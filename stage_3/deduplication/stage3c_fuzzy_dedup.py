"""Stage 3C — Cross-Dataset Fuzzy Deduplication.

Removes rows whose context prompt is a near-duplicate of a prompt already seen
in a *different* dataset, using RapidFuzz Levenshtein similarity.
Same-dataset duplicates are kept by design (handled in Stages 3A/3B).

This is the third of three sequential deduplication passes:

  Stage 3A  Same-ID dedup       (intra-dataset, by example_id)
  Stage 3B  Exact-hash dedup    (cross-dataset, SHA1 of normalized context)
  Stage 3C  Fuzzy dedup         ← this script

Inspired by the OpenThoughts deduplication approach:
  OpenThoughts3 paper: https://arxiv.org/abs/2506.04178
  OpenThoughts uses fuzz.ratio with threshold 95.
  For SFT-Collection-v2 we used threshold 78 (more aggressive dedup across
  the broader multilingual dataset mix — see README for rationale).

Algorithm
---------
For each incoming row:
  1. Extract + normalize context_messages (same normalization as Stage 3B).
  2. Identify candidates via bucketing:
       bucket key = first <bucket_tokens> non-role words  +  len(text) // len_bin_size
     Neighbour bins (±neighbor_bin_radius) are also checked to catch near-length matches.
  3. Compare against candidates with fuzz.ratio (Levenshtein normalized 0–100).
  4. If best match ≥ similarity_threshold and match is from a *different* dataset → drop.
  5. Otherwise → keep and add normalized text to the index.

All comparisons within a batch are parallelized with ProcessPoolExecutor.

Input
-----
A Parquet file from Stage 3B output (or Stage 2B / 3A if skipping earlier stages).
Required columns: dataset_id, example_id, context_messages.

Output (written to --output-dir)
----------------------------------
  merged_sft.fuzzy_dedup.kept.parquet    — unique rows
  merged_sft.fuzzy_dedup.dropped.parquet — near-duplicate rows removed
  report.json                            — counts, drop rate, per-dataset breakdown,
                                           top cross-dataset duplicate pairs
Intermediate files (can be deleted after a successful run):
  fuzzy_dedup_checkpoint.json            — batch-level resume state

Usage
-----
    # Reproduce SFT-Collection-v2 (threshold 78):
    python stage3c_fuzzy_dedup.py \\
        --input  /path/to/stage3b/merged_sft.exact_dedup.kept.parquet \\
        --output-dir /path/to/stage3c_fuzzy_dedup

    # OpenThoughts-style (threshold 95):
    python stage3c_fuzzy_dedup.py \\
        --input  /path/to/... \\
        --output-dir /path/to/... \\
        --similarity-threshold 95

    # Resume an interrupted run:
    python stage3c_fuzzy_dedup.py \\
        --input  /path/to/... --output-dir /path/to/... --resume

Requirements
------------
    pip install rapidfuzz pyarrow
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process as rfprocess


# ---------------------------------------------------------------------------
# Text normalization (identical to Stage 3B)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")
_ROLE_TOKENS = frozenset({"user", "assistant", "system"})


def _extract_context(context_messages: Any) -> str:
    if not isinstance(context_messages, list):
        return ""
    parts = []
    for msg in context_messages:
        if isinstance(msg, dict):
            role    = str(msg.get("role", "unknown")).lower()
            content = msg.get("content")
            text    = str(content) if content is not None else ""
            if text.strip():
                parts.append(f"{role}: {text}")
    return "\n".join(parts)


def _normalize(text: str) -> str:
    x = (text or "").lower().strip()
    x = _PUNCT_RE.sub(" ", x)
    return _WS_RE.sub(" ", x).strip()


def normalize_context(context_messages: Any) -> str:
    return _normalize(_extract_context(context_messages))


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------

def _bucket_head(text: str, n_tokens: int) -> str:
    tokens = [t for t in text.split() if t not in _ROLE_TOKENS]
    if not tokens:
        return "__EMPTY__"
    return "_".join(tokens[: max(1, n_tokens)])


def _bucket_key(text: str, n_tokens: int, bin_size: int) -> str:
    head = _bucket_head(text, n_tokens)
    if head == "__EMPTY__":
        return "__EMPTY__"
    return f"{head}__{len(text) // max(1, bin_size)}"


def _bucket_key_for_bin(head: str, length_bin: int) -> str:
    if head == "__EMPTY__":
        return "__EMPTY__"
    return f"{head}__{length_bin}"


def _add_to_bucket(
    index: dict[str, deque],
    key: str,
    text: str,
    max_per_bucket: int | None,
) -> None:
    bucket = index.setdefault(key, deque())
    bucket.append(text)
    if max_per_bucket is not None:
        while len(bucket) > max_per_bucket:
            bucket.popleft()


# ---------------------------------------------------------------------------
# Worker function for ProcessPoolExecutor (must be top-level / picklable)
# ---------------------------------------------------------------------------

def _find_match(
    args: tuple[str, list[str], int | None, float],
) -> tuple[str | None, float]:
    """Return (best_matching_candidate, score) or (None, 0.0)."""
    query, candidates, max_len_diff, threshold = args
    if not query or not candidates:
        return None, 0.0
    if max_len_diff is not None:
        q_len = len(query)
        candidates = [c for c in candidates if abs(len(c) - q_len) <= max_len_diff]
    if not candidates:
        return None, 0.0
    result = rfprocess.extractOne(
        query, candidates, scorer=fuzz.ratio, score_cutoff=threshold
    )
    if result:
        return result[0], result[1]
    return None, 0.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class _State:
    processed_rows: int = 0
    batch_index:    int = 0
    kept:           int = 0
    dropped:        int = 0
    kept_by_ds:     Counter = field(default_factory=Counter)
    dropped_by_ds:  Counter = field(default_factory=Counter)
    dropped_pairs:  Counter = field(default_factory=Counter)


def _save_checkpoint(path: Path, state: _State, t0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc":   datetime.now(timezone.utc).isoformat(),
        "processed_rows": state.processed_rows,
        "batch_index":    state.batch_index,
        "kept":           state.kept,
        "dropped":        state.dropped,
        "elapsed_sec":    time.time() - t0,
        "kept_by_ds":     dict(state.kept_by_ds),
        "dropped_by_ds":  dict(state.dropped_by_ds),
        "dropped_pairs":  {
            f"{a}|||{b}": c for (a, b), c in state.dropped_pairs.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_checkpoint(path: Path) -> _State | None:
    if not path.exists():
        return None
    p = json.loads(path.read_text())
    s = _State(
        processed_rows = int(p.get("processed_rows", 0)),
        batch_index    = int(p.get("batch_index", 0)),
        kept           = int(p.get("kept", 0)),
        dropped        = int(p.get("dropped", 0)),
        kept_by_ds     = Counter({k: int(v) for k, v in p.get("kept_by_ds", {}).items()}),
        dropped_by_ds  = Counter({k: int(v) for k, v in p.get("dropped_by_ds", {}).items()}),
    )
    for key, c in p.get("dropped_pairs", {}).items():
        if "|||" in key:
            a, b = key.split("|||", 1)
            s.dropped_pairs[(a, b)] = int(c)
    return s


# ---------------------------------------------------------------------------
# Main dedup loop
# ---------------------------------------------------------------------------

def run(
    input_path:          Path,
    output_dir:          Path,
    similarity_threshold: float = 78.0,
    batch_size:           int   = 20_000,
    resume:               bool  = False,
    num_workers:          int   = 32,
    bucket_tokens:        int   = 2,
    len_bin_size:         int   = 100,
    max_candidates_per_bucket: int | None = 5_000,
    max_len_diff:         int | None = 40,
    neighbor_bin_radius:  int   = 1,
    checkpoint_every:     int   = 20,
    log_every:            int   = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_path       = output_dir / "merged_sft.fuzzy_dedup.kept.parquet"
    dropped_path    = output_dir / "merged_sft.fuzzy_dedup.dropped.parquet"
    report_path     = output_dir / "report.json"
    checkpoint_path = output_dir / "fuzzy_dedup_checkpoint.json"

    pf = pq.ParquetFile(input_path)
    schema     = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    for col in ("dataset_id", "example_id", "context_messages"):
        if col not in schema.names:
            raise ValueError(f"Input file is missing required column: '{col}'")

    print(f"Input : {input_path}")
    print(f"  Rows          : {total_rows:,}")
    print(f"  Row groups    : {pf.metadata.num_row_groups}")
    print(f"  Sim threshold : {similarity_threshold} (fuzz.ratio)")
    print(f"  Batch size    : {batch_size:,}")
    print(f"  Workers       : {num_workers}")
    print(f"Output dir: {output_dir}")
    print()

    state = (_load_checkpoint(checkpoint_path) if resume else None) or _State()

    # In-memory prompt index: norm_text → (dataset_id, example_id)
    prompt_to_source: dict[str, tuple[str, str]] = {}
    bucket_index:     dict[str, deque]           = {}
    prompt_count = 0

    kept_writer:    pq.ParquetWriter | None = None
    dropped_writer: pq.ParquetWriter | None = None
    t0 = time.time()

    try:
        # --- Rebuild index for resume ---
        if resume and state.processed_rows > 0:
            print(f"[resume] Rebuilding index from first {state.processed_rows:,} rows ...")
            t_rebuild = time.time()
            for batch in pf.iter_batches(
                batch_size=50_000, columns=["dataset_id", "example_id", "context_messages"]
            ):
                ds_ids   = batch.column("dataset_id").to_pylist()
                ex_ids   = batch.column("example_id").to_pylist()
                ctx_list = batch.column("context_messages").to_pylist()
                for ds, ex, ctx in zip(ds_ids, ex_ids, ctx_list):
                    norm = normalize_context(ctx)
                    if norm and norm not in prompt_to_source:
                        ds_s = str(ds) if ds is not None else "__MISSING__"
                        ex_s = str(ex) if ex is not None else "__MISSING__"
                        _add_to_bucket(
                            bucket_index,
                            _bucket_key(norm, bucket_tokens, len_bin_size),
                            norm,
                            max_candidates_per_bucket,
                        )
                        prompt_to_source[norm] = (ds_s, ex_s)
                        prompt_count += 1
                        if prompt_count >= state.processed_rows:
                            break
                if prompt_count >= state.processed_rows:
                    break
            print(
                f"[resume] Index rebuilt: {prompt_count:,} prompts  "
                f"({time.time() - t_rebuild:.1f}s)\n"
            )

        executor = ProcessPoolExecutor(max_workers=num_workers)
        skip_rows = state.processed_rows

        try:
            for batch in pf.iter_batches(batch_size=batch_size):
                # Skip already-processed rows
                if skip_rows >= batch.num_rows:
                    skip_rows -= batch.num_rows
                    state.batch_index += 1
                    continue
                if skip_rows > 0:
                    batch = batch.slice(skip_rows)
                    skip_rows = 0

                state.batch_index += 1

                ds_ids   = batch.column("dataset_id").to_pylist()
                ex_ids   = batch.column("example_id").to_pylist()
                ctx_list = batch.column("context_messages").to_pylist()

                # Pre-normalize all rows in this batch
                norms = [normalize_context(ctx) for ctx in ctx_list]

                # Build per-batch local index (so rows within the batch can match each other)
                local_prompt_to_source: dict[str, tuple[str, str]] = {}
                local_bucket_index:     dict[str, deque]           = {}

                # Collect candidate lists for each row
                tasks: list[tuple[str, list[str], int | None, float]] = []
                for i, (norm, ds, ex) in enumerate(zip(norms, ds_ids, ex_ids)):
                    candidates: list[str] = []
                    if norm:
                        head     = _bucket_head(norm, bucket_tokens)
                        base_bin = len(norm) // max(1, len_bin_size)
                        for offset in range(-neighbor_bin_radius, neighbor_bin_radius + 1):
                            bkey = _bucket_key_for_bin(head, base_bin + offset)
                            if bkey in bucket_index:
                                candidates.extend(bucket_index[bkey])
                            if bkey in local_bucket_index:
                                candidates.extend(local_bucket_index[bkey])

                        # Add to local index *after* collecting candidates (so it can't match itself)
                        if norm not in local_prompt_to_source:
                            ds_s = str(ds) if ds is not None else "__MISSING__"
                            ex_s = str(ex) if ex is not None else "__MISSING__"
                            _add_to_bucket(
                                local_bucket_index,
                                _bucket_key(norm, bucket_tokens, len_bin_size),
                                norm,
                                max_candidates_per_bucket,
                            )
                            local_prompt_to_source[norm] = (ds_s, ex_s)

                    tasks.append((norm, candidates, max_len_diff, similarity_threshold))

                # Parallel fuzzy matching
                results = list(executor.map(_find_match, tasks, chunksize=100))

                # Decision pass
                keep_idx: list[int] = []
                drop_idx: list[int] = []
                drop_reasons:    list[str]   = []
                drop_scores:     list[float] = []
                drop_matched_ds: list[str]   = []
                drop_matched_ex: list[str]   = []

                for i, (ds, ex, norm, (matched_norm, score)) in enumerate(
                    zip(ds_ids, ex_ids, norms, results)
                ):
                    ds_s = str(ds) if ds is not None else "__MISSING__"
                    ex_s = str(ex) if ex is not None else "__MISSING__"

                    # Resolve who owns the matched prompt
                    matched_ds = matched_ex = ""
                    if matched_norm:
                        info = prompt_to_source.get(matched_norm) or local_prompt_to_source.get(matched_norm)
                        if info:
                            matched_ds, matched_ex = info
                        # Ignore same-dataset matches
                        if matched_ds == ds_s:
                            matched_ds = matched_ex = ""

                    if matched_ds:
                        drop_idx.append(i)
                        drop_reasons.append("fuzzy_duplicate")
                        drop_scores.append(score)
                        drop_matched_ds.append(matched_ds)
                        drop_matched_ex.append(matched_ex)
                        state.dropped += 1
                        state.dropped_by_ds[ds_s] += 1
                        state.dropped_pairs[(ds_s, matched_ds)] += 1
                    else:
                        keep_idx.append(i)
                        state.kept += 1
                        state.kept_by_ds[ds_s] += 1
                        # Add to global index
                        if norm and norm not in prompt_to_source:
                            _add_to_bucket(
                                bucket_index,
                                _bucket_key(norm, bucket_tokens, len_bin_size),
                                norm,
                                max_candidates_per_bucket,
                            )
                            prompt_to_source[norm] = (ds_s, ex_s)
                            prompt_count += 1

                raw_table = pa.Table.from_batches([batch], schema=schema)

                if keep_idx:
                    kept_t = raw_table.take(pa.array(keep_idx, pa.int64()))
                    if kept_writer is None:
                        kept_writer = pq.ParquetWriter(str(kept_path), kept_t.schema)
                    kept_writer.write_table(kept_t)

                if drop_idx:
                    drop_t = raw_table.take(pa.array(drop_idx, pa.int64()))
                    drop_t = drop_t.append_column("_drop_reason",           pa.array(drop_reasons))
                    drop_t = drop_t.append_column("_fuzzy_score",           pa.array(drop_scores))
                    drop_t = drop_t.append_column("_fuzzy_matched_dataset", pa.array(drop_matched_ds))
                    drop_t = drop_t.append_column("_fuzzy_matched_example", pa.array(drop_matched_ex))
                    if dropped_writer is None:
                        dropped_writer = pq.ParquetWriter(str(dropped_path), drop_t.schema)
                    dropped_writer.write_table(drop_t)

                state.processed_rows += batch.num_rows

                if state.batch_index % log_every == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    pct  = state.processed_rows / total_rows * 100
                    rate = state.processed_rows / elapsed
                    eta  = (total_rows - state.processed_rows) / rate if rate > 0 else 0
                    print(
                        f"  Batch {state.batch_index:>4d} | "
                        f"{state.processed_rows:>12,}/{total_rows:,} ({pct:5.1f}%) | "
                        f"kept={state.kept:,}  dropped={state.dropped:,} | "
                        f"indexed={prompt_count:,} | "
                        f"{rate:,.0f} rows/s  ETA ~{eta:.0f}s"
                    )

                if state.batch_index % checkpoint_every == 0:
                    _save_checkpoint(checkpoint_path, state, t0)

        finally:
            executor.shutdown(wait=False)

    finally:
        if kept_writer    is not None: kept_writer.close()
        if dropped_writer is not None: dropped_writer.close()

    elapsed = time.time() - t0
    _save_checkpoint(checkpoint_path, state, t0)

    # Report
    report = {
        "stage":              "3C_fuzzy_dedup",
        "generated_at_utc":   datetime.now(timezone.utc).isoformat(),
        "input_path":         str(input_path),
        "method":             "rapidfuzz fuzz.ratio (Levenshtein, normalized 0-100)",
        "similarity_threshold": similarity_threshold,
        "batch_size":         batch_size,
        "num_workers":        num_workers,
        "policy": {
            "first_seen_wins":           True,
            "cross_dataset_only":        True,
            "normalization":             ["lowercase", "remove_punctuation", "collapse_whitespace"],
        },
        "bucket_config": {
            "bucket_tokens":              bucket_tokens,
            "len_bin_size":               len_bin_size,
            "max_candidates_per_bucket":  max_candidates_per_bucket,
            "max_len_diff":               max_len_diff,
            "neighbor_bin_radius":        neighbor_bin_radius,
        },
        "total_rows_in":  total_rows,
        "total_kept":     state.kept,
        "total_dropped":  state.dropped,
        "drop_rate_pct":  round(state.dropped / total_rows * 100, 4) if total_rows else 0,
        "prompts_indexed": prompt_count,
        "elapsed_seconds": round(elapsed, 1),
        "kept_by_dataset": dict(sorted(state.kept_by_ds.items())),
        "dropped_by_dataset": dict(
            sorted(state.dropped_by_ds.items(), key=lambda x: -x[1])
        ),
        "dropped_cross_pairs": [
            {"dropped_from": a, "matched_in": b, "count": c}
            for (a, b), c in state.dropped_pairs.most_common(50)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}")
    print(f"Stage 3C complete  ({elapsed:.1f}s)")
    print(f"  Total in  : {total_rows:,}")
    print(f"  Kept      : {state.kept:,}")
    print(f"  Dropped   : {state.dropped:,}  ({report['drop_rate_pct']}%)")
    if state.dropped_by_ds:
        print("  Dropped by dataset (top 10):")
        for ds, cnt in state.dropped_by_ds.most_common(10):
            print(f"    {ds:<60s}  {cnt:>8,}")
    print(f"\nOutputs:")
    print(f"  {kept_path}  ({kept_path.stat().st_size / 1e9:.2f} GB)")
    if state.dropped > 0 and dropped_path.exists():
        print(f"  {dropped_path}  ({dropped_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 3C: Cross-dataset fuzzy deduplication using RapidFuzz fuzz.ratio. "
            "First-seen-wins policy; same-dataset duplicates are kept. "
            "Default threshold 78 (used for SFT-Collection-v2). "
            "OpenThoughts3 uses 95 — see README for discussion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,
                   help="Input Parquet file (Stage 3B output recommended).")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write outputs into.")
    p.add_argument("--similarity-threshold", type=float, default=78.0,
                   help=(
                       "fuzz.ratio threshold (0–100). "
                       "78 = SFT-Collection-v2 setting (aggressive). "
                       "95 = OpenThoughts3 setting (conservative)."
                   ))
    p.add_argument("--batch-size",    type=int, default=20_000,
                   help="Rows per batch.")
    p.add_argument("--num-workers",   type=int, default=32,
                   help="ProcessPoolExecutor workers for parallel matching.")
    p.add_argument("--bucket-tokens", type=int, default=2,
                   help="Leading content tokens used to build bucket key.")
    p.add_argument("--len-bin-size",  type=int, default=100,
                   help="Text length bin size for bucketing (len // bin_size).")
    p.add_argument("--max-candidates-per-bucket", type=int, default=5_000,
                   help="Cap per bucket; oldest entries evicted first.")
    p.add_argument("--max-len-diff",  type=int, default=40,
                   help="Skip candidates where |len(query) - len(candidate)| > this. -1 = disabled.")
    p.add_argument("--neighbor-bin-radius", type=int, default=1,
                   help="Also search ±radius length bins when looking up candidates.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint in --output-dir.")
    p.add_argument("--checkpoint-every", type=int, default=20,
                   help="Save checkpoint every N batches.")
    p.add_argument("--log-every",        type=int, default=5,
                   help="Print progress every N batches.")
    args = p.parse_args(argv)

    max_len_diff = None if args.max_len_diff < 0 else args.max_len_diff
    run(
        input_path           = Path(args.input),
        output_dir           = Path(args.output_dir),
        similarity_threshold = args.similarity_threshold,
        batch_size           = args.batch_size,
        resume               = args.resume,
        num_workers          = args.num_workers,
        bucket_tokens        = args.bucket_tokens,
        len_bin_size         = args.len_bin_size,
        max_candidates_per_bucket = args.max_candidates_per_bucket,
        max_len_diff         = max_len_diff,
        neighbor_bin_radius  = args.neighbor_bin_radius,
        checkpoint_every     = args.checkpoint_every,
        log_every            = args.log_every,
    )


if __name__ == "__main__":
    main()
