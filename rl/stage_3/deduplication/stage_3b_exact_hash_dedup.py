"""
stage_3b_exact_hash_dedup.py
────────────────────────────
RL Stage 3B — Cross-dataset exact-hash deduplication.

Input  : merged Stage 2B parquet (rl_v1_merged.kept.parquet)
Output : rl_v1.exact_dedup.{kept,dropped}.parquet  +  report.json

Policy
──────
* **Cross-dataset dedup.** Duplicates across different `dataset_id`s are
  dropped regardless of `ground_truth_text` (the cross-ds collision itself is
  the signal — keep the stronger-verifier version).
* **Intra-dataset dedup, GT-aware.** Within the same `dataset_id`:
    - Same prompt + **same** `ground_truth_text`  → DROP (pure redundancy).
    - Same prompt + **different** `ground_truth_text` → KEEP (legitimate
      multi-answer case; e.g. SynLogic / ReasoningGym procedurally generate
      prompts with different verifier configs).
* **First-seen-wins.** Because Stage 2B concatenated the 9 datasets in
  verifier-strength order, the row backed by the *stronger* verifier wins at
  cross-ds collisions.
* **Hash source = last user message** of `context_messages`.  Rationale: RL
  prompts re-use system-prompt boilerplate heavily; hashing the full context
  would under-deduplicate.  This matches the design in
  Filtering_Pipeline/rl/FILTERING_APPROACH.md.
* **SLR-Bench skip (7 multilingual subsets) is bypassed entirely** — synthetic
  Prolog tasks with no resource overlap; pass through without hashing.

Normalization (applied to the extracted last-user text)
────────────────────────────────────────────────────────
  1. lowercase
  2. strip non-word / non-space characters
  3. collapse whitespace

Hash : SHA-1 (configurable via --hash-algo).

State
─────
SQLite index `exact_dedup_state.sqlite` maps prompt-hash → first-seen dataset_id.
`exact_dedup_checkpoint.json` allows `--resume` after interruption.

CLI
───
    python3 -m Filtering_Pipeline.rl.stage_3.deduplication.stage_3b_exact_hash_dedup \\
        --input      /home/workdir/Master_Thesis/corpora/rl/merged_rl_ds/rl_v1_merged.kept.parquet \\
        --output-dir /home/workdir/Master_Thesis/corpora/rl/dedup_3b_exact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Set

import psutil
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

# SLR-Bench multilingual subsets — synthetic Prolog, no resource overlap with
# the other 8 datasets.  Pass through without hashing.
DEFAULT_SKIP_DATASETS: List[str] = [
    "AIML-TUDA/SLR-Bench",
    "AIML-TUDA/SLR-Bench-German",
    "AIML-TUDA/SLR-Bench-Spanish",
    "AIML-TUDA/SLR-Bench-French",
    "AIML-TUDA/SLR-Bench-Italian",
    "AIML-TUDA/SLR-Bench-Portuguese",
    "AIML-TUDA/SLR-Bench-Dutch",
]


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization & hashing
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")


def _extract_last_user(context_messages: Any) -> str:
    """Return the content of the LAST user-role message in context_messages.

    Falls back to:
      - last non-empty message of any role, if no user message present
      - empty string, if context_messages is empty or malformed
    """
    if not isinstance(context_messages, list) or not context_messages:
        return ""
    last_user = ""
    last_any = ""
    for msg in context_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None:
            continue
        text = str(content)
        if not text.strip():
            continue
        role = str(msg.get("role", "")).lower()
        last_any = text
        if role == "user":
            last_user = text
    return last_user if last_user else last_any


def _normalize(text: str) -> str:
    x = (text or "").lower().strip()
    x = _PUNCT_RE.sub(" ", x)
    return _WS_RE.sub(" ", x).strip()


def _hash(text: str, algo: str) -> str:
    data = text.encode("utf-8", errors="ignore")
    if algo == "sha1":   return hashlib.sha1(data).hexdigest()
    if algo == "md5":    return hashlib.md5(data).hexdigest()
    if algo == "sha256": return hashlib.sha256(data).hexdigest()
    raise ValueError(f"Unsupported hash algo: {algo!r}")


def compute_prompt_hash(context_messages: Any, algo: str) -> tuple[str, str]:
    """Return (hash, normalized_text) for the last-user message."""
    raw = _extract_last_user(context_messages)
    norm = _normalize(raw)
    return _hash(norm, algo), norm


# ─────────────────────────────────────────────────────────────────────────────
# SQLite hash index
# ─────────────────────────────────────────────────────────────────────────────

def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hash_index "
        "(hash TEXT PRIMARY KEY, dataset_id TEXT, gt_hash TEXT)"
    )
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _lookup(h: str, cache: dict[str, tuple[str, str]], conn: sqlite3.Connection) -> Optional[tuple[str, str]]:
    """Return (first_dataset_id, first_gt_hash) for prompt-hash h, or None."""
    if h in cache:
        return cache[h]
    row = conn.execute(
        "SELECT dataset_id, gt_hash FROM hash_index WHERE hash = ?", (h,)
    ).fetchone()
    if row:
        cache[h] = (row[0], row[1] or "")
        return cache[h]
    return None


def _insert(h: str, dataset_id: str, gt_hash: str,
            cache: dict[str, tuple[str, str]], conn: sqlite3.Connection) -> None:
    cache[h] = (dataset_id, gt_hash)
    conn.execute(
        "INSERT OR REPLACE INTO hash_index (hash, dataset_id, gt_hash) VALUES (?, ?, ?)",
        (h, dataset_id, gt_hash),
    )


def _gt_fingerprint(gt_text: Any, algo: str) -> str:
    """Hash a ground-truth value for cheap equality comparison.

    Normalization is intentionally light (strip only) — RL ground-truth is
    canonical-answer text, often case-sensitive, and we want to drop only
    *literal* duplicates.
    """
    s = "" if gt_text is None else str(gt_text).strip()
    if not s:
        return ""
    return _hash(s, algo)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _State:
    processed_rows: int = 0
    batch_index:    int = 0
    kept:           int = 0
    dropped_cross:  int = 0
    dropped_intra_same_gt: int = 0
    intra_kept_diff_gt: int = 0
    skipped_passthrough: int = 0
    kept_by_ds:     Counter = field(default_factory=Counter)
    dropped_cross_by_ds: Counter = field(default_factory=Counter)
    dropped_intra_by_ds: Counter = field(default_factory=Counter)
    intra_kept_diff_gt_by_ds: Counter = field(default_factory=Counter)
    passthrough_by_ds: Counter = field(default_factory=Counter)
    dropped_pairs:  Counter = field(default_factory=Counter)

    @property
    def dropped(self) -> int:
        return self.dropped_cross + self.dropped_intra_same_gt


def _save_checkpoint(path: Path, state: _State, t0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_rows": state.processed_rows,
        "batch_index": state.batch_index,
        "kept": state.kept,
        "dropped_cross": state.dropped_cross,
        "dropped_intra_same_gt": state.dropped_intra_same_gt,
        "intra_kept_diff_gt": state.intra_kept_diff_gt,
        "skipped_passthrough": state.skipped_passthrough,
        "elapsed_sec": time.time() - t0,
        "kept_by_ds": dict(state.kept_by_ds),
        "dropped_cross_by_ds": dict(state.dropped_cross_by_ds),
        "dropped_intra_by_ds": dict(state.dropped_intra_by_ds),
        "intra_kept_diff_gt_by_ds": dict(state.intra_kept_diff_gt_by_ds),
        "passthrough_by_ds": dict(state.passthrough_by_ds),
        "dropped_pairs": {f"{a}|||{b}": c for (a, b), c in state.dropped_pairs.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_checkpoint(path: Path) -> Optional[_State]:
    if not path.exists():
        return None
    p = json.loads(path.read_text())
    s = _State(
        processed_rows = int(p.get("processed_rows", 0)),
        batch_index    = int(p.get("batch_index", 0)),
        kept           = int(p.get("kept", 0)),
        dropped_cross  = int(p.get("dropped_cross", 0)),
        dropped_intra_same_gt = int(p.get("dropped_intra_same_gt", 0)),
        intra_kept_diff_gt = int(p.get("intra_kept_diff_gt", 0)),
        skipped_passthrough = int(p.get("skipped_passthrough", 0)),
        kept_by_ds     = Counter({k: int(v) for k, v in p.get("kept_by_ds", {}).items()}),
        dropped_cross_by_ds = Counter({k: int(v) for k, v in p.get("dropped_cross_by_ds", {}).items()}),
        dropped_intra_by_ds = Counter({k: int(v) for k, v in p.get("dropped_intra_by_ds", {}).items()}),
        intra_kept_diff_gt_by_ds = Counter({k: int(v) for k, v in p.get("intra_kept_diff_gt_by_ds", {}).items()}),
        passthrough_by_ds = Counter({k: int(v) for k, v in p.get("passthrough_by_ds", {}).items()}),
    )
    for key, c in p.get("dropped_pairs", {}).items():
        if "|||" in key:
            a, b = key.split("|||", 1)
            s.dropped_pairs[(a, b)] = int(c)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Main dedup loop
# ─────────────────────────────────────────────────────────────────────────────

def _free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def run(
    input_path:   Path,
    output_dir:   Path,
    hash_algo:    str,
    batch_size:   int,
    resume:       bool,
    checkpoint_every: int,
    log_every:        int,
    skip_datasets: Set[str],
    min_free_gb:  float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_path       = output_dir / "rl_v1.exact_dedup.kept.parquet"
    dropped_path    = output_dir / "rl_v1.exact_dedup.dropped.parquet"
    report_path     = output_dir / "report.json"
    checkpoint_path = output_dir / "exact_dedup_checkpoint.json"
    db_path         = output_dir / "exact_dedup_state.sqlite"

    pf = pq.ParquetFile(input_path)
    schema      = pf.schema_arrow
    total_rows  = pf.metadata.num_rows

    for col in ("dataset_id", "example_id", "context_messages", "ground_truth_text"):
        if col not in schema.names:
            print(f"ERROR: Input is missing required column '{col}'")
            sys.exit(1)

    print(f"Input        : {input_path}")
    print(f"  Rows       : {total_rows:,}")
    print(f"  Row groups : {pf.metadata.num_row_groups}")
    print(f"  Hash algo  : {hash_algo}")
    print(f"  Batch size : {batch_size:,}")
    print(f"Output dir   : {output_dir}")
    print(f"Skip datasets ({len(skip_datasets)}):")
    for d in sorted(skip_datasets):
        print(f"  - {d}")
    print()

    state = (_load_checkpoint(checkpoint_path) if resume else None) or _State()
    cache: dict[str, tuple[str, str]] = {}
    conn = _open_db(db_path)

    kept_writer:    Optional[pq.ParquetWriter] = None
    dropped_writer: Optional[pq.ParquetWriter] = None
    t0 = time.time()
    skip_rows = state.processed_rows

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            # RAM guard
            free_gb = _free_ram_gb()
            if free_gb < min_free_gb:
                print(f"⚠️  RAM guard tripped ({free_gb:.1f} GB free) — stopping cleanly.")
                _save_checkpoint(checkpoint_path, state, t0)
                break

            # Resume: skip already-processed rows
            if skip_rows >= batch.num_rows:
                skip_rows -= batch.num_rows
                state.batch_index += 1
                continue
            if skip_rows > 0:
                batch = batch.slice(skip_rows)
                skip_rows = 0

            state.batch_index += 1

            ds_ids   = batch.column("dataset_id").to_pylist()
            ctx_list = batch.column("context_messages").to_pylist()
            gt_list  = batch.column("ground_truth_text").to_pylist()

            keep_idx:    List[int] = []
            drop_idx:    List[int] = []
            keep_hashes: List[str] = []
            keep_norms:  List[str] = []
            keep_seen_in: List[str] = []
            drop_hashes:  List[str] = []
            drop_norms:   List[str] = []
            drop_matched: List[str] = []
            drop_reasons: List[str] = []

            for i, (ds, ctx, gt) in enumerate(zip(ds_ids, ctx_list, gt_list)):
                ds = str(ds) if ds is not None else "__MISSING__"

                if ds in skip_datasets:
                    # Pass-through: no hashing, no lookup, no insert.
                    keep_idx.append(i)
                    keep_hashes.append("")
                    keep_norms.append("")
                    keep_seen_in.append("__SKIPPED__")
                    state.kept += 1
                    state.skipped_passthrough += 1
                    state.kept_by_ds[ds] += 1
                    state.passthrough_by_ds[ds] += 1
                    state.processed_rows += 1
                    continue

                h, norm = compute_prompt_hash(ctx, hash_algo)
                gt_h = _gt_fingerprint(gt, hash_algo)
                lookup = _lookup(h, cache, conn)

                if lookup is None:
                    _insert(h, ds, gt_h, cache, conn)
                    keep_idx.append(i)
                    keep_hashes.append(h)
                    keep_norms.append(norm)
                    keep_seen_in.append("")
                    state.kept += 1
                    state.kept_by_ds[ds] += 1
                else:
                    seen_ds, seen_gt = lookup
                    if seen_ds != ds:
                        # Cross-dataset duplicate → DROP (regardless of GT)
                        drop_idx.append(i)
                        drop_hashes.append(h)
                        drop_norms.append(norm)
                        drop_matched.append(seen_ds)
                        drop_reasons.append("exact_hash_cross_dataset")
                        state.dropped_cross += 1
                        state.dropped_cross_by_ds[ds] += 1
                        state.dropped_pairs[(ds, seen_ds)] += 1
                    elif seen_gt == gt_h:
                        # Intra-ds same prompt + same GT → DROP (redundant)
                        drop_idx.append(i)
                        drop_hashes.append(h)
                        drop_norms.append(norm)
                        drop_matched.append(seen_ds)
                        drop_reasons.append("exact_hash_intra_dataset_same_gt")
                        state.dropped_intra_same_gt += 1
                        state.dropped_intra_by_ds[ds] += 1
                    else:
                        # Intra-ds same prompt + different GT → KEEP (legitimate alternate)
                        keep_idx.append(i)
                        keep_hashes.append(h)
                        keep_norms.append(norm)
                        keep_seen_in.append(seen_ds)
                        state.kept += 1
                        state.intra_kept_diff_gt += 1
                        state.intra_kept_diff_gt_by_ds[ds] += 1
                        state.kept_by_ds[ds] += 1

                state.processed_rows += 1

            raw_table = pa.Table.from_batches([batch], schema=schema)

            if keep_idx:
                kept_t = raw_table.take(pa.array(keep_idx, pa.int64()))
                kept_t = kept_t.append_column("_dedup_hash",       pa.array(keep_hashes))
                kept_t = kept_t.append_column("_dedup_norm",       pa.array(keep_norms))
                kept_t = kept_t.append_column("_dedup_seen_in_ds", pa.array(keep_seen_in))
                if kept_writer is None:
                    kept_writer = pq.ParquetWriter(str(kept_path), kept_t.schema, compression="snappy")
                kept_writer.write_table(kept_t)

            if drop_idx:
                drop_t = raw_table.take(pa.array(drop_idx, pa.int64()))
                drop_t = drop_t.append_column("_drop_reason",      pa.array(drop_reasons))
                drop_t = drop_t.append_column("_dedup_hash",       pa.array(drop_hashes))
                drop_t = drop_t.append_column("_dedup_norm",       pa.array(drop_norms))
                drop_t = drop_t.append_column("_dedup_matched_ds", pa.array(drop_matched))
                if dropped_writer is None:
                    dropped_writer = pq.ParquetWriter(str(dropped_path), drop_t.schema, compression="snappy")
                dropped_writer.write_table(drop_t)

            if state.batch_index % log_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                rate = state.processed_rows / elapsed
                pct = state.processed_rows / total_rows * 100
                proc_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
                print(
                    f"  Batch {state.batch_index:>4d} | "
                    f"{state.processed_rows:>10,}/{total_rows:,} ({pct:5.1f}%) | "
                    f"kept={state.kept:,}  drop_x={state.dropped_cross:,}  drop_i={state.dropped_intra_same_gt:,}  "
                    f"intraK={state.intra_kept_diff_gt:,}  pass={state.skipped_passthrough:,} | "
                    f"{rate:>6,.0f} r/s  proc={proc_gb:.2f} GB"
                )

            if state.batch_index % checkpoint_every == 0:
                conn.commit()
                _save_checkpoint(checkpoint_path, state, t0)

    finally:
        conn.commit()
        conn.close()
        if kept_writer    is not None: kept_writer.close()
        if dropped_writer is not None: dropped_writer.close()

    elapsed = time.time() - t0

    # ── Report ─────────────────────────────────────────────────────────────
    total_dropped = state.dropped
    report = {
        "stage": "3B_exact_hash_dedup_rl",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "hash_source": "last_user_message(context_messages)",
        "gt_fingerprint": "sha1(strip(ground_truth_text))",
        "hash_algo": hash_algo,
        "batch_size": batch_size,
        "policy": {
            "drop_cross_dataset_duplicates": True,
            "drop_intra_dataset_when_same_gt": True,
            "keep_intra_dataset_when_diff_gt": True,
            "skip_datasets_passthrough": sorted(skip_datasets),
            "prompt_normalization": ["lowercase", "remove_punctuation", "collapse_whitespace"],
            "gt_normalization": ["strip_only"],
        },
        "total_rows_in":          total_rows,
        "total_kept":             state.kept,
        "total_dropped":          total_dropped,
        "dropped_cross_dataset":  state.dropped_cross,
        "dropped_intra_same_gt":  state.dropped_intra_same_gt,
        "intra_kept_diff_gt":     state.intra_kept_diff_gt,
        "skipped_passthrough":    state.skipped_passthrough,
        "drop_rate_pct":          round(total_dropped / total_rows * 100, 4) if total_rows else 0,
        "elapsed_seconds":        round(elapsed, 1),
        "kept_by_dataset":        dict(sorted(state.kept_by_ds.items())),
        "dropped_cross_by_dataset":   dict(sorted(state.dropped_cross_by_ds.items(), key=lambda x: -x[1])),
        "dropped_intra_by_dataset":   dict(sorted(state.dropped_intra_by_ds.items(), key=lambda x: -x[1])),
        "intra_kept_diff_gt_by_dataset": dict(sorted(state.intra_kept_diff_gt_by_ds.items(), key=lambda x: -x[1])),
        "passthrough_by_dataset": dict(sorted(state.passthrough_by_ds.items())),
        "dropped_cross_pairs_top50": [
            {"dropped_from": a, "matched_in": b, "count": c}
            for (a, b), c in state.dropped_pairs.most_common(50)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Cleanup checkpoint on full success
    if state.processed_rows == total_rows:
        checkpoint_path.unlink(missing_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Stage 3B complete  ({elapsed:.1f}s)")
    print(f"  Total in                 : {total_rows:,}")
    print(f"  Kept                     : {state.kept:,}")
    print(f"    intra-ds kept (diff GT): {state.intra_kept_diff_gt:,}")
    print(f"    passthrough (SLR)      : {state.skipped_passthrough:,}")
    print(f"  Dropped (cross-ds)       : {state.dropped_cross:,}")
    print(f"  Dropped (intra-ds same GT): {state.dropped_intra_same_gt:,}")
    print(f"  Total dropped            : {total_dropped:,}  ({report['drop_rate_pct']}%)")
    if state.dropped_cross_by_ds:
        print(f"\n  Cross-ds dropped (top 10):")
        for ds, cnt in state.dropped_cross_by_ds.most_common(10):
            print(f"    {ds:<50s}  {cnt:>8,}")
    if state.dropped_intra_by_ds:
        print(f"\n  Intra-ds same-GT dropped (top 10):")
        for ds, cnt in state.dropped_intra_by_ds.most_common(10):
            print(f"    {ds:<50s}  {cnt:>8,}")
    if state.dropped_pairs:
        print(f"\n  Top cross-dataset duplicate pairs:")
        for (a, b), c in state.dropped_pairs.most_common(10):
            print(f"    {a:<35s} → matched in {b:<35s}  {c:>7,}")
    print(f"\nOutputs:")
    print(f"  {kept_path}  ({kept_path.stat().st_size / 1e9:.2f} GB)")
    if total_dropped > 0 and dropped_path.exists():
        print(f"  {dropped_path}  ({dropped_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {report_path}")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "RL Stage 3B — cross-dataset exact-hash dedup. "
            "Hash source = last user message of context_messages. "
            "SLR-Bench multilingual subsets are passed through (no overlap risk)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="/home/workdir/Master_Thesis/corpora/rl/merged_rl_ds/rl_v1_merged.kept.parquet",
        help="Stage 2B merged parquet.",
    )
    p.add_argument(
        "--output-dir",
        default="/home/workdir/Master_Thesis/corpora/rl/dedup_3b_exact",
        help="Output directory.",
    )
    p.add_argument("--hash-algo", default="sha1", choices=["sha1", "md5", "sha256"])
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--min-free-gb", type=float, default=40.0)
    p.add_argument(
        "--skip-datasets",
        nargs="*",
        default=None,
        help=(
            f"Dataset IDs to pass through without hashing (no cross-ds dedup applied). "
            f"Default: {len(DEFAULT_SKIP_DATASETS)} SLR-Bench multilingual subsets."
        ),
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable the default SLR-Bench skip list (dedup every row).",
    )
    args = p.parse_args(argv)

    if args.no_skip:
        skip_set: Set[str] = set()
    elif args.skip_datasets is None:
        skip_set = set(DEFAULT_SKIP_DATASETS)
    else:
        skip_set = set(args.skip_datasets)

    run(
        input_path = Path(args.input),
        output_dir = Path(args.output_dir),
        hash_algo  = args.hash_algo,
        batch_size = args.batch_size,
        resume     = args.resume,
        checkpoint_every = args.checkpoint_every,
        log_every  = args.log_every,
        skip_datasets = skip_set,
        min_free_gb = args.min_free_gb,
    )


if __name__ == "__main__":
    main()
