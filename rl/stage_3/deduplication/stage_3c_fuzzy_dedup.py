"""Stage 3C — Fuzzy Deduplication for the RL merge pipeline.

Adapted from `Filtering_Pipeline/stage_3/deduplication/stage3c_fuzzy_dedup.py`.

Policy (matches Stage 3B exact-hash dedup)
------------------------------------------
For each incoming row:
  1. Hash source: **last user message** of ``context_messages`` (not the full
     concatenated context — RL prompts heavily reuse system boilerplate, so
     hashing the whole context would never match).
  2. Pre-normalize: lowercase + strip punctuation + collapse whitespace.
  3. Find fuzzy-matching prompts in a bucketed index using RapidFuzz
     ``fuzz.ratio`` (Levenshtein, normalized 0–100). All matches above the
     threshold are returned (not just the best one — we need to inspect
     them under the policy below).
  4. Decision:
       • Any cross-dataset match (matched_ds != query_ds) → **DROP**
         (first-seen-wins by verifier strength, like Stage 3B).
       • Intra-dataset match where any matched record has the **exact same
         stripped ground_truth_text** → **DROP** (redundant near-duplicate).
       • Intra-dataset match with **different GT** → **KEEP** (legitimate
         multi-answer near-duplicate, e.g. procedurally generated tasks).
       • No match above threshold → **KEEP**.
  5. SLR-Bench multilingual subsets are passed through verbatim (synthetic
     Prolog tasks, no resource overlap with the other 8 datasets). They are
     never indexed, never fuzzy-matched.

  6. ``logicreasoning/logi_glue`` is excluded from **intra-dataset** dedup
     (still participates in cross-DS detection). Reason: its templated
     multiple-choice / NLI items share ~95 %% boilerplate prompt text with
     answer-relevant variation in only 1-2 tokens, and the GT label space is
     tiny (entailment/contradiction/neutral, True/False, 4-way MC), so the
     GT-aware policy cannot reliably distinguish near-duplicates from
     legitimately distinct items. Manual inspection at threshold 90 showed
     the majority of "drops" were genuine distinct tasks. See
     ``THRESHOLD_DECISION.md``.

Important: GT comparison is **exact** (strip + sha1), never fuzzy. Per user
requirement: "die gt müssen wirklich gleich sein und nicht fuzzy gleich".

Input
-----
Stage 3B output: ``rl_v1.exact_dedup.kept.parquet``.
Required columns: ``dataset_id``, ``example_id``, ``context_messages``,
``ground_truth_text``.

Output (written to --output-dir)
--------------------------------
  rl_v1.fuzzy_dedup.kept.parquet       — unique rows + passthrough rows
  rl_v1.fuzzy_dedup.dropped.parquet    — fuzzy-duplicate rows removed
  report.json                          — counts, drop rate, per-dataset breakdown
  fuzzy_dedup_checkpoint.json          — batch-level resume state

Usage
-----
    python -m Filtering_Pipeline.rl.stage_3.deduplication.stage_3c_fuzzy_dedup \\
        --input  /home/workdir/Master_Thesis/corpora/rl/dedup_3b_exact/rl_v1.exact_dedup.kept.parquet \\
        --output-dir /home/workdir/Master_Thesis/corpora/rl/dedup_3c_fuzzy \\
        --similarity-threshold 90
"""

from __future__ import annotations

import argparse
import hashlib
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
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SKIP_DATASETS = (
    "AIML-TUDA/SLR-Bench",
    "AIML-TUDA/SLR-Bench-Dutch",
    "AIML-TUDA/SLR-Bench-French",
    "AIML-TUDA/SLR-Bench-German",
    "AIML-TUDA/SLR-Bench-Italian",
    "AIML-TUDA/SLR-Bench-Portuguese",
    "AIML-TUDA/SLR-Bench-Spanish",
)

# Datasets that participate in CROSS-dataset detection (their prompts are
# indexed so other datasets can find duplicates against them, and they can
# themselves be cross-matched out), but where INTRA-dataset matches are
# ignored — even when the GT is identical.
#
# logi_glue is excluded from intra-dedup because its prompts are heavily
# templated (>95% shared boilerplate) with answer-relevant variation in only
# 1-2 tokens (multiple-choice distractors, NLI hypothesis). With a small GT
# label space ({entailment, contradiction, neutral} / {True, False} / 4-way
# MC), unrelated tasks that happen to share the correct label end up with
# identical GT fingerprints, so the GT-aware policy can no longer
# distinguish them. Manual inspection at threshold 90 confirmed the vast
# majority of intra-dataset "drops" in logi_glue were legitimately distinct
# multiple-choice / NLI items. See THRESHOLD_DECISION.md.
DEFAULT_SKIP_INTRA_DATASETS = (
    "logicreasoning/logi_glue",
)


# ---------------------------------------------------------------------------
# Text normalization (last-user-message only, identical to Stage 3B v2)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")
_ROLE_TOKENS = frozenset({"user", "assistant", "system"})


def _extract_last_user(context_messages: Any) -> str:
    """Return the content of the last user-role message, or ''."""
    if not isinstance(context_messages, list):
        return ""
    last = ""
    for msg in context_messages:
        if isinstance(msg, dict):
            if str(msg.get("role", "")).lower() == "user":
                content = msg.get("content")
                if content is not None:
                    last = str(content)
    return last


def _normalize(text: str) -> str:
    x = (text or "").lower().strip()
    x = _PUNCT_RE.sub(" ", x)
    return _WS_RE.sub(" ", x).strip()


def normalize_prompt(context_messages: Any) -> str:
    return _normalize(_extract_last_user(context_messages))


def gt_fingerprint(gt_text: Any) -> str:
    """sha1(strip(gt_text)) — exact (not fuzzy) GT fingerprint."""
    s = "" if gt_text is None else str(gt_text).strip()
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


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
# Worker: return ALL matches above threshold (need them for policy check)
# ---------------------------------------------------------------------------

def _find_matches(
    args: tuple[str, list[str], int | None, float, int],
) -> list[tuple[str, float]]:
    query, candidates, max_len_diff, threshold, limit = args
    if not query or not candidates:
        return []
    if max_len_diff is not None:
        q_len = len(query)
        candidates = [c for c in candidates if abs(len(c) - q_len) <= max_len_diff]
    if not candidates:
        return []
    # De-duplicate while preserving order (a candidate may appear in
    # both global and local buckets).
    seen: set[str] = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    results = rfprocess.extract(
        query, uniq, scorer=fuzz.ratio, score_cutoff=threshold, limit=limit
    )
    return [(r[0], r[1]) for r in results]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class _State:
    processed_rows:           int = 0
    batch_index:              int = 0
    kept:                     int = 0
    dropped:                  int = 0
    passthrough:              int = 0
    dropped_cross:            int = 0
    dropped_intra_same_gt:    int = 0
    intra_kept_diff_gt:       int = 0
    kept_by_ds:               Counter = field(default_factory=Counter)
    dropped_by_ds:            Counter = field(default_factory=Counter)
    passthrough_by_ds:        Counter = field(default_factory=Counter)
    dropped_cross_by_ds:      Counter = field(default_factory=Counter)
    dropped_intra_by_ds:      Counter = field(default_factory=Counter)
    intra_kept_diff_gt_by_ds: Counter = field(default_factory=Counter)
    dropped_pairs:            Counter = field(default_factory=Counter)


def _save_checkpoint(path: Path, state: _State, t0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc":        datetime.now(timezone.utc).isoformat(),
        "processed_rows":      state.processed_rows,
        "batch_index":         state.batch_index,
        "kept":                state.kept,
        "dropped":             state.dropped,
        "passthrough":         state.passthrough,
        "dropped_cross":       state.dropped_cross,
        "dropped_intra_same_gt": state.dropped_intra_same_gt,
        "intra_kept_diff_gt":  state.intra_kept_diff_gt,
        "elapsed_sec":         time.time() - t0,
        "kept_by_ds":          dict(state.kept_by_ds),
        "dropped_by_ds":       dict(state.dropped_by_ds),
        "passthrough_by_ds":   dict(state.passthrough_by_ds),
        "dropped_cross_by_ds": dict(state.dropped_cross_by_ds),
        "dropped_intra_by_ds": dict(state.dropped_intra_by_ds),
        "intra_kept_diff_gt_by_ds": dict(state.intra_kept_diff_gt_by_ds),
        "dropped_pairs":       {f"{a}|||{b}": c for (a, b), c in state.dropped_pairs.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_checkpoint(path: Path) -> _State | None:
    if not path.exists():
        return None
    p = json.loads(path.read_text())
    s = _State(
        processed_rows        = int(p.get("processed_rows", 0)),
        batch_index           = int(p.get("batch_index", 0)),
        kept                  = int(p.get("kept", 0)),
        dropped               = int(p.get("dropped", 0)),
        passthrough           = int(p.get("passthrough", 0)),
        dropped_cross         = int(p.get("dropped_cross", 0)),
        dropped_intra_same_gt = int(p.get("dropped_intra_same_gt", 0)),
        intra_kept_diff_gt    = int(p.get("intra_kept_diff_gt", 0)),
        kept_by_ds            = Counter({k: int(v) for k, v in p.get("kept_by_ds", {}).items()}),
        dropped_by_ds         = Counter({k: int(v) for k, v in p.get("dropped_by_ds", {}).items()}),
        passthrough_by_ds     = Counter({k: int(v) for k, v in p.get("passthrough_by_ds", {}).items()}),
        dropped_cross_by_ds   = Counter({k: int(v) for k, v in p.get("dropped_cross_by_ds", {}).items()}),
        dropped_intra_by_ds   = Counter({k: int(v) for k, v in p.get("dropped_intra_by_ds", {}).items()}),
        intra_kept_diff_gt_by_ds = Counter({k: int(v) for k, v in p.get("intra_kept_diff_gt_by_ds", {}).items()}),
    )
    for key, c in p.get("dropped_pairs", {}).items():
        if "|||" in key:
            a, b = key.split("|||", 1)
            s.dropped_pairs[(a, b)] = int(c)
    return s


# ---------------------------------------------------------------------------
# Main dedup loop
# ---------------------------------------------------------------------------

# prompt_records[norm] -> list of (dataset_id, example_id, gt_hash)
PromptRecord = tuple[str, str, str]


def _classify_match(
    query_ds: str,
    query_gt_hash: str,
    matched_norms: list[str],
    prompt_records: dict[str, list[PromptRecord]],
    local_records: dict[str, list[PromptRecord]],
    skip_intra: frozenset[str] = frozenset(),
) -> tuple[str, str | None, str | None, float]:
    """Decide policy outcome given matched norms (already above threshold).

    Returns (decision, matched_ds, matched_ex, score) where decision ∈
    {"keep", "drop_cross", "drop_intra_same_gt"}.

    score is the best matching score among all examined records; matched_ds/ex
    point to the record that triggered the drop (or to the best same-ds-diff-gt
    record if keep).
    """
    best_intra_diff: tuple[str, str, float] | None = None  # (ds, ex, score)
    best_intra_same: tuple[str, str, float] | None = None
    best_cross:      tuple[str, str, float] | None = None
    # NOTE: matched_norms are returned in descending score order, but we
    # iterate all to find first cross/intra-same-gt (priority order).

    for matched_norm, score in _iter_with_scores(matched_norms):
        recs: list[PromptRecord] = []
        if matched_norm in prompt_records:
            recs.extend(prompt_records[matched_norm])
        if matched_norm in local_records:
            recs.extend(local_records[matched_norm])
        for (ds, ex, gt_h) in recs:
            if ds != query_ds:
                if best_cross is None or score > best_cross[2]:
                    best_cross = (ds, ex, score)
            else:
                # Intra-dataset match. If this dataset is excluded from
                # intra-dedup, ignore — neither same-GT nor diff-GT match
                # should affect the keep/drop decision.
                if query_ds in skip_intra:
                    continue
                if gt_h == query_gt_hash:
                    if best_intra_same is None or score > best_intra_same[2]:
                        best_intra_same = (ds, ex, score)
                else:
                    if best_intra_diff is None or score > best_intra_diff[2]:
                        best_intra_diff = (ds, ex, score)

    # Drop priority: cross-ds > intra-ds same GT > keep
    if best_cross is not None:
        return ("drop_cross", best_cross[0], best_cross[1], best_cross[2])
    if best_intra_same is not None:
        return ("drop_intra_same_gt", best_intra_same[0], best_intra_same[1], best_intra_same[2])
    if best_intra_diff is not None:
        return ("keep", best_intra_diff[0], best_intra_diff[1], best_intra_diff[2])
    return ("keep", None, None, 0.0)


def _iter_with_scores(matches: list[tuple[str, float]]):
    for m in matches:
        yield m[0], m[1]


def run(
    input_path:           Path,
    output_dir:           Path,
    similarity_threshold: float = 90.0,
    batch_size:           int   = 20_000,
    resume:               bool  = False,
    num_workers:          int   = 32,
    bucket_tokens:        int   = 2,
    len_bin_size:         int   = 100,
    max_candidates_per_bucket: int | None = 5_000,
    max_len_diff:         int | None = 40,
    neighbor_bin_radius:  int   = 1,
    match_limit:          int   = 20,
    skip_datasets:        tuple[str, ...] = DEFAULT_SKIP_DATASETS,
    skip_intra_datasets:  tuple[str, ...] = DEFAULT_SKIP_INTRA_DATASETS,
    checkpoint_every:     int   = 20,
    log_every:            int   = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_path       = output_dir / "rl_v1.fuzzy_dedup.kept.parquet"
    dropped_path    = output_dir / "rl_v1.fuzzy_dedup.dropped.parquet"
    report_path     = output_dir / "report.json"
    checkpoint_path = output_dir / "fuzzy_dedup_checkpoint.json"

    pf = pq.ParquetFile(input_path)
    schema     = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    required = ("dataset_id", "example_id", "context_messages", "ground_truth_text")
    for col in required:
        if col not in schema.names:
            raise ValueError(f"Input file is missing required column: '{col}'")

    skip_set       = set(skip_datasets)
    skip_intra_set = frozenset(skip_intra_datasets)

    print(f"Input : {input_path}")
    print(f"  Rows          : {total_rows:,}")
    print(f"  Row groups    : {pf.metadata.num_row_groups}")
    print(f"  Sim threshold : {similarity_threshold} (fuzz.ratio, Levenshtein)")
    print(f"  Batch size    : {batch_size:,}")
    print(f"  Workers       : {num_workers}")
    print(f"  Match limit/row : {match_limit}")
    print(f"  Skip datasets  : {len(skip_set)} (passthrough)")
    print(f"  Intra-only skip: {len(skip_intra_set)} (indexed, cross-DS still active)")
    for ds in sorted(skip_intra_set):
        print(f"      • {ds}")
    print(f"Output dir: {output_dir}")
    print()

    state = (_load_checkpoint(checkpoint_path) if resume else None) or _State()

    # Global index: norm -> list of (ds, ex, gt_hash)
    prompt_records: dict[str, list[PromptRecord]] = {}
    bucket_index:   dict[str, deque]              = {}
    prompt_count = 0

    kept_writer:    pq.ParquetWriter | None = None
    dropped_writer: pq.ParquetWriter | None = None
    t0 = time.time()

    try:
        # --- Rebuild index for resume ---
        if resume and state.processed_rows > 0:
            print(f"[resume] Rebuilding index from first {state.processed_rows:,} rows ...")
            t_rebuild = time.time()
            seen = 0
            for batch in pf.iter_batches(
                batch_size=50_000,
                columns=["dataset_id", "example_id", "context_messages", "ground_truth_text"],
            ):
                ds_ids   = batch.column("dataset_id").to_pylist()
                ex_ids   = batch.column("example_id").to_pylist()
                ctx_list = batch.column("context_messages").to_pylist()
                gt_list  = batch.column("ground_truth_text").to_pylist()
                for ds, ex, ctx, gt in zip(ds_ids, ex_ids, ctx_list, gt_list):
                    seen += 1
                    ds_s = str(ds) if ds is not None else "__MISSING__"
                    if ds_s in skip_set:
                        if seen >= state.processed_rows:
                            break
                        continue
                    norm = normalize_prompt(ctx)
                    if not norm:
                        if seen >= state.processed_rows:
                            break
                        continue
                    ex_s = str(ex) if ex is not None else "__MISSING__"
                    gt_h = gt_fingerprint(gt)
                    if norm not in prompt_records:
                        _add_to_bucket(
                            bucket_index,
                            _bucket_key(norm, bucket_tokens, len_bin_size),
                            norm,
                            max_candidates_per_bucket,
                        )
                        prompt_records[norm] = [(ds_s, ex_s, gt_h)]
                        prompt_count += 1
                    else:
                        prompt_records[norm].append((ds_s, ex_s, gt_h))
                    if seen >= state.processed_rows:
                        break
                if seen >= state.processed_rows:
                    break
            print(
                f"[resume] Index rebuilt: {prompt_count:,} unique prompts  "
                f"({time.time() - t_rebuild:.1f}s)\n"
            )

        executor = ProcessPoolExecutor(max_workers=num_workers)
        skip_rows = state.processed_rows

        try:
            for batch in pf.iter_batches(batch_size=batch_size):
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
                gt_list  = batch.column("ground_truth_text").to_pylist()

                n = len(ds_ids)

                # Per-row precomputed: dataset, normalized prompt, gt hash,
                # and a flag "is passthrough".
                ds_s_list = [str(d) if d is not None else "__MISSING__" for d in ds_ids]
                pass_flags = [ds in skip_set for ds in ds_s_list]
                norms    = [
                    "" if pass_flags[i] else normalize_prompt(ctx_list[i])
                    for i in range(n)
                ]
                gt_hashes = [gt_fingerprint(gt_list[i]) for i in range(n)]

                # Build per-batch local index (so rows in the same batch can match each other)
                local_records:      dict[str, list[PromptRecord]] = {}
                local_bucket_index: dict[str, deque]              = {}

                tasks: list[tuple[str, list[str], int | None, float, int]] = []
                for i in range(n):
                    if pass_flags[i] or not norms[i]:
                        tasks.append(("", [], max_len_diff, similarity_threshold, match_limit))
                        continue
                    norm = norms[i]
                    candidates: list[str] = []
                    head     = _bucket_head(norm, bucket_tokens)
                    base_bin = len(norm) // max(1, len_bin_size)
                    for offset in range(-neighbor_bin_radius, neighbor_bin_radius + 1):
                        bkey = _bucket_key_for_bin(head, base_bin + offset)
                        if bkey in bucket_index:
                            candidates.extend(bucket_index[bkey])
                        if bkey in local_bucket_index:
                            candidates.extend(local_bucket_index[bkey])

                    # Add to LOCAL index now (so later rows in same batch can match it)
                    rec = (ds_s_list[i], str(ex_ids[i]) if ex_ids[i] is not None else "__MISSING__", gt_hashes[i])
                    if norm not in local_records and norm not in prompt_records:
                        _add_to_bucket(
                            local_bucket_index,
                            _bucket_key(norm, bucket_tokens, len_bin_size),
                            norm,
                            max_candidates_per_bucket,
                        )
                    local_records.setdefault(norm, []).append(rec)

                    tasks.append((norm, candidates, max_len_diff, similarity_threshold, match_limit))

                # Parallel fuzzy matching
                results = list(executor.map(_find_matches, tasks, chunksize=100))

                # Decision pass
                keep_idx: list[int] = []
                drop_idx: list[int] = []
                drop_reasons:    list[str]   = []
                drop_scores:     list[float] = []
                drop_matched_ds: list[str]   = []
                drop_matched_ex: list[str]   = []

                for i in range(n):
                    ds_s = ds_s_list[i]
                    ex_s = str(ex_ids[i]) if ex_ids[i] is not None else "__MISSING__"

                    if pass_flags[i]:
                        keep_idx.append(i)
                        state.kept += 1
                        state.passthrough += 1
                        state.kept_by_ds[ds_s] += 1
                        state.passthrough_by_ds[ds_s] += 1
                        continue

                    norm = norms[i]
                    if not norm:
                        keep_idx.append(i)
                        state.kept += 1
                        state.kept_by_ds[ds_s] += 1
                        continue

                    matches = results[i]
                    # Exclude self-matches via prompt_records (own batch self-entry won't appear
                    # because we only added to local_records, not local_bucket_index, when prompt
                    # was already in global). Need to ensure we don't classify our own record.
                    # Implementation detail: local_records[norm] contains the row itself if it's
                    # the first of that norm in the batch. _classify_match handles this by
                    # treating it as "intra, same GT" → drop. Prevent that:
                    matches_no_self = []
                    for (mn, sc) in matches:
                        # Skip if matched is exactly our own norm AND only record present is us.
                        if mn == norm:
                            # Check whether any OTHER record exists for this norm
                            other = (
                                len(prompt_records.get(mn, [])) > 0
                                or len([r for r in local_records.get(mn, []) if r[1] != ex_s or r[0] != ds_s]) > 0
                            )
                            if not other:
                                continue
                        matches_no_self.append((mn, sc))

                    decision, m_ds, m_ex, score = _classify_match(
                        ds_s, gt_hashes[i], matches_no_self, prompt_records, local_records,
                        skip_intra=skip_intra_set,
                    )

                    if decision == "drop_cross":
                        drop_idx.append(i)
                        drop_reasons.append("fuzzy_cross_dataset")
                        drop_scores.append(score)
                        drop_matched_ds.append(m_ds or "")
                        drop_matched_ex.append(m_ex or "")
                        state.dropped += 1
                        state.dropped_cross += 1
                        state.dropped_by_ds[ds_s] += 1
                        state.dropped_cross_by_ds[ds_s] += 1
                        if m_ds:
                            state.dropped_pairs[(ds_s, m_ds)] += 1
                    elif decision == "drop_intra_same_gt":
                        drop_idx.append(i)
                        drop_reasons.append("fuzzy_intra_dataset_same_gt")
                        drop_scores.append(score)
                        drop_matched_ds.append(m_ds or "")
                        drop_matched_ex.append(m_ex or "")
                        state.dropped += 1
                        state.dropped_intra_same_gt += 1
                        state.dropped_by_ds[ds_s] += 1
                        state.dropped_intra_by_ds[ds_s] += 1
                    else:
                        keep_idx.append(i)
                        state.kept += 1
                        state.kept_by_ds[ds_s] += 1
                        if decision == "keep" and m_ds is not None:
                            # was an intra-ds diff-GT match
                            state.intra_kept_diff_gt += 1
                            state.intra_kept_diff_gt_by_ds[ds_s] += 1
                        # Promote to global index (kept rows participate in future matching)
                        rec = (ds_s, ex_s, gt_hashes[i])
                        if norm not in prompt_records:
                            _add_to_bucket(
                                bucket_index,
                                _bucket_key(norm, bucket_tokens, len_bin_size),
                                norm,
                                max_candidates_per_bucket,
                            )
                            prompt_records[norm] = [rec]
                            prompt_count += 1
                        else:
                            prompt_records[norm].append(rec)

                # Write
                raw_table = pa.Table.from_batches([batch], schema=schema)

                if keep_idx:
                    kept_t = raw_table.take(pa.array(keep_idx, pa.int64()))
                    if kept_writer is None:
                        kept_writer = pq.ParquetWriter(str(kept_path), kept_t.schema, compression="snappy")
                    kept_writer.write_table(kept_t)

                if drop_idx:
                    drop_t = raw_table.take(pa.array(drop_idx, pa.int64()))
                    drop_t = drop_t.append_column("_drop_reason",           pa.array(drop_reasons))
                    drop_t = drop_t.append_column("_fuzzy_score",           pa.array(drop_scores))
                    drop_t = drop_t.append_column("_fuzzy_matched_dataset", pa.array(drop_matched_ds))
                    drop_t = drop_t.append_column("_fuzzy_matched_example", pa.array(drop_matched_ex))
                    if dropped_writer is None:
                        dropped_writer = pq.ParquetWriter(str(dropped_path), drop_t.schema, compression="snappy")
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
                        f"kept={state.kept:,} drop={state.dropped:,} "
                        f"(cross={state.dropped_cross:,} intra={state.dropped_intra_same_gt:,}) | "
                        f"idx={prompt_count:,} | "
                        f"{rate:,.0f} r/s ETA ~{eta:.0f}s"
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

    report = {
        "stage":              "3C_fuzzy_dedup_rl",
        "generated_at_utc":   datetime.now(timezone.utc).isoformat(),
        "input_path":         str(input_path),
        "method":             "rapidfuzz fuzz.ratio (Levenshtein, normalized 0-100)",
        "hash_source":        "last_user_message(context_messages)",
        "gt_fingerprint":     "sha1(strip(ground_truth_text))",
        "similarity_threshold": similarity_threshold,
        "batch_size":         batch_size,
        "num_workers":        num_workers,
        "match_limit_per_row": match_limit,
        "policy": {
            "drop_cross_dataset_duplicates":   True,
            "drop_intra_dataset_when_same_gt": True,
            "keep_intra_dataset_when_diff_gt": True,
            "gt_comparison":                   "exact (not fuzzy)",
            "skip_datasets_passthrough":       sorted(skip_set),
            "skip_intra_dedup_datasets":       sorted(skip_intra_set),
            "prompt_normalization":            ["lowercase", "remove_punctuation", "collapse_whitespace"],
            "gt_normalization":                ["strip_only"],
        },
        "bucket_config": {
            "bucket_tokens":              bucket_tokens,
            "len_bin_size":               len_bin_size,
            "max_candidates_per_bucket":  max_candidates_per_bucket,
            "max_len_diff":               max_len_diff,
            "neighbor_bin_radius":        neighbor_bin_radius,
        },
        "total_rows_in":         total_rows,
        "total_kept":            state.kept,
        "total_dropped":         state.dropped,
        "dropped_cross_dataset": state.dropped_cross,
        "dropped_intra_same_gt": state.dropped_intra_same_gt,
        "intra_kept_diff_gt":    state.intra_kept_diff_gt,
        "passthrough":           state.passthrough,
        "drop_rate_pct":         round(state.dropped / total_rows * 100, 4) if total_rows else 0,
        "prompts_indexed":       prompt_count,
        "elapsed_seconds":       round(elapsed, 1),
        "kept_by_dataset":          dict(sorted(state.kept_by_ds.items())),
        "dropped_by_dataset":       dict(sorted(state.dropped_by_ds.items(), key=lambda x: -x[1])),
        "dropped_cross_by_dataset": dict(sorted(state.dropped_cross_by_ds.items(), key=lambda x: -x[1])),
        "dropped_intra_by_dataset": dict(sorted(state.dropped_intra_by_ds.items(), key=lambda x: -x[1])),
        "intra_kept_diff_gt_by_dataset": dict(sorted(state.intra_kept_diff_gt_by_ds.items(), key=lambda x: -x[1])),
        "passthrough_by_dataset":   dict(sorted(state.passthrough_by_ds.items())),
        "dropped_cross_pairs_top50": [
            {"dropped_from": a, "matched_in": b, "count": c}
            for (a, b), c in state.dropped_pairs.most_common(50)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 70}")
    print(f"Stage 3C complete  ({elapsed:.1f}s)")
    print(f"  Total in        : {total_rows:,}")
    print(f"  Kept            : {state.kept:,}")
    print(f"    of which passthrough : {state.passthrough:,}")
    print(f"    of which intra-diff-GT: {state.intra_kept_diff_gt:,}")
    print(f"  Dropped         : {state.dropped:,}  ({report['drop_rate_pct']}%)")
    print(f"    cross-dataset           : {state.dropped_cross:,}")
    print(f"    intra-dataset same GT   : {state.dropped_intra_same_gt:,}")
    if state.dropped_by_ds:
        print("  Dropped by dataset (top 10):")
        for ds, cnt in state.dropped_by_ds.most_common(10):
            print(f"    {ds:<60s}  {cnt:>8,}")
    print(f"\nOutputs:")
    print(f"  {kept_path}")
    if dropped_path.exists():
        print(f"  {dropped_path}")
    print(f"  {report_path}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 3C — RL fuzzy dedup. RapidFuzz fuzz.ratio on the last user "
            "message. Cross-ds matches always drop; intra-ds matches drop only "
            "when ground_truth_text is exactly equal (after strip). SLR-Bench "
            "subsets are passthrough."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,
                   help="Input parquet (Stage 3B exact-dedup kept file).")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write outputs into.")
    p.add_argument("--similarity-threshold", type=float, default=90.0,
                   help="fuzz.ratio threshold (0–100). Default 90 (per user request).")
    p.add_argument("--batch-size",    type=int, default=20_000)
    p.add_argument("--num-workers",   type=int, default=32)
    p.add_argument("--bucket-tokens", type=int, default=2)
    p.add_argument("--len-bin-size",  type=int, default=100)
    p.add_argument("--max-candidates-per-bucket", type=int, default=5_000)
    p.add_argument("--max-len-diff",  type=int, default=40,
                   help="Skip candidates where |len(query) - len(candidate)| > this. -1 = disabled.")
    p.add_argument("--neighbor-bin-radius", type=int, default=1)
    p.add_argument("--match-limit",   type=int, default=20,
                   help="Max number of above-threshold matches to inspect per row.")
    p.add_argument("--skip-datasets", nargs="*", default=list(DEFAULT_SKIP_DATASETS),
                   help="dataset_id values to pass through (no hashing/matching at all).")
    p.add_argument("--skip-intra-datasets", nargs="*", default=list(DEFAULT_SKIP_INTRA_DATASETS),
                   help=("dataset_id values to exclude from INTRA-dataset dedup only "
                         "(prompts are still indexed so cross-DS dedup can find them). "
                         "Default: logi_glue, because its templated multiple-choice / NLI "
                         "items have ~95%% shared boilerplate and a tiny GT label space, "
                         "so the GT-aware policy cannot reliably distinguish near-duplicates "
                         "from legitimately distinct tasks. See THRESHOLD_DECISION.md."))
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint in --output-dir.")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--log-every",        type=int, default=5)
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
        match_limit          = args.match_limit,
        skip_datasets        = tuple(args.skip_datasets),
        skip_intra_datasets  = tuple(args.skip_intra_datasets),
        checkpoint_every     = args.checkpoint_every,
        log_every            = args.log_every,
    )


if __name__ == "__main__":
    main()
