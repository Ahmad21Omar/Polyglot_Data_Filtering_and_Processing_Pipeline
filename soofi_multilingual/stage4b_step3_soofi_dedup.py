"""Stage 4B (Step 3) — Soofi Multilingual Intra-Language Deduplication.

Runs a two-pass deduplication over each non-English Soofi language split
produced by Stage 4B Step 2 (quality filtering).  The two passes reproduce
the original SFT-Collection-v2 deduplication approach for multilingual data:

  Pass 1 — Exact dedup  (SHA-1 hash of normalised context_messages)
  Pass 2 — Fuzzy dedup  (RapidFuzz fuzz.ratio, Levenshtein, 98% threshold)

Both passes share the same source policy
-----------------------------------------
The Soofi multilingual dataset contains rows from multiple sub-sources
within a single language split.  The ``source_dataset_id`` column records
the sub-source (e.g. ``"HuggingFaceH4/aya_dataset/12345"``).  The *source
name* is the part before the last ``/``.

  Same source name  →  KEEP   (within-source duplicates are expected and fine)
  Cross-source      →  DROP   (same or near-identical prompt from a different subsource)

This mirrors the policy used in Stage 3B / 3C for the main English corpus,
adapted to use ``source_dataset_id`` instead of ``dataset_id``.

Pass 1 — Exact dedup
--------------------
Normalises context_messages (lowercase, strip punctuation, collapse whitespace),
computes SHA-1, and drops any row whose hash was already seen in a *different*
sub-source.  Uses SQLite for the hash index so the pass can checkpoint and
resume without re-reading the entire file.  Adds three audit columns to the
kept parquet:

  _dedup_hash           SHA-1 of the normalised prompt
  _dedup_prompt_norm    Normalised prompt text
  _dedup_seen_in_source Source name of the first row that introduced this hash
                        (empty string for the first occurrence)

These columns are read by Pass 2.

Pass 2 — Fuzzy dedup
--------------------
Uses RapidFuzz ``fuzz.ratio`` (Levenshtein) with a configurable threshold
(default 98 %).  Candidate lookup is accelerated with a bucket index keyed on
the first two non-role tokens of the prompt and a length-bin (binwidth 100
chars), reducing O(n²) comparisons to a small candidate set per row.
Parallel workers (``--num-workers``, default 8) handle the RapidFuzz comparisons.

Input (--input-root)
--------------------
The output directory of Stage 4B Step 2B (stage4b_step2b_soofi_quality_filter.py):

    <input-root>/
        german/kept.parquet
        french/kept.parquet
        italian/kept.parquet
        spanish/kept.parquet

Output (--output-dir)
---------------------
    <output-dir>/
        german/
            exact/
                kept.parquet            ← after pass 1
                dropped.parquet         ← exact cross-source duplicates
                report.json
            fuzzy/
                kept.parquet            ← final deduped output (use this downstream)
                dropped.parquet         ← fuzzy cross-source duplicates
                report.json
        french/  ...
        italian/ ...
        spanish/ ...
        summary.json                    ← combined stats

Usage
-----
    python stage4b_step3_soofi_dedup.py \\
        --input-root /path/to/stage4b_quality_filter \\
        --output-dir /path/to/stage4b_dedup

    # Smoke test — first 5 000 rows per language:
    python stage4b_step3_soofi_dedup.py \\
        --input-root /path/to/stage4b_quality_filter \\
        --output-dir /tmp/stage4b_dedup_test \\
        --max-rows   5000

    # Resume after interruption (both passes support resume):
    python stage4b_step3_soofi_dedup.py \\
        --input-root /path/to/stage4b_quality_filter \\
        --output-dir /path/to/stage4b_dedup \\
        --resume

Optional flags
--------------
  --languages           Comma-separated subfolder names (default: german,french,italian,spanish)
  --similarity-threshold Fuzzy threshold 0–100 (default: 98.0)
  --hash-algo           sha1 / md5 / sha256 (default: sha1)
  --batch-size-exact    Exact-pass batch size (default: 50 000)
  --batch-size-fuzzy    Fuzzy-pass batch size (default: 20 000)
  --num-workers         Parallel workers for fuzzy matching (default: 8)
  --max-rows            Stop after N rows per language (smoke test)
  --resume              Resume both passes from checkpoints if they exist
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import queue
import re
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Shared normalisation helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")
_ROLE_TOKENS = frozenset({"user", "assistant", "system"})


def _extract_context(context_messages: Any) -> str:
    """Concatenate all messages with role labels."""
    if not isinstance(context_messages, list):
        return ""
    parts = []
    for msg in context_messages:
        if isinstance(msg, dict):
            role    = str(msg.get("role", "unknown")).lower()
            content = str(msg.get("content") or "")
            if content.strip():
                parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _normalize(text: str) -> str:
    x = (text or "").lower().strip()
    x = _PUNCT_RE.sub(" ", x)
    x = _WS_RE.sub(" ", x).strip()
    return x


def _source_name(source_dataset_id: str) -> str:
    """Return the sub-source name: part of source_dataset_id before the last '/'."""
    sid = source_dataset_id or "__MISSING__"
    return sid.rsplit("/", 1)[0] if "/" in sid else sid


def _iter_batches(
    path: Path,
    columns: List[str],
    batch_size: int,
    max_rows: Optional[int] = None,
    skip_rows: int = 0,
) -> Iterable[pa.RecordBatch]:
    pf = pq.ParquetFile(path)
    yielded = 0
    skipped = 0
    for b in pf.iter_batches(columns=columns, batch_size=batch_size):
        if skipped < skip_rows:
            if skipped + b.num_rows <= skip_rows:
                skipped += b.num_rows
                continue
            b = b.slice(skip_rows - skipped)
            skipped = skip_rows
        if max_rows is not None and yielded >= max_rows:
            break
        if max_rows is not None and yielded + b.num_rows > max_rows:
            b = b.slice(0, max_rows - yielded)
        yielded += b.num_rows
        yield b


# ---------------------------------------------------------------------------
# Pass 1 — Exact SHA-1 deduplication
# ---------------------------------------------------------------------------

def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hash_index "
        "(hash TEXT PRIMARY KEY, source_name TEXT, example_id TEXT, prompt_norm TEXT)"
    )
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _db_lookup(h: str, cache: Dict[str, Tuple], conn: sqlite3.Connection) -> Optional[Tuple[str, str, str]]:
    if h in cache:
        return cache[h]
    row = conn.execute(
        "SELECT source_name, example_id, prompt_norm FROM hash_index WHERE hash = ?", (h,)
    ).fetchone()
    if row:
        cache[h] = (row[0], row[1], row[2])
        return cache[h]
    return None


def _db_insert(h: str, src: str, eid: str, norm: str, cache: Dict, conn: sqlite3.Connection) -> None:
    cache[h] = (src, eid, norm)
    conn.execute(
        "INSERT OR REPLACE INTO hash_index (hash, source_name, example_id, prompt_norm) VALUES (?,?,?,?)",
        (h, src, eid, norm),
    )


@dataclass
class _ExactState:
    processed_rows: int = 0
    batch_index:    int = 0
    total:   int = 0
    kept:    int = 0
    dropped: int = 0
    same_source_kept: int = 0


def _save_exact_checkpoint(path: Path, s: _ExactState, t0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "saved_at_utc":    datetime.now(timezone.utc).isoformat(),
        "elapsed_sec":     time.time() - t0,
        "processed_rows":  s.processed_rows,
        "batch_index":     s.batch_index,
        "total":           s.total,
        "kept":            s.kept,
        "dropped":         s.dropped,
        "same_source_kept": s.same_source_kept,
    }, indent=2, ensure_ascii=False))


def _load_exact_checkpoint(path: Path) -> Optional[_ExactState]:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    s = _ExactState()
    s.processed_rows  = d.get("processed_rows", 0)
    s.batch_index     = d.get("batch_index", 0)
    s.total           = d.get("total", 0)
    s.kept            = d.get("kept", 0)
    s.dropped         = d.get("dropped", 0)
    s.same_source_kept= d.get("same_source_kept", 0)
    return s


def run_exact_dedup(
    input_path:   Path,
    output_dir:   Path,
    hash_algo:    str  = "sha1",
    batch_size:   int  = 50_000,
    max_rows:     Optional[int] = None,
    resume:       bool = False,
    checkpoint_every: int = 20,
    log_every:        int = 5,
) -> Dict:
    """Pass 1: exact SHA-1 dedup within one language split.

    Returns a summary dict with n_in / n_kept / n_dropped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_path    = output_dir / "kept.parquet"
    dropped_path = output_dir / "dropped.parquet"
    report_path  = output_dir / "report.json"
    ckpt_path    = output_dir / "checkpoint.json"
    db_path      = output_dir / "hash_index.sqlite"

    state = (_load_exact_checkpoint(ckpt_path) if resume else None) or _ExactState()
    cache: Dict[str, Tuple] = {}
    conn  = _open_db(db_path)

    kept_by_src:    Counter = Counter()
    dropped_by_src: Counter = Counter()
    dropped_pairs:  Counter = Counter()

    kept_writer:    Optional[pq.ParquetWriter] = None
    dropped_writer: Optional[pq.ParquetWriter] = None
    t0 = time.time()

    cols = ["source_dataset_id", "example_id", "context_messages"]

    try:
        for batch in _iter_batches(input_path, cols, batch_size, max_rows, state.processed_rows):
            state.batch_index += 1

            srcs  = batch.column("source_dataset_id").to_pylist()
            eids  = batch.column("example_id").to_pylist()
            ctxs  = batch.column("context_messages").to_pylist()

            keep_idx: List[int] = []
            drop_idx: List[int] = []
            keep_hashes: List[str] = []
            keep_norms:  List[str] = []
            keep_seen:   List[str] = []
            drop_hashes: List[str] = []
            drop_norms:  List[str] = []
            drop_matched_src: List[str] = []
            drop_matched_eid: List[str] = []
            drop_matched_norm:List[str] = []
            drop_reasons:     List[str] = []

            for i, (src, eid, ctx) in enumerate(zip(srcs, eids, ctxs)):
                state.total += 1
                state.processed_rows += 1
                src_name  = _source_name(str(src) if src is not None else "")
                norm      = _normalize(_extract_context(ctx))
                h         = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest() \
                            if hash_algo == "sha1" else \
                            hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest() \
                            if hash_algo == "md5" else \
                            hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()

                seen = _db_lookup(h, cache, conn)

                if seen is None:
                    _db_insert(h, src_name, str(eid), norm, cache, conn)
                    keep_idx.append(i)
                    keep_hashes.append(h)
                    keep_norms.append(norm)
                    keep_seen.append("")
                    state.kept += 1
                    kept_by_src[src_name] += 1
                else:
                    seen_src, seen_eid, seen_norm = seen
                    if seen_src == src_name:
                        keep_idx.append(i)
                        keep_hashes.append(h)
                        keep_norms.append(norm)
                        keep_seen.append(seen_src)
                        state.kept += 1
                        state.same_source_kept += 1
                        kept_by_src[src_name] += 1
                    else:
                        drop_idx.append(i)
                        drop_hashes.append(h)
                        drop_norms.append(norm)
                        drop_matched_src.append(seen_src)
                        drop_matched_eid.append(seen_eid)
                        drop_matched_norm.append(seen_norm)
                        drop_reasons.append("exact_hash_cross_source")
                        state.dropped += 1
                        dropped_by_src[src_name] += 1
                        dropped_pairs[(src_name, seen_src)] += 1

            raw = pa.Table.from_batches([batch])

            if keep_idx:
                kb = raw.take(pa.array(keep_idx, type=pa.int64()))
                kb = kb.append_column("_dedup_hash",           pa.array(keep_hashes))
                kb = kb.append_column("_dedup_prompt_norm",    pa.array(keep_norms))
                kb = kb.append_column("_dedup_seen_in_source", pa.array(keep_seen))
                if kept_writer is None:
                    kept_writer = pq.ParquetWriter(str(kept_path), kb.schema)
                kept_writer.write_table(kb)

            if drop_idx:
                db_ = raw.take(pa.array(drop_idx, type=pa.int64()))
                db_ = db_.append_column("_drop_reason",              pa.array(drop_reasons))
                db_ = db_.append_column("_dedup_hash",               pa.array(drop_hashes))
                db_ = db_.append_column("_dedup_prompt_norm",        pa.array(drop_norms))
                db_ = db_.append_column("_dedup_matched_source",     pa.array(drop_matched_src))
                db_ = db_.append_column("_dedup_matched_example_id", pa.array(drop_matched_eid))
                db_ = db_.append_column("_dedup_matched_prompt_norm",pa.array(drop_matched_norm))
                if dropped_writer is None:
                    dropped_writer = pq.ParquetWriter(str(dropped_path), db_.schema)
                dropped_writer.write_table(db_)

            if state.batch_index % log_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"    [exact] batch={state.batch_index} "
                    f"rows={state.total:,} kept={state.kept:,} dropped={state.dropped:,} "
                    f"({state.total/elapsed:,.0f} rows/s)",
                    flush=True,
                )
            if state.batch_index % checkpoint_every == 0:
                _save_exact_checkpoint(ckpt_path, state, t0)
                conn.commit()

    finally:
        conn.commit()
        conn.close()
        if kept_writer    is not None: kept_writer.close()
        if dropped_writer is not None: dropped_writer.close()

    report = {
        "pass":        "exact",
        "hash_algo":   hash_algo,
        "policy":      "cross_source_drop / same_source_keep",
        "source_key":  "source_dataset_id (before last '/')",
        "n_in":        state.total,
        "n_kept":      state.kept,
        "n_dropped":   state.dropped,
        "same_source_duplicates_kept": state.same_source_kept,
        "drop_rate_pct": round(state.dropped / state.total * 100, 4) if state.total else 0,
        "kept_by_source":    dict(sorted(kept_by_src.items())),
        "dropped_by_source": dict(sorted(dropped_by_src.items())),
        "dropped_cross_pairs": [
            {"dropped_source": a, "kept_source": b, "count": c}
            for (a, b), c in dropped_pairs.most_common()
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# ---------------------------------------------------------------------------
# Pass 2 — Fuzzy RapidFuzz deduplication
# ---------------------------------------------------------------------------

def _bucket_head(text: str, n_tokens: int) -> str:
    if not text:
        return "__EMPTY__"
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
    index: Dict[str, collections.deque],
    key: str,
    norm: str,
    max_per_bucket: Optional[int],
) -> None:
    dq = index.setdefault(key, collections.deque())
    dq.append(norm)
    if max_per_bucket and len(dq) > max_per_bucket:
        dq.popleft()


def _worker_find_match(
    args: Tuple[str, List[str], Optional[int], float]
) -> Tuple[Optional[str], float]:
    p_norm, candidates, len_diff, sim_thres = args
    if not p_norm or not candidates:
        return None, 0.0
    if len_diff is not None:
        p_len = len(p_norm)
        candidates = [c for c in candidates if abs(len(c) - p_len) <= len_diff]
    if not candidates:
        return None, 0.0
    from rapidfuzz import fuzz, process as rf_process
    result = rf_process.extractOne(p_norm, candidates, scorer=fuzz.ratio, score_cutoff=sim_thres)
    if result:
        return result[0], result[1]
    return None, 0.0


@dataclass
class _FuzzyState:
    processed_rows: int = 0
    batch_index:    int = 0
    total:   int = 0
    kept:    int = 0
    dropped: int = 0


def _save_fuzzy_checkpoint(path: Path, s: _FuzzyState, t0: float,
                            kept_by_src: Counter, dropped_by_src: Counter,
                            dropped_pairs: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "saved_at_utc":    datetime.now(timezone.utc).isoformat(),
        "elapsed_sec":     time.time() - t0,
        "processed_rows":  s.processed_rows,
        "batch_index":     s.batch_index,
        "total":           s.total,
        "kept":            s.kept,
        "dropped":         s.dropped,
        "kept_by_source":    dict(kept_by_src),
        "dropped_by_source": dict(dropped_by_src),
        "dropped_pairs":  [
            {"dropped_source": a, "kept_source": b, "count": c}
            for (a, b), c in dropped_pairs.items()
        ],
    }, indent=2, ensure_ascii=False))


def _load_fuzzy_checkpoint(path: Path) -> Optional[_FuzzyState]:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    s = _FuzzyState()
    s.processed_rows = d.get("processed_rows", 0)
    s.batch_index    = d.get("batch_index", 0)
    s.total          = d.get("total", 0)
    s.kept           = d.get("kept", 0)
    s.dropped        = d.get("dropped", 0)
    return s


def run_fuzzy_dedup(
    input_path:   Path,
    output_dir:   Path,
    similarity_threshold: float = 98.0,
    batch_size:   int  = 20_000,
    max_rows:     Optional[int] = None,
    resume:       bool = False,
    num_workers:  int  = 8,
    checkpoint_every: int = 20,
    log_every:        int = 5,
    max_prompts_in_memory: int = 2_000_000,
    bucket_tokens:         int = 2,
    len_bin_size:          int = 100,
    max_candidates_per_bucket: int = 5_000,
    length_diff_threshold: int = 40,
    neighbor_bin_radius:   int = 1,
) -> Dict:
    """Pass 2: fuzzy dedup within one language split.

    Expects _dedup_hash / _dedup_prompt_norm / _dedup_seen_in_source columns
    to be present (added by run_exact_dedup).

    Returns a summary dict with n_in / n_kept / n_dropped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_path    = output_dir / "kept.parquet"
    dropped_path = output_dir / "dropped.parquet"
    report_path  = output_dir / "report.json"
    ckpt_path    = output_dir / "checkpoint.json"

    state = (_load_fuzzy_checkpoint(ckpt_path) if resume else None) or _FuzzyState()
    kept_by_src:    Counter = Counter()
    dropped_by_src: Counter = Counter()
    dropped_pairs:  Counter = Counter()

    bucket_index:   Dict[str, collections.deque] = {}
    prompt_source:  Dict[str, Tuple[str, str]]   = {}
    prompt_count    = 0

    executor = ProcessPoolExecutor(max_workers=num_workers)
    wq: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=16)
    _SHUTDOWN = "__SHUTDOWN__"

    kept_writer:    Optional[pq.ParquetWriter] = None
    dropped_writer: Optional[pq.ParquetWriter] = None

    def _writer_fn(q: "queue.Queue") -> None:
        nonlocal kept_writer, dropped_writer
        while True:
            item = q.get()
            try:
                if item is None or item[0] == _SHUTDOWN:
                    break
                tag, table = item
                if tag == "kept":
                    if kept_writer is None:
                        kept_writer = pq.ParquetWriter(str(kept_path), table.schema)
                    kept_writer.write_table(table)
                elif tag == "drop":
                    if dropped_writer is None:
                        dropped_writer = pq.ParquetWriter(str(dropped_path), table.schema)
                    dropped_writer.write_table(table)
            finally:
                q.task_done()

    writer_thread = threading.Thread(target=_writer_fn, args=(wq,), daemon=True)
    writer_thread.start()

    cols = [
        "source_dataset_id", "example_id", "context_messages",
        "_dedup_hash", "_dedup_prompt_norm", "_dedup_seen_in_source",
    ]
    t0 = time.time()

    # Rebuild bucket index for resume
    if resume and state.processed_rows > 0:
        print(f"    [fuzzy][resume] rebuilding index from {state.processed_rows:,} rows…", flush=True)
        for b in _iter_batches(input_path, cols, batch_size, state.processed_rows):
            srcs  = b.column("source_dataset_id").to_pylist()
            eids  = b.column("example_id").to_pylist()
            norms = b.column("_dedup_prompt_norm").to_pylist()
            for src, eid, norm in zip(srcs, eids, norms):
                if norm and norm not in prompt_source:
                    src_name = _source_name(str(src) if src is not None else "")
                    key = _bucket_key(norm, bucket_tokens, len_bin_size)
                    _add_to_bucket(bucket_index, key, norm, max_candidates_per_bucket)
                    prompt_source[norm] = (src_name, str(eid))
                    prompt_count += 1
        print(f"    [fuzzy][resume] index rebuilt: {prompt_count:,} prompts", flush=True)

    try:
        for batch in _iter_batches(input_path, cols, batch_size, max_rows, state.processed_rows):
            state.batch_index += 1

            srcs  = batch.column("source_dataset_id").to_pylist()
            eids  = batch.column("example_id").to_pylist()
            norms = batch.column("_dedup_prompt_norm").to_pylist()

            local_index:  Dict[str, collections.deque]  = {}
            local_source: Dict[str, Tuple[str, str]]    = {}
            tasks: List[Tuple] = []

            for src, eid, norm in zip(srcs, eids, norms):
                candidates: List[str] = []
                if norm:
                    head     = _bucket_head(norm, bucket_tokens)
                    base_bin = len(norm) // max(1, len_bin_size)
                    for offset in range(-neighbor_bin_radius, neighbor_bin_radius + 1):
                        bkey = _bucket_key_for_bin(head, base_bin + offset)
                        dq = bucket_index.get(bkey)
                        if dq:
                            candidates.extend(list(dq))
                        dq2 = local_index.get(bkey)
                        if dq2:
                            candidates.extend(list(dq2))
                    if norm not in local_source:
                        src_name = _source_name(str(src) if src is not None else "")
                        bkey2 = _bucket_key(norm, bucket_tokens, len_bin_size)
                        _add_to_bucket(local_index, bkey2, norm, max_candidates_per_bucket)
                        local_source[norm] = (src_name, str(eid))
                tasks.append((norm, candidates, length_diff_threshold, similarity_threshold))

            results = list(executor.map(_worker_find_match, tasks, chunksize=100))

            keep_idx: List[int] = []
            drop_idx: List[int] = []
            drop_reasons:      List[str]   = []
            drop_scores:       List[float] = []
            drop_match_src:    List[str]   = []
            drop_match_eid:    List[str]   = []
            drop_match_norms:  List[str]   = []
            drop_own_norms:    List[str]   = []

            for i, (src, eid, norm, (matched_norm, score)) in enumerate(
                zip(srcs, eids, norms, results)
            ):
                state.total          += 1
                state.processed_rows += 1
                src_name = _source_name(str(src) if src is not None else "")

                matched_src = ""
                matched_eid_str = ""

                if matched_norm:
                    info = prompt_source.get(matched_norm) or local_source.get(matched_norm)
                    if info:
                        matched_src, matched_eid_str = info
                    if matched_src == src_name:
                        matched_src = ""

                if matched_src:
                    drop_idx.append(i)
                    drop_reasons.append("fuzzy_duplicate")
                    drop_scores.append(float(score))
                    drop_match_src.append(matched_src)
                    drop_match_eid.append(matched_eid_str)
                    drop_match_norms.append(matched_norm or "")
                    drop_own_norms.append(norm or "")
                    state.dropped += 1
                    dropped_by_src[src_name] += 1
                    dropped_pairs[(src_name, matched_src)] += 1
                else:
                    keep_idx.append(i)
                    state.kept += 1
                    kept_by_src[src_name] += 1
                    if norm and norm not in prompt_source:
                        if prompt_count >= max_prompts_in_memory:
                            raise RuntimeError(
                                f"RAM guard: {prompt_count:,} prompts in memory. "
                                "Increase --max-prompts-in-memory."
                            )
                        bkey = _bucket_key(norm, bucket_tokens, len_bin_size)
                        _add_to_bucket(bucket_index, bkey, norm, max_candidates_per_bucket)
                        prompt_source[norm] = (src_name, str(eid))
                        prompt_count += 1

            raw = pa.Table.from_batches([batch])

            if keep_idx:
                kb = raw.take(pa.array(keep_idx, type=pa.int64()))
                wq.put(("kept", kb), block=True, timeout=30)

            if drop_idx:
                db_ = raw.take(pa.array(drop_idx, type=pa.int64()))
                db_ = db_.append_column("_fuzzy_drop_reason",          pa.array(drop_reasons))
                db_ = db_.append_column("_fuzzy_match_score",          pa.array(drop_scores))
                db_ = db_.append_column("_fuzzy_matched_source",       pa.array(drop_match_src))
                db_ = db_.append_column("_fuzzy_matched_example_id",   pa.array(drop_match_eid))
                db_ = db_.append_column("_fuzzy_prompt_norm",          pa.array(drop_own_norms))
                db_ = db_.append_column("_fuzzy_matched_prompt_norm",  pa.array(drop_match_norms))
                wq.put(("drop", db_), block=True, timeout=30)

            if state.batch_index % log_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"    [fuzzy] batch={state.batch_index} "
                    f"rows={state.total:,} kept={state.kept:,} dropped={state.dropped:,} "
                    f"indexed={prompt_count:,} ({state.total/elapsed:,.0f} rows/s)",
                    flush=True,
                )
            if state.batch_index % checkpoint_every == 0:
                _save_fuzzy_checkpoint(ckpt_path, state, t0, kept_by_src, dropped_by_src, dropped_pairs)

    finally:
        try:
            wq.put((_SHUTDOWN, None), block=True, timeout=5)
            wq.join()
        except Exception:
            pass
        if kept_writer    is not None: kept_writer.close()
        if dropped_writer is not None: dropped_writer.close()
        executor.shutdown(wait=False)

    report = {
        "pass":                 "fuzzy",
        "similarity_metric":    "fuzz.ratio (Levenshtein)",
        "similarity_threshold": similarity_threshold,
        "policy":               "cross_source_drop / same_source_keep",
        "source_key":           "source_dataset_id (before last '/')",
        "n_in":                 state.total,
        "n_kept":               state.kept,
        "n_dropped":            state.dropped,
        "drop_rate_pct":        round(state.dropped / state.total * 100, 4) if state.total else 0,
        "unique_prompts_indexed": prompt_count,
        "kept_by_source":        dict(sorted(kept_by_src.items())),
        "dropped_by_source":     dict(sorted(dropped_by_src.items())),
        "dropped_cross_pairs": [
            {"dropped_source": a, "kept_source": b, "count": c}
            for (a, b), c in dropped_pairs.most_common()
        ],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# ---------------------------------------------------------------------------
# Multi-language CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4B step 3: Intra-language two-pass deduplication (exact SHA-1 + "
            "fuzzy RapidFuzz) for non-English Soofi splits. "
            "Processes each language independently."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-root", required=True,
        help=(
            "Root directory of Stage 4B quality-filter output "
            "(contains german/kept.parquet, french/kept.parquet, …)."
        ),
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write per-language dedup outputs and summary.json.",
    )
    p.add_argument(
        "--languages", default="german,french,italian,spanish",
        help="Comma-separated language subfolder names to process.",
    )
    p.add_argument("--similarity-threshold", type=float, default=98.0,
                   help="Fuzzy similarity threshold (fuzz.ratio, 0–100).")
    p.add_argument("--hash-algo", default="sha1", choices=["sha1", "md5", "sha256"],
                   help="Hash algorithm for exact pass.")
    p.add_argument("--batch-size-exact", type=int, default=50_000,
                   help="Batch size for exact dedup pass.")
    p.add_argument("--batch-size-fuzzy", type=int, default=20_000,
                   help="Batch size for fuzzy dedup pass.")
    p.add_argument("--num-workers",  type=int, default=8,
                   help="Parallel workers for fuzzy RapidFuzz comparisons.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Stop after N rows per language (smoke test).")
    p.add_argument("--resume", action="store_true",
                   help="Resume both passes from checkpoints if they exist.")
    args = p.parse_args(argv)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    languages  = [s.strip() for s in args.languages.split(",") if s.strip()]

    if not languages:
        p.error("--languages is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    combined: dict = {
        "stage":      "4B_soofi_dedup",
        "languages":  languages,
        "similarity_threshold": args.similarity_threshold,
        "hash_algo":  args.hash_algo,
        "per_language": {},
        "totals": {"n_in": 0, "n_exact_kept": 0, "n_exact_dropped": 0,
                   "n_fuzzy_kept": 0, "n_fuzzy_dropped": 0},
    }

    t_global = time.time()

    for lang in languages:
        lang_in = input_root / lang / "kept.parquet"
        if not lang_in.exists():
            print(f"\n  [SKIP] {lang}: kept.parquet not found: {lang_in}", flush=True)
            continue

        lang_out = output_dir / lang
        print(f"\n{'='*60}", flush=True)
        print(f"Language: {lang}  →  {lang_out}", flush=True)

        # -- Pass 1: Exact --
        print(f"  Pass 1 — exact SHA-1 dedup …", flush=True)
        t0 = time.time()
        exact_report = run_exact_dedup(
            input_path  = lang_in,
            output_dir  = lang_out / "exact",
            hash_algo   = args.hash_algo,
            batch_size  = args.batch_size_exact,
            max_rows    = args.max_rows,
            resume      = args.resume,
        )
        exact_elapsed = time.time() - t0
        print(
            f"  Pass 1 done in {exact_elapsed:.1f}s — "
            f"in={exact_report['n_in']:,}  kept={exact_report['n_kept']:,}  "
            f"dropped={exact_report['n_dropped']:,}  ({exact_report['drop_rate_pct']}% drop)",
            flush=True,
        )

        # -- Pass 2: Fuzzy --
        exact_kept = lang_out / "exact" / "kept.parquet"
        if not exact_kept.exists():
            print(f"  [SKIP fuzzy] exact kept.parquet not found: {exact_kept}", flush=True)
            continue

        print(f"  Pass 2 — fuzzy dedup (threshold {args.similarity_threshold}%) …", flush=True)
        t0 = time.time()
        fuzzy_report = run_fuzzy_dedup(
            input_path            = exact_kept,
            output_dir            = lang_out / "fuzzy",
            similarity_threshold  = args.similarity_threshold,
            batch_size            = args.batch_size_fuzzy,
            max_rows              = args.max_rows,
            resume                = args.resume,
            num_workers           = args.num_workers,
        )
        fuzzy_elapsed = time.time() - t0
        print(
            f"  Pass 2 done in {fuzzy_elapsed:.1f}s — "
            f"in={fuzzy_report['n_in']:,}  kept={fuzzy_report['n_kept']:,}  "
            f"dropped={fuzzy_report['n_dropped']:,}  ({fuzzy_report['drop_rate_pct']}% drop)",
            flush=True,
        )

        combined["per_language"][lang] = {
            "exact": {k: exact_report[k] for k in ("n_in","n_kept","n_dropped","drop_rate_pct","elapsed_seconds")},
            "fuzzy": {k: fuzzy_report[k] for k in ("n_in","n_kept","n_dropped","drop_rate_pct","elapsed_seconds")},
        }
        combined["totals"]["n_in"]            += exact_report["n_in"]
        combined["totals"]["n_exact_kept"]    += exact_report["n_kept"]
        combined["totals"]["n_exact_dropped"] += exact_report["n_dropped"]
        combined["totals"]["n_fuzzy_kept"]    += fuzzy_report["n_kept"]
        combined["totals"]["n_fuzzy_dropped"] += fuzzy_report["n_dropped"]

    combined["total_elapsed_seconds"] = round(time.time() - t_global, 1)
    totals = combined["totals"]
    totals["overall_drop_rate_pct"] = round(
        (totals["n_in"] - totals["n_fuzzy_kept"]) / totals["n_in"] * 100, 2
    ) if totals["n_in"] else 0

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Stage 4B dedup complete in {combined['total_elapsed_seconds']:.1f}s")
    print(f"  Total in              : {totals['n_in']:,}")
    print(f"  After exact dedup     : {totals['n_exact_kept']:,}  (-{totals['n_exact_dropped']:,})")
    print(f"  After fuzzy dedup     : {totals['n_fuzzy_kept']:,}  (-{totals['n_fuzzy_dropped']:,})")
    print(f"  Overall drop rate     : {totals['overall_drop_rate_pct']}%")
    print(f"\nFinal outputs per language (use fuzzy/kept.parquet downstream):")
    for lang in languages:
        final = output_dir / lang / "fuzzy" / "kept.parquet"
        if final.exists():
            print(f"  {final}  ({final.stat().st_size / 1e6:.1f} MB)")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
