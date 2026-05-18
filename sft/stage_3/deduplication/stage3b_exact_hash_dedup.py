"""Stage 3B — Cross-Dataset Exact-Hash Deduplication.

Removes rows whose context prompt is an exact duplicate of a prompt already
seen in a *different* dataset.  Duplicates within the same dataset are kept by
design (Stage 3A handles intra-dataset same-ID cleanup).

This is the second of three sequential deduplication passes:

  Stage 3A  Same-ID dedup       (intra-dataset, by example_id)
  Stage 3B  Exact-hash dedup    ← this script
  Stage 3C  Fuzzy dedup         (cross-dataset, RapidFuzz Levenshtein ratio)

Algorithm
---------
For each row, the full conversation context (all turns in context_messages) is
extracted, normalized, and hashed:

  1. Extract: concatenate all turns from context_messages as "role: content\\n…"
  2. Normalize: lowercase → strip punctuation → collapse whitespace
  3. Hash: SHA-1 (configurable via --hash-algo)

Policy: first-seen-wins across datasets.
  - First row with hash H in dataset A → kept; hash recorded as "owned by A".
  - Later row with hash H in dataset B (B ≠ A) → dropped as cross-dataset duplicate.
  - Later row with hash H in dataset A again → kept (intra-dataset duplicate; Stage 3A handles these).

State persistence (SQLite)
--------------------------
The hash → first_dataset mapping is stored in a SQLite file (--state-db).
This allows resuming interrupted runs with --resume.

Input
-----
A Parquet file from Stage 3A output (or Stage 2B if skipping 3A).
Required columns: dataset_id, example_id, context_messages.

Output (written to --output-dir)
----------------------------------
  merged_sft.exact_dedup.kept.parquet    — unique rows (pipeline input for 3C)
  merged_sft.exact_dedup.dropped.parquet — cross-dataset duplicates removed
  report.json                            — counts, drop rate, per-dataset breakdown,
                                           cross-dataset duplicate pairs
Intermediate files (can be deleted after a successful run):
  exact_dedup_checkpoint.json            — batch-level resume checkpoint
  exact_dedup_state.sqlite               — hash → first_dataset index

Usage
-----
    python stage3b_exact_hash_dedup.py \\
        --input  /path/to/stage3a/merged_sft.same_id_dedup.kept.parquet \\
        --output-dir /path/to/stage3b_exact_dedup

    # Resume an interrupted run:
    python stage3b_exact_hash_dedup.py \\
        --input  /path/to/... --output-dir /path/to/... --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Text normalization & hashing
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")


def _extract_context(context_messages: Any) -> str:
    """Concatenate all turns in context_messages into a single string."""
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


def _hash(text: str, algo: str) -> str:
    data = text.encode("utf-8", errors="ignore")
    if algo == "sha1":   return hashlib.sha1(data).hexdigest()
    if algo == "md5":    return hashlib.md5(data).hexdigest()
    if algo == "sha256": return hashlib.sha256(data).hexdigest()
    raise ValueError(f"Unsupported hash algorithm: {algo!r}")


def compute_hash(context_messages: Any, algo: str) -> tuple[str, str]:
    """Return (prompt_hash, normalized_text) for a context_messages value."""
    raw  = _extract_context(context_messages)
    norm = _normalize(raw)
    return _hash(norm, algo), norm


# ---------------------------------------------------------------------------
# SQLite hash index
# ---------------------------------------------------------------------------

def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hash_index "
        "(hash TEXT PRIMARY KEY, dataset_id TEXT)"
    )
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _lookup(h: str, cache: dict[str, str], conn: sqlite3.Connection) -> str | None:
    if h in cache:
        return cache[h]
    row = conn.execute(
        "SELECT dataset_id FROM hash_index WHERE hash = ?", (h,)
    ).fetchone()
    if row:
        cache[h] = row[0]
        return row[0]
    return None


def _insert(h: str, dataset_id: str, cache: dict[str, str], conn: sqlite3.Connection) -> None:
    cache[h] = dataset_id
    conn.execute(
        "INSERT OR REPLACE INTO hash_index (hash, dataset_id) VALUES (?, ?)",
        (h, dataset_id),
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class _State:
    processed_rows: int = 0
    batch_index:    int = 0
    kept:           int = 0
    dropped:        int = 0
    same_ds_kept:   int = 0
    kept_by_ds:     Counter = field(default_factory=Counter)
    dropped_by_ds:  Counter = field(default_factory=Counter)
    dropped_pairs:  Counter = field(default_factory=Counter)


def _save_checkpoint(path: Path, state: _State, t0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc":    datetime.now(timezone.utc).isoformat(),
        "processed_rows":  state.processed_rows,
        "batch_index":     state.batch_index,
        "kept":            state.kept,
        "dropped":         state.dropped,
        "same_ds_kept":    state.same_ds_kept,
        "elapsed_sec":     time.time() - t0,
        "kept_by_ds":      dict(state.kept_by_ds),
        "dropped_by_ds":   dict(state.dropped_by_ds),
        "dropped_pairs":   {
            f"{a}|||{b}": c for (a, b), c in state.dropped_pairs.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_checkpoint(path: Path) -> _State | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    s = _State(
        processed_rows = int(payload.get("processed_rows", 0)),
        batch_index    = int(payload.get("batch_index", 0)),
        kept           = int(payload.get("kept", 0)),
        dropped        = int(payload.get("dropped", 0)),
        same_ds_kept   = int(payload.get("same_ds_kept", 0)),
        kept_by_ds     = Counter({k: int(v) for k, v in payload.get("kept_by_ds", {}).items()}),
        dropped_by_ds  = Counter({k: int(v) for k, v in payload.get("dropped_by_ds", {}).items()}),
    )
    for key, c in payload.get("dropped_pairs", {}).items():
        if "|||" in key:
            a, b = key.split("|||", 1)
            s.dropped_pairs[(a, b)] = int(c)
    return s


# ---------------------------------------------------------------------------
# Main dedup loop
# ---------------------------------------------------------------------------

def run(
    input_path:   Path,
    output_dir:   Path,
    hash_algo:    str  = "sha1",
    batch_size:   int  = 50_000,
    resume:       bool = False,
    checkpoint_every: int = 20,
    log_every:        int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_path       = output_dir / "merged_sft.exact_dedup.kept.parquet"
    dropped_path    = output_dir / "merged_sft.exact_dedup.dropped.parquet"
    report_path     = output_dir / "report.json"
    checkpoint_path = output_dir / "exact_dedup_checkpoint.json"
    db_path         = output_dir / "exact_dedup_state.sqlite"

    pf = pq.ParquetFile(input_path)
    schema      = pf.schema_arrow
    total_rows  = pf.metadata.num_rows

    for col in ("dataset_id", "example_id", "context_messages"):
        if col not in schema.names:
            raise ValueError(f"Input file is missing required column: '{col}'")

    print(f"Input : {input_path}")
    print(f"  Rows      : {total_rows:,}")
    print(f"  Row groups: {pf.metadata.num_row_groups}")
    print(f"  Hash algo : {hash_algo}")
    print(f"  Batch size: {batch_size:,}")
    print(f"Output dir: {output_dir}")
    print()

    state = (_load_checkpoint(checkpoint_path) if resume else None) or _State()
    cache: dict[str, str] = {}
    conn = _open_db(db_path)

    kept_writer:    pq.ParquetWriter | None = None
    dropped_writer: pq.ParquetWriter | None = None
    t0 = time.time()
    skip_rows = state.processed_rows

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            # Skip already-processed rows when resuming
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

            keep_idx: list[int] = []
            drop_idx: list[int] = []
            keep_hashes: list[str] = []
            keep_norms:  list[str] = []
            keep_seen_in: list[str] = []
            drop_hashes: list[str] = []
            drop_norms:  list[str] = []
            drop_matched: list[str] = []
            drop_reasons: list[str] = []

            for i, (ds, ctx) in enumerate(zip(ds_ids, ctx_list)):
                ds = str(ds) if ds is not None else "__MISSING__"
                h, norm = compute_hash(ctx, hash_algo)
                seen_in = _lookup(h, cache, conn)

                if seen_in is None:
                    _insert(h, ds, cache, conn)
                    keep_idx.append(i)
                    keep_hashes.append(h)
                    keep_norms.append(norm)
                    keep_seen_in.append("")
                    state.kept      += 1
                    state.kept_by_ds[ds] += 1
                elif seen_in == ds:
                    # Same-dataset duplicate — keep (Stage 3A handles intra-dataset IDs)
                    keep_idx.append(i)
                    keep_hashes.append(h)
                    keep_norms.append(norm)
                    keep_seen_in.append(seen_in)
                    state.kept      += 1
                    state.same_ds_kept += 1
                    state.kept_by_ds[ds] += 1
                else:
                    # Cross-dataset duplicate → drop
                    drop_idx.append(i)
                    drop_hashes.append(h)
                    drop_norms.append(norm)
                    drop_matched.append(seen_in)
                    drop_reasons.append("exact_hash_cross_dataset")
                    state.dropped     += 1
                    state.dropped_by_ds[ds] += 1
                    state.dropped_pairs[(ds, seen_in)] += 1

                state.processed_rows += 1

            raw_table = pa.Table.from_batches([batch], schema=schema)

            if keep_idx:
                kept_t = raw_table.take(pa.array(keep_idx, pa.int64()))
                kept_t = kept_t.append_column("_dedup_hash",       pa.array(keep_hashes))
                kept_t = kept_t.append_column("_dedup_norm",       pa.array(keep_norms))
                kept_t = kept_t.append_column("_dedup_seen_in_ds", pa.array(keep_seen_in))
                if kept_writer is None:
                    kept_writer = pq.ParquetWriter(str(kept_path), kept_t.schema)
                kept_writer.write_table(kept_t)

            if drop_idx:
                drop_t = raw_table.take(pa.array(drop_idx, pa.int64()))
                drop_t = drop_t.append_column("_drop_reason",       pa.array(drop_reasons))
                drop_t = drop_t.append_column("_dedup_hash",        pa.array(drop_hashes))
                drop_t = drop_t.append_column("_dedup_norm",        pa.array(drop_norms))
                drop_t = drop_t.append_column("_dedup_matched_ds",  pa.array(drop_matched))
                if dropped_writer is None:
                    dropped_writer = pq.ParquetWriter(str(dropped_path), drop_t.schema)
                dropped_writer.write_table(drop_t)

            if state.batch_index % log_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                rate    = state.processed_rows / elapsed
                pct     = state.processed_rows / total_rows * 100
                print(
                    f"  Batch {state.batch_index:>4d} | "
                    f"{state.processed_rows:>12,}/{total_rows:,} ({pct:5.1f}%) | "
                    f"kept={state.kept:,}  dropped={state.dropped:,} | "
                    f"{rate:,.0f} rows/s"
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

    # Write report
    report = {
        "stage":       "3B_exact_hash_dedup",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path":  str(input_path),
        "hash_algo":   hash_algo,
        "batch_size":  batch_size,
        "policy": {
            "drop_cross_dataset_duplicates": True,
            "keep_same_dataset_duplicates":  True,
            "normalization": ["lowercase", "remove_punctuation", "collapse_whitespace"],
        },
        "total_rows_in":  total_rows,
        "total_kept":     state.kept,
        "total_dropped":  state.dropped,
        "same_ds_kept":   state.same_ds_kept,
        "drop_rate_pct":  round(state.dropped / total_rows * 100, 4) if total_rows else 0,
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
    print(f"Stage 3B complete  ({elapsed:.1f}s)")
    print(f"  Total in  : {total_rows:,}")
    print(f"  Kept      : {state.kept:,}")
    print(f"  Dropped   : {state.dropped:,}  ({report['drop_rate_pct']}%)")
    print(f"  Same-ds kept (not counted as dup): {state.same_ds_kept:,}")
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
            "Stage 3B: Cross-dataset exact-hash deduplication. "
            "Hashes SHA1(normalize(context_messages)); first-seen-wins across datasets. "
            "Writes kept.parquet, dropped.parquet, and report.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input Parquet file (Stage 3A output, or Stage 2B if skipping 3A).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write outputs into.",
    )
    p.add_argument(
        "--hash-algo",
        default="sha1",
        choices=["sha1", "md5", "sha256"],
        help="Hash algorithm for the normalized prompt.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per batch.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if output-dir already contains one.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Flush checkpoint to disk every N batches.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print progress every N batches.",
    )
    args = p.parse_args(argv)
    run(
        input_path   = Path(args.input),
        output_dir   = Path(args.output_dir),
        hash_algo    = args.hash_algo,
        batch_size   = args.batch_size,
        resume       = args.resume,
        checkpoint_every = args.checkpoint_every,
        log_every        = args.log_every,
    )


if __name__ == "__main__":
    main()
