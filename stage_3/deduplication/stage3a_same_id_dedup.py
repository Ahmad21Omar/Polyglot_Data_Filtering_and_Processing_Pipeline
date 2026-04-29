"""Stage 3A — Intra-Dataset Same-ID Deduplication.

Removes rows where the same (dataset_id, example_id) pair appears more than
once within the same dataset.  Cross-dataset collisions on example_id are
intentionally left to Stage 3B (exact-hash dedup).

This is the first of three sequential deduplication passes:

  Stage 3A  Same-ID dedup       ← this script
  Stage 3B  Exact-hash dedup    (cross-dataset, SHA1 of normalized context)
  Stage 3C  Fuzzy dedup         (cross-dataset, RapidFuzz Levenshtein ratio)

Algorithm — two-pass streaming
-------------------------------
Pass 1  Reads only the two lightweight key columns (dataset_id, example_id).
        Streams all rows to build a set of global row-indices to keep (first
        occurrence of each pair).  RAM: ~200 MB for 20 M rows.

Pass 2  Streams the full schema.  For each row, checks whether its global index
        is in the keep-set and writes it to kept or dropped accordingly.
        RAM per batch: proportional to batch_size × average row size.

Input
-----
A Parquet file produced by Stage 2B (quality filtering).
Must contain at minimum the columns:
  - dataset_id   (string)  HuggingFace dataset name, e.g. "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
  - example_id   (string)  content hash assigned during dataset construction

Output (written to --output-dir)
----------------------------------
  merged_sft.same_id_dedup.kept.parquet    — unique rows (pipeline input for 3B)
  merged_sft.same_id_dedup.dropped.parquet — duplicate rows removed here
  summary.json                             — counts, drop rate, per-dataset breakdown

Usage
-----
    python stage3a_same_id_dedup.py \\
        --input  /path/to/quality_filtered/merged_sft.kept.parquet \\
        --output-dir /path/to/deduped/stage3a_same_id_dedup \\
        --batch-size 50000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Pass 1: build keep-index set (key columns only)
# ---------------------------------------------------------------------------

def _build_keep_indices(input_path: Path, batch_size: int) -> tuple[set[int], int]:
    """Return (keep_indices, total_rows).

    keep_indices is the set of global row positions (0-based) to retain —
    the first occurrence of each (dataset_id, example_id) pair.
    """
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    seen: set[tuple[str, str]] = set()
    keep_indices: set[int] = set()

    row_offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=["dataset_id", "example_id"]):
        ds_ids = batch.column("dataset_id").to_pylist()
        ex_ids = batch.column("example_id").to_pylist()
        for local_i, (ds, ex) in enumerate(zip(ds_ids, ex_ids)):
            key = (ds, ex)
            if key not in seen:
                seen.add(key)
                keep_indices.add(row_offset + local_i)
        row_offset += batch.num_rows

    return keep_indices, total_rows


# ---------------------------------------------------------------------------
# Pass 2: stream full schema, split kept / dropped
# ---------------------------------------------------------------------------

def _write_split(
    input_path: Path,
    keep_indices: set[int],
    total_rows: int,
    kept_path: Path,
    dropped_path: Path,
    batch_size: int,
) -> tuple[int, int]:
    """Stream full rows, write to kept/dropped. Returns (n_kept, n_dropped)."""
    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow

    kept_writer:    pq.ParquetWriter | None = None
    dropped_writer: pq.ParquetWriter | None = None
    total_kept = 0
    total_dropped = 0
    row_offset = 0
    batch_num = 0
    t0 = time.time()

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            batch_num += 1
            n = batch.num_rows

            keep_mask = [
                (row_offset + i) in keep_indices for i in range(n)
            ]
            row_offset += n

            mask_arr = pa.array(keep_mask, type=pa.bool_())
            inv_mask = pc.invert(mask_arr)

            kept_batch = batch.filter(mask_arr)
            drop_batch = batch.filter(inv_mask)

            n_kept = kept_batch.num_rows
            n_drop = drop_batch.num_rows
            total_kept    += n_kept
            total_dropped += n_drop

            if n_kept > 0:
                if kept_writer is None:
                    kept_writer = pq.ParquetWriter(str(kept_path), schema)
                kept_writer.write_table(
                    pa.Table.from_batches([kept_batch], schema=schema)
                )
            if n_drop > 0:
                if dropped_writer is None:
                    dropped_writer = pq.ParquetWriter(str(dropped_path), schema)
                dropped_writer.write_table(
                    pa.Table.from_batches([drop_batch], schema=schema)
                )

            processed = row_offset
            elapsed   = time.time() - t0
            pct       = processed / total_rows * 100
            if batch_num % 10 == 0 or processed == total_rows:
                rate = processed / elapsed if elapsed > 0 else 0
                eta  = (total_rows - processed) / rate if rate > 0 else 0
                print(
                    f"  [Pass 2] Batch {batch_num:>4d} | "
                    f"{processed:>12,}/{total_rows:,} ({pct:5.1f}%) | "
                    f"kept={total_kept:,}  dropped={total_dropped:,} | "
                    f"{elapsed:>5.0f}s  ETA ~{eta:.0f}s"
                )

    finally:
        if kept_writer    is not None: kept_writer.close()
        if dropped_writer is not None: dropped_writer.close()

    return total_kept, total_dropped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(input_path: Path, output_dir: Path, batch_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_path    = output_dir / "merged_sft.same_id_dedup.kept.parquet"
    dropped_path = output_dir / "merged_sft.same_id_dedup.dropped.parquet"
    summary_path = output_dir / "summary.json"

    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    for col in ("dataset_id", "example_id"):
        if col not in schema.names:
            raise ValueError(f"Input file is missing required column: '{col}'")

    print(f"Input : {input_path}")
    print(f"  Rows      : {total_rows:,}")
    print(f"  Row groups: {pf.metadata.num_row_groups}")
    print(f"  Batch size: {batch_size:,}")
    print(f"Output dir: {output_dir}")
    print()

    # ------------------------------------------------------------------
    # Pass 1 — lightweight key scan
    # ------------------------------------------------------------------
    print("Pass 1: scanning (dataset_id, example_id) pairs ...")
    t1 = time.time()
    keep_indices, _ = _build_keep_indices(input_path, batch_size=500_000)
    t1_elapsed = time.time() - t1
    n_kept_expected = len(keep_indices)
    n_dropped_expected = total_rows - n_kept_expected
    print(
        f"  Done in {t1_elapsed:.1f}s — "
        f"unique pairs: {n_kept_expected:,}  duplicates: {n_dropped_expected:,}\n"
    )

    # ------------------------------------------------------------------
    # Pass 2 — full-schema split
    # ------------------------------------------------------------------
    print("Pass 2: writing kept / dropped parquets ...")
    total_kept, total_dropped = _write_split(
        input_path=input_path,
        keep_indices=keep_indices,
        total_rows=total_rows,
        kept_path=kept_path,
        dropped_path=dropped_path,
        batch_size=batch_size,
    )

    # Per-dataset breakdown of dropped rows
    dataset_drop_counts: dict[str, int] = {}
    if total_dropped > 0 and dropped_path.exists():
        dpf = pq.ParquetFile(dropped_path)
        for b in dpf.iter_batches(batch_size=500_000, columns=["dataset_id"]):
            for ds in b.column("dataset_id").to_pylist():
                dataset_drop_counts[ds] = dataset_drop_counts.get(ds, 0) + 1

    summary = {
        "stage": "3A_same_id_dedup",
        "input_path": str(input_path),
        "total_rows_in": total_rows,
        "total_kept": total_kept,
        "total_dropped": total_dropped,
        "drop_rate_pct": round(total_dropped / total_rows * 100, 4) if total_rows else 0,
        "unique_dataset_example_id_pairs": n_kept_expected,
        "pass1_elapsed_seconds": round(t1_elapsed, 1),
        "dropped_by_dataset": dict(
            sorted(dataset_drop_counts.items(), key=lambda x: -x[1])
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}")
    print(f"Stage 3A complete")
    print(f"  Total in  : {total_rows:,}")
    print(f"  Kept      : {total_kept:,}")
    print(f"  Dropped   : {total_dropped:,}  ({summary['drop_rate_pct']}%)")
    if dataset_drop_counts:
        print("  Dropped by dataset (top 10):")
        for ds, cnt in list(summary["dropped_by_dataset"].items())[:10]:
            print(f"    {ds:<60s}  {cnt:>8,}")
    print(f"\nOutputs:")
    print(f"  {kept_path}  ({kept_path.stat().st_size / 1e9:.2f} GB)")
    if total_dropped > 0 and dropped_path.exists():
        print(f"  {dropped_path}  ({dropped_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 3A: Remove intra-dataset duplicate (dataset_id, example_id) pairs. "
            "Two-pass streaming — Pass 1 scans key columns only; Pass 2 writes full rows. "
            "Writes kept.parquet, dropped.parquet, and summary.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to the input Parquet file (Stage 2B output).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write kept.parquet, dropped.parquet, summary.json into.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per batch in Pass 2 (full-schema read). Lower = less RAM per batch.",
    )
    args = p.parse_args(argv)
    run(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
