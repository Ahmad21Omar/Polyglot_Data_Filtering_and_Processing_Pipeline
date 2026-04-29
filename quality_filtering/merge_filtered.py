"""
merge_filtered.py
─────────────────
RAM-safe streaming merge of all per-corpus ``*.kept.parquet`` files produced
by Stage 2B (``runner.run_quality_filter``) into one ``merged_sft.kept.parquet``.

Safety mechanisms
─────────────────
1. **Streaming / chunked writes** via PyArrow ParquetWriter — only one batch
   at a time is held in RAM.  Peak RAM ≈ one batch (~50 k rows × schema width).

2. **RAM guard** before every file: if free RAM < --min-free-gb (default 40 GB)
   the script exits cleanly with a clear message.  No OOM-kill.

3. **Checkpoint file** (``merge_checkpoint.json`` in ``--out-root``): after each
   source file is fully appended its name is recorded.  On re-run already-written
   files are skipped and the output is extended.

4. **Atomic rename**: written to ``merged_sft.kept.parquet.tmp`` and only renamed
   to the final name when ALL files are done — a partial run never overwrites a
   previously completed merge.

File discovery
──────────────
The script searches ``--out-root`` for files matching ``*.kept.parquet`` in all
immediate subdirectories (one level deep), and also at the root itself.
Files whose name starts with ``merged_sft`` are excluded (they are previous
merge outputs, not individual corpus files).

All Stage 2B columns are passed through verbatim.  No schema trimming is done
here — that is Stage 4's responsibility.  Files with differing schemas are
handled by casting each file's columns to the schema of the first file; missing
columns are null-filled.

Usage
─────
# Minimal:
python3 -m Filtering_Pipeline.quality_filtering.merge_filtered \\
    --out-root /home/workdir/Master_Thesis/corpora/quality_filtered_v2

# With explicit safety limits:
python3 -m Filtering_Pipeline.quality_filtering.merge_filtered \\
    --out-root /home/workdir/Master_Thesis/corpora/quality_filtered_v2 \\
    --min-free-gb 40 \\
    --batch-rows 50000

# Run in background:
nohup python3 -m Filtering_Pipeline.quality_filtering.merge_filtered \\
    --out-root /home/workdir/Master_Thesis/corpora/quality_filtered_v2 \\
    >> /home/workdir/Master_Thesis/corpora/merge_v2.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import psutil
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def _proc_ram_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"done_files": [], "total_rows_written": 0}


def _save_checkpoint(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _discover_files(out_root: Path) -> List[Path]:
    """Find all per-corpus *.kept.parquet files under out_root (1 level deep)."""
    found: List[Path] = []
    # Root level (flat layout)
    for f in out_root.glob("*.kept.parquet"):
        if not f.name.startswith("merged_sft"):
            found.append(f)
    # Subdirectory level (one subdir per corpus)
    for subdir in sorted(out_root.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.glob("*.kept.parquet"):
            if not f.name.startswith("merged_sft"):
                found.append(f)
    return sorted(found)


def _normalize_batch(batch: pa.RecordBatch, target_schema: pa.Schema) -> pa.Table:
    """Cast a batch to target_schema; fill missing columns with nulls.

    Uses pure Arrow — no Pandas round-trip.
    """
    import pyarrow.compute as pc

    cols = {}
    for field in target_schema:
        name = field.name
        target_type = field.type
        if name in batch.schema.names:
            col = batch.column(name)
            if col.type == target_type:
                cols[name] = col
            elif pa.types.is_null(col.type):
                cols[name] = pa.nulls(len(batch), type=target_type)
            else:
                try:
                    cols[name] = pc.cast(col, target_type, safe=False)
                except Exception:
                    try:
                        # Two-step: cast to string first, then to target
                        cols[name] = pc.cast(pc.cast(col, pa.string()), target_type, safe=False)
                    except Exception:
                        cols[name] = pa.nulls(len(batch), type=target_type)
        else:
            cols[name] = pa.nulls(len(batch), type=target_type)
    return pa.table(cols, schema=target_schema)


def _derive_target_schema(first_file: Path) -> pa.Schema:
    """Read the schema of the first file and normalise string types to large_string.

    ``context_messages`` keeps its nested list<struct> type as-is; all other
    string / large_string columns are promoted to ``large_string`` so that
    files written with different Arrow string encodings can be merged cleanly.
    """
    raw = pq.read_schema(first_file)
    fields = []
    for field in raw:
        if pa.types.is_string(field.type):
            fields.append(pa.field(field.name, pa.large_string(), nullable=True))
        else:
            fields.append(field)
    return pa.schema(fields)


# ─────────────────────────────────────────────────────────────────────────────
# Main merge logic
# ─────────────────────────────────────────────────────────────────────────────


def merge_streaming(
    out_root: Path,
    min_free_gb: float = 40.0,
    batch_rows: int = 50_000,
) -> None:
    t0 = time.time()

    kept_files = _discover_files(out_root)
    if not kept_files:
        print("[Merge] No *.kept.parquet files found. Aborting.")
        sys.exit(1)

    print(f"\n[Merge] Found {len(kept_files)} kept parquet files:")
    for f in kept_files:
        size_gb = f.stat().st_size / 1024 ** 3
        rel = f.relative_to(out_root)
        print(f"  {rel}  ({size_gb:.2f} GB)")

    final_path = out_root / "merged_sft.kept.parquet"
    tmp_path = out_root / "merged_sft.kept.parquet.tmp"
    checkpoint_path = out_root / "merge_checkpoint.json"

    ckpt = _load_checkpoint(checkpoint_path)
    done_files: Set[str] = set(ckpt["done_files"])
    total_rows: int = ckpt["total_rows_written"]

    if done_files:
        print(
            f"\n[Merge] Resuming from checkpoint: {len(done_files)} files done "
            f"({total_rows:,} rows written so far)"
        )
        # If final was already written from a prior complete run, rename back so
        # we can extend it (resume would only happen if checkpoint still exists).
        if final_path.exists() and not tmp_path.exists():
            final_path.rename(tmp_path)

    remaining = [f for f in kept_files if str(f.relative_to(out_root)) not in done_files]

    if not remaining:
        if tmp_path.exists():
            tmp_path.rename(final_path)
            print(f"\n[Merge] All files already written.  Renamed .tmp → {final_path}")
        elif final_path.exists():
            print(f"\n[Merge] Already complete: {final_path}")
        else:
            print("[Merge] Checkpoint says done but no output found.  Clear checkpoint and re-run.")
            sys.exit(1)
        return

    print(f"\n[Merge] {len(remaining)} file(s) remaining …\n")

    # Derive target schema from first remaining file (or from the first file overall
    # if tmp already exists, to stay consistent with prior batches).
    if tmp_path.exists():
        target_schema = pq.read_schema(tmp_path)
    else:
        target_schema = _derive_target_schema(remaining[0])

    writer: Optional[pq.ParquetWriter] = None

    if tmp_path.exists():
        # PyArrow cannot truly append to an existing Parquet file.
        # Write remaining files to a part2 tmp, then concatenate.
        part2_path = out_root / "merged_sft.kept.parquet.part2.tmp"
        print(
            f"[Merge] Existing partial output detected → new rows go to {part2_path.name}\n"
            f"        They will be combined with existing rows at the end.\n"
        )
        active_path = part2_path
        use_part2 = True
    else:
        active_path = tmp_path
        use_part2 = False

    # ── Stream each remaining file ─────────────────────────────────────────
    for file_idx, f in enumerate(remaining):
        rel = str(f.relative_to(out_root))
        size_gb = f.stat().st_size / 1024 ** 3

        free_gb = _free_ram_gb()
        proc_gb = _proc_ram_gb()
        print(f"[RAM]  Free: {free_gb:.1f} GB  |  This process: {proc_gb:.1f} GB")
        if free_gb < min_free_gb:
            print(
                f"\n⚠️  [RAM GUARD] Only {free_gb:.1f} GB free — threshold is {min_free_gb:.1f} GB.\n"
                f"   Stopping BEFORE loading next file to prevent OOM.\n"
                f"   Progress saved in checkpoint.  Re-run to continue."
            )
            _save_checkpoint(checkpoint_path, {"done_files": list(done_files), "total_rows_written": total_rows})
            if writer:
                writer.close()
            sys.exit(2)

        pf = pq.ParquetFile(f, memory_map=False)
        n_rows = pf.metadata.num_rows
        estimated_bytes_per_row = (size_gb * 1024 ** 3) / n_rows if n_rows > 0 else 10_000
        safe_batch = min(batch_rows, max(10_000, int(4 * 1024 ** 3 / max(estimated_bytes_per_row, 1))))

        print(
            f"[{file_idx + 1}/{len(remaining)}]  {rel}  ({size_gb:.2f} GB  "
            f"{n_rows:,} rows  batch={safe_batch:,})",
            flush=True,
        )
        t_file = time.time()
        rows_this_file = 0

        for batch in pf.iter_batches(batch_size=safe_batch):
            arrow_table = _normalize_batch(batch, target_schema)
            if writer is None:
                writer = pq.ParquetWriter(str(active_path), target_schema, compression="snappy")
            writer.write_table(arrow_table)
            rows_this_file += len(arrow_table)
            del arrow_table, batch

        elapsed_file = time.time() - t_file
        total_rows += rows_this_file
        done_files.add(rel)
        print(
            f"  ✅  {rows_this_file:>10,} rows  |  cumulative: {total_rows:>12,}  "
            f"|  {elapsed_file:.0f}s"
        )
        _save_checkpoint(checkpoint_path, {"done_files": list(done_files), "total_rows_written": total_rows})

    if writer:
        writer.close()
        writer = None

    # ── Final assembly ─────────────────────────────────────────────────────
    if use_part2:
        part2_path = active_path
        combined_tmp = out_root / "merged_sft.kept.parquet.combined.tmp"
        print(f"\n[Merge] Combining {tmp_path.name} + {part2_path.name} → {final_path.name} …")
        combined_writer = pq.ParquetWriter(str(combined_tmp), target_schema, compression="snappy")
        for src in [tmp_path, part2_path]:
            free_gb = _free_ram_gb()
            if free_gb < min_free_gb:
                print(f"⚠️  [RAM GUARD] {free_gb:.1f} GB free during final combine — stopping.")
                combined_writer.close()
                sys.exit(2)
            pf = pq.ParquetFile(src)
            for batch in pf.iter_batches(batch_size=batch_rows):
                combined_writer.write_table(_normalize_batch(batch, target_schema))
        combined_writer.close()
        combined_tmp.rename(final_path)
        tmp_path.unlink(missing_ok=True)
        part2_path.unlink(missing_ok=True)
    else:
        active_path.rename(final_path)

    # ── Done ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    size_gb = final_path.stat().st_size / 1024 ** 3

    print(f"\n{'=' * 70}")
    print(f"✅  MERGE COMPLETE")
    print(f"   Output : {final_path}")
    print(f"   Size   : {size_gb:.1f} GB")
    print(f"   Rows   : {total_rows:,}")
    print(f"   Time   : {int(elapsed_total) // 60}m {int(elapsed_total) % 60}s")
    print(f"{'=' * 70}\n")

    checkpoint_path.unlink(missing_ok=True)
    print("[Merge] Checkpoint file removed.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv=None):
    p = argparse.ArgumentParser(
        description="RAM-safe streaming merge of Stage 2B *.kept.parquet → merged_sft.kept.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--out-root",
        required=True,
        help=(
            "Directory that contains the per-corpus Stage 2B output subdirs "
            "(each subdir holds a *.kept.parquet from run_quality_filter)."
        ),
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=40.0,
        help="Minimum free system RAM in GB before loading the next file (default: 40). "
             "The script exits cleanly if the threshold is breached.",
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=50_000,
        help="Row-group batch size for streaming read/write (default: 50 000). "
             "The script auto-adjusts downward based on estimated row size.",
    )
    args = p.parse_args(argv)

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = Path("/home/workdir") / out_root
    if not out_root.exists():
        print(f"ERROR: --out-root does not exist: {out_root}")
        sys.exit(1)

    print(f"[Merge] out-root    : {out_root}")
    print(f"[Merge] min-free    : {args.min_free_gb} GB")
    print(f"[Merge] batch-rows  : {args.batch_rows:,}")
    print(f"[Merge] Free RAM now: {_free_ram_gb():.1f} GB")
    print()

    merge_streaming(
        out_root=out_root,
        min_free_gb=args.min_free_gb,
        batch_rows=args.batch_rows,
    )


if __name__ == "__main__":
    main()
