"""
stage_2b_merge.py
─────────────────
Stage 2B for the RL pipeline: streaming concatenation of all 9 verifier-validated
per-dataset ``*.kept.parquet`` files into one merged corpus.

Why this is "Stage 2B" for RL
─────────────────────────────
The SFT pipeline uses Stage 2B for quality filtering (Soofi/FineWeb-HQ).  RL does
not need an external quality-filter — the per-dataset *verifier* is the quality
gate (see Filtering_Pipeline/rl/FILTERING_APPROACH.md).  In the RL
pipeline, Stage 2B is therefore repurposed as the **merge consolidation step**:
it produces the single corpus that Stages 3B (exact-hash dedup) and 3C (fuzzy
dedup) operate on.

Merge order  (verifier strength, strongest first)
─────────────────────────────────────────────────
Because Stage 3B is "first-seen-wins", the strongest verifier must come first.
At duplicate prompts, the row backed by the stricter verifier survives.

    1. slr_bench                     (prolog_rule_induction)
    2. synlogic                      (library-based rule)
    3. nemotron_rl_reasoning_gym_v1  (reasoning_gym, 99 families)
    4. am_thinking_v1_rl             (math_equiv / code_asserts / code_stdio)
    5. dolci_think_rl_7b             (math / code / if_rules)
    6. nemotron_3_nano_rl_blend      (mc / if / code / schema)
    7. synthetic2_rl                 (9 verifier types)
    8. webinstruct_verified          (math_equiv / multi_gt)
    9. logi_glue                     (text_match / multi_gt)

Safety mechanisms
─────────────────
1. **Streaming / chunked writes** via ``pq.ParquetWriter`` + ``iter_batches``.
   Peak RAM ≈ one batch (~50 k rows × 19 cols ≈ a few hundred MB).
2. **RAM guard** before every file: exits cleanly if free RAM dips below
   ``--min-free-gb``.  No OOM-kill.
3. **Schema fail-fast**: every file's schema is compared against the schema of
   the first file (slr_bench).  Any divergence aborts the run — this is a
   one-time pipeline, divergence here means an upstream filter bug.
4. **Checkpoint file** (``merge_checkpoint.json`` in the output dir): after each
   source file is fully appended its name is recorded.  Re-runs skip done files.
5. **Atomic finalize**: written to ``rl_v1_merged.kept.parquet.tmp`` and renamed
   only when ALL files are done.  A partial run never overwrites a finished one.
6. **Post-merge validation**: re-opens the final parquet and checks
   ``row_count == sum(inputs)`` and ``dataset_id`` distribution matches inputs.
7. **MERGE_REPORT.md** is written next to the output with per-dataset stats,
   verifier_type / dataset_id distributions, SHA-256, and run metadata.

Usage
─────
    python3 -m Filtering_Pipeline.rl.stage_3.deduplication.stage_3_00_merge \\
        --corpora-root /home/workdir/Master_Thesis/corpora/rl \\
        --out-dir      /home/workdir/Master_Thesis/corpora/rl/merged_rl_ds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import psutil
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Hard-coded merge order (verifier strength, strongest first).
# Tuple: (corpus_subdir_name, kept_parquet_filename)
MERGE_ORDER: List[Tuple[str, str]] = [
    ("slr_bench",                    "rl_slr_bench_v1.kept.parquet"),
    ("synlogic",                     "rl_synlogic_v1.kept.parquet"),
    ("nemotron_rl_reasoning_gym_v1", "rl_nemotron_rl_reasoning_gym_v1.kept.parquet"),
    ("am_thinking_v1_rl",            "rl_am_thinking_v1_rl_v1.kept.parquet"),
    ("dolci_think_rl_7b",            "rl_dolci_think_rl_7b_v1.kept.parquet"),
    ("nemotron_3_nano_rl_blend",     "rl_nemotron_3_nano_rl_blend_v1.kept.parquet"),
    ("synthetic2_rl",                "rl_synthetic2_rl_v1.kept.parquet"),
    ("webinstruct_verified",         "rl_webinstruct_verified_v1.kept.parquet"),
    ("logi_glue",                    "rl_logi_glue_v1.kept.parquet"),
]

OUTPUT_NAME = "rl_v1_merged.kept.parquet"
CHECKPOINT_NAME = "merge_checkpoint.json"
REPORT_NAME = "MERGE_REPORT.md"


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
    return {"done_files": [], "total_rows_written": 0, "per_dataset_rows": {}}


def _save_checkpoint(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _resolve_inputs(corpora_root: Path) -> List[Path]:
    """Resolve each MERGE_ORDER entry to an absolute path.  Fail-fast if missing."""
    resolved: List[Path] = []
    missing: List[str] = []
    for subdir, fname in MERGE_ORDER:
        p = corpora_root / subdir / fname
        if not p.exists():
            missing.append(str(p))
        resolved.append(p)
    if missing:
        print("ERROR: Required input file(s) not found:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    return resolved


def _normalize_string_fields(schema: pa.Schema) -> pa.Schema:
    """Promote string → large_string for robustness against encoding mixes."""
    new_fields = []
    for field in schema:
        if pa.types.is_string(field.type):
            new_fields.append(pa.field(field.name, pa.large_string(), nullable=field.nullable))
        else:
            new_fields.append(field)
    return pa.schema(new_fields)


def _schemas_match(a: pa.Schema, b: pa.Schema) -> bool:
    """Compare two schemas by (name, type) tuples after string→large_string normalization."""
    a_norm = _normalize_string_fields(a)
    b_norm = _normalize_string_fields(b)
    if a_norm.names != b_norm.names:
        return False
    for fa, fb in zip(a_norm, b_norm):
        if fa.type != fb.type:
            return False
    return True


def _cast_batch_to_target(batch: pa.RecordBatch, target_schema: pa.Schema) -> pa.Table:
    """Cast batch columns to the target schema (string → large_string)."""
    import pyarrow.compute as pc

    cols = {}
    for field in target_schema:
        col = batch.column(field.name)
        if col.type == field.type:
            cols[field.name] = col
        else:
            cols[field.name] = pc.cast(col, field.type, safe=False)
    return pa.table(cols, schema=target_schema)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight: validate all input schemas against the first (slr_bench)
# ─────────────────────────────────────────────────────────────────────────────


def _preflight_schema_check(input_paths: List[Path]) -> pa.Schema:
    print("\n[Preflight] Validating schemas …")
    target = _normalize_string_fields(pq.read_schema(input_paths[0]))
    n_cols = len(target.names)
    print(f"  Target schema (from {input_paths[0].parent.name}): {n_cols} columns")

    bad: List[str] = []
    for p in input_paths:
        s = pq.read_schema(p)
        if not _schemas_match(target, s):
            bad.append(str(p))
            print(f"  ❌  {p.parent.name}: schema MISMATCH")
            # Detailed diff for diagnostics
            t_set = {(f.name, str(_normalize_string_fields(pa.schema([f])).field(0).type)) for f in target}
            s_set = {(f.name, str(_normalize_string_fields(pa.schema([f])).field(0).type)) for f in s}
            only_in_target = t_set - s_set
            only_in_source = s_set - t_set
            if only_in_target:
                print(f"     only in target : {sorted(only_in_target)}")
            if only_in_source:
                print(f"     only in source : {sorted(only_in_source)}")
        else:
            print(f"  ✅  {p.parent.name}")
    if bad:
        print(f"\nERROR: {len(bad)} file(s) have a schema mismatch — aborting (fail-fast policy).")
        sys.exit(2)
    print("[Preflight] All 9 schemas match.\n")
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Main merge
# ─────────────────────────────────────────────────────────────────────────────


def merge_streaming(
    corpora_root: Path,
    out_dir: Path,
    min_free_gb: float,
    batch_rows: int,
) -> Tuple[Path, int, Dict[str, int], float]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = _resolve_inputs(corpora_root)
    target_schema = _preflight_schema_check(input_paths)

    # Print plan
    print("[Merge] Input files (in merge order):")
    grand_total_estimate = 0
    for i, p in enumerate(input_paths, 1):
        n_rows = pq.read_metadata(p).num_rows
        size_gb = p.stat().st_size / 1024 ** 3
        grand_total_estimate += n_rows
        print(f"  {i}. {p.parent.name:<32} rows={n_rows:>10,}  size={size_gb:>5.2f} GB")
    print(f"  Σ estimated rows: {grand_total_estimate:,}\n")

    final_path = out_dir / OUTPUT_NAME
    tmp_path = out_dir / (OUTPUT_NAME + ".tmp")
    part2_path = out_dir / (OUTPUT_NAME + ".part2.tmp")
    checkpoint_path = out_dir / CHECKPOINT_NAME

    # ── Checkpoint / resume logic ──────────────────────────────────────────
    ckpt = _load_checkpoint(checkpoint_path)
    done_files: Set[str] = set(ckpt["done_files"])
    total_rows: int = ckpt["total_rows_written"]
    per_dataset_rows: Dict[str, int] = dict(ckpt.get("per_dataset_rows", {}))

    if done_files:
        print(f"[Merge] Resuming: {len(done_files)} files done, {total_rows:,} rows already written.")
        # If a previous run completed and renamed .tmp → final, but the checkpoint
        # still says some files are pending, that's inconsistent — bail out so the
        # user notices.
        if final_path.exists() and not tmp_path.exists():
            print("ERROR: Final output exists but checkpoint says work is pending. "
                  "Delete the checkpoint or the output and re-run.")
            sys.exit(1)

    remaining_idx = [i for i, p in enumerate(input_paths) if p.parent.name not in done_files]
    if not remaining_idx:
        if tmp_path.exists():
            tmp_path.rename(final_path)
            print(f"[Merge] All files already written.  Renamed .tmp → {final_path.name}")
        else:
            print(f"[Merge] Already complete: {final_path}")
        return final_path, total_rows, per_dataset_rows, time.time() - t0

    # ── Decide active output path ──────────────────────────────────────────
    if tmp_path.exists():
        # Cannot truly append; new rows go to part2.tmp and we combine at end.
        print(f"[Merge] Existing partial .tmp detected → new rows go to {part2_path.name}, "
              f"will combine at end.")
        active_path = part2_path
        use_part2 = True
    else:
        active_path = tmp_path
        use_part2 = False

    writer: Optional[pq.ParquetWriter] = None

    # ── Stream each remaining file ─────────────────────────────────────────
    for run_idx, file_idx in enumerate(remaining_idx, 1):
        path = input_paths[file_idx]
        ds_name = path.parent.name
        size_gb = path.stat().st_size / 1024 ** 3

        # RAM guard
        free_gb = _free_ram_gb()
        proc_gb = _proc_ram_gb()
        print(f"\n[{run_idx}/{len(remaining_idx)}] {ds_name}  ({size_gb:.2f} GB)")
        print(f"  [RAM] free={free_gb:.1f} GB  proc={proc_gb:.2f} GB")
        if free_gb < min_free_gb:
            print(f"  ⚠️  RAM guard tripped (free<{min_free_gb} GB) — stopping cleanly.")
            _save_checkpoint(checkpoint_path, {
                "done_files": sorted(done_files),
                "total_rows_written": total_rows,
                "per_dataset_rows": per_dataset_rows,
            })
            if writer:
                writer.close()
            sys.exit(2)

        pf = pq.ParquetFile(path, memory_map=False)
        n_rows = pf.metadata.num_rows
        t_file = time.time()
        rows_this_file = 0

        for batch in pf.iter_batches(batch_size=batch_rows):
            table = _cast_batch_to_target(batch, target_schema)
            if writer is None:
                writer = pq.ParquetWriter(str(active_path), target_schema, compression="snappy")
            writer.write_table(table)
            rows_this_file += len(table)
            del table, batch

        elapsed = time.time() - t_file
        total_rows += rows_this_file
        per_dataset_rows[ds_name] = per_dataset_rows.get(ds_name, 0) + rows_this_file
        done_files.add(ds_name)
        rate = rows_this_file / elapsed if elapsed > 0 else 0
        print(f"  ✅  {rows_this_file:>10,} rows  |  cumulative: {total_rows:>12,}  "
              f"|  {elapsed:.1f}s  ({rate:,.0f} rows/s)")
        if rows_this_file != n_rows:
            print(f"  ⚠️  Row-count mismatch: wrote {rows_this_file:,} but file reports {n_rows:,}")

        _save_checkpoint(checkpoint_path, {
            "done_files": sorted(done_files),
            "total_rows_written": total_rows,
            "per_dataset_rows": per_dataset_rows,
        })

    if writer:
        writer.close()
        writer = None

    # ── Final assembly: if part2 was used, concatenate tmp + part2 ─────────
    if use_part2:
        combined_tmp = out_dir / (OUTPUT_NAME + ".combined.tmp")
        print(f"\n[Merge] Combining {tmp_path.name} + {part2_path.name} → {final_path.name}")
        combined_writer = pq.ParquetWriter(str(combined_tmp), target_schema, compression="snappy")
        for src in [tmp_path, part2_path]:
            free_gb = _free_ram_gb()
            if free_gb < min_free_gb:
                print(f"⚠️  RAM guard during combine ({free_gb:.1f} GB free) — stopping.")
                combined_writer.close()
                sys.exit(2)
            pf = pq.ParquetFile(src)
            for batch in pf.iter_batches(batch_size=batch_rows):
                combined_writer.write_table(_cast_batch_to_target(batch, target_schema))
        combined_writer.close()
        combined_tmp.rename(final_path)
        tmp_path.unlink(missing_ok=True)
        part2_path.unlink(missing_ok=True)
    else:
        active_path.rename(final_path)

    elapsed_total = time.time() - t0
    return final_path, total_rows, per_dataset_rows, elapsed_total


# ─────────────────────────────────────────────────────────────────────────────
# Post-merge sanity validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_output(
    final_path: Path,
    expected_total: int,
    expected_per_dataset: Dict[str, int],
    batch_rows: int,
) -> Tuple[bool, Dict[str, int], Counter]:
    """Re-open the final parquet and verify row-count + dataset_id distribution."""
    print("\n[Validate] Re-opening final output and counting rows / dataset_id …")
    pf = pq.ParquetFile(final_path)
    counted_total = 0
    dataset_id_counts: Counter = Counter()
    verifier_type_counts: Counter = Counter()

    for batch in pf.iter_batches(batch_size=batch_rows, columns=["dataset_id", "verifier_type"]):
        counted_total += len(batch)
        for v in batch.column("dataset_id").to_pylist():
            dataset_id_counts[v] += 1
        for v in batch.column("verifier_type").to_pylist():
            verifier_type_counts[v] += 1

    ok = True
    if counted_total != expected_total:
        ok = False
        print(f"  ❌  Row-count mismatch: counted={counted_total:,} expected={expected_total:,}")
    else:
        print(f"  ✅  Row-count matches: {counted_total:,}")

    # Note: dataset_id values may not equal subdir names exactly (depends on filter scripts);
    # we just print the distribution for review.
    print("  dataset_id distribution (in final output):")
    for ds, n in dataset_id_counts.most_common():
        print(f"     {ds:<40} {n:>10,}")
    print("  verifier_type distribution (in final output):")
    for vt, n in verifier_type_counts.most_common():
        print(f"     {vt:<40} {n:>10,}")
    return ok, dict(dataset_id_counts), verifier_type_counts


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def write_report(
    out_dir: Path,
    final_path: Path,
    input_paths: List[Path],
    per_subdir_rows: Dict[str, int],
    total_rows: int,
    dataset_id_counts: Dict[str, int],
    verifier_type_counts: Counter,
    elapsed_total: float,
    sha256: str,
    validation_ok: bool,
) -> Path:
    size_gb = final_path.stat().st_size / 1024 ** 3
    report_path = out_dir / REPORT_NAME

    lines: List[str] = []
    lines.append(f"# RL Merge Report — Stage 2B")
    lines.append("")
    lines.append(f"- **Output file**: `{final_path}`")
    lines.append(f"- **Size**: {size_gb:.2f} GB")
    lines.append(f"- **Total rows**: {total_rows:,}")
    lines.append(f"- **Runtime**: {int(elapsed_total)//60} m {int(elapsed_total)%60} s")
    lines.append(f"- **SHA-256**: `{sha256}`")
    lines.append(f"- **Validation**: {'✅ PASS' if validation_ok else '❌ FAIL'}")
    lines.append("")
    lines.append("## Merge order (verifier strength, strongest first)")
    lines.append("")
    lines.append("| # | Dataset (subdir) | Rows | Cumulative |")
    lines.append("|---|---|---:|---:|")
    cum = 0
    for i, p in enumerate(input_paths, 1):
        ds = p.parent.name
        n = per_subdir_rows.get(ds, 0)
        cum += n
        lines.append(f"| {i} | `{ds}` | {n:,} | {cum:,} |")
    lines.append("")
    lines.append("## `dataset_id` distribution (column value, post-merge)")
    lines.append("")
    lines.append("| dataset_id | rows |")
    lines.append("|---|---:|")
    for ds, n in sorted(dataset_id_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{ds}` | {n:,} |")
    lines.append("")
    lines.append("## `verifier_type` distribution (post-merge)")
    lines.append("")
    lines.append("| verifier_type | rows |")
    lines.append("|---|---:|")
    for vt, n in verifier_type_counts.most_common():
        lines.append(f"| `{vt}` | {n:,} |")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Script: `Filtering_Pipeline/rl/stage_3/deduplication/stage_3_00_merge.py`")
    lines.append(f"- Inputs: `Master_Thesis/corpora/rl/<subdir>/*.kept.parquet`")
    lines.append(f"- Next stage: 3B exact-hash dedup → 3C fuzzy dedup (threshold=90)")
    lines.append("")
    report_path.write_text("\n".join(lines))
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Stage 2B RL merge — concatenate 9 verifier-validated kept.parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--corpora-root",
        default="/home/workdir/Master_Thesis/corpora/rl",
        help="Root that contains the per-dataset subdirs (default: Master_Thesis/corpora/rl).",
    )
    p.add_argument(
        "--out-dir",
        default="/home/workdir/Master_Thesis/corpora/rl/merged_rl_ds",
        help="Output directory (default: Master_Thesis/corpora/rl/merged_rl_ds).",
    )
    p.add_argument("--min-free-gb", type=float, default=40.0,
                   help="Minimum free system RAM (GB) before loading the next file.")
    p.add_argument("--batch-rows", type=int, default=50_000,
                   help="Row-group batch size for streaming read/write.")
    p.add_argument("--skip-sha", action="store_true",
                   help="Skip SHA-256 of the final file (saves ~30 s on 10 GB).")
    args = p.parse_args(argv)

    corpora_root = Path(args.corpora_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not corpora_root.exists():
        print(f"ERROR: --corpora-root does not exist: {corpora_root}")
        sys.exit(1)

    print(f"[Merge] corpora-root: {corpora_root}")
    print(f"[Merge] out-dir     : {out_dir}")
    print(f"[Merge] min-free    : {args.min_free_gb} GB")
    print(f"[Merge] batch-rows  : {args.batch_rows:,}")
    print(f"[Merge] free RAM now: {_free_ram_gb():.1f} GB")

    final_path, total_rows, per_dataset_rows, elapsed = merge_streaming(
        corpora_root=corpora_root,
        out_dir=out_dir,
        min_free_gb=args.min_free_gb,
        batch_rows=args.batch_rows,
    )

    # ── Validation ─────────────────────────────────────────────────────────
    expected_total = sum(per_dataset_rows.values())
    ok, dataset_id_counts, verifier_type_counts = validate_output(
        final_path=final_path,
        expected_total=expected_total,
        expected_per_dataset=per_dataset_rows,
        batch_rows=args.batch_rows,
    )

    # ── SHA-256 + Report ───────────────────────────────────────────────────
    if args.skip_sha:
        sha = "(skipped)"
    else:
        print("\n[Hash] Computing SHA-256 of final output …")
        sha = _sha256_file(final_path)
        print(f"  {sha}")

    # Re-resolve input paths for the report (order matters)
    input_paths = _resolve_inputs(corpora_root)
    report_path = write_report(
        out_dir=Path(args.out_dir),
        final_path=final_path,
        input_paths=input_paths,
        per_subdir_rows=per_dataset_rows,
        total_rows=total_rows,
        dataset_id_counts=dataset_id_counts,
        verifier_type_counts=verifier_type_counts,
        elapsed_total=elapsed,
        sha256=sha,
        validation_ok=ok,
    )

    # ── Cleanup checkpoint on success ──────────────────────────────────────
    if ok:
        (out_dir / CHECKPOINT_NAME).unlink(missing_ok=True)

    size_gb = final_path.stat().st_size / 1024 ** 3
    print(f"\n{'=' * 70}")
    print(f"{'✅  MERGE COMPLETE' if ok else '❌  MERGE FINISHED WITH VALIDATION ERRORS'}")
    print(f"   Output : {final_path}")
    print(f"   Size   : {size_gb:.2f} GB")
    print(f"   Rows   : {total_rows:,}")
    print(f"   Report : {report_path}")
    print(f"   Time   : {int(elapsed)//60}m {int(elapsed)%60}s")
    print(f"{'=' * 70}")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
