"""Stage 4 — Finalise RL training dataset (upload-ready schema).

Takes the Stage 3C kept parquet and writes a streaming, schema-clean version
that contains exactly the 19 canonical columns defined in the per-corpus
``filter_and_format_*.py`` ``_EMPTY_TEMPLATE``. All Stage 3B/3C internal
columns (``_dedup_hash``, ``_dedup_norm``, ``_dedup_seen_in_ds``) are dropped.

Output:
  rl_v1_final.parquet              — single parquet, snappy, 19 canonical cols
  rl_v1_final.schema.json          — column list + row count + sha256
  FINAL_DATASET_REPORT.md          — human-readable summary

Streams via PyArrow ``iter_batches`` → ``ParquetWriter``; never holds the
whole dataset in RAM.

Usage:
  python3 stage_4_finalize.py \\
      --input  /home/workdir/Master_Thesis/corpora/rl/dedup_3c_fuzzy/rl_v1.fuzzy_dedup.kept.parquet \\
      --output-dir /home/workdir/Master_Thesis/corpora/rl/final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Canonical RL schema — matches _EMPTY_TEMPLATE in filter_and_format_*.py
CANONICAL_COLUMNS = [
    "dataset_id",
    "dataset_version_date",
    "example_id",
    "row_id",
    "subsource_raw",
    "source_dataset_id",
    "license",
    "used_by_model",
    "context_messages",
    "language",
    "domain",
    "ability",
    "difficulty",
    "verifier_type",
    "verifier_source",
    "ground_truth_text",
    "verification_info_raw",
    "avg_reward",
    "reward_model_metadata",
]

# Internal columns we explicitly drop:
INTERNAL_COLUMNS = {"_dedup_hash", "_dedup_norm", "_dedup_seen_in_ds"}


def _sha256_of_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def finalize(input_path: Path, output_dir: Path, batch_rows: int = 50_000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path     = output_dir / "rl_v1_final.parquet"
    schema_path  = output_dir / "rl_v1_final.schema.json"
    report_path  = output_dir / "FINAL_DATASET_REPORT.md"

    pf = pq.ParquetFile(input_path)
    in_cols = set(pf.schema_arrow.names)

    missing = [c for c in CANONICAL_COLUMNS if c not in in_cols]
    if missing:
        raise SystemExit(f"Input is missing canonical columns: {missing}")
    unexpected = sorted(in_cols - set(CANONICAL_COLUMNS) - INTERNAL_COLUMNS)
    if unexpected:
        print(f"⚠  Unexpected extra columns (will be dropped): {unexpected}")

    total_in = pf.metadata.num_rows
    print(f"Input : {input_path}")
    print(f"  Rows : {total_in:,}")
    print(f"  Cols in: {len(in_cols)}")
    print(f"  Cols out: {len(CANONICAL_COLUMNS)} (canonical)")
    print(f"Output: {out_path}\n")

    # Per-dataset counts + verifier-type counts
    ds_counts = Counter()
    vt_counts = Counter()
    lang_counts = Counter()
    rows_written = 0

    writer: pq.ParquetWriter | None = None
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=batch_rows, columns=CANONICAL_COLUMNS):
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), batch.schema, compression="snappy")
        writer.write_table(pa.Table.from_batches([batch]))

        for ds in batch.column("dataset_id").to_pylist():
            ds_counts[str(ds) if ds is not None else "__MISSING__"] += 1
        for vt in batch.column("verifier_type").to_pylist():
            vt_counts[str(vt) if vt is not None else "__MISSING__"] += 1
        for lg in batch.column("language").to_pylist():
            lang_counts[str(lg) if lg is not None else "__MISSING__"] += 1

        rows_written += batch.num_rows
        if rows_written % 200_000 == 0 or rows_written == total_in:
            pct = rows_written / total_in * 100
            print(f"  {rows_written:>12,}/{total_in:,} ({pct:5.1f}%)", flush=True)

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    sha256 = _sha256_of_file(out_path)
    size_gb = out_path.stat().st_size / 1024 ** 3

    schema_payload = {
        "generated_at_utc":  datetime.now(timezone.utc).isoformat(),
        "input_path":        str(input_path),
        "output_path":       str(out_path),
        "rows":              rows_written,
        "size_gb":           round(size_gb, 3),
        "sha256":            sha256,
        "compression":       "snappy",
        "columns":           CANONICAL_COLUMNS,
        "n_columns":         len(CANONICAL_COLUMNS),
        "dropped_internal_columns": sorted(INTERNAL_COLUMNS & in_cols),
        "dataset_id_distribution": dict(ds_counts.most_common()),
        "verifier_type_distribution": dict(vt_counts.most_common()),
        "language_distribution": dict(lang_counts.most_common()),
        "elapsed_seconds":    round(elapsed, 1),
    }
    schema_path.write_text(json.dumps(schema_payload, indent=2, ensure_ascii=False))

    # Markdown report
    md = []
    md.append("# RL Dataset — Final Upload-Ready Version\n")
    md.append(f"- **Output**: `{out_path}`")
    md.append(f"- **Size**: {size_gb:.2f} GB (snappy parquet)")
    md.append(f"- **Rows**: {rows_written:,}")
    md.append(f"- **SHA-256**: `{sha256}`")
    md.append(f"- **Columns**: {len(CANONICAL_COLUMNS)} (canonical RL schema)")
    md.append(f"- **Generated**: {schema_payload['generated_at_utc']}")
    md.append(f"- **Source**: `{input_path}`\n")
    md.append("## Canonical columns\n")
    for c in CANONICAL_COLUMNS:
        md.append(f"- `{c}`")
    md.append("\nDropped internal columns: " + ", ".join(f"`{c}`" for c in sorted(INTERNAL_COLUMNS & in_cols)) + "\n")
    md.append("## `dataset_id` distribution\n")
    md.append("| dataset_id | rows |")
    md.append("|---|---:|")
    for ds, c in ds_counts.most_common():
        md.append(f"| `{ds}` | {c:,} |")
    md.append("\n## `verifier_type` distribution\n")
    md.append("| verifier_type | rows |")
    md.append("|---|---:|")
    for vt, c in vt_counts.most_common():
        md.append(f"| `{vt}` | {c:,} |")
    md.append("\n## `language` distribution\n")
    md.append("| language | rows |")
    md.append("|---|---:|")
    for lg, c in lang_counts.most_common():
        md.append(f"| `{lg}` | {c:,} |")
    md.append("\n## Pipeline lineage\n")
    md.append("| Stage | Rows |")
    md.append("|---|---:|")
    md.append("| Stage 2B merge | 1,127,950 |")
    md.append("| Stage 3B exact-hash dedup | 1,105,317 |")
    md.append("| Stage 3C fuzzy dedup (threshold=90, logi_glue intra-skip) | 1,084,071 |")
    md.append(f"| **Stage 4 finalise (this file)** | **{rows_written:,}** |")
    md.append("")

    report_path.write_text("\n".join(md))

    print(f"\n{'=' * 70}")
    print(f"Stage 4 complete  ({elapsed:.1f}s)")
    print(f"  Rows         : {rows_written:,}")
    print(f"  Size         : {size_gb:.2f} GB")
    print(f"  SHA-256      : {sha256}")
    print(f"  Output       : {out_path}")
    print(f"  Schema JSON  : {schema_path}")
    print(f"  Report       : {report_path}")
    print(f"{'=' * 70}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-rows", type=int, default=50_000)
    args = p.parse_args(argv)
    finalize(Path(args.input), Path(args.output_dir), args.batch_rows)


if __name__ == "__main__":
    main()
