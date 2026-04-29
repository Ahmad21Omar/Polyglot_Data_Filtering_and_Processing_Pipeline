"""Stage 4A — Post-Processing: Clean and Annotate the Deduplicated Dataset.

Applies four sequential steps to the output of Stage 3C (fuzzy dedup):

  Step 1  fix_ground_truth_present   — consistency fix: if ground_truth_present=True
                                        but final_answer_text is null/empty, set it to False.

  Step 2  fix_null_domains           — rule-based domain backfill for 5 known
                                        saumyamalik subsources whose domain can be
                                        inferred deterministically from subsource_raw.

  Step 3  apply_ml_predictions       — fills any remaining null-domain rows using a
                                        FastText prediction CSV produced by
                                        domain_classifier/predict.py.  Predictions are
                                        applied regardless of confidence score (having
                                        *some* domain is better than null for downstream use).

  Step 4  filter_language            — removes rows where the detected language does not
                                        match the expected language group for the row's
                                        category:
                                          • category = null / default  → language must be "en"
                                          • category = multilingual_*  → language must be in
                                            {de, it, fr, ja, es}

Steps 1–3 are *transforms* (they modify column values, they never drop rows).
Step 4 is a *filter* (rows that fail go to the dropped parquet).

Input
-----
A Parquet file produced by Stage 3C (fuzzy dedup).
Must contain at minimum: example_id, domain, subsource_raw,
ground_truth_present, final_answer_text, language, category,
context_messages.

Output (written to --output-dir)
---------------------------------
  merged_sft.v2.kept.parquet                    — cleaned dataset (pipeline output)
  merged_sft.v2.dropped.language_filter.parquet — rows removed by language filter
  summary.json                                  — row counts, step-by-step stats

Prerequisites
-------------
  • domain_classifier/predict.py must be run first to produce the predictions CSV.
    Pass its output path via --predictions-csv.

Usage
-----
    python stage4a_clean_dataset.py \\
        --input        /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \\
        --output-dir   /path/to/stage4a_out/ \\
        --predictions-csv /path/to/domain_classifier/null_domain_predictions.csv

    # Dry-run on first 1000 rows (no --predictions-csv needed — ML step is skipped)
    python stage4a_clean_dataset.py \\
        --input      /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \\
        --output-dir /tmp/stage4a_test/ \\
        --max-rows   1000

Optional flags
--------------
  --batch-size INT   Rows per streaming batch (default: 200 000)
  --max-rows   INT   Stop after N rows — for smoke tests / dry-runs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Language / category constants
# ---------------------------------------------------------------------------

VALID_LANG_DEFAULT     = {"en"}
VALID_LANG_MULTILINGUAL = {"de", "it", "fr", "ja", "es"}
MULTILINGUAL_CATEGORIES = {
    "multilingual_de", "multilingual_it", "multilingual_fr",
    "multilingual_ja", "multilingual_es",
}


# ---------------------------------------------------------------------------
# Step 1 — Ground-truth consistency fix
# ---------------------------------------------------------------------------

def fix_ground_truth_present(batch: pa.RecordBatch) -> pa.RecordBatch:
    """If ground_truth_present=True but final_answer_text is null/empty, set it False."""
    gt_col = batch.column("ground_truth_present")
    fa_col = batch.column("final_answer_text")

    new_gt = []
    for i in range(len(gt_col)):
        gt = gt_col[i].as_py()
        fa = fa_col[i].as_py()
        if gt is True and not fa:
            new_gt.append(False)
        else:
            new_gt.append(gt)

    new_col = pa.array(new_gt, type=pa.bool_())
    idx = batch.schema.get_field_index("ground_truth_present")
    return batch.set_column(idx, "ground_truth_present", new_col)


# ---------------------------------------------------------------------------
# Step 2 — Rule-based domain backfill
# ---------------------------------------------------------------------------

SUBSOURCE_DOMAIN_RULES: dict[str, str] = {
    "saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2":    "code",
    "saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2":       "code",
    "saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2":               "science",
    "saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2":                  "math",
    "saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2":                "reasoning",
}


def fix_null_domains(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Fill domain=null for rows where subsource_raw unambiguously indicates the domain."""
    dom_col = batch.column("domain")
    sub_col = batch.column("subsource_raw")

    new_dom = []
    changed = False
    for i in range(len(dom_col)):
        dom = dom_col[i].as_py()
        if not dom:
            sub = sub_col[i].as_py()
            rule_dom = SUBSOURCE_DOMAIN_RULES.get(sub)
            if rule_dom:
                new_dom.append(rule_dom)
                changed = True
                continue
        new_dom.append(dom)

    if not changed:
        return batch

    new_col = pa.array(new_dom, type=pa.large_string())
    idx = batch.schema.get_field_index("domain")
    return batch.set_column(idx, "domain", new_col)


# ---------------------------------------------------------------------------
# Step 3 — ML domain backfill (FastText predictions CSV)
# ---------------------------------------------------------------------------

def load_ml_predictions(csv_path: Path) -> dict[str, str]:
    """Load example_id → predicted_domain from a CSV produced by predict.py."""
    predictions: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions[row["example_id"]] = row["predicted_domain"]
    print(f"  Loaded {len(predictions):,} ML predictions from {csv_path.name}")
    return predictions


def make_apply_ml_predictions(predictions: dict[str, str]):
    """Return a transform that fills null-domain rows using the predictions dict."""
    def apply_ml_predictions(batch: pa.RecordBatch) -> pa.RecordBatch:
        dom_col = batch.column("domain")
        eid_col = batch.column("example_id")

        new_dom = []
        changed = False
        for i in range(len(dom_col)):
            dom = dom_col[i].as_py()
            if not dom:
                eid = eid_col[i].as_py()
                pred = predictions.get(eid)
                if pred:
                    new_dom.append(pred)
                    changed = True
                    continue
            new_dom.append(dom)

        if not changed:
            return batch

        new_col = pa.array(new_dom, type=pa.large_string())
        idx = batch.schema.get_field_index("domain")
        return batch.set_column(idx, "domain", new_col)

    return apply_ml_predictions


# ---------------------------------------------------------------------------
# Step 4 — Language / category filter
# ---------------------------------------------------------------------------

def filter_language(batch: pa.RecordBatch) -> pa.Array:
    """Keep rows whose language matches the expected group for their category.

    - category is null / not multilingual_* → language must be "en"
    - category is multilingual_*            → language must be in {de, it, fr, ja, es}
    """
    langs = batch.column("language")
    cats  = batch.column("category")

    keep = []
    for i in range(len(langs)):
        lang = langs[i].as_py()
        cat  = cats[i].as_py()
        if cat in MULTILINGUAL_CATEGORIES:
            keep.append(lang in VALID_LANG_MULTILINGUAL)
        else:
            keep.append(lang in VALID_LANG_DEFAULT)

    return pa.array(keep, type=pa.bool_())


# ---------------------------------------------------------------------------
# Step registries
# ---------------------------------------------------------------------------

TRANSFORM_STEPS: list[tuple[str, object]] = [
    ("fix_ground_truth_present", fix_ground_truth_present),
    ("fix_null_domains",         fix_null_domains),
    # "apply_ml_predictions" is appended at runtime when --predictions-csv is given
]

FILTER_STEPS: list[tuple[str, object]] = [
    ("language", filter_language),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    input_path: Path,
    output_dir: Path,
    predictions_csv: Path | None,
    batch_size: int,
    max_rows: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_path    = output_dir / "merged_sft.v2.kept.parquet"
    dropped_path = output_dir / "merged_sft.v2.dropped.language_filter.parquet"
    summary_path = output_dir / "summary.json"

    # Build transform list: fixed steps + optional ML predictions
    transform_steps = list(TRANSFORM_STEPS)
    if predictions_csv is not None:
        predictions = load_ml_predictions(predictions_csv)
        transform_steps.append(("apply_ml_predictions", make_apply_ml_predictions(predictions)))
    else:
        print("  --predictions-csv not provided: ML domain backfill step skipped.")

    pf         = pq.ParquetFile(str(input_path))
    schema     = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    is_dry_run = max_rows is not None
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print(f"Total rows in file: {total_rows:,}")
    if is_dry_run:
        print(f"** DRY-RUN: reading only first {max_rows:,} rows **")
    print(f"Transform steps : {[n for n, _ in transform_steps]}")
    print(f"Filter steps    : {[n for n, _ in FILTER_STEPS]}\n")

    kept_writer:    pq.ParquetWriter | None = None
    dropped_writer: pq.ParquetWriter | None = None

    total_kept    = 0
    total_dropped = 0
    rows_read     = 0
    t0 = time.time()

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            if is_dry_run:
                remaining = max_rows - rows_read
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)

            rows_read += batch.num_rows

            for _, transform_fn in transform_steps:
                batch = transform_fn(batch)

            keep_mask = pa.array([True] * batch.num_rows, type=pa.bool_())
            for _, filter_fn in FILTER_STEPS:
                keep_mask = pc.and_(keep_mask, filter_fn(batch))

            drop_mask    = pc.invert(keep_mask)
            kept_batch   = batch.filter(keep_mask)
            dropped_batch = batch.filter(drop_mask)

            if kept_batch.num_rows > 0:
                if kept_writer is None:
                    kept_writer = pq.ParquetWriter(str(kept_path), schema)
                kept_writer.write_table(pa.Table.from_batches([kept_batch], schema=schema))
                total_kept += kept_batch.num_rows

            if dropped_batch.num_rows > 0:
                if dropped_writer is None:
                    dropped_writer = pq.ParquetWriter(str(dropped_path), schema)
                dropped_writer.write_table(pa.Table.from_batches([dropped_batch], schema=schema))
                total_dropped += dropped_batch.num_rows

            elapsed = time.time() - t0
            pct     = rows_read / total_rows * 100
            print(
                f"  Read {rows_read:>10,} / {total_rows:,} ({pct:5.1f}%) | "
                f"kept={total_kept:,}  dropped={total_dropped:,} | {elapsed:.0f}s"
            )

    finally:
        if kept_writer    is not None: kept_writer.close()
        if dropped_writer is not None: dropped_writer.close()

    elapsed = time.time() - t0
    summary = {
        "stage": "4A_clean_dataset",
        "input_path": str(input_path),
        "total_rows_in": rows_read,
        "total_kept": total_kept,
        "total_dropped": total_dropped,
        "drop_rate_pct": round(total_dropped / rows_read * 100, 4) if rows_read else 0,
        "ml_predictions_used": predictions_csv is not None,
        "dry_run": is_dry_run,
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Stage 4A complete in {elapsed:.1f}s")
    print(f"  Rows read : {rows_read:,}")
    print(f"  Kept      : {total_kept:,}")
    print(f"  Dropped   : {total_dropped:,}  ({summary['drop_rate_pct']}%)")
    if is_dry_run:
        print(f"\n** DRY-RUN complete. Run without --max-rows for the full dataset. **")
    print(f"\nOutputs:")
    if kept_path.exists():
        print(f"  {kept_path}  ({kept_path.stat().st_size / 1e6:.1f} MB)")
    if dropped_path.exists():
        print(f"  {dropped_path}  ({dropped_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {summary_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4A: Post-process the deduplicated SFT dataset. "
            "Fixes ground-truth flags, fills null domains (rule-based + ML), "
            "and removes rows with unexpected language/category combinations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Path to the input Parquet file (Stage 3C output).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write kept.parquet, dropped.parquet, summary.json.",
    )
    p.add_argument(
        "--predictions-csv", default=None,
        help=(
            "Path to the FastText predictions CSV from domain_classifier/predict.py. "
            "Required for the ML domain backfill step. "
            "If omitted, that step is skipped."
        ),
    )
    p.add_argument("--batch-size", type=int, default=200_000)
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after N rows (dry-run / smoke test).",
    )
    args = p.parse_args(argv)

    run(
        input_path      = Path(args.input),
        output_dir      = Path(args.output_dir),
        predictions_csv = Path(args.predictions_csv) if args.predictions_csv else None,
        batch_size      = args.batch_size,
        max_rows        = args.max_rows,
    )


if __name__ == "__main__":
    main()
