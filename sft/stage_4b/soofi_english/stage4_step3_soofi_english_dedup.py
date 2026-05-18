"""Stage 4 (Step 3) — Soofi English Cross-Collection Fuzzy Dedup.

Cross-deduplicates the English Soofi split (Stage 4 Step 2B output) against the
existing main English SFT corpus using RapidFuzz Levenshtein similarity
(``fuzz.ratio``) at threshold τ = 90 %.

Reproduces ``Master_Thesis/Data_pipline/sft/dedup/dedup_fuzzy_cross_collection.py``
as a thin CLI wrapper. The Soofi English split is supplementing an already
high-quality English corpus, so the only dedup pass at this stage is
**cross-collection** against that reference (no intra-soofi exact/fuzzy pass —
the higher Stage 2B quality bar τ = 0.72 already keeps only the strongest rows).

Policy (matches the original script)
------------------------------------
- Match against **reference corpus** (main English SFT) → always **DROP**
  ("cross_collection_duplicate").
- Match against another **already-kept new row** with a *different*
  ``source_dataset_id`` → DROP ("intra_new_cross_source_duplicate").
- Match against another already-kept new row with the *same*
  ``source_dataset_id`` → KEEP (preserves intra-source diversity).
- No match → KEEP.

Algorithm
---------
- Phase 1: Stream reference parquet, normalize prompts, build bucket index
  (head-token + length-bin) of unique normalized prompts.
- Phase 2: Stream new parquet, fetch candidates from buckets, score with
  RapidFuzz ``fuzz.ratio`` (Levenshtein), keep / drop per policy above.

Input (--input-dir)
-------------------
Output of Stage 4 Step 2B (stage4_step2b_soofi_english_quality_filter.py):

    <input-dir>/
        kept.parquet  ← Soofi English rows surviving heuristic + quality filter

Reference (--reference-parquet)
-------------------------------
The merged main English SFT corpus parquet — every Soofi prompt is checked
against this collection.

Output (--output-dir)
---------------------
    <output-dir>/
        soofi_english.cross_dedup.kept.parquet
        soofi_english.cross_dedup.dropped.parquet  ← with audit columns:
            _fuzzy_drop_reason, _fuzzy_match_score,
            _fuzzy_matched_example_id, _fuzzy_matched_dataset_id,
            _fuzzy_new_context_text, _fuzzy_matched_context_text
        cross_dedup_report.json
        cross_dedup_report.csv

Usage
-----
    python stage4_step3_soofi_english_dedup.py \\
        --input-dir         /path/to/stage4_step2b_output \\
        --reference-parquet /path/to/merged_english_sft.kept.parquet \\
        --output-dir        /path/to/stage4_step3_output

    # Smoke test (first 5,000 new rows):
    python stage4_step3_soofi_english_dedup.py \\
        --input-dir         /path/to/stage4_step2b_output \\
        --reference-parquet /path/to/merged_english_sft.kept.parquet \\
        --output-dir        /tmp/stage4_step3_test \\
        --max-rows          5000

Optional flags (matching dedup_fuzzy_cross_collection.py)
---------------------------------------------------------
  --similarity-threshold F   fuzz.ratio threshold in [0, 100] (default 90.0).
  --batch-size N             Parquet batch size (default 20_000).
  --bucket-tokens N          Head-token count for bucket key (default 1).
  --len-bin-size N           Length-bin granularity (default 100).
  --max-candidates-per-bucket N   Cap per bucket (default 5_000).
  --length-diff-threshold N  Pre-filter on |len(a)-len(b)|, <0 disables (default 200).
  --neighbor-bin-radius N    Adjacent length-bins to check (default 1).
  --num-workers N            ProcessPool workers (default 52).
  --max-rows N               Stop after N new rows (smoke test).
  --log-every-batches N      Logging cadence (default 5).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Locate the existing cross-collection dedup implementation in Data_pipline.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CROSS_DEDUP_SCRIPT = (
    _REPO_ROOT
    / "Master_Thesis"
    / "Data_pipline"
    / "sft"
    / "dedup"
    / "dedup_fuzzy_cross_collection.py"
)


def _load_cross_dedup_module():
    if not _CROSS_DEDUP_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find cross-collection dedup module at {_CROSS_DEDUP_SCRIPT}"
        )
    spec = importlib.util.spec_from_file_location(
        "_dedup_fuzzy_cross_collection", _CROSS_DEDUP_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {_CROSS_DEDUP_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4 (Step 3): Cross-collection fuzzy dedup of English Soofi vs. main corpus. "
            "Wraps Data_pipline/sft/dedup/dedup_fuzzy_cross_collection.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory of Stage 4 Step 2B output containing kept.parquet (new data).",
    )
    p.add_argument(
        "--reference-parquet", required=True,
        help="Path to the merged main English SFT corpus parquet (reference collection).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write kept/dropped parquets and reports.",
    )
    p.add_argument("--similarity-threshold", type=float, default=90.0,
                   help="fuzz.ratio threshold in [0,100] (default 90.0).")
    p.add_argument("--batch-size", type=int, default=20_000)
    p.add_argument("--bucket-tokens", type=int, default=1)
    p.add_argument("--len-bin-size", type=int, default=100)
    p.add_argument("--max-candidates-per-bucket", type=int, default=5_000)
    p.add_argument("--length-diff-threshold", type=int, default=200,
                   help="Pre-filter |len(a)-len(b)|; negative disables (default 200).")
    p.add_argument("--neighbor-bin-radius", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=52)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Stop after N new rows (smoke test).")
    p.add_argument("--log-every-batches", type=int, default=5)
    args = p.parse_args(argv)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    new_parquet       = input_dir / "kept.parquet"
    reference_parquet = Path(args.reference_parquet)

    if not new_parquet.exists():
        p.error(f"Input file not found: {new_parquet}")
    if not reference_parquet.exists():
        p.error(f"Reference parquet not found: {reference_parquet}")

    output_dir.mkdir(parents=True, exist_ok=True)

    kept_out    = output_dir / "soofi_english.cross_dedup.kept.parquet"
    dropped_out = output_dir / "soofi_english.cross_dedup.dropped.parquet"
    report_json = output_dir / "cross_dedup_report.json"
    report_csv  = output_dir / "cross_dedup_report.csv"

    print("=" * 80)
    print("Stage 4 — Soofi English Cross-Collection Fuzzy Dedup")
    print("=" * 80)
    print(f"  new parquet (Soofi EN 2B): {new_parquet}")
    print(f"  reference (main corpus)  : {reference_parquet}")
    print(f"  output-dir               : {output_dir}")
    print(f"  similarity_threshold     : {args.similarity_threshold}% (fuzz.ratio)")
    print(f"  batch_size               : {args.batch_size:,}")
    print(f"  num_workers              : {args.num_workers}")
    if args.max_rows:
        print(f"  ** SMOKE TEST: first {args.max_rows:,} new rows **")
    print()

    cross = _load_cross_dedup_module()
    length_diff = None if args.length_diff_threshold < 0 else args.length_diff_threshold

    stats = cross.run_cross_fuzzy_dedup(
        new_parquet=new_parquet,
        reference_parquet=reference_parquet,
        kept_out=kept_out,
        dropped_out=dropped_out,
        report_json=report_json,
        report_csv=report_csv,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        bucket_tokens=args.bucket_tokens,
        len_bin_size=args.len_bin_size,
        max_candidates_per_bucket=args.max_candidates_per_bucket,
        length_diff_threshold=length_diff,
        neighbor_bin_radius=args.neighbor_bin_radius,
        num_workers=args.num_workers,
        max_rows=args.max_rows,
        log_every_batches=args.log_every_batches,
    )

    total_dropped = stats.dropped_vs_ref + stats.dropped_vs_new
    drop_rate = (total_dropped / stats.total_new * 100.0) if stats.total_new else 0.0

    print("\n" + "=" * 80)
    print("Stage 4 (cross-dedup) complete")
    print("=" * 80)
    print(f"  Reference indexed       : {stats.ref_indexed:,} unique prompts")
    print(f"  New rows processed      : {stats.total_new:,}")
    print(f"  Kept                    : {stats.kept:,}")
    print(f"  Dropped (vs reference)  : {stats.dropped_vs_ref:,}")
    print(f"  Dropped (intra-new x-source): {stats.dropped_vs_new:,}")
    print(f"  Total dropped           : {total_dropped:,}  ({drop_rate:.4f}%)")
    print(f"\nOutputs:")
    for fpath in (kept_out, dropped_out, report_json, report_csv):
        if fpath.exists():
            if fpath.suffix == ".parquet":
                print(f"  {fpath}  ({fpath.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"  {fpath}")
    print("=" * 80)


if __name__ == "__main__":
    main()
