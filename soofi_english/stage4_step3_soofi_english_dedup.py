"""Stage 4 (Step 3) — Soofi English Deduplication (Exact + Fuzzy).

Two-pass intra-language deduplication for the English Soofi split
produced by Stage 4 Step 2B (stage4_step2b_soofi_english_quality_filter.py).

Pass 1 (Exact): SHA-1 hash deduplication on normalized prompt.
Pass 2 (Fuzzy): RapidFuzz token set ratio similarity (threshold 0.99) on prompt,
                skipping rows already marked as exact duplicates.

This script is a thin wrapper around the existing dedup machinery:

    deduplication/exact.py           ← ExactDeduplicator (SHA-1 hashing)
    deduplication/fuzzy.py           ← FuzzyDeduplicator (RapidFuzz)
    deduplication/runner.py          ← run_dedup_exact, run_dedup_fuzzy

Schema & Audit columns
----------------------
Both passes annotate rows with audit columns before partitioning:
  - _dedup_hash          (Pass 1): SHA-1 of normalized prompt (kept rows only)
  - _dedup_prompt_norm   (Pass 1): Whitespace-normalized prompt (kept rows only)
  - _dedup_seen_in_source (Fuzzy): source_dataset_id of earlier matching row in same source

Input (--input-dir)
-------------------
Output of Stage 4 Step 2B (stage4_step2b_soofi_english_quality_filter.py):

    <input-dir>/
        kept.parquet      ← rows to be deduplicated
        dropped.parquet   ← (copied as-is)

Output (--output-dir)
---------------------
    <output-dir>/
        exact/
            kept.parquet     ← rows passing exact dedup
            dropped.parquet  ← rows failing exact dedup (hash collisions)
            summary.json     ← counts + hash statistics
        fuzzy/
            kept.parquet     ← rows passing fuzzy dedup (no prior similar in source)
            dropped.parquet  ← rows failing fuzzy dedup (similar prior row in source)
            summary.json     ← counts + similarity statistics
        deduped.kept.parquet     ← final kept rows (exact dedup output)
        deduped.dropped.parquet  ← final dropped rows (union of exact + fuzzy)
        summary.json             ← combined summary

Usage
-----
    python stage4_step3_soofi_english_dedup.py \\
        --input-dir /path/to/stage4_step2b_soofi_english_quality_filter \\
        --output-dir /path/to/stage4_step3_soofi_english_dedup

    # Smoke test — first 500 rows:
    python stage4_step3_soofi_english_dedup.py \\
        --input-dir /path/to/stage4_step2b_soofi_english_quality_filter \\
        --output-dir /tmp/stage4_step3_test \\
        --max-rows 500

Optional flags
--------------
  --max-rows             Stop after N rows (smoke test).
  --fuzzy-threshold      RapidFuzz token_set_ratio threshold (default: 0.99).
  --num-proc             Number of parallel workers (default: 1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add Master_Thesis/ to sys.path so Filtering_Pipeline is importable as a package.
_MASTER_THESIS_ROOT = Path(__file__).resolve().parents[2]
if str(_MASTER_THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MASTER_THESIS_ROOT))

from Filtering_Pipeline.deduplication import (
    run_dedup_exact,
    run_dedup_fuzzy,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4 (Step 3): Two-pass intra-language deduplication for English Soofi split. "
            "Pass 1 (exact): SHA-1 hash dedup. Pass 2 (fuzzy): RapidFuzz similarity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help=(
            "Directory of Stage 4 Step 2B output, containing kept.parquet "
            "to be deduplicated."
        ),
    )
    p.add_argument(
        "--output-dir", required=True,
        help=(
            "Directory to write exact/fuzzy subdirs, deduped.kept.parquet, "
            "deduped.dropped.parquet, and summary.json."
        ),
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after N rows (smoke test).",
    )
    p.add_argument(
        "--fuzzy-threshold", type=float, default=0.99,
        help="RapidFuzz token_set_ratio threshold for fuzzy dedup (default 0.99).",
    )
    p.add_argument(
        "--num-proc", type=int, default=1,
        help="Number of parallel workers for ds.map (>1 uses multiprocessing).",
    )
    args = p.parse_args(argv)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    input_path = input_dir / "kept.parquet"
    if not input_path.exists():
        p.error(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Stage 4 — Soofi English Deduplication (Exact + Fuzzy)")
    print(f"  input-dir   : {input_dir}")
    print(f"  output-dir  : {output_dir}")
    print(f"  fuzzy-threshold: {args.fuzzy_threshold}")
    if args.max_rows:
        print(f"  ** SMOKE TEST: first {args.max_rows:,} rows **")
    print()

    summary: dict = {
        "stage": "4_step3_soofi_english_dedup",
        "exact": {},
        "fuzzy": {},
        "totals": {"n_in": 0, "n_kept_exact": 0, "n_kept_fuzzy": 0, "n_dropped": 0},
    }

    t_global = time.time()

    # ─────────────────────────────────────────────────────────────
    # Pass 1: Exact SHA-1 hash deduplication
    # ─────────────────────────────────────────────────────────────
    print(f"\nPass 1: Exact SHA-1 hash deduplication")
    print("=" * 60)
    t0 = time.time()

    exact_output = output_dir / "exact"
    exact_output.mkdir(parents=True, exist_ok=True)

    exact_summary = run_dedup_exact(
        input_path=input_path,
        output_dir=exact_output,
        output_basename="kept",
        max_rows=args.max_rows,
        write_summary_json=True,
        num_proc=args.num_proc,
    )
    elapsed_exact = time.time() - t0

    n_in_exact      = exact_summary["n_in"]
    n_kept_exact    = exact_summary["n_kept"]
    n_dropped_exact = exact_summary["n_dropped"]
    drop_rate_exact = (n_dropped_exact / n_in_exact * 100) if n_in_exact else 0

    summary["exact"] = {
        "elapsed_seconds": round(elapsed_exact, 1),
        "n_in": n_in_exact,
        "n_kept": n_kept_exact,
        "n_dropped": n_dropped_exact,
        "drop_rate_pct": round(drop_rate_exact, 2),
    }

    print(f"Done in {elapsed_exact:.1f}s")
    print(f"  in={n_in_exact:,}  kept={n_kept_exact:,}  dropped={n_dropped_exact:,}  ({drop_rate_exact:.1f}%)")

    # ─────────────────────────────────────────────────────────────
    # Pass 2: Fuzzy RapidFuzz similarity deduplication
    # ─────────────────────────────────────────────────────────────
    print(f"\nPass 2: Fuzzy RapidFuzz similarity deduplication")
    print("=" * 60)
    t0 = time.time()

    fuzzy_output = output_dir / "fuzzy"
    fuzzy_output.mkdir(parents=True, exist_ok=True)

    exact_kept = exact_output / "kept.parquet"
    fuzzy_summary = run_dedup_fuzzy(
        input_path=exact_kept,
        output_dir=fuzzy_output,
        output_basename="kept",
        similarity_threshold=args.fuzzy_threshold,
        write_summary_json=True,
        num_proc=args.num_proc,
    )
    elapsed_fuzzy = time.time() - t0

    n_in_fuzzy      = fuzzy_summary["n_in"]
    n_kept_fuzzy    = fuzzy_summary["n_kept"]
    n_dropped_fuzzy = fuzzy_summary["n_dropped"]
    drop_rate_fuzzy = (n_dropped_fuzzy / n_in_fuzzy * 100) if n_in_fuzzy else 0

    summary["fuzzy"] = {
        "elapsed_seconds": round(elapsed_fuzzy, 1),
        "n_in": n_in_fuzzy,
        "n_kept": n_kept_fuzzy,
        "n_dropped": n_dropped_fuzzy,
        "drop_rate_pct": round(drop_rate_fuzzy, 2),
    }

    print(f"Done in {elapsed_fuzzy:.1f}s")
    print(f"  in={n_in_fuzzy:,}  kept={n_kept_fuzzy:,}  dropped={n_dropped_fuzzy:,}  ({drop_rate_fuzzy:.1f}%)")

    # ─────────────────────────────────────────────────────────────
    # Combine results
    # ─────────────────────────────────────────────────────────────
    print(f"\nCombining results...")

    fuzzy_kept = fuzzy_output / "kept.parquet"
    final_kept = output_dir / "deduped.kept.parquet"
    final_dropped = output_dir / "deduped.dropped.parquet"

    # Final kept = output of fuzzy pass
    import shutil
    shutil.copy(fuzzy_kept, final_kept)

    # Final dropped = union of exact dropped + fuzzy dropped
    exact_dropped = exact_output / "dropped.parquet"
    fuzzy_dropped = fuzzy_output / "dropped.parquet"

    import pandas as pd
    dropped_dfs = []
    if exact_dropped.exists():
        dropped_dfs.append(pd.read_parquet(exact_dropped))
    if fuzzy_dropped.exists():
        dropped_dfs.append(pd.read_parquet(fuzzy_dropped))

    if dropped_dfs:
        combined_dropped = pd.concat(dropped_dfs, ignore_index=True)
        combined_dropped.to_parquet(final_dropped, index=False)
    else:
        # Create empty parquet with same schema as kept
        import pyarrow as pa
        schema = pa.parquet.read_table(final_kept).schema
        pa.parquet.write_table(pa.table({name: [] for name in schema.names}, schema=schema), final_dropped)

    # Update summary
    summary["totals"] = {
        "n_in": n_in_exact,
        "n_kept_after_exact": n_kept_exact,
        "n_kept_after_fuzzy": n_kept_fuzzy,
        "n_dropped_total": n_dropped_exact + n_dropped_fuzzy,
    }
    summary["total_elapsed_seconds"] = round(time.time() - t_global, 1)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Final report
    print(f"\n{'='*60}")
    print(f"Stage 4 (dedup) complete in {summary['total_elapsed_seconds']:.1f}s")
    print(f"  In (before any dedup): {summary['totals']['n_in']:,}")
    print(f"  After exact dedup:     {summary['totals']['n_kept_after_exact']:,}")
    print(f"  After fuzzy dedup:     {summary['totals']['n_kept_after_fuzzy']:,}")
    print(f"  Total dropped:         {summary['totals']['n_dropped_total']:,}")
    print(f"\nOutputs:")
    for fname in ["deduped.kept.parquet", "deduped.dropped.parquet", "summary.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            if fname.endswith(".parquet"):
                print(f"  {fpath}  ({fpath.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"  {fpath}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
