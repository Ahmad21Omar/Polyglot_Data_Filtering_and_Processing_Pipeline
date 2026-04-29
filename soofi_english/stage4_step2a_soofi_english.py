"""Stage 4 (Step 2A) — Soofi English Heuristic Filter & Format.

Filters and normalises the English split of
``toroe/Soofi-Think-SFT-10B-multilingual`` into the canonical SFT-Collection-v2
schema using the full Stage 2A heuristic filter suite.

This script is a thin wrapper around the existing filter_and_format machinery
in this repository:

    filter_and_format/adapters/soofi_think_sft_10b_english.py  ← adapter
    filter_and_format/pipeline.py                               ← 6-phase pipeline
    filter_and_format/runner.py                                 ← load + map + save

Filter configuration (reproduces the original SFT-Collection-v2 run)
---------------------------------------------------------------------
Phase            Setting                           Value used here
--------         --------------------------------  ----------------------
2A.1 Structural  min_prompt_chars                  50
                 min_messages                      2
2A.2 Think       require_think_tags                True
                 min_think_chars                   50
2A.3 Question    enable_question_filter            True
                 question_min_length               50
2A.4 Language    enable_fasttext_english_filter    False  (upstream field guarantees English)
                 enable_mixed_language_filter      False  (monolingual data)
                 enable_chinese_ratio_filter       True
2A.5 Identity    enable_identity_filter            True
                 enable_cutoff_filter              True
2A.6 Repetition  enable_repetition_filter          True

Input
-----
A HuggingFace Arrow dataset saved by data_loader.py under:

    <input-path>/
        (English Soofi dataset, loadable by load_from_disk())

Columns per row:
  - messages      (list of {role, content})
  - source        (string, upstream subsource id)
  - dataset_name  (string, human-readable upstream dataset name)
  - ds_uid        (int64, stable id within source)
  - language      (string, always 'english' for this adapter)

Output (--output-dir)
---------------------
    <output-dir>/
        kept.parquet        ← filtered rows in canonical schema
        dropped.parquet     ← audit rows with _drop_reason
        summary.json        ← filter statistics

Usage
-----
    python stage4_step2a_soofi_english.py \\
        --input-path /path/to/english/soofi/dataset \\
        --output-dir /path/to/stage4_step2a_soofi_english

    # Smoke test — first 200 rows:
    python stage4_step2a_soofi_english.py \\
        --input-path /path/to/english/soofi/dataset \\
        --output-dir /tmp/stage4_test \\
        --max-examples 200

Optional flags
--------------
  --max-examples   Stop after N rows (smoke test).
  --num-proc       Number of parallel workers for ds.map (default: 1).
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

from Filtering_Pipeline.filter_and_format.adapters.soofi_think_sft_10b_english import (
    SoofiThinkSft10bEnglishAdapter,
)
from Filtering_Pipeline.filter_and_format.runner import run as _run_adapter


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4: Filter and normalise English Soofi split "
            "(toroe/Soofi-Think-SFT-10B-multilingual) into the canonical "
            "SFT-Collection-v2 schema with full Stage 2A heuristic filters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-path", required=True,
        help=(
            "Path to the English Arrow dataset saved by data_loader.py, "
            "loadable via load_from_disk()."
        ),
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write kept/dropped parquets and summary.json.",
    )
    p.add_argument(
        "--max-examples", type=int, default=None,
        help="Stop after N rows (smoke test).",
    )
    p.add_argument(
        "--num-proc", type=int, default=1,
        help="Parallel workers for ds.map (>1 uses multiprocessing).",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        p.error(f"Input path not found: {input_path}")

    adapter = SoofiThinkSft10bEnglishAdapter()
    cfg = adapter.recommended_config()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Stage 4 — Soofi English Heuristic Filter")
    print(f"  input-path  : {input_path}")
    print(f"  output-dir  : {output_dir}")
    if args.max_examples:
        print(f"  ** SMOKE TEST: first {args.max_examples:,} rows **")
    print()

    t0 = time.time()
    summary = _run_adapter(
        adapter=adapter,
        cfg=cfg,
        input_dataset_dir=input_path,
        output_dir=output_dir,
        split="train",
        num_proc=args.num_proc,
        max_examples=args.max_examples,
        save_kept=True,
        save_dropped=True,
        write_summary_json=True,
    )
    elapsed = time.time() - t0

    n_in      = summary["n_in"]
    n_kept    = summary["n_kept"]
    n_dropped = summary["n_dropped"]
    drop_rate = n_dropped / n_in * 100 if n_in else 0

    print(f"Done in {elapsed:.1f}s")
    print(f"  in={n_in:,}  kept={n_kept:,}  dropped={n_dropped:,}  ({drop_rate:.1f}% drop)")

    # Top drop reasons
    top_reasons = list(summary.get("by_reason", {}).items())[:5]
    if top_reasons:
        print("\nTop drop reasons:")
        for reason, cnt in top_reasons:
            print(f"  {reason:<35s}  {cnt:>8,}")

    print(f"\n{'='*60}")
    print(f"Stage 4 (heuristic filter) complete in {elapsed:.1f}s")
    print(f"  Total in      : {n_in:,}")
    print(f"  Total kept    : {n_kept:,}")
    print(f"  Total dropped : {n_dropped:,}  ({drop_rate:.1f}%)")
    print(f"\nOutputs:")
    for fname in ["kept.parquet", "dropped.parquet", "summary.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            if fname.endswith(".parquet"):
                print(f"  {fpath}  ({fpath.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"  {fpath}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
