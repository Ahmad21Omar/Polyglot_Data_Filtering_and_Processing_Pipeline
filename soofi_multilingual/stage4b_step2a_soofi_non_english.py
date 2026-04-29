"""Stage 4B (Step 2A) — Soofi Non-English Filter & Format.

Filters and normalises the non-English splits of
``toroe/Soofi-Think-SFT-10B-multilingual`` into the canonical SFT-Collection-v2
schema.  Each language is processed independently and gets its own
kept/dropped parquet pair.  A combined summary.json collects stats for all
languages.

This script is a thin wrapper around the existing filter_and_format machinery
in this repository:

    filter_and_format/adapters/soofi_think_sft_10b_multilingual.py  ← adapter
    filter_and_format/pipeline.py                                    ← 6-phase pipeline
    filter_and_format/runner.py                                      ← load + map + save

Filter configuration (reproduces the original SFT-Collection-v2 run)
---------------------------------------------------------------------
Phase            Setting                           Value used here
--------         --------------------------------  ----------------------
2A.1 Structural  min_prompt_chars                  50
                 min_messages                      2
2A.2 Think       require_think_tags                True
                 min_think_chars                   50  ← legacy value; default is 80
2A.3 Question    enable_question_filter            True
                 question_min_length               50  ← legacy value; default is 200
2A.4 Language    enable_fasttext_english_filter    False  (multilingual data — do NOT filter by EN)
                 enable_mixed_language_filter      False  (rows are intentionally non-English)
                 enable_chinese_ratio_filter       True
2A.5 Identity    enable_identity_filter            True
                 enable_cutoff_filter              True
2A.6 Repetition  enable_repetition_filter          True

Input
-----
A directory produced by the data loader containing one HuggingFace Arrow
dataset per language:

    <input-root>/
        german/     ← load_from_disk()-compatible
        french/
        italian/
        spanish/

The default language names correspond to the folder names used by
``data_loader.py``'s ``download_soofi_think_sft_10b_multilingual``.
Pass ``--languages`` to override.

Output (--output-dir)
---------------------
    <output-dir>/
        german/
            kept.parquet        ← filtered rows in canonical schema
            dropped.parquet     ← audit rows with _drop_reason
            summary.json        ← per-language counts
        french/
            ...
        italian/
            ...
        spanish/
            ...
        summary.json            ← combined stats for all languages

Usage
-----
    python stage4b_step2a_soofi_non_english.py \\
        --input-root /path/to/toroe__Soofi-Think-SFT-10B-multilingual \\
        --output-dir /path/to/stage4b_step2a_soofi_non_english

    # Smoke test — first 200 rows per language:
    python stage4b_step2a_soofi_non_english.py \\
        --input-root /path/to/toroe__Soofi-Think-SFT-10B-multilingual \\
        --output-dir /tmp/stage4b_test \\
        --max-examples 200

Optional flags
--------------
  --languages      Comma-separated list of language subfolder names.
                   Default: german,french,italian,spanish
  --max-examples   Stop after N rows per language (smoke test).
  --num-proc       Number of parallel workers for ds.map (default: 1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the Filtering_Pipeline importable when invoked directly (python stage4b_…).
# The filter_and_format package uses relative imports (from ..filters.*)  which
# require Filtering_Pipeline to be a sub-package, so we add its *parent*
# (Master_Thesis/) to sys.path rather than Filtering_Pipeline/ itself.
_MASTER_THESIS_ROOT = Path(__file__).resolve().parents[2]
if str(_MASTER_THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MASTER_THESIS_ROOT))

from Filtering_Pipeline.filter_and_format.adapters.soofi_think_sft_10b_multilingual import (
    SoofiThinkSft10bMultilingualAdapter,
)
from Filtering_Pipeline.filter_and_format.config import FilterConfig
from Filtering_Pipeline.filter_and_format.runner import run as _run_adapter


def _build_config() -> FilterConfig:
    """Return the FilterConfig that reproduces the original SFT-Collection-v2 run.

    Two values differ from the current defaults:
      - min_think_chars = 50   (legacy script used 50; current default 80)
      - question_min_length = 50  (legacy script used 50; current default 200)

    All language filters that would drop non-English text are disabled because
    these rows are intentionally in DE / FR / IT / ES.
    """
    return FilterConfig(
        # 2A.1
        min_prompt_chars=50,
        min_messages=2,
        # 2A.2
        require_think_tags=True,
        min_think_chars=50,
        # 2A.3
        enable_question_filter=True,
        question_min_length=50,
        # 2A.4 — no language gating (rows are non-English by design)
        enable_fasttext_english_filter=False,
        enable_mixed_language_filter=False,
        enable_chinese_ratio_filter=True,
        chinese_ratio_threshold=0.05,
        # 2A.5
        enable_identity_filter=True,
        enable_cutoff_filter=True,
        enable_safety_flag_filter=False,
        # 2A.6
        enable_repetition_filter=True,
        repetition_min_sentence_repeats=6,
        repetition_phrase_n=3,
        repetition_min_phrase_repeats=30,
        repetition_min_chars=200,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4B: Filter and normalise non-English Soofi splits "
            "(DE / FR / IT / ES) into the canonical SFT-Collection-v2 schema. "
            "Processes each language independently."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-root", required=True,
        help=(
            "Root directory containing per-language HuggingFace Arrow datasets "
            "saved by data_loader.py (e.g. .../toroe__Soofi-Think-SFT-10B-multilingual)."
        ),
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write per-language kept/dropped parquets and summary.json.",
    )
    p.add_argument(
        "--languages",
        default="german,french,italian,spanish",
        help="Comma-separated list of language subfolder names to process.",
    )
    p.add_argument(
        "--max-examples", type=int, default=None,
        help="Stop after N rows per language (smoke test).",
    )
    p.add_argument(
        "--num-proc", type=int, default=1,
        help="Parallel workers for ds.map (>1 uses multiprocessing).",
    )
    args = p.parse_args(argv)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    languages  = [s.strip() for s in args.languages.split(",") if s.strip()]

    if not languages:
        p.error("--languages is empty.")

    cfg     = _build_config()
    adapter = SoofiThinkSft10bMultilingualAdapter()

    combined_summary: dict = {
        "stage": "4B_soofi_non_english",
        "adapter": adapter.dataset_id,
        "languages": languages,
        "per_language": {},
        "totals": {"n_in": 0, "n_kept": 0, "n_dropped": 0},
    }

    t_global = time.time()

    for lang in languages:
        lang_input = input_root / lang
        if not lang_input.exists():
            print(f"  [SKIP] {lang}: input path not found: {lang_input}", flush=True)
            continue

        lang_output = output_dir / lang
        print(f"\n{'='*60}", flush=True)
        print(f"Language: {lang}  →  {lang_output}", flush=True)

        t0 = time.time()
        summary = _run_adapter(
            adapter=adapter,
            cfg=cfg,
            input_dataset_dir=lang_input,
            output_dir=lang_output,
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

        combined_summary["per_language"][lang] = {
            "n_in": n_in,
            "n_kept": n_kept,
            "n_dropped": n_dropped,
            "drop_rate_pct": round(drop_rate, 2),
            "elapsed_seconds": round(elapsed, 1),
            "by_reason": summary.get("by_reason", {}),
        }
        combined_summary["totals"]["n_in"]      += n_in
        combined_summary["totals"]["n_kept"]     += n_kept
        combined_summary["totals"]["n_dropped"]  += n_dropped

        print(f"  Done in {elapsed:.1f}s — "
              f"in={n_in:,}  kept={n_kept:,}  dropped={n_dropped:,}  ({drop_rate:.1f}% drop)",
              flush=True)
        top_reasons = list(summary.get("by_reason", {}).items())[:5]
        if top_reasons:
            print("  Top drop reasons:", flush=True)
            for reason, cnt in top_reasons:
                print(f"    {reason:<35s}  {cnt:>8,}", flush=True)

    # Write combined summary
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary["total_elapsed_seconds"] = round(time.time() - t_global, 1)
    totals = combined_summary["totals"]
    combined_summary["totals"]["drop_rate_pct"] = round(
        totals["n_dropped"] / totals["n_in"] * 100, 2
    ) if totals["n_in"] else 0

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Stage 4B (non-English) complete in {combined_summary['total_elapsed_seconds']:.1f}s")
    print(f"  Total in    : {totals['n_in']:,}")
    print(f"  Total kept  : {totals['n_kept']:,}")
    print(f"  Total dropped: {totals['n_dropped']:,}  ({combined_summary['totals']['drop_rate_pct']}%)")
    print(f"\nOutputs:")
    for lang in languages:
        lang_dir = output_dir / lang
        kept    = lang_dir / "kept.parquet"
        dropped = lang_dir / "dropped.parquet"
        if kept.exists():
            print(f"  {kept}  ({kept.stat().st_size / 1e6:.1f} MB)")
        if dropped.exists():
            print(f"  {dropped}  ({dropped.stat().st_size / 1e6:.1f} MB)")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
