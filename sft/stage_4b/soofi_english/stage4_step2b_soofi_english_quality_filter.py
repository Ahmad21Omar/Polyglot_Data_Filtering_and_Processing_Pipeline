"""Stage 4 (Step 2B) — Soofi English Quality Filter (FastText OH-ELI5).

Applies the ``mlfoundations/fasttext-oh-eli5`` binary quality classifier to the
English Soofi split produced by Stage 4 Step 2A
(stage4_step2a_soofi_english.py).

This reproduces ``Master_Thesis/Data_pipline/sft/quality_filtering/
quality_filter_soofi_english.py``, rewritten as a thin wrapper around the
shared quality-filtering machinery:

    quality_filtering/english_fasttext.py   ← EnglishFastTextQualityScorer
    quality_filtering/runner.py             ← run_quality_filter

Model
-----
``mlfoundations/fasttext-oh-eli5`` — binary FastText classifier trained on
OpenHermes + Reddit ELI5 (positive class ``__label__hq``) vs. Common Crawl /
RefinedWeb (negative class ``__label__cc``).

Download once before running:

    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="mlfoundations/fasttext-oh-eli5",
        filename="openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        local_dir="models/fasttext/quality_filter_oh",
    )

Threshold policy
----------------
Default: ``--base-threshold 0.0`` → all rows kept, scores saved.

If ``--base-threshold`` > 0 the script uses that threshold, but **caps
the drop rate** at ``--max-drop-rate`` (default 20 %): if the base threshold
would drop more than the cap, it is *lowered* to the percentile that keeps
exactly ``max_drop_rate`` of rows. The threshold is never *raised* above
``base_threshold``. (Reproduced from ``quality_filter_soofi_english.py``.)

Use ``--score-threshold`` to override calibration with a fixed cutoff.

Architecture
------------
::

    [context_messages + response_text]
        │ build_scoring_text() — join user turns + response with [SEP]
        ▼
    [text string (≤2000 chars, single-line)]
        │ fasttext-oh-eli5 bigram classifier
        ▼
    [__label__hq probability ∈ [0, 1]]  ← compared to threshold

Input (--input-dir)
-------------------
Output of Stage 4 Step 2A (stage4_step2a_soofi_english.py):

    <input-dir>/
        kept.parquet   ← rows to be quality-filtered

Output (--output-dir)
---------------------
    <output-dir>/
        kept.parquet     ← quality-filtered rows
        dropped.parquet  ← rows with quality_score < threshold
        stats.json       ← threshold, score percentiles, counts

Usage
-----
    # No filtering — score all rows and keep all (default):
    python stage4_step2b_soofi_english_quality_filter.py \\
        --input-dir   /path/to/stage4_step2a \\
        --model-path  /path/to/models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \\
        --output-dir  /path/to/stage4_step2b

    # With base threshold (auto-calibrated, ≤20% cap):
    python stage4_step2b_soofi_english_quality_filter.py \\
        --input-dir       /path/to/stage4_step2a \\
        --model-path      /path/to/models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \\
        --output-dir      /path/to/stage4_step2b \\
        --base-threshold  0.20

    # Fixed threshold (no calibration):
    python stage4_step2b_soofi_english_quality_filter.py \\
        --input-dir       /path/to/stage4_step2a \\
        --model-path      /path/to/models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \\
        --output-dir      /path/to/stage4_step2b \\
        --score-threshold 0.15

    # Smoke test (first 500 rows):
    python stage4_step2b_soofi_english_quality_filter.py \\
        --input-dir  /path/to/stage4_step2a \\
        --model-path /path/to/models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \\
        --output-dir /tmp/stage4_step2b_test \\
        --max-rows   500

Optional flags
--------------
  --base-threshold F    Preferred minimum HQ score for auto-calibration (default 0.0 = no filtering).
  --max-drop-rate  F    Hard cap on drop rate during calibration (default 0.20 = 20%).
  --score-threshold F   Fixed minimum score — skips calibration entirely.
  --batch-size     N    FastText scoring batch size (default 512).
  --num-proc       N    Parallel ds.map workers (default 4).
  --max-rows       N    Stop after N rows (smoke test).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add Master_Thesis/ to sys.path so Filtering_Pipeline is importable as a package.
_MASTER_THESIS_ROOT = Path(__file__).resolve().parents[3]
if str(_MASTER_THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MASTER_THESIS_ROOT))

from Filtering_Pipeline.stage_2b.quality_filtering import (
    EnglishFastTextQualityScorer,
    run_quality_filter,
)

_DEFAULT_BASE_THRESHOLD = 0.0
_DEFAULT_MAX_DROP_RATE  = 0.20


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4 (Step 2B): Quality-filter English Soofi split "
            "using mlfoundations/fasttext-oh-eli5. "
            "Reproduces quality_filter_soofi_english.py via the shared quality_filtering module."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory of Stage 4 Step 2A output containing kept.parquet.",
    )
    p.add_argument(
        "--model-path", required=True,
        help=(
            "Path to openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin. "
            "Download with: hf_hub_download(repo_id='mlfoundations/fasttext-oh-eli5', ...)."
        ),
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write kept/dropped parquets and stats.json.",
    )
    p.add_argument(
        "--base-threshold", type=float, default=_DEFAULT_BASE_THRESHOLD,
        help=(
            "Preferred minimum __label__hq score for calibrated filtering. "
            "0.0 (default) = no filtering, all rows kept."
        ),
    )
    p.add_argument(
        "--max-drop-rate", type=float, default=_DEFAULT_MAX_DROP_RATE,
        help="Hard cap on drop rate; threshold is lowered if base-threshold would exceed it.",
    )
    p.add_argument(
        "--score-threshold", type=float, default=None,
        help=(
            "Fixed threshold — bypasses calibration. "
            "If given, --base-threshold and --max-drop-rate are ignored."
        ),
    )
    p.add_argument(
        "--batch-size", type=int, default=512,
        help="FastText scoring batch size (passed as batch_rows to run_quality_filter).",
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after N rows (smoke test).",
    )
    args = p.parse_args(argv)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    input_path = input_dir / "kept.parquet"
    if not input_path.exists():
        p.error(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    scorer = EnglishFastTextQualityScorer(model_path=args.model_path)

    # Determine effective threshold / calibration params for run_quality_filter
    if args.score_threshold is not None:
        score_threshold  = args.score_threshold
        base_threshold   = None
        max_drop_rate    = None
        threshold_note   = f"fixed override {args.score_threshold:.4f}"
    else:
        score_threshold  = None
        base_threshold   = args.base_threshold
        max_drop_rate    = args.max_drop_rate
        threshold_note   = (
            f"auto-calibrated (base={base_threshold:.2f}, max_drop={max_drop_rate*100:.0f}%)"
        )

    print(f"Stage 4 — Soofi English Quality Filter (FastText OH-ELI5)")
    print(f"  input-dir       : {input_dir}")
    print(f"  model-path      : {args.model_path}")
    print(f"  threshold policy: {threshold_note}")
    print(f"  batch-size      : {args.batch_size}")
    if args.max_rows:
        print(f"  ** SMOKE TEST: first {args.max_rows:,} rows **")
    print()

    t0 = time.time()
    summary = run_quality_filter(
        scorer=scorer,
        input_path=input_path,
        output_dir=output_dir,
        # If fixed threshold given use it; otherwise pass calibration params
        score_threshold=score_threshold if score_threshold is not None else 0.0,
        base_threshold=base_threshold if base_threshold is not None else _DEFAULT_BASE_THRESHOLD,
        max_drop_rate=max_drop_rate if max_drop_rate is not None else _DEFAULT_MAX_DROP_RATE,
        max_rows=args.max_rows,
        batch_rows=args.batch_size,
        output_basename="kept",
        write_summary_json=True,
    )
    elapsed = time.time() - t0

    n_in      = summary["n_in"]
    n_kept    = summary["n_kept"]
    n_dropped = summary["n_dropped"]
    drop_rate = n_dropped / n_in * 100 if n_in else 0

    print(f"\n{'='*60}")
    print(f"Stage 4 (quality filter) complete in {elapsed:.1f}s")
    print(f"  Total in      : {n_in:,}")
    print(f"  Total kept    : {n_kept:,}")
    print(f"  Total dropped : {n_dropped:,}  ({drop_rate:.1f}%)")
    print(f"\nOutputs:")
    for fname in ["kept.parquet", "dropped.parquet", "stats.json"]:
        fpath = output_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            if fname.endswith(".parquet"):
                print(f"  {fpath}  ({size_mb:.1f} MB)")
            else:
                print(f"  {fpath}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
