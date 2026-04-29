"""Stage 4B (Step 2B) — Soofi Multilingual Quality Filter (FineWeb-HQ + XLM-R).

Applies the FineWeb-HQ quality classifier to each non-English Soofi language
split produced by Stage 4B Step 2A (stage4b_step2a_soofi_non_english.py).
Each language is scored independently using the language-specific classifier
head from ``epfml/FineWeb-HQ-Classifiers`` on top of frozen
``FacebookAI/xlm-roberta-base`` embeddings.

This script is a thin wrapper around the existing quality-filtering machinery:

    quality_filtering/multilingual_fineweb_hq.py   ← MultilingualFineWebHqScorer
    quality_filtering/runner.py                     ← run_quality_filter

Architecture
------------
::

    [context_messages + response_text]
        │ build_scoring_text() — join user turns with [SEP]
        ▼
    [text string]
        │ XLM-R tokeniser (max 512 tokens)
        ▼
    [token ids + attention mask]
        │ XLM-R base, frozen — mean pooling
        ▼
    [768-dim sentence embedding]
        │ language-specific MLP head (Linear→ReLU→Dropout→Linear, sigmoid)
        ▼
    [quality_score ∈ (0, 1)]   ← compared to --threshold

Classifier heads (one per language)
-------------------------------------
Download once before running this script:

    from huggingface_hub import hf_hub_download
    for fname in ("deu_Latn.pt", "fra_Latn.pt", "ita_Latn.pt", "spa_Latn.pt"):
        hf_hub_download(
            repo_id="epfml/FineWeb-HQ-Classifiers",
            filename=fname,
            local_dir="models/FineWeb-HQ-Classifiers",
        )

Input (--input-root)
--------------------
The output directory of Stage 4B (stage4b_step2a_soofi_non_english.py):

    <input-root>/
        german/kept.parquet
        french/kept.parquet
        italian/kept.parquet
        spanish/kept.parquet

Output (--output-dir)
---------------------
    <output-dir>/
        german/
            kept.parquet     ← quality-filtered rows
            dropped.parquet  ← rows with quality_score < threshold
            stats.json       ← threshold, score percentiles, counts
        french/  ...
        italian/ ...
        spanish/ ...
        summary.json         ← combined stats for all languages

Usage
-----
    python stage4c_soofi_quality_filter.py \\
        --input-root    /path/to/stage4b_step2a_soofi_non_english \\
        --classifiers-dir /path/to/models/FineWeb-HQ-Classifiers \\
        --output-dir    /path/to/stage4c_soofi_quality_filter \\
        --threshold     0.45

    # Smoke test — first 200 rows per language:
    python stage4c_soofi_quality_filter.py \\
        --input-root    /path/to/stage4b_step2a_soofi_non_english \\
        --classifiers-dir /path/to/models/FineWeb-HQ-Classifiers \\
        --output-dir    /tmp/stage4c_test \\
        --threshold     0.45 \\
        --max-rows      200

Optional flags
--------------
  --languages      Comma-separated language subfolder names.
                   Default: german,french,italian,spanish
  --batch-size     Scoring batch size (default: 8 — matches original script).
  --device         "cuda" / "cpu" — auto-detected if omitted.
  --max-rows       Stop after N rows per language (smoke test).
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
    MultilingualFineWebHqScorer,
    run_quality_filter,
)

# Map language folder name → ISO 639-1 code (used by the scorer + classifier filenames)
_LANGUAGE_TO_ISO: dict[str, str] = {
    "german":  "de",
    "french":  "fr",
    "italian": "it",
    "spanish": "es",
    "japanese": "ja",
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stage 4C: Quality-filter non-English Soofi splits "
            "(DE / FR / IT / ES) using FineWeb-HQ + XLM-R. "
            "Processes each language independently."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-root", required=True,
        help=(
            "Root directory of Stage 4B output, containing one subfolder "
            "per language (german/, french/, ...) each with a kept.parquet."
        ),
    )
    p.add_argument(
        "--classifiers-dir", required=True,
        help=(
            "Directory containing the per-language FineWeb-HQ classifier heads "
            "(deu_Latn.pt, fra_Latn.pt, ita_Latn.pt, spa_Latn.pt). "
            "Download with: hf_hub_download(repo_id='epfml/FineWeb-HQ-Classifiers', ...)."
        ),
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write per-language kept/dropped parquets and summary.json.",
    )
    p.add_argument(
        "--threshold", type=float, required=True,
        help="Fixed quality threshold in [0, 1]. Rows with score < threshold are dropped.",
    )
    p.add_argument(
        "--languages",
        default="german,french,italian,spanish",
        help="Comma-separated list of language subfolder names to process.",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Scoring batch size for XLM-R + classifier (default matches original script).",
    )
    p.add_argument(
        "--device", default=None,
        help=(
            "Torch device to use ('cuda', 'cpu', 'cuda:0', ...). "
            "Auto-detected (cuda if available, else cpu) if omitted."
        ),
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after N rows per language (smoke test).",
    )
    args = p.parse_args(argv)

    # Resolve device
    if args.device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    input_root    = Path(args.input_root)
    classifiers_dir = Path(args.classifiers_dir)
    output_dir    = Path(args.output_dir)
    languages     = [s.strip() for s in args.languages.split(",") if s.strip()]

    if not languages:
        p.error("--languages is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage 4C — Soofi Multilingual Quality Filter")
    print(f"  classifiers-dir : {classifiers_dir}")
    print(f"  threshold       : {args.threshold}")
    print(f"  device          : {device}")
    print(f"  batch-size      : {args.batch_size}")
    if args.max_rows:
        print(f"  ** SMOKE TEST: first {args.max_rows:,} rows per language **")

    combined_summary: dict = {
        "stage": "4C_soofi_quality_filter",
        "threshold": args.threshold,
        "device": device,
        "languages": languages,
        "per_language": {},
        "totals": {"n_in": 0, "n_kept": 0, "n_dropped": 0},
    }

    t_global = time.time()

    for lang in languages:
        lang_input = input_root / lang / "kept.parquet"
        if not lang_input.exists():
            print(f"\n  [SKIP] {lang}: kept.parquet not found: {lang_input}", flush=True)
            continue

        iso_code = _LANGUAGE_TO_ISO.get(lang.lower())
        if iso_code is None:
            print(
                f"\n  [SKIP] {lang}: no ISO code mapping — add it to _LANGUAGE_TO_ISO.",
                flush=True,
            )
            continue

        lang_output = output_dir / lang
        print(f"\n{'='*60}", flush=True)
        print(f"Language: {lang} ({iso_code})  →  {lang_output}", flush=True)

        scorer = MultilingualFineWebHqScorer(
            language=iso_code,
            classifiers_dir=classifiers_dir,
            device=device,
            batch_size=args.batch_size,
        )

        t0 = time.time()
        summary = run_quality_filter(
            scorer=scorer,
            input_path=lang_input,
            output_dir=lang_output,
            score_threshold=args.threshold,  # fixed threshold, no calibration
            max_rows=args.max_rows,
            output_basename="kept",
            write_summary_json=True,
        )
        elapsed = time.time() - t0

        n_in      = summary["n_in"]
        n_kept    = summary["n_kept"]
        n_dropped = summary["n_dropped"]
        drop_rate = n_dropped / n_in * 100 if n_in else 0

        combined_summary["per_language"][lang] = {
            "iso_code":           iso_code,
            "n_in":               n_in,
            "n_kept":             n_kept,
            "n_dropped":          n_dropped,
            "drop_rate_pct":      round(drop_rate, 2),
            "elapsed_seconds":    round(elapsed, 1),
            "score_stats":        summary.get("score_stats", {}),
        }
        combined_summary["totals"]["n_in"]      += n_in
        combined_summary["totals"]["n_kept"]     += n_kept
        combined_summary["totals"]["n_dropped"]  += n_dropped

        print(
            f"  Done in {elapsed:.1f}s — "
            f"in={n_in:,}  kept={n_kept:,}  dropped={n_dropped:,}  ({drop_rate:.1f}% drop)",
            flush=True,
        )

    # Combined summary
    combined_summary["total_elapsed_seconds"] = round(time.time() - t_global, 1)
    totals = combined_summary["totals"]
    combined_summary["totals"]["drop_rate_pct"] = (
        round(totals["n_dropped"] / totals["n_in"] * 100, 2) if totals["n_in"] else 0
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Stage 4C (quality filter) complete in {combined_summary['total_elapsed_seconds']:.1f}s")
    print(f"  Total in      : {totals['n_in']:,}")
    print(f"  Total kept    : {totals['n_kept']:,}")
    print(f"  Total dropped : {totals['n_dropped']:,}  ({combined_summary['totals']['drop_rate_pct']}%)")
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
