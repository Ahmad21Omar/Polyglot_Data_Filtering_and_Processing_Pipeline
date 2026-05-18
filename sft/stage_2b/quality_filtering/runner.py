from __future__ import annotations

"""Shared runner for Stage 2B quality filtering.

Reads one or more SFT-Collection-v2 ``kept.parquet`` files (the output of
Stage 2A), scores every row through a pluggable scorer, and writes per-input
kept / dropped Parquet pairs plus a JSON sidecar with the score statistics
and the threshold decision.

The same runner is used by both the English (FastText OH-ELI5) and the
multilingual (FineWeb-HQ + XLM-R) scorers — they implement the small
:class:`QualityScorer` protocol below.

Usage
-----

.. code-block:: python

    from Filtering_Pipeline.stage_2b.quality_filtering import (
        EnglishFastTextQualityScorer, run_quality_filter,
    )

    scorer = EnglishFastTextQualityScorer(
        model_path="models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
    )
    summary = run_quality_filter(
        scorer=scorer,
        input_path="/.../sft_dolci_v0.kept.parquet",
        output_dir="/.../quality_filtered/dolci",
        base_threshold=0.20,
        max_drop_rate=0.20,
    )
    print(summary["n_kept"], summary["n_dropped"], summary["threshold"])

The runner is **dataset-agnostic**: it only requires the input Parquet to
have ``context_messages`` and ``response_text`` columns (the two columns
used by :func:`...text_assembly.build_scoring_text`). All other Stage 2A
columns are passed through verbatim, plus two new columns are added:

- ``_quality_score``  : float in ``[0, 1]``
- ``_quality_label``  : ``"HIGH"`` (kept) or ``"LOW"`` (dropped)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .english_fasttext import ThresholdDecision


# ── scorer protocol ─────────────────────────────────────────────────────────


class QualityScorer(Protocol):
    """Tiny structural protocol satisfied by all Stage 2B scorers."""

    name: str

    def score_rows(self, rows: Sequence[dict]) -> List[float]:
        ...

    def calibrate(
        self,
        scores: Sequence[float],
        *,
        base_threshold: float,
        max_drop_rate: float,
    ) -> ThresholdDecision:
        ...


# ── helpers ─────────────────────────────────────────────────────────────────


def _score_stats(scores: np.ndarray) -> Dict[str, float]:
    if scores.size == 0:
        return {"min": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0,
                "p75": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min":  round(float(scores.min()), 4),
        "p10":  round(float(np.percentile(scores, 10)), 4),
        "p25":  round(float(np.percentile(scores, 25)), 4),
        "p50":  round(float(np.percentile(scores, 50)), 4),
        "p75":  round(float(np.percentile(scores, 75)), 4),
        "mean": round(float(scores.mean()), 4),
        "max":  round(float(scores.max()), 4),
    }


def _atomic_write_parquet(df, path: Path) -> None:
    """Write to ``<path>.tmp.parquet`` then rename — partial files never
    appear at the final path even if the process is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)


# ── public entry point ──────────────────────────────────────────────────────


def run_quality_filter(
    *,
    scorer: QualityScorer,
    input_path: str | Path,
    output_dir: str | Path,
    base_threshold: float = 0.20,
    max_drop_rate: float = 0.20,
    score_threshold: Optional[float] = None,
    max_rows: Optional[int] = None,
    batch_rows: int = 512,
    output_basename: Optional[str] = None,
    write_summary_json: bool = True,
) -> Dict[str, Any]:
    """Score one Stage 2A ``kept.parquet`` file and split it into kept / dropped.

    Parameters
    ----------
    scorer
        Any object that satisfies :class:`QualityScorer`. Typically one of
        :class:`...english_fasttext.EnglishFastTextQualityScorer` or
        :class:`...multilingual_fineweb_hq.MultilingualFineWebHqScorer`.
    input_path
        Path to the Stage 2A output (``*.kept.parquet``).
    output_dir
        Directory created for this run. Contents:

        - ``<basename>.kept.parquet``     — rows with score >= threshold
        - ``<basename>.dropped.parquet``  — rows with score < threshold
        - ``<basename>.stats.json``       — summary with threshold + score percentiles
    base_threshold
        Quality threshold the calibrator tries first. Default ``0.20``
        (matches the SFT-Collection-v2 setup). Pass ``0`` to disable
        filtering and just write the score column through.
    max_drop_rate
        Hard cap on the per-corpus drop rate. If ``base_threshold`` would
        drop more than this fraction, it is *lowered* to the corresponding
        percentile so that exactly ``max_drop_rate`` is dropped. Default
        ``0.20`` (20%).
    score_threshold
        Optional fixed override that bypasses calibration entirely.
    max_rows
        If set, only the first N rows are processed (useful for smoke tests).
    batch_rows
        Chunk size for scoring. Larger values are faster on GPU but use
        more memory.
    output_basename
        Stem used for the output files. Default: input file stem with the
        ``.kept`` suffix removed (so a ``sft_xyz.kept.parquet`` input
        produces ``sft_xyz.kept.parquet`` / ``sft_xyz.dropped.parquet`` /
        ``sft_xyz.stats.json``).

    Returns
    -------
    dict
        ``{"input_path", "n_in", "n_kept", "n_dropped", "threshold",
        "drop_rate", "score_stats", "scorer_name", "elapsed_s"}``.
    """
    import pandas as pd

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Stage 2B input not found: {input_path}")

    if output_basename is None:
        stem = input_path.stem
        if stem.endswith(".kept"):
            stem = stem[: -len(".kept")]
        output_basename = stem

    t0 = time.time()
    df = pd.read_parquet(input_path)
    if max_rows is not None and max_rows > 0:
        df = df.iloc[: max_rows].reset_index(drop=True)
    n_in = len(df)

    if "context_messages" not in df.columns and "response_text" not in df.columns:
        raise ValueError(
            f"Stage 2B input {input_path} has neither `context_messages` nor "
            f"`response_text` — cannot score. Did you point at a Stage 2A output?"
        )

    # ── 1) Score in batches ───────────────────────────────────────────────
    scores: List[float] = []
    for start in range(0, n_in, batch_rows):
        chunk = df.iloc[start : start + batch_rows].to_dict(orient="records")
        scores.extend(scorer.score_rows(chunk))
    scores_arr = np.asarray(scores, dtype=np.float32)

    # ── 2) Threshold ──────────────────────────────────────────────────────
    if score_threshold is not None:
        threshold = float(score_threshold)
        decision_note = f"explicit override {threshold:.4f}"
        decision_drop = float((scores_arr < threshold).mean()) if scores_arr.size else 0.0
    else:
        decision = scorer.calibrate(
            scores_arr,
            base_threshold=base_threshold,
            max_drop_rate=max_drop_rate,
        )
        threshold = float(decision.threshold)
        decision_note = decision.note
        decision_drop = float(decision.drop_rate)

    # ── 3) Split kept / dropped ───────────────────────────────────────────
    df["_quality_score"] = scores_arr
    if threshold > 0:
        keep_mask = scores_arr >= threshold
    else:
        keep_mask = np.ones(scores_arr.shape, dtype=bool)
    df["_quality_label"] = np.where(keep_mask, "HIGH", "LOW")

    kept_df = df.loc[keep_mask].reset_index(drop=True)
    dropped_df = df.loc[~keep_mask].reset_index(drop=True)
    n_kept = len(kept_df)
    n_dropped = len(dropped_df)

    # ── 4) Atomic write ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_path    = output_dir / f"{output_basename}.kept.parquet"
    dropped_path = output_dir / f"{output_basename}.dropped.parquet"
    stats_path   = output_dir / f"{output_basename}.stats.json"
    _atomic_write_parquet(kept_df, kept_path)
    _atomic_write_parquet(dropped_df, dropped_path)

    elapsed = time.time() - t0

    summary: Dict[str, Any] = {
        "input_path":  str(input_path),
        "scorer_name": scorer.name,
        "n_in":        n_in,
        "n_kept":      n_kept,
        "n_dropped":   n_dropped,
        "drop_rate":   round(decision_drop, 4),
        "threshold":   round(threshold, 6),
        "threshold_note":   decision_note,
        "base_threshold":   base_threshold,
        "max_drop_rate":    max_drop_rate,
        "score_stats":      _score_stats(scores_arr),
        "elapsed_s":        round(elapsed, 2),
        "kept_path":        str(kept_path),
        "dropped_path":     str(dropped_path),
    }

    if write_summary_json:
        stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary
