from __future__ import annotations

"""Stage 2B — English quality scorer (FastText OH-ELI5).

Model
-----
``mlfoundations/fasttext-oh-eli5`` is a binary FastText classifier trained
on **OpenHermes + Reddit ELI5** (positive class ``__label__hq``) versus
**Common Crawl / RefinedWeb** (negative class ``__label__cc``). It is fast,
deterministic, and runs entirely on CPU; for English-only corpora it is the
default Stage 2B classifier in SFT-Collection-v2.

Download (one-off)
------------------

.. code-block:: python

    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="mlfoundations/fasttext-oh-eli5",
        filename="openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        local_dir="models/fasttext/quality_filter_oh",
    )

The classifier file is ~800 MB.

Scoring rule
------------
For each row we score ``build_scoring_text(row)`` and read the probability
attached to ``__label__hq``. That probability becomes ``_quality_score``
in the output. A row is **kept** when ``_quality_score >= threshold``.

Calibration
-----------
SFT-Collection-v2 uses a **fixed base threshold** (default ``0.20``) with a
**hard drop-rate cap** (default ``20 %``). If the base threshold would drop
more than the cap, it is *lowered* to the corresponding percentile so that
exactly ``cap`` of the corpus is dropped — never raised above the base.
The calibration logic is implemented in :func:`calibrate_threshold` and
applied by :func:`...runner.run_quality_filter` when the caller does not
supply an explicit threshold.

This module **does not** download the model — callers pass an explicit
``model_path``. Missing files raise a clear ``FileNotFoundError`` with the
download command above.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .text_assembly import build_scoring_text


HQ_LABEL = "__label__hq"
CC_LABEL = "__label__cc"

# fastText has an internal token cap; cutting at 2000 chars keeps the model
# within that cap for nearly every realistic SFT row.
_MAX_CHARS_PER_INPUT = 2000


# ── model loader (cached per-process) ────────────────────────────────────────


def _require_fasttext():
    try:
        import fasttext  # type: ignore

        return fasttext
    except ImportError as e:
        raise RuntimeError(
            "fasttext is not installed. Run: pip install fasttext-wheel"
        ) from e


_DOWNLOAD_HINT = (
    "Download the OH-ELI5 quality classifier with:\n"
    "  from huggingface_hub import hf_hub_download\n"
    "  hf_hub_download(\n"
    "      repo_id='mlfoundations/fasttext-oh-eli5',\n"
    "      filename='openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin',\n"
    "      local_dir='models/fasttext/quality_filter_oh',\n"
    "  )"
)


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    fasttext = _require_fasttext()
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"OH-ELI5 quality classifier not found at: {model_path}\n{_DOWNLOAD_HINT}"
        )
    return fasttext.load_model(str(p))


# ── scoring API ──────────────────────────────────────────────────────────────


def _clean(text: str) -> str:
    """Single-line clean-up + length cap (fastText expects single-line input)."""
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").split())[:_MAX_CHARS_PER_INPUT]


def score_texts(
    texts: Sequence[str],
    *,
    model_path: str,
) -> List[float]:
    """Return ``__label__hq`` probability for each text in ``texts``.

    Texts that fall back to ``__label__cc`` as the top label receive
    ``0.0`` (i.e. lowest possible HQ score). Empty / whitespace-only inputs
    also receive ``0.0``.
    """
    if not texts:
        return []
    model = _load_model(model_path)
    cleaned = [_clean(t) for t in texts]
    labels_batch, probs_batch = model.predict(cleaned, k=2)
    out: List[float] = []
    for c, labs, prbs in zip(cleaned, labels_batch, probs_batch):
        if not c:
            # FastText returns non-zero for empty strings; we treat them as 0.0.
            out.append(0.0)
            continue
        labs = list(labs)
        if HQ_LABEL in labs:
            # clamp: fasttext softmax can produce values microscopically above 1.0
            out.append(min(1.0, float(prbs[labs.index(HQ_LABEL)])))
        else:
            out.append(0.0)
    return out


def score_rows(
    rows: Sequence[dict],
    *,
    model_path: str,
) -> List[float]:
    """Score a sequence of SFT rows. Equivalent to
    ``score_texts([build_scoring_text(r) for r in rows], model_path=...)``.
    """
    return score_texts(
        [build_scoring_text(r) for r in rows],
        model_path=model_path,
    )


# ── threshold calibration ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThresholdDecision:
    """Outcome of :func:`calibrate_threshold`."""

    threshold: float
    drop_rate: float
    note: str


def calibrate_threshold(
    scores: Sequence[float],
    *,
    base_threshold: float,
    max_drop_rate: float,
) -> ThresholdDecision:
    """Decide which threshold to apply for a single corpus.

    Rules (matches the legacy ``quality_filter_sft.py`` policy):

    - ``base_threshold <= 0`` → no filtering, every row is kept.
    - Otherwise: use ``base_threshold``, but if it would drop more than
      ``max_drop_rate`` of the corpus, lower it to the corresponding
      percentile so that exactly ``max_drop_rate`` is dropped. The
      threshold is **never raised** above ``base_threshold``.
    """
    arr = np.asarray(list(scores), dtype=np.float32)
    if base_threshold <= 0.0:
        return ThresholdDecision(
            threshold=0.0,
            drop_rate=0.0,
            note="no threshold — all rows kept",
        )
    if arr.size == 0:
        return ThresholdDecision(
            threshold=base_threshold,
            drop_rate=0.0,
            note=f"empty corpus; base threshold {base_threshold:.4f} unused",
        )
    drop_at_base = float((arr < base_threshold).mean())
    if drop_at_base <= max_drop_rate:
        return ThresholdDecision(
            threshold=float(base_threshold),
            drop_rate=drop_at_base,
            note=(
                f"fixed {base_threshold:.4f} "
                f"({drop_at_base * 100:.1f}% dropped, within {max_drop_rate * 100:.0f}% cap)"
            ),
        )
    capped = float(np.percentile(arr, max_drop_rate * 100))
    actual = float((arr < capped).mean())
    return ThresholdDecision(
        threshold=capped,
        drop_rate=actual,
        note=(
            f"capped at {capped:.4f}: base {base_threshold:.4f} would drop "
            f"{drop_at_base * 100:.1f}% > {max_drop_rate * 100:.0f}% cap → "
            f"lowered to {actual * 100:.1f}% dropped"
        ),
    )


# ── public scorer object (exposes a uniform interface for the runner) ────────


@dataclass(frozen=True)
class EnglishFastTextQualityScorer:
    """Stage 2B English quality scorer (FastText OH-ELI5)."""

    model_path: str
    name: str = "fasttext_oh_eli5"

    def score_rows(self, rows: Sequence[dict]) -> List[float]:
        return score_rows(rows, model_path=self.model_path)

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        return score_texts(texts, model_path=self.model_path)

    def calibrate(
        self,
        scores: Sequence[float],
        *,
        base_threshold: float,
        max_drop_rate: float,
    ) -> ThresholdDecision:
        return calibrate_threshold(
            scores, base_threshold=base_threshold, max_drop_rate=max_drop_rate
        )
