"""English language-ID filter for RL prompt text (FastText-based).

Thin wrapper around the SFT-pipeline FastText LID filter so that the RL
pipeline can import it without duplicating code.  Falls back gracefully when
the FastText model is not installed or the model file is absent.

Drop reason: ``non_english_fasttext`` (propagated from the SFT filter)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RLEnglishLidConfig:
    """Mirror of SFT ``EnglishLidConfig`` — kept separate so RL can be tuned independently."""
    threshold: float = 0.80
    min_chars: int = 80
    keep_if_too_short: bool = True


# ── Try to import the SFT filter ─────────────────────────────────────────────
try:
    from Filtering_Pipeline.sft.stage_2a.filters.english_fasttext_filter import (
        EnglishLidConfig as _SFTEnglishLidConfig,
        should_drop_non_english as _sft_should_drop_non_english,
    )
    _FASTTEXT_AVAILABLE = True
except Exception:
    _FASTTEXT_AVAILABLE = False


def is_fasttext_available() -> bool:
    """Return True if the SFT FastText filter was successfully imported."""
    return _FASTTEXT_AVAILABLE


def should_drop_non_english_prompt(
    prompt_text: str,
    *,
    model_path: str,
    cfg: Optional[RLEnglishLidConfig] = None,
) -> Optional[str]:
    """Return ``'non_english_fasttext'`` if the prompt is detected as non-English.

    Returns ``None`` if FastText is unavailable or the text passes the filter.

    Args:
        prompt_text: Flat prompt string to evaluate.
        model_path:  Absolute path to the FastText ``lid.176.bin`` model file.
        cfg:         Filter configuration.  Uses defaults when ``None``.

    Returns:
        Drop-reason string or ``None``.
    """
    if not _FASTTEXT_AVAILABLE:
        return None

    cfg = cfg or RLEnglishLidConfig()
    return _sft_should_drop_non_english(
        prompt_text,
        model_path=model_path,
        cfg=_SFTEnglishLidConfig(
            threshold=cfg.threshold,
            min_chars=cfg.min_chars,
            keep_if_too_short=cfg.keep_if_too_short,
        ),
    )
