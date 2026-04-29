from __future__ import annotations

"""English-only filtering using pretrained FastText language identification.

This implements an inexpensive language filter inspired by OpenThoughts' approach
("Removing Non-English Answers"), but without calling LLMs.

We use the standard pretrained FastText LID model (`lid.176.bin` or `lid.176.ftz`).

Notes
-----
- We intentionally do *not* download the model automatically to avoid making
  network calls implicitly. Instead, callers provide `model_path`.
- For speed, the module caches the loaded model per-process.

Typical policy:
- Keep only English (`__label__en`) with probability >= threshold (e.g. 0.8).

References
----------
- FastText language identification: https://fasttext.cc/docs/en/language-identification.html
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EnglishLidConfig:
    threshold: float = 0.8
    min_chars: int = 40
    # If True, short strings are kept because language-ID is unreliable on very short texts.
    keep_if_too_short: bool = True


def _require_fasttext():
    try:
        import fasttext  # type: ignore

        return fasttext
    except Exception as e:
        raise RuntimeError(
            "fasttext is not installed in the active Python environment. "
            "Install `fasttext` (system-wide in the container) to use the English LID filter."
        ) from e


@lru_cache(maxsize=8)
def _load_lid_model(model_path: str):
    fasttext = _require_fasttext()
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"FastText LID model not found: {model_path}")
    return fasttext.load_model(str(p))


def _clean_text(text: str) -> str:
    # FastText wants single-line input and behaves better with normalized whitespace.
    return " ".join((text or "").strip().split())


def predict_language(
    text: str,
    *,
    model_path: str,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """Return top-k predicted languages as (`__label__xx`, prob) pairs."""
    model = _load_lid_model(model_path)
    cleaned = _clean_text(text)
    labels, probs = model.predict(cleaned, k=k)
    return list(zip(labels, map(float, probs)))


def english_score(text: str, *, model_path: str) -> float:
    """Return probability for English label `__label__en` (0.0 if not in top-k)."""
    model = _load_lid_model(model_path)
    cleaned = _clean_text(text)
    labels, probs = model.predict(cleaned, k=10)
    try:
        idx = labels.index("__label__en")
        return float(probs[idx])
    except ValueError:
        return 0.0


def is_english_text(
    text: str,
    *,
    model_path: str,
    cfg: EnglishLidConfig = EnglishLidConfig(),
) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if len(text) < cfg.min_chars:
        return cfg.keep_if_too_short
    return english_score(text, model_path=model_path) >= cfg.threshold


def should_drop_non_english(
    text: str,
    *,
    model_path: str,
    cfg: EnglishLidConfig = EnglishLidConfig(),
) -> Optional[str]:
    """Return drop reason if text is confidently non-English, else None."""
    if not isinstance(text, str) or not text.strip():
        return "empty_text"

    # If the pipeline enabled this filter, missing dependencies/models should be
    # treated as configuration errors (fail fast) rather than silently dropping.
    _require_fasttext()

    # Also validate model path early to fail fast with a clear error.
    # (This call is cached after first load.)
    _load_lid_model(model_path)

    if len(text) < cfg.min_chars:
        return None if cfg.keep_if_too_short else "non_english_short"

    score = english_score(text, model_path=model_path)
    if score >= cfg.threshold:
        return None
    return "non_english_fasttext"
