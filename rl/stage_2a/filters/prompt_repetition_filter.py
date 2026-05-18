"""Repetition filter for RL prompt text.

Detects pathological repetition patterns in prompt text that indicate
broken or machine-generated low-quality data (e.g., copy-paste loops,
degenerate token repetitions).

Adapted from the SFT repetition filter
(`Data_pipline/sft/filters/repetition_filters.py`) but applied to the
*prompt* rather than the assistant response.

Drop reason: ``prompt_repetition``
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text.strip()).lower()


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]


@dataclass
class RepetitionResult:
    should_drop: bool = False
    reason: Optional[str] = None


def detect_prompt_repetition(
    prompt_text: str,
    *,
    min_sentence_repeats: int = 5,
    phrase_n: int = 4,
    min_phrase_repeats: int = 15,
    min_chars: int = 80,
) -> RepetitionResult:
    """Detect mass repetition in *prompt_text*.

    Two independent checks are run:

    1. **Sentence-level**: split on ``[.!?]+`` and count repeated sentences.
       Triggers when any normalised sentence appears ≥ *min_sentence_repeats* times.

    2. **N-gram-level**: build whitespace-token n-grams of length *phrase_n* and
       check if any phrase repeats ≥ *min_phrase_repeats* times.

    Both checks are skipped if the text is shorter than *min_chars*.

    Returns:
        :class:`RepetitionResult` with ``should_drop=True`` and a drop reason
        string if repetition is detected, otherwise ``should_drop=False``.
    """
    text = prompt_text or ""
    if len(text) < min_chars:
        return RepetitionResult()

    # ── 1) Sentence repetition ───────────────────────────────────────────
    sentences = _split_sentences(text)
    if sentences:
        counts = Counter(_norm(s) for s in sentences if _norm(s))
        if counts:
            top_sent, top_count = counts.most_common(1)[0]
            if top_count >= min_sentence_repeats and len(top_sent) >= 8:
                return RepetitionResult(
                    should_drop=True,
                    reason=f"prompt_repetition:sentence×{top_count}",
                )

    # ── 2) N-gram repetition ─────────────────────────────────────────────
    tokens = [t for t in _norm(text).split(" ") if t]
    if len(tokens) >= phrase_n:
        grams = [" ".join(tokens[i: i + phrase_n]) for i in range(len(tokens) - phrase_n + 1)]
        gram_counts = Counter(grams)
        top_gram, top_gram_count = gram_counts.most_common(1)[0]
        if top_gram_count >= min_phrase_repeats and len(top_gram) >= 12:
            return RepetitionResult(
                should_drop=True,
                reason=f"prompt_repetition:ngram×{top_gram_count}",
            )

    return RepetitionResult()


def should_drop_prompt_repetition(
    prompt_text: str,
    *,
    min_sentence_repeats: int = 5,
    phrase_n: int = 4,
    min_phrase_repeats: int = 15,
    min_chars: int = 80,
) -> Optional[str]:
    """Convenience wrapper — returns drop reason string or ``None``."""
    result = detect_prompt_repetition(
        prompt_text,
        min_sentence_repeats=min_sentence_repeats,
        phrase_n=phrase_n,
        min_phrase_repeats=min_phrase_repeats,
        min_chars=min_chars,
    )
    return result.reason if result.should_drop else None
