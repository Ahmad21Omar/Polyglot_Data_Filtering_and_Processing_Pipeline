from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# This is a deliberately smaller (but stricter) core extracted from Open-Instruct's
# `filter_ngram_repetitions.py`. Goal: catch "mass repetition" quickly and locally.
#
# Differences vs OLMo3 defaults:
# - stricter thresholds (configurable): by default we catch earlier
# - we focus on assistant completion text for Dolci
# - we return a structured result for drop_reason audits

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
WHITESPACE_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip()).lower()


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts


def find_repeated_sentences(sentences: List[str], *, min_count: int) -> Optional[Tuple[str, int]]:
    if not sentences:
        return None
    c = Counter(_norm(s) for s in sentences if _norm(s))
    if not c:
        return None
    sent, count = c.most_common(1)[0]
    if count >= min_count and len(sent) >= 8:
        return sent, count
    return None


def find_repeated_phrases(text: str, *, n: int, min_count: int) -> Optional[Tuple[str, int]]:
    """Very small n-gram finder over whitespace tokens.

    This intentionally ignores punctuation normalization beyond whitespace, but it catches
    pathological loops like "Therefore therefore therefore ...".
    """
    toks = [t for t in _norm(text).split(" ") if t]
    if len(toks) < n:
        return None

    grams = [" ".join(toks[i : i + n]) for i in range(0, len(toks) - n + 1)]
    c = Counter(grams)
    gram, count = c.most_common(1)[0]

    # Avoid flagging tiny / low-signal grams.
    if count >= min_count and len(gram) >= 12:
        return gram, count
    return None


@dataclass(frozen=True)
class RepetitionResult:
    should_drop: bool
    reason: Optional[str]
    details: Dict[str, object]


def detect_mass_repetition(
    text: str,
    *,
    # Stricter than OLMo3: they mention 10x+ sentences and 50x+ phrases.
    # We'll default to 6x sentences and 30x 3-grams. Tune later after stats.
    min_sentence_repeats: int = 6,
    phrase_n: int = 3,
    min_phrase_repeats: int = 30,
    min_chars: int = 200,
) -> RepetitionResult:
    if not isinstance(text, str) or not text.strip():
        return RepetitionResult(False, None, {})

    if len(text) < min_chars:
        return RepetitionResult(False, None, {"skipped": "too_short"})

    sentences = split_sentences(text)
    rep_sentence = find_repeated_sentences(sentences, min_count=min_sentence_repeats)
    if rep_sentence:
        sent, count = rep_sentence
        return RepetitionResult(
            True,
            "repetition_sentences",
            {
                "sentence": sent,
                "count": count,
                "min_sentence_repeats": min_sentence_repeats,
            },
        )

    rep_phrase = find_repeated_phrases(text, n=phrase_n, min_count=min_phrase_repeats)
    if rep_phrase:
        phrase, count = rep_phrase
        return RepetitionResult(
            True,
            "repetition_phrases",
            {
                "phrase": phrase,
                "count": count,
                "phrase_n": phrase_n,
                "min_phrase_repeats": min_phrase_repeats,
            },
        )

    return RepetitionResult(False, None, {})
