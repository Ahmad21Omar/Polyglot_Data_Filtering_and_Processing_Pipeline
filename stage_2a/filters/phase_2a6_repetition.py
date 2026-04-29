from __future__ import annotations

"""Phase 2A.6 — Repetition filters.

Sankey-Stage label: ``2A.6_repetition`` (sentences, phrases).

Facade re-exporting the canonical implementation from
:mod:`Filtering_Pipeline.stage_2a.filters.repetition_filters`.

Drop reasons typically attached at this phase
---------------------------------------------
- ``repetition_sentences`` : same normalised sentence repeats >= ``min_sentence_repeats`` times
                             (default 6, length >= 8 chars)
- ``repetition_phrases``   : same n-gram (default 3-gram) repeats >= ``min_phrase_repeats`` times
                             (default 30, n-gram length >= 12 chars)
"""

from .repetition_filters import (  # noqa: F401
    SENTENCE_SPLIT_RE,
    WHITESPACE_RE,
    RepetitionResult,
    split_sentences,
    find_repeated_sentences,
    find_repeated_phrases,
    detect_mass_repetition,
)

__all__ = [
    "SENTENCE_SPLIT_RE",
    "WHITESPACE_RE",
    "RepetitionResult",
    "split_sentences",
    "find_repeated_sentences",
    "find_repeated_phrases",
    "detect_mass_repetition",
]
