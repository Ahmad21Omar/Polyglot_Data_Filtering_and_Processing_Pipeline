from __future__ import annotations

"""Phase 2A.4 — Language filters.

Sankey-Stage label: ``2A.4_language`` (FastText LID, Chinese ratio).

Facade module that bundles the three canonical language-related modules:

- :mod:`Filtering_Pipeline.stage_2a.filters.language_filters`        — Chinese-character ratio
- :mod:`Filtering_Pipeline.stage_2a.filters.english_fasttext_filter` — English-only FastText LID
- :mod:`Filtering_Pipeline.stage_2a.filters.language_detector`       — generic 218-language detector

The original modules remain the canonical implementation; this file just
re-exports them in one place so each new pipeline script can do a single
``from Filtering_Pipeline.stage_2a.filters import phase_2a4_language as lang`` import.

Drop reasons typically attached at this phase
---------------------------------------------
- ``chinese_ratio``         : assistant content has CJK ratio above threshold
- ``non_english_fasttext``  : FastText English score below threshold (English-only datasets)
- ``mixed_language``        : top language confidence too low (multilingual datasets)
"""

# ── Chinese ratio (language_filters) ─────────────────────────────────────────
from .language_filters import (  # noqa: F401
    CJK_UNIFIED_IDEOGRAPHS_RE,
    chinese_character_ratio,
    has_too_much_chinese,
    should_filter_example_by_chinese,
)

# ── English-only FastText (english_fasttext_filter) ──────────────────────────
from .english_fasttext_filter import (  # noqa: F401
    EnglishLidConfig,
    predict_language,
    english_score,
    is_english_text,
    should_drop_non_english,
)

# ── 218-language detector (language_detector) ────────────────────────────────
from .language_detector import (  # noqa: F401
    MODEL_PATH as LANGUAGE_DETECTOR_MODEL_PATH,
    label_to_short,
    detect_language,
    detect_language_with_confidence,
    detect_language_top_k,
    is_mixed_language,
)

__all__ = [
    # Chinese ratio
    "CJK_UNIFIED_IDEOGRAPHS_RE",
    "chinese_character_ratio",
    "has_too_much_chinese",
    "should_filter_example_by_chinese",
    # English-only FastText
    "EnglishLidConfig",
    "predict_language",
    "english_score",
    "is_english_text",
    "should_drop_non_english",
    # 218-language detector
    "LANGUAGE_DETECTOR_MODEL_PATH",
    "label_to_short",
    "detect_language",
    "detect_language_with_confidence",
    "detect_language_top_k",
    "is_mixed_language",
]
