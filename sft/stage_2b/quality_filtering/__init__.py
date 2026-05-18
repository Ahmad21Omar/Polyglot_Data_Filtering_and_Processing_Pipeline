"""Stage 2B — Quality filtering for SFT-Collection-v2.

Public API:

- :class:`EnglishFastTextQualityScorer`     — FastText OH-ELI5 (English-only)
- :class:`MultilingualFineWebHqScorer`      — XLM-R + FineWeb-HQ classifier head (multilingual)
- :func:`run_quality_filter`                — shared runner (load → score → split kept/dropped → save)
- :func:`build_scoring_text`                — shared text-assembly helper
- :func:`calibrate_threshold`               — shared threshold calibration policy
- :class:`ThresholdDecision`                — calibration result
- :data:`DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME`
                                            — language → FineWeb-HQ classifier filename mapping

This module never downloads any model files. The first call to a scorer
that cannot find its model on disk raises ``FileNotFoundError`` with the
``huggingface_hub.hf_hub_download`` snippet to fetch it.
"""

from .english_fasttext import (  # noqa: F401
    EnglishFastTextQualityScorer,
    ThresholdDecision,
    calibrate_threshold,
    score_rows as english_score_rows,
    score_texts as english_score_texts,
)
from .multilingual_fineweb_hq import (  # noqa: F401
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME,
    MultilingualFineWebHqScorer,
    load_classifier_head,
)
from .runner import QualityScorer, run_quality_filter  # noqa: F401
from .text_assembly import build_scoring_text  # noqa: F401


__all__ = [
    # scorers
    "EnglishFastTextQualityScorer",
    "MultilingualFineWebHqScorer",
    # runner
    "run_quality_filter",
    "QualityScorer",
    # helpers
    "build_scoring_text",
    "calibrate_threshold",
    "ThresholdDecision",
    "english_score_rows",
    "english_score_texts",
    "load_classifier_head",
    # constants
    "DEFAULT_EMBEDDING_MODEL_ID",
    "DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME",
]
