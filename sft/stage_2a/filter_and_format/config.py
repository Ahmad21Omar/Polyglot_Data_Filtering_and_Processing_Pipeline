from __future__ import annotations

"""Configuration for the generic Stage 2A filter pipeline.

A single ``FilterConfig`` controls every threshold and every on/off toggle
used by :mod:`Filtering_Pipeline.stage_2a.filter_and_format.pipeline`. Per-dataset
adapters do *not* override these values directly; if a particular dataset
needs different thresholds, callers pass a tweaked config to the runner.

All defaults match the values used in the legacy per-dataset scripts that
produced SFT-Collection-v2 (``min_prompt_chars=50`` from ``FilterCfg``,
``min_think_chars=80``, ``question_min_length=200``, FastText English
threshold ``0.8``, repetition thresholds ``6 / 30``).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── default model paths (relative to Master_Thesis/) ─────────────────────────
# These paths match the layout used by the legacy Stage 2A scripts that
# produced SFT-Collection-v2; they are *defaults*, callers can always
# override them via FilterConfig fields.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # Master_Thesis/

# Legacy used `lid.176.bin` (Fasttext.cc original filename); the same model
# is also published on HF as ``model.bin``. Either filename works as long as
# the file exists at this path.
_DEFAULT_FASTTEXT_LID_PATH = _REPO_ROOT / "models" / "fasttext" / "lid.176.bin"
_DEFAULT_LANGUAGE_DETECTOR_PATH = (
    _REPO_ROOT / "models" / "fasttext" / "language_detector" / "model.bin"
)


@dataclass
class FilterConfig:
    """All thresholds and toggles for a Stage 2A run.

    Phase 2A.1 — Structural
    -----------------------
    min_prompt_chars
        Minimum length (after whitespace normalisation) of the joined non-
        assistant prompt text. Drop reason: ``prompt_too_short``.
    min_messages
        Minimum number of turns in the ``messages`` list. Drop reason:
        ``bad_messages``.

    Phase 2A.2 — Think / Reasoning
    ------------------------------
    require_think_tags
        If True, drop rows whose assistant content has no ``<think>...</think>``
        block. Drop reason: ``missing_think``.
    min_think_chars
        Minimum length (after whitespace normalisation) of the extracted think
        text. Drop reason: ``think_too_short``. Only applied when
        ``require_think_tags`` is True.

    Phase 2A.3 — Prompt / Question
    ------------------------------
    enable_question_filter
        If False, skip the entire question-quality filter.
    question_min_length
        Minimum length of the prompt text inside the question filter. Drop
        reason: ``question_too_short``.

    Phase 2A.4 — Language
    ---------------------
    enable_chinese_ratio_filter
        If True, drop rows whose final assistant turn has too many CJK chars.
        Drop reason: ``chinese_ratio``.
    chinese_ratio_threshold
        CJK-char ratio threshold (default 5%).

    enable_fasttext_english_filter
        If True, drop rows whose (prompt + assistant content) is not English.
        Drop reason: ``non_english_fasttext``. Mutually exclusive with the
        mixed-language filter.
    fasttext_threshold, fasttext_min_chars, fasttext_keep_if_too_short
        Forwarded to :class:`Filtering_Pipeline.stage_2a.filters.phase_2a4_language.EnglishLidConfig`.
    fasttext_model_path
        Path to ``lid.176.bin`` / ``model.bin`` for the English-only filter.

    enable_mixed_language_filter
        If True, drop rows where the 218-language detector top confidence is
        below ``mixed_language_confidence_threshold``. Drop reason:
        ``mixed_language``. Used for multilingual datasets instead of the
        English-only filter.
    mixed_language_min_chars, mixed_language_confidence_threshold
        Forwarded to :func:`...phase_2a4_language.is_mixed_language`.
    language_detector_model_path
        Path to ``facebook/fasttext-language-identification`` / ``model.bin``
        for the 218-language detector.

    Phase 2A.5 — Identity / Safety
    ------------------------------
    enable_identity_filter
        If True, drop rows containing model self-identification phrases.
        Drop reason: ``identity_self_id``.
    enable_cutoff_filter
        If True, drop rows mentioning a knowledge-cutoff date. Drop reason:
        ``cutoff_mention``.
    enable_safety_flag_filter
        If True, honour ``toxic`` / ``redacted`` flags in the source row.
        Drop reasons: ``unsafe_toxic`` / ``unsafe_redacted``.

    Phase 2A.6 — Repetition
    -----------------------
    enable_repetition_filter
        If True, drop rows whose assistant content shows mass repetition.
    repetition_min_sentence_repeats, repetition_phrase_n,
    repetition_min_phrase_repeats, repetition_min_chars
        Forwarded to :func:`...phase_2a6_repetition.detect_mass_repetition`.

    Audit
    -----
    prompt_text_preview_chars, assistant_content_preview_chars
        How many characters of the (potentially dropped) prompt / assistant
        text to keep in the audit row, so dropped rows can be re-inspected
        without re-loading the source dataset.
    """

    # --- 2A.1 ---
    min_prompt_chars: int = 50
    min_messages: int = 2

    # --- 2A.2 ---
    require_think_tags: bool = True
    min_think_chars: int = 80

    # --- 2A.3 ---
    enable_question_filter: bool = True
    question_min_length: int = 200

    # --- 2A.4 ---
    enable_chinese_ratio_filter: bool = True
    chinese_ratio_threshold: float = 0.05

    enable_fasttext_english_filter: bool = False
    fasttext_threshold: float = 0.8
    # Defaults match the legacy Stage 2A dolci script (FilterCfg + CLI defaults):
    # the underlying EnglishLidConfig uses min_chars=40, but the SFT-Collection-v2
    # run set min_chars=80 to be more conservative on short snippets.
    fasttext_min_chars: int = 80
    fasttext_keep_if_too_short: bool = True
    fasttext_model_path: str = str(_DEFAULT_FASTTEXT_LID_PATH)

    enable_mixed_language_filter: bool = False
    mixed_language_min_chars: int = 20
    mixed_language_confidence_threshold: float = 0.75
    language_detector_model_path: str = str(_DEFAULT_LANGUAGE_DETECTOR_PATH)

    # --- 2A.5 ---
    enable_identity_filter: bool = True
    enable_cutoff_filter: bool = True
    enable_safety_flag_filter: bool = False

    # --- 2A.6 ---
    enable_repetition_filter: bool = True
    repetition_min_sentence_repeats: int = 6
    repetition_phrase_n: int = 3
    repetition_min_phrase_repeats: int = 30
    repetition_min_chars: int = 200

    # --- Audit ---
    prompt_text_preview_chars: int = 2000
    assistant_content_preview_chars: int = 4000

    def validate(self) -> None:
        """Sanity-check the configuration."""
        if self.enable_fasttext_english_filter and self.enable_mixed_language_filter:
            raise ValueError(
                "FilterConfig: enable_fasttext_english_filter and "
                "enable_mixed_language_filter are mutually exclusive. "
                "Use the English-only filter for English-only datasets and the "
                "mixed-language filter for multilingual datasets."
            )
        if self.min_prompt_chars < 0 or self.min_messages < 0:
            raise ValueError("FilterConfig: lengths must be non-negative.")


# Common presets used by per-dataset adapters.
def english_only_preset() -> FilterConfig:
    """Default for English-only source datasets (Dolci, OpenThoughts3, ...)."""
    return FilterConfig(
        enable_fasttext_english_filter=True,
        enable_mixed_language_filter=False,
    )


def multilingual_preset() -> FilterConfig:
    """Default for multilingual source datasets (Soofi, AIML-dolci-4-lang, ...)."""
    return FilterConfig(
        enable_fasttext_english_filter=False,
        enable_mixed_language_filter=True,
    )
