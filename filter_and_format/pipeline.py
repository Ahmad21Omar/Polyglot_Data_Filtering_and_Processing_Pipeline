from __future__ import annotations

"""Generic Stage 2A pipeline.

Runs the six Sankey sub-stages **in canonical order** for one row at a time:

    2A.1 Structural        -> bad_messages / last_not_assistant
                              prompt_too_short / empty_assistant
    2A.2 Think / Reasoning -> truncated_think / missing_think / think_too_short
    2A.3 Prompt / Question -> question_has_http / question_has_image_ref
                              question_multipart / question_too_short
    2A.4 Language          -> chinese_ratio / non_english_fasttext / mixed_language
    2A.5 Identity / Safety -> identity_self_id / cutoff_mention
                              unsafe_toxic / unsafe_redacted
    2A.6 Repetition        -> repetition_sentences / repetition_phrases

This canonical order **differs slightly** from the legacy per-dataset scripts
(e.g. the dolci script ran the question filter between 2A.1 sub-checks and
moved identity in front of think extraction). The pipeline accepts that the
absolute drop-counts will not be byte-identical to the historical Stage 2A
run, but the *output schema* (``output_schema.KEPT_COLUMNS``) and the *set
of drop reasons* are identical, which is what downstream consumers depend on.

Pre-checks supplied by the adapter (``adapter.pre_check``) run *before* 2A.1
and produce the dataset-specific drop reasons (``missing_dataset_source``,
``no_verified_proof``, ``empty_reasoning_content``, ``missing_language``,
...).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..filters.phase_2a1_structural import (
    evaluate_structural,
    messages_to_prompt_text,
    normalize_whitespace,
)
from ..filters.phase_2a2_think_reasoning import (
    extract_think_and_answer,
    is_truncated_think,
)
from ..filters.phase_2a3_prompt_question import should_drop_question_text
from ..filters.phase_2a4_language import (
    EnglishLidConfig,
    is_mixed_language,
    should_drop_non_english,
    should_filter_example_by_chinese,
)
from ..filters.phase_2a5_identity_safety import (
    has_cutoff_mention,
    should_drop_identity_text,
    should_drop_wildchat_like_row,
)
from ..filters.phase_2a6_repetition import detect_mass_repetition
from .adapter import DatasetAdapter
from .config import FilterConfig
from .output_schema import (
    KeptRowFields,
    build_dropped_row,
    build_kept_row,
    messages_to_context_messages,
    stable_sha256,
)


@dataclass
class _PhaseOutcome:
    """Internal: result of running the 6 phases on a single row."""

    drop_reason: Optional[str]
    prompt_text: str
    assistant_content: str
    reasoning_text: Optional[str]
    response_text: Optional[str]
    final_answer_text: Optional[str]


def _run_phases(
    *,
    messages: Any,
    cfg: FilterConfig,
    safety_flags_row: Optional[Dict[str, Any]] = None,
) -> _PhaseOutcome:
    """Run the six Stage-2A phases in canonical order on one row's messages.

    Returns the first drop reason encountered (short-circuit), and — for kept
    rows — the extracted reasoning and response text.
    """
    # ── Phase 2A.1: Structural ────────────────────────────────────────────
    structural = evaluate_structural(
        messages,
        min_prompt_chars=cfg.min_prompt_chars,
        min_messages=cfg.min_messages,
    )
    if structural.drop_reason is not None:
        return _PhaseOutcome(
            drop_reason=structural.drop_reason,
            prompt_text=structural.prompt_text,
            assistant_content=structural.assistant_content,
            reasoning_text=None,
            response_text=None,
            final_answer_text=None,
        )

    prompt_text = structural.prompt_text
    assistant_content = structural.assistant_content

    # ── Phase 2A.2: Think / Reasoning ─────────────────────────────────────
    if is_truncated_think(assistant_content):
        return _PhaseOutcome(
            "truncated_think", prompt_text, assistant_content, None, None, None
        )

    reasoning_text, response_text, answer_tag_text = extract_think_and_answer(
        assistant_content
    )

    if cfg.require_think_tags:
        if reasoning_text is None:
            return _PhaseOutcome(
                "missing_think", prompt_text, assistant_content, None, None, None
            )
        if len(normalize_whitespace(reasoning_text)) < cfg.min_think_chars:
            return _PhaseOutcome(
                "think_too_short", prompt_text, assistant_content, None, None, None
            )

    # ── Phase 2A.3: Prompt / Question ─────────────────────────────────────
    if cfg.enable_question_filter:
        q_reason = should_drop_question_text(
            prompt_text,
            min_length=cfg.question_min_length,
            drop_if_has_http=True,
            drop_if_has_images=True,
            drop_if_multipart=True,
        )
        if q_reason is not None:
            return _PhaseOutcome(
                q_reason, prompt_text, assistant_content, None, None, None
            )

    # ── Phase 2A.4: Language ──────────────────────────────────────────────
    if cfg.enable_chinese_ratio_filter and should_filter_example_by_chinese(
        messages=messages,
        threshold=cfg.chinese_ratio_threshold,
        include_user_turns=False,
    ):
        return _PhaseOutcome(
            "chinese_ratio", prompt_text, assistant_content, None, None, None
        )

    if cfg.enable_fasttext_english_filter:
        lang_reason = should_drop_non_english(
            prompt_text + "\n\n" + assistant_content,
            model_path=cfg.fasttext_model_path,
            cfg=EnglishLidConfig(
                threshold=cfg.fasttext_threshold,
                min_chars=cfg.fasttext_min_chars,
                keep_if_too_short=cfg.fasttext_keep_if_too_short,
            ),
        )
        if lang_reason is not None:
            return _PhaseOutcome(
                lang_reason, prompt_text, assistant_content, None, None, None
            )

    if cfg.enable_mixed_language_filter and is_mixed_language(
        prompt_text + "\n\n" + assistant_content,
        model_path=cfg.language_detector_model_path,
        min_chars=cfg.mixed_language_min_chars,
        confidence_threshold=cfg.mixed_language_confidence_threshold,
    ):
        return _PhaseOutcome(
            "mixed_language", prompt_text, assistant_content, None, None, None
        )

    # ── Phase 2A.5: Identity / Safety ─────────────────────────────────────
    if cfg.enable_identity_filter:
        id_reason = should_drop_identity_text(assistant_content)
        if id_reason is not None:
            return _PhaseOutcome(
                id_reason, prompt_text, assistant_content, None, None, None
            )

    if cfg.enable_cutoff_filter and has_cutoff_mention(messages):
        return _PhaseOutcome(
            "cutoff_mention", prompt_text, assistant_content, None, None, None
        )

    if cfg.enable_safety_flag_filter and safety_flags_row is not None:
        safety_reason = should_drop_wildchat_like_row(safety_flags_row)
        if safety_reason is not None:
            return _PhaseOutcome(
                safety_reason, prompt_text, assistant_content, None, None, None
            )

    # ── Phase 2A.6: Repetition ────────────────────────────────────────────
    if cfg.enable_repetition_filter:
        rep = detect_mass_repetition(
            assistant_content,
            min_sentence_repeats=cfg.repetition_min_sentence_repeats,
            phrase_n=cfg.repetition_phrase_n,
            min_phrase_repeats=cfg.repetition_min_phrase_repeats,
            min_chars=cfg.repetition_min_chars,
        )
        if rep.should_drop:
            return _PhaseOutcome(
                rep.reason or "repetition",
                prompt_text,
                assistant_content,
                None,
                None,
                None,
            )

    # All 6 phases passed.
    return _PhaseOutcome(
        drop_reason=None,
        prompt_text=prompt_text,
        assistant_content=assistant_content,
        reasoning_text=reasoning_text,
        response_text=response_text or "",
        final_answer_text=answer_tag_text,
    )


# ── public row-mapping function ──────────────────────────────────────────────


def map_row(
    *,
    ex: Dict[str, Any],
    idx: int,
    adapter: DatasetAdapter,
    cfg: FilterConfig,
    split: str,
) -> Dict[str, Any]:
    """Process one source row through the full Stage 2A pipeline.

    Always returns a dict with the canonical
    :data:`...output_schema.KEPT_COLUMNS` keys. ``_keep`` indicates whether
    the row survived all 6 phases; ``_drop_reason`` is set when ``_keep`` is
    False.
    """
    # ── 1) Adapter pre-check ──────────────────────────────────────────────
    pre = adapter.pre_check(ex)
    if pre is not None:
        return build_dropped_row(drop_reason=pre)

    # ── 2) Adapter extracts messages ──────────────────────────────────────
    messages = adapter.extract_messages(ex)

    # Build the audit context now so dropped rows still carry the chat
    # history (matches legacy script behaviour: dropped rows include
    # ``context_messages`` for inspection).
    context_messages = messages_to_context_messages(messages)

    # ── 3) Run the 6 phases ───────────────────────────────────────────────
    outcome = _run_phases(
        messages=messages,
        cfg=cfg,
        safety_flags_row=ex if cfg.enable_safety_flag_filter else None,
    )

    if outcome.drop_reason is not None:
        out = build_dropped_row(
            drop_reason=outcome.drop_reason,
            context_messages=context_messages,
            prompt_text_preview=(outcome.prompt_text or "")[
                : cfg.prompt_text_preview_chars
            ]
            if outcome.prompt_text
            else None,
            assistant_last_content=(outcome.assistant_content or "")[
                : cfg.assistant_content_preview_chars
            ]
            if outcome.assistant_content
            else None,
        )
        return out

    # ── 4) Kept row: ask the adapter for dataset-specific metadata ────────
    row_id = adapter.get_row_id(ex, idx)
    example_id = stable_sha256(f"{adapter.dataset_id}|{split}|{row_id}")

    subsource_raw = adapter.extract_subsource_raw(ex)
    language = adapter.infer_language(
        ex=ex, messages=messages, default_lang="en"
    )
    domain = adapter.infer_domain(
        subsource_raw=subsource_raw,
        prompt_text=outcome.prompt_text,
        reasoning_text=outcome.reasoning_text,
    )
    reasoning_type = adapter.infer_reasoning_type(
        domain=domain, reasoning_text=outcome.reasoning_text
    )
    extras = adapter.extract_extras(ex) or {}

    fields = KeptRowFields(
        dataset_id=adapter.dataset_id,
        dataset_version_date=adapter.dataset_version_date,
        example_id=example_id,
        row_id=str(row_id),
        source_dataset_id=extras.get("source_dataset_id"),
        subsource_raw=subsource_raw,
        language=language,
        translation_of_example_id=extras.get("translation_of_example_id"),
        domain=domain,
        reasoning_type=reasoning_type,
        license=adapter.license,
        used_by_model=adapter.used_by_model,
        context_messages=context_messages,
        reasoning_text=outcome.reasoning_text,
        response_text=outcome.response_text,
        ground_truth_present=str(extras.get("ground_truth_present", "no")),
        final_answer_text=extras.get("final_answer_text", outcome.final_answer_text),
        difficulty=extras.get("difficulty"),
        category=extras.get("category"),
    )
    return build_kept_row(fields)
