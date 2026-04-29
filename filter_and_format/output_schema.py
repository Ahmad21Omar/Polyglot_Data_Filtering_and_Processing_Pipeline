from __future__ import annotations

"""Unified row schema for SFT-Collection-v2.

Every adapter — regardless of source dataset — produces rows with this exact
schema. The order of the fields is the canonical order used by all downstream
stages (Stage 2B quality filter, Stage 3 dedup, Stage 4 post-processing) and
by the published HuggingFace dataset.

The schema mirrors the ``empty_out`` dict that was duplicated in every
``filter_and_format_*.py`` script (see e.g. the dolci script lines 276-305),
so existing downstream consumers continue to work without changes.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── canonical column order (used for ``ds.select_columns`` / Parquet) ────────
KEPT_COLUMNS: List[str] = [
    # bookkeeping
    "_keep",
    "_drop_reason",
    # shared identifiers
    "dataset_id",
    "dataset_version_date",
    "example_id",
    "row_id",
    "source_dataset_id",
    "subsource_raw",
    "language",
    "translation_of_example_id",
    "domain",
    "reasoning_type",
    "license",
    "used_by_model",
    # SFT payload
    "context_messages",
    "reasoning_text",
    "response_text",
    "ground_truth_present",
    "final_answer_text",
    # audit-only (None for kept rows, populated for dropped rows)
    "prompt_text_preview",
    "assistant_last_content",
    # optional metadata
    "difficulty",
    "category",
]


def empty_row() -> Dict[str, Any]:
    """Return a fresh dict with every canonical column set to its null default.

    Mirrors ``empty_out`` from the legacy per-dataset scripts. ``_keep`` is
    ``False`` and every payload field is ``None``; the row is *only* kept if
    a later step explicitly sets ``_keep = True`` and ``_drop_reason = None``.
    """
    return {
        "_keep": False,
        "_drop_reason": None,
        "dataset_id": None,
        "dataset_version_date": None,
        "example_id": None,
        "row_id": None,
        "source_dataset_id": None,
        "subsource_raw": None,
        "language": None,
        "translation_of_example_id": None,
        "domain": None,
        "reasoning_type": None,
        "license": None,
        "used_by_model": None,
        "context_messages": None,
        "reasoning_text": None,
        "response_text": None,
        "ground_truth_present": None,
        "final_answer_text": None,
        "prompt_text_preview": None,
        "assistant_last_content": None,
        "difficulty": None,
        "category": None,
    }


# ── helpers ──────────────────────────────────────────────────────────────────


def stable_sha256(text: str) -> str:
    """Stable hex digest used to derive ``example_id`` from
    ``f"{dataset_id}|{split}|{row_id}"`` so the IDs survive re-shuffles.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages_to_context_messages(messages: Any) -> List[Dict[str, Any]]:
    """Strip the source ``messages`` list down to ``role`` + ``content`` only.

    The chat history *without* the final assistant turn is what downstream
    training actually consumes as the prompt. The final assistant turn is
    represented separately via ``reasoning_text`` + ``response_text``.

    Mirrors ``_messages_to_context_messages`` from the legacy scripts: keeps
    every turn *except* the very last assistant turn, and reduces each kept
    turn to a 2-key ``{"role", "content"}`` dict so downstream Parquet
    schemas are deterministic.
    """
    if not isinstance(messages, list) or not messages:
        return []
    # The final assistant turn is the *target* — exclude it from context.
    if (
        isinstance(messages[-1], dict)
        and messages[-1].get("role") == "assistant"
    ):
        history = messages[:-1]
    else:
        history = messages
    out: List[Dict[str, Any]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            continue
        out.append({"role": str(role), "content": str(content)})
    return out


# ── schema-builder used by the pipeline on a kept row ────────────────────────


@dataclass
class KeptRowFields:
    """Bundle of fields that the pipeline + adapter together compute for a kept row.

    The pipeline computes ``reasoning_text``, ``response_text``,
    ``context_messages`` and ``example_id`` itself; the adapter contributes
    everything that depends on the source dataset (dataset_id, license,
    domain, language, ...). ``build_kept_row`` merges them onto the canonical
    schema.
    """

    dataset_id: str
    dataset_version_date: Optional[str]
    example_id: str
    row_id: str
    source_dataset_id: Optional[str]
    subsource_raw: Optional[str]
    language: str
    translation_of_example_id: Optional[str]
    domain: Optional[str]
    reasoning_type: Optional[str]
    license: Optional[str]
    used_by_model: Optional[str]
    context_messages: List[Dict[str, Any]]
    reasoning_text: Optional[str]
    response_text: Optional[str]
    ground_truth_present: str  # "yes" / "no" / "unknown"
    final_answer_text: Optional[str]
    difficulty: Optional[str]
    category: Optional[str]


def build_kept_row(fields: KeptRowFields) -> Dict[str, Any]:
    """Build a kept row in canonical-column order.

    The audit fields ``prompt_text_preview`` and ``assistant_last_content``
    are *not* populated for kept rows (they are only useful when inspecting
    dropped rows).
    """
    out = empty_row()
    out.update(
        {
            "_keep": True,
            "_drop_reason": None,
            "dataset_id": fields.dataset_id,
            "dataset_version_date": fields.dataset_version_date,
            "example_id": fields.example_id,
            "row_id": fields.row_id,
            "source_dataset_id": fields.source_dataset_id,
            "subsource_raw": fields.subsource_raw,
            "language": fields.language,
            "translation_of_example_id": fields.translation_of_example_id,
            "domain": fields.domain,
            "reasoning_type": fields.reasoning_type,
            "license": fields.license,
            "used_by_model": fields.used_by_model,
            "context_messages": fields.context_messages,
            "reasoning_text": fields.reasoning_text,
            "response_text": fields.response_text,
            "ground_truth_present": fields.ground_truth_present,
            "final_answer_text": fields.final_answer_text,
            "difficulty": fields.difficulty,
            "category": fields.category,
        }
    )
    return out


def build_dropped_row(
    *,
    drop_reason: str,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    prompt_text_preview: Optional[str] = None,
    assistant_last_content: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an audit-only dropped row.

    The schema is the same as a kept row (so kept + dropped can share a
    Parquet writer); only the bookkeeping + the two ``*_preview`` audit
    fields are populated.
    """
    out = empty_row()
    out["_keep"] = False
    out["_drop_reason"] = drop_reason
    if context_messages is not None:
        out["context_messages"] = context_messages
    if prompt_text_preview is not None:
        out["prompt_text_preview"] = prompt_text_preview
    if assistant_last_content is not None:
        out["assistant_last_content"] = assistant_last_content
    return out
