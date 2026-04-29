from __future__ import annotations

"""Phase 2A.1 — Structural filters.

Sankey-Stage label: ``2A.1_structural`` (messages / assistant presence, prompt length).

These checks operate on the raw chat-style ``messages`` list of an example and
catch rows that are unusable before any text-quality analysis. They are the
*first* gate of Stage 2A and were previously inlined in every per-dataset
``filter_and_format_*.py`` script — this module bundles the canonical version
so each new pipeline script gets the same behaviour for free.

Drop reasons produced by this module
------------------------------------
- ``bad_messages``       : ``messages`` is missing, not a list, or contains < 2 turns
- ``last_not_assistant`` : the final message is not an assistant turn
- ``prompt_too_short``   : the joined user/system prompt is shorter than ``min_prompt_chars``
- ``empty_assistant``    : the final assistant turn has empty content after stripping

The thresholds and field semantics match what the legacy
``filter_and_format_dolci_think_sft_7b.py`` pipeline used (and which was then
reused — with the same defaults — across the other 12 dataset scripts):

    min_prompt_chars = 30   # CLI default in dolci script
    min_prompt_chars = 50   # default of FilterCfg in dolci script
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ── prompt joining ───────────────────────────────────────────────────────────
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends.

    Mirrors the ``_normalize_ws`` helper used in the per-dataset scripts.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def messages_to_prompt_text(messages: Any) -> str:
    """Concatenate the content of all non-assistant turns into a single string.

    Mirrors the ``_messages_to_prompt_text`` helper used by the per-dataset
    scripts: only ``system`` / ``user`` (and any non-assistant) roles contribute,
    joined by a blank line. The final assistant turn is *not* included because
    it is the target the model is supposed to produce.
    """
    if not isinstance(messages, list):
        return ""
    parts: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n\n".join(parts)


# ── structural drop checks ───────────────────────────────────────────────────


@dataclass(frozen=True)
class StructuralCheckResult:
    """Outcome of running the four structural checks against a row.

    Attributes
    ----------
    drop_reason
        One of ``"bad_messages"``, ``"last_not_assistant"``, ``"prompt_too_short"``,
        ``"empty_assistant"`` if the row should be dropped, else ``None``.
    last_message
        The last message dict if the row passed ``bad_messages`` and
        ``last_not_assistant``, else ``None``. Useful so callers don't repeat
        the indexing.
    prompt_text
        The result of ``messages_to_prompt_text(messages)`` for downstream use
        (question filter, language detector, ...).
    assistant_content
        The stripped content of the final assistant turn, or ``""`` if the row
        was dropped before that point.
    """

    drop_reason: Optional[str]
    last_message: Optional[Dict[str, Any]]
    prompt_text: str
    assistant_content: str


def evaluate_structural(
    messages: Any,
    *,
    min_prompt_chars: int = 50,
    min_messages: int = 2,
) -> StructuralCheckResult:
    """Run the four Phase 2A.1 checks in their canonical order.

    The checks short-circuit: as soon as one fails, the remaining ones are
    skipped (matching the original per-script behaviour, which used a chain of
    early returns).

    Parameters
    ----------
    messages
        Raw ``messages`` field from the source row.
    min_prompt_chars
        Lower bound on ``len(normalize_whitespace(prompt_text))``. Default 50,
        matching the ``FilterCfg.min_prompt_chars`` default in the legacy
        per-dataset scripts. Many CLIs override this to 30.
    min_messages
        Minimum number of turns. Default 2 (one user, one assistant), matching
        the legacy scripts.
    """
    # 1) bad_messages
    if not isinstance(messages, list) or len(messages) < min_messages:
        return StructuralCheckResult("bad_messages", None, "", "")

    # 2) last_not_assistant
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return StructuralCheckResult("last_not_assistant", None, "", "")

    # 3) prompt_too_short
    prompt_text = messages_to_prompt_text(messages)
    prompt_text_norm = normalize_whitespace(prompt_text)
    if len(prompt_text_norm) < min_prompt_chars:
        return StructuralCheckResult("prompt_too_short", last, prompt_text, "")

    # 4) empty_assistant
    assistant_content = (last.get("content") or "").strip()
    if not assistant_content:
        return StructuralCheckResult("empty_assistant", last, prompt_text, "")

    return StructuralCheckResult(None, last, prompt_text, assistant_content)


# ── individual predicates (kept for callers that prefer atomic functions) ────


def is_bad_messages(messages: Any, *, min_messages: int = 2) -> bool:
    return not isinstance(messages, list) or len(messages) < min_messages


def is_last_not_assistant(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return True
    last = messages[-1]
    return not isinstance(last, dict) or last.get("role") != "assistant"


def is_prompt_too_short(messages: Any, *, min_prompt_chars: int = 50) -> bool:
    prompt = normalize_whitespace(messages_to_prompt_text(messages))
    return len(prompt) < min_prompt_chars


def is_empty_assistant(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return True
    last = messages[-1]
    if not isinstance(last, dict):
        return True
    return not (last.get("content") or "").strip()
