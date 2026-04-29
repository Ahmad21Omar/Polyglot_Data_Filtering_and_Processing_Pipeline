from __future__ import annotations

"""Shared text-assembly helper for Stage 2B quality scoring.

The English (FastText OH-ELI5) and multilingual (FineWeb-HQ + XLM-R) scorers
both consume the same kind of input: ``"<user turns> [SEP] <response_text>"``.
Putting that one helper here keeps both scorers identical in what they feed
the model, so a row's quality decision is purely a function of *which*
classifier ran, not of how the text was concatenated.

The legacy SFT-Collection-v2 scripts intentionally **omit** ``reasoning_text``
from the scoring text:

- it is often very long (>2000 chars) → fastText would truncate it anyway,
- it is mostly model "thinking", not user-facing content → not a great
  signal for quality classifiers trained on web text.

The user prompt + final response is therefore the canonical scoring text.
"""

import json
from typing import Any, Dict, List


SEPARATOR = " [SEP] "


def _normalise_context_messages(value: Any) -> List[Dict[str, Any]]:
    """Accept list, JSON string, numpy/pyarrow array, or None and return a list."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except Exception:
            return []
        return decoded if isinstance(decoded, list) else []
    try:
        return list(value)
    except Exception:
        return []


def build_scoring_text(row: Dict[str, Any]) -> str:
    """Build the text that gets scored by a Stage 2B quality classifier.

    Parameters
    ----------
    row
        A dict with at least ``context_messages`` (list of
        ``{role, content}``) and ``response_text``. Any other keys are
        ignored. The format mirrors the canonical SFT-Collection-v2 row
        schema (``output_schema.KEPT_COLUMNS``).

    Returns
    -------
    str
        ``"<user_1> [SEP] <user_2> [SEP] ... [SEP] <response>"`` — empty
        string if the row carries neither user turns nor a response.
    """
    parts: List[str] = []

    ctx = _normalise_context_messages(row.get("context_messages"))
    for msg in ctx:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if content:
            parts.append(content)

    response = str(row.get("response_text") or "").strip()
    if response:
        parts.append(response)

    return SEPARATOR.join(parts)
