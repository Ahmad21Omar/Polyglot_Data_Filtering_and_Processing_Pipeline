from __future__ import annotations

"""Adapter for ``nvidia/Nemotron-Math-v2``.

Specifics
---------
- Unlike most other Nemotron datasets, math-v2 stores the reasoning trace in
  a **separate field** ``reasoning_content`` on the assistant turn — there
  are no ``<think>...</think>`` tags. The adapter wraps the raw
  ``reasoning_content`` into ``<think>...</think>`` before the generic
  pipeline runs, so the standard Phase 2A.2 extraction code can use it.
- Rows with no ``reasoning_content`` (or with whitespace-only content) are
  dropped early with a dedicated reason ``empty_reasoning_content`` instead
  of falling through to ``missing_think``.
- ``expected_answer`` is the verified ground-truth final answer.
- All splits are math, English-only.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter


def _get_last_assistant_turn(messages: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(messages, list):
        return None
    last = None
    for turn in messages:
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            last = turn
    return last


def _wrap_reasoning_into_think(messages: Any) -> List[Dict[str, str]]:
    """Splice the assistant's ``reasoning_content`` into ``<think>...</think>``.

    The generic pipeline's Phase 2A.2 code (``extract_think_and_answer``)
    expects an inline ``<think>`` block on the assistant content. Math-v2
    stores the reasoning out-of-band, so we splice it back in here.

    Each non-assistant turn is reduced to ``{role, content}``; the last
    assistant turn becomes ``{role: "assistant", content: "<think>{R}</think>{C}"}``
    where ``R`` is the assistant's ``reasoning_content`` and ``C`` is the
    final solution string.
    """
    if not isinstance(messages, list):
        return []

    out: List[Dict[str, str]] = []
    last_idx = -1
    for i, turn in enumerate(messages):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            last_idx = i

    for i, turn in enumerate(messages):
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or "").strip()
        content = (turn.get("content") or "").strip()
        if i == last_idx:
            reasoning = (turn.get("reasoning_content") or "").strip()
            if reasoning:
                content = f"<think>{reasoning}</think>\n\n{content}".strip()
        if role:
            out.append({"role": role, "content": content})
    return out


class NemotronMathV2Adapter(DatasetAdapter):
    """Adapter for ``nvidia/Nemotron-Math-v2``."""

    dataset_id: str = "nvidia/Nemotron-Math-v2"
    license: Optional[str] = "CC-BY-4.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    _DOMAIN: str = "math"
    _LANGUAGE: str = "en"

    def pre_check(self, ex: Dict[str, Any]) -> Optional[str]:
        # Drop rows whose assistant has no reasoning_content; the generic
        # `missing_think` would catch these too, but math-v2 historically
        # tagged them with a more specific reason.
        last = _get_last_assistant_turn(ex.get("messages"))
        if last is None:
            return None  # let the structural phase emit `last_not_assistant`
        if not (last.get("reasoning_content") or "").strip():
            return "empty_reasoning_content"
        return None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _wrap_reasoning_into_think(ex.get("messages"))

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        uid = (ex.get("uuid") or "").strip()
        return uid or str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return ex.get("data_source") or self.dataset_id

    def infer_language(
        self,
        *,
        ex: Dict[str, Any],
        messages: Any,
        default_lang: str = "en",
    ) -> str:
        return self._LANGUAGE

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        return self._DOMAIN

    def infer_reasoning_type(
        self,
        *,
        domain: Optional[str],
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        if reasoning_text is None or not reasoning_text.strip():
            return None
        return "deductive"

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        ans = ex.get("expected_answer")
        return {
            "ground_truth_present": "yes" if ans else "no",
            "final_answer_text": ans or None,
            "category": "math",
        }
