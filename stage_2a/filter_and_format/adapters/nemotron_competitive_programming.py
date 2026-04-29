from __future__ import annotations

"""Adapter for ``nvidia/Nemotron-Competitive-Programming-v1``.

Specifics
---------
- Pure code-task dataset (C++/Python competitive programming).
- Constant domain (``code``), language (``en``) and reasoning_type
  (``procedural``).
- The reasoning trace is stored on the assistant turn under
  ``reasoning_content`` (separate field, no ``<think>...</think>`` tags).
  The adapter wraps it the same way :mod:`...nemotron_math_v2` does.
- ``difficulty`` and ``question_id`` are surfaced through
  ``extract_extras``; ``question_id`` is also used as ``source_dataset_id``
  to preserve the link to the original problem.
- The legacy script flagged most drops as ``prompt_too_short`` because
  competitive-programming prompts often consist of just an algorithmic
  problem statement that is below the 200-char question filter — the new
  pipeline reproduces that automatically through the standard 2A.1 / 2A.3
  thresholds.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter


def _wrap_reasoning_into_think(messages: Any) -> List[Dict[str, str]]:
    """Same trick as nemotron_math_v2: splice ``reasoning_content`` into
    ``<think>...</think>`` on the last assistant turn so the generic Phase
    2A.2 extraction picks it up.
    """
    if not isinstance(messages, list):
        return []
    last_idx = -1
    for i, turn in enumerate(messages):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            last_idx = i

    out: List[Dict[str, str]] = []
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


class NemotronCompetitiveProgrammingAdapter(DatasetAdapter):
    """Adapter for ``nvidia/Nemotron-Competitive-Programming-v1``."""

    dataset_id: str = "nvidia/Nemotron-Competitive-Programming-v1"
    license: Optional[str] = "CC-BY-4.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    _DOMAIN: str = "code"
    _LANGUAGE: str = "en"
    _REASONING_TYPE: str = "procedural"

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _wrap_reasoning_into_think(ex.get("messages"))

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        uid = (ex.get("uuid") or "").strip()
        if uid:
            return uid
        qid = ex.get("question_id")
        if qid:
            return str(qid)
        return str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return ex.get("source") or self.dataset_id

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
        return self._REASONING_TYPE

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        used_in = ex.get("used_in")
        if isinstance(used_in, list):
            used_models_str = ", ".join(str(x) for x in used_in if x) or None
        else:
            used_models_str = used_in if used_in else None
        return {
            "ground_truth_present": "no",
            "difficulty": ex.get("difficulty"),
            "category": "competitive_programming",
            "source_dataset_id": (
                str(ex.get("question_id")) if ex.get("question_id") else None
            ),
            # Note: used_by_model is set per-row here (overrides the class-level value)
            # by surfacing it via extras. The pipeline does not currently honour
            # extras["used_by_model"] — left as adapter-level constant. If per-row
            # override is needed later, extend KeptRowFields.
            "_used_by_model_per_row": used_models_str,
        }
