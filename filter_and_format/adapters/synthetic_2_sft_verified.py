from __future__ import annotations

"""Adapter for ``PrimeIntellect/SYNTHETIC-2-SFT-verified``.

Specifics
---------
- ``task_type`` is the per-row label (``verifiable_math`` / ``prime_rl_code``
  / ``ifeval`` / ``reasoning_gym`` / ...). It maps directly to the canonical
  domain via :data:`_TASK_TYPE_TO_DOMAIN`; unknown task types fall back to
  the umbrella domain ``"general"``.
- All rows are English; reasoning_type defaults to ``"deductive"`` because
  the dataset contains verifiable reasoning trajectories.
- Per-row extras: ``problem_id``, ``task_type`` (also used as ``category``)
  and ``reward``.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter


_TASK_TYPE_TO_DOMAIN: Dict[str, str] = {
    "verifiable_math":        "math",
    "no_verification":        "math",
    "prime_rl_code":          "code",
    "code_output_prediction": "code",
    "ifeval":                 "general",
    "complex_json_output":    "general",
    "reasoning_gym":          "general",
    "unscramble_sentence":    "general",
    "pydantic_adherance":     "general",
    "ascii_tree_formatting":  "general",
}
_DEFAULT_DOMAIN = "general"


def _strip_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, str]] = []
    for t in messages:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or "").strip()
        if not role:
            continue
        out.append({"role": role, "content": (t.get("content") or "").strip()})
    return out


class Synthetic2SftVerifiedAdapter(DatasetAdapter):
    """Adapter for ``PrimeIntellect/SYNTHETIC-2-SFT-verified``."""

    dataset_id: str = "PrimeIntellect/SYNTHETIC-2-SFT-verified"
    license: Optional[str] = "Apache-2.0"
    used_by_model: Optional[str] = "deepseek-ai/DeepSeek-R1"
    dataset_version_date: Optional[str] = None

    _LANGUAGE: str = "en"

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _strip_messages(ex.get("messages"))

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        pid = (ex.get("problem_id") or "").strip()
        return pid or str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        tt = ex.get("task_type")
        return tt if isinstance(tt, str) and tt.strip() else None

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
        if not subsource_raw:
            return _DEFAULT_DOMAIN
        return _TASK_TYPE_TO_DOMAIN.get(
            subsource_raw.strip().lower(), _DEFAULT_DOMAIN
        )

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
        return {
            "ground_truth_present": "no",
            "category": ex.get("task_type"),
        }
