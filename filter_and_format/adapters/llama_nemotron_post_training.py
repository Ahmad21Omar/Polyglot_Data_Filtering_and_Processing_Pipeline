from __future__ import annotations

"""Adapter for ``nvidia/Llama-Nemotron-Post-Training-Dataset``.

Specifics
---------
- Source rows store the conversation in **three separate fields**:
  ``system_prompt`` (string), ``input`` (list of ``{role, content}`` turns) and
  ``output`` (assistant string). The adapter assembles these into a single
  ``messages`` list before the pipeline sees the row.
- An explicit ``category`` field on every row (``math`` / ``code`` /
  ``science`` / ``chat`` / ``safety``) maps directly to our domain taxonomy.
- The dataset is English-only across all 5 task splits.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter
from .dolci_think_sft_7b import _infer_reasoning_type


_CATEGORY_TO_DOMAIN: Dict[str, str] = {
    "math":    "math",
    "code":    "code",
    "science": "science",
    "chat":    "chat",
    "safety":  "safety",
}


def _build_messages(
    system_prompt: Optional[str],
    input_field: Any,
    assistant_output: str,
) -> List[Dict[str, str]]:
    """Assemble (system / input-turns / output) into a chat ``messages`` list."""
    msgs: List[Dict[str, str]] = []

    if isinstance(system_prompt, str) and system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt.strip()})

    if isinstance(input_field, list):
        for turn in input_field:
            if not isinstance(turn, dict):
                continue
            role = (turn.get("role") or "user").strip()
            content = (turn.get("content") or "").strip()
            if content:
                msgs.append({"role": role, "content": content})
    elif isinstance(input_field, str) and input_field.strip():
        msgs.append({"role": "user", "content": input_field.strip()})

    msgs.append({"role": "assistant", "content": (assistant_output or "").strip()})
    return msgs


class LlamaNemotronPostTrainingAdapter(DatasetAdapter):
    """Adapter for ``nvidia/Llama-Nemotron-Post-Training-Dataset``."""

    dataset_id: str = "nvidia/Llama-Nemotron-Post-Training-Dataset"
    license: Optional[str] = "CC-BY-4.0"
    used_by_model: Optional[str] = "nvidia/Llama-3.1-Nemotron"
    dataset_version_date: Optional[str] = None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _build_messages(
            ex.get("system_prompt"),
            ex.get("input"),
            ex.get("output") or "",
        )

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        # Source rows have no stable id; the legacy pipeline computed a
        # content-hash. Here we let the runner derive example_id from the
        # adapter's row_id (idx-based), and the content hash is implicit in
        # example_id = sha256(dataset_id|split|row_id) — stable for a fixed
        # row order and split name, which is enough for downstream dedup.
        return str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        cat = ex.get("category")
        return f"{self.dataset_id}:{cat}" if cat else self.dataset_id

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        # The category field is read by extract_extras from the original row,
        # but we don't have that here. We re-derive from subsource_raw.
        if subsource_raw and ":" in subsource_raw:
            cat = subsource_raw.split(":", 1)[1].strip().lower()
            return _CATEGORY_TO_DOMAIN.get(cat)
        return None

    def infer_reasoning_type(
        self,
        *,
        domain: Optional[str],
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        return _infer_reasoning_type(domain=domain, reasoning_text=reasoning_text)

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ground_truth_present": "no",
            "category": ex.get("category"),
        }
