from __future__ import annotations

"""Adapter for ``nvidia/Nemotron-Math-Proofs-v1``.

Specifics
---------
- Each row has ``messages`` (chat trace) plus a ``formal_statement`` field
  containing the Lean 4 theorem the model is supposed to prove. The Lean
  statement is preserved as ``final_answer_text`` and ``ground_truth_present``
  is ``"yes"`` for kept rows.
- Roughly 34% of source rows have an *empty* ``messages`` list — these are
  prompts without a verified proof attempt. The legacy script dropped them
  with a dedicated reason ``no_verified_proof`` (rather than the generic
  ``bad_messages``); this adapter preserves that distinction by short-
  circuiting in :meth:`pre_check`.
"""

from typing import Any, Dict, Optional

from ..adapter import DatasetAdapter


class NemotronMathProofsV1Adapter(DatasetAdapter):
    """Adapter for ``nvidia/Nemotron-Math-Proofs-v1``."""

    dataset_id: str = "nvidia/Nemotron-Math-Proofs-v1"
    license: Optional[str] = "CC-BY-4.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    # Domain / language are constant for this dataset.
    _DOMAIN: str = "math"
    _LANGUAGE: str = "en"

    def pre_check(self, ex: Dict[str, Any]) -> Optional[str]:
        # Rows with no verified proof attempt have empty messages. The
        # legacy script tags those with a dedicated reason rather than
        # letting them fall through into the generic `bad_messages` drop.
        msgs = ex.get("messages")
        if not isinstance(msgs, list) or len(msgs) == 0:
            return "no_verified_proof"
        return None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return ex.get("messages")

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        uuid_val = (ex.get("uuid") or "").strip()
        return uuid_val or str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return ex.get("source") or ex.get("data_source")

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
        # Lean proof attempts are deductive by definition.
        if reasoning_text is None or not reasoning_text.strip():
            return None
        return "deductive"

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        formal = ex.get("formal_statement")
        if isinstance(formal, str) and not formal.strip():
            formal = None
        return {
            "ground_truth_present": "yes" if formal else "no",
            "final_answer_text": formal,
            "category": "lean4_proof",
        }
