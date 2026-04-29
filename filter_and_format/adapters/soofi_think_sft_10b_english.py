from __future__ import annotations

"""Adapter for the **English** subset of ``toroe/Soofi-Think-SFT-10B-multilingual``.

Soofi rows expose the following columns:

- ``messages``     — list of ``{role, content}``
- ``source``       — upstream subsource id
- ``dataset_name`` — human-readable name of the upstream dataset
- ``ds_uid``       — int64 stable id within the source
- ``language``     — string (``"english"`` for this adapter)

Domain is inferred via a keyword sweep over ``(dataset_name, source)``;
language is forced to ``"en"`` regardless of the ``language`` column.

Language filter note
--------------------
The FastText English LID filter (``enable_fasttext_english_filter``) is
**disabled** for Soofi English.  The original pipeline ran without it because
``fasttext-wheel`` is incompatible with NumPy ≥ 2.x on some servers, and
because the upstream ``language`` field already guarantees the content is
English — making automatic LID redundant.  The Chinese-ratio filter remains
active as a secondary safeguard.

For the multilingual splits use
:class:`...soofi_think_sft_10b_multilingual.SoofiThinkSft10bMultilingualAdapter`.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter
from ..config import FilterConfig


def _infer_domain(dataset_name: Optional[str], source: Optional[str]) -> Optional[str]:
    text = " ".join(str(x) for x in (dataset_name, source) if x).lower()
    if "math" in text:
        return "math"
    if "code" in text or "program" in text:
        return "code"
    if "science" in text or "stem" in text:
        return "science"
    return "general"


_REASONING_TYPE_BY_DOMAIN = {
    "math": "deductive",
    "science": "explanatory",
    "code": "procedural",
    "general": "inductive",
}


def _strip_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip()
        if not role:
            continue
        out.append({"role": role, "content": (m.get("content") or "").strip()})
    return out


class SoofiThinkSft10bEnglishAdapter(DatasetAdapter):
    """Adapter for the English split of ``toroe/Soofi-Think-SFT-10B-multilingual``."""

    dataset_id: str = "toroe/Soofi-Think-SFT-10B-multilingual"
    license: Optional[str] = "Apache-2.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    _LANGUAGE: str = "en"

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _strip_messages(ex.get("messages"))

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        uid = ex.get("ds_uid")
        if uid is not None:
            return str(uid)
        return str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        # Prefer the upstream HF id (``source``); fall back to ``dataset_name``.
        return ex.get("source") or ex.get("dataset_name")

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
        # ``subsource_raw`` here is ex["source"] | ex["dataset_name"]; that
        # captures the original signal.
        return _infer_domain(dataset_name=subsource_raw, source=subsource_raw)

    def infer_reasoning_type(
        self,
        *,
        domain: Optional[str],
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        if reasoning_text is None or not reasoning_text.strip():
            return None
        return _REASONING_TYPE_BY_DOMAIN.get((domain or "general").lower())

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ground_truth_present": "no",
            "source_dataset_id": ex.get("dataset_name"),
        }

    @classmethod
    def recommended_config(cls) -> FilterConfig:
        """FilterConfig used in SFT-Collection-v2 for this adapter.

        FastText English filter is **OFF** — upstream ``language`` field
        already guarantees English content, and fasttext-wheel has NumPy≥2
        incompatibilities.  Chinese-ratio filter stays ON.
        """
        cfg = FilterConfig()
        cfg.enable_fasttext_english_filter = False
        cfg.enable_mixed_language_filter = False
        cfg.enable_chinese_ratio_filter = True
        return cfg
