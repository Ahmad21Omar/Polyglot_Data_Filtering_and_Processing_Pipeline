from __future__ import annotations

"""Adapter for ``allenai/Dolci-Think-SFT-7B``.

Maps each Dolci row (``{messages, dataset_source, id}``) to the canonical
SFT-Collection-v2 schema. The domain / language / reasoning-type lookup
tables are ported verbatim from the legacy
``Master_Thesis/Data_pipline/sft/filter_and_format_dolci_think_sft_7b.py``
script (functions ``_infer_domain_from_subsource``,
``_infer_language_from_subsource``, ``_infer_reasoning_type``).
"""

from typing import Any, Dict, Optional

from ..adapter import DatasetAdapter
from ...filters.phase_2a1_structural import normalize_whitespace


# ── helpers (ported from legacy dolci script) ────────────────────────────────


def _infer_domain_from_subsource(subsource_raw: Optional[str]) -> Optional[str]:
    """Conservative domain inference from the Dolci ``dataset_source`` label."""
    if not subsource_raw:
        return None
    s = subsource_raw.lower()

    # OpenThoughts3 sub-mixes
    if "open-thoughts/openthoughts3" in s and "math" in s:
        return "math"
    if "open-thoughts/openthoughts3" in s and "stem" in s:
        return "science"
    if "open-thoughts/openthoughts3" in s and "code" in s:
        return "code"

    # Nemotron code splits
    if "nemotron" in s and "code" in s:
        return "code"

    # Code-flavoured sources
    if "python algorithms" in s or "algorithms" in s or "leetcode" in s:
        return "code"
    if "tablegpt" in s:
        return "code"

    # Math-ish synthetic mixes
    if "math" in s:
        return "math"

    # STEM => science
    if "stem" in s:
        return "science"

    # Chat / safety-ish mixes
    if "wildchat" in s or "openassistant" in s or "guanaco" in s:
        return "chat"
    if "wildjailbreak" in s or "wildguard" in s:
        return "chat"

    # IF-style sources
    if "precise if" in s or "persona" in s:
        return "instruction_following"

    # Safety / identity prompt sources
    if "identity" in s:
        return "chat"

    return None


def _infer_language_from_subsource(
    subsource_raw: Optional[str],
    default_language: str = "en",
) -> Optional[str]:
    """Conservative language inference. Dolci is predominantly English; only
    override when the subsource label makes a non-English origin obvious.
    """
    if not subsource_raw:
        return default_language
    s = subsource_raw.lower()
    if "aya" in s and "multilingual" in s:
        return None
    return default_language


def _infer_reasoning_type(
    *,
    domain: Optional[str],
    reasoning_text: Optional[str],
) -> Optional[str]:
    """Conservative reasoning-type inference based on domain."""
    if reasoning_text is None or not normalize_whitespace(reasoning_text):
        return None
    if domain == "code":
        return "procedural"
    if domain == "math":
        return "deductive"
    if domain == "science":
        return "explanatory"
    if domain == "chat":
        return "dialogue-based"
    return None


# ── adapter ──────────────────────────────────────────────────────────────────


class DolciThinkSft7BAdapter(DatasetAdapter):
    """Adapter for ``allenai/Dolci-Think-SFT-7B``."""

    dataset_id: str = "allenai/Dolci-Think-SFT-7B"
    license: Optional[str] = "ODC-BY-1.0"
    used_by_model: Optional[str] = "allenai/OLMo-3-Think"
    dataset_version_date: Optional[str] = None  # set by caller

    def pre_check(self, ex: Dict[str, Any]) -> Optional[str]:
        # Dolci rows must carry a dataset_source label; rows without one
        # cannot be attributed to a sub-mixture and are dropped early.
        if ex.get("dataset_source") is None:
            return "missing_dataset_source"
        return None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return ex.get("messages")

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        return str(ex.get("id") or idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return ex.get("dataset_source")

    def infer_language(
        self,
        *,
        ex: Dict[str, Any],
        messages: Any,
        default_lang: str = "en",
    ) -> str:
        sub = ex.get("dataset_source")
        lang = _infer_language_from_subsource(sub, default_language=default_lang)
        return lang or default_lang

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        return _infer_domain_from_subsource(subsource_raw)

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
            "final_answer_text": ex.get("final_answer_text"),
        }
