from __future__ import annotations

"""Adapter for the **multilingual** subsets of ``toroe/Soofi-Think-SFT-10B-multilingual``.

Soofi multilingual rows expose the same columns as the English subset, but
the ``language`` field carries the human-readable language name
(``"german"``, ``"french"``, ``"italian"``, ``"spanish"``, ``"japanese"``,
...). The adapter converts that to an ISO 639-1 code.

Pre-check: rows without a ``language`` value are dropped with
``missing_language`` (matches the legacy script behaviour).
"""

from typing import Any, Dict, Optional

from ..adapter import DatasetAdapter
from .soofi_think_sft_10b_english import (
    _REASONING_TYPE_BY_DOMAIN,
    _infer_domain,
    _strip_messages,
)


_LANGUAGE_NAME_TO_ISO: Dict[str, str] = {
    "english":   "en",
    "german":    "de",
    "deutsch":   "de",
    "french":    "fr",
    "francais":  "fr",
    "français":  "fr",
    "italian":   "it",
    "italiano":  "it",
    "spanish":   "es",
    "espanol":   "es",
    "español":   "es",
    "japanese":  "ja",
}


def _normalize_language(value: Any, default: str = "en") -> str:
    if not isinstance(value, str):
        return default
    s = value.strip().lower()
    if not s:
        return default
    if len(s) == 2:
        return s  # already an ISO code
    return _LANGUAGE_NAME_TO_ISO.get(s, default)


class SoofiThinkSft10bMultilingualAdapter(DatasetAdapter):
    """Adapter for non-English splits of ``toroe/Soofi-Think-SFT-10B-multilingual``."""

    dataset_id: str = "toroe/Soofi-Think-SFT-10B-multilingual"
    license: Optional[str] = "Apache-2.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    def pre_check(self, ex: Dict[str, Any]) -> Optional[str]:
        # The multilingual splits MUST carry a language label; rows without
        # one cannot be assigned to a target language and were historically
        # dropped under ``missing_language``.
        lang_raw = ex.get("language")
        if not (isinstance(lang_raw, str) and lang_raw.strip()):
            return "missing_language"
        return None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _strip_messages(ex.get("messages"))

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        uid = ex.get("ds_uid")
        if uid is not None:
            return str(uid)
        return str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return ex.get("source") or ex.get("dataset_name")

    def infer_language(
        self,
        *,
        ex: Dict[str, Any],
        messages: Any,
        default_lang: str = "en",
    ) -> str:
        return _normalize_language(ex.get("language"), default=default_lang)

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
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
        dataset_name = ex.get("dataset_name")
        ds_uid = ex.get("ds_uid")
        sdid = (
            f"{dataset_name}/{ds_uid}"
            if dataset_name and ds_uid is not None
            else dataset_name
        )
        return {
            "ground_truth_present": "no",
            "source_dataset_id": sdid,
        }
