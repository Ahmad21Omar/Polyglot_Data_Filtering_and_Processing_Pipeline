from __future__ import annotations

"""Adapter for ``open-thoughts/OpenThoughts3-1.2M``.

OpenThoughts ships its chat data under a ``conversations`` column with
``{from, value}`` entries (ShareGPT-style) instead of the usual
``messages``. This adapter normalises that into a standard ``messages`` list
and uses OpenThoughts' ``domain`` / ``difficulty`` metadata to fill the
canonical schema fields.

Domain & reasoning-type inference reuses the dolci adapter helpers so the
behaviour matches what the published dataset already contains.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter
from .dolci_think_sft_7b import _infer_reasoning_type


# ── helpers (ported from legacy openthoughts3 script) ────────────────────────


def _conversations_to_messages(conv: Any) -> List[Dict[str, str]]:
    """Convert OpenThoughts ``conversations`` (ShareGPT-style) to ``messages``."""
    out: List[Dict[str, str]] = []
    if not isinstance(conv, list):
        return out
    for item in conv:
        if not isinstance(item, dict):
            continue
        role_raw = (item.get("from") or "").strip().lower()
        content = (item.get("value") or "").strip()
        if not content:
            continue
        if any(k in role_raw for k in ("assistant", "bot", "gpt", "model")):
            role = "assistant"
        elif any(k in role_raw for k in ("user", "human", "human:", "person")):
            role = "user"
        else:
            role = role_raw if role_raw else "user"
        out.append({"role": role, "content": content})
    return out


# ── adapter ──────────────────────────────────────────────────────────────────


class OpenThoughts3Adapter(DatasetAdapter):
    """Adapter for ``open-thoughts/OpenThoughts3-1.2M``."""

    dataset_id: str = "open-thoughts/OpenThoughts3-1.2M"
    license: Optional[str] = "Apache-2.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _conversations_to_messages(ex.get("conversations"))

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        # OpenThoughts has no stable row id; fall back to enumeration.
        return str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return ex.get("source") or "open-thoughts"

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        # OpenThoughts ships an explicit ``domain`` column on the source row;
        # because the adapter API only sees ``subsource_raw`` here, the
        # adapter overrides ``extract_extras`` instead to surface the raw
        # domain on every kept row, and we mirror it as the canonical domain.
        return None

    def infer_reasoning_type(
        self,
        *,
        domain: Optional[str],
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        return _infer_reasoning_type(domain=domain, reasoning_text=reasoning_text)

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        # Preserve OpenThoughts metadata under the canonical column names.
        return {
            "ground_truth_present": "no",
            "difficulty": ex.get("difficulty"),
            "category": ex.get("domain"),
        }
