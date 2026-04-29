from __future__ import annotations

"""Adapter for ``open-r1/Mixture-of-Thoughts``.

Mixture-of-Thoughts ships **three configs** (``math`` / ``code`` /
``science``) as separate snapshots. The domain is therefore not heuristic —
it equals the config name and is set authoritatively by the adapter.

Source rows have either ``messages`` (chat shape) or ``conversations``
(ShareGPT shape); the adapter normalises both.

Adapter is *parameterised* by ``config_name`` because each config produces a
different ``dataset_source`` label (``"open-r1/Mixture-of-Thoughts:math"``)
and a different forced domain.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter
from .dolci_think_sft_7b import _infer_reasoning_type


_FIXED_REASONING_TYPE = {
    "code": "procedural",
    "math": "deductive",
    "science": "explanatory",
}


def _normalize_messages(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    """Accept both Dolci-style ``messages`` and ShareGPT-style ``conversations``."""
    msgs = ex.get("messages")
    if isinstance(msgs, list) and all(isinstance(m, dict) for m in msgs):
        out: List[Dict[str, str]] = []
        for m in msgs:
            role = (m.get("role") or "").strip() or "user"
            content = (m.get("content") or "").strip()
            if not content:
                continue
            out.append({"role": role, "content": content})
        return out

    conv = ex.get("conversations")
    if isinstance(conv, list) and all(isinstance(m, dict) for m in conv):
        out = []
        for m in conv:
            role_raw = (m.get("from") or "").strip().lower()
            content = (m.get("value") or "").strip()
            if not content:
                continue
            if any(k in role_raw for k in ("assistant", "bot", "gpt", "model")):
                role = "assistant"
            elif any(k in role_raw for k in ("user", "human")):
                role = "user"
            else:
                role = role_raw if role_raw else "user"
            out.append({"role": role, "content": content})
        return out

    return []


class MixtureOfThoughtsAdapter(DatasetAdapter):
    """Adapter for one config of ``open-r1/Mixture-of-Thoughts``.

    Parameters
    ----------
    config_name
        One of ``"math"``, ``"code"``, ``"science"``. Becomes the canonical
        ``domain`` for every kept row and is appended to ``dataset_source``.
    """

    dataset_id: str = "open-r1/Mixture-of-Thoughts"
    license: Optional[str] = "Apache-2.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    def __init__(self, config_name: str):
        cfg = (config_name or "").strip().lower()
        if cfg not in _FIXED_REASONING_TYPE:
            raise ValueError(
                f"MixtureOfThoughtsAdapter: unsupported config_name={config_name!r}; "
                f"expected one of {sorted(_FIXED_REASONING_TYPE)}."
            )
        self.config_name = cfg

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return _normalize_messages(ex)

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        rid = ex.get("id") or ex.get("row_id") or ex.get("uuid")
        return str(rid) if rid else str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return f"{self.dataset_id}:{self.config_name}"

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        # Config name is authoritative.
        return self.config_name

    def infer_reasoning_type(
        self,
        *,
        domain: Optional[str],
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        if reasoning_text is None or not reasoning_text.strip():
            return None
        rt = _FIXED_REASONING_TYPE.get((domain or "").lower())
        if rt:
            return rt
        return _infer_reasoning_type(domain=domain, reasoning_text=reasoning_text)

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ground_truth_present": "no",
            "category": self.config_name,
        }
