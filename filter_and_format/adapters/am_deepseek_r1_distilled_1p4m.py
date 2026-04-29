from __future__ import annotations

"""Adapter for ``a-m-team/AM-DeepSeek-R1-Distilled-1.4M``.

Specifics
---------
- AM messages carry an extra ``info`` dict on each turn with provenance
  (``source``, ``reference_answer``, ``test_case``) and pre-extracted
  ``think_content`` / ``answer_content`` on the assistant turn. The adapter
  uses those when present and falls back to inline ``<think>``/``<answer>``
  tag parsing otherwise.
- Domain inference uses a multi-tier strategy:
    1. exact instruction-source lookup (``kodcode`` → ``code``,
       ``numinamath_1.5`` → ``math``, ...)
    2. substring heuristic (anything containing ``"math"`` → ``math``)
    3. fallback by prompt-text signals (``def``, ``class``, ``integral``,
       ``equation``, ...) → finally ``reasoning``.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..adapter import DatasetAdapter


def _normalize_messages(messages: Any) -> List[Dict[str, str]]:
    """Drop the AM-specific ``info`` key, keep ``{role, content}``."""
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip() or "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _splice_pre_extracted_think_answer(messages: Any) -> List[Dict[str, str]]:
    """If the assistant turn has pre-extracted ``info.think_content`` /
    ``info.answer_content`` but no inline ``<think>``/``<answer>`` tags,
    splice them inline so the generic Phase 2A.2 picks them up.
    """
    if not isinstance(messages, list):
        return []

    last_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_idx = i

    out: List[Dict[str, str]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip() or "user"
        content = (m.get("content") or "").strip()
        if not content and i != last_idx:
            continue
        if i == last_idx:
            info = m.get("info") if isinstance(m.get("info"), dict) else {}
            tc = info.get("think_content") if isinstance(info, dict) else None
            ac = info.get("answer_content") if isinstance(info, dict) else None
            tc = tc.strip() if isinstance(tc, str) and tc.strip() else None
            ac = ac.strip() if isinstance(ac, str) and ac.strip() else None
            # Only splice if the content doesn't already contain <think>.
            if "<think>" not in content.lower() and tc:
                pieces = [f"<think>{tc}</think>"]
                if ac:
                    pieces.append(ac)
                else:
                    pieces.append(content)
                content = "\n\n".join(pieces).strip()
        if role:
            out.append({"role": role, "content": content})
    return out


def _extract_ground_truth(
    messages: Any,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(reference_answer, test_case_json, instruction_source)``."""
    ref: Optional[str] = None
    test_case: Optional[str] = None
    instr_source: Optional[str] = None
    if not isinstance(messages, list):
        return None, None, None
    for m in messages:
        if not isinstance(m, dict):
            continue
        info = m.get("info")
        if not isinstance(info, dict):
            continue
        if instr_source is None:
            s = info.get("source")
            if isinstance(s, str) and s.strip():
                instr_source = s.strip()
        if ref is None:
            ra = info.get("reference_answer")
            if isinstance(ra, str) and ra.strip():
                ref = ra.strip()
        if test_case is None:
            tc = info.get("test_case")
            if isinstance(tc, str) and tc.strip():
                test_case = tc
            elif tc is not None:
                try:
                    test_case = json.dumps(tc, ensure_ascii=False)
                except Exception:
                    test_case = str(tc)
        if ref and test_case and instr_source:
            break
    return ref, test_case, instr_source


_MATH_EXACT = {
    "numinamath_1.5",
    "metamathqa",
    "openr1math_default",
    "openr1math_extended",
    "omni-math",
    "aime",
}
_REASONING_EXACT = {
    "natural_reasoning",
    "infinityinstruct",
    "generalthought - feb25",
    "generalthought-feb25",
    "openthoughts",
    "dolphin - r1",
    "dolphin-r1",
    "data_ablation_full59k",
    "bespoke17k",
    "am-0309",
}


def _infer_domain_from_instruction_source(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lower()
    if s in {"kodcode", "codeio", "opencoder"}:
        return "code"
    if s in _MATH_EXACT:
        return "math"
    if "math" in s:
        return "math"
    if s in _REASONING_EXACT:
        return "reasoning"
    return None


def _infer_domain_from_prompt(prompt: str) -> Optional[str]:
    if not isinstance(prompt, str):
        return None
    p = prompt.lower()
    if any(k in p for k in ("def ", "class ", "import ", "#include", "public static", "console.log", "function ")):
        return "code"
    if any(k in p for k in ("leetcode", "complexity", "big-o", "time complexity", "runtime", "compile", "debug")):
        return "code"
    if any(k in p for k in ("integral", "derivative", "solve for", "∫", "√", "π", "equation", "matrix", "proof")):
        return "math"
    return "reasoning"


class AmDeepSeekR1DistilledAdapter(DatasetAdapter):
    """Adapter for ``a-m-team/AM-DeepSeek-R1-Distilled-1.4M``."""

    dataset_id: str = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
    license: Optional[str] = "Apache-2.0"
    used_by_model: Optional[str] = "deepseek-ai/DeepSeek-R1"
    dataset_version_date: Optional[str] = None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        # Try the pre-extracted think/answer first; if absent, fall back to plain normalisation.
        spliced = _splice_pre_extracted_think_answer(ex.get("messages"))
        return spliced if spliced else _normalize_messages(ex.get("messages"))

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        rid = ex.get("id") or ex.get("row_id") or ex.get("uuid")
        return str(rid) if rid else str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        _, _, instr = _extract_ground_truth(ex.get("messages"))
        return instr

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        d = _infer_domain_from_instruction_source(subsource_raw)
        if d is not None:
            return d
        return _infer_domain_from_prompt(prompt_text)

    def infer_reasoning_type(
        self,
        *,
        domain: Optional[str],
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        if reasoning_text is None or not reasoning_text.strip():
            return None
        return {
            "math": "deductive",
            "code": "procedural",
            "science": "explanatory",
            "chat": "dialogue-based",
            "reasoning": "deductive",
        }.get((domain or "").lower())

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        ref, _tc, instr = _extract_ground_truth(ex.get("messages"))
        return {
            "ground_truth_present": "yes" if ref else "no",
            "final_answer_text": ref,
            "category": instr,
        }
