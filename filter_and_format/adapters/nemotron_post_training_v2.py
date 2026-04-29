from __future__ import annotations

"""Adapter for ``nvidia/Nemotron-Post-Training-Dataset-v2``.

Dataset structure
-----------------
Shipped as **9 splits**: ``math`` / ``code`` / ``stem`` / ``chat`` /
``multilingual_de`` / ``multilingual_es`` / ``multilingual_fr`` /
``multilingual_it`` / ``multilingual_ja``.

**Only the 5 multilingual splits are used** in SFT-Collection-v2.
The English splits (math / code / stem / chat) all have ``reasoning="off"``
and therefore contain **zero CoT traces** — they are incompatible with the
reasoning-trace SFT objective and are excluded entirely.

Language & reasoning layout
---------------------------
- Multilingual splits (de / es / fr / it / ja):
    * **Prompt** is in the target language (German, French, ...).
    * **Reasoning trace** (``<think>…</think>``) is **always in English**,
      even when the prompt is not. This is how NVIDIA generated the data.
    * Language is assigned directly from the split name — no FastText
      detection is needed or wanted.
- English splits (math / code / stem / chat):
    * Prompt and response are both English.
    * No CoT trace → not used.

Per-split FilterConfig recommendations
---------------------------------------
Use ``recommended_config(split_name)`` to get the right ``FilterConfig``
for each split. The key per-split decisions are:

+-------------------+-------------------+-------------------+-------------------+
| split             | FastText English  | mixed-language    | why               |
|                   | filter            | filter            |                   |
+===================+===================+===================+===================+
| math              | **OFF**           | OFF               | LaTeX tokens      |
|                   |                   |                   | (\int, \frac,     |
|                   |                   |                   | \boxed) cause     |
|                   |                   |                   | ~63% false-drop   |
|                   |                   |                   | rate; language    |
|                   |                   |                   | guaranteed EN via |
|                   |                   |                   | split name.       |
+-------------------+-------------------+-------------------+-------------------+
| code / chat / stem| ON                | ON                | Freeform EN text, |
|                   |                   |                   | filter is reliable|
+-------------------+-------------------+-------------------+-------------------+
| multilingual_*    | **OFF**           | **OFF**           | Prompts are       |
|                   |                   |                   | intentionally non-|
|                   |                   |                   | English; reasoning|
|                   |                   |                   | in English is     |
|                   |                   |                   | expected.         |
+-------------------+-------------------+-------------------+-------------------+

Domain inference
----------------
- Non-multilingual splits: domain comes directly from the split name
  (``_CATEGORY_TO_DOMAIN``).
- Multilingual splits: domain is inferred via English keyword matching on
  ``reasoning_text``, which is always English (reliable signal). Falls back
  to ``"multilingual"`` when the keyword match is ambiguous.
"""

from typing import Any, Dict, List, Optional

from ..adapter import DatasetAdapter
from ..config import FilterConfig


_SPLIT_TO_LANGUAGE: Dict[str, str] = {
    "math":            "en",
    "code":            "en",
    "stem":            "en",
    "chat":            "en",
    "multilingual_de": "de",
    "multilingual_es": "es",
    "multilingual_fr": "fr",
    "multilingual_it": "it",
    "multilingual_ja": "ja",
}

_CATEGORY_TO_DOMAIN: Dict[str, str] = {
    "math":            "math",
    "code":            "code",
    "stem":            "science",
    "chat":            "chat",
    "multilingual_de": "multilingual",
    "multilingual_es": "multilingual",
    "multilingual_fr": "multilingual",
    "multilingual_it": "multilingual",
    "multilingual_ja": "multilingual",
}


# ── domain-from-reasoning fallback (multilingual splits only) ────────────────
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "code": [
        "def ", "function", "algorithm", "implement", "class ", "array",
        "string", "loop", "recursion", "complexity", "data structure",
        "binary", "sort", "graph ", "stack", "queue", "sql", "database",
        "api", "python", "javascript", "program",
    ],
    "math": [
        "equation", "theorem", "prove", "proof", "integer", "prime",
        "polynomial", "matrix", "vector", "calculus", "derivative",
        "integral", "sequence", "function f", "modulo", "combinat",
        "triangle", "angle", "arithmetic", "geometric", "algebra",
        "digit", "divisib", "modular",
    ],
    "science": [
        "chemistry", "physics", "biology", "reaction", "molecule", "atom",
        "force ", "energy", "velocity", "acceleration", "electric",
        "magnetic", "cell ", "protein", "dna", "enzyme", "compound",
        "concentration", "diffusion", "pressure", "temperature",
        "coefficient", "gene", "nucleus", "electron", "quantum",
    ],
}


def _infer_domain_from_reasoning(reasoning_text: str) -> Optional[str]:
    """Best-of-keywords disambiguation; returns None if the winner isn't clear."""
    if not reasoning_text:
        return None
    rt = reasoning_text.lower()
    scores: Dict[str, int] = {
        dom: sum(1 for kw in kws if kw in rt) for dom, kws in _DOMAIN_KEYWORDS.items()
    }
    best_dom, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score == 0:
        return None
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 2:
        return None
    return best_dom


class NemotronPostTrainingV2Adapter(DatasetAdapter):
    """Adapter for one split of ``nvidia/Nemotron-Post-Training-Dataset-v2``."""

    dataset_id: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    license: Optional[str] = "CC-BY-4.0"
    used_by_model: Optional[str] = None
    dataset_version_date: Optional[str] = None

    def __init__(self, split_name: str):
        s = (split_name or "").strip().lower()
        if s not in _SPLIT_TO_LANGUAGE:
            raise ValueError(
                f"NemotronPostTrainingV2Adapter: unsupported split={split_name!r}; "
                f"expected one of {sorted(_SPLIT_TO_LANGUAGE)}."
            )
        self.split_name = s
        self._domain = _CATEGORY_TO_DOMAIN[s]
        self._language = _SPLIT_TO_LANGUAGE[s]

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        return ex.get("messages")

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        uid = (ex.get("uuid") or "").strip()
        return uid or str(idx)

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        return f"{self.dataset_id}:{self.split_name}"

    def infer_language(
        self,
        *,
        ex: Dict[str, Any],
        messages: Any,
        default_lang: str = "en",
    ) -> str:
        return self._language

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        # For non-multilingual splits the split name is already the canonical domain.
        if self._domain != "multilingual":
            return self._domain
        # For multilingual splits, refine to math/code/science from English reasoning_text.
        refined = _infer_domain_from_reasoning(reasoning_text or "")
        return refined or self._domain  # fall back to "multilingual"

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
        }.get((domain or "").lower())

    def extract_extras(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ground_truth_present": "no",
            "category": ex.get("category") or self.split_name,
        }

    @classmethod
    def recommended_config(cls, split_name: str) -> FilterConfig:
        """Return the FilterConfig tuned for ``split_name``.

        Key decisions (see module docstring for rationale):

        - ``math``:           FastText English OFF (LaTeX false-drop rate ~63%).
        - ``code``/``chat``/``stem``: FastText English ON, mixed-language ON.
        - ``multilingual_*``: both filters OFF; prompts are intentionally
          non-English, reasoning is English by NVIDIA's design.
        """
        s = (split_name or "").strip().lower()
        if s not in _SPLIT_TO_LANGUAGE:
            raise ValueError(f"Unknown split: {split_name!r}")

        if s == "math":
            # LaTeX tokens (\int, \frac, \boxed) cause ~63% false-drop rate.
            # Language is guaranteed EN via split name — no detection needed.
            cfg = FilterConfig()
            cfg.enable_fasttext_english_filter = False
            cfg.enable_mixed_language_filter = False
            return cfg

        if s in {"code", "chat", "stem"}:
            from ..config import english_only_preset
            return english_only_preset()

        # multilingual_de / es / fr / it / ja
        from ..config import multilingual_preset
        return multilingual_preset()
