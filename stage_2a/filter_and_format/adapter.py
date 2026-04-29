from __future__ import annotations

"""Per-dataset adapter interface for the generic Stage 2A pipeline.

A ``DatasetAdapter`` knows everything about *one* source dataset that the
generic pipeline cannot derive from a chat-style ``messages`` list. Most
methods have sensible defaults that the dolci-style adapters can use as-is;
only the methods marked **required** must always be implemented.

The adapter is *stateless* for the row-level methods — they receive a single
example dict and return a small piece of derived data. The pipeline calls
them in this order for every row:

    1. ``adapter.pre_check(ex)``
    2. ``adapter.extract_messages(ex)``
    3. (pipeline runs the 6 phases)
    4. for kept rows: ``adapter.extract_subsource_raw(ex)``,
       ``infer_language``, ``infer_domain``, ``infer_reasoning_type``,
       ``extract_extras``
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class DatasetAdapter(ABC):
    """Abstract per-dataset adapter."""

    # ── identity (required) ────────────────────────────────────────────────

    #: HuggingFace dataset id, e.g. ``"allenai/Dolci-Think-SFT-7B"``.
    dataset_id: str = ""

    #: License string written into every kept row.
    license: Optional[str] = None

    #: Originating model, written into every kept row (or ``None``).
    used_by_model: Optional[str] = None

    #: ISO date string identifying the snapshot of the source dataset
    #: used for this run; written into every kept row.
    dataset_version_date: Optional[str] = None

    # ── row-level hooks ────────────────────────────────────────────────────

    def pre_check(self, ex: Dict[str, Any]) -> Optional[str]:
        """Optional dataset-specific drop reasons that fire **before** the
        generic 6-phase pipeline.

        Examples (legacy scripts):
        - dolci: ``missing_dataset_source`` if ``ex["dataset_source"]`` is None
        - nemotron-math-proofs: ``no_verified_proof`` if no verified proof exists
        - nemotron-math-v2: ``empty_reasoning_content`` if reasoning column empty
        - soofi: ``missing_language`` if language metadata absent

        Default: no pre-check.
        """
        return None

    @abstractmethod
    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        """Return the chat-style ``messages`` list for this row.

        For datasets that already have a ``messages`` field this is just
        ``ex["messages"]``. For datasets that store the prompt/response in
        separate fields (e.g. Llama-Nemotron's ``input`` / ``output``) the
        adapter assembles a synthetic ``messages`` list.

        Returning a non-list (or an empty list) makes Phase 2A.1 emit
        ``bad_messages``.
        """

    @abstractmethod
    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        """Return a stable per-row identifier.

        Used as the third component of ``f"{dataset_id}|{split}|{row_id}"`` →
        ``example_id``. Most adapters return ``str(ex.get("id") or idx)``.
        """

    # ── kept-row hooks (only called when the row passes all 6 phases) ──────

    def extract_subsource_raw(self, ex: Dict[str, Any]) -> Optional[str]:
        """Return the raw subsource label for ``subsource_raw``.

        Default: ``None`` (no subsource info).
        """
        return None

    def infer_language(
        self,
        *,
        ex: Dict[str, Any],
        messages: Any,
        default_lang: str = "en",
    ) -> str:
        """Return ISO 639-1 language code for the kept row. Default: English.

        Multilingual adapters override this either via a metadata lookup or
        via :func:`...phase_2a4_language.detect_language` on the assistant
        text.
        """
        return default_lang

    def infer_domain(
        self,
        *,
        subsource_raw: Optional[str],
        prompt_text: str,
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        """Return a domain label (``"math"``, ``"code"``, ``"science"``,
        ``"chat"``, ...) or ``None``.

        Default: ``None``.
        """
        return None

    def infer_reasoning_type(
        self,
        *,
        domain: Optional[str],
        reasoning_text: Optional[str],
    ) -> Optional[str]:
        """Return a reasoning-type label (e.g. ``"think"``, ``"none"``).

        Default: ``"think"`` if reasoning text is present, else ``None``.
        """
        return "think" if reasoning_text else None

    def extract_extras(
        self,
        ex: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a dict of optional extras to merge onto the kept row.

        Recognised keys: ``difficulty``, ``category``, ``ground_truth_present``,
        ``final_answer_text``, ``source_dataset_id``,
        ``translation_of_example_id``. Unknown keys are ignored by the
        pipeline (so adapters may add audit-only keys without harm).

        Default: empty dict.
        """
        return {}
