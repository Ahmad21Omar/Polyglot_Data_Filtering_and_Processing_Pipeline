from __future__ import annotations

"""Phase 2A.2 — Think / Reasoning filters.

Sankey-Stage label: ``2A.2_think_reasoning`` (think-tag presence, min chars,
truncation).

This module is a *facade* — it re-exports the canonical implementations from
:mod:`Filtering_Pipeline.stage_2a.filters.format_filters` so the public API is grouped
by Sankey phase while the underlying logic stays byte-identical with the
original Stage 2A pipeline that produced SFT-Collection-v2.

Drop reasons typically attached at this phase
---------------------------------------------
- ``truncated_think``           : ``<think>`` present but ``</think>`` missing
- ``missing_think``             : pipeline-level: no think tag where required
- ``think_too_short``           : pipeline-level: think text shorter than threshold
- ``empty_reasoning_content``   : pipeline-level: explicit reasoning column empty
- ``no_verified_proof``         : pipeline-level (math-proofs only): no verified proof found
- ``split_no_reasoning_traces`` : pipeline-level (nemotron v2 only): split has no traces

Only ``truncated_think`` and the parsing helpers live here; the others are
emitted by the per-dataset filter_and_format scripts based on dataset-specific
fields.
"""

# Re-export the canonical implementations.
from .format_filters import (  # noqa: F401  (re-exported)
    THINK_OPEN_RE,
    THINK_CLOSE_RE,
    THINK_ONLY_FULLMATCH,
    THINK_ANSWER_FULLMATCH,
    THINK_BLOCK_EXTRACT,
    ANSWER_BLOCK_EXTRACT,
    has_balanced_think_tags,
    is_truncated_think,
    matches_olmo_think_format,
    matches_olmo_think_answer_format,
    extract_think_and_answer,
)

__all__ = [
    "THINK_OPEN_RE",
    "THINK_CLOSE_RE",
    "THINK_ONLY_FULLMATCH",
    "THINK_ANSWER_FULLMATCH",
    "THINK_BLOCK_EXTRACT",
    "ANSWER_BLOCK_EXTRACT",
    "has_balanced_think_tags",
    "is_truncated_think",
    "matches_olmo_think_format",
    "matches_olmo_think_answer_format",
    "extract_think_and_answer",
]
