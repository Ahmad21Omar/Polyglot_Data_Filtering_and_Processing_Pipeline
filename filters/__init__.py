"""Reusable filtering utilities for SFT-Collection-v2 (Stage 2A).

The public API is grouped by Sankey-pipeline phase:

- :mod:`Filtering_Pipeline.filters.phase_2a1_structural`     — messages / assistant presence, prompt length
- :mod:`Filtering_Pipeline.filters.phase_2a2_think_reasoning` — think-tag presence, min chars, truncation
- :mod:`Filtering_Pipeline.filters.phase_2a3_prompt_question` — URL / image / length heuristics
- :mod:`Filtering_Pipeline.filters.phase_2a4_language`        — FastText LID, Chinese ratio
- :mod:`Filtering_Pipeline.filters.phase_2a5_identity_safety` — identity self-id, cutoff mention, safety flags
- :mod:`Filtering_Pipeline.filters.phase_2a6_repetition`      — sentence / phrase repetition

Cross-cutting utilities live under :mod:`Filtering_Pipeline.filters._utils`
(code-task structural checks, FastText classifier wrapper).

Stability note
--------------
The original modules (``format_filters``, ``question_filters``,
``language_filters``, ``english_fasttext_filter``, ``language_detector``,
``identity_filters``, ``content_filters``, ``domain_filters``,
``repetition_filters``, ``code_task_filters``, ``fasttext_filters``) remain in
this package and are still the *canonical* implementations — the phase modules
above are thin facades that re-export them. This guarantees byte-for-byte
identical behaviour with the Stage 2A run that produced SFT-Collection-v2,
while giving new pipeline scripts a phase-organised public API.
"""

# Re-export the 6 phase modules at package level for convenience.
from . import (  # noqa: F401
    phase_2a1_structural,
    phase_2a2_think_reasoning,
    phase_2a3_prompt_question,
    phase_2a4_language,
    phase_2a5_identity_safety,
    phase_2a6_repetition,
)

__all__ = [
    "phase_2a1_structural",
    "phase_2a2_think_reasoning",
    "phase_2a3_prompt_question",
    "phase_2a4_language",
    "phase_2a5_identity_safety",
    "phase_2a6_repetition",
]
