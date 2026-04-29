from __future__ import annotations

"""Phase 2A.3 — Prompt / Question filters.

Sankey-Stage label: ``2A.3_prompt_question`` (URL / image / length heuristics).

Facade re-exporting the canonical implementation from
:mod:`Filtering_Pipeline.stage_2a.filters.question_filters`.

Drop reasons typically attached at this phase
---------------------------------------------
- ``question_has_http``       : prompt contains a URL
- ``question_has_image_ref``  : prompt references an image / figure / diagram
- ``question_multipart``      : prompt is structured as a multipart "a)..b).." question
- ``question_too_short``      : prompt shorter than ``min_length`` (default 200 chars)
- ``empty_question``          : prompt is empty after stripping
"""

from .question_filters import (  # noqa: F401  (re-exported)
    DEFAULT_IMAGE_KEYWORDS,
    HTTP_PATTERN,
    contains_any_keyword,
    is_multipart_like,
    should_drop_question_text,
)

__all__ = [
    "DEFAULT_IMAGE_KEYWORDS",
    "HTTP_PATTERN",
    "contains_any_keyword",
    "is_multipart_like",
    "should_drop_question_text",
]
