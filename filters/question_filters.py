from __future__ import annotations

"""Cheap question/prompt filters inspired by OpenThoughts.

These are *domain-agnostic* helpers that catch common low-quality / unusable
questions without calling any LLMs.

Provenance:
- Inspired by `Master_Thesis/Code_Templat/open-thoughts/open_thoughts/math/filter.py`
  (image/diagram keyword filtering, multipart detection)
- Inspired by `Master_Thesis/Code_Templat/open-thoughts/open_thoughts/code/filters.py`
  (problem description sanity checks: min length, http links, [image] markers)

The functions here return either a boolean (keep/drop) or an optional drop reason
string so they can be integrated into `_drop_reason` pipelines.
"""

import re
from typing import Iterable, Optional


DEFAULT_IMAGE_KEYWORDS = (
    "figure",
    "diagram",
    "jpeg",
    "png",
    "jpg",
    "svg",
    "[image]",
)

HTTP_PATTERN = re.compile(r"https?://", flags=re.IGNORECASE)


def contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def is_multipart_like(text: str) -> bool:
    """Heuristic for questions like 'a) ... b) ...'.

    Ported from OpenThoughts math filter: they drop multipart questions.
    """
    t = (text or "").strip().lower()
    return t.startswith("a)") and "b)" in t


def should_drop_question_text(
    text: str,
    *,
    min_length: int = 200,
    drop_if_has_http: bool = True,
    drop_if_has_images: bool = True,
    image_keywords: Iterable[str] = DEFAULT_IMAGE_KEYWORDS,
    drop_if_multipart: bool = True,
) -> Optional[str]:
    """Return a drop reason if the question text looks unusable, else None."""
    t = (text or "")
    if not t.strip():
        return "empty_question"

    if drop_if_has_http and HTTP_PATTERN.search(t):
        return "question_has_http"

    if drop_if_has_images and contains_any_keyword(t, image_keywords):
        return "question_has_image_ref"

    if drop_if_multipart and is_multipart_like(t):
        return "question_multipart"

    if len(t) < min_length:
        return "question_too_short"

    return None
