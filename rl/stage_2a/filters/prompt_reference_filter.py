"""HTTP-link and image-reference filters for RL prompt text.

Prompts containing bare HTTP/HTTPS URLs or image/figure references are
likely unusable: they depend on external resources that are not part of
the text and cannot be resolved during training or inference.

Drop reasons:
  ``prompt_has_http``   — an HTTP/HTTPS URL was found in the prompt.
  ``prompt_has_image``  — an image or figure reference was found.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# ── HTTP/URL detection ───────────────────────────────────────────────────────
_HTTP_RE = re.compile(r"https?://", flags=re.IGNORECASE)


def has_http_link(text: str) -> bool:
    """Return True if *text* contains at least one HTTP/HTTPS URL."""
    return bool(_HTTP_RE.search(text or ""))


# ── Image / figure reference detection ──────────────────────────────────────
_DEFAULT_IMAGE_KEYWORDS: tuple[str, ...] = (
    "[image]",
    "[figure]",
    "[img]",
    "[photo]",
    "figure ",       # "Figure 1", "figure 2", ...
    "diagram",
    "jpeg",
    "jpg",
    "png",
    "svg",
    ".gif",
)


def has_image_reference(
    text: str,
    keywords: Iterable[str] = _DEFAULT_IMAGE_KEYWORDS,
) -> bool:
    """Return True if *text* contains any image/figure keyword."""
    t = (text or "").lower()
    return any(k in t for k in keywords)


# ── Combined convenience function ────────────────────────────────────────────

def should_drop_prompt_references(
    prompt_text: str,
    *,
    drop_if_has_http: bool = True,
    drop_if_has_images: bool = True,
    image_keywords: Iterable[str] = _DEFAULT_IMAGE_KEYWORDS,
) -> Optional[str]:
    """Return a drop reason if the prompt contains unusable references, else None.

    Args:
        prompt_text:        Flat prompt string.
        drop_if_has_http:   Drop if an HTTP/HTTPS URL is present.
        drop_if_has_images: Drop if an image/figure keyword is present.
        image_keywords:     Override the default image keyword list.

    Returns:
        ``'prompt_has_http'``, ``'prompt_has_image'``, or ``None``.
    """
    if drop_if_has_http and has_http_link(prompt_text):
        return "prompt_has_http"
    if drop_if_has_images and has_image_reference(prompt_text, image_keywords):
        return "prompt_has_image"
    return None
