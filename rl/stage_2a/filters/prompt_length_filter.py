"""Prompt length filter for RL datasets.

Drops prompts that are shorter than a configurable minimum character count
(after whitespace normalisation).  Short prompts are typically ambiguous,
degenerate, or incomplete tasks that provide no useful learning signal.

Drop reason: ``prompt_too_short``
"""

from __future__ import annotations

import re
from typing import Optional

_WS_RE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def should_drop_prompt_length(
    prompt_text: str,
    *,
    min_chars: int = 30,
) -> Optional[str]:
    """Return ``'prompt_too_short'`` if the prompt is below *min_chars*, else ``None``.

    Args:
        prompt_text: The flat prompt string to check.
        min_chars:   Minimum number of characters after whitespace normalisation.

    Returns:
        Drop-reason string or ``None``.
    """
    if len(normalize_ws(prompt_text)) < min_chars:
        return "prompt_too_short"
    return None
