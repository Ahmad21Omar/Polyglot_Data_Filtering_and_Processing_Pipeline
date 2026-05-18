from __future__ import annotations

"""Cheap identity / boilerplate filters.

These filters remove assistant responses that contain obvious model self-
identification or boilerplate such as:
- "As an AI language model ..."
- "I am ChatGPT"
- "I am DeepSeek"

Rationale
---------
For SFT corpora these phrases are typically undesirable artifacts that reduce
usefulness and can leak model branding.

The intent is to be conservative: flag only very common, explicit patterns.
"""

import re
from typing import Optional


# Common self-identification / boilerplate patterns.
# Keep this intentionally short and high-precision.
_IDENTITY_PATTERNS = [
    # Generic
    r"\bas an ai language model\b",
    r"\bi am an ai\b",
    r"\bi am an artificial intelligence\b",
    r"\bi(?:'| a)m a language model\b",
    # Popular model names / brands
    r"\bi am chatgpt\b",
    r"\bi(?:'| a)m gpt[- ]?\d\b",
    r"\bi am openai\b",
    r"\bi am claude\b",
    r"\bi am anthropic\b",
    r"\bi am gemini\b",
    r"\bi am deepseek\b",
    r"\bi am llama\b",
    r"\bi am mistral\b",
]

IDENTITY_RE = re.compile("|".join(f"(?:{p})" for p in _IDENTITY_PATTERNS), flags=re.IGNORECASE)


def should_drop_identity_text(text: str) -> Optional[str]:
    """Return drop reason if text contains explicit model self-identification."""
    if not isinstance(text, str) or not text.strip():
        return None
    if IDENTITY_RE.search(text):
        return "identity_self_id"
    return None
