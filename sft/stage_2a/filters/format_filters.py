from __future__ import annotations

import re
from typing import Optional, Tuple

# We keep the regex simple and close to Open-Instruct's `filter_cots.py`.
# Dolci commonly contains: <think>...</think> + (final answer as plain text)

THINK_OPEN_RE = re.compile(r"<think>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)

# Strict full-match variants like Open-Instruct uses.
THINK_ONLY_FULLMATCH = re.compile(r"^<think>[\s\S]*?</think>[\s\S]*?$", flags=re.IGNORECASE)
THINK_ANSWER_FULLMATCH = re.compile(
    r"^<think>[\s\S]*?</think><answer>[\s\S]*?</answer>$", flags=re.IGNORECASE
)

THINK_BLOCK_EXTRACT = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", flags=re.IGNORECASE)
ANSWER_BLOCK_EXTRACT = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", flags=re.IGNORECASE)


def has_balanced_think_tags(text: str) -> bool:
    """Return True if both <think> and </think> exist and are ordered."""
    if not text:
        return False
    m_open = THINK_OPEN_RE.search(text)
    if not m_open:
        return False
    m_close = THINK_CLOSE_RE.search(text)
    if not m_close:
        return False
    return m_open.start() < m_close.start()


def is_truncated_think(text: str) -> bool:
    """Detect likely truncation: has <think> but missing </think>."""
    if not text:
        return False
    has_open = THINK_OPEN_RE.search(text) is not None
    has_close = THINK_CLOSE_RE.search(text) is not None
    return has_open and not has_close


def matches_olmo_think_format(text: str) -> bool:
    """Close to Open-Instruct is_think(): must start with <think> and have </think>."""
    if not isinstance(text, str) or not text:
        return False
    return bool(THINK_ONLY_FULLMATCH.fullmatch(text))


def matches_olmo_think_answer_format(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return bool(THINK_ANSWER_FULLMATCH.fullmatch(text))


def extract_think_and_answer(text: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Return (think_text, response_without_tags, answer_tag_text).

    - think_text is inner <think>...</think> content if present
    - response_without_tags removes the whole <think> block and <answer> block
    - answer_tag_text is inner <answer>...</answer> content if present

    Note: Dolci often doesn't have <answer> tags, so answer_tag_text is usually None.
    """
    if not isinstance(text, str):
        return None, "", None

    think_m = THINK_BLOCK_EXTRACT.search(text)
    answer_m = ANSWER_BLOCK_EXTRACT.search(text)

    think_text = think_m.group(1).strip() if think_m else None
    answer_text = answer_m.group(1).strip() if answer_m else None

    # Remove blocks using regex substitution to avoid index-shift bugs after the first removal.
    response_wo = THINK_BLOCK_EXTRACT.sub("", text).strip()
    response_wo = ANSWER_BLOCK_EXTRACT.sub("", response_wo).strip()

    return think_text, response_wo, answer_text
