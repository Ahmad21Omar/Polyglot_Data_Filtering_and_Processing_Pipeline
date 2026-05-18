from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

# Ported (lightly) from Open-Instruct `filter_cutoff_date.py`.
CUT_OFF_PATTERN = re.compile(
    r"""(?ix)
    (?:
        as\s+(?:of\s+)?
        (?:my\s+)?
        (?:last|latest)\s+
        (?:update|knowledge)
        (?:\s+was)?\s+in
        |
        knowledge\s+cutoff
        |
        last\s+updated?\s+in
    )
    \s+
    (?:
        (?:January|February|March|April|May|June|July|
           August|September|October|November|December)
        \s+
    )?
    (\d{4})
    """
)


def _iter_message_texts(messages: Any) -> Iterable[str]:
    if not isinstance(messages, list):
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            yield content


def find_cutoff_mentions_in_messages(messages: Any) -> List[str]:
    matches: List[str] = []
    for content in _iter_message_texts(messages):
        matches.extend(m.group(0) for m in CUT_OFF_PATTERN.finditer(content))
    return matches


def has_cutoff_mention(messages: Any) -> bool:
    for content in _iter_message_texts(messages):
        if CUT_OFF_PATTERN.search(content):
            return True
    return False
