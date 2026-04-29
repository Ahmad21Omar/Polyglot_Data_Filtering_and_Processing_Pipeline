from __future__ import annotations

import re
from typing import Any

# Ported from Open-Instruct `filter_chinese.py`.
CJK_UNIFIED_IDEOGRAPHS_RE = re.compile(r"[\u4e00-\u9fff]")


def chinese_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    matches = CJK_UNIFIED_IDEOGRAPHS_RE.findall(text)
    return len(matches) / max(1, len(text))


def has_too_much_chinese(text: str, *, threshold: float = 0.05) -> bool:
    return chinese_character_ratio(text) >= threshold


def should_filter_example_by_chinese(
    *,
    messages: Any,
    threshold: float = 0.05,
    include_user_turns: bool = False,
) -> bool:
    if not isinstance(messages, list):
        return False

    if include_user_turns:
        # last two messages ~ (user, assistant) if present
        messages_to_check = messages[-2:] if len(messages) >= 2 else messages
    else:
        # only final assistant message
        assistant_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        messages_to_check = [assistant_msgs[-1]] if assistant_msgs else []

    for msg in messages_to_check:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue
        if has_too_much_chinese(content, threshold=threshold):
            return True

    return False
