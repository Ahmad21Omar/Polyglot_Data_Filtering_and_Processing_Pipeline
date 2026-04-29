from __future__ import annotations

"""Domain-specific filters (adapter layer).

OLMo3 describes "domain specific accuracy filtering" (math/code/IF + chat safety tags).
Not all datasets (including Dolci) expose the required metadata.

This module provides *optional* predicates you can turn on per-dataset.
For Dolci we generally won't apply these yet (user requested: create module but do not use).
"""

from typing import Any, Dict, Optional


def should_drop_wildchat_like_row(ex: Dict[str, Any]) -> Optional[str]:
    """If row looks like WildChat and has safety flags, return a drop reason.

    Returns:
      - 'unsafe_toxic' if ex['toxic'] is True
      - 'unsafe_redacted' if ex['redacted'] is True
      - None otherwise
    """
    if ex.get("toxic") is True:
        return "unsafe_toxic"
    if ex.get("redacted") is True:
        return "unsafe_redacted"
    return None
