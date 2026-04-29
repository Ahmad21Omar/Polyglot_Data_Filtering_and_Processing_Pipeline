from __future__ import annotations

"""Phase 2A.5 — Identity / Safety filters.

Sankey-Stage label: ``2A.5_identity_safety`` (identity self-id, cutoff mention).

Facade module that bundles the three canonical identity/safety modules:

- :mod:`Filtering_Pipeline.filters.identity_filters` — model self-identification
- :mod:`Filtering_Pipeline.filters.content_filters`  — knowledge-cutoff phrasing
- :mod:`Filtering_Pipeline.filters.domain_filters`   — optional toxicity/redaction flags

Drop reasons typically attached at this phase
---------------------------------------------
- ``identity_self_id``  : assistant says "I am ChatGPT", "as an AI language model", etc.
- ``cutoff_mention``    : assistant mentions "as of my last update in 2023", etc.
- ``unsafe_toxic``      : optional, when source dataset exposes a ``toxic`` flag
- ``unsafe_redacted``   : optional, when source dataset exposes a ``redacted`` flag
"""

# ── Identity (identity_filters) ──────────────────────────────────────────────
from .identity_filters import (  # noqa: F401
    IDENTITY_RE,
    should_drop_identity_text,
)

# ── Knowledge-cutoff phrasing (content_filters) ──────────────────────────────
from .content_filters import (  # noqa: F401
    CUT_OFF_PATTERN,
    find_cutoff_mentions_in_messages,
    has_cutoff_mention,
)

# ── Optional safety flags (domain_filters) ───────────────────────────────────
from .domain_filters import (  # noqa: F401
    should_drop_wildchat_like_row,
)

__all__ = [
    # identity
    "IDENTITY_RE",
    "should_drop_identity_text",
    # cutoff
    "CUT_OFF_PATTERN",
    "find_cutoff_mentions_in_messages",
    "has_cutoff_mention",
    # safety flags
    "should_drop_wildchat_like_row",
]
