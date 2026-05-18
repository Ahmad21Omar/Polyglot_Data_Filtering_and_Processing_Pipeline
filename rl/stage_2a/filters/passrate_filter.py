"""Passrate / difficulty filter for RL datasets.

Motivation (OLMo-3 paper §4.4.2, "Step 2: offline difficulty filtering"):
  "We remove all samples that the model easily solves (pass rate > 62.5%)."

  Prompts where the current policy already succeeds with very high probability
  produce near-zero GRPO advantage (std ≈ 0 across rollouts) and therefore
  contribute almost no gradient signal.  Keeping them wastes compute and can
  destabilise training (spuriously inflated batch statistics).

We use a slightly more conservative default of 0.90 instead of 0.625 because:
  - We are not filtering against a specific base model (unlike OLMo-3).
  - A stricter threshold avoids accidentally removing moderately hard examples
    that would still provide positive signal for weaker models.

This filter is **optional**: many RL datasets do not expose a passrate field.
When the field is absent, the filter is simply skipped (returns None).

Drop reasons:
  ``passrate_too_high``  — passrate > max_passrate  (too easy)
  ``passrate_zero``      — passrate == 0.0           (always fails; optional)
"""

from __future__ import annotations

from typing import Optional


def should_drop_by_passrate(
    passrate: Optional[float],
    *,
    max_passrate: float = 0.90,
    drop_if_zero: bool = False,
) -> Optional[str]:
    """Return a drop reason based on *passrate*, or ``None`` if the row passes.

    Args:
        passrate:      Float in [0, 1] representing the fraction of rollouts
                       that were correct.  ``None`` means the field is absent
                       — the filter is skipped and ``None`` is returned.
        max_passrate:  Upper bound (exclusive).  Rows with passrate strictly
                       above this value are dropped as "too easy".
                       Default: 0.90 (conservative; see module docstring).
        drop_if_zero:  If True, also drop rows where passrate == 0.0 exactly
                       ("always fails" — no positive reward signal).
                       Default: False (we keep them; they may still be useful
                       as hard negatives or after curriculum ordering).

    Returns:
        Drop-reason string or ``None``.
    """
    if passrate is None:
        return None  # field absent → skip filter

    try:
        p = float(passrate)
    except (TypeError, ValueError):
        return None  # unparseable → skip

    if p > max_passrate:
        return "passrate_too_high"

    if drop_if_zero and p == 0.0:
        return "passrate_zero"

    return None
