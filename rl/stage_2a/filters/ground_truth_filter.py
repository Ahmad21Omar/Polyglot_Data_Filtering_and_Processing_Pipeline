"""Ground-truth quality filters for RL datasets.

RL training requires a reliable verifier signal.  These filters catch the
two most common failure modes in ground-truth fields:

1. **Missing ground truth** — the field is ``None`` / empty string.
2. **Placeholder ground truth** — the field contains a token that looks like
   a valid answer but is actually a stand-in (e.g. ``"N/A"``, ``"?"``,
   ``"none"``, ``"unknown"``).

Drop reasons:
  ``missing_ground_truth``         — ground_truth is None or empty.
  ``placeholder_ground_truth``     — ground_truth is a known placeholder token.
  ``ground_truth_too_short``       — ground_truth is shorter than min_gt_chars.
  ``missing_verification_info``    — schema_* verifier has no verification_info_raw.
  ``invalid_verification_info``    — verification_info_raw is not valid JSON.
  ``missing_verification_info_<k>``— schema_* verifier is missing expected key <k>.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

# ── Placeholder token list ────────────────────────────────────────────────────
# Lower-cased, stripped strings that look like answers but carry no signal.
_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {
        "n/a",
        "na",
        "n.a.",
        "none",
        "null",
        "nil",
        "unknown",
        "?",
        "??",
        "???",
        "-",
        "--",
        "---",
        "tbd",
        "todo",
        "placeholder",
        "answer",          # literal word "answer" without content
        "answer here",
        "<answer>",
        "</answer>",
        "<answer></answer>",
        "no answer",
        "not provided",
        "not available",
        "not applicable",
    }
)


# ── Verifier-type classification sets ────────────────────────────────────────

# These verifiers carry ALL payload in verification_info_raw — ground_truth_text
# is not populated and must not be required.
_GT_NOT_IN_GT_FIELD: frozenset[str] = frozenset({
    "schema_pydantic",
    "schema_structured_outputs",
})

# These verifiers have an optional ground_truth_text (may be empty / None).
_GT_OPTIONAL: frozenset[str] = frozenset({
    "llm_judge_open",
})

# Per-verifier key expected in verification_info_raw (for _GT_NOT_IN_GT_FIELD).
_VERIF_INFO_REQUIRED_KEY: Dict[str, str] = {
    "schema_pydantic": "schema_code",
    "schema_structured_outputs": "schema_json",
}


def is_placeholder_ground_truth(ground_truth_text: Optional[str]) -> bool:
    """Return True if *ground_truth_text* is a known placeholder token."""
    if not ground_truth_text:
        return False
    return ground_truth_text.strip().lower() in _PLACEHOLDER_TOKENS


# ── Combined gate ─────────────────────────────────────────────────────────────

def should_drop_ground_truth(
    ground_truth_text: Optional[str],
    *,
    require_present: bool = True,
    check_placeholder: bool = True,
    min_gt_chars: int = 1,
) -> Optional[str]:
    """Return a drop reason if the ground truth fails quality checks, else None.

    Checks (in order):
      1. Missing / empty  → ``'missing_ground_truth'``
      2. Placeholder token → ``'placeholder_ground_truth'``
      3. Too short         → ``'ground_truth_too_short'``

    Args:
        ground_truth_text: Normalised ground-truth string (or None).
        require_present:   Trigger check (1) — set False to skip.
        check_placeholder: Trigger check (2).
        min_gt_chars:      Minimum character length after stripping for check (3).
                           Default 1 catches empty strings that slipped past (1).

    Returns:
        Drop-reason string or ``None``.
    """
    if require_present and not (ground_truth_text and ground_truth_text.strip()):
        return "missing_ground_truth"

    if ground_truth_text and ground_truth_text.strip():
        if check_placeholder and is_placeholder_ground_truth(ground_truth_text):
            return "placeholder_ground_truth"

        if len(ground_truth_text.strip()) < min_gt_chars:
            return "ground_truth_too_short"

    return None


# ── Verifier-aware GT gate ────────────────────────────────────────────────────

def should_drop_ground_truth_for_verifier(
    ground_truth_text: Optional[str],
    verifier_type: str,
    verification_info_raw: Optional[str] = None,
    *,
    check_placeholder: bool = True,
    min_gt_chars: int = 1,
) -> Optional[str]:
    """Verifier-aware ground-truth quality gate.

    Dispatches based on ``verifier_type``:

    * ``schema_pydantic`` / ``schema_structured_outputs`` — GT lives exclusively
      in ``verification_info_raw``.  Validates that the expected key
      (``schema_code`` / ``schema_json``) is present in the JSON blob.
      ``ground_truth_text`` is not checked.

    * ``llm_judge_open`` — GT is optional.  Skips the ``require_present``
      check but still rejects placeholder / too-short values if GT is supplied.

    * All other verifier types — delegates to :func:`should_drop_ground_truth`
      with ``require_present=True``.

    Args:
        ground_truth_text:    Normalised GT string (or None).
        verifier_type:        Schema verifier_type enum value.
        verification_info_raw: Raw JSON string from ``verification_info_raw``
                               column (used for schema_* verifiers).
        check_placeholder:    Forward to :func:`should_drop_ground_truth`.
        min_gt_chars:         Forward to :func:`should_drop_ground_truth`.

    Returns:
        Drop-reason string or ``None``.
    """
    if verifier_type in _GT_NOT_IN_GT_FIELD:
        # GT is in verification_info_raw only — validate that blob.
        if not verification_info_raw:
            return "missing_verification_info"
        try:
            extra = json.loads(verification_info_raw)
        except (json.JSONDecodeError, TypeError):
            return "invalid_verification_info"
        required_key = _VERIF_INFO_REQUIRED_KEY.get(verifier_type)
        if required_key and not extra.get(required_key):
            return f"missing_verification_info_{required_key}"
        return None

    if verifier_type in _GT_OPTIONAL:
        # GT is optional — skip require_present but still check quality.
        return should_drop_ground_truth(
            ground_truth_text,
            require_present=False,
            check_placeholder=check_placeholder,
            min_gt_chars=min_gt_chars,
        )

    # Default: full GT required.
    return should_drop_ground_truth(
        ground_truth_text,
        require_present=True,
        check_placeholder=check_placeholder,
        min_gt_chars=min_gt_chars,
    )
