"""RL-specific filtering utilities.

All filters operate on prompt text or ground-truth text (not on model outputs).
Each function returns either a boolean or an Optional[str] drop-reason so they
can be wired directly into the `_drop_reason` audit pipeline.
"""
