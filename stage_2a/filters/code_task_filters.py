from __future__ import annotations

"""Cheap structural filters for code-task datasets (OpenThoughts-like).

Provenance:
- Ported from `Master_Thesis/Code_Templat/open-thoughts/open_thoughts/code/filters.py`

These helpers are intended for datasets that have fields like:
- `problem` / `description`
- `tests` (or similar) containing `inputs` and `outputs`
- `solutions`

They are *not* used in the Dolci Think SFT pipeline (Dolci is chat-style messages)
but can be used for later code-focused datasets in your collection.
"""

import ast
import json
from typing import Any, Dict, List, Optional


def filter_problem(description: str, min_description_length: int = 200) -> bool:
    """Keep only problems with enough text and without obvious image/link reliance."""
    if description is None:
        return False
    desc = description.lower()
    if "http://" in desc or "https://" in desc:
        return False
    if "[image]" in desc:
        return False
    if len(description) < min_description_length:
        return False
    return True


def _loads_maybe_json_or_literal(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return None


def filter_tests(tests: Any) -> bool:
    """Keep only tasks with non-empty test inputs and outputs."""
    tests_obj = _loads_maybe_json_or_literal(tests)
    if tests_obj is None or not isinstance(tests_obj, dict):
        return False
    if len(tests_obj.get("inputs", []) or []) == 0:
        return False
    if len(tests_obj.get("outputs", []) or []) == 0:
        return False
    return True


def filter_solutions(solutions: Any) -> bool:
    """Keep only tasks with at least one solution."""
    if solutions is None:
        return False

    sol_obj = _loads_maybe_json_or_literal(solutions)

    if isinstance(sol_obj, list):
        return len(sol_obj) > 0

    if isinstance(sol_obj, dict):
        # OpenThoughts uses `solutions.get("solution", [])` sometimes.
        return len(sol_obj.get("solution", []) or []) > 0

    return False


def filter_num_solutions(num_solutions: Any) -> bool:
    try:
        return int(num_solutions) > 0
    except Exception:
        return False


def should_drop_code_task_row(
    ex: Dict[str, Any],
    *,
    problem_field: str = "problem",
    tests_field: str = "tests",
    solutions_field: str = "solutions",
    min_description_length: int = 200,
    require_tests: bool = True,
    require_solutions: bool = True,
) -> Optional[str]:
    """Return drop reason if a code-task row is structurally unusable, else None."""

    problem = ex.get(problem_field)
    if not isinstance(problem, str) or not filter_problem(problem, min_description_length=min_description_length):
        return "code_problem_bad_description"

    if require_tests and not filter_tests(ex.get(tests_field)):
        return "code_problem_bad_tests"

    if require_solutions and not filter_solutions(ex.get(solutions_field)):
        return "code_problem_bad_solutions"

    return None
