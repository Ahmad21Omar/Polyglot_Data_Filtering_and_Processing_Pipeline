"""Filter + normalize `PrimeIntellect/SYNTHETIC-2-RL` into RL v1 schema.

SYNTHETIC-2-RL is a heterogeneous mixture — 10 distinct verifier dispatches
keyed by `(problem_id-prefix, sorted(verification_info.keys()))`:

| prefix           | vi keys                                                | verifier_type      | domain                |
|------------------|--------------------------------------------------------|--------------------|-----------------------|
| sky_work_math    | {ground_truth}                                         | math_equiv         | math                  |
| prime_rl_code    | {style, test_cases}  (fn_name in test_cases → asserts) | code_asserts       | code                  |
| prime_rl_code    | {style, test_cases}  (no fn_name → stdio)              | code_stdio         | code                  |
| ifeval           | {ground_truth}  (nested {func_name, …})                | if_rules           | instruction_following |
| formatask_e{0-3} | {ground_truth}                                         | text_match         | text_extraction       |
| unscramble       | {ground_truth, scrambled}                              | text_match         | text_extraction       |
| formatask_dual_* | {ground_truth1, ground_truth2}                         | multi_gt           | text_extraction       |
| ascii_format_*   | {description, ground_truth}                            | text_match         | structured_outputs    |
| (null pid)       | {code_output}                                          | structured_match   | code_output_pred      |
| (null pid)       | {ground_truth, reasoning_gym_dataset, reasoning_gym_entry} | puzzle_match  | puzzle                |
| pydantic_*       | {model_name, pydantic_config}                          | schema_pydantic    | structured_outputs    |

verifier_source = "prime_intellect" (rl_schema_v1 §3.1).

Input prompt is a flat string — wrap as a single user turn for context_messages.

Output: corpora/rl/synthetic2_rl/rl_synthetic2_rl_v1.{kept,dropped}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, Features, Value, load_from_disk

from Filtering_Pipeline.rl.stage_2a.filters.prompt_length_filter import (
    should_drop_prompt_length,
)
from Filtering_Pipeline.rl.stage_2a.filters.prompt_reference_filter import (
    should_drop_prompt_references,
)
from Filtering_Pipeline.rl.stage_2a.filters.prompt_repetition_filter import (
    should_drop_prompt_repetition,
)
from Filtering_Pipeline.rl.stage_2a.filters.ground_truth_filter import (
    should_drop_ground_truth_for_verifier,
)
from Filtering_Pipeline.rl.stage_2a.filters.passrate_filter import (
    should_drop_by_passrate,
)
from Filtering_Pipeline.rl.stage_2a.filters.english_filter import (
    RLEnglishLidConfig,
    is_fasttext_available,
    should_drop_non_english_prompt,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PID_SUFFIX_RE = re.compile(r"^(.*?)[_-]\d+$")


def _problem_family(problem_id: Optional[str]) -> Optional[str]:
    """Extract the canonical task-family prefix from a problem_id."""
    if problem_id is None:
        return None
    pid = problem_id.strip()
    # ascii_format_* IDs do not end in a number; they keep their full repo-path suffix.
    if pid.startswith("ascii_format_"):
        return "ascii_format"
    if pid.startswith("pydantic_adherance"):
        return "pydantic_adherance"
    if pid.startswith("complex_json_output_"):
        return "complex_json_output"
    if pid.startswith("formatask_dual_"):
        # formatask_dual_e0_022866 → "formatask_dual_e0"
        return pid.rsplit("_", 1)[0] if "_" in pid else pid
    if pid.startswith("formatask_"):
        return pid.rsplit("_", 1)[0] if "_" in pid else pid
    # Strip trailing _NNN or -NNN
    m = _PID_SUFFIX_RE.match(pid)
    return m.group(1) if m else pid


def _parse_verification_info(vi_str: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(vi_str, str) or not vi_str:
        return None
    try:
        parsed = json.loads(vi_str)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _maybe_parse_nested_json(v: Any) -> Any:
    """If v is a JSON string, parse it; else return as-is."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def _classify_row(
    problem_id: Optional[str],
    vi: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (domain, verifier_type, ability, subfamily) or (None,)*4 if unclassifiable.

    `ability` is the fine-grained upstream tag; `subfamily` carries a finer
    distinction (e.g. reasoning_gym task name, ifeval func, code stdio vs asserts).
    """
    if vi is None:
        return (None, None, None, None)
    keys = frozenset(vi.keys())
    fam = _problem_family(problem_id)

    # sky_work_math → math_equiv
    if fam == "sky_work_math" and keys == {"ground_truth"}:
        return ("math", "math_equiv", "sky_work_math", None)

    # ifeval → if_rules
    if fam == "ifeval" and keys == {"ground_truth"}:
        return ("instruction_following", "if_rules", "ifeval", None)

    # formatask_e{0-3} → text_match
    if fam and re.match(r"^formatask_e\d+$", fam) and keys == {"ground_truth"}:
        return ("text_extraction", "text_match", fam, fam)

    # unscramble → text_match (with scrambled context)
    if fam == "unscramble" and keys == {"ground_truth", "scrambled"}:
        return ("text_extraction", "text_match", "unscramble", None)

    # formatask_dual_e* → multi_gt
    if fam and re.match(r"^formatask_dual_e\d+$", fam) and keys == {"ground_truth1", "ground_truth2"}:
        return ("text_extraction", "multi_gt", fam, fam)

    # ascii_format_* → text_match (long-form tree comparison)
    if fam == "ascii_format" and keys == {"description", "ground_truth"}:
        return ("structured_outputs", "text_match", "ascii_format", None)

    # prime_rl_code → code_asserts (fn_name present) or code_stdio (absent)
    if fam == "prime_rl_code" and keys == {"style", "test_cases"}:
        tc = _maybe_parse_nested_json(vi.get("test_cases"))
        if isinstance(tc, dict) and tc.get("fn_name"):
            return ("code", "code_asserts", "competitive_code", "asserts")
        return ("code", "code_stdio", "competitive_code", "stdio")

    # null-pid: code_output → structured_match
    if problem_id is None and keys == {"code_output"}:
        return ("code_output_pred", "structured_match", "code_output", None)

    # complex_json_output_*  → structured_match (nested JSON Q&A dict)
    if fam == "complex_json_output" and keys == {"ground_truth"}:
        return ("structured_outputs", "structured_match", "complex_json_output", None)

    # null-pid: reasoning_gym → puzzle_match
    if problem_id is None and keys == {"ground_truth", "reasoning_gym_dataset", "reasoning_gym_entry"}:
        rg_task = vi.get("reasoning_gym_dataset")
        return ("puzzle", "puzzle_match", "reasoning_gym", str(rg_task) if rg_task else None)

    # pydantic adherence → schema_pydantic
    if fam == "pydantic_adherance" and keys == {"model_name", "pydantic_config"}:
        return ("structured_outputs", "schema_pydantic", "pydantic_adherence", None)

    return (None, None, None, None)


# ── IFEval payload normalisation ─────────────────────────────────────────────
# Maps SYNTHETIC-2-RL's per-row `func_name` into the IFEval `instruction_id`
# vocabulary that our [[dolci-think-rl-nemo-gym]] verifier expects.

_IFEVAL_FUNC_TO_INSTRUCTION: Dict[str, str] = {
    "validate_highlighted_sections":   "detectable_format:number_highlighted_sections",
    "validate_quotation":              "startend:quotation",
    "verify_postscript":               "detectable_content:postscript",
    "validate_lowercase":              "change_case:english_lowercase",
    "verify_keywords":                 "keywords:existence",
    "validate_sections":               "detectable_format:multiple_sections",
    "validate_no_commas":              "punctuation:no_comma",
    "validate_repeat_prompt":          "combination:repeat_prompt",
    "verify_keyword_frequency":        "keywords:frequency",
    "validate_frequency_capital_words": "change_case:capital_word_frequency",
    "verify_bullet_points":            "detectable_format:number_bullet_lists",
    "validate_end":                    "startend:end_checker",
    "verify_sentence_constraint":      "length_constraints:number_sentences",
    "validate_json_format":            "detectable_format:json_format",
    "validate_uppercase":              "change_case:english_capital",
    "validate_title":                  "detectable_format:title",
    "verify_paragraph_count":          "length_constraints:number_paragraphs",
    "validate_paragraphs":             "length_constraints:number_paragraphs",
    "validate_word_constraint":        "length_constraints:number_words",
    "validate_choice":                 "detectable_format:constrained_response",
    "validate_forbidden_words":        "keywords:forbidden_words",
    "validate_placeholders":           "detectable_content:number_placeholders",
    "validate_two_responses":          "combination:two_responses",
    "verify_letter_frequency":         "keywords:letter_frequency",
}


def _ifeval_gt_to_constraint(nested_gt: Any) -> Optional[Dict[str, Any]]:
    """Convert SYNTHETIC-2-RL ifeval ground_truth (dict with func_name + per-func args)
    into the IFEval canonical {instruction_id_list, kwargs} pair.

    Returns None if func_name is missing/unknown.
    """
    if isinstance(nested_gt, str):
        nested_gt = _maybe_parse_nested_json(nested_gt)
    if not isinstance(nested_gt, dict):
        return None
    func = nested_gt.get("func_name")
    if not func:
        return None
    instr_id = _IFEVAL_FUNC_TO_INSTRUCTION.get(func)
    if not instr_id:
        return None
    # Strip None values from the per-row kwargs dict; only forward keys that are
    # actually populated. The downstream IFEval verifier already handles aliases.
    kwargs = {k: v for k, v in nested_gt.items() if v is not None and k != "func_name"}
    return {"instruction_id_list": [instr_id], "kwargs": [kwargs]}


# ── Verifier-payload builders ────────────────────────────────────────────────


def _build_verifier_payload(
    verifier_type: str,
    vi: Dict[str, Any],
    subfamily: Optional[str],
    problem_id: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (ground_truth_text, verification_info_raw_json) for one row."""
    extra: Dict[str, Any] = {}
    primary: Optional[str] = None

    if verifier_type == "math_equiv":
        gt = vi.get("ground_truth")
        primary = str(gt).strip() if gt is not None else None

    elif verifier_type == "if_rules":
        # nested ground_truth dict → canonical IFEval struct
        nested = _maybe_parse_nested_json(vi.get("ground_truth"))
        constraint = _ifeval_gt_to_constraint(nested)
        if constraint is None:
            return (None, None)  # caller drops as missing_ground_truth
        ids = constraint["instruction_id_list"]
        kwargs = constraint["kwargs"]
        gt_struct = [
            {"instruction_id": ids[i], "kwargs": kwargs[i]}
            for i in range(len(ids))
        ]
        primary = json.dumps(gt_struct, ensure_ascii=False, sort_keys=True)
        extra["constraint_format"] = "ifeval_native"
        extra["instruction_id_list"] = ids
        extra["kwargs"] = kwargs
        if isinstance(nested, dict):
            extra["constraint_func_name"] = nested.get("func_name")

    elif verifier_type == "text_match":
        # formatask_e* / unscramble / ascii_format
        gt = vi.get("ground_truth")
        primary = str(gt).strip() if gt is not None else None
        if "scrambled" in vi:
            extra["scrambled"] = vi["scrambled"]
        if "description" in vi:
            extra["description"] = vi["description"]
        if subfamily:
            extra["subfamily"] = subfamily

    elif verifier_type == "multi_gt":
        gt1 = vi.get("ground_truth1")
        gt2 = vi.get("ground_truth2")
        primary = str(gt1).strip() if gt1 is not None else None
        alternatives = [str(gt2).strip()] if gt2 is not None else []
        if alternatives:
            extra["alternative_ground_truths"] = alternatives
        if subfamily:
            extra["subfamily"] = subfamily

    elif verifier_type == "code_asserts":
        tc = _maybe_parse_nested_json(vi.get("test_cases"))
        if not isinstance(tc, dict):
            return (None, None)
        inputs = tc.get("inputs") or []
        outputs = tc.get("outputs") or []
        fn_name = tc.get("fn_name")
        if not (inputs and outputs and fn_name) or len(inputs) != len(outputs):
            return (None, None)
        # Build assert statements from (inputs, outputs) pairs.
        # inputs[i] is a list of positional args (or scalar); outputs[i] is the expected return.
        asserts: List[str] = []
        for inp, out in zip(inputs, outputs):
            args_repr = ", ".join(repr(a) for a in (inp if isinstance(inp, list) else [inp]))
            # outputs may be wrapped in a 1-elem list; unwrap for clarity.
            if isinstance(out, list) and len(out) == 1:
                out = out[0]
            asserts.append(f"assert {fn_name}({args_repr}) == {out!r}")
        primary = json.dumps(asserts, ensure_ascii=False)
        extra["test_format"] = "asserts"
        extra["language"] = "python"
        extra["fn_name"] = fn_name
        extra["num_tests"] = len(asserts)

    elif verifier_type == "code_stdio":
        tc = _maybe_parse_nested_json(vi.get("test_cases"))
        if not isinstance(tc, dict):
            return (None, None)
        inputs = list(tc.get("inputs") or [])
        outputs = list(tc.get("outputs") or [])
        if not (inputs and outputs) or len(inputs) != len(outputs):
            return (None, None)
        extra["unit_tests"] = {"inputs": inputs, "outputs": outputs}
        extra["test_format"] = "stdio"
        extra["language"] = "python"
        extra["num_tests"] = len(inputs)
        # joined outputs as primary so the GT-quality gate sees a non-empty string
        primary = "\n---\n".join(str(o) for o in outputs)[:8000] or None

    elif verifier_type == "structured_match":
        # Two sub-shapes share structured_match:
        #  * {"code_output": <any json value>}  (null-pid)
        #  * {"ground_truth": <json string>}    (complex_json_output_*)
        if "code_output" in vi:
            co = vi.get("code_output")
            primary = json.dumps(co, ensure_ascii=False, sort_keys=True) if co is not None else None
        else:
            gt = vi.get("ground_truth")
            # gt may already be a JSON string; either way we store as canonical str.
            parsed = _maybe_parse_nested_json(gt) if isinstance(gt, str) else gt
            primary = json.dumps(parsed, ensure_ascii=False, sort_keys=True) if parsed is not None else None
        extra["match_kind"] = "deep_equal_json"
        if subfamily:
            extra["subfamily"] = subfamily

    elif verifier_type == "puzzle_match":
        # reasoning_gym row: keep ground_truth + rg metadata in payload
        gt = vi.get("ground_truth")
        primary = str(gt).strip() if gt is not None else None
        if subfamily:
            extra["game_type"] = subfamily
        rg_entry = _maybe_parse_nested_json(vi.get("reasoning_gym_entry"))
        if rg_entry is not None:
            extra["reasoning_gym_entry"] = rg_entry

    elif verifier_type == "schema_pydantic":
        schema_code = vi.get("pydantic_config")
        model_name = vi.get("model_name")
        if not schema_code:
            return (None, None)
        extra["schema_code"] = schema_code
        if model_name:
            extra["model_name"] = model_name
        # ground_truth_text is not used for schema_* verifiers; intentionally None.

    info_json = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return primary, info_json


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class FilterConfig:
    min_prompt_chars: int = 30
    drop_prompt_http: bool = True
    drop_prompt_images: bool = True
    enable_prompt_repetition_filter: bool = True
    enable_fasttext_english_filter: bool = False
    fasttext_model_path: str = "/home/workdir/Master_Thesis/models/fasttext/lid.176.bin"
    fasttext_threshold: float = 0.80
    fasttext_min_chars: int = 80
    fasttext_keep_if_too_short: bool = True

    check_gt_placeholder: bool = True
    min_gt_chars: int = 1

    # Pass-rate gate: average across the three reliably-populated reward models.
    enable_passrate_filter: bool = True
    max_passrate: float = 0.90
    drop_passrate_zero: bool = False


# ── Schema & empty output ───────────────────────────────────────────────────


_EMPTY_TEMPLATE: Dict[str, Any] = {
    "_keep": False,
    "_drop_reason": None,
    "dataset_id": None,
    "dataset_version_date": None,
    "example_id": None,
    "row_id": None,
    "subsource_raw": None,
    "source_dataset_id": None,
    "license": None,
    "used_by_model": None,
    "context_messages": None,
    "language": "en",
    "domain": None,
    "ability": None,
    "difficulty": None,
    "verifier_type": None,
    "verifier_source": "prime_intellect",
    "ground_truth_text": None,
    "verification_info_raw": None,
    "avg_reward": None,
    "reward_model_metadata": None,
}


def _make_empty_out() -> Dict[str, Any]:
    return dict(_EMPTY_TEMPLATE)


_MAPPED_FEATURES = Features({
    "_keep": Value("bool"),
    "_drop_reason": Value("string"),
    "dataset_id": Value("string"),
    "dataset_version_date": Value("string"),
    "example_id": Value("string"),
    "row_id": Value("string"),
    "subsource_raw": Value("string"),
    "source_dataset_id": Value("string"),
    "license": Value("string"),
    "used_by_model": Value("string"),
    "context_messages": [{
        "role": Value("string"),
        "content": Value("string"),
    }],
    "language": Value("string"),
    "domain": Value("string"),
    "ability": Value("string"),
    "difficulty": Value("string"),
    "verifier_type": Value("string"),
    "verifier_source": Value("string"),
    "ground_truth_text": Value("string"),
    "verification_info_raw": Value("string"),
    "avg_reward": Value("float32"),
    "reward_model_metadata": Value("string"),
})


_REWARD_COLS = (
    "Qwen/Qwen3-32B_avg_reward",
    "Qwen/Qwen3-4B_avg_reward",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B_avg_reward",
)


def _avg_reward(ex: Dict[str, Any]) -> Optional[float]:
    vals = []
    for k in _REWARD_COLS:
        v = ex.get(k)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    if not vals:
        return None
    return sum(vals) / len(vals)


# ── Core mapping ────────────────────────────────────────────────────────────


def map_row(
    ex: Dict[str, Any],
    idx: int,
    *,
    split: str,
    dataset_id: str,
    dataset_version: Optional[str],
    license_str: Optional[str],
    used_by_model: str,
    cfg: FilterConfig,
) -> Dict[str, Any]:
    out = _make_empty_out()

    problem_id = ex.get("problem_id")
    row_id = problem_id if problem_id else f"row{idx}"

    out["dataset_id"] = dataset_id
    out["dataset_version_date"] = dataset_version
    out["row_id"] = row_id
    out["license"] = license_str
    out["used_by_model"] = used_by_model
    out["subsource_raw"] = problem_id

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt_raw = ex.get("prompt")
    if not isinstance(prompt_raw, str) or not prompt_raw.strip():
        out["_drop_reason"] = "missing_prompt"
        return out
    prompt_text = prompt_raw.strip()
    context_messages = [{"role": "user", "content": prompt_text}]

    reason = should_drop_prompt_length(prompt_text, min_chars=cfg.min_prompt_chars)
    if reason:
        out["_drop_reason"] = reason
        return out

    reason = should_drop_prompt_references(
        prompt_text,
        drop_if_has_http=cfg.drop_prompt_http,
        drop_if_has_images=cfg.drop_prompt_images,
    )
    if reason:
        out["_drop_reason"] = reason
        return out

    if cfg.enable_prompt_repetition_filter:
        reason = should_drop_prompt_repetition(prompt_text)
        if reason:
            out["_drop_reason"] = reason
            return out

    if cfg.enable_fasttext_english_filter:
        reason = should_drop_non_english_prompt(
            prompt_text,
            model_path=cfg.fasttext_model_path,
            cfg=RLEnglishLidConfig(
                threshold=cfg.fasttext_threshold,
                min_chars=cfg.fasttext_min_chars,
                keep_if_too_short=cfg.fasttext_keep_if_too_short,
            ),
        )
        if reason:
            out["_drop_reason"] = reason
            return out

    # ── Classification ────────────────────────────────────────────────────
    vi = _parse_verification_info(ex.get("verification_info"))
    if vi is None:
        out["_drop_reason"] = "invalid_verification_info"
        return out

    domain, verifier_type, ability, subfamily = _classify_row(problem_id, vi)
    if verifier_type is None:
        out["_drop_reason"] = "unknown_verifier"
        return out

    gt_text, verif_info = _build_verifier_payload(verifier_type, vi, subfamily, problem_id)

    # GT quality gate
    reason = should_drop_ground_truth_for_verifier(
        gt_text,
        verifier_type=verifier_type,
        verification_info_raw=verif_info,
        check_placeholder=cfg.check_gt_placeholder,
        min_gt_chars=cfg.min_gt_chars,
    )
    if reason:
        out["_drop_reason"] = reason
        return out

    # Pass-rate gate (averaged across the three populated reward columns)
    avg_r = _avg_reward(ex)
    if cfg.enable_passrate_filter:
        reason = should_drop_by_passrate(
            avg_r,
            max_passrate=cfg.max_passrate,
            drop_if_zero=cfg.drop_passrate_zero,
        )
        if reason:
            out["_drop_reason"] = reason
            return out

    example_id = _stable_sha256(f"{dataset_id}|{split}|{idx}|{problem_id}")

    out.update({
        "_keep": True,
        "_drop_reason": None,
        "example_id": example_id,
        "context_messages": context_messages,
        "language": "en",
        "domain": domain,
        "ability": ability,
        "difficulty": None,
        "verifier_type": verifier_type,
        "verifier_source": "prime_intellect",
        "ground_truth_text": gt_text,
        "verification_info_raw": verif_info,
        "avg_reward": float(avg_r) if avg_r is not None else None,
        "reward_model_metadata": json.dumps(
            {"models": list(_REWARD_COLS), "aggregation": "mean"},
            ensure_ascii=False, sort_keys=True,
        ),
        "source_dataset_id": ability,
    })
    return out


def map_only(ds: Dataset, **kw) -> Dataset:
    cfg = kw["cfg"]
    dataset_id = kw["dataset_id"]
    dataset_version = kw["dataset_version"]
    license_str = kw["license_str"]
    used_by_model = kw["used_by_model"]
    split = kw["split"]

    def _row(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return map_row(
            ex, idx,
            split=split,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            license_str=license_str,
            used_by_model=used_by_model,
            cfg=cfg,
        )

    return ds.map(
        _row,
        with_indices=True,
        features=_MAPPED_FEATURES,
        remove_columns=ds.column_names,
    )


def filter_kept(mapped: Dataset) -> Dataset:
    kept = mapped.filter(lambda x: x["_keep"])
    return kept.remove_columns(["_keep", "_drop_reason"])


def filter_dropped(mapped: Dataset) -> Dataset:
    dropped = mapped.filter(lambda x: not x["_keep"])
    return dropped.remove_columns(["_keep"])


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter + normalize PrimeIntellect/SYNTHETIC-2-RL to RL v1 schema."
    )
    parser.add_argument(
        "--input_dir",
        default="/home/workdir/Master_Thesis/Datasets/rl/PrimeIntellect__SYNTHETIC-2-RL/train",
    )
    parser.add_argument(
        "--output_parquet",
        default="/home/workdir/Master_Thesis/corpora/rl/synthetic2_rl/rl_synthetic2_rl_v1.kept.parquet",
    )
    parser.add_argument(
        "--output_parquet_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/synthetic2_rl/rl_synthetic2_rl_v1.dropped.parquet",
    )
    parser.add_argument(
        "--output_jsonl_sample",
        default="/home/workdir/Master_Thesis/corpora/rl/synthetic2_rl/kept.sample.jsonl",
    )
    parser.add_argument(
        "--output_jsonl_sample_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/synthetic2_rl/dropped.sample.jsonl",
    )
    parser.add_argument(
        "--output_report",
        default="/home/workdir/Master_Thesis/corpora/rl/synthetic2_rl/FILTERING_REPORT.md",
    )
    parser.add_argument("--test_n", type=int, default=0)
    parser.add_argument("--sample_n", type=int, default=200)
    parser.add_argument("--min_prompt_chars", type=int, default=30)
    parser.add_argument("--drop_prompt_http", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_prompt_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_prompt_repetition_filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_fasttext_english_filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable_passrate_filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_passrate", type=float, default=0.90)
    parser.add_argument("--drop_passrate_zero", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    dataset_id = "PrimeIntellect/SYNTHETIC-2-RL"
    license_str = "apache-2.0"
    used_by_model = "SYNTHETIC-2-RL-Reasoner"  # informal
    split = "train"
    dataset_version = datetime.now(timezone.utc).date().isoformat()

    cfg = FilterConfig(
        min_prompt_chars=args.min_prompt_chars,
        drop_prompt_http=args.drop_prompt_http,
        drop_prompt_images=args.drop_prompt_images,
        enable_prompt_repetition_filter=args.enable_prompt_repetition_filter,
        enable_fasttext_english_filter=args.enable_fasttext_english_filter,
        enable_passrate_filter=args.enable_passrate_filter,
        max_passrate=args.max_passrate,
        drop_passrate_zero=args.drop_passrate_zero,
    )

    print(f"Loading dataset from: {input_dir}")
    ds = load_from_disk(str(input_dir))
    print(ds)

    if args.test_n and args.test_n > 0:
        n = min(args.test_n, len(ds))
        print(f"TEST MODE: limiting to {n:,} rows")
        ds = ds.select(range(n))

    total_in = len(ds)
    print(f"FastText available: {is_fasttext_available()}")
    print("Mapping + filtering...")
    mapped = map_only(
        ds,
        split=split, dataset_id=dataset_id, dataset_version=dataset_version,
        license_str=license_str, used_by_model=used_by_model, cfg=cfg,
    )

    kept = filter_kept(mapped)
    out_kept = Path(args.output_parquet)
    out_kept.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing kept parquet: {out_kept}")
    kept.to_parquet(str(out_kept))
    print(f"  -> {len(kept):,} kept rows")

    dropped = filter_dropped(mapped)
    out_dropped = Path(args.output_parquet_dropped)
    out_dropped.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing dropped parquet: {out_dropped}")
    dropped.to_parquet(str(out_dropped))
    print(f"  -> {len(dropped):,} dropped rows")

    if args.sample_n > 0:
        for rows, path_str in ((kept, args.output_jsonl_sample), (dropped, args.output_jsonl_sample_dropped)):
            if len(rows) == 0: continue
            sample = rows.select(range(min(args.sample_n, len(rows))))
            p = Path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                for ex in sample:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Wrote sample: {p}")

    from collections import Counter
    drop_counts = Counter()
    if "_drop_reason" in dropped.column_names and len(dropped) > 0:
        drop_counts = Counter(r for r in dropped["_drop_reason"] if r)
        print("\nDrop reason summary:")
        for reason, count in drop_counts.most_common():
            print(f"  {reason:40s} {count:>7,}")

    vt_counts = Counter(); dom_counts = Counter(); ab_counts = Counter()
    if len(kept) > 0:
        vt_counts = Counter(kept["verifier_type"])
        dom_counts = Counter(kept["domain"])
        ab_counts = Counter(kept["ability"])
        print("\nVerifier type summary (kept):")
        for k, v in vt_counts.most_common():
            print(f"  {k:30s} {v:>7,}")
        print("\nDomain summary (kept):")
        for k, v in dom_counts.most_common():
            print(f"  {k:30s} {v:>7,}")
        print("\nAbility summary (kept):")
        for k, v in ab_counts.most_common():
            print(f"  {k:30s} {v:>7,}")

    # FILTERING_REPORT.md
    report = Path(args.output_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# PrimeIntellect/SYNTHETIC-2-RL — Filtering Report\n\n")
        f.write(f"- **dataset_id:** `{dataset_id}`\n")
        f.write(f"- **license:** `{license_str}`\n")
        f.write(f"- **dataset_version_date:** `{dataset_version}`\n")
        f.write(f"- **schema target:** `rl_schema_v1`\n")
        f.write(f"- **verifier_source:** `prime_intellect`\n\n")
        f.write("## Counts\n\n| Metric | Count |\n|---|---|\n")
        f.write(f"| Input rows | {total_in:,} |\n")
        f.write(f"| Kept | {len(kept):,} |\n")
        f.write(f"| Dropped | {len(dropped):,} |\n")
        if total_in:
            f.write(f"| Keep rate | {len(kept)/total_in*100:.2f}% |\n")
        f.write("\n")
        for title, counter in (
            ("Drop reasons", drop_counts),
            ("verifier_type (kept)", vt_counts),
            ("domain (kept)", dom_counts),
            ("ability (kept)", ab_counts),
        ):
            if counter:
                f.write(f"## {title}\n\n| key | count |\n|---|---|\n")
                for k, v in counter.most_common():
                    f.write(f"| `{k}` | {v:,} |\n")
                f.write("\n")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
