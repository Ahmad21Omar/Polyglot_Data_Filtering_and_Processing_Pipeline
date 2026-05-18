"""Filter + normalize `nvidia/Nemotron-3-Nano-RL-Training-Blend` into RL v1 schema.

The Nemotron blend ships seven sub-corpora under one HF dataset.  Each sub-corpus
is identified by the `dataset` column and pre-paired with a NeMo Gym agent name
in `agent_ref.name`.  We map each sub-corpus to a concrete RL v1 verifier_type:

| dataset                                       | verifier_type             | kept |
|-----------------------------------------------|---------------------------|------|
| nano_v3_sft_profiled_stem_mcqa                | multiple_choice           |  ✔   |
| nano_v3_sft_profiled_instruction_following    | if_rules                  |  ✔   |
| nano_v3_sft_profiled_comp_coding_50tests      | code_stdio                |  ✔   |
| nano_v3_sft_profiled_structured_outputs       | schema_structured_outputs |  ✔   |
| nano_v3_sft_profiled_dapo17k                  | math_with_judge           |  ✘   |
| nano_v3_sft_profiled_skywork_no_omni          | math_with_judge           |  ✘   |
| nano_v3_sft_profiled_workbench                | tool_calls                |  ✘   |

Drop rationale:
  * `math_with_judge` — both math sub-corpora ship `_hf_placeholder` rows only
    (the actual prompt + reference answer must be re-fetched from the original
    HuggingFace datasets) AND require an LLM judge for scoring.  Two independent
    blockers, drop reason `placeholder_only_prompt`.
  * `tool_calls` — the workbench sub-corpus needs the full WorkBench tool
    environment (Email/Calendar/CRM mocks) to grade tool-call sequences.  Out
    of scope for the verifier-only NeMo Gym setup; drop reason
    `ungradeable_verifier_tool_calls`.

Output:
  Master_Thesis/corpora/rl/nemotron_3_nano_rl_blend/
    - rl_nemotron_3_nano_rl_blend_v1.kept.parquet
    - rl_nemotron_3_nano_rl_blend_v1.dropped.parquet
    - kept.sample.jsonl, dropped.sample.jsonl

Filter stages (heuristic):
  [A] prompt_length     [B] prompt_references     [C] unknown_verifier
  [D] english LID (opt) [E] ungradeable_verifier  [F] ground_truth_quality
  [G] passrate
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

from datasets import Dataset, Features, Sequence, Value, load_from_disk

from Filtering_Pipeline.rl.stage_2a.filters.prompt_length_filter import (
    should_drop_prompt_length,
)
from Filtering_Pipeline.rl.stage_2a.filters.prompt_reference_filter import (
    should_drop_prompt_references,
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


def _input_to_context_messages(rcp: Any) -> List[Dict[str, str]]:
    """Extract `responses_create_params.input` → list[{role,content}]."""
    if not isinstance(rcp, dict):
        return []
    inp = rcp.get("input")
    if not isinstance(inp, list):
        return []
    out: List[Dict[str, str]] = []
    for m in inp:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if role and content:
            out.append({"role": role, "content": content})
    return out


def _messages_to_flat_text(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n---\n\n".join(parts)


# Nemotron `dataset` label → (our domain, our verifier_type, ability)
_NEMOTRON_VERIFIER_MAP: Dict[str, Tuple[str, str, str]] = {
    "nano_v3_sft_profiled_stem_mcqa":             ("mcqa",                  "multiple_choice",           "stem_mcqa"),
    "nano_v3_sft_profiled_instruction_following": ("instruction_following", "if_rules",                  "ifeval"),
    "nano_v3_sft_profiled_comp_coding_50tests":   ("code",                  "code_stdio",                "competitive_programming"),
    "nano_v3_sft_profiled_structured_outputs":    ("structured_outputs",    "schema_structured_outputs", "json_schema"),
    "nano_v3_sft_profiled_dapo17k":               ("math",                  "math_with_judge",           "dapo_math"),
    "nano_v3_sft_profiled_skywork_no_omni":       ("math",                  "math_with_judge",           "skywork_math"),
    "nano_v3_sft_profiled_workbench":             ("tool_use",              "tool_calls",                "workbench"),
}

# Verifier types we cannot grade in the NeMo Gym setup we ship.
_UNGRADEABLE_VERIFIER_TYPES: set = {"math_with_judge", "tool_calls"}

# Sub-datasets that consist entirely of `_hf_placeholder` rows (the actual
# prompt and answer live in the upstream HF dataset and were not materialised
# into this blend).  Detected up-front so we can use a sharper drop reason.
_PLACEHOLDER_ONLY_DATASETS: frozenset = frozenset({
    "nano_v3_sft_profiled_dapo17k",
    "nano_v3_sft_profiled_skywork_no_omni",
})


def _classify_row(dataset_label: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not isinstance(dataset_label, str):
        return (None, None, None)
    return _NEMOTRON_VERIFIER_MAP.get(dataset_label.strip(), (None, None, None))


def _normalize_mcqa_options(options_raw: Any) -> Dict[str, str]:
    """Nemotron stores options as list[dict[letter→str|None]] with one filled key per dict.

    Flatten to {letter: text}.
    """
    flat: Dict[str, str] = {}
    if not isinstance(options_raw, list):
        return flat
    for slot in options_raw:
        if not isinstance(slot, dict):
            continue
        for letter, text in slot.items():
            if text is None:
                continue
            text = str(text).strip()
            if text:
                flat[str(letter).strip()] = text
    return flat


def _normalize_kwargs(kwargs_raw: Any) -> List[Dict[str, Any]]:
    """Strip None values from each per-instruction kwargs dict."""
    if not isinstance(kwargs_raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for k in kwargs_raw:
        if isinstance(k, dict):
            out.append({kk: vv for kk, vv in k.items() if vv is not None})
        else:
            out.append({})
    return out


def _build_verifier_payload(
    verifier_type: str,
    ex: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (ground_truth_text, verification_info_raw_json) for a Nemotron row."""
    extra: Dict[str, Any] = {}
    primary: Optional[str] = None

    if verifier_type == "multiple_choice":
        # expected_answer is the canonical letter (e.g. "F").
        primary = (ex.get("expected_answer") or "").strip() or None
        extra["options"] = _normalize_mcqa_options(ex.get("options"))
        tm = ex.get("template_metadata") or {}
        if isinstance(tm, dict):
            for k in ("format_type", "output_regex", "prompt_type", "template_id"):
                if tm.get(k):
                    extra[k] = tm[k]

    elif verifier_type == "if_rules":
        # Native IFEval format: list of instruction_ids + per-instruction kwargs.
        ids = ex.get("instruction_id_list") or []
        kwargs = _normalize_kwargs(ex.get("kwargs"))
        # ground_truth_text mirrors the dolci convention: a JSON string of the
        # constraint list, so generic verifiers can parse it directly.
        gt_struct = [
            {"instruction_id": iid, "kwargs": (kwargs[i] if i < len(kwargs) else {})}
            for i, iid in enumerate(ids)
        ]
        primary = json.dumps(gt_struct, ensure_ascii=False, sort_keys=True) if gt_struct else None
        extra["constraint_format"] = "ifeval_native"
        extra["instruction_id_list"] = list(ids)
        extra["kwargs"] = kwargs

    elif verifier_type == "code_stdio":
        # verifier_metadata.unit_tests = {"inputs": [...], "outputs": [...]}
        vm = ex.get("verifier_metadata") or {}
        unit_tests = vm.get("unit_tests") if isinstance(vm, dict) else None
        if isinstance(unit_tests, dict):
            inputs = list(unit_tests.get("inputs") or [])
            outputs = list(unit_tests.get("outputs") or [])
            extra["unit_tests"] = {"inputs": inputs, "outputs": outputs}
            extra["test_format"] = "stdio"
            extra["language"] = "python"
            # Use joined outputs as ground_truth_text to satisfy GT-quality gate.
            primary = "\n---\n".join(str(o) for o in outputs)[:8000] or None
        if ex.get("source"):
            extra["origin_platform"] = ex["source"]

    elif verifier_type == "schema_structured_outputs":
        # Schema lives in schema_str (JSON-encoded JSON Schema).
        schema_str = ex.get("schema_str")
        if isinstance(schema_str, str) and schema_str.strip():
            try:
                # Validate JSON and re-serialise canonically.
                schema_obj = json.loads(schema_str)
                extra["schema_json"] = json.dumps(schema_obj, ensure_ascii=False, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                extra["schema_json"] = None
        if ex.get("schema_type"):
            extra["schema_type"] = ex["schema_type"]
        if ex.get("schema_fields_count") is not None:
            extra["schema_fields_count"] = int(ex["schema_fields_count"])
        # ground_truth_text is conventionally not used for schema verifiers
        # (the GT-quality gate routes around it via verification_info_raw).
        primary = None

    elif verifier_type == "math_with_judge":
        extra["judge_template"] = "math_quality"
        extra["placeholder_only"] = ex.get("_hf_placeholder")

    elif verifier_type == "tool_calls":
        # Tool-call ground truth comes as list[{name, arguments}].
        gt = ex.get("ground_truth")
        if isinstance(gt, list) and gt:
            primary = json.dumps(gt, ensure_ascii=False, sort_keys=True)
        extra["environment_name"] = ex.get("environment_name")

    info_json = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return primary, info_json


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    min_prompt_chars: int = 30

    drop_prompt_http: bool = True
    drop_prompt_images: bool = True

    enable_fasttext_english_filter: bool = False
    fasttext_model_path: str = "/home/workdir/Master_Thesis/models/fasttext/lid.176.bin"
    fasttext_threshold: float = 0.80
    fasttext_min_chars: int = 80
    fasttext_keep_if_too_short: bool = True

    check_gt_placeholder: bool = True
    min_gt_chars: int = 1

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
    "verifier_source": "nemo_gym",
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


# ── Core mapping function ───────────────────────────────────────────────────

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

    row_id = ex.get("uuid") or ex.get("hash_id") or ex.get("id")
    row_id = str(row_id) if row_id is not None else str(idx)

    out["dataset_id"] = dataset_id
    out["dataset_version_date"] = dataset_version
    out["row_id"] = row_id
    out["license"] = license_str
    out["used_by_model"] = used_by_model
    out["subsource_raw"] = ex.get("dataset")

    dataset_label = ex.get("dataset")

    # ── Early drop: placeholder-only sub-corpora ──────────────────────────
    if dataset_label in _PLACEHOLDER_ONLY_DATASETS:
        out["_drop_reason"] = "placeholder_only_prompt"
        return out

    # ── Prompt ────────────────────────────────────────────────────────────
    context_messages = _input_to_context_messages(ex.get("responses_create_params"))
    if not context_messages:
        out["_drop_reason"] = "empty_context_messages"
        return out

    flat_text = _messages_to_flat_text(context_messages)

    # [A] length
    reason = should_drop_prompt_length(flat_text, min_chars=cfg.min_prompt_chars)
    if reason:
        out["_drop_reason"] = reason
        return out

    # [B] references
    reason = should_drop_prompt_references(
        flat_text,
        drop_if_has_http=cfg.drop_prompt_http,
        drop_if_has_images=cfg.drop_prompt_images,
    )
    if reason:
        out["_drop_reason"] = reason
        return out

    # [D] English LID
    if cfg.enable_fasttext_english_filter:
        reason = should_drop_non_english_prompt(
            flat_text,
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

    # ── Classify verifier ─────────────────────────────────────────────────
    domain, verifier_type, ability_str = _classify_row(dataset_label)
    if verifier_type is None:
        out["_drop_reason"] = "unknown_verifier"
        return out

    if verifier_type in _UNGRADEABLE_VERIFIER_TYPES:
        out["_drop_reason"] = f"ungradeable_verifier_{verifier_type}"
        return out

    # ── Build verification payload ────────────────────────────────────────
    gt_text, verif_info = _build_verifier_payload(verifier_type, ex)

    # [F] GT quality gate — verifier-aware
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

    # [G] passrate
    if cfg.enable_passrate_filter:
        reason = should_drop_by_passrate(
            ex.get("pass_rate"),
            max_passrate=cfg.max_passrate,
            drop_if_zero=cfg.drop_passrate_zero,
        )
        if reason:
            out["_drop_reason"] = reason
            return out

    # ── Finalise ──────────────────────────────────────────────────────────
    pr_val = ex.get("pass_rate")
    avg_reward: Optional[float] = None
    if pr_val is not None:
        try:
            avg_reward = float(pr_val)
        except (TypeError, ValueError):
            avg_reward = None

    # Include the per-blend row index alongside row_id: in `comp_coding_50tests`
    # the upstream `hash_id` is shared across many distinct problems (each row
    # carries a different unit-test slice), so hash_id alone collides.
    example_id = _stable_sha256(f"{dataset_id}|{split}|{dataset_label}|{idx}|{row_id}")

    out.update({
        "_keep": True,
        "_drop_reason": None,
        "example_id": example_id,
        "source_dataset_id": ex.get("source"),
        "context_messages": context_messages,
        "language": "en",
        "domain": domain,
        "ability": ability_str,
        "difficulty": None,
        "verifier_type": verifier_type,
        "verifier_source": "nemo_gym",
        "ground_truth_text": gt_text,
        "verification_info_raw": verif_info,
        "avg_reward": avg_reward,
        "reward_model_metadata": None,
    })
    return out


# ── Map + filter wrappers ───────────────────────────────────────────────────

def map_only(
    ds: Dataset,
    *,
    split: str,
    dataset_id: str,
    dataset_version: Optional[str],
    license_str: Optional[str],
    used_by_model: str,
    cfg: FilterConfig,
) -> Dataset:
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
        description="Filter + normalize nvidia/Nemotron-3-Nano-RL-Training-Blend to RL v1 schema."
    )
    parser.add_argument(
        "--input_dir",
        default="/home/workdir/Master_Thesis/Datasets/rl/nvidia__Nemotron-3-Nano-RL-Training-Blend/train",
    )
    parser.add_argument(
        "--output_parquet",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_3_nano_rl_blend/rl_nemotron_3_nano_rl_blend_v1.kept.parquet",
    )
    parser.add_argument(
        "--output_parquet_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_3_nano_rl_blend/rl_nemotron_3_nano_rl_blend_v1.dropped.parquet",
    )
    parser.add_argument(
        "--output_jsonl_sample",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_3_nano_rl_blend/kept.sample.jsonl",
    )
    parser.add_argument(
        "--output_jsonl_sample_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_3_nano_rl_blend/dropped.sample.jsonl",
    )
    parser.add_argument("--test_n", type=int, default=0)
    parser.add_argument("--sample_n", type=int, default=200)

    parser.add_argument("--min_prompt_chars", type=int, default=30)
    parser.add_argument("--drop_prompt_http",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_prompt_images",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_fasttext_english_filter",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fasttext_model_path",
                        default="/home/workdir/Master_Thesis/models/fasttext/lid.176.bin")
    parser.add_argument("--fasttext_threshold", type=float, default=0.80)
    parser.add_argument("--check_gt_placeholder",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_gt_chars", type=int, default=1)
    parser.add_argument("--enable_passrate_filter",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_passrate", type=float, default=0.90)
    parser.add_argument("--drop_passrate_zero",
                        action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    dataset_id = "nvidia/Nemotron-3-Nano-RL-Training-Blend"
    license_str = "CC-BY-4.0"  # per Nvidia HF dataset card
    used_by_model = "Nemotron-3-Nano"
    split = "train"
    dataset_version = datetime.now(timezone.utc).date().isoformat()

    cfg = FilterConfig(
        min_prompt_chars=args.min_prompt_chars,
        drop_prompt_http=args.drop_prompt_http,
        drop_prompt_images=args.drop_prompt_images,
        enable_fasttext_english_filter=args.enable_fasttext_english_filter,
        fasttext_model_path=args.fasttext_model_path,
        fasttext_threshold=args.fasttext_threshold,
        check_gt_placeholder=args.check_gt_placeholder,
        min_gt_chars=args.min_gt_chars,
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

    print(f"FastText available: {is_fasttext_available()}")
    print("Mapping + filtering...")
    mapped = map_only(
        ds,
        split=split,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        license_str=license_str,
        used_by_model=used_by_model,
        cfg=cfg,
    )

    kept = filter_kept(mapped)
    output_parquet = Path(args.output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing kept parquet: {output_parquet}")
    kept.to_parquet(str(output_parquet))
    print(f"  → {len(kept):,} kept rows")

    dropped = filter_dropped(mapped)
    output_dropped = Path(args.output_parquet_dropped)
    output_dropped.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing dropped parquet: {output_dropped}")
    dropped.to_parquet(str(output_dropped))
    print(f"  → {len(dropped):,} dropped rows")

    if args.sample_n > 0:
        for rows, path_str in (
            (kept, args.output_jsonl_sample),
            (dropped, args.output_jsonl_sample_dropped),
        ):
            if len(rows) == 0:
                continue
            sample = rows.select(range(min(args.sample_n, len(rows))))
            p = Path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                for ex in sample:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"Wrote sample: {p}")

    if "_drop_reason" in dropped.column_names and len(dropped) > 0:
        from collections import Counter
        counts = Counter(r for r in dropped["_drop_reason"] if r)
        print("\nDrop reason summary:")
        for reason, count in counts.most_common():
            print(f"  {reason:40s} {count:>6,}")

    if "verifier_type" in kept.column_names and len(kept) > 0:
        from collections import Counter
        vt = Counter(kept["verifier_type"])
        print("\nVerifier type summary (kept):")
        for k, v in vt.most_common():
            print(f"  {k:40s} {v:>6,}")
        dom = Counter(kept["domain"])
        print("\nDomain summary (kept):")
        for k, v in dom.most_common():
            print(f"  {k:40s} {v:>6,}")


if __name__ == "__main__":
    main()
