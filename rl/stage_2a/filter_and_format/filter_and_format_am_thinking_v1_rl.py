"""Filter + normalize `a-m-team/AM-Thinking-v1-RL-Dataset` into RL v1 schema.

Upstream is verl-standard:
  data_source, prompt (list[{role,content}]), ability ('math'|'code'),
  reward_model {ground_truth: str, style: 'rule'}, extra_info {index, split}

Two verifier families only — no LLM-judge / IFEval / MCQA in this dataset:

| ability | reward_model.ground_truth.call_type | verifier_type | domain |
|---------|-------------------------------------|---------------|--------|
| math    | (raw string)                        | math_equiv    | math   |
| code    | assert                              | code_asserts  | code   |
| code    | std                                 | code_stdio    | code   |

`verifier_source = "open_instruct"` per rl_schema_v1 §3.1
(AllenAI ground_truth_utils, shared with Dolci-Think-RL-7B).

Output: corpora/rl/am_thinking_v1_rl/rl_am_thinking_v1_rl_v1.{kept,dropped}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from Filtering_Pipeline.rl.stage_2a.filters.english_filter import (
    RLEnglishLidConfig,
    is_fasttext_available,
    should_drop_non_english_prompt,
)
from Filtering_Pipeline.sft.stage_2a.filters.content_filters import has_cutoff_mention


# ── Helpers ─────────────────────────────────────────────────────────────────

def _stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_to_context_messages(prompt_raw: Any) -> List[Dict[str, str]]:
    """AM-Thinking prompts are already list[{role,content}]; normalise + filter."""
    if not isinstance(prompt_raw, list):
        return []
    out: List[Dict[str, str]] = []
    for msg in prompt_raw:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if role and content:
            out.append({"role": role, "content": content})
    return out


def _messages_to_flat_text(messages: List[Dict[str, str]]) -> str:
    """Join messages for prompt-hygiene filters."""
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n---\n\n".join(parts)


def _user_only_text(messages: List[Dict[str, str]]) -> str:
    """Pull just user turns for LID / cutoff checks (the system prompt is boilerplate)."""
    parts = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
    return "\n\n".join(parts)


def _parse_reward_model(rm: Any) -> Dict[str, Any]:
    if isinstance(rm, dict):
        return rm
    if isinstance(rm, str):
        try:
            return json.loads(rm)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _classify_row(ability: Any, gt_struct: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Map (ability, code GT call_type) → (domain, verifier_type)."""
    if not isinstance(ability, str):
        return (None, None)
    a = ability.strip().lower()
    if a == "math":
        return ("math", "math_equiv")
    if a == "code":
        # Sub-split by call_type from the parsed code GT struct.
        ct = (gt_struct or {}).get("call_type")
        if ct == "assert":
            return ("code", "code_asserts")
        if ct == "std":
            return ("code", "code_stdio")
        # Code row without a parseable GT struct → unknown call_type
        return ("code", None)
    return (None, None)


def _build_verifier_payload(
    verifier_type: str,
    gt_raw: Any,
    gt_struct: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (ground_truth_text, verification_info_raw_json)."""
    extra: Dict[str, Any] = {}
    primary: Optional[str] = None

    if verifier_type == "math_equiv":
        # Raw string answer (e.g. "90\\%", "69375"). Keep verbatim.
        if isinstance(gt_raw, str):
            primary = gt_raw.strip() or None
        elif gt_raw is not None:
            primary = str(gt_raw).strip() or None
        # no extra info

    elif verifier_type == "code_asserts":
        # ground_truth = JSON-string of assert_case list (one assert per item).
        asserts = list((gt_struct or {}).get("assert_case") or [])
        if asserts:
            primary = json.dumps(asserts, ensure_ascii=False)
        extra["test_format"] = "asserts"
        extra["language"] = "python"
        fn_name = (gt_struct or {}).get("fn_name")
        if fn_name:
            extra["fn_name"] = fn_name
        extra["num_tests"] = len(asserts)

    elif verifier_type == "code_stdio":
        inputs = list((gt_struct or {}).get("inputs") or [])
        outputs = list((gt_struct or {}).get("outputs") or [])
        if inputs and outputs:
            extra["unit_tests"] = {"inputs": inputs, "outputs": outputs}
            # GT-quality gate needs a non-empty primary string → joined outputs (truncated).
            primary = "\n---\n".join(str(o) for o in outputs)[:8000] or None
        extra["test_format"] = "stdio"
        extra["language"] = "python"
        extra["num_tests"] = len(inputs)

    info_json = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return primary, info_json


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    min_prompt_chars: int = 30

    drop_prompt_http: bool = True
    drop_prompt_images: bool = True

    enable_prompt_repetition_filter: bool = True
    repetition_min_sentence_repeats: int = 5
    repetition_phrase_n: int = 4
    repetition_min_phrase_repeats: int = 15
    repetition_min_chars: int = 80

    enable_fasttext_english_filter: bool = False
    fasttext_model_path: str = "/home/workdir/Master_Thesis/models/fasttext/lid.176.bin"
    fasttext_threshold: float = 0.80
    fasttext_min_chars: int = 80
    fasttext_keep_if_too_short: bool = True

    check_cutoff_mention: bool = True

    check_gt_placeholder: bool = True
    min_gt_chars: int = 1


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
    "verifier_source": "open_instruct",
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

    # row_id: prefer upstream extra_info.index combined with positional idx
    # (extra_info.index is 0 for many rows so it's not unique on its own).
    ei = ex.get("extra_info") or {}
    upstream_idx = ei.get("index") if isinstance(ei, dict) else None
    row_id = f"{upstream_idx}-{idx}" if upstream_idx is not None else str(idx)

    out["dataset_id"] = dataset_id
    out["dataset_version_date"] = dataset_version
    out["row_id"] = row_id
    out["license"] = license_str
    out["used_by_model"] = used_by_model

    data_source = ex.get("data_source")
    out["subsource_raw"] = data_source if isinstance(data_source, str) else None
    out["source_dataset_id"] = data_source if isinstance(data_source, str) else None

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt_raw = ex.get("prompt")
    if not prompt_raw:
        out["_drop_reason"] = "missing_prompt"
        return out

    context_messages = _prompt_to_context_messages(prompt_raw)
    if not context_messages:
        out["_drop_reason"] = "empty_context_messages"
        return out

    flat_text = _messages_to_flat_text(context_messages)
    user_text = _user_only_text(context_messages) or flat_text

    # [A] prompt length (uses user-only text — system prompt is boilerplate)
    reason = should_drop_prompt_length(user_text, min_chars=cfg.min_prompt_chars)
    if reason:
        out["_drop_reason"] = reason
        return out

    # [B] HTTP / image references
    reason = should_drop_prompt_references(
        flat_text,
        drop_if_has_http=cfg.drop_prompt_http,
        drop_if_has_images=cfg.drop_prompt_images,
    )
    if reason:
        out["_drop_reason"] = reason
        return out

    # [C] prompt repetition
    if cfg.enable_prompt_repetition_filter:
        reason = should_drop_prompt_repetition(
            flat_text,
            min_sentence_repeats=cfg.repetition_min_sentence_repeats,
            phrase_n=cfg.repetition_phrase_n,
            min_phrase_repeats=cfg.repetition_min_phrase_repeats,
            min_chars=cfg.repetition_min_chars,
        )
        if reason:
            out["_drop_reason"] = reason
            return out

    # [D] FastText English LID
    if cfg.enable_fasttext_english_filter:
        reason = should_drop_non_english_prompt(
            user_text,
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

    # [E] knowledge-cutoff mention (user turn only)
    if cfg.check_cutoff_mention:
        if has_cutoff_mention([{"role": "user", "content": user_text}]):
            out["_drop_reason"] = "cutoff_mention"
            return out

    # ── Classify verifier ─────────────────────────────────────────────────
    ability = ex.get("ability")
    rm = _parse_reward_model(ex.get("reward_model"))
    gt_raw = rm.get("ground_truth")

    # For ability=code the GT is a JSON string with call_type. Parse once here.
    gt_struct: Optional[Dict[str, Any]] = None
    if isinstance(ability, str) and ability.strip().lower() == "code":
        if isinstance(gt_raw, str):
            try:
                parsed = json.loads(gt_raw)
                if isinstance(parsed, dict):
                    gt_struct = parsed
            except (json.JSONDecodeError, TypeError):
                gt_struct = None

    domain, verifier_type = _classify_row(ability, gt_struct)
    if verifier_type is None:
        out["_drop_reason"] = "unknown_verifier"
        return out

    # ── Build verifier payload ────────────────────────────────────────────
    gt_text, verif_info = _build_verifier_payload(verifier_type, gt_raw, gt_struct)

    # [F] GT quality gate (verifier-aware: stdio/asserts have non-empty primary)
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

    # ── Finalise ──────────────────────────────────────────────────────────
    example_id = _stable_sha256(f"{dataset_id}|{split}|{idx}")

    out.update({
        "_keep": True,
        "_drop_reason": None,
        "example_id": example_id,
        "context_messages": context_messages,
        "language": "en",
        "domain": domain,
        "ability": ability.strip().lower() if isinstance(ability, str) else None,
        "difficulty": None,
        "verifier_type": verifier_type,
        "verifier_source": "open_instruct",
        "ground_truth_text": gt_text,
        "verification_info_raw": verif_info,
        "avg_reward": None,
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
        description="Filter + normalize a-m-team/AM-Thinking-v1-RL-Dataset to RL v1 schema."
    )
    parser.add_argument(
        "--input_dir",
        default="/home/workdir/Master_Thesis/Datasets/rl/a-m-team__AM-Thinking-v1-RL-Dataset/train",
    )
    parser.add_argument(
        "--output_parquet",
        default="/home/workdir/Master_Thesis/corpora/rl/am_thinking_v1_rl/rl_am_thinking_v1_rl_v1.kept.parquet",
    )
    parser.add_argument(
        "--output_parquet_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/am_thinking_v1_rl/rl_am_thinking_v1_rl_v1.dropped.parquet",
    )
    parser.add_argument(
        "--output_jsonl_sample",
        default="/home/workdir/Master_Thesis/corpora/rl/am_thinking_v1_rl/kept.sample.jsonl",
    )
    parser.add_argument(
        "--output_jsonl_sample_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/am_thinking_v1_rl/dropped.sample.jsonl",
    )
    parser.add_argument(
        "--output_report",
        default="/home/workdir/Master_Thesis/corpora/rl/am_thinking_v1_rl/FILTERING_REPORT.md",
    )
    parser.add_argument("--test_n", type=int, default=0)
    parser.add_argument("--sample_n", type=int, default=200)

    parser.add_argument("--min_prompt_chars", type=int, default=30)
    parser.add_argument("--drop_prompt_http",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_prompt_images",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_prompt_repetition_filter",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_fasttext_english_filter",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fasttext_model_path",
                        default="/home/workdir/Master_Thesis/models/fasttext/lid.176.bin")
    parser.add_argument("--fasttext_threshold", type=float, default=0.80)
    parser.add_argument("--check_cutoff_mention",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_gt_placeholder",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_gt_chars", type=int, default=1)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    dataset_id = "a-m-team/AM-Thinking-v1-RL-Dataset"
    license_str = "apache-2.0"
    used_by_model = "AM-Thinking-v1"
    split = "train"
    dataset_version = datetime.now(timezone.utc).date().isoformat()

    cfg = FilterConfig(
        min_prompt_chars=args.min_prompt_chars,
        drop_prompt_http=args.drop_prompt_http,
        drop_prompt_images=args.drop_prompt_images,
        enable_prompt_repetition_filter=args.enable_prompt_repetition_filter,
        enable_fasttext_english_filter=args.enable_fasttext_english_filter,
        fasttext_model_path=args.fasttext_model_path,
        fasttext_threshold=args.fasttext_threshold,
        check_cutoff_mention=args.check_cutoff_mention,
        check_gt_placeholder=args.check_gt_placeholder,
        min_gt_chars=args.min_gt_chars,
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
    print(f"  -> {len(kept):,} kept rows")

    dropped = filter_dropped(mapped)
    output_dropped = Path(args.output_parquet_dropped)
    output_dropped.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing dropped parquet: {output_dropped}")
    dropped.to_parquet(str(output_dropped))
    print(f"  -> {len(dropped):,} dropped rows")

    # ── Samples ──────────────────────────────────────────────────────────
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

    # ── Console stats ────────────────────────────────────────────────────
    from collections import Counter

    drop_counts: Counter = Counter()
    if "_drop_reason" in dropped.column_names and len(dropped) > 0:
        drop_counts = Counter(r for r in dropped["_drop_reason"] if r)
        print("\nDrop reason summary:")
        for reason, count in drop_counts.most_common():
            print(f"  {reason:40s} {count:>6,}")

    vt_counts: Counter = Counter()
    dom_counts: Counter = Counter()
    ability_counts: Counter = Counter()
    source_counts: Counter = Counter()
    if len(kept) > 0:
        vt_counts = Counter(kept["verifier_type"])
        dom_counts = Counter(kept["domain"])
        ability_counts = Counter(kept["ability"])
        source_counts = Counter(kept["source_dataset_id"])
        print("\nVerifier type summary (kept):")
        for k, v in vt_counts.most_common():
            print(f"  {k:40s} {v:>6,}")
        print("\nDomain summary (kept):")
        for k, v in dom_counts.most_common():
            print(f"  {k:40s} {v:>6,}")

    # ── FILTERING_REPORT.md ─────────────────────────────────────────────
    report = Path(args.output_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# AM-Thinking-v1-RL-Dataset — Filtering Report\n\n")
        f.write(f"- **dataset_id:** `{dataset_id}`\n")
        f.write(f"- **license:** `{license_str}`\n")
        f.write(f"- **dataset_version_date:** `{dataset_version}`\n")
        f.write(f"- **schema target:** `rl_schema_v1`\n")
        f.write(f"- **verifier_source:** `open_instruct`\n\n")
        f.write("## Counts\n\n")
        f.write(f"| Metric | Count |\n|---|---|\n")
        f.write(f"| Input rows | {total_in:,} |\n")
        f.write(f"| Kept | {len(kept):,} |\n")
        f.write(f"| Dropped | {len(dropped):,} |\n")
        if total_in:
            f.write(f"| Keep rate | {len(kept)/total_in*100:.2f}% |\n")
        f.write("\n")
        if drop_counts:
            f.write("## Drop reasons\n\n| Reason | Count |\n|---|---|\n")
            for reason, count in drop_counts.most_common():
                f.write(f"| `{reason}` | {count:,} |\n")
            f.write("\n")
        if vt_counts:
            f.write("## verifier_type distribution (kept)\n\n| verifier_type | Count |\n|---|---|\n")
            for k, v in vt_counts.most_common():
                f.write(f"| `{k}` | {v:,} |\n")
            f.write("\n")
        if dom_counts:
            f.write("## domain distribution (kept)\n\n| domain | Count |\n|---|---|\n")
            for k, v in dom_counts.most_common():
                f.write(f"| `{k}` | {v:,} |\n")
            f.write("\n")
        if ability_counts:
            f.write("## ability distribution (kept)\n\n| ability | Count |\n|---|---|\n")
            for k, v in ability_counts.most_common():
                f.write(f"| `{k}` | {v:,} |\n")
            f.write("\n")
        if source_counts:
            f.write("## source_dataset_id distribution (kept, top 30)\n\n| source | Count |\n|---|---|\n")
            for k, v in source_counts.most_common(30):
                f.write(f"| `{k}` | {v:,} |\n")
            f.write("\n")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
