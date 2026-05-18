"""Filter + normalize `nvidia/Nemotron-RL-ReasoningGym-v1` into RL v1 schema.

The dataset is a procedurally-generated mixture of 104 reasoning-gym tasks
(15 000 rows, ~144 per task).  Unlike the Nemotron-3-Nano blend, there is
**no explicit `verifier_type`** column — every row is conceptually scored by
the reasoning-gym `score_answer()` function for its `metadata.source_dataset`.

We collapse this into a single new RL v1 verifier_type, ``reasoning_gym``.
A future NeMo Gym resource server (`reasoning_gym/`) will dispatch on
``verification_info.source_dataset`` to the correct task scorer.

| metadata.source_dataset        | answer | kept |
|--------------------------------|--------|------|
| 99 single-answer tasks         | str    |  ✔   |
| rubiks_cube                    | None   |  ✘   |
| propositional_logic            | None   |  ✘   |
| graph_color                    | None   |  ✘   |
| rush_hour                      | None   |  ✘   |
| boxnet                         | None   |  ✘   |

Drop rationale:
  * Multi-solution tasks (`rubiks_cube`, `propositional_logic`, `graph_color`,
    `rush_hour`, `boxnet`) ship `answer = null` because any valid solution
    sequence works and string-equality is meaningless.  Grading them requires
    a task-specific simulator (the reasoning-gym Python lib).  Drop reason
    `ungradeable_verifier_reasoning_gym_multi_solution`.  Re-enable once a
    reasoning-gym backed NeMo Gym verifier exists.

Output:
  Master_Thesis/corpora/rl/nemotron_rl_reasoning_gym_v1/
    - rl_nemotron_rl_reasoning_gym_v1.kept.parquet
    - rl_nemotron_rl_reasoning_gym_v1.dropped.parquet
    - kept.sample.jsonl, dropped.sample.jsonl

Filter stages:
  [A] prompt_length     [B] prompt_references     [C] empty_context
  [D] english LID (opt) [E] ungradeable_multi_sol [F] ground_truth_quality
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, Features, Value, load_from_disk

from Filtering_Pipeline.rl.stage_2a.filters.prompt_length_filter import (
    should_drop_prompt_length,
)
from Filtering_Pipeline.rl.stage_2a.filters.prompt_reference_filter import (
    should_drop_prompt_references,
)
from Filtering_Pipeline.rl.stage_2a.filters.ground_truth_filter import (
    should_drop_ground_truth_for_verifier,
)
from Filtering_Pipeline.rl.stage_2a.filters.english_filter import (
    RLEnglishLidConfig,
    is_fasttext_available,
    should_drop_non_english_prompt,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_json_loads(s: Any) -> Any:
    if isinstance(s, str) and s:
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None
    return s if not isinstance(s, str) else None


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


# Multi-solution reasoning-gym tasks — `answer` is always None because any
# valid solution sequence is acceptable.  Need an in-process simulator (the
# reasoning-gym lib) to score, which we don't ship in this filter step.
_MULTI_SOLUTION_TASKS: frozenset = frozenset({
    "rubiks_cube",
    "propositional_logic",
    "graph_color",
    "rush_hour",
    "boxnet",
})

# Single verifier_type for every kept row.  Downstream NeMo Gym resource server
# dispatches on `verification_info.source_dataset` to the per-task scorer.
_VERIFIER_TYPE = "reasoning_gym"
_VERIFIER_SOURCE = "reasoning_gym"
_DOMAIN = "reasoning_gym"


def _build_verification_info(
    metadata: Dict[str, Any],
    *,
    source_dataset: str,
    uuid: str,
) -> str:
    """Build the verification_info_raw JSON for a reasoning-gym row.

    The full original metadata dict (all task-specific keys) is preserved so
    the future verifier can call ``reasoning_gym.score_answer(answer, entry={
        "metadata": <this dict>,
    })`` directly.
    """
    extra: Dict[str, Any] = {
        "source_dataset": source_dataset,
        "uuid": uuid,
        "metadata": metadata,
    }
    # Promote a few common keys to the top level for cheap downstream access.
    if "source_index" in metadata:
        extra["source_index"] = metadata["source_index"]
    if "difficulty" in metadata:
        extra["difficulty"] = metadata["difficulty"]
    if "task_type" in metadata:
        extra["task_type"] = metadata["task_type"]
    return json.dumps(extra, ensure_ascii=False, sort_keys=True)


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
    "verifier_source": _VERIFIER_SOURCE,
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

    uuid = ex.get("uuid") or str(idx)
    out["dataset_id"] = dataset_id
    out["dataset_version_date"] = dataset_version
    out["row_id"] = str(uuid)
    out["license"] = license_str or ex.get("license")
    out["used_by_model"] = used_by_model

    # responses_create_params, metadata, agent_ref are stored as JSON strings
    # by the loader (heterogeneous nested schemas across 104 tasks).
    rcp = _safe_json_loads(ex.get("responses_create_params")) or {}
    md = _safe_json_loads(ex.get("metadata")) or {}
    source_dataset = md.get("source_dataset") if isinstance(md, dict) else None
    out["subsource_raw"] = source_dataset
    out["source_dataset_id"] = source_dataset

    # ── Prompt ────────────────────────────────────────────────────────────
    context_messages = _input_to_context_messages(rcp)
    if not context_messages:
        # Fallback: synth single-turn user message from `question` column.
        q = (ex.get("question") or "").strip()
        if q:
            context_messages = [{"role": "user", "content": q}]
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

    # ── Verifier classification (single type) ─────────────────────────────
    if not source_dataset:
        out["_drop_reason"] = "missing_source_dataset"
        return out

    # [E] Ungradeable multi-solution tasks (no canonical string answer)
    if source_dataset in _MULTI_SOLUTION_TASKS:
        out["_drop_reason"] = "ungradeable_verifier_reasoning_gym_multi_solution"
        return out

    # ── Build verification payload ────────────────────────────────────────
    gt_text = ex.get("answer")
    gt_text = gt_text.strip() if isinstance(gt_text, str) else None
    verif_info = _build_verification_info(md, source_dataset=source_dataset, uuid=str(uuid))

    # [F] GT quality gate (delegates to default require_present=True path
    # because reasoning_gym is NOT in _GT_NOT_IN_GT_FIELD / _GT_OPTIONAL).
    reason = should_drop_ground_truth_for_verifier(
        gt_text,
        verifier_type=_VERIFIER_TYPE,
        verification_info_raw=verif_info,
        check_placeholder=cfg.check_gt_placeholder,
        min_gt_chars=cfg.min_gt_chars,
    )
    if reason:
        out["_drop_reason"] = reason
        return out

    # ── Finalise ──────────────────────────────────────────────────────────
    difficulty = md.get("difficulty") if isinstance(md, dict) else None
    diff_str = json.dumps(difficulty, ensure_ascii=False, sort_keys=True) if difficulty is not None else None

    example_id = _stable_sha256(f"{dataset_id}|{split}|{source_dataset}|{idx}|{uuid}")

    out.update({
        "_keep": True,
        "_drop_reason": None,
        "example_id": example_id,
        "context_messages": context_messages,
        "language": "en",
        "domain": _DOMAIN,
        "ability": source_dataset,
        "difficulty": diff_str,
        "verifier_type": _VERIFIER_TYPE,
        "verifier_source": _VERIFIER_SOURCE,
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
        description="Filter + normalize nvidia/Nemotron-RL-ReasoningGym-v1 to RL v1 schema."
    )
    parser.add_argument(
        "--input_dir",
        default="/home/workdir/Master_Thesis/Datasets/rl/nvidia__Nemotron-RL-ReasoningGym-v1/train",
    )
    parser.add_argument(
        "--output_parquet",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_rl_reasoning_gym_v1/rl_nemotron_rl_reasoning_gym_v1.kept.parquet",
    )
    parser.add_argument(
        "--output_parquet_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_rl_reasoning_gym_v1/rl_nemotron_rl_reasoning_gym_v1.dropped.parquet",
    )
    parser.add_argument(
        "--output_jsonl_sample",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_rl_reasoning_gym_v1/kept.sample.jsonl",
    )
    parser.add_argument(
        "--output_jsonl_sample_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/nemotron_rl_reasoning_gym_v1/dropped.sample.jsonl",
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

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    dataset_id = "nvidia/Nemotron-RL-ReasoningGym-v1"
    license_str = "CC-BY-4.0"
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
            print(f"  {reason:60s} {count:>6,}")

    if "ability" in kept.column_names and len(kept) > 0:
        from collections import Counter
        ab = Counter(kept["ability"])
        print(f"\nKept by source_dataset (ability) — {len(ab)} unique:")
        for k, v in ab.most_common():
            print(f"  {k:40s} {v:>5,}")


if __name__ == "__main__":
    main()
