"""Filter + normalize `MiniMaxAI/SynLogic` to RL v1.

SynLogic (MiniMax, arXiv 2505.19641) is a 35-task-family logical reasoning RL
corpus designed for RLVR with **rule-based, task-specific verifiers** shipped
in the official MIT-licensed code repo (https://github.com/MiniMax-AI/SynLogic).

Per-row payload of interest:
- `prompt` (chat list) — instruction with `<think>...</think><answer>...</answer>`
  formatting requirement (English + Chinese mixed; HF card lists both).
- `data_source` — task family name; the verifier dispatch key.
- `extra_info.game_data_str` — JSON string containing the canonical puzzle
  state. `json.loads(game_data_str)` yields `{question, answer, difficulty,
  metadata, gpt_response}` which is the `base.data.Data` payload the upstream
  verifier consumes.

The dataset's `reward_model.answer/solution` fields are empty across the board
(verified on 200 inspected rows); the actual ground truth lives inside
`game_data_str.answer`. Our filter pulls that through into `ground_truth_text`
and packs the full `game_data_str` into `verification_info_raw` so the server
can reconstruct the upstream `Data` object.

verifier_type   = `synlogic_rule_based`
verifier_source = `synlogic`

Output: corpora/rl/synlogic/rl_synlogic_v1.{kept,dropped}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset, Features, Value, load_from_disk


DATASET_ID = "MiniMaxAI/SynLogic"
LICENSE = "mit"  # upstream code MIT; HF card unspecified; we use code license


# Load the upstream verifier dispatch table to gate `data_source` membership.
# We add the vendored path so this import works without installing the package.
_VENDOR_PATH = (
    "/home/workdir/Master_Thesis/Code_Templat/Gym/resources_servers/synlogic/synlogic_vendor"
)
sys.path.insert(0, _VENDOR_PATH)
try:
    from task2verifier import verifier_classes as _UPSTREAM_VERIFIERS  # type: ignore
    _SUPPORTED_TASK_FAMILIES = set(_UPSTREAM_VERIFIERS.keys())
except Exception as e:  # pragma: no cover
    print(f"WARN: failed to import upstream verifier_classes: {e}")
    _SUPPORTED_TASK_FAMILIES = set()


_MIN_PROMPT_LEN = 30


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _detect_language(text: str) -> str:
    """Coarse English/Chinese/mixed detector — sufficient for `language` field.

    SynLogic's hard config interleaves Chinese prompts; we tag rows so the
    multilingual provenance is queryable downstream.
    """
    if not text:
        return "en"
    chinese = sum(1 for c in text if "一" <= c <= "鿿")
    ratio = chinese / max(len(text), 1)
    if ratio > 0.2:
        return "zh"
    if chinese > 0:
        return "en_zh_mixed"
    return "en"


_FEATURES = Features({
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


def _empty_record() -> Dict[str, Any]:
    return {
        "_keep": False,
        "_drop_reason": None,
        "dataset_id": DATASET_ID,
        "dataset_version_date": None,
        "example_id": None,
        "row_id": None,
        "subsource_raw": None,
        "source_dataset_id": DATASET_ID,
        "license": LICENSE,
        "used_by_model": "",
        "context_messages": None,
        "language": "en",
        "domain": "logic_puzzles",
        "ability": "logical_reasoning",
        "difficulty": "",
        "verifier_type": None,
        "verifier_source": "synlogic",
        "ground_truth_text": None,
        "verification_info_raw": None,
        "avg_reward": None,
        "reward_model_metadata": None,
    }


def _transform_row(row: Dict[str, Any], split_cfg: str, version_date: str) -> Dict[str, Any]:
    out = _empty_record()
    out["dataset_version_date"] = version_date
    out["row_id"] = str((row.get("extra_info") or {}).get("index") or "")

    data_source = (row.get("data_source") or "").strip()
    ability = (row.get("ability") or "").strip()
    out["subsource_raw"] = data_source
    out["difficulty"] = split_cfg  # 'easy' or 'hard'
    out["ability"] = ability or "logical_reasoning"

    # Extract the first/last user message — SynLogic prompts are single-turn.
    prompt_msgs = row.get("prompt") or []
    if not isinstance(prompt_msgs, list) or not prompt_msgs:
        out["_drop_reason"] = "empty_prompt"
        return out
    prompt_text = (prompt_msgs[-1].get("content") or "").strip()
    if not prompt_text:
        out["_drop_reason"] = "empty_prompt"
        return out
    if len(prompt_text) < _MIN_PROMPT_LEN:
        out["_drop_reason"] = "prompt_too_short"
        return out

    # game_data_str must parse as JSON and contain non-empty `answer`.
    gds_raw = (row.get("extra_info") or {}).get("game_data_str") or ""
    if not gds_raw:
        out["_drop_reason"] = "empty_game_data_str"
        return out
    try:
        gd = json.loads(gds_raw)
    except (json.JSONDecodeError, TypeError):
        out["_drop_reason"] = "game_data_str_not_json"
        return out
    answer = gd.get("answer")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        out["_drop_reason"] = "empty_ground_truth_answer"
        return out

    # Task family must have an upstream verifier.
    if data_source not in _SUPPORTED_TASK_FAMILIES:
        out["_drop_reason"] = f"no_verifier_for_task_family"
        return out

    out["context_messages"] = [
        {"role": (m.get("role") or "user"), "content": (m.get("content") or "")}
        for m in prompt_msgs
    ]
    out["language"] = _detect_language(prompt_text)
    out["example_id"] = _sha256(f"{data_source}\0{prompt_text}")
    out["verifier_type"] = "synlogic_rule_based"
    # Stringify the canonical answer for the schema (verifier reads vi instead).
    out["ground_truth_text"] = (
        answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    )

    # verification_info carries the full upstream `Data` payload + dispatch key.
    vi = {
        "data_source": data_source,
        "game_data": gd,
        "difficulty_tier": split_cfg,
    }
    out["verification_info_raw"] = json.dumps(vi, ensure_ascii=False, sort_keys=True)
    out["_keep"] = True
    return out


def process_split(input_dir: Path, cfg: str, version_date: str) -> Tuple[List[Dict], Counter]:
    ds = load_from_disk(str(input_dir / cfg / "train"))
    seen = set()
    records: List[Dict] = []
    drops: Counter = Counter()
    for row in ds:
        rec = _transform_row(dict(row), cfg, version_date)
        if not rec["_keep"]:
            drops[rec["_drop_reason"]] += 1
            records.append(rec)
            continue
        # Dedup on sha256(data_source + prompt) within full corpus.
        h = rec["example_id"]
        if h in seen:
            rec["_keep"] = False
            rec["_drop_reason"] = "duplicate_prompt"
            drops["duplicate_prompt"] += 1
        else:
            seen.add(h)
        records.append(rec)
    return records, drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        default="/home/workdir/Master_Thesis/Datasets/rl/MiniMaxAI__SynLogic",
    )
    ap.add_argument(
        "--output-dir",
        default="/home/workdir/Master_Thesis/corpora/rl/synlogic",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[synlogic] processing {DATASET_ID} (supported task families: "
          f"{len(_SUPPORTED_TASK_FAMILIES)})…")

    all_records: List[Dict] = []
    all_drops: Counter = Counter()
    kept_by_split: Counter = Counter()

    for cfg in ("easy", "hard"):
        print(f"[synlogic/{cfg}] …")
        records, drops = process_split(in_dir, cfg, version_date)
        kept_n = sum(1 for r in records if r["_keep"])
        print(f"  rows={len(records)} kept={kept_n} dropped={len(records)-kept_n}")
        all_records.extend(records)
        all_drops.update(drops)
        kept_by_split[cfg] = kept_n

    kept = [r for r in all_records if r["_keep"]]
    dropped = [r for r in all_records if not r["_keep"]]
    total = len(all_records)
    print(f"\nTOTAL: input={total} kept={len(kept)} dropped={len(dropped)} "
          f"({len(kept)/total:.1%})")

    vt_dist = Counter(r["verifier_type"] for r in kept)
    src_dist = Counter(r["subsource_raw"] for r in kept)
    lang_dist = Counter(r["language"] for r in kept)
    print(f"verifier_type: {dict(vt_dist)}")
    print(f"language: {dict(lang_dist)}")

    def _strip_internal(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    kept_clean = [_strip_internal(r) for r in kept]
    dropped_clean = [
        {**_strip_internal(r), "_drop_reason": r["_drop_reason"]} for r in dropped
    ]

    kept_features = Features({k: v for k, v in _FEATURES.items() if not k.startswith("_")})
    dropped_features = Features({
        **{k: v for k, v in _FEATURES.items() if not k.startswith("_")},
        "_drop_reason": Value("string"),
    })

    Dataset.from_list(kept_clean, features=kept_features).to_parquet(
        out_dir / "rl_synlogic_v1.kept.parquet"
    )
    Dataset.from_list(dropped_clean, features=dropped_features).to_parquet(
        out_dir / "rl_synlogic_v1.dropped.parquet"
    )

    write_report(
        out_dir, all_drops, vt_dist, src_dist, lang_dist, kept_by_split,
        total, len(kept), len(dropped), version_date,
    )

    print(f"\nWrote: {out_dir}/rl_synlogic_v1.kept.parquet ({len(kept)} rows)")
    print(f"Wrote: {out_dir}/rl_synlogic_v1.dropped.parquet ({len(dropped)} rows)")
    print(f"Wrote: {out_dir}/FILTERING_REPORT.md")


def write_report(out_dir: Path, drops: Counter, vt_dist: Counter, src_dist: Counter,
                 lang_dist: Counter, kept_by_split: Counter, total_in: int,
                 kept_n: int, dropped_n: int, version_date: str) -> None:
    lines = [
        "# SynLogic → RL v1 Filtering Report",
        "",
        f"**Generated:** {version_date}",
        f"**Dataset:** `{DATASET_ID}` (configs: easy + hard, split: train)",
        f"**Paper:** Liu et al., \"SynLogic\", arXiv 2505.19641",
        f"**Verifier repo:** https://github.com/MiniMax-AI/SynLogic (MIT)",
        "",
        "## Summary",
        "",
        f"- Input rows: **{total_in:,}**",
        f"- Kept: **{kept_n:,}** ({kept_n/total_in:.1%})",
        f"- Dropped: **{dropped_n:,}** ({dropped_n/total_in:.1%})",
        f"- Easy kept: **{kept_by_split.get('easy', 0):,}**",
        f"- Hard kept: **{kept_by_split.get('hard', 0):,}**",
        "",
        "## Verifier-scope policy",
        "",
        "SynLogic is a fully **rule-based** RLVR corpus: each task family ships a",
        "task-specific Python verifier in the upstream MIT-licensed repo. We",
        "vendor the repo into the NeMo Gym server (",
        "`Code_Templat/Gym/resources_servers/synlogic/synlogic_vendor/`) and",
        "dispatch on `data_source`. No LLM judge.",
        "",
        "Fits the thesis-wide rule/library-based verifier policy",
        "(see `Filtering_Pipeline/rl/VERIFIER_POLICY.md`).",
        "",
        "## Important caveat — `reward_model` is empty",
        "",
        "Across all 200 inspected rows the dataset's top-level `reward_model.answer`",
        "and `.solution` are empty strings. The actual canonical ground truth lives",
        "in `extra_info.game_data_str` (JSON with key `answer`). Our filter pulls",
        "the answer from there into `ground_truth_text` and bundles the full",
        "`game_data` dict into `verification_info` so the server reconstructs the",
        "upstream `base.data.Data` object for `verifier.verify(data, rollout)`.",
        "",
        "## Drop reasons",
        "",
        "| Reason | n |",
        "|---|---:|",
    ]
    for reason, c in drops.most_common():
        lines.append(f"| `{reason}` | {c:,} |")
    lines += [
        "",
        "## Verifier-type distribution (kept rows)",
        "",
        "| verifier_type | rows |",
        "|---|---:|",
    ]
    for vt, c in vt_dist.most_common():
        lines.append(f"| {vt} | {c:,} |")
    lines += [
        "",
        "## Task-family distribution (kept rows, `subsource_raw`)",
        "",
        "| task family | rows |",
        "|---|---:|",
    ]
    for src, c in src_dist.most_common():
        lines.append(f"| {src} | {c:,} |")
    lines += [
        "",
        "## Language distribution (heuristic CJK detector)",
        "",
        "| language | rows |",
        "|---|---:|",
    ]
    for lang, c in lang_dist.most_common():
        lines.append(f"| {lang} | {c:,} |")
    lines += [
        "",
        "## License",
        "",
        f"`{LICENSE}` — HF dataset card unspecified; upstream verifier code is",
        "MIT-licensed (https://github.com/MiniMax-AI/SynLogic/blob/main/LICENSE).",
        "We mark records as MIT under the same license as the verifier code we",
        "depend on.",
        "",
    ]
    (out_dir / "FILTERING_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
