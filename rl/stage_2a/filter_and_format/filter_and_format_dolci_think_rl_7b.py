"""Filter + normalize `allenai/Dolci-Think-RL-7B` into our RL v1 schema.

Design goals (RL v1, 2026 rewrite):
  - Verifier-type names describe the concrete check, not a generic bucket.
    Dolci ships 6 distinct verifiers and we expose each with a precise name:
        math_equiv, if_rules, code_asserts, code_stdio,
        llm_judge_ref, llm_judge_open
  - Prompt stored as `context_messages` (list of {role, content}) like SFT.
    (We do NOT store a flat prompt_text column; it can be rebuilt from
    context_messages at any time.)
  - Verifier payload split cleanly:
        ground_truth_text      — raw GT string from upstream
        verification_info_raw  — JSON blob with extra info (IF constraint,
                                 code-test format, judge template)
  - Dolci's `constraint` / `constraint_type` columns fold into
    `verification_info_raw` for IF rows — no top-level carry-over.

Input:
  Master_Thesis/Datasets/rl/allenai__Dolci-Think-RL-7B/train

Output:
  kept.parquet, dropped.parquet, kept.sample.jsonl, dropped.sample.jsonl

Filter stages (heuristic):
  [A] prompt_length    [B] prompt_references   [D] english LID
  [F] ground_truth_quality   [G] passrate

Note: RL prompts have no completions yet (rollouts happen in the trainer),
so SFT-style response filters (repetition, cutoff-mention) are not applied
here — they would belong to the rollout, not the prompt.
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


_ROLE_PREFIX_RE = re.compile(
    r"^\s*(system|user|assistant)\s*:\s*", re.IGNORECASE | re.MULTILINE
)


def _prompt_to_context_messages(prompt: Any) -> List[Dict[str, str]]:
    """Parse Dolci's `prompt` into a list of chat messages.

    Dolci stores prompts as a flat string starting with "user: ...".
    Occasionally multiple role markers appear (system + user).
    If no role marker is found we wrap the whole text as a single user turn.
    """
    if isinstance(prompt, list):
        out: List[Dict[str, str]] = []
        for m in prompt:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if role and content:
                out.append({"role": role, "content": content})
        return out

    if not isinstance(prompt, str):
        return []

    text = prompt.strip()
    if not text:
        return []

    matches = list(_ROLE_PREFIX_RE.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}]

    messages: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        role = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _messages_to_flat_text(messages: List[Dict[str, str]]) -> str:
    """Internal helper: flatten context_messages for the heuristic filters.

    Not stored in the output — rebuilt on the fly whenever a filter needs
    a single string to inspect.
    """
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n---\n\n".join(parts)


# Dolci `dataset` label → (our domain, our verifier_type)
_DOLCI_VERIFIER_MAP: Dict[str, Tuple[str, str]] = {
    "math":                ("math",                   "math_equiv"),
    "ifeval":              ("instruction_following",  "if_rules"),
    "code":                ("code",                   "code_asserts"),
    "code_stdio":          ("code",                   "code_stdio"),
    "general-quality_ref": ("chat",                   "llm_judge_ref"),
    "general-quality":     ("chat",                   "llm_judge_open"),
}

# Verifier types we cannot grade automatically with NeMo Gym (no judge
# endpoint wired up). Dropped at Stage 1a regardless of upstream advice.
_UNGRADEABLE_VERIFIER_TYPES: set = {"llm_judge_ref", "llm_judge_open"}


def _classify_row(dataset_col: Any) -> Tuple[Optional[str], Optional[str]]:
    """Map Dolci `dataset` column to (domain, verifier_type)."""
    labels: List[str] = []
    if isinstance(dataset_col, list):
        labels = [str(x).strip().lower() for x in dataset_col if x]
    elif isinstance(dataset_col, str):
        labels = [dataset_col.strip().lower()]
    for lbl in labels:
        if lbl in _DOLCI_VERIFIER_MAP:
            return _DOLCI_VERIFIER_MAP[lbl]
    return (None, None)


def _build_verifier_payload(
    verifier_type: str,
    ground_truth_raw: Any,
    constraint: Optional[str],
    constraint_type: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Build (ground_truth_text, verification_info_raw) for a row.

    - `ground_truth_text`: primary answer string from Dolci (first list
       element preserved as-is; can be a JSON-string for code tests or a
       python-repr for IF constraints — downstream verifier interprets).
    - `verification_info_raw`: JSON string with per-verifier extras.
    """
    primary: Optional[str] = None
    alternatives: List[str] = []
    if isinstance(ground_truth_raw, list) and ground_truth_raw:
        items = [str(x).strip() for x in ground_truth_raw if x is not None]
        items = [x for x in items if x]
        if items:
            primary = items[0]
            alternatives = items[1:]
    elif isinstance(ground_truth_raw, str):
        primary = ground_truth_raw.strip() or None

    extra: Dict[str, Any] = {}

    if verifier_type == "if_rules":
        extra["constraint_text"] = constraint or ""
        extra["constraint_type"] = constraint_type or "multi"
    elif verifier_type == "code_asserts":
        extra["test_format"] = "asserts"
    elif verifier_type == "code_stdio":
        extra["test_format"] = "stdio"
    elif verifier_type == "llm_judge_ref":
        extra["judge_template"] = "quality_ref"
    elif verifier_type == "llm_judge_open":
        extra["judge_template"] = "quality"
    # math_equiv: no extra info

    if alternatives:
        extra["alternative_ground_truths"] = alternatives

    info_json = (
        json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    )
    return primary, info_json


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    # [A] Prompt length
    min_prompt_chars: int = 30

    # [B] Prompt references
    drop_prompt_http: bool = True
    drop_prompt_images: bool = True

    # [D] FastText English LID
    enable_fasttext_english_filter: bool = False
    fasttext_model_path: str = "/home/workdir/Master_Thesis/models/fasttext/lid.176.bin"
    fasttext_threshold: float = 0.80
    fasttext_min_chars: int = 80
    fasttext_keep_if_too_short: bool = True

    # [F] Ground truth
    check_gt_placeholder: bool = True
    min_gt_chars: int = 1

    # [G] Passrate
    enable_passrate_filter: bool = True
    max_passrate: float = 0.90
    drop_passrate_zero: bool = False


# ── Schema & empty output ───────────────────────────────────────────────────

_EMPTY_TEMPLATE: Dict[str, Any] = {
    # control (stripped before writing kept parquet)
    "_keep": False,
    "_drop_reason": None,
    # IDs & provenance
    "dataset_id": None,
    "dataset_version_date": None,
    "example_id": None,
    "row_id": None,
    "subsource_raw": None,
    "source_dataset_id": None,
    "license": None,
    "used_by_model": None,
    # content
    "context_messages": None,
    "language": "en",
    # domain
    "domain": None,
    "ability": None,
    "difficulty": None,
    # verifier
    "verifier_type": None,
    "verifier_source": "nemo_gym",
    "ground_truth_text": None,
    "verification_info_raw": None,
    # signals
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

    row_id = ex.get("custom_id") or ex.get("id") or str(idx)
    out["dataset_id"] = dataset_id
    out["dataset_version_date"] = dataset_version
    out["row_id"] = str(row_id)
    out["license"] = license_str
    out["used_by_model"] = used_by_model

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt_raw = ex.get("prompt")
    if not prompt_raw:
        out["_drop_reason"] = "missing_prompt"
        return out

    context_messages = _prompt_to_context_messages(prompt_raw)
    if not context_messages:
        out["_drop_reason"] = "empty_context_messages"
        return out

    flat_text = _messages_to_flat_text(context_messages)  # internal only

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
    dataset_col = ex.get("dataset")
    domain, verifier_type = _classify_row(dataset_col)
    if verifier_type is None:
        out["_drop_reason"] = "unknown_verifier"
        return out

    # NeMo Gym has no judge endpoint — drop llm_judge_* rows so RL training
    # never sees prompts it cannot score. (Phase 3 decision, 2026-05-14.)
    if verifier_type in _UNGRADEABLE_VERIFIER_TYPES:
        out["_drop_reason"] = f"ungradeable_verifier_{verifier_type}"
        return out

    # ── Build verification payload ────────────────────────────────────────
    gt_text, verif_info = _build_verifier_payload(
        verifier_type=verifier_type,
        ground_truth_raw=ex.get("ground_truth"),
        constraint=ex.get("constraint"),
        constraint_type=ex.get("constraint_type"),
    )

    # [F] GT quality gate — verifier-aware
    # schema_pydantic / schema_structured_outputs: GT is in verification_info_raw.
    # llm_judge_open: GT is optional.
    # All others: GT required.
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
            ex.get("passrate"),
            max_passrate=cfg.max_passrate,
            drop_if_zero=cfg.drop_passrate_zero,
        )
        if reason:
            out["_drop_reason"] = reason
            return out

    # ── Finalise ──────────────────────────────────────────────────────────
    passrate_val = ex.get("passrate")
    avg_reward: Optional[float] = None
    if passrate_val is not None:
        try:
            avg_reward = float(passrate_val)
        except (TypeError, ValueError):
            avg_reward = None

    example_id = _stable_sha256(f"{dataset_id}|{split}|{row_id}")

    ability_str: Optional[str] = None
    if isinstance(dataset_col, list):
        ability_str = ", ".join(str(x) for x in dataset_col if x) or None
    elif dataset_col:
        ability_str = str(dataset_col)

    out.update({
        "_keep": True,
        "_drop_reason": None,
        "example_id": example_id,
        "subsource_raw": ex.get("dataset_source"),
        "source_dataset_id": ex.get("original_dataset"),
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

def _read_local_revision(dataset_info_path: Path) -> Optional[str]:
    if not dataset_info_path.exists():
        return None
    try:
        info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        for k in info.get("download_checksums", {}).keys():
            m = re.search(r"Dolci-Think-RL-7B@([0-9a-f]{8,40})/", k)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter + normalize allenai/Dolci-Think-RL-7B to RL v1 schema."
    )
    parser.add_argument(
        "--input_dir",
        default="/home/workdir/Master_Thesis/Datasets/rl/allenai__Dolci-Think-RL-7B/train",
    )
    parser.add_argument(
        "--output_parquet",
        default="/home/workdir/Master_Thesis/corpora/rl/dolci_think_rl_7b/rl_dolci_think_rl_7b_v1.kept.parquet",
    )
    parser.add_argument(
        "--output_parquet_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/dolci_think_rl_7b/rl_dolci_think_rl_7b_v1.dropped.parquet",
    )
    parser.add_argument(
        "--output_jsonl_sample",
        default="/home/workdir/Master_Thesis/corpora/rl/dolci_think_rl_7b/kept.sample.jsonl",
    )
    parser.add_argument(
        "--output_jsonl_sample_dropped",
        default="/home/workdir/Master_Thesis/corpora/rl/dolci_think_rl_7b/dropped.sample.jsonl",
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

    dataset_id = "allenai/Dolci-Think-RL-7B"
    license_str = "ODC-BY"
    used_by_model = "Olmo-3-7B-Think-RL"
    split = "train"

    dataset_version = _read_local_revision(input_dir / "dataset_info.json")
    if dataset_version is None:
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

    # Kept
    kept = filter_kept(mapped)
    output_parquet = Path(args.output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing kept parquet: {output_parquet}")
    kept.to_parquet(str(output_parquet))
    print(f"  → {len(kept):,} kept rows")

    # Dropped
    dropped = filter_dropped(mapped)
    output_dropped = Path(args.output_parquet_dropped)
    output_dropped.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing dropped parquet: {output_dropped}")
    dropped.to_parquet(str(output_dropped))
    print(f"  → {len(dropped):,} dropped rows")

    # Samples
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

    # Drop-reason summary
    if "_drop_reason" in dropped.column_names and len(dropped) > 0:
        from collections import Counter
        counts = Counter(r for r in dropped["_drop_reason"] if r)
        print("\nDrop reason summary:")
        for reason, count in counts.most_common():
            print(f"  {reason:40s} {count:>6,}")

    # Verifier-type summary on kept rows
    if "verifier_type" in kept.column_names and len(kept) > 0:
        from collections import Counter
        vt = Counter(kept["verifier_type"])
        print("\nVerifier type summary (kept):")
        for k, v in vt.most_common():
            print(f"  {k:40s} {v:>6,}")


if __name__ == "__main__":
    main()
