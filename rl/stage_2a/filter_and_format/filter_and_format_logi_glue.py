"""Filter + normalize `logicreasoning/logi_glue` (train-only subsets) to RL v1.

logi_glue is a heterogeneous collection of 24 logical-reasoning datasets.
We use the 10 subsets that ship a `train` split (the original benchmark's
fine-tuning set):

| subset           | rows    | category | choices |
|------------------|---------|----------|---------|
| abduction_animal | 23.100  | ff       | -       |
| adv (αARCT)      | 2.420   | mcqa     | 2       |
| alpha_nli (αNLI) | 169.654 | nli      | 2       |
| anli             | 162.865 | ff       | [""]    |
| cluttr           | 10.100  | ff       | [""]    |
| folio            | 1.004   | nli      | 3       |
| logiQA           | 7.376   | mcqa     | 4       |
| logicNLI         | 16.000  | nli      | 4       |
| proofwriter      | 69.814  | fv       | 2       |
| rulebert         | 56.000  | fv       | 2       |

Verifier dispatch:
  * category ∈ {mcqa, nli, fv} AND len(choices)>1 → `multi_gt`
      primary GT = answer_text, alternatives = [letter("A"/"B"/...) for answer_choice]
  * else (ff OR len(choices)<=1) → `text_match`

Schema notes:
  * `input` is pre-formatted (Context + Question + Options). One user turn.
  * `id_` is NOT unique across rows in 5/10 subsets (alpha_nli 90% dup, proofwriter 95% dup).
    We dedup on sha256(input) within each subset.
  * `alpha_nli.answer_choice` is 1-indexed (upstream bug). `answer_text` matches
    `choices` in 100% of rows — we trust `answer_text` and reconstruct the letter
    from `choices.index(answer_text)`.
  * License: logi_glue is a research compilation of mixed-license sources;
    we record `composite_research_only` per-row.

Downsampling: cap at 25.000 rows per subset post-filter (random sample, seed=42).
Reason: alpha_nli + anli + proofwriter + rulebert would otherwise be 88% of the
mix and drown out the smaller reasoning-diverse subsets (folio, logiQA, adv).
See FILTERING_REPORT.md for full rationale.

verifier_source = "logi_glue".

Output: corpora/rl/logi_glue/rl_logi_glue_v1.{kept,dropped}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, Features, Value, load_from_disk


DATASET_ID = "logicreasoning/logi_glue"
LICENSE = "composite_research_only"
PER_SUBSET_CAP = 25_000
SEED = 42

TRAIN_SUBSETS = [
    "abduction_animal", "adv", "alpha_nli", "anli", "cluttr",
    "folio", "logiQA", "logicNLI", "proofwriter", "rulebert",
]

# Category → reasoning ability label (logged into rl_schema_v1.ability)
ABILITY_BY_SUBSET = {
    "abduction_animal": "abductive_reasoning",
    "adv":              "abductive_reasoning",
    "alpha_nli":        "abductive_reasoning",
    "anli":             "deductive_reasoning",
    "cluttr":           "inductive_reasoning",
    "folio":            "deductive_reasoning",
    "logiQA":           "mixed_reasoning",
    "logicNLI":         "deductive_reasoning",
    "proofwriter":      "deductive_reasoning",
    "rulebert":         "deductive_reasoning",
}

ALPHABET = "ABCDEFGHIJKLMNOP"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


_HTTP_RE = re.compile(r"https?://", re.IGNORECASE)


# ── Schema features ─────────────────────────────────────────────────────────


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
        "source_dataset_id": None,
        "license": LICENSE,
        "used_by_model": "",
        "context_messages": None,
        "language": "en",
        "domain": "logical_reasoning",
        "ability": None,
        "difficulty": "",
        "verifier_type": None,
        "verifier_source": "logi_glue",
        "ground_truth_text": None,
        "verification_info_raw": None,
        "avg_reward": None,
        "reward_model_metadata": None,
    }


# ── Per-row transform ───────────────────────────────────────────────────────


def _transform_row(subset: str, row: Dict[str, Any], version_date: str) -> Dict[str, Any]:
    out = _empty_record()
    out["dataset_version_date"] = version_date
    out["subsource_raw"] = subset
    out["row_id"] = str(row.get("id_"))
    out["source_dataset_id"] = row.get("original_dataset") or subset
    out["ability"] = ABILITY_BY_SUBSET[subset]

    input_text = (row.get("input") or "").strip()
    answer_text = (row.get("answer_text") or "").strip()
    category = (row.get("category") or "").strip().lower()
    choices = row.get("choices") if "choices" in row else None
    if not isinstance(choices, list):
        choices = None

    if not input_text:
        out["_drop_reason"] = "empty_input"
        return out
    if not answer_text:
        out["_drop_reason"] = "empty_answer_text"
        return out
    if _HTTP_RE.search(input_text):
        out["_drop_reason"] = "prompt_has_http"
        return out

    out["context_messages"] = [{"role": "user", "content": input_text}]
    out["example_id"] = _sha256(f"{subset}\0{input_text}\0{answer_text}")

    # Detect "real" MCQA-style: more than one non-empty choice AND answer ∈ choices.
    has_real_choices = (
        choices is not None
        and len(choices) > 1
        and any((c or "").strip() for c in choices)
        and answer_text in choices
    )

    vi: Dict[str, Any] = {
        "category": category,
        "original_dataset": row.get("original_dataset"),
        "subset": subset,
    }

    if has_real_choices:
        # Letter alternative (e.g. "A", "B"). We compute from answer_text-position
        # to dodge the alpha_nli 1-indexed answer_choice upstream bug.
        idx = choices.index(answer_text)
        if 0 <= idx < len(ALPHABET):
            letter = ALPHABET[idx]
            vi["choices"] = list(choices)
            vi["answer_letter"] = letter
            vi["alternative_ground_truths"] = [letter]
            out["verifier_type"] = "multi_gt"
        else:
            out["verifier_type"] = "text_match"
    else:
        out["verifier_type"] = "text_match"

    out["ground_truth_text"] = answer_text
    out["verification_info_raw"] = json.dumps(vi, ensure_ascii=False, sort_keys=True)
    out["_keep"] = True
    return out


# ── Pipeline ────────────────────────────────────────────────────────────────


def process_subset(subset: str, base_dir: Path, version_date: str) -> Tuple[List[Dict], Counter]:
    ds = load_from_disk(str(base_dir / subset / "train"))
    seen_input_hashes = set()
    records: List[Dict] = []
    drop_counter: Counter = Counter()

    for row in ds:
        rec = _transform_row(subset, dict(row), version_date)
        if not rec["_keep"]:
            drop_counter[rec["_drop_reason"]] += 1
            records.append(rec)
            continue
        # Within-subset dedup on input hash (id_ is unreliable).
        input_hash = _sha256((row.get("input") or "").strip())
        if input_hash in seen_input_hashes:
            rec["_keep"] = False
            rec["_drop_reason"] = "duplicate_input_within_subset"
            drop_counter["duplicate_input_within_subset"] += 1
        else:
            seen_input_hashes.add(input_hash)
        records.append(rec)
    return records, drop_counter


def downsample(records: List[Dict], cap: int, seed: int) -> Tuple[List[Dict], int]:
    kept = [r for r in records if r["_keep"]]
    if len(kept) <= cap:
        return records, 0
    rng = random.Random(seed)
    indices_to_keep = set(rng.sample(range(len(kept)), cap))
    new_records = []
    keep_idx = 0
    dropped_for_cap = 0
    for r in records:
        if not r["_keep"]:
            new_records.append(r)
            continue
        if keep_idx in indices_to_keep:
            new_records.append(r)
        else:
            r2 = dict(r)
            r2["_keep"] = False
            r2["_drop_reason"] = "subset_cap_downsample"
            dropped_for_cap += 1
            new_records.append(r2)
        keep_idx += 1
    return new_records, dropped_for_cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/home/workdir/Master_Thesis/Datasets/rl/logicreasoning__logi_glue")
    parser.add_argument("--output-dir", default="/home/workdir/Master_Thesis/corpora/rl/logi_glue")
    parser.add_argument("--cap", type=int, default=PER_SUBSET_CAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_records: List[Dict] = []
    per_subset_stats: Dict[str, Dict[str, int]] = {}

    cap_disabled = args.cap <= 0
    for subset in TRAIN_SUBSETS:
        print(f"[{subset}] processing…")
        records, drops = process_subset(subset, base_dir, version_date)
        if not cap_disabled:
            records, downsample_drops = downsample(records, args.cap, args.seed)
            if downsample_drops:
                drops["subset_cap_downsample"] = downsample_drops
        kept_n = sum(1 for r in records if r["_keep"])
        per_subset_stats[subset] = {
            "input_rows": len(records),
            "kept": kept_n,
            "dropped": len(records) - kept_n,
            **dict(drops),
        }
        all_records.extend(records)
        print(f"[{subset}] in={len(records)} kept={kept_n} drops={dict(drops)}")

    kept = [r for r in all_records if r["_keep"]]
    dropped = [r for r in all_records if not r["_keep"]]
    print(f"\nTOTAL: input={len(all_records)} kept={len(kept)} dropped={len(dropped)} ({len(kept)/len(all_records):.1%})")

    # Verifier-type distribution on kept
    vt_dist = Counter(r["verifier_type"] for r in kept)
    print(f"verifier_type distribution: {dict(vt_dist)}")

    # Write parquet
    def _strip_internal(r):
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        return out

    kept_clean = [_strip_internal(r) for r in kept]
    dropped_clean = [{**_strip_internal(r), "_drop_reason": r["_drop_reason"]} for r in dropped]

    kept_features = Features({k: v for k, v in _FEATURES.items() if not k.startswith("_")})
    dropped_features = Features({**{k: v for k, v in _FEATURES.items() if not k.startswith("_")}, "_drop_reason": Value("string")})

    Dataset.from_list(kept_clean, features=kept_features).to_parquet(out_dir / "rl_logi_glue_v1.kept.parquet")
    Dataset.from_list(dropped_clean, features=dropped_features).to_parquet(out_dir / "rl_logi_glue_v1.dropped.parquet")

    # Write report
    write_report(out_dir, per_subset_stats, vt_dist, len(all_records), len(kept), len(dropped), args.cap, args.seed, version_date)

    print(f"\nWrote: {out_dir}/rl_logi_glue_v1.kept.parquet ({len(kept)} rows)")
    print(f"Wrote: {out_dir}/rl_logi_glue_v1.dropped.parquet ({len(dropped)} rows)")
    print(f"Wrote: {out_dir}/FILTERING_REPORT.md")


def write_report(out_dir: Path, stats: Dict[str, Dict[str, int]], vt_dist: Counter,
                  total_in: int, kept_n: int, dropped_n: int, cap: int, seed: int,
                  version_date: str) -> None:
    cap_active = cap > 0
    cap_str = f"{cap:,} (random sample, seed={seed})" if cap_active else "disabled (keep everything)"
    lines = [
        "# logi_glue → RL v1 Filtering Report",
        "",
        f"**Generated:** {version_date}",
        f"**Dataset:** `{DATASET_ID}` (train-only subsets)",
        f"**Per-subset cap:** {cap_str}",
        "",
        "## Summary",
        "",
        f"- Input rows: **{total_in:,}**",
        f"- Kept: **{kept_n:,}** ({kept_n/total_in:.1%})",
        f"- Dropped: **{dropped_n:,}** ({dropped_n/total_in:.1%})",
        "",
    ]
    if cap_active:
        lines += [
            "## Why we downsampled (cap = {:,} per subset)".format(cap),
            "",
            "Without a cap, the 4 largest subsets (alpha_nli 170k + anli 163k + proofwriter 70k + rulebert 56k)",
            "constitute **88% of the mix** and structurally drown out the reasoning-diverse smaller",
            "subsets (folio 1k, adv 2.4k, logiQA 7.4k). A per-subset cap balances exposure across",
            "the 4 reasoning types (abductive / deductive / inductive / mixed) and prevents",
            "over-fitting to a single template (e.g. proofwriter's True/False/Unknown form).",
            "",
            "Random subsampling is seeded for reproducibility.",
            "",
        ]
    else:
        lines += [
            "## No-cap rationale",
            "",
            "Subset-cap is disabled — every row that passes the quality filters is kept.",
            "Trade-off: the 4 largest subsets (alpha_nli + anli + proofwriter + rulebert)",
            "constitute ~88% of the kept mix, which may bias the model toward their",
            "specific templates. RL trainer can rebalance via per-subset sampling weights",
            "(use `subsource_raw` column).",
            "",
        ]
    lines += [
        "## Per-subset breakdown",
        "",
        "| Subset | Input | Kept | Drops |",
        "|---|---:|---:|---|",
    ]
    for s, st in stats.items():
        drop_descr = ", ".join(f"{k}={v}" for k, v in st.items() if k not in ("input_rows","kept","dropped"))
        lines.append(f"| {s} | {st['input_rows']:,} | {st['kept']:,} | {drop_descr or '—'} |")
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
        "## Drop reasons",
        "",
        "- `empty_input` / `empty_answer_text` — defensive, should be 0 in this corpus.",
        "- `prompt_has_http` — drops rows whose prompt embeds URLs (training-distribution-shift risk).",
        "- `duplicate_input_within_subset` — `id_` is reused across rows in 5/10 subsets; we",
        "  dedupe on `sha256(input)` so semantic duplicates within a single subset are collapsed.",
        "  (No cross-subset dedup — semantically distinct datasets keep their entries.)",
        "- `subset_cap_downsample` — random rows dropped to hit the per-subset cap of 25.000.",
        "",
        "## Verifier-mapping rationale",
        "",
        "- **mcqa / nli / fv with ≥2 real choices** → `multi_gt`: primary GT is `answer_text`,",
        "  alternative is the choice **letter** (`A`/`B`/`C`/…). This lets a rollout that emits",
        "  just `B` instead of the full option text still score 1.0.",
        "- **ff (anli, cluttr, abduction_animal)** → `text_match`: the answer is canonical text",
        "  (e.g. `entailment`, `brother`); tolerant whitespace + punctuation normalisation is sufficient.",
        "- The `alpha_nli.answer_choice` field is 1-indexed in the upstream data (off-by-one bug",
        "  vs. the 0-indexed `choices` list). We bypass `answer_choice` entirely and reconstruct",
        "  the letter from `choices.index(answer_text)` — which matches 100% of rows.",
        "",
        "## License",
        "",
        f"`{LICENSE}` — logi_glue compiles 24 upstream datasets, each with its own license.",
        "Downstream redistribution should preserve the per-source attribution stored in",
        "`verification_info_raw.original_dataset` and `subsource_raw`.",
        "",
    ]
    (out_dir / "FILTERING_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
