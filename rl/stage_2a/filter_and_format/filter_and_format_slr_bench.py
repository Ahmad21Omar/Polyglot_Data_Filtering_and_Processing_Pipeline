"""Filter + normalize the AIML-TUDA/SLR-Bench multilingual family to RL v1.

Combines the English `AIML-TUDA/SLR-Bench` (config `v1-All`) with the six
translated variants:

- German     — `ost(Zug)` / `west(Zug)`
- Spanish    — `este(Tren)` / `oeste(Tren)`
- French     — `est(Train)` / `ouest(Train)`
- Italian    — `est(Treno)` / `ovest(Treno)`
- Portuguese — `leste(Trem)` / `oeste(Trem)`
- Dutch      — `oost(Trein)` / `west(Trein)`

In each translated variant *both the natural-language prompt and the Prolog
predicate vocabulary* are translated. The validation program and ground-truth
rule are still executable Prolog, so the same `prolog_rule_induction` verifier
scores all 7 languages — we only need per-row `positive_predicate` /
`negative_predicate` in `verification_info.evaluation_config`. The
`evaluate_with_prolog` function in the server is language-agnostic; spot-
checked 3 GT rules per language → 3/3 score 1.0 on every language.

Dedup is per-language (a translated row is a *distinct* training example, not
a duplicate of the English source). `language` field is set to ISO-639-1 code.

Output: corpora/rl/slr_bench/rl_slr_bench_v1.{kept,dropped}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset, Features, Value, load_from_disk
import pyarrow.parquet as pq


DATASET_ID_BY_LANG = {
    "en": "AIML-TUDA/SLR-Bench",
    "de": "AIML-TUDA/SLR-Bench-German",
    "es": "AIML-TUDA/SLR-Bench-Spanish",
    "fr": "AIML-TUDA/SLR-Bench-French",
    "it": "AIML-TUDA/SLR-Bench-Italian",
    "pt": "AIML-TUDA/SLR-Bench-Portuguese",
    "nl": "AIML-TUDA/SLR-Bench-Dutch",
}

# (positive_predicate, negative_predicate) per language.
PREDICATES_BY_LANG = {
    "en": ("eastbound", "westbound"),
    "de": ("ost", "west"),
    "es": ("este", "oeste"),
    "fr": ("est", "ouest"),
    "it": ("est", "ovest"),
    "pt": ("leste", "oeste"),
    "nl": ("oost", "west"),
}

# Input layouts differ:
#  - English lives in HF arrow form at .../v1-All/train (load_from_disk)
#  - Translations live as parquet at /home/workdir/<lang>_slr_bench/dataset/v1-All/train.parquet
INPUT_DIR_BY_LANG = {
    "en": ("arrow", "/home/workdir/Master_Thesis/Datasets/rl/AIML-TUDA__SLR-Bench/v1-All/train"),
    "de": ("parquet", "/home/workdir/german_slr_bench/dataset/v1-All/train.parquet"),
    "es": ("parquet", "/home/workdir/spanish_slr_bench/dataset/v1-All/train.parquet"),
    "fr": ("parquet", "/home/workdir/french_slr_bench/dataset/v1-All/train.parquet"),
    "it": ("parquet", "/home/workdir/italian_slr_bench/dataset/v1-All/train.parquet"),
    "pt": ("parquet", "/home/workdir/portuguese_slr_bench/dataset/v1-All/train.parquet"),
    "nl": ("parquet", "/home/workdir/dutch_slr_bench/dataset/v1-All/train.parquet"),
}

LICENSE = "cc-by-4.0"
_MIN_PROMPT_LEN = 100


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _has_body_re(positive_pred: str) -> re.Pattern:
    return re.compile(rf"{re.escape(positive_pred)}\s*\([^()]*\)\s*:-", re.IGNORECASE)


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


def _empty_record(lang: str) -> Dict[str, Any]:
    return {
        "_keep": False,
        "_drop_reason": None,
        "dataset_id": DATASET_ID_BY_LANG[lang],
        "dataset_version_date": None,
        "example_id": None,
        "row_id": None,
        "subsource_raw": None,
        "source_dataset_id": DATASET_ID_BY_LANG[lang],
        "license": LICENSE,
        "used_by_model": "",
        "context_messages": None,
        "language": lang,
        "domain": "logical_reasoning_ilp",
        "ability": "inductive_logic_programming",
        "difficulty": "",
        "verifier_type": None,
        "verifier_source": "slr_bench",
        "ground_truth_text": None,
        "verification_info_raw": None,
        "avg_reward": None,
        "reward_model_metadata": None,
    }


def _transform_row(row: Dict[str, Any], lang: str, version_date: str) -> Dict[str, Any]:
    out = _empty_record(lang)
    out["dataset_version_date"] = version_date
    out["row_id"] = str(row.get("id") or "")

    prompt = (row.get("prompt") or "").strip()
    gt_rule = (row.get("ground-truth rule") or "").strip()
    vp = (row.get("validation program") or "").strip()
    level = row.get("curriculum level")
    tier = (row.get("curriculum tier") or "").strip()
    out["subsource_raw"] = f"L{level:02d}_{tier}" if isinstance(level, int) else tier
    out["difficulty"] = tier or ""

    pos_pred, neg_pred = PREDICATES_BY_LANG[lang]

    if not prompt:
        out["_drop_reason"] = "empty_prompt"
        return out
    if len(prompt) < _MIN_PROMPT_LEN:
        out["_drop_reason"] = "prompt_too_short"
        return out
    if not gt_rule:
        out["_drop_reason"] = "empty_ground_truth_rule"
        return out
    if not vp:
        out["_drop_reason"] = "empty_validation_program"
        return out
    if not _has_body_re(pos_pred).search(gt_rule):
        out["_drop_reason"] = "ground_truth_rule_has_no_body"
        return out
    if f"{pos_pred}(" not in vp or f"{neg_pred}(" not in vp:
        out["_drop_reason"] = "validation_program_missing_pos_or_neg"
        return out

    out["context_messages"] = [{"role": "user", "content": prompt}]
    out["example_id"] = _sha256(f"{lang}\0{prompt}")
    out["verifier_type"] = "prolog_rule_induction"
    out["ground_truth_text"] = gt_rule

    vi: Dict[str, Any] = {
        "validation_program": vp,
        "evaluation_config": {
            "positive_predicate": pos_pred,
            "negative_predicate": neg_pred,
        },
        "allow_multiple_rules": False,
        "language": lang,
        "curriculum_level": level,
        "curriculum_tier": tier,
        "rule_complexity": row.get("rule complexity"),
        "rule_sampling": row.get("rule sampling"),
        "background_sampling": row.get("background sampling"),
        "problem_size": row.get("problem size"),
        "vocabulary_predicates": row.get("vocabulary predicates"),
    }
    out["verification_info_raw"] = json.dumps(vi, ensure_ascii=False, sort_keys=True)
    out["_keep"] = True
    return out


def _iter_rows(lang: str):
    kind, path = INPUT_DIR_BY_LANG[lang]
    if kind == "arrow":
        ds = load_from_disk(path)
        for row in ds:
            yield dict(row)
    else:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000):
            for row in batch.to_pylist():
                yield row


def process_language(lang: str, version_date: str) -> Tuple[List[Dict], Counter]:
    seen = set()
    records: List[Dict] = []
    drops: Counter = Counter()
    for row in _iter_rows(lang):
        rec = _transform_row(row, lang, version_date)
        if not rec["_keep"]:
            drops[rec["_drop_reason"]] += 1
            records.append(rec)
            continue
        prompt = (row.get("prompt") or "").strip()
        h = _sha256(f"{lang}\0{prompt}")
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
        "--output-dir",
        default="/home/workdir/Master_Thesis/corpora/rl/slr_bench",
    )
    ap.add_argument(
        "--languages",
        default="en,de,es,fr,it,pt,nl",
        help="Comma-separated ISO-639-1 codes.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    version_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_records: List[Dict] = []
    drops_by_lang: Dict[str, Counter] = {}
    kept_by_lang: Counter = Counter()

    for lang in languages:
        print(f"[slr_bench/{lang}] processing {DATASET_ID_BY_LANG[lang]}…")
        records, drops = process_language(lang, version_date)
        kept_n = sum(1 for r in records if r["_keep"])
        print(f"  rows={len(records)} kept={kept_n} dropped={len(records)-kept_n}")
        all_records.extend(records)
        drops_by_lang[lang] = drops
        kept_by_lang[lang] = kept_n

    kept = [r for r in all_records if r["_keep"]]
    dropped = [r for r in all_records if not r["_keep"]]
    total = len(all_records)
    print(
        f"\nTOTAL: input={total} kept={len(kept)} dropped={len(dropped)} "
        f"({len(kept)/total:.1%})"
    )

    vt_dist = Counter(r["verifier_type"] for r in kept)
    lang_dist = Counter(r["language"] for r in kept)
    diff_dist = Counter(r["difficulty"] for r in kept)
    print(f"verifier_type: {dict(vt_dist)}")
    print(f"by language: {dict(lang_dist)}")

    def _strip_internal(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    kept_clean = [_strip_internal(r) for r in kept]
    dropped_clean = [{**_strip_internal(r), "_drop_reason": r["_drop_reason"]} for r in dropped]

    kept_features = Features({k: v for k, v in _FEATURES.items() if not k.startswith("_")})
    dropped_features = Features({
        **{k: v for k, v in _FEATURES.items() if not k.startswith("_")},
        "_drop_reason": Value("string"),
    })

    Dataset.from_list(kept_clean, features=kept_features).to_parquet(
        out_dir / "rl_slr_bench_v1.kept.parquet"
    )
    Dataset.from_list(dropped_clean, features=dropped_features).to_parquet(
        out_dir / "rl_slr_bench_v1.dropped.parquet"
    )

    write_report(out_dir, drops_by_lang, vt_dist, lang_dist, diff_dist,
                 total, len(kept), len(dropped), version_date, languages)

    print(f"\nWrote: {out_dir}/rl_slr_bench_v1.kept.parquet ({len(kept)} rows)")
    print(f"Wrote: {out_dir}/rl_slr_bench_v1.dropped.parquet ({len(dropped)} rows)")
    print(f"Wrote: {out_dir}/FILTERING_REPORT.md")


def write_report(out_dir: Path, drops_by_lang: Dict[str, Counter], vt_dist: Counter,
                 lang_dist: Counter, diff_dist: Counter, total_in: int,
                 kept_n: int, dropped_n: int, version_date: str,
                 languages: List[str]) -> None:
    lines = [
        "# SLR-Bench Multilingual → RL v1 Filtering Report",
        "",
        f"**Generated:** {version_date}",
        f"**Languages:** {', '.join(languages)}",
        f"**Paper:** Helff et al., \"SLR: An Automated Synthesis Framework for "
        f"Scalable Logical Reasoning\", arXiv 2506.15787",
        "",
        "## Summary",
        "",
        f"- Input rows (across all languages): **{total_in:,}**",
        f"- Kept: **{kept_n:,}** ({kept_n/total_in:.1%})",
        f"- Dropped: **{dropped_n:,}** ({dropped_n/total_in:.1%})",
        "",
        "## Per-language counts (kept)",
        "",
        "| language | dataset                              | kept rows |",
        "|----------|--------------------------------------|----------:|",
    ]
    for lang in languages:
        lines.append(
            f"| {lang} | `{DATASET_ID_BY_LANG[lang]}` | {lang_dist.get(lang, 0):,} |"
        )
    lines += [
        "",
        "## Per-language verifier predicate configuration",
        "",
        "Each row carries `verification_info.evaluation_config.{positive_predicate,",
        "negative_predicate}` so the shared `prolog_rule_induction` verifier dispatches",
        "correctly across languages. Spot-checked: 3 GT rules / language → 3/3 score 1.0.",
        "",
        "| language | positive_predicate | negative_predicate |",
        "|----------|--------------------|--------------------|",
    ]
    for lang in languages:
        pos, neg = PREDICATES_BY_LANG[lang]
        lines.append(f"| {lang} | `{pos}` | `{neg}` |")
    lines += [
        "",
        "## Verifier-scope policy",
        "",
        "Identical to the English-only pipeline — see",
        "`Filtering_Pipeline/rl/VERIFIER_POLICY.md` and the original",
        "`FILTERING_REPORT.md`. All kept rows share `verifier_type = "
        "\"prolog_rule_induction\"` and are scored deterministically by SWI-Prolog.",
        "",
        "## Per-language drop reasons",
        "",
    ]
    for lang in languages:
        lines.append(f"### {lang} (`{DATASET_ID_BY_LANG[lang]}`)")
        lines.append("")
        lines.append("| Reason | n |")
        lines.append("|---|---:|")
        drops = drops_by_lang.get(lang, Counter())
        if not drops:
            lines.append("| (no drops) | 0 |")
        else:
            for reason, c in drops.most_common():
                lines.append(f"| `{reason}` | {c:,} |")
        lines.append("")
    lines += [
        "",
        "## Curriculum tier distribution (kept rows, all languages)",
        "",
        "| tier | rows |",
        "|---|---:|",
    ]
    for diff, c in diff_dist.most_common():
        lines.append(f"| {diff} | {c:,} |")
    lines += [
        "",
        "## Reward-hacking caveat (Isomorphic Perturbation Testing)",
        "",
        "Same as English: the upstream symbolic judge is vulnerable to extensional",
        "shortcuts (fact enumeration). Defense in depth applies in both the filter",
        "(`ground_truth_rule_has_no_body` drop with language-specific head predicate)",
        "and the server (rule extractor requires `<pos_pred>(...) :- body.` clause).",
        "",
        "## License",
        "",
        f"`{LICENSE}` — CC-BY-4.0 across all 7 SLR-Bench language variants",
        "(English + 6 translations on the AIML-TUDA HF organisation).",
        "",
    ]
    (out_dir / "FILTERING_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
