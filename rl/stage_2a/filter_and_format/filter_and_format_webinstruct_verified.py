"""Filter + normalize `TIGER-Lab/WebInstruct-verified` to RL v1.

WebInstruct-verified (Ma et al., NeurIPS 2025; "General-Reasoner") is a
228,736-row cross-discipline reasoning corpus (Math/Physics/Chem/Finance/...
across 11 answer_types). The upstream paper relies on a 1.5B Qwen-based
*model verifier* (`TIGER-Lab/general-verifier`) to score open-form answers.

Our thesis applies a uniform **rule-based-only verifier policy** across all
RL datasets (cf. Dolci, AM-Thinking, logi_glue, SYNTHETIC-2-RL, ...). Rows
whose answer requires a model judge are excluded here for the same reason
we excluded LLM-judge rows from Dolci-Think-RL — this keeps verifier silent-
pass=0 hardenable across the entire pipeline.

Kept answer_types (143k rows, ~63%) — all deterministically rule-verifiable:

| answer_type     | rows   | verifier_type | rule                                  |
|-----------------|--------|---------------|---------------------------------------|
| Multiple Choice | 27,764 | multi_gt      | letter (A-Z, case-insensitive)        |
| Boolean         | 12,655 | multi_gt      | {Yes,No,True,False} (case-insensitive)|
| Integer         | 22,950 | math_equiv    | extracted numeric token               |
| Float           | 67,506 | math_equiv    | extracted numeric token (math_verify) |
| Percentage      | 7,036  | math_equiv    | numeric % value (math_verify)         |
| Fraction        | 5,494  | math_equiv    | LaTeX fraction (math_verify)          |

Dropped answer_types (verifier-scope policy):
- Expression (49k), String (16k), List (13k), Matrix (4.8k), Other (2.1k)
- These require model-based equivalence judgment (semantic prose, LaTeX
  proof equivalence, ordering-aware lists, etc.). Our rule-based stack
  cannot guarantee silent-pass=0 on them, so they are excluded.

Plus quality filters:
- empty Q/A
- prompt_has_http / image refs
- malformed boolean (LaTeX-as-answer, MCQ-letter-as-Boolean)
- non-parseable numeric (Integer/Float/Percentage without extractable value)
- range-percentage ("-11.7% to 24.3%" — single value required)
- MCQ with multi-character (>1 char) non-letter answer
- duplicate on sha256(question)

verifier_source = "webinstruct_verified".

Output: corpora/rl/webinstruct_verified/rl_webinstruct_verified_v1.{kept,dropped}.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, Features, Value, load_from_disk


DATASET_ID = "TIGER-Lab/WebInstruct-verified"
# Upstream license is "unknown" on HF; the paper's code repo is Apache-2.0.
# We use a conservative marker. Researcher-level redistribution only.
LICENSE = "tiger_lab_research_use"

# Answer types we keep (rule-verifiable end-to-end).
KEEP_ANSWER_TYPES = {
    "Multiple Choice", "Boolean", "Integer", "Float", "Percentage", "Fraction",
}
# Answer types we drop with reason "requires_model_verifier".
DROP_ANSWER_TYPES = {
    "Expression", "String", "List", "Matrix", "Other",
}

# Category → broader domain bucket for rl_schema_v1.domain
DOMAIN_BY_CATEGORY = {
    "Mathematics": "math",
    "Physics": "physics",
    "Chemistry": "chemistry",
    "Biology": "biology",
    "Computer Science": "computer_science",
    "Engineering": "engineering",
    "Business": "business_finance",
    "Finance": "business_finance",
    "Economics": "business_finance",
    "Health": "health",
    "Psychology": "social_science",
    "History": "humanities",
    "Philosophy": "humanities",
    "Law": "humanities",
    "Other STEM": "stem_other",
    "Other": "other",
}

# Allowed MCQ letters (extend if dataset has E/F/G).
MCQ_LETTERS = set("ABCDEFGH")

# Canonical Boolean normalizations.
BOOL_TRUE_TOKENS = {"yes", "true", "y", "t"}
BOOL_FALSE_TOKENS = {"no", "false", "n", "f"}

_HTTP_RE = re.compile(r"https?://", re.IGNORECASE)
_IMG_RE = re.compile(r"\b(?:see\s+(?:image|figure|diagram)|\[image\]|<img\b)", re.IGNORECASE)

# Match a signed decimal number (with optional thousands-separators) anywhere.
_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# Percentage range detector: "X% to Y%" or "X-Y%".
_PCT_RANGE_RE = re.compile(r"%\s*(?:to|–|—|-)\s*[-\d]")
# LaTeX-y fraction marker (so we can keep them even if no plain numeric exists).
_LATEX_FRAC_RE = re.compile(r"\\(?:frac|tfrac|dfrac|cfrac)\b|/")


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _strip_currency(s: str) -> str:
    return re.sub(r"[\$€£¥]", "", s).strip()


def _extract_numeric(s: str) -> Optional[str]:
    """First numeric token in `s`, comma-stripped. None if no numeric."""
    s = _strip_currency(s)
    m = _NUM_RE.search(s)
    if not m:
        return None
    return m.group(0).replace(",", "")


def _normalize_pct(s: str) -> Optional[Tuple[str, str]]:
    """Return (numeric_value, raw_with_pct). None if not a single-value pct."""
    s = s.strip()
    if _PCT_RANGE_RE.search(s):
        return None
    num = _extract_numeric(s)
    if num is None:
        return None
    return num, num + "%"


def _normalize_bool(s: str) -> Optional[str]:
    """Return 'True' or 'False' iff `s` is a canonical boolean token."""
    t = s.strip().rstrip(".").lower()
    if t in BOOL_TRUE_TOKENS:
        return "True"
    if t in BOOL_FALSE_TOKENS:
        return "False"
    return None


def _normalize_mcq_letter(s: str) -> Optional[str]:
    """Return single-letter A-H if `s` is just a (possibly punctuated) MCQ letter."""
    t = s.strip().rstrip(".").rstrip(")").strip().upper()
    if len(t) == 1 and t in MCQ_LETTERS:
        return t
    return None


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
        "source_dataset_id": DATASET_ID,
        "license": LICENSE,
        "used_by_model": "",
        "context_messages": None,
        "language": "en",
        "domain": "general_reasoning",
        "ability": "cross_domain_reasoning",
        "difficulty": "",
        "verifier_type": None,
        "verifier_source": "webinstruct_verified",
        "ground_truth_text": None,
        "verification_info_raw": None,
        "avg_reward": None,
        "reward_model_metadata": None,
    }


# ── Per-row transform ───────────────────────────────────────────────────────


def _transform_row(row: Dict[str, Any], version_date: str) -> Dict[str, Any]:
    out = _empty_record()
    out["dataset_version_date"] = version_date
    out["row_id"] = str(row.get("id") or "")

    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    ans_type = (row.get("answer_type") or "").strip()
    category = (row.get("category") or "").strip()
    difficulty = (row.get("difficulty") or "").strip()

    out["subsource_raw"] = ans_type
    out["difficulty"] = difficulty
    out["domain"] = DOMAIN_BY_CATEGORY.get(category, "other")

    # 1) Hard verifier-scope policy: drop the 5 model-judge-required answer_types.
    if ans_type in DROP_ANSWER_TYPES:
        out["_drop_reason"] = "requires_model_verifier"
        return out
    if ans_type not in KEEP_ANSWER_TYPES:
        out["_drop_reason"] = "unknown_answer_type"
        return out

    # 2) Quality filters.
    if not question:
        out["_drop_reason"] = "empty_question"
        return out
    if not answer:
        out["_drop_reason"] = "empty_answer"
        return out
    if len(question) < 30:
        out["_drop_reason"] = "question_too_short"
        return out
    if _HTTP_RE.search(question):
        out["_drop_reason"] = "prompt_has_http"
        return out
    if _IMG_RE.search(question):
        out["_drop_reason"] = "prompt_has_image_ref"
        return out

    out["context_messages"] = [{"role": "user", "content": question}]
    out["example_id"] = _sha256(f"{ans_type}\0{question}\0{answer}")

    vi: Dict[str, Any] = {
        "category": category,
        "answer_type": ans_type,
        "raw_answer": answer,
    }

    # 3) Per-answer-type validation + verifier dispatch.
    if ans_type == "Multiple Choice":
        letter = _normalize_mcq_letter(answer)
        if letter is None:
            out["_drop_reason"] = "mcq_non_letter_answer"
            return out
        out["verifier_type"] = "multi_gt"
        out["ground_truth_text"] = letter
        vi["alternative_ground_truths"] = [letter.lower()]
        vi["letter"] = letter

    elif ans_type == "Boolean":
        canonical = _normalize_bool(answer)
        if canonical is None:
            out["_drop_reason"] = "boolean_non_canonical"
            return out
        out["verifier_type"] = "multi_gt"
        out["ground_truth_text"] = canonical  # "True" or "False"
        if canonical == "True":
            vi["alternative_ground_truths"] = ["true", "yes", "Yes", "TRUE", "YES", "y", "Y"]
        else:
            vi["alternative_ground_truths"] = ["false", "no", "No", "FALSE", "NO", "n", "N"]

    elif ans_type == "Integer":
        num = _extract_numeric(answer)
        if num is None:
            out["_drop_reason"] = "integer_no_numeric"
            return out
        # Confirm it parses as int (allow trailing .0).
        try:
            f = float(num)
            if not f.is_integer():
                out["_drop_reason"] = "integer_not_integer"
                return out
            num = str(int(f))
        except ValueError:
            out["_drop_reason"] = "integer_unparseable"
            return out
        out["verifier_type"] = "math_equiv"
        out["ground_truth_text"] = num
        vi["alternative_ground_truths"] = [answer]  # also accept raw form

    elif ans_type == "Float":
        num = _extract_numeric(answer)
        if num is None:
            out["_drop_reason"] = "float_no_numeric"
            return out
        try:
            float(num)
        except ValueError:
            out["_drop_reason"] = "float_unparseable"
            return out
        out["verifier_type"] = "math_equiv"
        out["ground_truth_text"] = num
        vi["alternative_ground_truths"] = [answer]
        vi["numeric_tolerance"] = "rel_1e-3"

    elif ans_type == "Percentage":
        norm = _normalize_pct(answer)
        if norm is None:
            out["_drop_reason"] = "percentage_invalid_or_range"
            return out
        num, raw_with_pct = norm
        out["verifier_type"] = "math_equiv"
        out["ground_truth_text"] = num  # numeric form, e.g. "85.39"
        vi["alternative_ground_truths"] = [raw_with_pct, answer]
        vi["numeric_tolerance"] = "rel_1e-3"
        vi["unit"] = "percent"

    elif ans_type == "Fraction":
        # Strict-string match against the raw LaTeX/text form. We deliberately do
        # NOT add an "extracted numeric" alternative: `_extract_numeric("1/5")`
        # would return "1", which would silent-pass any rollout emitting "1".
        # Numeric-vs-symbolic fraction equivalence is a model-judge problem
        # (excluded by our verifier-scope policy); strict-string is the
        # hardenable rule.
        if not _LATEX_FRAC_RE.search(answer):
            out["_drop_reason"] = "fraction_no_recognizable_form"
            return out
        out["verifier_type"] = "math_equiv"
        out["ground_truth_text"] = answer

    out["verification_info_raw"] = json.dumps(vi, ensure_ascii=False, sort_keys=True)
    out["_keep"] = True
    return out


# ── Pipeline ────────────────────────────────────────────────────────────────


def process_dataset(input_dir: Path, version_date: str) -> Tuple[List[Dict], Counter]:
    ds = load_from_disk(str(input_dir / "train"))
    seen_q_hashes = set()
    records: List[Dict] = []
    drop_counter: Counter = Counter()

    for row in ds:
        rec = _transform_row(dict(row), version_date)
        if not rec["_keep"]:
            drop_counter[rec["_drop_reason"]] += 1
            records.append(rec)
            continue
        # Dedup on sha256(question) — id field is unique but we want semantic dedup.
        q_hash = _sha256((row.get("question") or "").strip())
        if q_hash in seen_q_hashes:
            rec["_keep"] = False
            rec["_drop_reason"] = "duplicate_question"
            drop_counter["duplicate_question"] += 1
        else:
            seen_q_hashes.add(q_hash)
        records.append(rec)
    return records, drop_counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/home/workdir/Master_Thesis/Datasets/rl/TIGER-Lab__WebInstruct-verified")
    parser.add_argument("--output-dir", default="/home/workdir/Master_Thesis/corpora/rl/webinstruct_verified")
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[webinstruct] processing {DATASET_ID}…")
    records, drops = process_dataset(base_dir, version_date)
    kept = [r for r in records if r["_keep"]]
    dropped = [r for r in records if not r["_keep"]]
    print(f"TOTAL: input={len(records)} kept={len(kept)} dropped={len(dropped)} ({len(kept)/len(records):.1%})")

    # Verifier-type distribution on kept
    vt_dist = Counter(r["verifier_type"] for r in kept)
    # answer_type distribution among kept
    at_dist = Counter(r["subsource_raw"] for r in kept)
    # domain distribution
    dom_dist = Counter(r["domain"] for r in kept)
    print(f"verifier_type: {dict(vt_dist)}")
    print(f"answer_type kept: {dict(at_dist)}")

    # Write parquet
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
        out_dir / "rl_webinstruct_verified_v1.kept.parquet"
    )
    Dataset.from_list(dropped_clean, features=dropped_features).to_parquet(
        out_dir / "rl_webinstruct_verified_v1.dropped.parquet"
    )

    write_report(out_dir, drops, vt_dist, at_dist, dom_dist,
                 len(records), len(kept), len(dropped), version_date)

    print(f"\nWrote: {out_dir}/rl_webinstruct_verified_v1.kept.parquet ({len(kept)} rows)")
    print(f"Wrote: {out_dir}/rl_webinstruct_verified_v1.dropped.parquet ({len(dropped)} rows)")
    print(f"Wrote: {out_dir}/FILTERING_REPORT.md")


def write_report(out_dir: Path, drops: Counter, vt_dist: Counter, at_dist: Counter,
                  dom_dist: Counter, total_in: int, kept_n: int, dropped_n: int,
                  version_date: str) -> None:
    lines = [
        "# WebInstruct-verified → RL v1 Filtering Report",
        "",
        f"**Generated:** {version_date}",
        f"**Dataset:** `{DATASET_ID}`",
        f"**Paper:** Ma et al., \"General-Reasoner\", NeurIPS 2025 (arXiv 2505.14652)",
        "",
        "## Summary",
        "",
        f"- Input rows: **{total_in:,}**",
        f"- Kept: **{kept_n:,}** ({kept_n/total_in:.1%})",
        f"- Dropped: **{dropped_n:,}** ({dropped_n/total_in:.1%})",
        "",
        "## Verifier-scope policy (why this dataset is filtered more aggressively)",
        "",
        "WebInstruct-verified was designed to be scored with a **1.5B model-based",
        "verifier** (`TIGER-Lab/general-verifier`) so that open-form answers like",
        "LaTeX expressions, free-text strings, lists and matrices can be judged",
        "for semantic equivalence. The upstream methodology explicitly argues that",
        "rule-based scorers cannot handle these answer types.",
        "",
        "Our thesis applies a **uniform rule-based-only verifier policy** across all",
        "RL datasets in the corpus (Dolci-Think-RL, AM-Thinking-v1-RL, logi_glue,",
        "SYNTHETIC-2-RL, Nemotron-3-Nano-RL-Blend, Nemotron-RL-ReasoningGym-v1,",
        "WebInstruct-verified). The policy is:",
        "",
        "> *Keep only rows whose answer can be scored by a deterministic, rule-based",
        "> or library-based verifier that we can adversarially harden to silent-pass=0.*",
        "",
        "This is the same rule that drove the LLM-judge row exclusion in",
        "Dolci-Think-RL. For WebInstruct-verified it means we drop 5 of 11",
        "answer_types upfront — not because they are uninteresting, but because",
        "scoring them faithfully would require deploying the upstream 1.5B model",
        "verifier (out of scope for this thesis's verifier-architecture chapter).",
        "",
        "### Answer types kept (rule-verifiable)",
        "",
        "| answer_type     | verifier_type | rule                                       |",
        "|-----------------|---------------|--------------------------------------------|",
        "| Multiple Choice | multi_gt      | single letter A-H, case-insensitive        |",
        "| Boolean         | multi_gt      | canonical {Yes,No,True,False} ± variants   |",
        "| Integer         | math_equiv    | extracted numeric token + math_verify      |",
        "| Float           | math_equiv    | extracted numeric token + math_verify      |",
        "| Percentage      | math_equiv    | numeric % value (single, no ranges)        |",
        "| Fraction        | math_equiv    | LaTeX fraction symbolic via math_verify    |",
        "",
        "### Answer types dropped (model-verifier required)",
        "",
        "- `Expression` (49k) — symbolic equivalence of free LaTeX/algebra/calculus",
        "  expressions; math_verify works only for canonical-mathy subset and would",
        "  produce false-negatives on cross-domain (physics/chem) formulae.",
        "- `String` (16k) — open-form prose; rule-based exact match would silent-fail",
        "  on paraphrases.",
        "- `List` (13k) — ordering and synonym variation; no hardenable rule.",
        "- `Matrix` (4.8k) — free LaTeX matrix forms; not handled by math_verify.",
        "- `Other` (2.1k) — by-definition heterogeneous.",
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
        "**Drop categories explained:**",
        "",
        "- `requires_model_verifier` — answer_type in {Expression, String, List, Matrix, Other}.",
        "  See verifier-scope policy above.",
        "- `empty_question` / `empty_answer` — defensive.",
        "- `question_too_short` — q < 30 chars (degenerate prompts).",
        "- `prompt_has_http` / `prompt_has_image_ref` — references to external/visual",
        "  content (training-distribution-shift risk).",
        "- `mcq_non_letter_answer` — Multiple-Choice rows where the answer is not a single",
        "  letter A-H (full option text or malformed).",
        "- `boolean_non_canonical` — Boolean rows containing LaTeX expressions or MCQ-style",
        "  letters (upstream labeling errors). We accept only {Yes,No,True,False} variants.",
        "- `integer_no_numeric` / `integer_not_integer` / `integer_unparseable` —",
        "  Integer rows without an extractable integer value (e.g. 'A-1').",
        "- `float_no_numeric` / `float_unparseable` — Float without an extractable numeric.",
        "- `percentage_invalid_or_range` — Percentage rows that encode a *range*",
        "  ('-11.7% to 24.3%') rather than a single value. Range-judging is model-judge",
        "  territory; we exclude.",
        "- `fraction_no_recognizable_form` — Fraction answers with no `\\frac{}{}` and no",
        "  numeric token.",
        "- `duplicate_question` — semantic dedup on `sha256(question)` after row-level",
        "  filtering.",
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
        "## answer_type distribution (kept rows, recorded as `subsource_raw`)",
        "",
        "| answer_type | rows |",
        "|---|---:|",
    ]
    for at, c in at_dist.most_common():
        lines.append(f"| {at} | {c:,} |")
    lines += [
        "",
        "## Domain distribution (kept rows)",
        "",
        "| domain | rows |",
        "|---|---:|",
    ]
    for dom, c in dom_dist.most_common():
        lines.append(f"| {dom} | {c:,} |")
    lines += [
        "",
        "## Contamination caveat",
        "",
        "WebInstruct-verified is sourced from the open web (re-crawled QA pages). The",
        "upstream paper evaluates on **MMLU-Pro, SuperGPQA, GPQA, AIME** — benchmarks",
        "whose questions are themselves indexed on the public web. RL-training on this",
        "corpus and then evaluating on those benchmarks therefore carries non-trivial",
        "contamination risk by construction. We do not perform a benchmark-level dedup",
        "(out of scope for this thesis's data-curation chapter), but any downstream",
        "evaluation report should disclose this risk.",
        "",
        "## License",
        "",
        f"`{LICENSE}` — the HF dataset card lists license=unknown; the General-Reasoner",
        "code repo is Apache-2.0 but the dataset re-crawl is from third-party pages.",
        "We mark this conservatively as research-use; downstream redistribution should",
        "preserve attribution to TIGER-Lab and the original web sources.",
        "",
    ]
    (out_dir / "FILTERING_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
