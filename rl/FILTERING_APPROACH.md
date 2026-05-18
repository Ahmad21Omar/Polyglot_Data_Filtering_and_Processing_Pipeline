# RL Filtering Pipeline — Approach & Justification

> **Companion document to [`VERIFIER_POLICY.md`](VERIFIER_POLICY.md).** Where the
> Verifier Policy answers *"how do we score model outputs?"*, this document
> answers *"how does raw data become training-ready RL rows?"*.

This file documents the end-to-end filtering pipeline for the 9 RL datasets
in this thesis, mirroring the structure of the SFT pipeline at
`Master_Thesis/Filtering_Pipeline/README.md` so a reader familiar with the SFT
work can pick up RL quickly.

The high-level goal is the same as for SFT: produce a single, deduplicated,
quality-verified parquet that downstream training consumes. The mechanics
differ in two important places (Stage 2B and Stage 4) and those differences
are justified explicitly below — that justification is the load-bearing part
of this document.

---

## Pipeline overview

```
Stage 0  Download                 9 public HuggingFace datasets   ~1.59M rows
Stage 1  Dataset elimination      verifier-scope policy           9 datasets pass (1 skipped)
Stage 2A Per-dataset filter       structural + verifier policy    1,127,950 rows
Stage 2B Quality scoring          SKIPPED — see "Why no Stage 2B"
Stage 3A Intra-dataset dedup      sha256(prompt) within DS        already in 2A
Stage 3  step 0  Merge            concat in verifier-strength order  1,127,950
Stage 3B Cross-dataset exact dedup SHA1(last_user_msg) + GT-aware  1,105,317 (−2.0%)
Stage 3C Cross-dataset fuzzy dedup RapidFuzz ≥ 90, GT-aware,
                                   logi_glue intra-skip            1,084,071 (−1.9%)
Stage 4  Finalise + E2E validation 19-column upload schema         1,084,071 (final)
```

After Stage 2A the 9 datasets total **1,127,950 rows / 9.0 GB** in parquet,
all sharing the `rl_schema_v1` schema (20 columns; see any
`filter_and_format_*.py` for the canonical `Features` definition).

---

## Stage 0 — Download

Same as SFT: HuggingFace artifacts downloaded into
`Master_Thesis/Datasets/rl/<dataset>/`. No filtering at this stage.

| Source | Dataset on disk |
|---|---|
| `am-team/AM-Thinking-v1-RL-Dataset` | `am_thinking_v1_rl` |
| `allenai/Dolci-Think-RL-7B` | `dolci_think_rl_7b` |
| `logicreasoning/logi_glue` | `logi_glue` |
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` | `nemotron_3_nano_rl_blend` |
| `nvidia/Nemotron-RL-ReasoningGym-v1` | `nemotron_rl_reasoning_gym_v1` |
| `AIML-TUDA/SLR-Bench` (+ 6 lang variants) | `AIML-TUDA__SLR-Bench`, `*_slr_bench/` |
| `MiniMaxAI/SynLogic` | `MiniMaxAI__SynLogic` |
| `PrimeIntellect/SYNTHETIC-2-RL` | `synthetic2_rl` |
| `TIGER-Lab/WebInstruct-verified` | `TIGER-Lab__WebInstruct-verified` |

---

## Stage 1 — Dataset-level elimination

Two rejection criteria, applied before any row is examined:

**1A — Verifier-scope policy.** The dataset's answer types must be scoreable
by a deterministic, rule-based or library-based verifier (no LLM judge). See
[`VERIFIER_POLICY.md`](VERIFIER_POLICY.md) for the full policy and the
consistency proof across all 9 datasets.

**1B — Overlap with already-included sources.** A dataset is rejected when
its content is fully subsumed by another already-accepted source.

| Decision | Dataset | Reason |
|---|---|---|
| ❌ skipped | `CLUTRR/v1` | Already covered via `logi_glue.cluttr` (10,099 rows); CC-BY-NC; no new verifier code. See `Master_Thesis/rl_datasets/rl_clutrr_v1.md`. |

All 9 listed datasets pass Stage 1 with documented reasoning.

---

## Stage 2A — Per-dataset filter & format

Each dataset has its own script under
`Master_Thesis/Data_pipline/rl/filter_and_format_<dataset>.py`. Unlike SFT
(which extracted a *shared* 6-phase pipeline because all SFT rows have the
same `{reasoning_text, response_text}` shape), RL rows differ wildly by
verifier type — code rows need `assert` extraction, math rows need numeric
normalisation, prolog rows need `head :- body` extraction, etc. — so the
filter logic *cannot* be cleanly shared without forcing per-verifier
branching back in. Each script is therefore self-contained but applies the
same conceptual steps:

1. **Per-row verifier-scope check** — drop rows the policy excludes
   (e.g. `requires_model_verifier` in WebInstruct; `ungradeable_verifier_llm_judge_*` in Dolci).
2. **Structural quality filters** — empty prompt, prompt-too-short
   (typically <30 chars; SLR uses <100), URL/image references, malformed
   answer encoding (e.g. `boolean_non_canonical`, `integer_not_integer`).
3. **Per-verifier validation** — for example, math_equiv rows must have an
   extractable numeric; prolog_rule_induction rows must have a `head :- body`
   shape (defense-in-depth against the IPT shortcut hack); fraction rows must
   contain a recognisable LaTeX/text fraction.
4. **rl_schema_v1 conversion** — produce the 20-column canonical row with
   `verifier_type`, `verifier_source`, `ground_truth_text`,
   `verification_info_raw` (JSON), plus standard metadata.
5. **Intra-dataset dedup** — `sha256(prompt)` (and in slr_bench, per-language
   `sha256(lang \0 prompt)`). This is the Stage 3A-equivalent done inline; the
   global Stage 3A from the SFT pipeline is therefore unnecessary as a
   separate step.

Output per dataset: `corpora/rl/<dataset>/rl_<dataset>_v1.{kept,dropped}.parquet`
plus a `FILTERING_REPORT.md`.

---

## Stage 2B — Quality scoring **(intentionally skipped)**

The SFT pipeline runs a learned quality classifier here (FastText OH-ELI5
for English, FineWeb-HQ on XLM-R for multilingual) and drops the lowest-
scoring 20%. For RL data this step is **deliberately omitted**. Three reasons:

1. **Row shape is different.** SFT rows contain `reasoning_text` and
   `response_text` — natural inputs for "is this good chain-of-thought / good
   answer?" classifiers. RL rows contain only a `prompt` plus
   `verification_info` — the answer comes from the policy model at training
   time, not from the dataset. There is no answer to score.

2. **RL prompts are synthetic templates by construction.** Most of our RL
   data is procedurally generated (SLR-Bench, SynLogic, reasoning_gym) or
   tightly-formatted instruction-following (IFEval, MCQ, code asserts). A
   FineWeb-HQ-style classifier scores naturalistic web text — on
   synthetic templates its score distribution is flat and non-discriminative.
   Dropping the bottom 20% would mostly drop *short* prompts, not *low-quality*
   prompts, because the classifier has no signal in this regime.

3. **The verifier is the RL-time quality gate.** Bad prompts produce
   rollouts that never get reward 1.0, so the RL policy gradient simply
   ignores them. This is structurally different from SFT, where bad rows go
   directly into the cross-entropy loss and must be excluded pre-training.
   The verifier-scope policy (Stage 1A + Stage 2A.1) already enforces the
   property we actually care about: every kept row admits a hardenable
   reward signal.

Empirical check on this argument: the silent-pass-safety audits across the
9 datasets (per-dataset hardening reports) show 0 silent-passes across
≥30k adversarial probes. Any prompt that the verifier *could* misbehave on
is already excluded.

**Where the trade-off bites.** Without Stage 2B we may include prompts that
are valid but pedagogically weak (trivial / too-easy, exotic edge cases,
out-of-distribution puzzles). For RL this manifests as low signal density,
not a wrong training signal — a stronger policy model just learns these
quickly and moves on. If we ever observe pathological training behaviour
traceable to specific prompt families, the right response is a curriculum
or sampling-weight adjustment in Stage 4, not a learned quality filter.

---

## Stage 3A — Intra-dataset dedup

Done inline inside each `filter_and_format_<dataset>.py` via
`sha256(prompt)` (or `sha256(lang \0 prompt)` for slr_bench multilingual).
Drop reasons are recorded as `duplicate_prompt` / `duplicate_question` in
the per-dataset `dropped.parquet` and `FILTERING_REPORT.md`.

This is the equivalent of `Filtering_Pipeline/stage_3/deduplication/stage3a_same_id_dedup.py`
in the SFT pipeline; we keep it inline rather than as a separate stage
because RL datasets are small enough individually that the streaming
two-pass design of SFT 3A would be overkill.

---

## Stage 3 — Merge then cross-dataset deduplication **(DONE)**

The merge sits **at the start of Stage 3** (logically Stage 4 step 1 in the
original sketch below, but operationally inseparable from the cross-DS
dedup that follows), mirroring the SFT layout where
`merge_filtered.py` runs immediately before `stage_3/deduplication/`.

### Step 0 — Merge in verifier-strength order

Script: [`stage_3/deduplication/stage_3_00_merge.py`](stage_3/deduplication/stage_3_00_merge.py).

The 9 per-dataset `*.kept.parquet` files are concatenated in
**verifier-strength order**. Because the subsequent dedup is
first-seen-wins, the strongest verifier "claims" each duplicate prompt:

1. `slr_bench`           (Prolog symbolic execution)
2. `synlogic`            (per-task rule-based dispatch)
3. `nemotron_rl_reasoning_gym_v1`  (reasoning_gym lib)
4. `am_thinking_v1_rl`   (math_equiv + code execution)
5. `dolci_think_rl_7b`   (math_equiv + code + if_rules)
6. `nemotron_3_nano_rl_blend`
7. `synthetic2_rl`
8. `webinstruct_verified`
9. `logi_glue`           (text_match / multi_gt only — weakest)

Schema is identical across datasets (`rl_schema_v1`), so concat is a
straight PyArrow streaming operation. **Output: 1,127,950 rows in one
parquet, no rows dropped.**

### Step 3B — Exact-hash dedup, GT-aware

Script: [`stage_3/deduplication/stage_3b_exact_hash_dedup.py`](stage_3/deduplication/stage_3b_exact_hash_dedup.py).

- Hash key: `SHA1(lowercase(strip_punctuation(collapse_whitespace(last_user_message))))`.
  RL prompts heavily reuse system boilerplate, so hashing the full context
  would never match — we hash only the last user turn.
- Policy:
  - **Cross-DS duplicate prompt → DROP** (first-seen-wins).
  - **Intra-DS duplicate prompt + identical GT** (sha1(strip(GT))) → **DROP**.
  - **Intra-DS duplicate prompt + different GT → KEEP** (legitimate
    multi-answer, e.g. procedurally-generated tasks).
  - 7 SLR-Bench multilingual subsets → **passthrough** (synthetic, no
    resource overlap).
- SQLite-backed prompt index for memory efficiency; resumable.

**Result:** 1,127,950 → 1,105,317 (−22,633, −2.0 %)
- dropped cross-DS: 5,370
- dropped intra-DS same-GT: 17,263
- kept intra-DS diff-GT (multi-answer): 5,263
- passthrough: 126,189

### Step 3C — Fuzzy dedup, GT-aware

Script: [`stage_3/deduplication/stage_3c_fuzzy_dedup.py`](stage_3/deduplication/stage_3c_fuzzy_dedup.py).

- Algorithm: RapidFuzz `fuzz.ratio` on the same normalised last-user-message,
  with leading-token + length-bin bucketing and 32-way `ProcessPoolExecutor`
  fan-out.
- **Threshold = 90**, chosen after a 4-way empirical sweep (78 / 85 / 90 / 95)
  on a 100,000-row sample. See
  [`THRESHOLD_DECISION.md`](stage_3/deduplication/THRESHOLD_DECISION.md)
  for the rationale (cross-DS detectability, false-positive characterisation,
  literature comparison).
- Policy carries over from 3B with one exception:
  - **`logicreasoning/logi_glue` is excluded from intra-DS dedup**
    (cross-DS still active). Reason: templated MC/NLI prompts share ~95 %
    of their text with answer-relevant variation in 1-2 tokens, and the GT
    label space is tiny (e.g. {entailment, contradiction, neutral}), so
    GT-aware filtering cannot distinguish near-duplicates from
    legitimately distinct items. Manual inspection of an unrestricted run
    confirmed the vast majority of "duplicate" drops in logi_glue were
    real distinct tasks.

**Result:** 1,105,317 → 1,084,071 (−21,246, −1.9 %)
- dropped cross-DS: 1,109
- dropped intra-DS same-GT: 20,137
- kept intra-DS diff-GT (multi-answer): 74,053
- passthrough: 126,189

---

## Stage 4 — Finalise + E2E validation **(DONE)**

The original sketch reserved Stage 4 for "merge + sampling weights".
Since merge moved up into Stage 3, Stage 4 now contains the two steps
needed to produce an upload-ready dataset.

### Step 4.1 — Finalise

Script: [`stage_4/stage_4_finalize.py`](stage_4/stage_4_finalize.py).

Strips the three internal dedup columns (`_dedup_hash`, `_dedup_norm`,
`_dedup_seen_in_ds`) and writes a parquet with exactly the **19 canonical
columns** in canonical order. Streaming snappy parquet; emits a
`schema.json` (SHA-256 + counts) and a `FINAL_DATASET_REPORT.md`.

### Step 4.2 — End-to-end validation

Script: [`stage_4/e2e_test.py`](stage_4/e2e_test.py).

Streaming validation suite: schema, NULL checks on critical fields,
`context_messages` structure, `verification_info_raw` JSON parse +
per-verifier required-key check, global `example_id` uniqueness,
distribution sanity, and light-weight in-process verifier roundtrips for
`math_equiv` / `text_match` / `multiple_choice` / `structured_match`.

**Result:** 1,084,071 rows · 4.73 GB · SHA-256
`fa76f51cfa3fa3e6151686f334f7d4a9a42214660479baa2041366888a256a8f`.

### Sampling weights

The original sketch reserved a Step 4.3 for sampling weights (to avoid
the flat concat over-weighting large datasets like logi_glue, which is
~48 % of all kept rows). This step was **deferred** to downstream RL
training: when reward curves are available, a per-dataset sampling
mixture can be defined on top of the canonical parquet without
re-running the pipeline. The canonical parquet keeps all rows so
downstream consumers can pick any mixture.

---

## Disk space budget (actual final run)

| File | Size |
|---|---|
| Sum of 9 per-dataset `kept.parquet` | 9.0 GB |
| Merged `rl_v1_merged.kept.parquet` | 6.08 GB |
| 3B output (`exact_dedup.kept` + `dropped`) | 5.53 + 1.3 GB |
| 3C output (`fuzzy_dedup.kept` + `dropped`) | 5.1 + 145 MB |
| **4 Final upload parquet** | **4.73 GB** |

Plus the per-dataset dropped files (~6 GB across all 9) which stay on
disk as the audit trail.

---

## Files & locations (cheat sheet)

- Verifier-scope policy: [`VERIFIER_POLICY.md`](VERIFIER_POLICY.md)
- Stage 1 decisions: [`stage_1_dataset_elimination/`](stage_1_dataset_elimination/)
- Stage 2A per-dataset filter scripts: [`stage_2a/filter_and_format/`](stage_2a/filter_and_format/)
- Stage 2A shared filters: [`stage_2a/filters/`](stage_2a/filters/)
- Stage 2B skip rationale: [`stage_2b_skipped/`](stage_2b_skipped/)
- Stage 3 merge + dedup scripts: [`stage_3/deduplication/`](stage_3/deduplication/)
- Stage 4 finalise + E2E: [`stage_4/`](stage_4/)
- Per-dataset outputs (working tree): `Master_Thesis/corpora/rl/<dataset>/`
- Final upload parquet (working tree): `Master_Thesis/corpora/rl/final/rl_v1_final.parquet`
- Reference SFT pipeline: [`../sft/README.md`](../sft/README.md)

---

**History.** *Created 2026-05-17, after all 9 datasets passed Stage 2A and
the quality-skip decision was made. Updated 2026-05-17 with the
implemented Stage 3 / Stage 4 details and the empirical threshold
decision (90, logi_glue intra-skip) after manual inspection of false
positives.*
