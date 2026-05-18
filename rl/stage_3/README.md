# Stage 3 — Merge + Cross-dataset Deduplication

Stage 3 takes the 9 per-dataset `*.kept.parquet` files produced by
[Stage 2A](../stage_2a/) and produces a single deduplicated parquet ready
for [Stage 4](../stage_4/) finalisation. There are three sequential steps:

```
9 per-dataset kept.parquet
        │
        ▼
   (3 step 0) merge in verifier-strength order      ── stage_3_00_merge.py
        │
        ▼
   merged.parquet   (1,127,950 rows)
        │
        ▼
   (3B) exact-hash dedup, GT-aware                  ── stage_3b_exact_hash_dedup.py
        │
        ▼
   exact_dedup.kept.parquet   (1,105,317 rows)
        │
        ▼
   (3C) fuzzy dedup, GT-aware                       ── stage_3c_fuzzy_dedup.py
        │       (preceded by threshold_sweep.py)
        ▼
   fuzzy_dedup.kept.parquet   (1,084,071 rows)
```

> **Note on Stage 3A.** The SFT pipeline has a separate "same-ID dedup" at
> Stage 3A. In RL this is performed **inline** inside each Stage 2A script
> via `sha256(prompt)` keyed dedup (per-language for SLR-Bench), so there
> is no separate Stage 3A directory.

## Why the merge sits at the start of Stage 3

The merge is *technically* its own stage (Stage 4 step 1 in the original
[`FILTERING_APPROACH.md`](../FILTERING_APPROACH.md)), but logically it
only exists to set up the cross-dataset dedup that follows. Mirroring the
SFT pipeline's layout (where `merge_filtered.py` sits inside `stage_2b/`
right before Stage 3), we place the merge script inside
`stage_3/deduplication/` here.

## Merge order — verifier strength, strongest first

Cross-dataset dedup uses **first-seen-wins**: the first dataset to
introduce a given prompt hash keeps the row; later datasets with the same
prompt are dropped. To make this policy meaningful, datasets are
concatenated in **verifier-strength order**, so the row that survives is
preferentially scored by the strictest verifier available for that
prompt.

| # | Dataset | Rows | Cumulative | Verifier rationale |
|---|---|---:|---:|---|
| 1 | `slr_bench` (7 langs) | 126,189 | 126,189 | Prolog symbolic execution |
| 2 | `synlogic` | 36,277 | 162,466 | Per-task rule-based dispatch |
| 3 | `nemotron_rl_reasoning_gym_v1` | 14,143 | 176,609 | `reasoning_gym` library |
| 4 | `am_thinking_v1_rl` | 53,635 | 230,244 | math_equiv + code execution |
| 5 | `dolci_think_rl_7b` | 78,225 | 308,469 | math_equiv + code + if_rules |
| 6 | `nemotron_3_nano_rl_blend` | 58,657 | 367,126 | code_stdio + if_rules + MCQ + schema |
| 7 | `synthetic2_rl` | 109,269 | 476,395 | 9 verifier types |
| 8 | `webinstruct_verified` | 134,855 | 611,250 | math_equiv + multi_gt (text-match) |
| 9 | `logi_glue` | 516,700 | 1,127,950 | text_match / multi_gt only — weakest |

## Layout

```
stage_3/
├── README.md                              # this file
└── deduplication/
    ├── stage_3_00_merge.py                # concat the 9 kept.parquet files
    ├── stage_3b_exact_hash_dedup.py       # SHA1(last_user_msg) dedup, GT-aware
    ├── stage_3c_fuzzy_dedup.py            # RapidFuzz dedup, GT-aware
    ├── threshold_sweep.py                 # sweep N thresholds on a sample
    └── THRESHOLD_DECISION.md              # why threshold 90 + logi_glue intra-skip
```

## Step details

### Step 0 — Merge

Streaming PyArrow `ParquetWriter`. Pre-flight schema validation confirms
all 9 files share the identical 19-column `rl_schema_v1` schema. RAM-guard
(default 40 GB free) before each file. Resumable via JSON checkpoint.

**Result:** 1,127,950 rows in one parquet (no rows dropped at this step).

### Step 3B — Exact-hash dedup, GT-aware

Cross-dataset SHA1 hash of the **normalised last user message** of
`context_messages`. (Hashing the full context would never match because
RL prompts heavily reuse system boilerplate.) Normalisation: lowercase →
strip punctuation → collapse whitespace.

Policy:
- **Cross-dataset duplicate prompt** → DROP (first-seen-wins, by verifier
  strength).
- **Intra-dataset duplicate prompt + identical** ground-truth fingerprint
  (`sha1(strip(ground_truth_text))`) → DROP.
- **Intra-dataset duplicate prompt + different GT** → KEEP (legitimate
  multi-answer).
- 7 SLR-Bench multilingual subsets → **passthrough** (synthetic, no
  resource overlap; never indexed, never hashed).

**Result:** 1,127,950 → 1,105,317 (−22,633, −2.0 %)
* dropped cross-DS: 5,370
* dropped intra-DS same-GT: 17,263
* kept intra-DS diff-GT (multi-answer): 5,263
* passthrough (SLR-Bench): 126,189

### Step 3C — Fuzzy dedup, GT-aware

RapidFuzz `fuzz.ratio` (Levenshtein, normalised 0–100) on the normalised
last user message. Candidates are bucketed by leading-token prefix and
length bin to keep cost tractable; 32-way `ProcessPoolExecutor` for
parallel matching.

**Threshold = 90**, chosen after a 4-way empirical sweep (78 / 85 / 90 / 95)
on a 100,000-row sample. Full rationale in
[`THRESHOLD_DECISION.md`](deduplication/THRESHOLD_DECISION.md).

Policy carries over from 3B with one exception:

- **`logicreasoning/logi_glue` is excluded from intra-DS dedup**, but still
  participates in cross-DS detection. Manual inspection at threshold 90
  showed that templated multiple-choice / NLI prompts in logi_glue share
  ~95 % of their text with answer-relevant variation in only 1-2 tokens
  (distractor options, NLI hypothesis), and with a tiny GT label space
  ({entailment, contradiction, neutral} / {True, False}), unrelated tasks
  frequently share the correct label, defeating the GT-aware safeguard.

**Result:** 1,105,317 → 1,084,071 (−21,246, −1.9 %)
* dropped cross-DS: 1,109
* dropped intra-DS same-GT: 20,137
* kept intra-DS diff-GT (multi-answer): 74,053
* passthrough (SLR-Bench): 126,189

## Running

```bash
python3 stage_3/deduplication/stage_3_00_merge.py \
    --in-root corpora/rl/ \
    --out-root corpora/rl/merged_rl_ds/

python3 stage_3/deduplication/stage_3b_exact_hash_dedup.py \
    --input corpora/rl/merged_rl_ds/rl_v1_merged.kept.parquet \
    --output-dir corpora/rl/dedup_3b_exact/

# Optional sanity check before fuzzy:
python3 stage_3/deduplication/threshold_sweep.py \
    --input corpora/rl/dedup_3b_exact/rl_v1.exact_dedup.kept.parquet \
    --sample-size 100000 \
    --thresholds 78 85 90 95

python3 stage_3/deduplication/stage_3c_fuzzy_dedup.py \
    --input corpora/rl/dedup_3b_exact/rl_v1.exact_dedup.kept.parquet \
    --output-dir corpora/rl/dedup_3c_fuzzy/ \
    --similarity-threshold 90
```

All scripts support `--resume` and write a `report.json` next to their
output. Stage 3B uses a SQLite-backed prompt index for memory efficiency;
Stage 3C uses an in-memory bucketed index plus a `ProcessPoolExecutor`
fan-out.
