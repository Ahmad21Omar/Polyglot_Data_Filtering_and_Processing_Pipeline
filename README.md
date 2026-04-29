# SFT-Collection-v2 — Filtering Pipeline

> **Master Thesis: Polyglot Thoughts — Synthesizing Reasoning Data for Multilingual Post-Training**
> Ahmad Omar | TU Darmstadt, AIML Lab | Supervisors: Prof. Kersting, Lukas Helff

This repository documents and reproduces the full data filtering pipeline used to build
**[SFT-Collection-v2](https://huggingface.co/datasets/ahmad21omar/SFT-Collection-v2)**,
a multilingual supervised fine-tuning dataset for reasoning-capable language models.

---

## Dataset at a glance

| Property | Value |
|---|---|
| HuggingFace ID | `ahmad21omar/SFT-Collection-v2` |
| Total rows | 23,896,757 |
| Tokens | ~89.2B |
| Languages | EN, DE, FR, IT, ES, JA |
| Configs | 11 |
| Key columns | `reasoning_text`, `response_text`, `language`, `domain`, `quality_score` |

---

## Pipeline overview

```
Stage 0 — Download             15 public HuggingFace datasets   85.7M rows
              ↓
Stage 1 — Dataset Elimination  source-level quality gates        57.0M  (-33.5%)
              ↓
Stage 2A — Heuristic Filter    per-row text quality checks       19.9M  (-65.1%)
              ↓
Stage 2B — Quality Score       FineWeb-HQ classifier ≥ 0.2      19.3M  (-3.1%)
              ↓
Stage 3A — Same-ID Dedup        intra-dataset exact-ID dedup      19.3M  (-0.0%)
              ↓
Stage 3B — Exact-Hash Dedup    SHA1(normalize(context)), cross-ds  ~19M  (-1–2%)
              ↓
Stage 3C — Fuzzy Dedup         RapidFuzz Levenshtein, cross-ds    17.1M  (-11%)
              ↓
Stage 4a — Post-processing     schema clean-up, domain tags      17.1M  (-0.17%)
              ↓
Stage 4b — Soofi Multilingual  add translated data (DE/FR/IT/ES/JA) +6.8M
              ↓
                                                   Final: 23.9M rows
```

Each stage is documented in its own section of this repo (added incrementally).

---

## Repository structure

```
Filtering_Pipeline/
├── README.md                  ← this file
├── data_loader.py             ← Stage 0: download source datasets         ✅
├── filters/                   ← Stage 2A: heuristic filter modules         ✅
│   ├── README.md
│   ├── phase_2a1_structural.py     2A.1  messages / assistant presence, prompt length
│   ├── phase_2a2_think_reasoning.py 2A.2  think-tag presence, min chars, truncation
│   ├── phase_2a3_prompt_question.py 2A.3  URL / image / multipart / length heuristics
│   ├── phase_2a4_language.py       2A.4  FastText English LID, 218-lang detector, Chinese ratio
│   ├── phase_2a5_identity_safety.py 2A.5  identity self-id, cutoff phrasing, safety flags
│   ├── phase_2a6_repetition.py     2A.6  sentence / phrase mass repetition
│   ├── _utils/                     cross-cutting helpers (code-task, FastText classifier)
│   └── (canonical originals: format_filters.py, question_filters.py, language_filters.py,
│        english_fasttext_filter.py, language_detector.py, identity_filters.py,
│        content_filters.py, domain_filters.py, repetition_filters.py,
│        code_task_filters.py, fasttext_filters.py — re-exported by the phase modules above)
├── filter_and_format/         ← Stage 2A: generic pipeline + per-dataset adapters ✅
│   ├── README.md
│   ├── config.py              FilterConfig + presets (english_only / multilingual)
│   ├── output_schema.py       canonical SFT-Collection-v2 row schema
│   ├── adapter.py             DatasetAdapter ABC
│   ├── pipeline.py            6-phase row-level filter (canonical Sankey order)
│   ├── runner.py              shared CLI: load + adapt + filter + save
│   └── adapters/              13 per-dataset adapters (one per source dataset)
├── quality_filtering/         ← Stage 2B: quality scoring + threshold filtering ✅
│   ├── README.md
│   ├── text_assembly.py       build_scoring_text(row) — shared text builder
│   ├── english_fasttext.py    EnglishFastTextQualityScorer (FastText OH-ELI5)
│   ├── multilingual_fineweb_hq.py  MultilingualFineWebHqScorer (XLM-R + FineWeb-HQ heads)
│   └── runner.py              run_quality_filter(scorer, input_path, output_dir, …)
├── deduplication/             ← Stage 3: three-pass deduplication               ✅
│   ├── stage3a_same_id_dedup.py      3A  intra-dataset duplicate example_ids   ✅
│   ├── stage3b_exact_hash_dedup.py   3B  cross-dataset SHA1 exact dedup        ✅
│   └── stage3c_fuzzy_dedup.py        3C  cross-dataset RapidFuzz fuzzy dedup   ✅
├── postprocess/               ← Stage 4A: clean-up, domain tags, lang filter   ✅
│   ├── stage4a_clean_dataset.py      4A  ground-truth fix, domain backfill, language filter
│   └── domain_classifier/
│       ├── train.py                  train FastText domain classifier on labeled rows
│       └── predict.py                predict domains for null-domain rows → CSV
└── soofi_multilingual/        ← Stage 4B: Soofi multilingual pipeline            ✅
    ├── stage4b_step2a_soofi_non_english.py   4B step 2A  filter + normalise DE/FR/IT/ES splits
    ├── stage4b_step2b_soofi_quality_filter.py  4B step 2B  FineWeb-HQ quality filter per language
    └── stage4b_step3_soofi_dedup.py          4B step 3   intra-language dedup (exact SHA-1 → fuzzy RapidFuzz)
```

---

## Stage 0 — Downloading source datasets (`data_loader.py`)

### Source datasets

| # | HuggingFace ID | Lang | Domain | Raw columns (verified) |
|---|---|---|---|---|
| 1 | `allenai/Dolci-Think-SFT-7B` | EN | Mixed | `messages`, `dataset_source`, `id` |
| 2 | `open-thoughts/OpenThoughts3-1.2M` | EN | Math/Code/Science | `difficulty`, `source`, `domain`, `conversations` |
| 3 | `open-r1/Mixture-of-Thoughts` | EN | Math/Code/Science | `messages`, `num_tokens`, `source` |
| 4 | `nvidia/Nemotron-Post-Training-Dataset-v2` | EN+Multilingual | Math/Code/STEM/Chat | `uuid`, `license`, `generator`, `version`, `category`, `reasoning`, `messages` |
| 5 | `nvidia/Llama-Nemotron-Post-Training-Dataset` | EN | Math/Code/Science/Chat | `input`, `output`, `category`, `license`, `reasoning`, `generator` |
| 6 | `nvidia/Nemotron-Math-v2` | EN | Math | `uuid`, `expected_answer`, `problem`, `messages`, `data_source`, `metadata` |
| 7 | `nvidia/Nemotron-Math-Proofs-v1` | EN | Math (Lean proofs) | `problem`, `source`, `formal_statement`, `lean_header`, `messages` |
| 8 | `nvidia/Nemotron-Competitive-Programming-v1` | EN | Code (C++/Python) | `uuid`, `messages`, `license`, `used_in`, `tools`, `difficulty`, `question_id` |
| 9 | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | EN | Mixed | `messages` (with nested `info`) |
| 10 | `PrimeIntellect/SYNTHETIC-2-SFT-verified` | EN | Mixed | `problem_id`, `task_type`, `reward`, `messages` |
| 11 | `toroe/Soofi-Think-SFT-10B-multilingual` | DE/FR/IT/ES/EN | Multilingual | `messages`, `source`, `dataset_name`, `ds_uid`, `language` |

> Datasets 12–14 are internal AIML-Lab datasets not publicly available.

### Verified downloads

All functions were smoke-tested with `max_examples=5` against a temporary directory:

| Function | Status | Notes |
|---|---|---|
| `download_openthoughts3` | ✅ | |
| `download_dolci_think_sft_7b` | ✅ | |
| `download_mixture_of_thoughts` | ✅ | 3 configs: math/code/science |
| `download_nemotron_math_v2` | ✅ | needs `split` kwarg (default: all 5 splits) |
| `download_nemotron_math_proofs_v1` | ✅ | default split: `lean` |
| `download_nemotron_competitive_programming_v1` | ✅ | requires explicit Features schema (HF bug) |
| `download_synthetic_2_sft_verified` | ✅ | |
| `download_nemotron_post_training_v2` | ✅ | 9 splits incl. multilingual_de/fr/it/es/ja |
| `download_llama_nemotron_post_training` | ✅ | subset=SFT; 5 task splits |
| `download_soofi_think_sft_10b_multilingual` | ✅ | languages arg: english/italian/french/spanish/german |
| `download_am_deepseek_r1_distilled_1p4m` | ✅ | requires explicit Features schema (HF bug) |

### How to download

**Smoke-test via CLI (5 rows, output to custom dir):**

```bash
python data_loader.py --dataset openthoughts3 --max-examples 5 --output-dir /tmp/sft_test
```

Available `--dataset` values:
`openthoughts3`, `dolci`, `synthetic2`, `mixture_of_thoughts`, `nemotron_math_proofs`,
`nemotron_competitive`, `nemotron_math_v2`, `am_deepseek`, `llama_nemotron`, `nemotron_v2`,
`soofi_multilingual`

**Full download via Python:**

```python
from Filtering_Pipeline.data_loader import download_nemotron_post_training_v2

paths = download_nemotron_post_training_v2(
    splits=["math", "code"],       # omit for all 9 splits
    output_dir="/mnt/data/sft",    # optional; falls back to SFT_DATA_ROOT env var
)
```

**Via environment variable (applies to all functions without explicit output_dir):**

```bash
export SFT_DATA_ROOT=/mnt/data/sft
python data_loader.py --dataset dolci
```

### Output layout

Datasets are saved in HuggingFace Arrow format, loadable with `datasets.load_from_disk()`:

```
<output_dir>/
  open-thoughts__OpenThoughts3-1.2M/
    train/                          ← full download
    train__first_1000/              ← smoke-test (max_examples=1000)
  nvidia__Nemotron-Post-Training-Dataset-v2/
    SFT/
      math/
      code/
      multilingual_de/
      ...
  translated_ds/
    toroe__Soofi-Think-SFT-10B-multilingual/
      english/
      german/
      ...
```

### Special cases

**`nvidia/Nemotron-Competitive-Programming-v1`** and **`a-m-team/AM-DeepSeek-R1-Distilled-1.4M`**
require an explicit `Features` schema passed to `load_dataset`. Without it, HuggingFace fails
to infer nullable or nested fields. The schema is hardcoded in the respective download functions.

### Storage requirements

| Dataset | Approx. disk (full download) |
|---|---|
| `nvidia/Llama-Nemotron-Post-Training-Dataset` | ~200 GB |
| `nvidia/Nemotron-Post-Training-Dataset-v2` | ~80 GB |
| `open-thoughts/OpenThoughts3-1.2M` | ~20 GB |
| `toroe/Soofi-Think-SFT-10B-multilingual` | ~15 GB |
| All others combined | ~30 GB |

Use `max_examples=N` for development and testing.

### Requirements

```bash
pip install datasets huggingface_hub
```

For gated datasets, authenticate first:

```bash
huggingface-cli login
```

---

## Reproducibility notes

- All scripts are designed to run locally without GPU.
- On shared servers, limit RAM per process before running heavy scripts:
  ```bash
  ulimit -v $((350 * 1024 * 1024)) && python data_loader.py --dataset llama_nemotron
  ```
- Never use `pd.read_parquet()` on large files — use PyArrow batched reading (see pipeline scripts).

---

---

## Stage 2B — Quality filtering (`quality_filtering/`)

After Stage 2A removes structurally bad rows, Stage 2B scores each surviving row with a
learned quality classifier and drops the lowest-quality ones.

| Language | Classifier | Module |
|---|---|---|
| English-only | FastText OH-ELI5 (`mlfoundations/fasttext-oh-eli5`) | `english_fasttext.py` |
| Multilingual (DE, FR, IT, ES, JA, …) | FineWeb-HQ head on XLM-RoBERTa-base | `multilingual_fineweb_hq.py` |

### Threshold policy

1. If `score_threshold` is passed explicitly → use it as-is.
2. Else if `base_threshold == 0` → keep everything (scores written through).
3. Else if `< base_threshold` would drop **at most** `max_drop_rate` → use `base_threshold`.
4. Else → **lower** the threshold to the `max_drop_rate` percentile (never raised above base).

Default values: `base_threshold=0.20`, `max_drop_rate=0.20` (English); `base_threshold=0.50` (multilingual, sigmoid-calibrated).

### Quickstart

```python
from Filtering_Pipeline.quality_filtering import EnglishFastTextQualityScorer, run_quality_filter

scorer = EnglishFastTextQualityScorer(
    model_path="models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
)
summary = run_quality_filter(
    scorer=scorer,
    input_path="/path/to/sft_dolci_v0.kept.parquet",
    output_dir="/path/to/quality_filtered/dolci",
)
print(summary["n_kept"], summary["n_dropped"], summary["threshold"])
```

See [`quality_filtering/README.md`](quality_filtering/README.md) for the full documentation,
model download snippets, and multilingual usage.

---

---

## Stage 3 — Deduplication (`deduplication/`)

Three sequential passes remove duplicate rows from the quality-filtered corpus.
Each pass is a standalone script with no hardcoded paths.

```
Stage 3A → Stage 3B → Stage 3C
```

### Stage 3A — Intra-dataset Same-ID Deduplication (`stage3a_same_id_dedup.py`)

Removes rows where the same `(dataset_id, example_id)` pair appears more than
once **within the same dataset**.  Cross-dataset collisions on `example_id` are
left to Stage 3B.

**Algorithm (two-pass streaming):**
- Pass 1 reads only the two key columns (`dataset_id`, `example_id`) — lightweight, ~200 MB RAM for 20 M rows.
- Pass 2 streams the full schema and writes rows based on the keep-set from Pass 1.

```bash
python deduplication/stage3a_same_id_dedup.py \
    --input  /path/to/quality_filtered/merged_sft.kept.parquet \
    --output-dir /path/to/stage3a_same_id_dedup
```

Outputs: `merged_sft.same_id_dedup.kept.parquet`, `merged_sft.same_id_dedup.dropped.parquet`, `summary.json`

---

### Stage 3B — Cross-Dataset Exact-Hash Deduplication (`stage3b_exact_hash_dedup.py`)

Removes rows whose context prompt is an exact duplicate of a prompt already seen in a
**different** dataset.  Same-dataset duplicates are kept (Stage 3A handles those).

**Hash function:** `SHA1(lowercase(strip_punctuation(collapse_whitespace(context_messages))))`

**Policy:** first-seen-wins across datasets.

**State persistence:** a SQLite file stores the hash → first_dataset mapping, enabling
`--resume` to continue interrupted runs.

```bash
python deduplication/stage3b_exact_hash_dedup.py \
    --input  /path/to/stage3a/merged_sft.same_id_dedup.kept.parquet \
    --output-dir /path/to/stage3b_exact_dedup

# Resume an interrupted run:
python deduplication/stage3b_exact_hash_dedup.py \
    --input  /path/to/... --output-dir /path/to/... --resume
```

| Flag | Default | Description |
|---|---|---|
| `--hash-algo` | `sha1` | Hash algorithm: `sha1`, `md5`, `sha256` |
| `--batch-size` | `50000` | Rows per batch |
| `--resume` | off | Resume from checkpoint in `--output-dir` |
| `--checkpoint-every` | `20` | Flush checkpoint every N batches |

Outputs: `merged_sft.exact_dedup.kept.parquet`, `merged_sft.exact_dedup.dropped.parquet`, `report.json`

Extra debug columns added to output parquets (can be dropped in post-processing):
`_dedup_hash`, `_dedup_norm`, `_dedup_seen_in_ds` (kept) / `_dedup_matched_ds` (dropped)

---

### Stage 3C — Cross-Dataset Fuzzy Deduplication (`stage3c_fuzzy_dedup.py`)

Removes rows whose context prompt is a *near-duplicate* of a prompt already seen in a
different dataset, using **RapidFuzz `fuzz.ratio`** (Levenshtein similarity, normalized 0–100).

Inspired by the [OpenThoughts3 deduplication approach](https://arxiv.org/abs/2506.04178).

#### Similarity threshold

| Setting | Threshold | Behaviour |
|---|---|---|
| **SFT-Collection-v2** (default) | **78** | Aggressive — catches paraphrases and lightly reworded prompts |
| **OpenThoughts3** | 95 | Conservative — only near-identical prompts are removed |

We chose **threshold 78** for SFT-Collection-v2 because the dataset mix contains several
overlapping sources (e.g. Nemotron variants, OpenThoughts3, Soofi) where prompts are
frequently paraphrased rather than copied verbatim.  A higher threshold leaves many
near-duplicates in.  If you are working with a single-source dataset or want to preserve
borderline cases, use `--similarity-threshold 95`.

#### Bucketing for efficiency

Comparing every pair of 19 M rows with fuzz.ratio would be O(n²).  We use a two-level
bucket filter to reduce candidate sets:

1. **Bucket key** = first 2 non-role words of the normalized text + `len(text) // 100`
2. **Neighbour bins** (±1 length bin) are also checked to catch near-length matches.
3. **Length diff pre-filter**: candidates with `|len(query) − len(candidate)| > 40` are skipped.
4. **Parallel matching**: comparisons inside each batch are distributed across workers via `ProcessPoolExecutor`.

#### Usage

```bash
# Reproduce SFT-Collection-v2 (threshold 78, default):
python deduplication/stage3c_fuzzy_dedup.py \
    --input  /path/to/stage3b/merged_sft.exact_dedup.kept.parquet \
    --output-dir /path/to/stage3c_fuzzy_dedup

# OpenThoughts3-style (threshold 95):
python deduplication/stage3c_fuzzy_dedup.py \
    --input  /path/to/... \
    --output-dir /path/to/... \
    --similarity-threshold 95

# Resume an interrupted run:
python deduplication/stage3c_fuzzy_dedup.py \
    --input  /path/to/... --output-dir /path/to/... --resume
```

| Flag | Default | Description |
|---|---|---|
| `--similarity-threshold` | `78` | fuzz.ratio cutoff (0–100).  78 = SFT-Collection-v2; 95 = OpenThoughts3 |
| `--batch-size` | `20000` | Rows per batch |
| `--num-workers` | `32` | ProcessPoolExecutor workers |
| `--bucket-tokens` | `2` | Leading tokens in bucket key |
| `--len-bin-size` | `100` | Length bin granularity |
| `--max-candidates-per-bucket` | `5000` | Cap per bucket (oldest evicted first) |
| `--max-len-diff` | `40` | Skip candidates with length difference larger than this (`-1` = disabled) |
| `--neighbor-bin-radius` | `1` | Search ±N adjacent length bins |
| `--resume` | off | Continue from checkpoint in `--output-dir` |

Outputs: `merged_sft.fuzzy_dedup.kept.parquet`, `merged_sft.fuzzy_dedup.dropped.parquet`, `report.json`

Extra debug columns in dropped parquet: `_drop_reason`, `_fuzzy_score`, `_fuzzy_matched_dataset`, `_fuzzy_matched_example`

#### Requirements

```bash
pip install rapidfuzz pyarrow
```

---

---

## Stage 4A — Post-Processing (`postprocess/`)

After fuzzy deduplication (Stage 3C), Stage 4A applies four sequential cleanup steps
to produce the final English SFT corpus before the multilingual extension (Stage 4B).

### Steps

| # | Step | Type | Effect |
|---|---|---|---|
| 1 | `fix_ground_truth_present` | Transform | If `ground_truth_present=True` but `final_answer_text` is null/empty → set to `False` |
| 2 | `fix_null_domains` | Transform | Rule-based domain backfill for 5 known `saumyamalik` subsources |
| 3 | `apply_ml_predictions` | Transform | FastText ML backfill for remaining null-domain rows |
| 4 | `filter_language` | Filter | Drop rows where language doesn't match category (e.g. non-EN in default category) |

Steps 1–3 are **transforms** (column values are modified, no rows are dropped).
Step 4 is a **filter** (failing rows go to the dropped parquet).

### Domain backfill — rule-based (Step 2)

Five saumyamalik subsources have an unambiguous domain that can be assigned from `subsource_raw`:

| subsource_raw | domain assigned |
|---|---|
| `saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2` | `code` |
| `saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2` | `code` |
| `saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2` | `science` |
| `saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2` | `math` |
| `saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2` | `reasoning` |

### Domain backfill — ML (Step 3)

Remaining null-domain rows (not covered by Step 2) are classified by a FastText model
trained on the labeled majority of the corpus.

**Workflow:**

```
1. Train:   domain_classifier/train.py   → domain_classifier.bin
2. Predict: domain_classifier/predict.py → null_domain_predictions.csv
3. Apply:   stage4a_clean_dataset.py --predictions-csv null_domain_predictions.csv
```

FastText hyperparameters follow [OpenThoughts3 Appendix R.2.1](https://arxiv.org/abs/2506.04178):
`dim=256, epoch=3, lr=0.1, wordNgrams=2, minCount=3`, with class balancing (max 500k per domain).

### Language filter (Step 4)

| Row category | Allowed language |
|---|---|
| `null` / anything not `multilingual_*` | `en` |
| `multilingual_de/fr/it/es/ja` | `de`, `fr`, `it`, `es`, `ja` |

Rows failing this check are written to the dropped parquet for inspection.

### Usage

**Step 1 — Train the domain classifier:**

```bash
python postprocess/domain_classifier/train.py \
    --input  /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \
    --output /path/to/domain_classifier/domain_classifier.bin
```

**Step 2 — Predict null-domain rows:**

```bash
python postprocess/domain_classifier/predict.py \
    --input  /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \
    --model  /path/to/domain_classifier/domain_classifier.bin \
    --output /path/to/domain_classifier/null_domain_predictions.csv
```

**Step 3 — Run the full clean pass:**

```bash
python postprocess/stage4a_clean_dataset.py \
    --input           /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \
    --output-dir      /path/to/stage4a_out/ \
    --predictions-csv /path/to/domain_classifier/null_domain_predictions.csv
```

**Smoke test (no predictions CSV needed — ML step is skipped):**

```bash
python postprocess/stage4a_clean_dataset.py \
    --input      /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \
    --output-dir /tmp/stage4a_test/ \
    --max-rows   1000
```

| Flag | Default | Description |
|---|---|---|
| `--predictions-csv` | _(none)_ | CSV from `predict.py`. If omitted, ML step is skipped. |
| `--batch-size` | `200000` | Rows per streaming batch |
| `--max-rows` | _(none)_ | Stop after N rows (dry-run) |

**Outputs:**

| File | Description |
|---|---|
| `merged_sft.v2.kept.parquet` | Cleaned dataset (pipeline input for Stage 4B) |
| `merged_sft.v2.dropped.language_filter.parquet` | Rows removed by Step 4 |
| `summary.json` | Row counts, drop rate, config |

### Requirements

```bash
pip install fasttext pyarrow
```

> **Note:** fasttext requires a C++ compiler.  On most Linux systems:
> `apt-get install -y build-essential` is sufficient.
> For numpy ≥ 2.0, both `train.py` and `predict.py` apply a compatibility patch automatically.

---

---

## Stage 4B — Step 2A: Soofi Non-English Filter & Format (`soofi_multilingual/`)

Filters and normalises the four non-English splits of
**`toroe/Soofi-Think-SFT-10B-multilingual`** (DE, FR, IT, ES) into the canonical
SFT-Collection-v2 schema.  Each language is processed independently and contributes
~1.4–1.7M rows to the final multilingual extension.

### How it works

`stage4b_step2a_soofi_non_english.py` is a thin wrapper around the existing
`filter_and_format/` framework already present in this repository:

```
SoofiThinkSft10bMultilingualAdapter  (filter_and_format/adapters/)
        ↓
filter_and_format/pipeline.py   ← 6-phase filter (same as Stage 2A)
        ↓
filter_and_format/runner.py     ← load_from_disk + ds.map + save parquet
```

The adapter handles:
- Language name → ISO 639-1 conversion (`"german"` → `"de"`, etc.)
- `domain` inference from `dataset_name` / `source`
- `example_id` derivation (SHA-256 of `dataset_id|split|ds_uid`)

### Filter configuration

The configuration reproduces the original SFT-Collection-v2 run.
Two thresholds differ from the pipeline's current defaults:

| Phase | Setting | Value used | Pipeline default |
|---|---|---|---|
| 2A.2 Think | `min_think_chars` | **50** | 80 |
| 2A.3 Question | `question_min_length` | **50** | 200 |
| 2A.4 Language | `enable_fasttext_english_filter` | **False** | False |
| 2A.4 Language | `enable_mixed_language_filter` | **False** | False |

Both language filters are disabled because the rows are intentionally non-English.

### Input layout

The input directory must contain one HuggingFace Arrow dataset (saved by
`data_loader.py`) per language:

```
<input-root>/
    german/         ← load_from_disk()-compatible Arrow dataset
    french/
    italian/
    spanish/
```

Use `data_loader.py --dataset soofi_multilingual` to produce this layout.

### Usage

```bash
# Full run (all four languages):
python soofi_multilingual/stage4b_step2a_soofi_non_english.py \
    --input-root /path/to/toroe__Soofi-Think-SFT-10B-multilingual \
    --output-dir /path/to/stage4b_step2a_soofi_non_english

# Smoke test (200 rows per language):
python soofi_multilingual/stage4b_step2a_soofi_non_english.py \
    --input-root /path/to/toroe__Soofi-Think-SFT-10B-multilingual \
    --output-dir /tmp/stage4b_test \
    --max-examples 200

# Only German and French:
python soofi_multilingual/stage4b_step2a_soofi_non_english.py \
    --input-root /path/to/... \
    --output-dir /path/to/... \
    --languages german,french
```

> **Note:** Run from `Master_Thesis/` (the parent of `Filtering_Pipeline/`) or
> from within `Filtering_Pipeline/` — the script adjusts `sys.path` automatically.

| Flag | Default | Description |
|---|---|---|
| `--languages` | `german,french,italian,spanish` | Language subfolder names to process |
| `--max-examples` | _(none)_ | Stop after N rows per language (smoke test) |
| `--num-proc` | `1` | ds.map parallel workers |

### Output layout

```
<output-dir>/
    german/
        kept.parquet       ← filtered rows in canonical schema
        dropped.parquet    ← audit rows with _drop_reason
        summary.json       ← per-language counts + drop reasons
    french/
        ...
    italian/
        ...
    spanish/
        ...
    summary.json           ← combined stats for all languages
```

### Requirements

```bash
pip install datasets huggingface_hub pyarrow
```

---

## Stage 4B — Step 2B: Soofi Multilingual Quality Filter (`soofi_multilingual/`)

Scores each non-English Soofi split (output of Stage 4B) with the
**FineWeb-HQ + XLM-R** quality classifier and removes low-quality rows.

### How it works

`stage4b_step2b_soofi_quality_filter.py` is a thin wrapper around the existing
`quality_filtering/` framework already present in this repository:

```
MultilingualFineWebHqScorer          (quality_filtering/multilingual_fineweb_hq.py)
        ↓  XLM-R base, frozen — mean pooling → language-specific MLP head → sigmoid
run_quality_filter(...)              (quality_filtering/runner.py)
        ↓  reads kept.parquet, scores, writes kept/dropped + stats
```

For each language the scorer loads the corresponding FineWeb-HQ classifier head
(`deu_Latn.pt` → DE, `fra_Latn.pt` → FR, `ita_Latn.pt` → IT, `spa_Latn.pt` → ES)
and scores the concatenation of all user turns and the response with `[SEP]` delimiters.

### Classifier heads — one-off download

```python
from huggingface_hub import hf_hub_download
for fname in ("deu_Latn.pt", "fra_Latn.pt", "ita_Latn.pt", "spa_Latn.pt"):
    hf_hub_download(
        repo_id="epfml/FineWeb-HQ-Classifiers",
        filename=fname,
        local_dir="models/FineWeb-HQ-Classifiers",
    )
```

### Usage

```bash
# Full run:
python soofi_multilingual/stage4c_soofi_quality_filter.py \
    --input-root      /path/to/stage4b_step2a_soofi_non_english \
    --classifiers-dir /path/to/models/FineWeb-HQ-Classifiers \
    --output-dir      /path/to/stage4c_soofi_quality_filter \
    --threshold       0.45

# Smoke test (200 rows per language):
python soofi_multilingual/stage4c_soofi_quality_filter.py \
    --input-root      /path/to/stage4b_step2a_soofi_non_english \
    --classifiers-dir /path/to/models/FineWeb-HQ-Classifiers \
    --output-dir      /tmp/stage4c_test \
    --threshold       0.45 \
    --max-rows        200

# GPU with larger batch:
python soofi_multilingual/stage4c_soofi_quality_filter.py \
    --input-root      /path/to/stage4b_step2a_soofi_non_english \
    --classifiers-dir /path/to/models/FineWeb-HQ-Classifiers \
    --output-dir      /path/to/stage4c_out \
    --threshold       0.45 \
    --device          cuda \
    --batch-size      64
```

| Flag | Default | Description |
|---|---|---|
| `--input-root` | _(required)_ | Stage 4B output root (contains `german/kept.parquet`, etc.) |
| `--classifiers-dir` | _(required)_ | Directory with `deu_Latn.pt`, `fra_Latn.pt`, etc. |
| `--output-dir` | _(required)_ | Where to write per-language output + `summary.json` |
| `--threshold` | _(required)_ | Quality score cutoff in [0, 1] |
| `--languages` | `german,french,italian,spanish` | Language subfolder names to process |
| `--batch-size` | `8` | XLM-R scoring batch size |
| `--device` | auto (cuda if available) | `"cuda"` / `"cpu"` |
| `--max-rows` | _(none)_ | Stop after N rows per language (smoke test) |

### Output layout

```
<output-dir>/
    german/
        kept.parquet     ← quality_score >= threshold  (+_quality_score, _quality_label columns)
        dropped.parquet  ← quality_score < threshold
        stats.json       ← threshold, score percentiles, n_kept, n_dropped
    french/
        ...
    italian/
        ...
    spanish/
        ...
    summary.json         ← combined totals for all languages
```

### Requirements

```bash
pip install torch transformers pyarrow pandas
# (huggingface_hub for the one-off head download)
pip install huggingface_hub
```

---

## Stage 4B — Step 3: Soofi Intra-Language Deduplication (`soofi_multilingual/`)

Deduplicates each non-English Soofi split (output of Step 2B) within the
language split itself.  The Soofi multilingual dataset contains rows from
several sub-sources (e.g. `HuggingFaceH4/aya_dataset`, `CohereForAI/aya_dataset`,
…).  Cross-source exact and near-duplicate pairs are removed; within-source
duplicates are kept.

### Two-pass approach

```
quality_filtered kept.parquet
        │
        ▼ Pass 1 — Exact (SHA-1)
  SHA-1( normalize(context_messages) )
  seen in different sub-source? → DROP   same sub-source? → KEEP
        │
        ▼ adds _dedup_hash / _dedup_prompt_norm / _dedup_seen_in_source
        │
        ▼ Pass 2 — Fuzzy (RapidFuzz fuzz.ratio, default 98%)
  bucket-indexed candidate lookup (first-2-token head + length bin)
  match ≥ threshold in different sub-source? → DROP   else → KEEP
        │
        ▼ final kept.parquet  (use this downstream)
```

### Source policy

The sub-source name is derived from `source_dataset_id` (everything before
the last `/`).  Two rows are considered cross-source if their sub-source
names differ — this is the same policy used in Stage 3B/3C for the main
English corpus.

### Usage

```bash
# Full run:
python soofi_multilingual/stage4b_step3_soofi_dedup.py \
    --input-root /path/to/stage4b_quality_filter \
    --output-dir /path/to/stage4b_dedup

# Smoke test (5 000 rows per language):
python soofi_multilingual/stage4b_step3_soofi_dedup.py \
    --input-root /path/to/stage4b_quality_filter \
    --output-dir /tmp/stage4b_dedup_test \
    --max-rows   5000

# Resume after interruption:
python soofi_multilingual/stage4b_step3_soofi_dedup.py \
    --input-root /path/to/stage4b_quality_filter \
    --output-dir /path/to/stage4b_dedup \
    --resume
```

| Flag | Default | Description |
|---|---|---|
| `--input-root` | _(required)_ | Stage 4B Step 2B output root (`german/kept.parquet`, …) |
| `--output-dir` | _(required)_ | Where to write per-language outputs + `summary.json` |
| `--languages` | `german,french,italian,spanish` | Subfolder names to process |
| `--similarity-threshold` | `98.0` | Fuzzy threshold (fuzz.ratio, 0–100) |
| `--hash-algo` | `sha1` | Hash for exact pass (`sha1` / `md5` / `sha256`) |
| `--batch-size-exact` | `50000` | Batch size for exact pass |
| `--batch-size-fuzzy` | `20000` | Batch size for fuzzy pass |
| `--num-workers` | `8` | Parallel workers for RapidFuzz comparisons |
| `--max-rows` | _(none)_ | Stop after N rows per language (smoke test) |
| `--resume` | `False` | Resume both passes from checkpoints |

### Output layout

```
<output-dir>/
    german/
        exact/
            kept.parquet          ← after pass 1 (input to pass 2)
            dropped.parquet       ← exact cross-source duplicates
            report.json
        fuzzy/
            kept.parquet          ← FINAL output — use this downstream
            dropped.parquet       ← fuzzy cross-source duplicates
            report.json
    french/  ...
    italian/ ...
    spanish/ ...
    summary.json                  ← combined stats (exact + fuzzy, all languages)
```

### Requirements

```bash
pip install pyarrow rapidfuzz
```
