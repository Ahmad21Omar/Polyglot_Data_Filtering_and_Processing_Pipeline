# Stage 4 — Soofi English Filtering Pipeline

Three-step pipeline for the English split of `toroe/Soofi-Think-SFT-10B-multilingual`,
producing data that **supplements the main English SFT corpus**.

Because the Soofi English split is added on top of an already high-quality corpus,
Step 2B uses a **stricter quality threshold (τ = 0.72)** than the main corpus (τ = 0.20).
Step 3 is **cross-collection fuzzy deduplication** of the survivors against the full
merged main English SFT corpus at τ_fuzzy = 90 % (`fuzz.ratio` / Levenshtein) —
there is no intra-Soofi exact/fuzzy pass here.

```
soofi_english/
  ├─ stage4_step2a_soofi_english.py
  │    Heuristic filter + format (structural, prompt, think, etc.)
  │
  ├─ stage4_step2b_soofi_english_quality_filter.py
  │    Quality filtering with FastText OH-ELI5 (recommended: τ = 0.72)
  │
  ├─ stage4_step3_soofi_english_dedup.py
  │    Cross-collection fuzzy dedup vs. main English SFT corpus (τ_fuzzy = 90 %)
  │
  └─ README.md
       This file
```

## Reference numbers (Thesis Table 9.11)

| Step | In | Out | Dropped | Drop % |
|---|---|---|---|---|
| 2A Heuristic filter | 2,283,204 | 1,701,713 | −581,491 | −25.5 % |
| 2B Quality (τ = 0.72) | 1,701,713 | 1,535,565 | −166,148 | −9.8 % |
| 3 Cross-dedup vs. main corpus (fuzzy, τ = 90 %) | 1,535,565 | 1,314,868 | −220,697 | −14.4 % |

## Stage 4 — Step 2A: Soofi English Heuristic Filter

**Input**: Arrow dataset from data_loader.py (English split of `toroe/Soofi-Think-SFT-10B-multilingual`)

**Output**: `kept.parquet` + `dropped.parquet` in canonical schema

**Filter suite** (reproduces original SFT-Collection-v2 English Soofi run):
- 2A.1 Structural: min_prompt_chars=50, min_messages=2
- 2A.2 Think: require_think_tags=True, min_think_chars=50
- 2A.3 Question: enable_question_filter=True, question_min_length=50
- 2A.4 Language: enable_chinese_ratio_filter=True (no FastText LID; upstream field guarantees English)
- 2A.5 Identity: enable_identity_filter=True, enable_cutoff_filter=True
- 2A.6 Repetition: detect_mass_repetition (6 sentence repeats, 30 phrase repeats)

**Usage**:
```bash
python stage4_step2a_soofi_english.py \
    --input-path /path/to/english/soofi/dataset \
    --output-dir /path/to/stage4_step2a_output

# Smoke test (first 200 rows):
python stage4_step2a_soofi_english.py \
    --input-path /path/to/english/soofi/dataset \
    --output-dir /tmp/stage4_test \
    --max-examples 200
```

**Optional flags**:
- `--max-examples N` — limit to N rows (smoke test)
- `--num-proc N` — parallel workers (default: 1)

## Stage 4 — Step 2B: Soofi English Quality Filter

Reproduces `Master_Thesis/Data_pipline/sft/quality_filtering/quality_filter_soofi_english.py`
as a thin wrapper around `quality_filtering/english_fasttext.py`.

**Input**: `kept.parquet` from Step 2A

**Output**: Quality-filtered `kept.parquet` + `dropped.parquet` + `stats.json`

**Classifier**: `mlfoundations/fasttext-oh-eli5` — binary FastText classifier
trained on OpenHermes + Reddit ELI5 (`__label__hq`) vs. Common Crawl (`__label__cc`).
Scores each row's `context_messages + response_text` text (≤2000 chars, single-line).

**Threshold policy**:
- Default (`--base-threshold 0.0`): all rows kept, scores written for analysis
- **`--score-threshold 0.72`**: thesis run (stricter than main-corpus τ = 0.20) — bypasses calibration
- `--base-threshold 0.20`: use 0.20, but auto-lower if it would drop >20 % (cap)
- `--score-threshold F`: fixed threshold, bypass calibration entirely

**Setup**: Download model once:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="mlfoundations/fasttext-oh-eli5",
    filename="openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
    local_dir="models/fasttext/quality_filter_oh",
)
```

**Usage**:
```bash
# No filtering — score all rows (default):
python stage4_step2b_soofi_english_quality_filter.py \
    --input-dir  /path/to/stage4_step2a_output \
    --model-path /path/to/models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \
    --output-dir /path/to/stage4_step2b_output

# With base threshold (auto-calibrated, ≤20% drop cap):
python stage4_step2b_soofi_english_quality_filter.py \
    --input-dir       /path/to/stage4_step2a_output \
    --model-path      /path/to/models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \
    --output-dir      /path/to/stage4_step2b_output \
    --base-threshold  0.20

# Smoke test (first 500 rows):
python stage4_step2b_soofi_english_quality_filter.py \
    --input-dir  /path/to/stage4_step2a_output \
    --model-path /path/to/models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \
    --output-dir /tmp/stage4_step2b_test \
    --max-rows   500
```

**Optional flags**:
- `--base-threshold F` — preferred minimum HQ score (default: 0.0 = no filtering)
- `--max-drop-rate F` — hard drop-rate cap for calibration (default: 0.20)
- `--score-threshold F` — fixed threshold, bypasses calibration
- `--batch-size N` — FastText scoring batch size (default: 512)
- `--max-rows N` — limit to N rows (smoke test)

## Stage 4 — Step 3: Soofi English Cross-Collection Fuzzy Dedup

Reproduces `Master_Thesis/Data_pipline/sft/dedup/dedup_fuzzy_cross_collection.py`
as a thin CLI wrapper. **No intra-Soofi exact/fuzzy pass** — the Soofi English split
is deduplicated *against the merged main English SFT corpus* using RapidFuzz
`fuzz.ratio` (Levenshtein) at τ_fuzzy = 90 %.

**Input**:
- `kept.parquet` from Step 2B (new data)
- merged main English SFT corpus parquet (reference)

**Output**:
- `soofi_english.cross_dedup.kept.parquet` — survivors
- `soofi_english.cross_dedup.dropped.parquet` — dropped rows with audit columns
  (`_fuzzy_drop_reason`, `_fuzzy_match_score`, `_fuzzy_matched_example_id`,
  `_fuzzy_matched_dataset_id`, `_fuzzy_new_context_text`, `_fuzzy_matched_context_text`)
- `cross_dedup_report.json` + `cross_dedup_report.csv`

**Policy**:

| Match against … | Action |
|---|---|
| Reference (main corpus) | always **DROP** (`cross_collection_duplicate`) |
| Already-kept new row, *different* `source_dataset_id` | **DROP** (`intra_new_cross_source_duplicate`) |
| Already-kept new row, *same* `source_dataset_id` | **KEEP** |
| No match | **KEEP** |

**Algorithm**:
1. **Phase 1**: Stream reference parquet, normalize prompts, build a bucket index
   keyed by head-token + length-bin (deduplicated by normalized prompt).
2. **Phase 2**: Stream new parquet, fetch candidates from neighbour buckets,
   score with `fuzz.ratio` in a `ProcessPoolExecutor`, apply policy above.

**Usage**:
```bash
python stage4_step3_soofi_english_dedup.py \
    --input-dir         /path/to/stage4_step2b_output \
    --reference-parquet /path/to/merged_english_sft.kept.parquet \
    --output-dir        /path/to/stage4_step3_output

# Smoke test (first 5,000 new rows):
python stage4_step3_soofi_english_dedup.py \
    --input-dir         /path/to/stage4_step2b_output \
    --reference-parquet /path/to/merged_english_sft.kept.parquet \
    --output-dir        /tmp/stage4_step3_test \
    --max-rows          5000
```

**Optional flags** (forwarded to `dedup_fuzzy_cross_collection.run_cross_fuzzy_dedup`):
- `--similarity-threshold F` — `fuzz.ratio` threshold in [0,100] (default 90.0)
- `--batch-size N` — parquet batch size (default 20,000)
- `--bucket-tokens N` — head-token count for bucket key (default 1)
- `--len-bin-size N` — length-bin granularity (default 100)
- `--max-candidates-per-bucket N` — cap per bucket (default 5,000)
- `--length-diff-threshold N` — pre-filter on `|len(a)−len(b)|`, <0 disables (default 200)
- `--neighbor-bin-radius N` — adjacent length-bins to check (default 1)
- `--num-workers N` — `ProcessPoolExecutor` workers (default 52)
- `--max-rows N` — limit new rows (smoke test)
- `--log-every-batches N` — logging cadence (default 5)

## Full Pipeline Example

```bash
# Step 2A: Heuristic filter
python stage_4b/soofi_english/stage4_step2a_soofi_english.py \
    --input-path /data/english_soofi \
    --output-dir /out/stage4_step2a \
    --num-proc 4

# Step 2B: Quality filter (thesis run uses τ = 0.72)
python stage_4b/soofi_english/stage4_step2b_soofi_english_quality_filter.py \
    --input-dir       /out/stage4_step2a \
    --model-path      /models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \
    --output-dir      /out/stage4_step2b \
    --score-threshold 0.72

# Step 3: Cross-collection fuzzy dedup vs. main English SFT corpus
python stage_4b/soofi_english/stage4_step3_soofi_english_dedup.py \
    --input-dir            /out/stage4_step2b \
    --reference-parquet    /path/to/merged_english_sft.kept.parquet \
    --output-dir           /out/stage4_step3 \
    --similarity-threshold 90.0 \
    --num-workers          52
```

## Comparison to Multilingual (Stage 4B) Pipeline

The English pipeline (`stage4_stepXX_soofi_english.py`) supplements the main English SFT
corpus, so its Step 3 is a **cross-collection fuzzy dedup against that main corpus**.

The multilingual pipeline (`stage_4b/soofi_multilingual/stage4b_stepXX_soofi_non_english.py`)
processes per-language splits (German, French, Italian, Spanish) independently, with
language-specific classifiers at Step 2B and intra-language exact + fuzzy dedup at Step 3.

| | English (Stage 4) | Multilingual (Stage 4B) |
|---|---|---|
| **2B Model** | FastText OH-ELI5 (`mlfoundations/fasttext-oh-eli5`) | XLM-R + FineWeb-HQ heads (`epfml/FineWeb-HQ-Classifiers`) |
| **2B Speed** | Very fast (CPU, batch 512) | Slower (GPU recommended, batch 8) |
| **2B Threshold (thesis)** | τ = 0.72 (stricter — supplementing high-quality EN corpus) | τ = 0.20 (per-language) |
| **Step 3 Dedup** | Cross-collection fuzzy vs. main EN corpus, τ_fuzzy = 90 % | Intra-language exact (SHA-1) + fuzzy (token_set_ratio 0.99) |
