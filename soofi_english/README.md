# Stage 4 — Soofi English Filtering Pipeline

Three-step processing pipeline for the English split of `toroe/Soofi-Think-SFT-10B-multilingual`, producing deduplicated, quality-filtered data in the canonical SFT-Collection-v2 schema.

```
soofi_english/
  ├─ stage4_step2a_soofi_english.py
  │    Heuristic filter + format (structural, prompt, think, etc.)
  │
  ├─ stage4_step2b_soofi_english_quality_filter.py
  │    Quality filtering using FineWeb-HQ + XLM-R
  │
  ├─ stage4_step3_soofi_english_dedup.py
  │    Two-pass deduplication (exact SHA-1 hash + fuzzy RapidFuzz)
  │
  └─ README.md
       This file
```

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
- `--base-threshold 0.20`: use 0.20, but auto-lower if it would drop >20% (cap)
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

## Stage 4 — Step 3: Soofi English Deduplication

**Input**: `kept.parquet` from Step 2B

**Output**: 
- `deduped.kept.parquet` — final deduplicated rows
- `deduped.dropped.parquet` — dropped rows (union of exact + fuzzy)
- `exact/` and `fuzzy/` subdirectories with intermediate results

**Two-pass approach**:

1. **Pass 1 (Exact)**: SHA-1 hash deduplication on whitespace-normalized prompt
   - Drops rows with identical normalized prompt to earlier rows
   
2. **Pass 2 (Fuzzy)**: RapidFuzz token_set_ratio similarity (default threshold 0.99)
   - Only on rows that passed exact dedup
   - Drops rows with >99% prompt similarity to earlier row from same source_dataset_id

**Usage**:
```bash
python stage4_step3_soofi_english_dedup.py \
    --input-dir /path/to/stage4_step2b_output \
    --output-dir /path/to/stage4_step3_output

# Smoke test:
python stage4_step3_soofi_english_dedup.py \
    --input-dir /path/to/stage4_step2b_output \
    --output-dir /tmp/stage4_step3_test \
    --max-rows 500
```

**Optional flags**:
- `--fuzzy-threshold F` — RapidFuzz token_set_ratio threshold (default: 0.99)
- `--max-rows N` — limit to N rows (smoke test)
- `--num-proc N` — parallel workers for ds.map (default: 1)

## Full Pipeline Example

```bash
# Step 2A: Heuristic filter
python soofi_english/stage4_step2a_soofi_english.py \
    --input-path /data/english_soofi \
    --output-dir /out/stage4_step2a \
    --num-proc 4

# Step 2B: Quality filter
python soofi_english/stage4_step2b_soofi_english_quality_filter.py \
    --input-dir      /out/stage4_step2a \
    --model-path     /models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \
    --output-dir     /out/stage4_step2b \
    --base-threshold 0.20

# Step 3: Deduplication
python soofi_english/stage4_step3_soofi_english_dedup.py \
    --input-dir /out/stage4_step2b \
    --output-dir /out/stage4_step3 \
    --fuzzy-threshold 0.99 \
    --num-proc 4
```

## Comparison to Multilingual (Stage 4B) Pipeline

The English pipeline (`stage4_stepXX_soofi_english.py`) processes the English split of Soofi in isolation.

The multilingual pipeline (`soofi_multilingual/stage4b_stepXX_soofi_non_english.py`) processes per-language splits (German, French, Italian, Spanish) independently, with language-specific classifiers at Step 2B.

Both pipelines use the same adapter framework (`Filtering_Pipeline/filter_and_format/adapters/`) and deduplication machinery, but differ in the **Step 2B quality classifier**:

| | English (Stage 4) | Multilingual (Stage 4B) |
|---|---|---|
| **Model** | FastText OH-ELI5 (`mlfoundations/fasttext-oh-eli5`) | XLM-R + FineWeb-HQ heads (`epfml/FineWeb-HQ-Classifiers`) |
| **Speed** | Very fast (CPU, batch 512) | Slower (GPU recommended, batch 8) |
| **Score type** | `__label__hq` probability [0,1] | MLP sigmoid output [0,1] |
| **Default threshold** | 0.0 (no filtering, score only) | Fixed (e.g. 0.45) |
| **Threshold policy** | Auto-calibrated with 20% drop cap | Fixed threshold |
