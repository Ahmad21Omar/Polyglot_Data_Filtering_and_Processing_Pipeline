# SFT-Collection-v2 — Filtering Pipeline

This repository reproduces the full data-filtering pipeline behind
**[SFT-Collection-v2](https://huggingface.co/datasets/ahmad21omar/SFT-Collection-v2)**,
a multilingual supervised fine-tuning corpus for reasoning-capable language models.

## Dataset at a glance

| Property | Value |
|---|---|
| HuggingFace ID | `ahmad21omar/SFT-Collection-v2` |
| Total rows | 23,896,757 |
| Tokens | ~89.2 B |
| Languages | EN, DE, FR, IT, ES, JA |
| Configs | 11 |
| Key columns | `reasoning_text`, `response_text`, `language`, `domain`, `quality_score` |

## Pipeline overview

```
Stage 0  Download                15 public HuggingFace datasets   85.7M rows
Stage 1  Dataset elimination     dataset-level rejection          57.0M  (-33.5%)
Stage 2A Heuristic filter        per-row text quality checks      19.9M  (-65.1%)
Stage 2B Quality score           FineWeb-HQ classifier >= 0.20    19.3M  ( -3.1%)
Stage 3A Same-ID dedup           intra-dataset exact-ID dedup     19.3M  ( -0.0%)
Stage 3B Exact-hash dedup        SHA1(normalize(context)) cross   ~19M   ( -1-2%)
Stage 3C Fuzzy dedup             RapidFuzz Levenshtein cross      17.1M  (-11.0%)
Stage 4A Post-processing         schema clean-up + domain tags    17.1M  ( -0.2%)
Stage 4B Soofi multilingual      add translated DE/FR/IT/ES/JA    +6.8M
                                                          Final:  23.9M rows
```

## Repository layout

```
Filtering_Pipeline/
  README.md
  data_loader.py                       Stage 0 — download source datasets

  stage_2a/                            Stage 2A — heuristic filter + format
    filter_and_format/                   adapters + 6-phase pipeline + runner
      adapter.py
      config.py
      output_schema.py
      pipeline.py
      runner.py
      adapters/                          one adapter per source dataset
    filters/                             phase modules used by the pipeline
      phase_2a1_structural.py
      phase_2a2_think_reasoning.py
      phase_2a3_prompt_question.py
      phase_2a4_language.py
      phase_2a5_identity_safety.py
      phase_2a6_repetition.py
      _utils/

  stage_2b/                            Stage 2B — quality scoring
    quality_filtering/
      english_fasttext.py                EnglishFastTextQualityScorer
      multilingual_fineweb_hq.py         MultilingualFineWebHqScorer
      runner.py                          run_quality_filter(...)
      text_assembly.py                   build_scoring_text(row)
      merge_filtered.py

  stage_3/                             Stage 3 — three-pass deduplication
    deduplication/
      stage3a_same_id_dedup.py           intra-dataset same-ID dedup
      stage3b_exact_hash_dedup.py        cross-dataset SHA-1 hash dedup
      stage3c_fuzzy_dedup.py             cross-dataset RapidFuzz dedup

  stage_4a/                            Stage 4A — post-processing
    postprocess/
      stage4a_clean_dataset.py           ground-truth fix + domain backfill + lang filter
      domain_classifier/
        train.py
        predict.py

  stage_4b/                            Stage 4B — Soofi multilingual extension
    soofi_english/                       English split, supplements main corpus
      stage4_step2a_soofi_english.py
      stage4_step2b_soofi_english_quality_filter.py
      stage4_step3_soofi_english_dedup.py
    soofi_multilingual/                  DE / FR / IT / ES splits
      stage4b_step2a_soofi_non_english.py
      stage4b_step2b_soofi_quality_filter.py
      stage4b_step3_soofi_dedup.py
```

All Python imports use the package path
`Filtering_Pipeline.stage_<N>.<module>`, e.g.
`from Filtering_Pipeline.stage_2b.quality_filtering import EnglishFastTextQualityScorer`.

## Stage 0 — Download (`data_loader.py`)

### Source datasets

| # | HuggingFace ID | Lang | Domain | Raw columns |
|---|---|---|---|---|
| 1 | `allenai/Dolci-Think-SFT-7B` | EN | Mixed | `messages`, `dataset_source`, `id` |
| 2 | `open-thoughts/OpenThoughts3-1.2M` | EN | Math/Code/Science | `difficulty`, `source`, `domain`, `conversations` |
| 3 | `open-r1/Mixture-of-Thoughts` | EN | Math/Code/Science | `messages`, `num_tokens`, `source` |
| 4 | `nvidia/Nemotron-Post-Training-Dataset-v2` | EN+ML | Math/Code/STEM/Chat | `uuid`, `category`, `reasoning`, `messages`, … |
| 5 | `nvidia/Llama-Nemotron-Post-Training-Dataset` | EN | Math/Code/Science/Chat | `input`, `output`, `category`, `reasoning`, … |
| 6 | `nvidia/Nemotron-Math-v2` | EN | Math | `uuid`, `expected_answer`, `problem`, `messages`, … |
| 7 | `nvidia/Nemotron-Math-Proofs-v1` | EN | Math (Lean) | `problem`, `formal_statement`, `lean_header`, `messages` |
| 8 | `nvidia/Nemotron-Competitive-Programming-v1` | EN | Code | `uuid`, `messages`, `difficulty`, `question_id`, … |
| 9 | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | EN | Mixed | `messages` (with nested `info`) |
| 10 | `PrimeIntellect/SYNTHETIC-2-SFT-verified` | EN | Mixed | `problem_id`, `task_type`, `reward`, `messages` |
| 11 | `toroe/Soofi-Think-SFT-10B-multilingual` | DE/FR/IT/ES/EN | Multilingual | `messages`, `source`, `dataset_name`, `ds_uid`, `language` |

Datasets 12-14 are internal AIML-Lab datasets, not publicly available.

### Usage

CLI smoke test (5 rows):

```bash
python data_loader.py --dataset openthoughts3 --max-examples 5 --output-dir /tmp/sft_test
```

Available `--dataset` values:
`openthoughts3`, `dolci`, `synthetic2`, `mixture_of_thoughts`, `nemotron_math_proofs`,
`nemotron_competitive`, `nemotron_math_v2`, `am_deepseek`, `llama_nemotron`,
`nemotron_v2`, `soofi_multilingual`.

Programmatic usage:

```python
from Filtering_Pipeline.data_loader import download_nemotron_post_training_v2

paths = download_nemotron_post_training_v2(
    splits=["math", "code"],          # omit for all 9 splits
    output_dir="/mnt/data/sft",       # or set SFT_DATA_ROOT
)
```

Datasets are saved in HuggingFace Arrow format, loadable via
`datasets.load_from_disk()`.

### Storage

| Dataset | Approx. disk |
|---|---|
| `nvidia/Llama-Nemotron-Post-Training-Dataset` | ~200 GB |
| `nvidia/Nemotron-Post-Training-Dataset-v2` | ~80 GB |
| `open-thoughts/OpenThoughts3-1.2M` | ~20 GB |
| `toroe/Soofi-Think-SFT-10B-multilingual` | ~15 GB |
| All others combined | ~30 GB |

For gated datasets, run `huggingface-cli login` first.

## Stage 1 — Dataset-level elimination

Stage 1 operates **at the dataset level**: entire datasets are accepted or
rejected before any row is examined individually. There is no executable
script for this stage — the result is encoded in which datasets enter Stage 2A.

Two rejection criteria are applied:

**1A — No chain-of-thought traces.** Datasets containing only final answers or
plain instruction-following pairs (no step-by-step reasoning) are discarded.
CoT traces are a prerequisite for the SFT objective of this work.

**1B — Superseded or redundant.** Legacy versions of a dataset that are fully
subsumed by a newer release are dropped; keeping both would introduce heavily
overlapping content that downstream deduplication cannot fully resolve.

The decisions rest on per-dataset analyses of subsource composition, content
overlap with already-included sources, and CoT availability. Those analyses
live under `analytics/sft/` (outside this package) — see in particular the
sub-source intersection scripts (`find_subsource_alias_candidates.py`) used
to detect the overlaps that motivated the 1B drops.

After Stage 1, eleven public datasets enter Stage 2A (rows 1-11 in the table
above) plus internal AIML-Lab sources.

## Stage 2A — Heuristic filter & adapter framework (`stage_2a/`)

`stage_2a/filter_and_format/` defines:

- `DatasetAdapter` (abstract) — knows how to load one source dataset and map
  its raw rows into the canonical SFT-Collection-v2 schema.
- `pipeline.py` — six row-level filter phases applied in fixed Sankey order.
- `runner.py` — generic CLI: load via adapter -> apply pipeline -> write
  `kept.parquet` + `dropped.parquet` + `summary.json`.
- `adapters/` — one concrete adapter per source dataset.

The six filter phases live in `stage_2a/filters/`:

| Phase | Module | Checks |
|---|---|---|
| 2A.1 | `phase_2a1_structural.py` | min messages, assistant presence, prompt length |
| 2A.2 | `phase_2a2_think_reasoning.py` | think-tag presence, min think chars, truncation |
| 2A.3 | `phase_2a3_prompt_question.py` | URL / image / multipart / length heuristics |
| 2A.4 | `phase_2a4_language.py` | FastText English LID, 218-lang detector, Chinese ratio |
| 2A.5 | `phase_2a5_identity_safety.py` | identity self-id, training-cutoff phrasing, safety |
| 2A.6 | `phase_2a6_repetition.py` | sentence / phrase mass repetition |

See [`stage_2a/filter_and_format/README.md`](stage_2a/filter_and_format/README.md)
for adapter authoring and configuration details, and
[`stage_2a/filters/README.md`](stage_2a/filters/README.md) for per-phase
parameters.

## Stage 2B — Quality filtering (`stage_2b/`)

After Stage 2A removes structurally bad rows, Stage 2B scores each surviving
row with a learned quality classifier and drops the lowest-quality rows.

| Language | Classifier | Module |
|---|---|---|
| English | FastText OH-ELI5 (`mlfoundations/fasttext-oh-eli5`) | `english_fasttext.py` |
| Multilingual (DE, FR, IT, ES, JA, ...) | FineWeb-HQ head on XLM-RoBERTa-base | `multilingual_fineweb_hq.py` |

### Threshold policy

1. If `score_threshold` is passed explicitly, use it as-is.
2. Else if `base_threshold == 0`, keep everything (scores written through).
3. Else if filtering at `base_threshold` would drop at most `max_drop_rate`,
   use `base_threshold`.
4. Else lower the threshold to the `max_drop_rate` percentile (never raised
   above `base_threshold`).

Defaults: `base_threshold=0.20`, `max_drop_rate=0.20` for English;
`base_threshold=0.50` for multilingual (sigmoid-calibrated).

### Quickstart

```python
from Filtering_Pipeline.stage_2b.quality_filtering import (
    EnglishFastTextQualityScorer, run_quality_filter,
)

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

Full documentation incl. model download snippets and multilingual usage:
[`stage_2b/quality_filtering/README.md`](stage_2b/quality_filtering/README.md).

## Stage 3 — Deduplication (`stage_3/`)

Three sequential passes remove duplicate rows from the quality-filtered
corpus. Each pass is a standalone CLI script with no hardcoded paths.

### Stage 3A — Intra-dataset same-ID dedup

`stage_3/deduplication/stage3a_same_id_dedup.py` removes rows whose
`(dataset_id, example_id)` pair appears more than once **within the same
dataset**. Cross-dataset collisions on `example_id` are left to Stage 3B.

Two-pass streaming: pass 1 reads only the two key columns (~200 MB RAM for
20 M rows), pass 2 streams the full schema and writes rows based on the
keep-set from pass 1.

```bash
python stage_3/deduplication/stage3a_same_id_dedup.py \
    --input  /path/to/quality_filtered/merged_sft.kept.parquet \
    --output-dir /path/to/stage3a_same_id_dedup
```

Outputs: `merged_sft.same_id_dedup.kept.parquet`,
`merged_sft.same_id_dedup.dropped.parquet`, `summary.json`.

### Stage 3B — Cross-dataset exact-hash dedup

`stage3b_exact_hash_dedup.py` removes rows whose context prompt is an exact
duplicate of a prompt already seen in a **different** dataset. Same-dataset
duplicates are kept (handled by Stage 3A).

Hash function:
`SHA1(lowercase(strip_punctuation(collapse_whitespace(context_messages))))`.
Policy: first-seen-wins across datasets. A SQLite file persists
`hash -> first_dataset` so `--resume` can continue interrupted runs.

```bash
python stage_3/deduplication/stage3b_exact_hash_dedup.py \
    --input  /path/to/stage3a/merged_sft.same_id_dedup.kept.parquet \
    --output-dir /path/to/stage3b_exact_dedup
```

| Flag | Default | Description |
|---|---|---|
| `--hash-algo` | `sha1` | `sha1` / `md5` / `sha256` |
| `--batch-size` | `50000` | Rows per batch |
| `--resume` | off | Resume from checkpoint in `--output-dir` |
| `--checkpoint-every` | `20` | Flush checkpoint every N batches |

Outputs: `merged_sft.exact_dedup.kept.parquet`,
`merged_sft.exact_dedup.dropped.parquet`, `report.json`. Audit columns
(`_dedup_hash`, `_dedup_norm`, `_dedup_seen_in_ds` / `_dedup_matched_ds`) can
be dropped post-hoc.

### Stage 3C — Cross-dataset fuzzy dedup

`stage3c_fuzzy_dedup.py` removes prompts that are *near-duplicates* of a
prompt already seen in a different dataset, using **RapidFuzz `fuzz.ratio`**
(Levenshtein similarity, 0-100).

| Setting | Threshold | Behaviour |
|---|---|---|
| SFT-Collection-v2 (default) | **78** | Catches paraphrases and lightly reworded prompts |
| OpenThoughts3-style | 95 | Conservative, only near-identical prompts dropped |

Bucketing: bucket key = `head_tokens(text, 2) + "_" + len(text)//100`,
neighbour bins are checked within `±1` length bin, candidates with
`|len(query) - len(cand)| > 40` are skipped, comparisons inside each batch
run in a `ProcessPoolExecutor`.

```bash
python stage_3/deduplication/stage3c_fuzzy_dedup.py \
    --input  /path/to/stage3b/merged_sft.exact_dedup.kept.parquet \
    --output-dir /path/to/stage3c_fuzzy_dedup
```

| Flag | Default | Description |
|---|---|---|
| `--similarity-threshold` | `78` | `fuzz.ratio` cutoff (0-100) |
| `--batch-size` | `20000` | Rows per batch |
| `--num-workers` | `32` | ProcessPoolExecutor workers |
| `--bucket-tokens` | `2` | Leading tokens in bucket key |
| `--len-bin-size` | `100` | Length-bin granularity |
| `--max-candidates-per-bucket` | `5000` | Cap per bucket (oldest evicted first) |
| `--max-len-diff` | `40` | Skip candidates with greater length difference (`-1` disables) |
| `--neighbor-bin-radius` | `1` | Search +/- N adjacent length bins |
| `--resume` | off | Continue from checkpoint |

Outputs: `merged_sft.fuzzy_dedup.kept.parquet`,
`merged_sft.fuzzy_dedup.dropped.parquet`, `report.json`. Audit columns in
dropped: `_drop_reason`, `_fuzzy_score`, `_fuzzy_matched_dataset`,
`_fuzzy_matched_example`.

Requires `rapidfuzz` and `pyarrow`.

## Stage 4A — Post-processing (`stage_4a/`)

`stage_4a/postprocess/stage4a_clean_dataset.py` applies four sequential
clean-up steps after Stage 3C, producing the final English SFT corpus before
the multilingual extension.

| # | Step | Type | Effect |
|---|---|---|---|
| 1 | `fix_ground_truth_present` | Transform | If `ground_truth_present=True` but `final_answer_text` is null/empty, set to `False` |
| 2 | `fix_null_domains` | Transform | Rule-based domain backfill for known `saumyamalik` subsources |
| 3 | `apply_ml_predictions` | Transform | FastText backfill for remaining null-domain rows |
| 4 | `filter_language` | Filter | Drop rows whose `language` does not match the row category |

### Rule-based domain backfill (Step 2)

| `subsource_raw` | Domain assigned |
|---|---|
| `saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2` | `code` |
| `saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2` | `code` |
| `saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2` | `science` |
| `saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2` | `math` |
| `saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2` | `reasoning` |

### ML domain backfill (Step 3)

Remaining null-domain rows are classified by a FastText model trained on the
labeled majority of the corpus.

```
1. Train:   stage_4a/postprocess/domain_classifier/train.py   -> domain_classifier.bin
2. Predict: stage_4a/postprocess/domain_classifier/predict.py -> null_domain_predictions.csv
3. Apply:   stage_4a/postprocess/stage4a_clean_dataset.py --predictions-csv ...
```

FastText hyperparameters follow OpenThoughts3 Appendix R.2.1
(`dim=256, epoch=3, lr=0.1, wordNgrams=2, minCount=3`) with class balancing
(max 500 k per domain).

### Language filter (Step 4)

| Row category | Allowed `language` |
|---|---|
| `null` or anything not `multilingual_*` | `en` |
| `multilingual_de/fr/it/es/ja` | `de`, `fr`, `it`, `es`, `ja` |

Failing rows go to the dropped parquet for inspection.

### Usage

```bash
python stage_4a/postprocess/domain_classifier/train.py \
    --input  /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \
    --output /path/to/domain_classifier/domain_classifier.bin

python stage_4a/postprocess/domain_classifier/predict.py \
    --input  /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \
    --model  /path/to/domain_classifier/domain_classifier.bin \
    --output /path/to/domain_classifier/null_domain_predictions.csv

python stage_4a/postprocess/stage4a_clean_dataset.py \
    --input           /path/to/stage3c/merged_sft.fuzzy_dedup.kept.parquet \
    --output-dir      /path/to/stage4a_out/ \
    --predictions-csv /path/to/domain_classifier/null_domain_predictions.csv
```

| Flag | Default | Description |
|---|---|---|
| `--predictions-csv` | none | If omitted, ML step is skipped |
| `--batch-size` | `200000` | Rows per streaming batch |
| `--max-rows` | none | Stop after N rows (dry-run) |

Outputs: `merged_sft.v2.kept.parquet`,
`merged_sft.v2.dropped.language_filter.parquet`, `summary.json`.

`fasttext` requires a C++ compiler (e.g. `apt-get install -y build-essential`).
Both `train.py` and `predict.py` apply a NumPy>=2.0 compatibility patch
automatically.

## Stage 4B — Soofi multilingual extension (`stage_4b/`)

Two parallel pipelines extend the corpus with the Soofi dataset:

- **`stage_4b/soofi_english/`** — the English split, supplementing an
  already high-quality English corpus. Uses a stricter quality threshold
  (τ = 0.72) and a single dedup pass: cross-collection fuzzy match against
  the merged main English SFT corpus at τ_fuzzy = 90 %.
- **`stage_4b/soofi_multilingual/`** — the DE / FR / IT / ES splits
  processed independently, each with its own FineWeb-HQ classifier head and
  intra-language exact + fuzzy dedup.

| | English (`soofi_english/`) | Non-English (`soofi_multilingual/`) |
|---|---|---|
| 2B classifier | FastText OH-ELI5 | XLM-R + FineWeb-HQ heads |
| 2B speed | CPU, batch 512 | GPU recommended, batch 8 |
| 2B threshold (thesis) | τ = 0.72 | τ = 0.45 (per language) |
| Step 3 dedup | Cross-collection fuzzy vs. main EN corpus, τ = 90 % | Intra-language exact (SHA-1) + fuzzy (98 %) |

### Soofi English

Documentation: [`stage_4b/soofi_english/README.md`](stage_4b/soofi_english/README.md).

```bash
python stage_4b/soofi_english/stage4_step2a_soofi_english.py \
    --input-path /data/english_soofi --output-dir /out/stage4_step2a --num-proc 4

python stage_4b/soofi_english/stage4_step2b_soofi_english_quality_filter.py \
    --input-dir       /out/stage4_step2a \
    --model-path      /models/fasttext/quality_filter_oh/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin \
    --output-dir      /out/stage4_step2b \
    --score-threshold 0.72

python stage_4b/soofi_english/stage4_step3_soofi_english_dedup.py \
    --input-dir            /out/stage4_step2b \
    --reference-parquet    /path/to/merged_english_sft.kept.parquet \
    --output-dir           /out/stage4_step3 \
    --similarity-threshold 90.0
```

### Soofi non-English (Step 2A)

Wraps the same `filter_and_format/` framework as Stage 2A, with two
configuration overrides for non-English content:

| Phase | Setting | Value used | Pipeline default |
|---|---|---|---|
| 2A.2 Think | `min_think_chars` | 50 | 80 |
| 2A.3 Question | `question_min_length` | 50 | 200 |
| 2A.4 Language | `enable_fasttext_english_filter` | False | False |
| 2A.4 Language | `enable_mixed_language_filter` | False | False |

Input layout (one HuggingFace Arrow dataset per language, produced by
`data_loader.py --dataset soofi_multilingual`):

```
<input-root>/
    german/
    french/
    italian/
    spanish/
```

```bash
python stage_4b/soofi_multilingual/stage4b_step2a_soofi_non_english.py \
    --input-root /path/to/toroe__Soofi-Think-SFT-10B-multilingual \
    --output-dir /path/to/stage4b_step2a

# Subset of languages:
python stage_4b/soofi_multilingual/stage4b_step2a_soofi_non_english.py \
    --input-root /path/to/... --output-dir /path/to/... \
    --languages german,french
```

| Flag | Default | Description |
|---|---|---|
| `--languages` | `german,french,italian,spanish` | Subfolder names to process |
| `--max-examples` | none | Stop after N rows per language (smoke test) |
| `--num-proc` | `1` | `ds.map` parallel workers |

Output (per language): `kept.parquet`, `dropped.parquet`, `summary.json`,
plus a top-level `summary.json` with combined stats.

### Soofi non-English (Step 2B)

Loads the per-language FineWeb-HQ classifier head
(`deu_Latn.pt` / `fra_Latn.pt` / `ita_Latn.pt` / `spa_Latn.pt`) on top of
XLM-RoBERTa-base and scores `[user-turns ... [SEP] response]`.

One-off head download:

```python
from huggingface_hub import hf_hub_download
for fname in ("deu_Latn.pt", "fra_Latn.pt", "ita_Latn.pt", "spa_Latn.pt"):
    hf_hub_download(
        repo_id="epfml/FineWeb-HQ-Classifiers",
        filename=fname,
        local_dir="models/FineWeb-HQ-Classifiers",
    )
```

```bash
python stage_4b/soofi_multilingual/stage4b_step2b_soofi_quality_filter.py \
    --input-root      /path/to/stage4b_step2a \
    --classifiers-dir /path/to/models/FineWeb-HQ-Classifiers \
    --output-dir      /path/to/stage4b_step2b \
    --threshold       0.45
```

| Flag | Default | Description |
|---|---|---|
| `--threshold` | required | Quality cutoff in [0, 1] |
| `--languages` | `german,french,italian,spanish` | Subfolders to process |
| `--batch-size` | `8` | XLM-R scoring batch size |
| `--device` | auto | `cuda` / `cpu` |
| `--max-rows` | none | Stop after N rows per language |

Per-language output: `kept.parquet`, `dropped.parquet`, `stats.json`.

### Soofi non-English (Step 3) — intra-language dedup

`stage4b_step3_soofi_dedup.py` runs a two-pass intra-language dedup:

1. Exact SHA-1 hash on normalized prompt — cross-source duplicates dropped,
   within-source kept.
2. Fuzzy `fuzz.ratio` (default 98 %) on prompts surviving pass 1, with the
   same cross-source policy.

The sub-source name is `source_dataset_id` up to the last `/` — the same
policy used by Stage 3B/3C for the main English corpus.

```bash
python stage_4b/soofi_multilingual/stage4b_step3_soofi_dedup.py \
    --input-root /path/to/stage4b_step2b \
    --output-dir /path/to/stage4b_step3
```

| Flag | Default | Description |
|---|---|---|
| `--similarity-threshold` | `98.0` | Fuzzy threshold (`fuzz.ratio`, 0-100) |
| `--hash-algo` | `sha1` | `sha1` / `md5` / `sha256` |
| `--batch-size-exact` | `50000` | |
| `--batch-size-fuzzy` | `20000` | |
| `--num-workers` | `8` | RapidFuzz workers |
| `--max-rows` | none | Per-language smoke-test cap |
| `--resume` | off | Resume both passes from checkpoints |

Per-language output: `exact/{kept,dropped}.parquet` + `report.json`,
`fuzzy/{kept,dropped}.parquet` + `report.json`. The downstream input is
`fuzzy/kept.parquet`. A combined `summary.json` is written at the top level.
