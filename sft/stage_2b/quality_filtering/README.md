# Stage 2B — Quality filtering

After Stage 2A removes structurally bad rows, Stage 2B scores each surviving
row with a learned **quality classifier** and drops the lowest-quality ones.
There are two classifiers — pick by language:

| Language | Classifier | Module |
|---|---|---|
| English-only | FastText OH-ELI5 (binary HQ vs CC) | [`english_fasttext.py`](english_fasttext.py) |
| Multilingual (DE, FR, IT, ES, JA, …) | FineWeb-HQ classifier head on top of XLM-RoBERTa-base embeddings | [`multilingual_fineweb_hq.py`](multilingual_fineweb_hq.py) |

Both share:

- the **scoring text** (user-prompt turns + final response, see
  [`text_assembly.py`](text_assembly.py))
- the **threshold-calibration policy** (fixed base threshold, capped at
  ``max_drop_rate``)
- the **runner** ([`runner.py`](runner.py)) — load Parquet → score → split
  kept / dropped → save + JSON sidecar

```
quality_filtering/
├── README.md                       this file
├── text_assembly.py                build_scoring_text(row) — shared
├── english_fasttext.py             English (FastText OH-ELI5)
├── multilingual_fineweb_hq.py      Multilingual (XLM-R + FineWeb-HQ)
├── runner.py                       run_quality_filter(...) — per-corpus filter
└── merge_filtered.py               merge all *.kept.parquet → merged_sft.kept.parquet
```

## Models — download once, no auto-download

This module never downloads anything. The first call to a scorer that
cannot find its model file on disk raises ``FileNotFoundError`` with the
exact ``huggingface_hub.hf_hub_download`` snippet to fetch it. Run the
snippets below once before processing data.

### English — FastText OH-ELI5  (~800 MB)

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="mlfoundations/fasttext-oh-eli5",
    filename="openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
    local_dir="models/fasttext/quality_filter_oh",
)
```

### Multilingual — FineWeb-HQ classifier heads  (~1.5 MB each)

```python
from huggingface_hub import hf_hub_download
for fname in ("deu_Latn.pt", "fra_Latn.pt", "ita_Latn.pt",
              "spa_Latn.pt", "jpn_Jpan.pt"):
    hf_hub_download(
        repo_id="epfml/FineWeb-HQ-Classifiers",
        filename=fname,
        local_dir="models/FineWeb-HQ-Classifiers",
    )
```

The XLM-RoBERTa embedding model (``FacebookAI/xlm-roberta-base``, ~1 GB) is
fetched automatically by ``transformers`` on first use.

## Quickstart — English

```python
from Filtering_Pipeline.stage_2b.quality_filtering import (
    EnglishFastTextQualityScorer, run_quality_filter,
)

scorer = EnglishFastTextQualityScorer(
    model_path="models/fasttext/quality_filter_oh/"
               "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
)

summary = run_quality_filter(
    scorer=scorer,
    input_path="/path/to/sft_dolci_v0.kept.parquet",
    output_dir="/path/to/quality_filtered/dolci",
    base_threshold=0.20,   # default — drop rows with HQ score < 0.20
    max_drop_rate=0.20,    # cap drop rate at 20%; lower threshold if needed
)
print(summary["n_kept"], summary["n_dropped"], summary["threshold"])
```

## Quickstart — Multilingual

```python
from Filtering_Pipeline.stage_2b.quality_filtering import (
    MultilingualFineWebHqScorer, run_quality_filter,
)

scorer = MultilingualFineWebHqScorer(
    language="de",
    classifiers_dir="models/FineWeb-HQ-Classifiers",
    device="cuda",          # or "cpu"
    batch_size=16,
)

for lang_input in [
    ("de", "/path/to/sft_soofi_german.kept.parquet"),
    ("fr", "/path/to/sft_soofi_french.kept.parquet"),
]:
    lang, in_path = lang_input
    scorer.language = lang   # reuse the loaded XLM-R encoder across languages
    scorer._classifier = None  # force the per-language head to reload
    run_quality_filter(
        scorer=scorer,
        input_path=in_path,
        output_dir=f"/path/to/quality_filtered/soofi_{lang}",
        base_threshold=0.50,   # FineWeb-HQ heads are sigmoid-calibrated → 0.5 is the natural midpoint
        max_drop_rate=0.20,
    )
```

## Threshold policy

For every input file the runner computes a per-row HQ score in ``[0, 1]``
and then decides the threshold:

1. If ``score_threshold`` is passed explicitly → use it as-is.
2. Else if ``base_threshold == 0`` → keep everything (just write scores
   through).
3. Else if ``< base_threshold`` would drop **at most** ``max_drop_rate`` of
   the corpus → use ``base_threshold``.
4. Else → **lower** the threshold to the ``max_drop_rate`` percentile so
   that exactly ``max_drop_rate`` is dropped. The threshold is **never
   raised** above ``base_threshold``.

This matches the policy used to produce SFT-Collection-v2.

## Output layout

For each input ``foo.kept.parquet`` the runner writes:

```
<output_dir>/
├── foo.kept.parquet      surviving rows (with two new columns: _quality_score, _quality_label="HIGH")
├── foo.dropped.parquet   dropped rows (with _quality_label="LOW")
└── foo.stats.json        threshold + score percentiles + scorer metadata
```

Both output Parquet files are written *atomically* (write to ``.tmp.parquet``
then rename) so a killed process never leaves a partial file at the final
path.

## Step 2 — Merge all per-corpus outputs

After all corpora have been filtered individually, run
[`merge_filtered.py`](merge_filtered.py) to stream-merge every
``*.kept.parquet`` file into one ``merged_sft.kept.parquet``.

```bash
# Minimal (defaults: min-free-gb=40, batch-rows=50 000)
python3 -m Filtering_Pipeline.stage_2b.quality_filtering.merge_filtered \
    --out-root /home/workdir/Master_Thesis/corpora/quality_filtered_v2

# Background run
nohup python3 -m Filtering_Pipeline.stage_2b.quality_filtering.merge_filtered \
    --out-root /home/workdir/Master_Thesis/corpora/quality_filtered_v2 \
    >> /home/workdir/Master_Thesis/corpora/merge_v2.log 2>&1 &
```

The script:

- **Discovers files automatically** — scans ``--out-root`` and its immediate
  subdirectories for ``*.kept.parquet`` (skips any ``merged_sft.*`` files from
  prior runs).
- **Streams via PyArrow** — one batch at a time; peak RAM ≈ one batch × schema
  width.  Never calls ``pd.read_parquet`` on the merged output.
- **RAM guard** — aborts cleanly before loading the next file if free system RAM
  falls below ``--min-free-gb`` (default 40 GB).
- **Checkpoint** — saves progress after each file to ``merge_checkpoint.json``
  so a killed run can be resumed without re-processing completed files.
- **Atomic rename** — writes to ``.tmp`` first; the final path only appears when
  all files are done.
- **Schema normalisation** — schema is derived from the first file; subsequent
  files are cast to match (missing columns are null-filled, ``string`` columns
  promoted to ``large_string``).  No columns are dropped — schema trimming is
  left to Stage 4.

Output:

```
<out-root>/
├── merged_sft.kept.parquet   ← all corpora combined
└── merge_checkpoint.json     ← deleted on successful completion
```

---

## Scoring text (both English and multilingual)

```text
<user_turn_1> [SEP] <user_turn_2> [SEP] ... [SEP] <response_text>
```

- ``reasoning_text`` is intentionally **not** included — it is often very
  long (>2 000 chars), would be truncated by both classifiers, and is
  internal model thinking rather than user-facing content.
- Empty rows (no user turn AND no response) get ``_quality_score = 0.0``.
- The ``[SEP]`` marker is a literal string the classifier was *not* trained
  on; both models ignore it (they predict from word/sub-word n-grams).

## Adding a new language

1. Make sure the ISO 639-1 code is in
   ``DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME`` (in
   [`multilingual_fineweb_hq.py`](multilingual_fineweb_hq.py)). Otherwise
   pass ``classifier_path=...`` directly when constructing the scorer.
2. Download the classifier head from ``epfml/FineWeb-HQ-Classifiers`` (the
   repo lists ~100 languages).

## Requirements

```bash
pip install datasets pandas pyarrow numpy fasttext-wheel
# Multilingual path also needs:
pip install transformers torch huggingface_hub
```
