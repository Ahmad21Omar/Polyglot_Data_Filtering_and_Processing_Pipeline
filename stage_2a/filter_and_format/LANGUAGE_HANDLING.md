# Language Handling in Stage 2A

This document explains exactly how language detection and language-based filtering
are applied across all source datasets. The decisions here are faithfully reproduced
from the original `Data_pipline/sft/filter_and_format_*.py` scripts.

---

## Overview: two filter modes, never both at once

`FilterConfig` exposes two mutually exclusive language-filtering modes:

| Mode | Config flag | Drop reason | When to use |
|---|---|---|---|
| **FastText English LID** | `enable_fasttext_english_filter=True` | `non_english_fasttext` | English-only sources where the prompt is free-form text |
| **Mixed-language detector** | `enable_mixed_language_filter=True` | `mixed_language` | Multilingual sources — rejects rows whose text is a garbled mix of languages (low top-confidence) |

Setting both at the same time raises a `ValueError`.  Use neither when the
dataset's language is externally guaranteed (e.g. Nemotron multilingual splits).

In addition, the **Chinese-ratio filter** (`enable_chinese_ratio_filter=True`,
drop reason `chinese_ratio`) is always active regardless of mode.  It drops
rows where the assistant turn contains ≥ 5 % CJK characters — a heuristic
that catches Chinese model contamination in English/European datasets.

---

## Per-dataset decisions

### 1. `allenai/Dolci-Think-SFT-7B`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | **ON** (threshold 0.8, min 80 chars, keep-if-too-short) |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Inferred from `dataset_source` subsource lookup table; default `"en"` |

Language for Dolci rows is inferred from the `dataset_source` field using a
lookup table (`_SUBSOURCE_TO_LANGUAGE` in the adapter). All subsources in
Dolci-7B are English, so the effective language is always `"en"`. The FastText
LID filter is the primary guard against occasional non-English rows that slipped
through Dolci's own curation.

---

### 2. `open-thoughts/OpenThoughts3-1.2M`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | ON (via `english_only_preset()`) |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

---

### 3. `open-r1/Mixture-of-Thoughts`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | ON |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

---

### 4. `nvidia/Nemotron-Post-Training-Dataset-v2`  →  **Multilingual; only 5 of 9 splits used**

This dataset ships 9 splits but **only the 5 multilingual splits are used** in
SFT-Collection-v2. The 4 English splits (math / code / stem / chat) have
`reasoning="off"` throughout — they contain zero CoT traces and are therefore
incompatible with the reasoning-trace SFT objective.

#### Reasoning language

The multilingual splits have a structural quirk: **the prompt is in the target
language** (German, French, Italian, Spanish, Japanese) but **the reasoning
trace (`<think>…</think>`) is always in English**. NVIDIA generated these rows
by running an English-reasoning model on translated prompts. The English CoT is
preserved as-is in `reasoning_text`.

#### Per-split filter configuration

Use `NemotronPostTrainingV2Adapter.recommended_config(split_name)`:

| split | FastText EN filter | Mixed-lang filter | Rationale |
|---|---|---|---|
| `math` | **OFF** | OFF | LaTeX tokens (`\int`, `\frac`, `\boxed`) cause **~63 % false-drop rate**. Language is guaranteed `en` from the split name — no detection needed. |
| `code` | ON | ON | Free-form English text; both filters are reliable. |
| `chat` | ON | ON | Same as code. |
| `stem` | ON | ON | Same as code. |
| `multilingual_de/es/fr/it/ja` | **OFF** | **OFF** | Prompts are intentionally non-English; FastText would drop the entire split. Mixed-language filter is also off — cross-language reasoning traces are expected by design. |

#### Language assignment

Language comes **directly from the split name** — no runtime FastText detection:

```
multilingual_de → "de"
multilingual_es → "es"
multilingual_fr → "fr"
multilingual_it → "it"
multilingual_ja → "ja"
math / code / stem / chat → "en"
```

#### Domain inference for multilingual splits

Because the multilingual splits all have domain `"multilingual"` in
`_CATEGORY_TO_DOMAIN`, the adapter refines the domain by keyword-matching
against `reasoning_text` (which is English and therefore reliable). Keywords
cover math / code / science; if the match is ambiguous the domain stays
`"multilingual"`.

---

### 5. `nvidia/Llama-Nemotron-Post-Training-Dataset`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | ON |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

---

### 6. `nvidia/Nemotron-Math-v2`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | **OFF** | 
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

FastText is off for the same reason as the Nemotron v2 math split: heavy LaTeX
in the prompts yields a high false-drop rate. Language is guaranteed English.

---

### 7. `nvidia/Nemotron-Math-Proofs-v1`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | OFF (LaTeX) |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

Lean 4 formal proofs contain non-English mathematical syntax. FastText is off.

---

### 8. `nvidia/Nemotron-Competitive-Programming-v1`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | OFF |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

Competitive programming prompts often contain code, pseudocode, and symbols
that confuse FastText. Off by design.

---

### 9. `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | ON |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

---

### 10. `PrimeIntellect/SYNTHETIC-2-SFT-verified`  →  English-only

| Filter | Setting |
|---|---|
| FastText English LID | ON |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

---

### 11. `toroe/Soofi-Think-SFT-10B-multilingual` — English subset

| Filter | Setting |
|---|---|
| FastText English LID | **OFF** |
| Mixed-language | OFF |
| Chinese ratio | ON |
| Language assignment | Constant `"en"` |

FastText English LID is **off** for Soofi English. The upstream `language`
field already certifies the content as English, making LID redundant. The
original pipeline also encountered `fasttext-wheel` / NumPy ≥ 2.x
incompatibilities on the training servers, which was a secondary reason to
leave it disabled.

---

### 12. `toroe/Soofi-Think-SFT-10B-multilingual` — Multilingual subsets (DE/FR/IT/ES)

| Filter | Setting |
|---|---|
| FastText English LID | OFF |
| Mixed-language | **ON** |
| Chinese ratio | ON |
| Language assignment | From `language` field (`"german"` → `"de"`, etc.) |

The `language` field carries a human-readable name which the adapter normalises
to ISO 639-1. Rows without a `language` value are dropped with pre-check reason
`missing_language` — in a multilingual dataset a row without a language label
cannot be assigned to a target language and is therefore unusable.

The mixed-language filter guards against rows where the text is a garbled mix
of multiple languages (low top-1 confidence in the 218-language FastText
detector). Threshold: 0.75.

The reasoning trace in Soofi multilingual rows is in the **target language**
(unlike Nemotron v2, where it is always English).

---

## Chinese-ratio filter — why always on

The Chinese-ratio filter (`enable_chinese_ratio_filter=True`) is applied to
**all datasets**, including purely English ones. It drops rows where the
assistant turn contains ≥ 5 % CJK characters. This catches a class of
contamination where a Chinese-language model (e.g., DeepSeek) occasionally
outputs Chinese text in what is supposed to be an English response. The filter
runs only on the assistant content, not on user turns.

Drop reason: `chinese_ratio`

---

## Models used for language detection

| Model | Path | Used for |
|---|---|---|
| FastText OH-ELI5 LID (`facebook/fasttext-language-identification`, `lid.176.bin`) | `models/fasttext/lid.176.bin` | `non_english_fasttext` filter (218 languages, binary: EN vs. not-EN at threshold 0.8) |
| FastText language-ID 218 (`facebook/fasttext-language-identification`, `model.bin`) | `models/fasttext/language_detector/model.bin` | `mixed_language` filter (top-1 confidence check, threshold 0.75) |

Both are the **same underlying HuggingFace repo** (`facebook/fasttext-language-identification`)
but different filenames:

- `lid.176.bin` — the **176-language** model used as English LID binary classifier.
- `model.bin`   — the **218-language** model used for mixed-language confidence scoring.

Neither is downloaded automatically. See `filters/README.md` for download snippets.

---

## Presets

`FilterConfig` ships two convenience presets:

```python
from Filtering_Pipeline.stage_2a.filter_and_format.config import (
    english_only_preset,
    multilingual_preset,
)

# English-only sources (Dolci, OpenThoughts3, LlamaNemotron, ...)
cfg = english_only_preset()
# → enable_fasttext_english_filter=True, enable_mixed_language_filter=False

# Multilingual sources (Soofi multilingual, Nemotron-v2 multilingual_* splits)
cfg = multilingual_preset()
# → enable_fasttext_english_filter=False, enable_mixed_language_filter=True
```

For datasets where neither filter applies (math-heavy LaTeX, formal proofs,
competitive programming, Soofi English), use a bare `FilterConfig()` and
set the flags explicitly — or call the adapter's `recommended_config()`.
