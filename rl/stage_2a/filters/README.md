# `rl/filters/` — Heuristic Filter Modules for RL Datasets

This package contains all **Step 2a heuristic filters** for the RL data
pipeline.  Each module is self-contained, returns `Optional[str]` (a drop
reason or `None` = keep), and can be imported individually by any
dataset-specific pipeline script (e.g. `filter_and_format_dolci_think_rl_7b.py`).

---

## Overview of Filter Modules

| Module | Function | Drop Reason(s) | Trigger |
|---|---|---|---|
| `prompt_length_filter.py` | `should_drop_prompt_length` | `prompt_too_short` | `len(prompt) < min_chars` (default 30) |
| `prompt_reference_filter.py` | `should_drop_prompt_references` | `prompt_has_http`, `prompt_has_image` | URL or image keyword in prompt |
| `prompt_repetition_filter.py` | `should_drop_prompt_repetition` | `prompt_repetition:sentence×N`, `prompt_repetition:ngram×N` | Repeated sentence (≥5×) or 4-gram loop (≥15×) |
| `ground_truth_filter.py` | `should_drop_ground_truth` | `missing_ground_truth`, `placeholder_ground_truth`, `ground_truth_too_short` | Missing, placeholder string, or too short GT |
| `passrate_filter.py` | `should_drop_by_passrate` | `passrate_too_high`, `passrate_zero` | passrate > max (default 0.90) or == 0 (optional) |
| `english_filter.py` | `should_drop_non_english_prompt` | `non_english_fasttext` | FastText LID confidence < threshold (default 0.80) |

The SFT filter `content_filters.has_cutoff_mention` is **reused directly**
from `sft/filters/` (domain-agnostic, no RL-specific wrapping needed).

---

## Design Rationale

### Why a separate `rl/filters/` instead of reusing `sft/filters/`?

The SFT pipeline applies filters to **conversation turns with reasoning traces**.
The RL pipeline applies filters to **bare prompts + verifiable ground truths**.
This leads to a different set of concerns:

- **No `<think>` trace** → trace-format filters (`is_truncated_think`,
  `require_think_tags`) are not applicable.
- **Ground truth is a hard verifier gate** → GT quality matters more than in
  SFT (a wrong GT poisons the reward signal for all rollouts).
- **Passrate is an RL-specific signal** → too-easy examples (passrate ≈ 1.0)
  provide no training gradient; too-hard ones (passrate ≈ 0.0) provide no
  positive signal either.
- **Prompt repetition** → RL prompt loops confuse policy models differently
  than SFT models (reward hacking on repeated tokens is more likely).

The SFT filters for identity text, HTTP links, and images were merged into
cleaner RL-specific functions (`prompt_reference_filter`, `ground_truth_filter`).

---

## Filter Logic Details

### `prompt_length_filter.py`
Drops prompts that are too short to carry meaningful content after whitespace
normalisation.  `min_chars=30` is conservative — nearly every real task
description is longer.

### `prompt_reference_filter.py`
Drops prompts that:
- contain HTTP/HTTPS links (the model cannot fetch them at inference time)
- reference images or figures by keyword (`[image]`, `diagram`, `.png`, etc.)

Both checks can be toggled independently (`drop_if_has_http`,
`drop_if_has_images`).  The image-keyword list covers common Markdown, LaTeX,
and plain-text image patterns.

### `prompt_repetition_filter.py`
Two independent checks:
1. **Sentence-level**: split on `[.!?]+`, count exact-string repeats. Fires
   at ≥ 5 repetitions of any sentence (≥ 10 chars).
2. **N-gram level**: 4-grams over whitespace tokens.  Fires at ≥ 15 repeats.

`min_chars=80` guard prevents false positives on ultra-short prompts.

### `ground_truth_filter.py`
Three sequential checks:
1. **Missing** (`require_present=True`): `None` or empty string → drop.
2. **Placeholder** (`check_placeholder=True`): exact match against a frozenset
   of ~20 sentinel strings (`"n/a"`, `"none"`, `"?"`, `"<answer>"`, etc.).
3. **Too short** (`min_gt_chars=1`): character count gate (default 1 = just
   requires non-empty; raise to e.g. 3 for stricter quality).

### `passrate_filter.py`
Based on the **OLMo-3 offline difficulty filtering** (§4.4.2 of the paper):
> "We apply offline difficulty filtering… removing problems where all or
>  nearly all rollouts were correct."

- `max_passrate=0.90` (conservative default; OLMo-3 uses 0.625 against their
  base model — we keep 0.90 because we are not filtering against a fixed base).
- `drop_if_zero=False` (optional; zero-passrate prompts are unsolvable for the
  current model but may be reachable after SFT warm-up — keep them by default).
- If `passrate` field is absent the filter is silently skipped (`None` → keep).

### `english_filter.py`
Thin wrapper around the SFT `english_fasttext_filter` that accepts the RL
`RLEnglishLidConfig` dataclass.  Gracefully returns `None` (= keep) if the
FastText model is not available.  `min_chars=80` avoids dropping very short
code-only prompts that are language-neutral.

---

## Applying Filters to Other RL Datasets

The modules are dataset-agnostic.  To add a new RL source, create a new
pipeline script `filter_and_format_<dataset_name>.py` and call the same
functions.  Below are dataset-specific considerations:

### `NovaSky-Berkeley/Sky-T1_data_17k`
- **Ground truth**: answers are plain strings → `ground_truth_filter` applies
  directly.
- **Passrate**: not available → set `enable_passrate_filter=False`.
- **Language**: mostly English, FastText filter useful as a soft guard.
- **Extra consideration**: contains some prompts with LaTeX `\boxed{}` answers
  embedded in the prompt text — consider a custom check for answer leakage.

### `open-r1/MATH-lighteval`
- **Ground truth**: LaTeX math expressions.  `min_gt_chars` should be ≥ 2
  (single digits are valid answers).  Placeholder check still useful (`"?"`,
  `"none"` can appear in scraped data).
- **Passrate**: not available → `enable_passrate_filter=False`.
- **Reference filter**: `drop_prompt_http=True` is fine; `drop_prompt_images`
  may need loosening if geometry problems reference figures.

### `AI-MO/NuminaMath-CoT`
- **Ground truth**: multi-step solution strings.  `min_gt_chars` should be
  higher (e.g. 10) because single-character solutions are likely truncation
  artefacts.
- **Passrate**: not available → `enable_passrate_filter=False`.
- **Repetition**: solution strings can contain repeated proof steps —
  consider running `prompt_repetition_filter` on the GT as well.

### `codeparrot/apps` / `deepmind/code_contests`
- **Ground truth**: test cases (list of input/output pairs).  The
  `ground_truth_filter` placeholder check is less relevant; missing-GT check
  is critical (some APPS problems lack test cases entirely).
- **Passrate**: if a solve-rate column is available, use
  `max_passrate=0.95` (code problems have sparser solution distributions).
- **Reference filter**: `drop_prompt_http` should stay `True`.
  `drop_prompt_images` is usually fine (code problems rarely have figures).

### Datasets with non-English prompts (e.g. future multilingual RL)
- Set `enable_fasttext_english_filter=False`.
- Add a language-tag column to the schema and filter by allowed language codes
  instead of relying on the FastText gate.

---

## Adding a New Filter

1. Create `<name>_filter.py` in this folder.
2. Export one primary function with signature:
   ```python
   def should_drop_<name>(value, *, ...) -> Optional[str]:
       ...
   ```
   Return a string drop reason or `None` to keep.
3. Export it from `__init__.py` (optional but recommended for discoverability).
4. Add an entry to the table at the top of this README.
5. Add the corresponding `FilterConfig` field + CLI argument in the pipeline
   script, following the `[A]`–`[G]` pattern.
