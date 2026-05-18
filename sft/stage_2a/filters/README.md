# Stage 2A — Heuristic Filter Modules

Reusable filter utilities used in Stage 2A. The public API is grouped by the
six sub-stages of the Stage 2A Sankey diagram:

| Sub-stage | Module | What it catches |
|---|---|---|
| **2A.1 Structural**         | [`phase_2a1_structural.py`](phase_2a1_structural.py)         | messages / assistant presence, prompt length |
| **2A.2 Think / Reasoning**  | [`phase_2a2_think_reasoning.py`](phase_2a2_think_reasoning.py) | think-tag presence, min chars, truncation |
| **2A.3 Prompt / Question**  | [`phase_2a3_prompt_question.py`](phase_2a3_prompt_question.py) | URL / image / multipart / length heuristics |
| **2A.4 Language**           | [`phase_2a4_language.py`](phase_2a4_language.py)             | FastText English LID, 218-language detector, Chinese ratio |
| **2A.5 Identity / Safety**  | [`phase_2a5_identity_safety.py`](phase_2a5_identity_safety.py) | model self-id, knowledge-cutoff phrasing, optional toxicity flags |
| **2A.6 Repetition**         | [`phase_2a6_repetition.py`](phase_2a6_repetition.py)         | sentence / phrase mass repetition |

Cross-cutting helpers live in [`_utils/`](_utils/):

| Helper | Purpose |
|---|---|
| [`_utils/code_task.py`](_utils/code_task.py)               | structural validation for code-task datasets (`problem` / `tests` / `solutions`) |
| [`_utils/fasttext_classifier.py`](_utils/fasttext_classifier.py) | generic FastText binary classifier wrapper (training + scoring), used at Stage 2B |

## Properties

- All filters are local, deterministic, CPU-only.
- Each filter returns a `bool` or a string drop-reason.
- Every dropped row gets a stable `_drop_reason` label that maps 1:1 to a
  phase in the Sankey diagram.

## Module layout

The phase modules (`phase_2a*.py`) are facades that re-export the exact
function objects from the canonical implementation files in this folder
(`format_filters.py`, `question_filters.py`, `language_filters.py`,
`english_fasttext_filter.py`, `language_detector.py`, `identity_filters.py`,
`content_filters.py`, `domain_filters.py`, `repetition_filters.py`,
`code_task_filters.py`, `fasttext_filters.py`). Identity is verified at
import time (`phase_module.fn is original_module.fn`), so behaviour is
byte-for-byte identical to the original Stage 2A run.

`phase_2a1_structural.py` is the only module containing new code: it bundles
the four structural checks (`bad_messages`, `last_not_assistant`,
`prompt_too_short`, `empty_assistant`) that were previously inlined in every
per-dataset script. Check order, defaults (`min_prompt_chars=50`,
`min_messages=2`), and drop-reason labels match the legacy scripts.

## How filters are used by pipeline scripts

Pipeline scripts in `../filter_and_format/`:

1. Load a raw dataset from disk via `datasets.load_from_disk()`.
2. `map()` each row, calling these filters to decide keep / drop.
3. Set `_keep=False` and `_drop_reason="<label>"` for dropped rows.
4. Partition on `_keep` and write both kept and dropped rows to Parquet
   (the dropped split is preserved for audit / Sankey reporting).

## Public API per phase

### Phase 2A.1 — Structural ([`phase_2a1_structural.py`](phase_2a1_structural.py))

| Function | Returns | Drop reason produced |
|---|---|---|
| `evaluate_structural(messages, *, min_prompt_chars=50, min_messages=2)` | `StructuralCheckResult` | `bad_messages` / `last_not_assistant` / `prompt_too_short` / `empty_assistant` |
| `messages_to_prompt_text(messages)` | `str` (joins non-assistant turns) | — |
| `normalize_whitespace(text)` | `str` | — |
| `is_bad_messages(messages, *, min_messages=2)` | `bool` | — |
| `is_last_not_assistant(messages)` | `bool` | — |
| `is_prompt_too_short(messages, *, min_prompt_chars=50)` | `bool` | — |
| `is_empty_assistant(messages)` | `bool` | — |

### Phase 2A.2 — Think / Reasoning ([`phase_2a2_think_reasoning.py`](phase_2a2_think_reasoning.py))

| Function | Description |
|---|---|
| `is_truncated_think(text)` | `<think>` present but `</think>` missing → drop reason `truncated_think` |
| `has_balanced_think_tags(text)` | both tags present and ordered |
| `matches_olmo_think_format(text)` | strict full-match `^<think>...</think>...$` |
| `matches_olmo_think_answer_format(text)` | strict full-match `^<think>...</think><answer>...</answer>$` |
| `extract_think_and_answer(text)` | returns `(think_text, response_without_tags, answer_text)` |

Pipeline-level drop reasons emitted at this phase (by the per-dataset script):
`missing_think`, `think_too_short`, `empty_reasoning_content`,
`no_verified_proof` (math-proofs only),
`split_no_reasoning_traces` (Nemotron-v2 only).

### Phase 2A.3 — Prompt / Question ([`phase_2a3_prompt_question.py`](phase_2a3_prompt_question.py))

| Function | Drops when |
|---|---|
| `should_drop_question_text(text, *, min_length=200, ...)` | URL / image keyword / multipart / too short / empty |

Drop reasons: `question_has_http`, `question_has_image_ref`,
`question_multipart`, `question_too_short`, `empty_question`.

### Phase 2A.4 — Language ([`phase_2a4_language.py`](phase_2a4_language.py))

| Function | Description | Drop reason |
|---|---|---|
| `chinese_character_ratio(text)` | helper: CJK-char ratio | — |
| `should_filter_example_by_chinese(messages=..., threshold=0.05)` | CJK ratio in assistant turn ≥ threshold | `chinese_ratio` |
| `english_score(text, *, model_path)` | English LID confidence | — |
| `should_drop_non_english(text, *, model_path, cfg)` | English score below threshold (default 0.8) | `non_english_fasttext` |
| `detect_language(text, ...)` | top language as ISO 639-1 | — |
| `detect_language_with_confidence(text, ...)` | `(code, prob)` | — |
| `detect_language_top_k(text, ..., k=3)` | top-k `[(code, prob), ...]` | — |
| `is_mixed_language(text, ..., confidence_threshold=0.75)` | top language confidence below threshold (multilingual datasets) | `mixed_language` |

Required model file (download once):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    "facebook/fasttext-language-identification",
    "model.bin",
    local_dir="Master_Thesis/models/fasttext/language_detector",
)
hf_hub_download(
    "facebook/fasttext-language-identification",
    "model.bin",
    local_dir="Master_Thesis/models/fasttext",
)
```

### Phase 2A.5 — Identity / Safety ([`phase_2a5_identity_safety.py`](phase_2a5_identity_safety.py))

| Function | Drops when | Drop reason |
|---|---|---|
| `should_drop_identity_text(text)` | text contains "I am ChatGPT", "as an AI language model", etc. | `identity_self_id` |
| `has_cutoff_mention(messages)` | text contains "as of my last update in 2023", etc. | `cutoff_mention` |
| `find_cutoff_mentions_in_messages(messages)` | helper: returns matched substrings | — |
| `should_drop_wildchat_like_row(ex)` | row has `toxic` / `redacted` flag (optional, only used for datasets that ship those flags) | `unsafe_toxic` / `unsafe_redacted` |

### Phase 2A.6 — Repetition ([`phase_2a6_repetition.py`](phase_2a6_repetition.py))

| Function | Description |
|---|---|
| `detect_mass_repetition(text, *, min_sentence_repeats=6, phrase_n=3, min_phrase_repeats=30, min_chars=200)` | returns `RepetitionResult(should_drop, reason, details)` |

Drop reasons: `repetition_sentences` (same sentence ≥ `min_sentence_repeats`,
length ≥ 8 chars), `repetition_phrases` (same n-gram ≥ `min_phrase_repeats`,
n-gram text ≥ 12 chars).

### Cross-cutting — `_utils/code_task.py`

| Function | Drops when | Drop reason |
|---|---|---|
| `should_drop_code_task_row(ex, *, problem_field, tests_field, solutions_field, min_description_length=200)` | `problem` empty / has URL / has `[image]` / shorter than `min_description_length` | `code_problem_bad_description` |
|  | `tests` JSON has empty inputs or outputs | `code_problem_bad_tests` |
|  | `solutions` JSON empty | `code_problem_bad_solutions` |

## Drop-reason → Sankey-stage reference

This is the canonical mapping used by the Sankey reporting scripts
(`Master_Thesis/analytics/sft/Filtering_process/plot_step2a_filtering.py`):

| `_drop_reason` label | Sankey stage |
|---|---|
| `bad_messages` | `2A.1_structural` |
| `last_not_assistant` | `2A.1_structural` |
| `prompt_too_short` | `2A.1_structural` |
| `empty_assistant` | `2A.1_structural` |
| `truncated_think` | `2A.2_think_reasoning` |
| `missing_think` | `2A.2_think_reasoning` |
| `think_too_short` | `2A.2_think_reasoning` |
| `empty_reasoning_content` | `2A.2_think_reasoning` |
| `no_verified_proof` | `2A.2_think_reasoning` |
| `split_no_reasoning_traces` | `2A.2_think_reasoning` |
| `question_has_http` | `2A.3_prompt_question` |
| `question_has_image_ref` | `2A.3_prompt_question` |
| `question_multipart` | `2A.3_prompt_question` |
| `question_too_short` | `2A.3_prompt_question` |
| `empty_question` | `2A.3_prompt_question` |
| `chinese_ratio` | `2A.4_language` |
| `non_english_fasttext` | `2A.4_language` |
| `mixed_language` | `2A.4_language` |
| `identity_self_id` | `2A.5_identity_safety` |
| `cutoff_mention` | `2A.5_identity_safety` |
| `unsafe_toxic` | `2A.5_identity_safety` |
| `unsafe_redacted` | `2A.5_identity_safety` |
| `repetition_sentences` | `2A.6_repetition` |
| `repetition_phrases` | `2A.6_repetition` |
| `code_problem_bad_description` | (cross-cutting, attached during 2A.1) |
| `code_problem_bad_tests` | (cross-cutting, attached during 2A.1) |
| `code_problem_bad_solutions` | (cross-cutting, attached during 2A.1) |

## Requirements

```bash
pip install datasets fasttext-wheel huggingface_hub
```
