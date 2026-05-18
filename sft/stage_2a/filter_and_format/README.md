# Stage 2A — Generic filter & format pipeline

Shared Stage 2A pipeline used by every source dataset in SFT-Collection-v2.
Each source dataset has one adapter under [`adapters/`](adapters/); the
filter pipeline, output schema, and CLI runner are shared across all of them.

```
filter_and_format/
  config.py              FilterConfig — all thresholds & toggles
  output_schema.py       canonical row schema (KEPT_COLUMNS) + builders
  adapter.py             DatasetAdapter ABC — per-dataset hooks
  pipeline.py            6-phase row-level filter (canonical Sankey order)
  runner.py              shared CLI: load + adapt + filter + save
  LANGUAGE_HANDLING.md   per-dataset language-filter decisions (read first)
  adapters/              one file per source dataset (12 total)
```

## Pipeline order (canonical)

```
adapter.pre_check(ex)        → e.g. missing_dataset_source / no_verified_proof
        │
adapter.extract_messages(ex) → list of {role, content}
        │
2A.1 Structural        → bad_messages / last_not_assistant / prompt_too_short / empty_assistant
2A.2 Think / Reasoning → truncated_think / missing_think / think_too_short
2A.3 Prompt / Question → question_has_http / question_has_image_ref / question_multipart / question_too_short
2A.4 Language          → chinese_ratio / non_english_fasttext / mixed_language
2A.5 Identity / Safety → identity_self_id / cutoff_mention / unsafe_toxic / unsafe_redacted
2A.6 Repetition        → repetition_sentences / repetition_phrases
        │
adapter.infer_language / infer_domain / infer_reasoning_type / extract_extras
        │
output_schema.build_kept_row(...)
```

The order matches the Sankey labels in
`analytics/sft/Filtering_process/plot_step2a_filtering.py` 1:1. The legacy
per-dataset scripts had a slightly different ordering (question filter
before `empty_assistant`, identity before think extraction); the unified
pipeline adopts the canonical Sankey order. Output schema and the *set* of
drop reasons are identical to the legacy scripts; absolute drop counts may
differ marginally for rows that would have been caught by an earlier phase
under the canonical order.

## Output schema

All adapters produce rows with the same columns (defined once in
[`output_schema.KEPT_COLUMNS`](output_schema.py)):

| Field | Type | Description |
|---|---|---|
| `_keep` | bool | True iff the row passed all 6 phases |
| `_drop_reason` | string \| null | first drop reason; null for kept rows |
| `dataset_id` | string | HuggingFace ID, e.g. `allenai/Dolci-Think-SFT-7B` |
| `dataset_version_date` | string | snapshot date used for this run |
| `example_id` | string | `sha256(f"{dataset_id}|{split}|{row_id}")` |
| `row_id` | string | per-row source id |
| `source_dataset_id` | string \| null | upstream dataset id (set when the source aggregates other datasets) |
| `subsource_raw` | string \| null | raw subsource label (e.g. Dolci's `dataset_source`) |
| `language` | string | ISO 639-1 |
| `translation_of_example_id` | string \| null | for translated rows: the source `example_id` |
| `domain` | string \| null | `math` / `code` / `science` / `chat` / `instruction_following` / null |
| `reasoning_type` | string \| null | e.g. `deductive` / `procedural` / `explanatory` |
| `license` | string \| null | from the source dataset |
| `used_by_model` | string \| null | which model was used to generate this row |
| `context_messages` | list of `{role, content}` | chat history *without* the final assistant turn |
| `reasoning_text` | string \| null | extracted `<think>...</think>` content |
| `response_text` | string \| null | assistant content with think/answer tags removed |
| `ground_truth_present` | string | `yes` / `no` / `unknown` |
| `final_answer_text` | string \| null | dataset-provided gold answer (when applicable) |
| `prompt_text_preview` | string \| null | first 2000 chars of prompt — only filled for *dropped* rows |
| `assistant_last_content` | string \| null | first 4000 chars of final assistant turn — only filled for *dropped* rows |
| `difficulty` | string \| null | optional |
| `category` | string \| null | optional |

## Quickstart

```python
from Filtering_Pipeline.stage_2a.filter_and_format import (
    english_only_preset, run,
)
from Filtering_Pipeline.stage_2a.filter_and_format.adapters.dolci_think_sft_7b import (
    DolciThinkSft7BAdapter,
)

summary = run(
    adapter=DolciThinkSft7BAdapter(),
    cfg=english_only_preset(),
    input_dataset_dir="/.../allenai__Dolci-Think-SFT-7B/train",
    output_dir="/.../sft_dolci_v0",
    split="train",
    max_examples=100,    # smoke test; omit for full run
)
print(summary["n_kept"], summary["n_dropped"], summary["by_reason"])
```

The same call shape works for every dataset — just swap the adapter:

```python
from Filtering_Pipeline.stage_2a.filter_and_format.adapters.openthoughts3 import OpenThoughts3Adapter
from Filtering_Pipeline.stage_2a.filter_and_format.adapters.nemotron_math_proofs_v1 import NemotronMathProofsV1Adapter
```

## Config presets

| Preset | When to use |
|---|---|
| `english_only_preset()`  | English-only sources (Dolci, OpenThoughts3, Nemotron-* English splits) |
| `multilingual_preset()`  | Multilingual sources (Soofi, Nemotron-PT-v2 multilingual splits) |
| `FilterConfig()`         | bare default — both language filters off (use only when one of the language signals will be applied at a later stage) |

All thresholds (think length, question length, FastText threshold, repetition
counts, …) live in [`config.py`](config.py) and are documented inline.

## Writing a new adapter

A typical adapter is 50–150 lines:

```python
from typing import Any, Dict, Optional
from ..adapter import DatasetAdapter

class MyDatasetAdapter(DatasetAdapter):
    dataset_id = "myorg/my-dataset"
    license = "Apache-2.0"

    def pre_check(self, ex: Dict[str, Any]) -> Optional[str]:
        # dataset-specific drop reasons that fire BEFORE the 6 phases
        if ex.get("status") == "rejected":
            return "rejected_upstream"
        return None

    def extract_messages(self, ex: Dict[str, Any]) -> Any:
        # return a list of {role, content} (assemble from input/output if needed)
        return ex.get("messages")

    def get_row_id(self, ex: Dict[str, Any], idx: int) -> str:
        return str(ex.get("id") or idx)

    # optional overrides
    def extract_subsource_raw(self, ex): ...
    def infer_language(self, *, ex, messages, default_lang="en"): ...
    def infer_domain(self, *, subsource_raw, prompt_text, reasoning_text): ...
    def infer_reasoning_type(self, *, domain, reasoning_text): ...
    def extract_extras(self, ex) -> Dict[str, Any]: ...
```

Add the file under [`adapters/`](adapters/) — pipeline, schema, filters,
and CLI are shared.

## Adapters

| Adapter | Source dataset | Notes |
|---|---|---|
| `dolci_think_sft_7b`            | `allenai/Dolci-Think-SFT-7B`                      | pre-check: `missing_dataset_source` |
| `openthoughts3`                 | `open-thoughts/OpenThoughts3-1.2M`                | normalises `conversations` → `messages` (ShareGPT shape) |
| `mixture_of_thoughts`           | `open-r1/Mixture-of-Thoughts`                     | parameterised by `config_name` (`math`/`code`/`science`) |
| `nemotron_post_training_v2`     | `nvidia/Nemotron-Post-Training-Dataset-v2`        | parameterised by `split_name` (9 splits, 5 multilingual) |
| `llama_nemotron_post_training`  | `nvidia/Llama-Nemotron-Post-Training-Dataset`     | assembles messages from `system_prompt` + `input` + `output` |
| `nemotron_math_v2`              | `nvidia/Nemotron-Math-v2`                         | wraps `reasoning_content` → `<think>`; pre-check: `empty_reasoning_content`; ground truth via `expected_answer` |
| `nemotron_math_proofs_v1`       | `nvidia/Nemotron-Math-Proofs-v1`                  | pre-check: `no_verified_proof` (empty `messages`); ground truth via `formal_statement` (Lean 4) |
| `nemotron_competitive_programming` | `nvidia/Nemotron-Competitive-Programming-v1`   | wraps `reasoning_content` → `<think>`; constant `domain="code"` |
| `am_deepseek_r1_distilled_1p4m` | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`          | reads pre-extracted `info.think_content`/`info.answer_content`; multi-tier domain inference |
| `synthetic_2_sft_verified`      | `PrimeIntellect/SYNTHETIC-2-SFT-verified`         | domain via `task_type` lookup (`verifiable_math` → `math`, ...) |
| `soofi_think_sft_10b_english`   | `toroe/Soofi-Think-SFT-10B-multilingual` (EN)     | constant `language="en"` |
| `soofi_think_sft_10b_multilingual` | `toroe/Soofi-Think-SFT-10B-multilingual` (other) | pre-check: `missing_language`; converts `"german"` → `"de"` etc. |
