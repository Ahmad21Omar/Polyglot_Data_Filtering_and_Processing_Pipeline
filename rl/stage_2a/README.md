# Stage 2A — Per-dataset Filter & Format

This stage applies row-level filtering and produces a row in the canonical
**`rl_schema_v1`** (20 internal columns, 19 in the upload-ready schema).
**Stage 3A — intra-dataset deduplication — is performed inline** at the
end of each script (via `sha256(prompt)` or `sha256(lang \0 prompt)` for
multilingual datasets), so there is no separate Stage 3A directory.

## Why one script per dataset

The SFT pipeline factored Stage 2A into a shared 6-phase pipeline because
all SFT rows have the same `{reasoning_text, response_text}` shape. RL
rows are heterogeneous by verifier — code rows need `assert` extraction,
math rows need numeric normalisation, prolog rows need `head :- body`
extraction — so a shared pipeline would force per-verifier branching back
in. Each script in [`filter_and_format/`](filter_and_format/) is therefore
self-contained, but they all apply the same five conceptual steps:

1. **Per-row verifier-scope check** — drop rows the policy excludes
   (e.g. `requires_model_verifier` in WebInstruct,
   `ungradeable_verifier_llm_judge_*` in Dolci).
2. **Structural quality filters** — empty prompt, prompt-too-short
   (typically < 30 chars; SLR uses < 100), URL / image references,
   malformed answer encoding (e.g. `boolean_non_canonical`,
   `integer_not_integer`).
3. **Per-verifier validation** — for example, `math_equiv` rows must have
   an extractable numeric; `prolog_rule_induction` rows must have a
   `head :- body` shape (defense-in-depth against the IPT shortcut
   exploit); fraction rows must contain a recognisable LaTeX / text
   fraction.
4. **`rl_schema_v1` conversion** — produce the canonical row with
   `verifier_type`, `verifier_source`, `ground_truth_text`,
   `verification_info_raw` (JSON), plus standard metadata.
5. **Intra-dataset dedup (Stage 3A inline)** — `sha256(prompt)` keyed
   dedup. Drop reasons recorded as `duplicate_prompt` /
   `duplicate_question` in the per-dataset `dropped.parquet` and
   `FILTERING_REPORT.md`.

## Layout

```
stage_2a/
├── README.md                        # this file
├── filter_and_format/               # 9 dataset-specific scripts
│   ├── data_loader.py               # shared HF-dataset loader
│   ├── filter_and_format_am_thinking_v1_rl.py
│   ├── filter_and_format_dolci_think_rl_7b.py
│   ├── filter_and_format_logi_glue.py
│   ├── filter_and_format_nemotron_3_nano_rl_blend.py
│   ├── filter_and_format_nemotron_rl_reasoning_gym_v1.py
│   ├── filter_and_format_slr_bench.py
│   ├── filter_and_format_synlogic.py
│   ├── filter_and_format_synthetic2_rl.py
│   └── filter_and_format_webinstruct_verified.py
└── filters/                         # shared reusable filters
    ├── english_filter.py            # FastText English filter
    ├── ground_truth_filter.py       # GT-shape validators (per verifier)
    ├── passrate_filter.py           # AM-Thinking / Dolci passrate cutoff
    ├── prompt_length_filter.py
    ├── prompt_reference_filter.py   # URL / image references
    ├── prompt_repetition_filter.py
    └── README.md
```

## Running

Each script runs independently and writes to a per-dataset output dir:

```bash
python3 -m Filtering_Pipeline.rl.stage_2a.filter_and_format.filter_and_format_<dataset>
```

Output per dataset:

```
corpora/rl/<dataset>/
├── rl_<dataset>_v1.kept.parquet     # rows that passed all filters
├── rl_<dataset>_v1.dropped.parquet  # rows that failed, with _drop_reason
└── FILTERING_REPORT.md              # per-filter drop counts, distributions
```

## Aggregate result

After Stage 2A the 9 datasets total **1,127,950 rows / 9.0 GB** of parquet,
all sharing the `rl_schema_v1` schema. These 9 files are the input to
[`Stage 3`](../stage_3/).
