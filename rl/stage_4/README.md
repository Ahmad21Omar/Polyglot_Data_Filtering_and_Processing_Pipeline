# Stage 4 — Finalise & End-to-End Validation

Stage 4 takes the deduplicated parquet from [Stage 3](../stage_3/) and
produces the **upload-ready** dataset: a single parquet with exactly the
19 canonical `rl_schema_v1` columns, plus a structured validation report
that documents every property a downstream consumer (RL trainer, NeMo
Gym Resource Server) can rely on.

Unlike SFT, this pipeline has **no Stage 4B multilingual extension** —
SLR-Bench already supplies the 7 language variants directly during Stage
2A, so the multilingual coverage is fixed at Stage 2A time.

## Layout

```
stage_4/
├── README.md             # this file
├── stage_4_finalize.py   # strip dedup internals, emit canonical schema
└── e2e_test.py           # streaming validation suite
```

## Step 4.1 — Finalise

`stage_4_finalize.py` performs a single streaming pass over the Stage 3C
output and writes a new parquet that contains **exactly** the 19 canonical
columns in canonical order:

```
dataset_id, dataset_version_date, example_id, row_id, subsource_raw,
source_dataset_id, license, used_by_model, context_messages, language,
domain, ability, difficulty, verifier_type, verifier_source,
ground_truth_text, verification_info_raw, avg_reward, reward_model_metadata
```

Internal dedup columns (`_dedup_hash`, `_dedup_norm`, `_dedup_seen_in_ds`)
are dropped. The script also emits a `schema.json` (column list + per-row
counts + SHA-256 of the output) and a `FINAL_DATASET_REPORT.md`.

**Running:**

```bash
python3 stage_4_finalize.py \
    --input corpora/rl/dedup_3c_fuzzy/rl_v1.fuzzy_dedup.kept.parquet \
    --output-dir corpora/rl/final/
```

**Output:**

```
corpora/rl/final/
├── rl_v1_final.parquet            # 4.73 GB, 1,084,071 rows, 19 cols, snappy
├── rl_v1_final.schema.json
└── FINAL_DATASET_REPORT.md
```

## Step 4.2 — End-to-end validation

`e2e_test.py` performs a streaming validation suite over the final
parquet. Exit code is non-zero on any CRITICAL failure.

Checks:

1. **Schema** — exactly 19 canonical columns, correct Arrow types
   (`context_messages` must be `list<struct<role, content>>`).
2. **NULL / empty** on critical fields (`dataset_id`, `example_id`,
   `context_messages`, `verifier_type`, `verifier_source`). For
   `ground_truth_text`, NULL is only allowed for the two schema verifiers
   (`schema_structured_outputs`, `schema_pydantic`) where the "ground
   truth" is the schema itself, stored in `verification_info_raw`.
3. **`context_messages` structure** — each list has ≥ 1 user message,
   valid roles (`user` / `assistant` / `system` / `tool`), non-empty
   contents.
4. **`verification_info_raw`** — JSON-parseable for every non-NULL row;
   per `verifier_type` a soft check on required keys.
5. **`example_id` global uniqueness** — single-pass hash-set comparison.
6. **Verifier smoke tests** — feed the GT back to a simple in-process
   verifier for `math_equiv`, `text_match`, `multiple_choice`, and
   `structured_match`. 50 samples per type. Heavy verifiers (`code_*`,
   `prolog_rule_induction`, `reasoning_gym`, `synlogic_rule_based`,
   `puzzle_match`, `if_rules`, `schema_*`) are not executed here — they
   were validated per-dataset in the NeMo Gym Resource Server test suites
   before merging.
7. **Distributions** — `dataset_id`, `verifier_type`, `language`,
   `domain`, `ability`, `difficulty`, `license`, plus a `dataset_id ×
   verifier_type` cross-tab printed for visual sanity check.

**Running:**

```bash
python3 e2e_test.py \
    --input corpora/rl/final/rl_v1_final.parquet \
    --output-dir corpora/rl/final/
```

**Output:**

```
corpora/rl/final/
├── E2E_REPORT.md      # markdown summary
└── e2e_run.log
```

## Final-run results

The production run on `rl_v1_final.parquet` (1,084,071 rows) passed all
checks:

- ✅ 19 / 19 canonical columns, correct Arrow types
- ✅ 0 NULLs in critical fields
- ✅ 1,084,071 / 1,084,071 unique `example_id`
- ✅ All `verification_info_raw` parse cleanly as JSON
- ✅ All `context_messages` valid (≥ 1 user message, valid roles)
- ✅ Smoke tests: 50 / 50 for math_equiv, text_match, multiple_choice,
  structured_match

Output SHA-256:
`fa76f51cfa3fa3e6151686f334f7d4a9a42214660479baa2041366888a256a8f`
