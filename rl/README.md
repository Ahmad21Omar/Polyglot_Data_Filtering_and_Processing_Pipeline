# RL-Collection-v1 — Filtering Pipeline

This directory reproduces the full data-filtering pipeline behind
**[RL-Collection-v1](https://huggingface.co/datasets/ahmad21omar/RL-Collection-v1)**,
a unified, verifiable-rewards corpus for reinforcement learning of
reasoning-capable language models.

## Dataset at a glance

| Property | Value |
|---|---|
| HuggingFace ID | `ahmad21omar/RL-Collection-v1` |
| Total rows | 1,084,071 |
| Size | 4.73 GB (snappy parquet) |
| Languages | EN (majority), DE, ES, FR, IT, PT, NL (SLR-Bench multilingual), ZH (SynLogic) |
| Source datasets | 9 |
| Verifier types | 14 (math_equiv, code_stdio, prolog_rule_induction, …) |
| Key columns | `context_messages`, `verifier_type`, `verifier_source`, `ground_truth_text`, `verification_info_raw` |

## Pipeline overview

```
Stage 0  Download                   9 public HuggingFace datasets    ~1.59 M rows
Stage 1  Dataset-level elimination  verifier-scope policy            9 pass (1 skipped)
Stage 2A Filter & Format            per-dataset, structural + scope  1,127,950 rows
Stage 2B Quality scoring            SKIPPED (see stage_2b_skipped/)  —
Stage 3A Intra-dataset dedup        inline inside Stage 2A           (no separate step)
Stage 3  Merge → 3B → 3C            cross-dataset dedup              1,084,071 rows
Stage 4  Finalise + E2E validation  19-column upload schema          1,084,071 rows (final)
```

## Design rationale

For the **conceptual rationale** of why this pipeline diverges from SFT
(skipped Stage 2B, inline Stage 3A, GT-aware dedup, verifier-strength merge
ordering) read [`FILTERING_APPROACH.md`](FILTERING_APPROACH.md) first.

For the **single unified verifier rule** that all 9 datasets are gated on
(deterministic / rule-based / library-based, hardenable to silent-pass=0)
read [`VERIFIER_POLICY.md`](VERIFIER_POLICY.md).

These two documents are the load-bearing rationale: they explain *why* the
pipeline is shaped the way it is. The per-stage READMEs below explain
*what* runs and *what* it produces.

## Stages

| Stage | Directory | What it does |
|---|---|---|
| **1** | [`stage_1_dataset_elimination/`](stage_1_dataset_elimination/) | Dataset-level rejection by verifier scope |
| **2A** | [`stage_2a/`](stage_2a/) | Per-dataset Format & Filter (9 scripts + shared filters) |
| **2B** | [`stage_2b_skipped/`](stage_2b_skipped/) | Why we deliberately skip quality scoring for RL |
| **3** | [`stage_3/`](stage_3/) | Merge + cross-dataset exact-hash + fuzzy dedup |
| **4** | [`stage_4/`](stage_4/) | Finalise schema + end-to-end validation suite |

## Reproducibility

All scripts are **streaming** (PyArrow `iter_batches` + `ParquetWriter`),
**RAM-safe** (default 40 GB free guard), and **resumable** (JSON checkpoints
in the output dir). The pipeline can be re-run from any stage given the
output of the previous stage.

### End-to-end command sequence

```bash
# Stage 2A — Filter & format each of the 9 datasets independently
for ds in am_thinking_v1_rl dolci_think_rl_7b logi_glue \
          nemotron_3_nano_rl_blend nemotron_rl_reasoning_gym_v1 \
          slr_bench synlogic synthetic2_rl webinstruct_verified; do
    python3 -m Filtering_Pipeline.rl.stage_2a.filter_and_format.filter_and_format_${ds}
done

# Stage 3 — Merge then deduplicate
python3 Filtering_Pipeline/rl/stage_3/deduplication/stage_3_00_merge.py \
    --in-root corpora/rl/ \
    --out-root corpora/rl/merged_rl_ds/

python3 Filtering_Pipeline/rl/stage_3/deduplication/stage_3b_exact_hash_dedup.py \
    --input corpora/rl/merged_rl_ds/rl_v1_merged.kept.parquet \
    --output-dir corpora/rl/dedup_3b_exact/

# (Optional) Threshold sweep before the final fuzzy run
python3 Filtering_Pipeline/rl/stage_3/deduplication/threshold_sweep.py \
    --input corpora/rl/dedup_3b_exact/rl_v1.exact_dedup.kept.parquet \
    --sample-size 100000 \
    --thresholds 78 85 90 95

python3 Filtering_Pipeline/rl/stage_3/deduplication/stage_3c_fuzzy_dedup.py \
    --input corpora/rl/dedup_3b_exact/rl_v1.exact_dedup.kept.parquet \
    --output-dir corpora/rl/dedup_3c_fuzzy/ \
    --similarity-threshold 90

# Stage 4 — Finalise + validate
python3 Filtering_Pipeline/rl/stage_4/stage_4_finalize.py \
    --input corpora/rl/dedup_3c_fuzzy/rl_v1.fuzzy_dedup.kept.parquet \
    --output-dir corpora/rl/final/

python3 Filtering_Pipeline/rl/stage_4/e2e_test.py \
    --input corpora/rl/final/rl_v1_final.parquet \
    --output-dir corpora/rl/final/
```

Every script writes a structured JSON/Markdown report next to its parquet
output, so the run is fully auditable after the fact.

## Numbers (final pipeline run)

| Stage | Input | Output | Dropped | Drop % |
|---|---:|---:|---:|---:|
| **2A** Filter & Format (sum over 9 datasets) | ~1,590,000 | 1,127,950 | — | — |
| **3 merge** | 1,127,950 | 1,127,950 | 0 | — |
| **3B** Exact-hash dedup | 1,127,950 | 1,105,317 | −22,633 | −2.0 % |
| **3C** Fuzzy dedup (threshold 90, logi_glue intra-skip) | 1,105,317 | 1,084,071 | −21,246 | −1.9 % |
| **4** Finalise (drop dedup internals) | 1,084,071 | **1,084,071** | 0 | — |

Final output: **1,084,071 rows · 4.73 GB · SHA-256
`fa76f51cfa3fa3e6151686f334f7d4a9a42214660479baa2041366888a256a8f`**
(verified by `Stage 4`).

## Source datasets

| Dataset | Verifier family | Rows kept | License |
|---|---|---:|---|
| `logicreasoning/logi_glue` | `multi_gt`, `text_match` | 516,447 | composite_research_only |
| `TIGER-Lab/WebInstruct-verified` | `math_equiv`, `multi_gt` | 133,192 | tiger_lab_research_use |
| `PrimeIntellect/SYNTHETIC-2-RL` | 9 verifier types | 99,589 | apache-2.0 |
| `allenai/Dolci-Think-RL-7B` | `code_*`, `if_rules`, `math_equiv` | 65,244 | ODC-BY |
| `a-m-team/AM-Thinking-v1-RL-Dataset` | `code_*`, `math_equiv` | 52,962 | apache-2.0 |
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` | `code_stdio`, `if_rules`, `mcqa`, `schema` | 50,673 | CC-BY-4.0 |
| `MiniMaxAI/SynLogic` | `synlogic_rule_based` | 28,871 | mit |
| `AIML-TUDA/SLR-Bench` (+ 6 lang variants) | `prolog_rule_induction` | 126,189 (18,027 ea.) | cc-by-4.0 |
| `nvidia/Nemotron-RL-ReasoningGym-v1` | `reasoning_gym` (99 families) | 10,904 | CC-BY-4.0 |
