# Polyglot Data Filtering & Processing Pipeline

This repository reproduces two end-to-end data-processing pipelines that
produce two distinct training corpora used in the thesis:

| Corpus | Pipeline | Purpose |
|---|---|---|
| **[SFT-Collection-v2](https://huggingface.co/datasets/ahmad21omar/SFT-Collection-v2)** | [`sft/`](sft/) | Multilingual supervised fine-tuning corpus for reasoning-capable LLMs (23.9 M rows). |
| **[RL-Collection-v1](https://huggingface.co/datasets/ahmad21omar/RL-Collection-v1)** | [`rl/`](rl/) | Verifiable-rewards corpus for reasoning-LLM RL (RLVR / GRPO / PPO). (1.08 M rows.) |

Both pipelines start from public HuggingFace datasets, apply a sequence of
stage-by-stage filtering and deduplication steps, and emit one parquet ready
to upload to the Hub. The structure of the two pipelines is similar but not
identical — see the per-pipeline READMEs for the differences and rationale.

## Repository layout

```
Filtering_Pipeline/
├── README.md                    # this file (index)
├── sft/                         # SFT-Collection-v2 pipeline
│   ├── README.md
│   ├── data_loader.py
│   ├── stage_2a/                # heuristic row-level filter
│   ├── stage_2b/                # quality scoring (FineWeb-HQ, threshold 0.2)
│   ├── stage_3/                 # cross-dataset deduplication (3A, 3B, 3C)
│   ├── stage_4a/                # post-processing (cleanup, domain backfill)
│   └── stage_4b/                # multilingual extension (Soofi Think SFT 10B)
└── rl/                          # RL-Collection-v1 pipeline
    ├── README.md
    ├── FILTERING_APPROACH.md
    ├── VERIFIER_POLICY.md
    ├── stage_1_dataset_elimination/
    ├── stage_2a/                # per-dataset Format & Filter + verifier policy
    ├── stage_2b_skipped/        # quality scoring intentionally omitted
    ├── stage_3/                 # merge + exact-hash dedup + fuzzy dedup
    └── stage_4/                 # finalise schema + end-to-end validation
```

## Pipeline structural differences

The two pipelines share the **same stage numbering** so a reader familiar
with one can read the other, but the contents of each stage differ because
the data shapes and training objectives are different:

| Stage | SFT pipeline | RL pipeline |
|---|---|---|
| **0** | Download 15 public SFT datasets | Download 9 public RL datasets |
| **1** | Dataset-level rejection (no-CoT, redundant, superseded) | Dataset-level rejection by **verifier-scope policy** (no LLM judge, no model verifier) |
| **2A** | Shared 6-phase heuristic row filter | Per-dataset `filter_and_format` script (one per dataset) |
| **2B** | Learned quality classifier (FineWeb-HQ on XLM-R / FastText English) | **Intentionally skipped** — see [`rl/stage_2b_skipped/README.md`](rl/stage_2b_skipped/README.md) |
| **3A** | `Filtering_Pipeline/sft/stage_3/deduplication/stage3a_same_id_dedup.py` | Done **inline** inside each Stage 2A script (per-dataset) |
| **3B** | Cross-dataset SHA1-of-context dedup | Cross-dataset SHA1-of-**last-user-message** + intra-dataset same-GT dedup |
| **3C** | Cross-dataset fuzzy dedup (threshold 78) | Cross-dataset + intra-DS-same-GT fuzzy dedup (threshold 90, `logi_glue` intra-skip) |
| **4** | 4A post-processing + 4B multilingual extension | Finalise upload schema + E2E validation |

## Quick start

To reproduce SFT-Collection-v2:

```bash
cd sft/
# follow stage-by-stage instructions in sft/README.md
```

To reproduce RL-Collection-v1:

```bash
cd rl/
# follow stage-by-stage instructions in rl/README.md
```

Both pipelines use **PyArrow streaming** (no `pd.read_parquet` on large
files), **RAM guards** (default 40 GB free required before loading the next
batch), and **checkpoint/resume** support for the long-running dedup stages.

## License of this repository

The pipeline source code is released under the same license as the rest of
the thesis materials. Note that *the datasets produced* by these pipelines
inherit the composite licensing of their upstream sources — consult each
upstream license before redistribution. License columns on the resulting
parquet rows record the source license per row.
