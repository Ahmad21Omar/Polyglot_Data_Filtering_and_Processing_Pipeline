# Stage 1 — Dataset-level elimination

This stage rejects entire datasets **before any row is examined**. There
are two rejection criteria.

## 1A — Verifier-scope policy

The dataset's answer types must be scoreable by a **deterministic,
rule-based or library-based verifier** that can be hardened to
silent-pass = 0 under adversarial probing. LLM-as-judge, model-based
verifiers, and open-form answer classes without a deterministic comparison
procedure are excluded.

The full policy, the list of accepted verifier categories, and the
per-dataset consistency proof are in
[`../VERIFIER_POLICY.md`](../VERIFIER_POLICY.md).

## 1B — Overlap with already-accepted sources

A dataset is rejected when its content is fully subsumed by another
already-accepted source — independent of whether its verifier would pass
1A.

## Decisions

| Decision | Dataset | Reason |
|---|---|---|
| ✅ kept | `a-m-team/AM-Thinking-v1-RL-Dataset` | math_equiv + code_asserts + code_stdio all in scope; verifier source = `open_instruct`. |
| ✅ kept | `allenai/Dolci-Think-RL-7B` | math_equiv + code_* + if_rules all in scope; LLM-judge subsets dropped per 1A. |
| ✅ kept | `logicreasoning/logi_glue` | multi_gt / text_match in scope; templated MC / NLI. |
| ✅ kept | `nvidia/Nemotron-3-Nano-RL-Training-Blend` | code_stdio + if_rules + multiple_choice + schema_structured_outputs all in scope. |
| ✅ kept | `nvidia/Nemotron-RL-ReasoningGym-v1` | reasoning_gym library is deterministic and importable. |
| ✅ kept | `AIML-TUDA/SLR-Bench` (× 7 lang variants) | prolog_rule_induction via SWI-Prolog subprocess; language-agnostic verifier. |
| ✅ kept | `MiniMaxAI/SynLogic` | vendored MIT-licensed verifier classes (synlogic_rule_based). |
| ✅ kept | `PrimeIntellect/SYNTHETIC-2-RL` | 9 verifier types, all deterministic; LLM-judge subsets dropped per 1A. |
| ✅ kept | `TIGER-Lab/WebInstruct-verified` | math_equiv + multi_gt in scope; LLM-verifier subsets dropped per 1A. |
| ❌ skipped | `CLUTRR/v1` | Already covered via `logi_glue.cluttr` (10,099 rows in scope); CC-BY-NC prevents re-use; no new verifier code needed. See `Master_Thesis/rl_datasets/rl_clutrr_v1.md` for the analysis. |

**Result:** 9 datasets pass Stage 1 and proceed to Stage 2A.
