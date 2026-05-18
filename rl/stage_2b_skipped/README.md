# Stage 2B — Quality scoring (intentionally skipped)

The SFT pipeline runs a learned quality classifier here
([SFT Stage 2B](../../sft/stage_2b/quality_filtering/)) and drops the
lowest-scoring 20 % of rows. For RL data this step is **deliberately
omitted**. This directory exists only to make the omission explicit and to
document the reasoning, so a reader auditing the pipeline does not assume
Stage 2B was forgotten.

## Why Stage 2B is skipped for RL

### 1. Row shape is different

SFT rows contain `reasoning_text` and `response_text` — natural inputs for
a "is this good chain-of-thought / good answer?" classifier. RL rows
contain only a *prompt* plus a *verifier configuration* — the answer comes
from the policy model at training time, not from the dataset. **There is
no answer to score.**

### 2. RL prompts are synthetic templates by construction

Most of the RL data is procedurally generated (SLR-Bench, SynLogic,
reasoning_gym) or tightly-formatted instruction-following (IFEval, MCQ,
code asserts). A FineWeb-HQ-style classifier scores naturalistic web text
— on synthetic templates its score distribution is flat and
non-discriminative. Dropping the bottom 20 % would mostly drop *short*
prompts, not *low-quality* prompts, because the classifier has no signal
in this regime.

### 3. The verifier is the RL-time quality gate

Bad prompts produce rollouts that never get reward 1.0, so the RL policy
gradient simply ignores them. This is structurally different from SFT,
where bad rows go directly into the cross-entropy loss and must be
excluded pre-training. The verifier-scope policy
([Stage 1A](../stage_1_dataset_elimination/) + Stage 2A.1) already
enforces the property we actually care about: every kept row admits a
hardenable reward signal.

## Empirical justification

The per-dataset hardening reports show **0 silent-pass results across
≥ 30 k adversarial probes** in total. Any prompt that the verifier could
misbehave on is already excluded by the time we leave Stage 2A. There is
no marginal value left for a learned quality filter to find.

## Where the trade-off bites

Without Stage 2B we may include prompts that are technically valid but
pedagogically weak (trivial / too easy, exotic edge cases, out-of-
distribution puzzles). For RL this manifests as low signal density, **not
a wrong training signal**. A stronger policy model simply learns these
quickly and moves on.

If pathological training behaviour is ever observed and traced to specific
prompt families, the correct response is a **curriculum or sampling-weight
adjustment** in downstream training, not a learned quality filter applied
pre-training.
