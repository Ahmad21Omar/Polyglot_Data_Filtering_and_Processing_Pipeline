# Stage 3C Fuzzy Dedup — Threshold Decision

**Decision:** `similarity_threshold = 90` (RapidFuzz `fuzz.ratio`, Levenshtein 0–100).

## Method

Before running Stage 3C on the full 1,105,317-row input
(`rl_v1.exact_dedup.kept.parquet`), we ran a **threshold sweep** on a
uniform-random 100,000-row sample (9.05 %, seed = 42) to measure how each
threshold affects drop rate, cross-dataset vs. intra-dataset behavior, and
the rate of legitimate multi-answer rows that survive.

- Sweep script: [threshold_sweep.py](threshold_sweep.py)
- Full results: [threshold_sweep.json](../../corpora/rl/dedup_3c_fuzzy/threshold_sweep.json)
- Sweep log:    [threshold_sweep.log](../../corpora/rl/dedup_3c_fuzzy/threshold_sweep.log)

Sample order preserves verifier-strength ordering so first-seen-wins behaves
as in the full run. Same fuzzy logic, bucketing, and policy as the production
script — only the input is a sample.

## Results

| Thr | Kept | Dropped | Drop % | Cross-DS | Intra=GT | Intra≠GT kept |
|----:|------:|--------:|-------:|---------:|---------:|---------------:|
| 78  | 85,474 | 14,526 | 14.53 % | 135 | 14,391 | 12,484 |
| 85  | 86,042 | 13,958 | 13.96 % | 126 | 13,832 | 10,463 |
| **90** | **89,092** | **10,908** | **10.91 %** | **45** | **10,863** | **9,217** |
| 95  | 92,218 |  7,782 |  7.78 % |  10 |  7,772 |  7,115 |

Extrapolated to full input (×11.05):

| Thr | Drops est. | Kept est. |
|----:|-----------:|----------:|
| 78  | ~160,500 |   ~944,800 |
| 85  | ~154,200 |   ~951,100 |
| **90** | **~120,600** | **~984,700** |
| 95  |  ~86,000 | ~1,019,300 |

## Reasoning for 90

1. **Cross-DS effectiveness.** Cross-dataset fuzzy drops collapse rapidly as
   the threshold rises (135 → 126 → 45 → 10 in the sample). Above 95 we lose
   roughly 80 % of the still-findable cross-DS near-duplicates that Stage 3B
   exact-hash could not catch. At 90 we still recover ~500 cross-DS dupes in
   the full set; at 95 only ~110.

2. **Intra-DS same-GT signal.** The big mass of drops is intra-dataset with
   identical GT — mostly trivial paraphrases of the same task with the same
   answer. The drop count steps down meaningfully between 85 → 90
   (13,832 → 10,863, –22 %), then again 90 → 95 (10,863 → 7,772, –28 %). The
   85 → 90 step removes near-duplicates with minor word-order or whitespace
   edits; the 90 → 95 step starts to require almost character-identical
   prompts. 90 is the inflection point where we stop catching legitimate
   paraphrases but still catch true near-dupes.

3. **Legitimate multi-answer rows protected.** Across all thresholds the
   intra-DS *different*-GT rows are kept (per policy). At 90 we keep 9,217
   multi-answer rows in the sample (extrapolated ~102k full) — confirming
   the GT-aware policy is doing its job: only redundant same-GT pairs are
   removed.

4. **Comparison with literature.** OpenThoughts3 uses 95 (very conservative,
   web-scraped instruction data). SFT-Collection-v2 used 78 (very
   aggressive, multilingual mixed corpus). RL data sits in between: the
   prompts are more diverse than OpenThoughts but already exact-deduped by
   Stage 3B, so 90 fits — strong enough to catch near-duplicates, lenient
   enough not to merge semantically distinct prompts.

5. **Drop-rate plausibility.** A ~11 % drop after exact-hash dedup is
   consistent with what we observed in SFT-Collection-v2 (where 78 dropped
   ~17 % from a dirtier mix). Anything above 15 % at this stage would warn
   us we are killing legitimate variation; anything below 5 % would warn us
   the threshold is so conservative it adds little over Stage 3B.

## What 90 does, concretely

- Two prompts that differ only in trailing punctuation, an inserted/removed
  filler word, or word-order swaps → **drop** (>= 90).
- Two prompts with one substantive token changed (different number,
  different entity, different sub-question) → **keep** (< 90).
- All decisions still subject to the **exact-GT guard** for intra-dataset
  matches: same prompt + different ground-truth ⇒ keep.

## How to reproduce

```bash
python3 -m Filtering_Pipeline.rl.stage_3.deduplication.threshold_sweep \\
    --input /home/workdir/Master_Thesis/corpora/rl/dedup_3b_exact/rl_v1.exact_dedup.kept.parquet \\
    --sample-size 100000 \\
    --thresholds 78 85 90 95
```

## Post-run finding: `logicreasoning/logi_glue` excluded from intra-DS dedup

The first full run at threshold 90 dropped 376,652 rows (34.08 %), of which
**355,406 (94 %) came from `logicreasoning/logi_glue`** alone (–68.8 % of
that dataset). We manually inspected 20 random logi_glue drops with their
matched kept partner. Findings:

- The vast majority of "duplicates" are **legitimately distinct tasks** that
  happen to share most of the prompt text:
  - **Multiple-choice items** with the same context/question but *different
    distractor options* (e.g. "consider wearing the silly hat" vs. "decline
    wearing the silly hat"). Same correct answer text, fuzz.ratio ~97.
  - **NLI items** with the same premise but a *different hypothesis* (e.g.
    "first leader was a man" vs. "founded in November"). Both labelled
    `contradiction` for unrelated reasons, fuzz.ratio ~97.
- Only fuzz.ratio scores ≥ 99 looked like true whitespace/punctuation-only
  variants of the same task.

Root cause: logi_glue prompts are heavily **templated** (>95 % shared
boilerplate), with answer-relevant variation in only 1–2 tokens, and the GT
label space is **tiny** ({entailment, contradiction, neutral} / {True,
False} / 4-way MC). With such a small label space, unrelated items often
share the same correct answer, so the GT-aware policy can no longer
distinguish near-duplicates from legitimately distinct items. This is a
fundamental limitation of Levenshtein-on-prompt for templated MC/NLI data,
not a bug in the implementation.

**Decision**: keep threshold 90, but add `logicreasoning/logi_glue` to
`--skip-intra-datasets`. Effect:

- Cross-DS dedup is **still active** for logi_glue (its prompts are still
  indexed, so other datasets can drop a row that fuzzily matches a logi_glue
  prompt; a logi_glue row can also be dropped if it cross-matches another DS).
- Intra-DS dedup is **disabled** for logi_glue — every row in logi_glue is
  kept unless it cross-matches another dataset.

This is consistent with how we treat SLR-Bench (passthrough) but less
aggressive: logi_glue still contributes to cross-DS detection.

All other 7 deduped datasets retain the standard intra-DS-same-GT policy at
threshold 90 (per Stage 3B v2). The intra-DS drops we saw at threshold 90
for those datasets (Dolci-Think 4.8k, Synthetic-2 5.3k, SynLogic 5.0k,
etc.) are all in the few-thousand range and inspecting them is reasonable
to revisit if needed — but they are not dominated by templated MC/NLI in
the same way logi_glue is.
