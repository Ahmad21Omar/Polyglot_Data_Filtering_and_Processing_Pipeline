"""Stage 3C — Threshold sweep on a sample.

Runs the Stage 3C fuzzy-dedup logic on a random sample of rows for several
candidate thresholds, then prints a comparison table. No parquet output —
this script exists purely to inform the choice of ``--similarity-threshold``
for the full Stage 3C run.

Sample design
-------------
- Reads the Stage 3B kept file's required columns into memory.
- Selects ``--sample-size`` row indices uniformly at random (seed=42).
- Sample is sorted back into the original (verifier-strength) order so
  first-seen-wins matches the full-run behaviour.

Per-threshold algorithm
-----------------------
For each threshold, the script reruns the full bucket+match pipeline from
scratch on the same sample so the first-seen-wins effect is honoured per
threshold (index grows differently when more rows are dropped).

Usage
-----
    python -m Filtering_Pipeline.rl.stage_3.deduplication.threshold_sweep \\
        --input /home/workdir/Master_Thesis/corpora/rl/dedup_3b_exact/rl_v1.exact_dedup.kept.parquet \\
        --sample-size 100000 \\
        --thresholds 78 85 90 95
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from stage_3c_fuzzy_dedup import (
    DEFAULT_SKIP_DATASETS,
    _add_to_bucket,
    _bucket_head,
    _bucket_key,
    _bucket_key_for_bin,
    _classify_match,
    _find_matches,
    gt_fingerprint,
    normalize_prompt,
)


# ---------------------------------------------------------------------------
# Sample loader
# ---------------------------------------------------------------------------

def load_sample(input_path: Path, sample_size: int, seed: int = 42):
    """Return list of (ds_id, ex_id, norm_prompt, gt_hash, is_passthrough).

    Sample is uniform random across rows, then sorted by original index to
    preserve verifier-strength ordering for first-seen-wins.
    """
    skip_set = set(DEFAULT_SKIP_DATASETS)
    pf = pq.ParquetFile(input_path)
    total = pf.metadata.num_rows

    print(f"[sample] Input rows: {total:,}")
    rng = random.Random(seed)
    if sample_size >= total:
        sample_size = total
        keep = set(range(total))
    else:
        keep = set(rng.sample(range(total), sample_size))
    print(f"[sample] Drawing {sample_size:,} rows (seed={seed})")

    rows: list[tuple[int, str, str, str, str, bool]] = []  # (orig_idx, ds, ex, norm, gt_h, pass)
    row_idx = 0
    t0 = time.time()
    for batch in pf.iter_batches(
        batch_size=50_000,
        columns=["dataset_id", "example_id", "context_messages", "ground_truth_text"],
    ):
        ds_ids   = batch.column("dataset_id").to_pylist()
        ex_ids   = batch.column("example_id").to_pylist()
        ctx_list = batch.column("context_messages").to_pylist()
        gt_list  = batch.column("ground_truth_text").to_pylist()
        for j in range(len(ds_ids)):
            idx = row_idx + j
            if idx in keep:
                ds_s = str(ds_ids[j]) if ds_ids[j] is not None else "__MISSING__"
                ex_s = str(ex_ids[j]) if ex_ids[j] is not None else "__MISSING__"
                is_pass = ds_s in skip_set
                norm = "" if is_pass else normalize_prompt(ctx_list[j])
                gt_h = gt_fingerprint(gt_list[j])
                rows.append((idx, ds_s, ex_s, norm, gt_h, is_pass))
        row_idx += len(ds_ids)
    rows.sort(key=lambda r: r[0])
    print(f"[sample] Loaded in {time.time() - t0:.1f}s\n")
    return rows


# ---------------------------------------------------------------------------
# Evaluate one threshold
# ---------------------------------------------------------------------------

def evaluate(
    sample,
    threshold: float,
    executor: ProcessPoolExecutor,
    num_workers: int,
    bucket_tokens: int = 2,
    len_bin_size: int = 100,
    max_candidates_per_bucket: int | None = 5_000,
    max_len_diff: int | None = 40,
    neighbor_bin_radius: int = 1,
    match_limit: int = 20,
    batch_size: int = 20_000,
) -> dict:
    prompt_records: dict[str, list[tuple[str, str, str]]] = {}
    bucket_index:   dict[str, deque]                       = {}

    kept = dropped = passthrough = 0
    dropped_cross = dropped_intra_same_gt = intra_kept_diff_gt = 0
    kept_by_ds = Counter()
    dropped_cross_by_ds = Counter()
    dropped_intra_by_ds = Counter()
    intra_kept_diff_gt_by_ds = Counter()
    dropped_pairs = Counter()

    n_total = len(sample)
    t0 = time.time()

    for start in range(0, n_total, batch_size):
        chunk = sample[start : start + batch_size]
        local_records: dict[str, list[tuple[str, str, str]]] = {}
        local_bucket: dict[str, deque]                       = {}

        tasks = []
        for (_, ds_s, ex_s, norm, gt_h, is_pass) in chunk:
            if is_pass or not norm:
                tasks.append(("", [], max_len_diff, threshold, match_limit))
                continue
            candidates: list[str] = []
            head     = _bucket_head(norm, bucket_tokens)
            base_bin = len(norm) // max(1, len_bin_size)
            for offset in range(-neighbor_bin_radius, neighbor_bin_radius + 1):
                bkey = _bucket_key_for_bin(head, base_bin + offset)
                if bkey in bucket_index:
                    candidates.extend(bucket_index[bkey])
                if bkey in local_bucket:
                    candidates.extend(local_bucket[bkey])
            rec = (ds_s, ex_s, gt_h)
            if norm not in local_records and norm not in prompt_records:
                _add_to_bucket(
                    local_bucket,
                    _bucket_key(norm, bucket_tokens, len_bin_size),
                    norm,
                    max_candidates_per_bucket,
                )
            local_records.setdefault(norm, []).append(rec)
            tasks.append((norm, candidates, max_len_diff, threshold, match_limit))

        results = list(executor.map(_find_matches, tasks, chunksize=100))

        for i, (_, ds_s, ex_s, norm, gt_h, is_pass) in enumerate(chunk):
            if is_pass:
                kept += 1
                passthrough += 1
                kept_by_ds[ds_s] += 1
                continue
            if not norm:
                kept += 1
                kept_by_ds[ds_s] += 1
                continue

            matches = results[i]
            # Drop self if it's the only occurrence
            matches_no_self = []
            for (mn, sc) in matches:
                if mn == norm:
                    other = (
                        len(prompt_records.get(mn, [])) > 0
                        or len([r for r in local_records.get(mn, []) if r[1] != ex_s or r[0] != ds_s]) > 0
                    )
                    if not other:
                        continue
                matches_no_self.append((mn, sc))

            decision, m_ds, m_ex, score = _classify_match(
                ds_s, gt_h, matches_no_self, prompt_records, local_records,
            )

            if decision == "drop_cross":
                dropped += 1
                dropped_cross += 1
                dropped_cross_by_ds[ds_s] += 1
                if m_ds:
                    dropped_pairs[(ds_s, m_ds)] += 1
            elif decision == "drop_intra_same_gt":
                dropped += 1
                dropped_intra_same_gt += 1
                dropped_intra_by_ds[ds_s] += 1
            else:
                kept += 1
                kept_by_ds[ds_s] += 1
                if m_ds is not None:
                    intra_kept_diff_gt += 1
                    intra_kept_diff_gt_by_ds[ds_s] += 1
                rec = (ds_s, ex_s, gt_h)
                if norm not in prompt_records:
                    _add_to_bucket(
                        bucket_index,
                        _bucket_key(norm, bucket_tokens, len_bin_size),
                        norm,
                        max_candidates_per_bucket,
                    )
                    prompt_records[norm] = [rec]
                else:
                    prompt_records[norm].append(rec)

    elapsed = time.time() - t0
    return {
        "threshold":             threshold,
        "n_total":               n_total,
        "kept":                  kept,
        "dropped":               dropped,
        "passthrough":           passthrough,
        "dropped_cross":         dropped_cross,
        "dropped_intra_same_gt": dropped_intra_same_gt,
        "intra_kept_diff_gt":    intra_kept_diff_gt,
        "drop_rate_pct":         round(dropped / n_total * 100, 3) if n_total else 0,
        "elapsed_sec":           round(elapsed, 1),
        "kept_by_ds":            dict(kept_by_ds),
        "dropped_cross_by_ds":   dict(dropped_cross_by_ds),
        "dropped_intra_by_ds":   dict(dropped_intra_by_ds),
        "intra_kept_diff_gt_by_ds": dict(intra_kept_diff_gt_by_ds),
        "top_pairs":             dropped_pairs.most_common(10),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]) -> None:
    print(f"\n{'=' * 95}")
    print("THRESHOLD SWEEP — COMPARISON")
    print(f"{'=' * 95}")
    header = (
        f"{'thr':>5} | {'kept':>9} | {'drop':>9} | {'%':>6} | "
        f"{'cross':>7} | {'intra=gt':>9} | {'intra≠gt':>9} | {'pass':>9} | {'sec':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['threshold']:>5.1f} | "
            f"{r['kept']:>9,} | "
            f"{r['dropped']:>9,} | "
            f"{r['drop_rate_pct']:>6.2f} | "
            f"{r['dropped_cross']:>7,} | "
            f"{r['dropped_intra_same_gt']:>9,} | "
            f"{r['intra_kept_diff_gt']:>9,} | "
            f"{r['passthrough']:>9,} | "
            f"{r['elapsed_sec']:>5.0f}"
        )
    print(f"{'=' * 95}\n")

    for r in results:
        print(f"\n[threshold={r['threshold']}]  cross-ds drops by source dataset (top 8):")
        rows = sorted(r["dropped_cross_by_ds"].items(), key=lambda x: -x[1])[:8]
        for ds, c in rows:
            print(f"   {ds:<55s} {c:>6,}")
        print(f"[threshold={r['threshold']}]  intra-ds same-GT drops by dataset (top 8):")
        rows = sorted(r["dropped_intra_by_ds"].items(), key=lambda x: -x[1])[:8]
        for ds, c in rows:
            print(f"   {ds:<55s} {c:>6,}")
        print(f"[threshold={r['threshold']}]  intra-ds diff-GT KEPT by dataset (top 8):")
        rows = sorted(r["intra_kept_diff_gt_by_ds"].items(), key=lambda x: -x[1])[:8]
        for ds, c in rows:
            print(f"   {ds:<55s} {c:>6,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Threshold sweep for Stage 3C fuzzy dedup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True)
    p.add_argument("--sample-size", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thresholds", type=float, nargs="+", default=[78.0, 85.0, 90.0, 95.0])
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--output-json",
                   default="/home/workdir/Master_Thesis/corpora/rl/dedup_3c_fuzzy/threshold_sweep.json")
    args = p.parse_args(argv)

    sample = load_sample(Path(args.input), args.sample_size, args.seed)

    results: list[dict] = []
    print(f"[sweep] Evaluating {len(args.thresholds)} thresholds on {len(sample):,} rows ...\n")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for thr in args.thresholds:
            print(f"--- threshold = {thr} ---")
            r = evaluate(sample, thr, executor, args.num_workers)
            results.append(r)
            print(
                f"  kept={r['kept']:,}  dropped={r['dropped']:,}  "
                f"cross={r['dropped_cross']:,}  intra=gt={r['dropped_intra_same_gt']:,}  "
                f"intra≠gt-kept={r['intra_kept_diff_gt']:,}  "
                f"({r['elapsed_sec']}s)"
            )

    print_comparison(results)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    out_path.write_text(json.dumps({
        "input":       args.input,
        "sample_size": len(sample),
        "seed":        args.seed,
        "results":     [{k: v for k, v in r.items() if k != "top_pairs"} for r in results],
        "top_pairs":   {str(r["threshold"]): [{"from": a, "to": b, "count": c} for (a, b), c in r["top_pairs"]] for r in results},
    }, indent=2, ensure_ascii=False))
    print(f"\n[sweep] Written: {out_path}")


if __name__ == "__main__":
    main()
