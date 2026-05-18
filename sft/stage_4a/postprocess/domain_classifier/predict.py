"""Predict domains for null-domain rows using the trained FastText classifier.

Reads the SFT Parquet, finds rows where domain is still null AFTER the
rule-based backfill in stage4a_clean_dataset.py (i.e. not covered by the 5
known saumyamalik subsources), predicts their domain using the FastText model,
and writes a CSV mapping:

    example_id, predicted_domain, confidence, subsource_raw

This CSV is then passed to stage4a_clean_dataset.py via --predictions-csv.

Design note on "multilingual" label
------------------------------------
The FastText model is trained on English data that includes a "multilingual"
domain class.  "multilingual" is a language tag, not a content domain.  If the
model predicts "multilingual" for an English null-domain row we skip it and
use the second-best prediction instead (the actual content domain: math, code,
etc.).  This fallback does not apply to non-English rows.

Input
-----
A Parquet file containing columns:
  example_id, domain, subsource_raw, context_messages, language

Output
------
  A CSV file: example_id, predicted_domain, confidence, subsource_raw

Usage
-----
    python predict.py \\
        --input  /path/to/stage4a_input.parquet \\
        --model  /path/to/domain_classifier.bin \\
        --output /path/to/null_domain_predictions.csv

    # Smoke test (first 1 000 rows):
    python predict.py \\
        --input    /path/to/stage4a_input.parquet \\
        --model    /path/to/domain_classifier.bin \\
        --output   /tmp/predictions_test.csv \\
        --max-rows 1000
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _patch_fasttext_numpy2() -> None:
    """Patch fasttext for numpy ≥ 2.0 compatibility (see train.py for details)."""
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
        return
    import fasttext.FastText as ft_mod  # type: ignore
    _orig = np.array

    def _patched(obj, *args, copy=None, **kwargs):
        if copy is False:
            return np.asarray(obj, *args, **kwargs)
        if copy is not None:
            kwargs["copy"] = copy
        return _orig(obj, *args, **kwargs)

    ft_mod.np.array = _patched


# Subsources covered by the rule-based step in stage4a_clean_dataset.py.
# We skip these rows here so we don't overwrite what the rule already fills.
RULE_BASED_SUBSOURCES: frozenset[str] = frozenset({
    "saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2",
    "saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2",
    "saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2",
    "saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2",
    "saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2",
})


def extract_user_text(context_messages: list[dict] | None) -> str:
    """Return the first user message content (single line), or 'empty' if none."""
    if not context_messages:
        return "empty"
    for msg in context_messages:
        if msg.get("role") == "user":
            text = msg.get("content", "")
            if text:
                return text.replace("\n", " ").strip()
    return "empty"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Predict domains for null-domain rows using the trained FastText model. "
            "Writes a CSV (example_id, predicted_domain, confidence, subsource_raw) "
            "for use by stage4a_clean_dataset.py --predictions-csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Input Parquet file (same file that will be fed to stage4a_clean_dataset.py).",
    )
    p.add_argument(
        "--model", required=True,
        help="Path to the trained FastText .bin model (output of train.py).",
    )
    p.add_argument(
        "--output", required=True,
        help="Path to write the predictions CSV.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.3,
        help="Min confidence to accept a prediction. Rows below threshold are still written "
             "(just flagged); stage4a_clean_dataset.py applies all predictions regardless.",
    )
    p.add_argument("--max-rows",   type=int, default=None)
    p.add_argument("--batch-size", type=int, default=200_000)
    args = p.parse_args(argv)

    import fasttext  # type: ignore
    _patch_fasttext_numpy2()

    print(f"Loading model: {args.model}")
    model = fasttext.load_model(args.model)
    print(f"  Labels: {model.labels}")

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf         = pq.ParquetFile(str(input_path))
    total_rows = pf.metadata.num_rows

    print(f"Input       : {input_path}")
    print(f"Output CSV  : {output_path}")
    print(f"Total rows  : {total_rows:,}")
    print(f"Conf threshold: {args.threshold}")
    if args.max_rows:
        print(f"** SMOKE TEST: first {args.max_rows:,} rows only **")

    cols_needed = ["example_id", "domain", "subsource_raw", "context_messages", "language"]

    pred_counts    = Counter()
    low_conf_count = 0
    total_null     = 0
    total_predicted = 0
    rows_read      = 0
    t0 = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["example_id", "predicted_domain", "confidence", "subsource_raw"])

        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols_needed):
            if args.max_rows:
                remaining = args.max_rows - rows_read
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)

            rows_read += batch.num_rows
            dom_col  = batch.column("domain")
            sub_col  = batch.column("subsource_raw")
            eid_col  = batch.column("example_id")
            msg_col  = batch.column("context_messages")
            lang_col = batch.column("language")

            # Collect indices of null-domain rows not covered by rule-based step
            indices: list[int] = []
            texts:   list[str] = []
            for i in range(batch.num_rows):
                if dom_col[i].as_py():
                    continue
                if sub_col[i].as_py() in RULE_BASED_SUBSOURCES:
                    continue
                total_null += 1
                indices.append(i)
                texts.append(extract_user_text(msg_col[i].as_py()))

            if not texts:
                continue

            # Predict top-3 so we can skip "multilingual" for English rows
            pred_labels, pred_probs = model.predict(texts, k=3)

            for idx, labels, probs in zip(indices, pred_labels, pred_probs):
                lang = lang_col[idx].as_py()

                pred_domain = None
                confidence  = 0.0
                for lbl, prob in zip(labels, probs):
                    candidate = lbl.replace("__label__", "")
                    if candidate == "multilingual" and lang == "en":
                        continue
                    pred_domain = candidate
                    confidence  = float(prob)
                    break

                if pred_domain is None:
                    pred_domain = labels[0].replace("__label__", "")
                    confidence  = float(probs[0])

                if confidence < args.threshold:
                    low_conf_count += 1

                eid = eid_col[idx].as_py()
                sub = sub_col[idx].as_py()
                writer.writerow([eid, pred_domain, f"{confidence:.4f}", sub])
                pred_counts[pred_domain] += 1
                total_predicted += 1

            elapsed = time.time() - t0
            pct     = rows_read / total_rows * 100
            print(
                f"  Read {rows_read:>10,} / {total_rows:,} ({pct:5.1f}%) | "
                f"null={total_null:,}  predicted={total_predicted:,} | {elapsed:.0f}s",
                end="\r",
            )

    print()
    output_size = output_path.stat().st_size / 1e6

    print(f"\n{'='*60}")
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"  Null-domain rows (excl. rule-based): {total_null:,}")
    print(f"  Predictions written                : {total_predicted:,}")
    print(f"  Low confidence (< {args.threshold}): {low_conf_count:,}")
    print(f"\nPredicted domain distribution:")
    for dom, count in pred_counts.most_common():
        print(f"  {dom:25s}  {count:>8,}")
    print(f"\nOutput: {output_path} ({output_size:.1f} MB)")


if __name__ == "__main__":
    main()
