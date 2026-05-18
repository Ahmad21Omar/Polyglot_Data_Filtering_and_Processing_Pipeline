"""Train a multiclass FastText domain classifier for null-domain rows.

Reads the labeled SFT parquet (rows where domain IS already set), extracts the
first user message as input text, and trains a FastText supervised model.

Training data is balanced: each domain class is capped at --max-per-class
samples using reservoir sampling, so the dominant class (math, ~58%) does not
overwhelm smaller ones.

The training text file is written to a temp directory and deleted after
training.  Only the final .bin model file is persisted.

The trained model is then used by predict.py to fill null-domain rows.

Input
-----
A Parquet file containing columns: domain, context_messages.
Rows with null domain are skipped (they are what we want to predict).

Hyperparameters (following OpenThoughts Appendix R.2.1)
-------------------------------------------------------
  dim=256, epoch=3, lr=0.1, wordNgrams=2, minCount=3

Usage
-----
    python train.py \\
        --input  /path/to/stage4a_input.parquet \\
        --output /path/to/domain_classifier.bin

    # Smoke test (first 50 000 rows):
    python train.py \\
        --input    /path/to/stage4a_input.parquet \\
        --output   /tmp/domain_classifier_test.bin \\
        --max-rows 50000
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _patch_fasttext_numpy2() -> None:
    """Patch fasttext for numpy ≥ 2.0 compatibility.

    numpy 2.x changed np.array(obj, copy=False) to raise ValueError.
    FastText calls this internally, so we swap in np.asarray when copy=False.
    """
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


def extract_user_text(context_messages: list[dict] | None) -> str | None:
    """Return the content of the first 'user' message, or None if not found."""
    if not context_messages:
        return None
    for msg in context_messages:
        if msg.get("role") == "user":
            text = msg.get("content", "")
            if text:
                return text
    return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Train FastText domain classifier on labeled rows of an SFT Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Input Parquet file (Stage 4A input or earlier). Must have 'domain' and 'context_messages'.",
    )
    p.add_argument(
        "--output", required=True,
        help="Path to write the trained FastText model (.bin).",
    )
    p.add_argument(
        "--max-per-class", type=int, default=500_000,
        help="Maximum training samples per domain class (reservoir sampling).",
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop reading after N rows (smoke test).",
    )
    p.add_argument("--batch-size", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    # FastText hyperparams
    p.add_argument("--dim",         type=int,   default=256)
    p.add_argument("--epoch",       type=int,   default=3)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--word-ngrams", type=int,   default=2)
    p.add_argument("--min-count",   type=int,   default=3)
    args = p.parse_args(argv)

    random.seed(args.seed)
    import fasttext  # type: ignore
    _patch_fasttext_numpy2()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf         = pq.ParquetFile(str(input_path))
    total_rows = pf.metadata.num_rows

    print(f"Input       : {input_path}")
    print(f"Output model: {output_path}")
    print(f"Total rows  : {total_rows:,}")
    print(f"Max per class: {args.max_per_class:,}")
    if args.max_rows:
        print(f"** SMOKE TEST: reading only first {args.max_rows:,} rows **")

    # ------------------------------------------------------------------
    # Pass 1 — Reservoir sampling of labeled rows per domain
    # ------------------------------------------------------------------
    reservoir:    dict[str, list[str]] = {}
    domain_total: Counter              = Counter()
    rows_read = 0
    t0 = time.time()

    print("\n--- Pass 1: collecting training samples ---")
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=["domain", "context_messages"]):
        if args.max_rows:
            remaining = args.max_rows - rows_read
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)

        rows_read += batch.num_rows
        dom_col = batch.column("domain")
        msg_col = batch.column("context_messages")

        for i in range(batch.num_rows):
            dom = dom_col[i].as_py()
            if not dom:
                continue

            text = extract_user_text(msg_col[i].as_py())
            if not text or len(text.strip()) < 10:
                continue

            domain_total[dom] += 1
            if dom not in reservoir:
                reservoir[dom] = []
            bucket = reservoir[dom]
            n = domain_total[dom]
            if len(bucket) < args.max_per_class:
                bucket.append(text)
            else:
                j = random.randint(0, n - 1)
                if j < args.max_per_class:
                    bucket[j] = text

        elapsed = time.time() - t0
        pct = rows_read / total_rows * 100
        print(f"  {rows_read:>10,} / {total_rows:,} ({pct:5.1f}%) | {elapsed:.0f}s", end="\r")

    print()
    print("\nSamples collected per domain:")
    total_samples = 0
    for dom in sorted(reservoir):
        n = len(reservoir[dom])
        total_samples += n
        print(f"  {dom:25s}  {n:>8,}  (total seen: {domain_total[dom]:>10,})")
    print(f"  {'TOTAL':25s}  {total_samples:>8,}")

    # ------------------------------------------------------------------
    # Pass 2 — Write temp training file and train
    # ------------------------------------------------------------------
    print("\n--- Training FastText model ---")
    model_dir = output_path.parent

    with tempfile.TemporaryDirectory(dir=str(model_dir)) as td:
        train_file = os.path.join(td, "train.txt")
        n_written  = 0

        with open(train_file, "w", encoding="utf-8") as f:
            for dom, texts in reservoir.items():
                for text in texts:
                    clean = text.replace("\n", " ").replace("\r", " ").strip()
                    if clean:
                        f.write(f"__label__{dom} {clean}\n")
                        n_written += 1

        train_size_mb = os.path.getsize(train_file) / 1e6
        print(f"  Training file : {n_written:,} lines ({train_size_mb:.1f} MB)")
        print(f"  Hyperparams   : dim={args.dim}, epoch={args.epoch}, lr={args.lr}, "
              f"wordNgrams={args.word_ngrams}, minCount={args.min_count}")

        t1    = time.time()
        model = fasttext.train_supervised(
            input       = train_file,
            dim         = args.dim,
            epoch       = args.epoch,
            lr          = args.lr,
            wordNgrams  = args.word_ngrams,
            minCount    = args.min_count,
        )
        print(f"  Training done in {time.time() - t1:.1f}s")

        model.save_model(str(output_path))
        print(f"  Model saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # Quick self-test
    # ------------------------------------------------------------------
    print("\n--- Quick validation (5 samples per class) ---")
    correct = 0
    tested  = 0
    for dom, texts in reservoir.items():
        for text in texts[:5]:
            clean = text.replace("\n", " ").strip()
            pred_labels, pred_probs = model.predict(clean, k=1)
            pred    = pred_labels[0].replace("__label__", "")
            mark    = "OK" if pred == dom else "WRONG"
            correct += int(pred == dom)
            tested  += 1
            print(f"  [{mark}] true={dom:20s} pred={pred:20s} conf={pred_probs[0]:.3f}  "
                  f"text={clean[:80]}...")

    print(f"\n  Quick accuracy: {correct}/{tested} ({correct/tested*100:.1f}%)")
    print(f"\nDone. Model at: {output_path}")


if __name__ == "__main__":
    main()
