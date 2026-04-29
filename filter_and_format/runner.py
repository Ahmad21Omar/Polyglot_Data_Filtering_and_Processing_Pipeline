from __future__ import annotations

"""Shared runner for any per-dataset adapter.

Loads a HuggingFace dataset from disk, applies the generic 6-phase pipeline
through ``adapter.map_row``, then saves the kept and dropped rows separately
as Parquet files. Used by every per-dataset CLI script (and directly callable
from notebooks / tests).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, load_from_disk

from .adapter import DatasetAdapter
from .config import FilterConfig
from .output_schema import KEPT_COLUMNS
from .pipeline import map_row


def _ensure_canonical_columns(ds: Dataset) -> Dataset:
    """Reorder columns to match :data:`...output_schema.KEPT_COLUMNS`.

    Tolerates extra columns (drops them) and missing columns (raises).
    """
    missing = [c for c in KEPT_COLUMNS if c not in ds.column_names]
    if missing:
        raise RuntimeError(
            f"Mapped dataset is missing canonical columns: {missing}"
        )
    return ds.select_columns(KEPT_COLUMNS)


def _split_kept_dropped(mapped: Dataset) -> tuple[Dataset, Dataset]:
    kept = mapped.filter(lambda ex: ex.get("_keep") is True)
    dropped = mapped.filter(lambda ex: ex.get("_keep") is not True)
    return kept, dropped


def _drop_reason_counts(dropped: Dataset) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in dropped.select_columns(["_drop_reason"]):
        reason = r.get("_drop_reason") or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def run(
    *,
    adapter: DatasetAdapter,
    cfg: FilterConfig,
    input_dataset_dir: str | Path,
    output_dir: str | Path,
    split: str = "train",
    num_proc: int = 1,
    max_examples: Optional[int] = None,
    save_kept: bool = True,
    save_dropped: bool = True,
    write_summary_json: bool = True,
) -> Dict[str, Any]:
    """Run the generic Stage 2A pipeline against a single source dataset.

    Parameters
    ----------
    adapter
        A concrete :class:`DatasetAdapter` for the source dataset.
    cfg
        Pipeline configuration (thresholds, toggles, model paths).
    input_dataset_dir
        Path passed to :func:`datasets.load_from_disk`.
    output_dir
        Directory created for the run. Receives:

        - ``kept.parquet``        — kept rows in canonical column order
        - ``dropped.parquet``     — dropped rows (audit)
        - ``summary.json``        — counts (kept / dropped / by reason)
    split
        Logical split name used inside ``example_id`` derivation.
    num_proc
        Forwarded to ``ds.map`` / ``ds.filter`` for parallelism.
    max_examples
        If set, only the first N rows are processed (useful for smoke tests).

    Returns
    -------
    dict
        ``{"n_in": int, "n_kept": int, "n_dropped": int, "by_reason": {...}}``
    """
    cfg.validate()

    input_dataset_dir = Path(input_dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(input_dataset_dir))
    if max_examples is not None and max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    n_in = len(ds)

    def _map_fn(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return map_row(ex=ex, idx=idx, adapter=adapter, cfg=cfg, split=split)

    mapped = ds.map(
        _map_fn,
        with_indices=True,
        remove_columns=ds.column_names,
        num_proc=num_proc if num_proc > 1 else None,
        load_from_cache_file=False,
        desc=f"Stage 2A | {adapter.dataset_id}",
    )
    mapped = _ensure_canonical_columns(mapped)

    kept, dropped = _split_kept_dropped(mapped)
    n_kept = len(kept)
    n_dropped = len(dropped)
    by_reason = _drop_reason_counts(dropped)

    if save_kept:
        kept.to_parquet(str(output_dir / "kept.parquet"))
    if save_dropped:
        dropped.to_parquet(str(output_dir / "dropped.parquet"))

    summary = {
        "dataset_id": adapter.dataset_id,
        "split": split,
        "n_in": n_in,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "by_reason": by_reason,
        "config": {
            k: v for k, v in cfg.__dict__.items() if not k.startswith("_")
        },
    }
    if write_summary_json:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
