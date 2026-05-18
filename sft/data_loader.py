"""Dataset downloader for SFT-Collection-v2.

Downloads the 15 source datasets used to build SFT-Collection-v2
(ahmad21omar/SFT-Collection-v2 on HuggingFace) and saves them locally
using HuggingFace Arrow format (save_to_disk).

Default output layout:
  <output_dir>/<dataset_id>/
    <split>/   ← Arrow dataset, compatible with datasets.load_from_disk()

The default output_dir is either:
  - the env variable SFT_DATA_ROOT (if set), or
  - <repo_root>/datasets/sft/  (repo_root = parent of Filtering_Pipeline/)

Usage
-----
  # smoke-test: stream only the first 1000 rows
  python data_loader.py --dataset openthoughts3 --max-examples 1000 --output-dir /tmp/sft

  # full download
  python data_loader.py --dataset nemotron_v2

Or import and call directly:
  from Filtering_Pipeline.data_loader import download_openthoughts3
  path = download_openthoughts3(max_examples=1000, output_dir=Path("/tmp/sft"))

Every download function accepts:
  max_examples : int | None   stream only first N rows (smoke-test mode)
  output_dir   : Path | None  override the default download root
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, Features, Value, load_dataset, Sequence


# ---------------------------------------------------------------------------
# Default output root (override via env var or per-function parameter)
# ---------------------------------------------------------------------------

def _default_sft_root() -> Path:
    env = os.environ.get("SFT_DATA_ROOT")
    if env:
        return Path(env)
    # Filtering_Pipeline/data_loader.py → parents[0] = Filtering_Pipeline/
    #                                    → parents[1] = Master_Thesis/  (repo root)
    return Path(__file__).resolve().parents[1] / "datasets" / "sft"


SFT_ROOT: Path = _default_sft_root()


def dataset_id_to_folder(dataset_id: str) -> str:
    """Convert a HuggingFace dataset ID to a filesystem-safe folder name."""
    return dataset_id.replace("/", "__")


def _resolve_root(output_dir: Optional[Path | str]) -> Path:
    """Return the effective download root (explicit arg > env var > default)."""
    if output_dir is not None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return SFT_ROOT


# ---------------------------------------------------------------------------
# Generic helpers used by the per-dataset functions below
# ---------------------------------------------------------------------------

def _download_single_split(
    dataset_id: str,
    split: str,
    *,
    revision: Optional[str],
    max_examples: Optional[int],
    out_dir: Path,
    hf_kwargs: Optional[dict] = None,
) -> Path:
    """Download one split and save it under out_dir/<split>/."""
    hf_kwargs = hf_kwargs or {}
    save_path = out_dir / (split if max_examples is None else f"{split}__first_{max_examples}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if max_examples is not None:
        ds_stream = load_dataset(
            dataset_id, revision=revision, split=split, streaming=True, **hf_kwargs
        )
        from datasets import Dataset
        rows = list(ds_stream.take(max_examples))  # type: ignore[attr-defined]
        materialized = Dataset.from_list(rows, features=hf_kwargs.get("features"))
        materialized.save_to_disk(str(save_path))
    else:
        ds = load_dataset(dataset_id, revision=revision, split=split, **hf_kwargs)
        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))  # type: ignore[union-attr]

    return save_path


def _download_configs(
    dataset_id: str,
    configs: list[str],
    split: str,
    *,
    revision: Optional[str],
    max_examples: Optional[int],
    out_dir: Path,
    features: Optional[Features] = None,
) -> list[Path]:
    """Download multiple HF dataset configs into subfolders of out_dir."""
    saved: list[Path] = []
    for cfg in configs:
        if max_examples is not None:
            ds_stream = load_dataset(
                dataset_id, cfg, revision=revision, split=split,
                streaming=True, **({"features": features} if features else {}),
            )
            from datasets import Dataset
            rows = list(ds_stream.take(max_examples))
            materialized = Dataset.from_list(rows)
            save_path = out_dir / cfg / f"{split}__first_{max_examples}"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            materialized.save_to_disk(str(save_path))
        else:
            ds: DatasetDict = load_dataset(
                dataset_id, cfg, revision=revision,
                **({"features": features} if features else {}),
            )  # type: ignore[assignment]
            if split not in ds:
                raise ValueError(f"Split '{split}' not found for config '{cfg}'. Available: {list(ds.keys())}")
            save_path = out_dir / cfg / split
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ds[split].save_to_disk(str(save_path))
        saved.append(save_path)
    return saved


# ---------------------------------------------------------------------------
# Per-dataset download functions
# ---------------------------------------------------------------------------

def download_openthoughts3(
    dataset_id: str = "open-thoughts/OpenThoughts3-1.2M",
    *,
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Download open-thoughts/OpenThoughts3-1.2M (~1.2M rows, EN, Math/Code/Science)."""
    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = _download_single_split(dataset_id, split, revision=revision, max_examples=max_examples, out_dir=out_dir)
    print(f"[OpenThoughts3] saved to {save_path}")
    return save_path


def download_dolci_think_sft_7b(
    dataset_id: str = "allenai/Dolci-Think-SFT-7B",
    *,
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Download allenai/Dolci-Think-SFT-7B (~102k rows, EN, mixed domains)."""
    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = _download_single_split(dataset_id, split, revision=revision, max_examples=max_examples, out_dir=out_dir)
    print(f"[Dolci-Think-SFT-7B] saved to {save_path}")
    return save_path


def download_synthetic_2_sft_verified(
    dataset_id: str = "PrimeIntellect/SYNTHETIC-2-SFT-verified",
    *,
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Download PrimeIntellect/SYNTHETIC-2-SFT-verified."""
    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = _download_single_split(dataset_id, split, revision=revision, max_examples=max_examples, out_dir=out_dir)
    print(f"[SYNTHETIC-2-SFT-verified] saved to {save_path}")
    return save_path


def download_mixture_of_thoughts(
    dataset_id: str = "open-r1/Mixture-of-Thoughts",
    *,
    configs: Optional[list[str]] = None,
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> list[Path]:
    """Download open-r1/Mixture-of-Thoughts (configs: math / code / science)."""
    if configs is None:
        configs = ["math", "code", "science"]
    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _download_configs(dataset_id, configs, split, revision=revision, max_examples=max_examples, out_dir=out_dir)
    for p in paths:
        print(f"[Mixture-of-Thoughts] saved to {p}")
    return paths


def download_nemotron_math_proofs_v1(
    dataset_id: str = "nvidia/Nemotron-Math-Proofs-v1",
    *,
    split: str = "lean",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Download nvidia/Nemotron-Math-Proofs-v1 (Lean proof split)."""
    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = _download_single_split(dataset_id, split, revision=revision, max_examples=max_examples, out_dir=out_dir)
    print(f"[Nemotron-Math-Proofs-v1] saved to {save_path}")
    return save_path


def download_nemotron_competitive_programming_v1(
    dataset_id: str = "nvidia/Nemotron-Competitive-Programming-v1",
    *,
    splits: Optional[list[str]] = None,
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> list[Path]:
    """Download nvidia/Nemotron-Competitive-Programming-v1 (C++/Python competitive coding).

    Requires an explicit Features schema — HF fails to infer the nullable fields otherwise.
    """
    if splits is None:
        splits = [
            "competitive_coding_cpp_part00",
            "competitive_coding_cpp_part01",
            "competitive_coding_python_part00",
            "competitive_coding_python_part01",
            "infinibyte_part00",
            "infinibyte_part01",
        ]

    cp_features = Features({
        "uuid": Value("string"),
        "messages": [{"role": Value("string"), "content": Value("string"), "reasoning_content": Value("string")}],
        "license": Value("string"),
        "used_in": [Value("string")],
        "tools": Sequence(Value("null")),
        "dataset": Value("string"),
        "split": Value("string"),
        "index": Value("string"),
        "source": Value("string"),
        "difficulty": Value("string"),
        "question_id": Value("string"),
    })

    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for sp in splits:
        print(f"[Nemotron-Competitive-Programming-v1] Downloading split={sp!r} ...")
        save_path = _download_single_split(
            dataset_id, sp, revision=revision, max_examples=max_examples,
            out_dir=out_dir, hf_kwargs={"features": cp_features},
        )
        print(f"  -> {save_path}")
        saved.append(save_path)
    return saved


def download_nemotron_math_v2(
    dataset_id: str = "nvidia/Nemotron-Math-v2",
    *,
    splits: Optional[list[str]] = None,
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> list[Path]:
    """Download nvidia/Nemotron-Math-v2 (difficulty splits: low / medium / high_part00-02)."""
    if splits is None:
        splits = ["low", "medium", "high_part00", "high_part01", "high_part02"]

    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id) / "SFT"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for sp in splits:
        print(f"[Nemotron-Math-v2] Downloading split={sp!r} ...")
        save_path = _download_single_split(dataset_id, sp, revision=revision, max_examples=max_examples, out_dir=out_dir)
        print(f"  -> {save_path}")
        saved.append(save_path)
    return saved


def download_am_deepseek_r1_distilled_1p4m(
    dataset_id: str = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
    *,
    configs: Optional[list[str]] = None,
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> list[Path]:
    """Download a-m-team/AM-DeepSeek-R1-Distilled-1.4M (configs: am_0.5M / am_0.9M).

    Requires an explicit Features schema — HF's JSON config loader fails to infer
    the nested `info` dict without it.
    """
    if configs is None:
        configs = ["am_0.5M", "am_0.9M", "am_0.9M_sample_1k"]

    am_features = Features({
        "messages": [{
            "role": Value("string"),
            "content": Value("string"),
            "info": {
                "source": Value("string"),
                "reference_answer": Value("string"),
                "test_case": Value("string"),
                "think_content": Value("string"),
                "answer_content": Value("string"),
            },
        }]
    })

    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _download_configs(
        dataset_id, configs, split, revision=revision, max_examples=max_examples,
        out_dir=out_dir, features=am_features,
    )
    for p in paths:
        print(f"[AM-DeepSeek-R1-Distilled-1.4M] saved to {p}")
    return paths


def download_llama_nemotron_post_training(
    dataset_id: str = "nvidia/Llama-Nemotron-Post-Training-Dataset",
    *,
    subset: str = "SFT",
    splits: Optional[list[str]] = None,
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> list[Path]:
    """Download nvidia/Llama-Nemotron-Post-Training-Dataset (SFT subset).

    SFT splits: code (~657k), math (~22M), science (~709k), chat (~40k), safety (~31k).
    HF load pattern: load_dataset(..., "SFT", split="code")
    """
    if splits is None:
        splits = ["code", "math", "science", "chat", "safety"]

    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id) / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for sp in splits:
        print(f"[Llama-Nemotron-Post-Training] Downloading subset={subset!r} split={sp!r} ...")
        if max_examples is not None:
            ds_stream = load_dataset(dataset_id, subset, revision=revision, split=sp, streaming=True)
            from datasets import Dataset
            rows = list(ds_stream.take(max_examples))  # type: ignore[attr-defined]
            materialized = Dataset.from_list(rows)
            save_path = out_dir / sp / f"first_{max_examples}"
            save_path.mkdir(parents=True, exist_ok=True)
            materialized.save_to_disk(str(save_path))
        else:
            ds = load_dataset(dataset_id, subset, revision=revision, split=sp)
            save_path = out_dir / sp
            save_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(save_path))  # type: ignore[union-attr]
        print(f"  -> {save_path}")
        saved.append(save_path)
    return saved


def download_nemotron_post_training_v2(
    dataset_id: str = "nvidia/Nemotron-Post-Training-Dataset-v2",
    *,
    splits: Optional[list[str]] = None,
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> list[Path]:
    """Download nvidia/Nemotron-Post-Training-Dataset-v2.

    Splits: math, code, stem, chat, multilingual_ja/de/it/es/fr.
    HF load pattern: load_dataset(..., "default", split="math")
    """
    if splits is None:
        splits = [
            "math", "code", "stem", "chat",
            "multilingual_ja", "multilingual_de", "multilingual_it",
            "multilingual_es", "multilingual_fr",
        ]

    root = _resolve_root(output_dir)
    out_dir = root / dataset_id_to_folder(dataset_id) / "SFT"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for sp in splits:
        print(f"[Nemotron-Post-Training-v2] Downloading split={sp!r} ...")
        if max_examples is not None:
            ds_stream = load_dataset(dataset_id, "default", revision=revision, split=sp, streaming=True)
            from datasets import Dataset
            rows = list(ds_stream.take(max_examples))  # type: ignore[attr-defined]
            materialized = Dataset.from_list(rows)
            save_path = out_dir / sp / f"first_{max_examples}"
            save_path.mkdir(parents=True, exist_ok=True)
            materialized.save_to_disk(str(save_path))
        else:
            ds = load_dataset(dataset_id, "default", revision=revision, split=sp)
            save_path = out_dir / sp
            save_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(save_path))  # type: ignore[union-attr]
        print(f"  -> {save_path}")
        saved.append(save_path)
    return saved


def download_soofi_think_sft_10b_multilingual(
    dataset_id: str = "toroe/Soofi-Think-SFT-10B-multilingual",
    *,
    languages: Optional[list[str]] = None,
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
) -> dict[str, Path]:
    """Download toroe/Soofi-Think-SFT-10B-multilingual.

    Available language splits: english, italian, french, spanish, german.
    """
    if languages is None:
        languages = ["italian", "french", "spanish", "german"]

    root = _resolve_root(output_dir)
    out_root = root / "translated_ds" / dataset_id_to_folder(dataset_id)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for lang in languages:
        print(f"[Soofi-Think-SFT-10B-multilingual] Downloading language: {lang} ...")
        save_path = _download_single_split(
            dataset_id, lang, revision=revision, max_examples=max_examples, out_dir=out_root
        )
        print(f"  -> {save_path}")
        results[lang] = save_path
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    DATASETS = {
        "openthoughts3": download_openthoughts3,
        "dolci": download_dolci_think_sft_7b,
        "synthetic2": download_synthetic_2_sft_verified,
        "mixture_of_thoughts": download_mixture_of_thoughts,
        "nemotron_math_proofs": download_nemotron_math_proofs_v1,
        "nemotron_competitive": download_nemotron_competitive_programming_v1,
        "nemotron_math_v2": download_nemotron_math_v2,
        "am_deepseek": download_am_deepseek_r1_distilled_1p4m,
        "llama_nemotron": download_llama_nemotron_post_training,
        "nemotron_v2": download_nemotron_post_training_v2,
        "soofi_multilingual": download_soofi_think_sft_10b_multilingual,
    }

    parser = argparse.ArgumentParser(description="Download SFT source datasets.")
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASETS.keys()),
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save the dataset. Defaults to SFT_DATA_ROOT env var or <repo>/datasets/sft/.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Stream only the first N rows (smoke-test mode).",
    )
    args = parser.parse_args()

    fn = DATASETS[args.dataset]
    result = fn(max_examples=args.max_examples, output_dir=args.output_dir)  # type: ignore[call-arg]
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
