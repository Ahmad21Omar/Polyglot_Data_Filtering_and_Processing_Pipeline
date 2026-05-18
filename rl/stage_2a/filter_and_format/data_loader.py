"""Minimal dataset downloader for RL thesis data curation.

Saves RL datasets under:
  Master_Thesis/Datasets/rl/<dataset_id>/

Example:
  allenai/Dolci-Think-RL-7B -> Master_Thesis/Datasets/rl/allenai__Dolci-Think-RL-7B/

We use Hugging Face `datasets` and persist with `save_to_disk()`.

RL datasets in scope (from rl_subsources_before_1a.json):
  allenai/Dolci-Think-RL-7B          102,014  ODC-BY
  allenai/Dolci-Instruct-RL          169,964  ODC-BY
  a-m-team/AM-Thinking-v1-RL-Dataset  54,765  Apache-2.0
  nvidia/Llama-Nemotron-Post-Training-Dataset (RL split "instruction_following") 56,339
  PrimeIntellect/SYNTHETIC-2-RL      155,638  Apache-2.0
  MiniMaxAI/SynLogic                  48,677  MIT
  TIGER-Lab/WebInstruct-verified     228,736  Apache-2.0
  AIML-TUDA/SLR-Bench                 19,253  CC-BY-4.0
  CLUTRR/v1                           70,631  MIT
  logicreasoning/logi_glue           616,762  Apache-2.0

Stage 1b redundant (subsamples of Dolci-Think-RL-7B — NOT downloaded):
  allenai/Dolci-RL-Zero-Code-7B      (13,312)
  allenai/Dolci-RL-Zero-IF-7B        (13,179)
  allenai/Dolci-RL-Zero-Math-7B      (13,314)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, Features, IterableDataset, Value, Sequence, load_dataset


ROOT = Path(__file__).resolve().parents[2]   # Master_Thesis/ (go up 2 levels from rl/)
RL_ROOT = ROOT / "Datasets" / "rl"


def dataset_id_to_folder(dataset_id: str) -> str:
    """Create a filesystem-safe folder name while keeping the dataset_id readable."""
    return dataset_id.replace("/", "__")


@dataclass(frozen=True)
class DownloadSpec:
    dataset_id: str
    split: str = "train"
    revision: Optional[str] = None
    # If set, we stream and only materialise the first N examples (smoke test).
    max_examples: Optional[int] = None


# ── Generic helpers (mirrors sft/data_loader.py) ─────────────────────────────

def download_and_save_rl(spec: DownloadSpec) -> Path:
    """Download (or stream) a single-split RL dataset and save it as-is.

    Always uses streaming + Dataset.from_list() to avoid schema casting issues.
    Returns the output folder path.
    """
    out_dir = RL_ROOT / dataset_id_to_folder(spec.dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset

    # Always stream to avoid schema validation during loading
    try:
        ids: IterableDataset = load_dataset(  # type: ignore
            spec.dataset_id,
            revision=spec.revision,
            split=spec.split,
            streaming=True,
        )
    except Exception as e:
        print(f"  WARNING: Failed to load with streaming: {e}")
        print(f"  Retrying with trust_remote_code=True...")
        ids: IterableDataset = load_dataset(  # type: ignore
            spec.dataset_id,
            revision=spec.revision,
            split=spec.split,
            streaming=True,
            trust_remote_code=True,
        )

    if spec.max_examples is None:
        print(f"  Streaming all rows...")
        rows = list(ids)  # type: ignore
    else:
        print(f"  Streaming first {spec.max_examples} rows...")
        rows = list(ids.take(spec.max_examples))  # type: ignore

    # Convert to regular Dataset and save
    materialized = Dataset.from_list(rows)
    if spec.max_examples is None:
        save_path = out_dir / spec.split
    else:
        save_path = out_dir / f"{spec.split}__first_{spec.max_examples}"
    
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"  Saving {len(materialized)} rows with {len(materialized.column_names)} fields...")
    materialized.save_to_disk(str(save_path))
    return save_path


def download_and_save_rl_configs(
    dataset_id: str,
    configs: list[str],
    *,
    split: str = "train",
    revision: Optional[str] = None,
    features: Optional[Features] = None,
    max_examples: Optional[int] = None,
) -> list[Path]:
    """Download multiple HF *configs* of an RL dataset into subfolders.
    
    Always uses streaming to avoid schema validation issues.
    Falls back to 'test' or 'validation' if 'train' split is not available.
    """
    from datasets import Dataset

    out_dir = RL_ROOT / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for cfg in configs:
        print(f"  [{dataset_id}] config={cfg!r} split={split!r} ...")
        
        # Try to load, with fallback splits if needed
        fallback_splits = [split, "test", "validation", "train"] if split != "train" else ["train", "test", "validation"]
        ids = None
        actual_split = split
        
        for try_split in fallback_splits:
            try:
                ids = load_dataset(
                    dataset_id, cfg, revision=revision,
                    split=try_split, streaming=True,
                )
                actual_split = try_split
                if try_split != split:
                    print(f"    (using fallback split '{try_split}' instead of '{split}')")
                break
            except Exception as e:
                continue
        
        if ids is None:
            # Last resort: try with trust_remote_code
            try:
                ids = load_dataset(
                    dataset_id, cfg, revision=revision,
                    split=split, streaming=True,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"    ERROR: Could not load config '{cfg}': {type(e).__name__}")
                continue
        
        if max_examples is not None:
            rows = list(ids.take(max_examples))  # type: ignore
        else:
            rows = list(ids)  # type: ignore

        materialized = Dataset.from_list(rows)
        save_path = out_dir / cfg / actual_split
        if max_examples is not None:
            save_path = out_dir / cfg / f"{actual_split}__first_{max_examples}"
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    -> {len(materialized)} rows with {len(materialized.column_names)} fields")
        materialized.save_to_disk(str(save_path))
        saved.append(save_path)
    return saved


def download_and_save_rl_splits(
    dataset_id: str,
    splits: list[str],
    *,
    revision: Optional[str] = None,
    features: Optional[Features] = None,
    max_examples: Optional[int] = None,
) -> list[Path]:
    """Download multiple *splits* of an RL dataset into subfolders.
    
    Always uses streaming to avoid schema validation issues.
    """
    from datasets import Dataset

    out_dir = RL_ROOT / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for sp in splits:
        print(f"  [{dataset_id}] split={sp!r} ...")
        
        try:
            # Always stream to avoid schema casting
            ids = load_dataset(
                dataset_id, revision=revision, split=sp, streaming=True,
            )
        except Exception as e:
            print(f"    WARNING: Failed to load: {e}")
            print(f"    Retrying with trust_remote_code=True...")
            ids = load_dataset(
                dataset_id, revision=revision, split=sp, streaming=True,
                trust_remote_code=True,
            )
        
        if max_examples is not None:
            rows = list(ids.take(max_examples))  # type: ignore
        else:
            rows = list(ids)  # type: ignore

        materialized = Dataset.from_list(rows)
        save_path = out_dir / sp
        if max_examples is not None:
            save_path = out_dir / sp / f"first_{max_examples}"

        save_path.mkdir(parents=True, exist_ok=True)
        print(f"    -> {len(materialized)} rows with {len(materialized.column_names)} fields")
        materialized.save_to_disk(str(save_path))
        saved.append(save_path)
    return saved


# ── Dataset-specific helpers ─────────────────────────────────────────────────

def download_dolci_think_rl_7b(
    *,
    dataset_id: str = "allenai/Dolci-Think-RL-7B",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download allenai/Dolci-Think-RL-7B (102,014 rows, ODC-BY).

    Single train split; no configs.  Fields:
      prompt, ground_truth (list), dataset (list), dataset_source,
      original_dataset, custom_id, passrate, total_rollouts,
      total_correct_rollouts, constraint, constraint_type.

    Saved to:
      Master_Thesis/Datasets/rl/allenai__Dolci-Think-RL-7B/train/
    """
    print(f"[Dolci-Think-RL-7B] Downloading split={split!r} ...")
    path = download_and_save_rl(
        DownloadSpec(
            dataset_id=dataset_id,
            split=split,
            revision=revision,
            max_examples=max_examples,
        )
    )
    print(f"  -> {path}")
    return path


def download_dolci_instruct_rl(
    *,
    dataset_id: str = "allenai/Dolci-Instruct-RL",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download allenai/Dolci-Instruct-RL (169,964 rows, ODC-BY).

    OLMo-3 Instruct RL mixture.  Same schema as Dolci-Think-RL-7B.

    Saved to:
      Master_Thesis/Datasets/rl/allenai__Dolci-Instruct-RL/train/
    """
    print(f"[Dolci-Instruct-RL] Downloading split={split!r} ...")
    path = download_and_save_rl(
        DownloadSpec(
            dataset_id=dataset_id,
            split=split,
            revision=revision,
            max_examples=max_examples,
        )
    )
    print(f"  -> {path}")
    return path


def download_am_thinking_v1_rl(
    *,
    dataset_id: str = "a-m-team/AM-Thinking-v1-RL-Dataset",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download a-m-team/AM-Thinking-v1-RL-Dataset (54,765 rows, Apache-2.0).

    verl-format RL dataset with reward_model payloads (ground_truth / testcases).
    Fields: messages (list of role/content), data_source, reward_model (dict).

    Note: HF may require a `trust_remote_code=True` flag depending on version.
    If loading fails, pass revision= a known commit hash.

    Saved to:
      Master_Thesis/Datasets/rl/a-m-team__AM-Thinking-v1-RL-Dataset/train/
    """
    print(f"[AM-Thinking-v1-RL] Downloading split={split!r} ...")
    path = download_and_save_rl(
        DownloadSpec(
            dataset_id=dataset_id,
            split=split,
            revision=revision,
            max_examples=max_examples,
        )
    )
    print(f"  -> {path}")
    return path


def download_llama_nemotron_post_training_rl(
    *,
    dataset_id: str = "nvidia/Llama-Nemotron-Post-Training-Dataset",
    splits: Optional[list[str]] = None,
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> list[Path]:
    """Download the RL portion of nvidia/Llama-Nemotron-Post-Training-Dataset.

    The RL portion exposes the split `instruction_following` (56,339 rows).
    The SFT portion (code/math/science/chat/safety) is handled in
    sft/data_loader.py.

    Always uses streaming to avoid schema casting issues.

    Saved to:
      Master_Thesis/Datasets/rl/nvidia__Llama-Nemotron-Post-Training-Dataset/RL/<split>/
    """
    from datasets import Dataset

    if splits is None:
        splits = ["instruction_following"]

    out_dir = RL_ROOT / dataset_id_to_folder(dataset_id) / "RL"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for sp in splits:
        print(f"[Llama-Nemotron-RL] Downloading subset=RL split={sp!r} ...")
        
        try:
            # Always stream to avoid schema validation
            ids = load_dataset(
                dataset_id, "RL", revision=revision, split=sp, streaming=True,
            )
        except Exception as e:
            print(f"  WARNING: Failed to load: {e}")
            print(f"  Retrying with trust_remote_code=True...")
            ids = load_dataset(
                dataset_id, "RL", revision=revision, split=sp, streaming=True,
                trust_remote_code=True,
            )
        
        if max_examples is not None:
            print(f"  Streaming first {max_examples} rows...")
            rows = list(ids.take(max_examples))  # type: ignore
        else:
            print(f"  Streaming all rows...")
            rows = list(ids)  # type: ignore

        # Convert to regular Dataset and save
        materialized = Dataset.from_list(rows)
        save_path = out_dir / sp
        if max_examples is not None:
            save_path = out_dir / sp / f"first_{max_examples}"

        save_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving {len(materialized)} rows with {len(materialized.column_names)} fields...")
        materialized.save_to_disk(str(save_path))
        print(f"  -> {save_path}")
        saved.append(save_path)
    return saved


def download_nemotron_3_nano_rl(
    *,
    dataset_id: str = "nvidia/Nemotron-3-Nano-RL-Training-Blend",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download nvidia/Nemotron-3-Nano-RL-Training-Blend (~93k rows, Apache-2.0).

    NeMo-Gym RL training mixture used for Nemotron-3-Nano post-training.
    Verifier types include: if_rules, multiple_choice, code_asserts,
    math_with_judge, schema_structured_outputs.
    verifier_source = "nemo_gym".

    Saved to:
      Master_Thesis/Datasets/rl/nvidia__Nemotron-3-Nano-RL-Training-Blend/train/
    """
    print(f"[Nemotron-3-Nano-RL] Downloading split={split!r} ...")
    path = download_and_save_rl(
        DownloadSpec(
            dataset_id=dataset_id,
            split=split,
            revision=revision,
            max_examples=max_examples,
        )
    )
    print(f"  -> {path}")
    return path


def download_nemotron_rl_reasoning_gym_v1(
    *,
    dataset_id: str = "nvidia/Nemotron-RL-ReasoningGym-v1",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download nvidia/Nemotron-RL-ReasoningGym-v1 (15,000 rows, CC-BY-4.0).

    Procedurally generated RL data spanning 104 reasoning-gym environments
    across 12 categories (algebra, arithmetic, computation, cognition,
    geometry, graph theory, logic, games, etc.). Single config "default",
    single split "train" backed by `data/train.jsonl` (~50 MB on disk).

    NOTE: bypasses HF `datasets` streaming because some rows contain huge
    integers (graph-theory tasks) that ujson refuses to parse. We download
    the raw `data/train.jsonl` via `hf_hub_download` and parse with stdlib
    `json` line-by-line.

    Saved to:
      Master_Thesis/Datasets/rl/nvidia__Nemotron-RL-ReasoningGym-v1/train/
    """
    import json as _json
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    print(f"[Nemotron-RL-ReasoningGym-v1] Downloading raw train.jsonl ...")
    raw_path = hf_hub_download(
        repo_id=dataset_id,
        filename="data/train.jsonl",
        repo_type="dataset",
        revision=revision,
    )

    # Nested dict columns vary across the 104 reasoning-gym task families
    # (heterogeneous metadata sub-schemas). Stringify them so PyArrow can
    # build a single uniform table; downstream filter_and_format step parses
    # them back with json.loads().
    _STRINGIFY_KEYS = ("responses_create_params", "metadata", "agent_ref")

    rows: list[dict] = []
    with open(raw_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            for k in _STRINGIFY_KEYS:
                if k in obj and not isinstance(obj[k], str):
                    obj[k] = _json.dumps(obj[k], ensure_ascii=False)
            rows.append(obj)
            if max_examples is not None and len(rows) >= max_examples:
                break
    print(f"  Parsed {len(rows)} rows from {raw_path}")

    out_dir = RL_ROOT / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = (
        out_dir / split if max_examples is None
        else out_dir / f"{split}__first_{max_examples}"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    materialized = Dataset.from_list(rows)
    print(f"  Saving {len(materialized)} rows with {len(materialized.column_names)} fields ({materialized.column_names}) ...")
    materialized.save_to_disk(str(save_path))
    print(f"  -> {save_path}")
    return save_path


def download_synthetic2_rl(
    *,
    dataset_id: str = "PrimeIntellect/SYNTHETIC-2-RL",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download PrimeIntellect/SYNTHETIC-2-RL (155,638 rows, Apache-2.0).

    Fields include: problem_id (prefix encodes task family), problem, solution,
    verifier_type (execution / rule-based).

    Saved to:
      Master_Thesis/Datasets/rl/PrimeIntellect__SYNTHETIC-2-RL/train/
    """
    print(f"[SYNTHETIC-2-RL] Downloading split={split!r} ...")
    path = download_and_save_rl(
        DownloadSpec(
            dataset_id=dataset_id,
            split=split,
            revision=revision,
            max_examples=max_examples,
        )
    )
    print(f"  -> {path}")
    return path


def download_synlogic(
    *,
    dataset_id: str = "MiniMaxAI/SynLogic",
    configs: Optional[list[str]] = None,
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> list[Path]:
    """Download MiniMaxAI/SynLogic (48,677 rows total, MIT).

    Two HF configs: `easy` (27 tasks) and `hard` (35 tasks).
    Fields: data_source (task name), problem, solution, difficulty.

    Saved to:
      Master_Thesis/Datasets/rl/MiniMaxAI__SynLogic/easy/train/
      Master_Thesis/Datasets/rl/MiniMaxAI__SynLogic/hard/train/
    """
    if configs is None:
        configs = ["easy", "hard"]
    print(f"[SynLogic] Downloading configs={configs} ...")
    return download_and_save_rl_configs(
        dataset_id,
        configs=configs,
        split=split,
        revision=revision,
        max_examples=max_examples,
    )


def download_general_reasoner(
    *,
    dataset_id: str = "TIGER-Lab/WebInstruct-verified",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download TIGER-Lab/WebInstruct-verified (228,736 rows, Apache-2.0).

    WebInstruct-verified QA pairs covering diverse domains.
    Fields: question, answer, source (web URL).

    Saved to:
      Master_Thesis/Datasets/rl/TIGER-Lab__WebInstruct-verified/train/
    """
    print(f"[General-Reasoner] Downloading split={split!r} ...")
    path = download_and_save_rl(
        DownloadSpec(
            dataset_id=dataset_id,
            split=split,
            revision=revision,
            max_examples=max_examples,
        )
    )
    print(f"  -> {path}")
    return path


def download_slr_bench(
    *,
    dataset_id: str = "AIML-TUDA/SLR-Bench",
    config: str = "v1-All",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download AIML-TUDA/SLR-Bench (19,253 rows, CC-BY-4.0).

    Syntactically generated spatial/logical reasoning benchmark.
    Available configs: v1-All, v1-Basic, v1-Easy, v1-Hard, v1-Medium.
    Default: v1-All (all samples).

    Saved to:
      Master_Thesis/Datasets/rl/AIML-TUDA__SLR-Bench/v1-All/train/
    """
    print(f"[SLR-Bench] Downloading config={config!r} split={split!r} ...")
    return download_and_save_rl_configs(
        dataset_id,
        configs=[config],
        split=split,
        revision=revision,
        max_examples=max_examples,
    )[0]


def download_clutrr(
    *,
    dataset_id: str = "CLUTRR/v1",
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Path:
    """Download CLUTRR/v1 (70,631 rows, MIT).

    Compositional Language Understanding and Text-based Relational Reasoning.
    Fields: story, query, target, relation_list, etc.

    Note: CLUTRR has known HF metadata issues that may cause loading failures.
    Falls back gracefully if the dataset cannot be loaded.

    Saved to:
      Master_Thesis/Datasets/rl/CLUTRR__v1/train/
    """
    from datasets import Dataset
    
    print(f"[CLUTRR/v1] Downloading split={split!r} ...")
    
    out_dir = RL_ROOT / dataset_id_to_folder(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / split
    
    try:
        # Try to load with trust_remote_code
        ids = load_dataset(
            dataset_id,
            revision=revision,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        if max_examples is None:
            print(f"  Streaming all rows...")
            rows = list(ids)
        else:
            print(f"  Streaming first {max_examples} rows...")
            rows = list(ids.take(max_examples))
        
        materialized = Dataset.from_list(rows)
        if max_examples is not None:
            save_path = out_dir / f"{split}__first_{max_examples}"
        
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving {len(materialized)} rows with {len(materialized.column_names)} fields...")
        materialized.save_to_disk(str(save_path))
    except Exception as e:
        print(f"  ERROR: Failed to download CLUTRR: {type(e).__name__}: {str(e)[:100]}")
        print(f"  CLUTRR will be skipped - it has known HF metadata issues")
        return None  # type: ignore
    
    print(f"  -> {save_path}")
    return save_path


def download_logi_glue(
    *,
    dataset_id: str = "logicreasoning/logi_glue",
    configs: Optional[list[str]] = None,
    split: str = "train",
    revision: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> list[Path]:
    """Download logicreasoning/logi_glue (616,762 rows total, Apache-2.0).

    Suite of 24 logic-reasoning HF configs covering various logical reasoning tasks.
    Available configs (actual names from HF):
      logiQA, cluttr, abduction_animal, adv, alpha_nli, logicNLI, folio, proofwriter,
      rulebert, anli, logiQA_2.0, cluttr_systematic, bigbench-logical-Args, natlang,
      babi_task_16, wanli, abduction_person, prontoqa, babi_task_15, winologic,
      birdelectricity, bigbench_deduction, reclor, Rulebert-Union-Rules.

    Default: download all 24 known configs. Pass `configs=["anli", "folio"]`
    to download a subset only.

    Saved to:
      Master_Thesis/Datasets/rl/logicreasoning__logi_glue/<config>/train/
    """
    if configs is None:
        configs = [
            "logiQA",
            "cluttr",
            "abduction_animal",
            "adv",
            "alpha_nli",
            "logicNLI",
            "folio",
            "proofwriter",
            "rulebert",
            "anli",
            "logiQA_2.0",
            "cluttr_systematic",
            "bigbench-logical-Args",
            "natlang",
            "babi_task_16",
            "wanli",
            "abduction_person",
            "prontoqa",
            "babi_task_15",
            "winologic",
            "birdelectricity",
            "bigbench_deduction",
            "reclor",
            "Rulebert-Union-Rules",
        ]
    print(f"[logi_glue] Downloading {len(configs)} configs ...")
    return download_and_save_rl_configs(
        dataset_id,
        configs=configs,
        split=split,
        revision=revision,
        max_examples=max_examples,
    )


# ── Convenience: download ALL RL datasets at once ────────────────────────────

def download_all_rl_datasets(
    *,
    max_examples: Optional[int] = None,
    skip: Optional[list[str]] = None,
) -> dict[str, list[Path] | Path]:
    """Download every RL dataset in scope.

    Args:
        max_examples: If set, only stream and save the first N rows per
                      dataset/split (smoke-test mode).
        skip:         List of dataset_ids to skip, e.g.
                      ["logicreasoning/logi_glue"] to avoid the 616k download.

    Returns:
        Dict mapping dataset_id -> saved Path(s).

    Stage 1b redundant datasets (Dolci-RL-Zero-*) are intentionally excluded.
    """
    skip = skip or []
    results: dict[str, list[Path] | Path] = {}

    def _run(key: str, fn, *args, **kwargs):
        if key in skip:
            print(f"[SKIP] {key}")
            return
        print(f"\n{'='*60}")
        print(f"Downloading: {key}")
        print(f"{'='*60}")
        results[key] = fn(*args, **kwargs)

    _run(
        "allenai/Dolci-Think-RL-7B",
        download_dolci_think_rl_7b,
        max_examples=max_examples,
    )
    _run(
        "allenai/Dolci-Instruct-RL",
        download_dolci_instruct_rl,
        max_examples=max_examples,
    )
    _run(
        "a-m-team/AM-Thinking-v1-RL-Dataset",
        download_am_thinking_v1_rl,
        max_examples=max_examples,
    )
    _run(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        download_llama_nemotron_post_training_rl,
        max_examples=max_examples,
    )
    _run(
        "nvidia/Nemotron-3-Nano-RL-Training-Blend",
        download_nemotron_3_nano_rl,
        max_examples=max_examples,
    )
    _run(
        "nvidia/Nemotron-RL-ReasoningGym-v1",
        download_nemotron_rl_reasoning_gym_v1,
        max_examples=max_examples,
    )
    _run(
        "PrimeIntellect/SYNTHETIC-2-RL",
        download_synthetic2_rl,
        max_examples=max_examples,
    )
    _run(
        "MiniMaxAI/SynLogic",
        download_synlogic,
        max_examples=max_examples,
    )
    _run(
        "TIGER-AI-Lab/General-Reasoner",
        download_general_reasoner,
        max_examples=max_examples,
    )
    _run(
        "AIML-TUDA/SLR-Bench",
        download_slr_bench,
        max_examples=max_examples,
    )
    _run(
        "CLUTRR/v1",
        download_clutrr,
        max_examples=max_examples,
    )
    _run(
        "logicreasoning/logi_glue",
        download_logi_glue,
        max_examples=max_examples,
    )

    print(f"\n{'='*60}")
    print("All RL datasets downloaded.")
    for ds_id, path in results.items():
        if isinstance(path, list):
            for p in path:
                print(f"  {ds_id}: {p}")
        else:
            print(f"  {ds_id}: {path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download RL datasets for the thesis data pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "dolci_think_rl",
            "dolci_instruct_rl",
            "am_thinking_v1_rl",
            "llama_nemotron_rl",
            "nemotron_3_nano_rl",
            "nemotron_rl_reasoning_gym_v1",
            "synthetic2_rl",
            "synlogic",
            "general_reasoner",
            "slr_bench",
            "clutrr",
            "logi_glue",
            "all",
        ],
        default="all",
        help="Which dataset(s) to download (default: all).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="If set, only stream the first N rows per dataset (smoke test).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="DATASET_ID",
        help="Dataset IDs to skip when using --dataset=all.",
    )
    args = parser.parse_args()

    n = args.max_examples

    if args.dataset == "all":
        download_all_rl_datasets(max_examples=n, skip=args.skip)
    elif args.dataset == "dolci_think_rl":
        download_dolci_think_rl_7b(max_examples=n)
    elif args.dataset == "dolci_instruct_rl":
        download_dolci_instruct_rl(max_examples=n)
    elif args.dataset == "am_thinking_v1_rl":
        download_am_thinking_v1_rl(max_examples=n)
    elif args.dataset == "llama_nemotron_rl":
        download_llama_nemotron_post_training_rl(max_examples=n)
    elif args.dataset == "nemotron_3_nano_rl":
        download_nemotron_3_nano_rl(max_examples=n)
    elif args.dataset == "nemotron_rl_reasoning_gym_v1":
        download_nemotron_rl_reasoning_gym_v1(max_examples=n)
    elif args.dataset == "synthetic2_rl":
        download_synthetic2_rl(max_examples=n)
    elif args.dataset == "synlogic":
        download_synlogic(max_examples=n)
    elif args.dataset == "general_reasoner":
        download_general_reasoner(max_examples=n)
    elif args.dataset == "slr_bench":
        download_slr_bench(max_examples=n)
    elif args.dataset == "clutrr":
        download_clutrr(max_examples=n)
    elif args.dataset == "logi_glue":
        download_logi_glue(max_examples=n)


if __name__ == "__main__":
    main()
