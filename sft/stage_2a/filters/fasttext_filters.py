from __future__ import annotations

"""FastText helper wrapper (OpenThoughts-style, but lightweight).

OpenThoughts uses FastText classifiers as a cheap learned filter stage.
Their OT3 engine provides Ray-based operators (train + inference). For our thesis
pipeline we also want a simple local API:

- train a binary classifier from positive/negative text lists
- score a list of texts

This module does *not* integrate with Ray/DAG; it is intended for small scripts
that pre-train a model and then run it offline.

Paper hyperparams (OpenThoughts Appendix R.2.1) we match by default:
- dim=256, epoch=3, lr=0.1, wordNgrams=2, minCount=3

Note: requires the `fasttext` Python package.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FastTextTrainConfig:
    dim: int = 256
    epoch: int = 3
    lr: float = 0.1
    word_ngrams: int = 2
    min_count: int = 3


def _require_fasttext():
    try:
        import fasttext  # type: ignore

        return fasttext
    except Exception as e:
        raise RuntimeError(
            "fasttext package is not installed. Install it to use FastText filters. "
            "(In OpenThoughts it's a core dependency.)"
        ) from e


def train_binary_fasttext(
    *,
    positives: Sequence[str],
    negatives: Sequence[str],
    save_path: str,
    cfg: FastTextTrainConfig = FastTextTrainConfig(),
    positive_label: str = "QA_doc",
    negative_label: str = "Not_QA_doc",
    tmp_dir: Optional[str] = None,
) -> str:
    """Train and save a binary FastText supervised classifier."""
    fasttext = _require_fasttext()

    save_path_p = Path(save_path)
    save_path_p.parent.mkdir(parents=True, exist_ok=True)

    import tempfile

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        train_file = Path(td) / "train.txt"
        with train_file.open("w", encoding="utf-8") as f:
            for t in positives:
                f.write(f"__label__{positive_label} {t.replace(chr(10), ' ')}\n")
            for t in negatives:
                f.write(f"__label__{negative_label} {t.replace(chr(10), ' ')}\n")

        model = fasttext.train_supervised(
            input=str(train_file),
            dim=cfg.dim,
            epoch=cfg.epoch,
            lr=cfg.lr,
            wordNgrams=cfg.word_ngrams,
            minCount=cfg.min_count,
        )
        model.save_model(str(save_path_p))

    return str(save_path_p)


def score_texts(
    *,
    model_path: str,
    texts: Sequence[str],
    target_label: str = "__label__QA_doc",
    k: int = 10,
) -> List[float]:
    """Return probability scores for `target_label` for each text."""
    fasttext = _require_fasttext()
    model = fasttext.load_model(model_path)

    cleaned = [" ".join((t or "").strip().split("\n")) for t in texts]
    labels, probs = model.predict(cleaned, k=k)

    scores: List[float] = []
    for lab_list, prob_list in zip(labels, probs):
        try:
            idx = lab_list.index(target_label)
            scores.append(float(prob_list[idx]))
        except ValueError:
            scores.append(0.0)
    return scores
