"""Re-export of :mod:`Filtering_Pipeline.filters.fasttext_filters`.

Generic FastText binary classifier (training + scoring). Used at Stage 2B
(quality filtering) but kept under ``_utils/`` because it is a low-level
utility, not a phase filter.
"""

from ..fasttext_filters import (  # noqa: F401
    FastTextTrainConfig,
    train_binary_fasttext,
    score_texts,
)

__all__ = [
    "FastTextTrainConfig",
    "train_binary_fasttext",
    "score_texts",
]
