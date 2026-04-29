from __future__ import annotations

"""Stage 2B — Multilingual quality scorer (FineWeb-HQ + XLM-R).

Architecture
------------
For non-English corpora we use the **FineWeb-HQ classifier heads**
(``epfml/FineWeb-HQ-Classifiers``) on top of frozen
**XLM-RoBERTa base** embeddings (``FacebookAI/xlm-roberta-base``):

::

    [text]
        │ tokenise (XLM-R tokenizer, max_len 512)
        ▼
    [token ids + attention mask]
        │ XLM-R base, frozen
        ▼
    [last hidden state]
        │ mean-pool over the attention mask
        ▼
    [768-dim sentence embedding]
        │ language-specific classifier head (256→1 MLP, sigmoid)
        ▼
    [quality_score ∈ (0, 1)]

The classifier heads are tiny PyTorch state-dicts (~1.5 MB each), one per
language. The reference filename convention used by ``epfml/FineWeb-HQ-
Classifiers`` is ``{ISO 639-3}_{Script}.pt``, e.g.:

==========  ==================  ========================
ISO 639-1   ISO 639-3 + script  classifier filename
==========  ==================  ========================
de          deu_Latn            ``deu_Latn.pt``
fr          fra_Latn            ``fra_Latn.pt``
it          ita_Latn            ``ita_Latn.pt``
es          spa_Latn            ``spa_Latn.pt``
ja          jpn_Jpan            ``jpn_Jpan.pt``
en          eng_Latn            ``eng_Latn.pt``
==========  ==================  ========================

(:data:`DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME` is the full mapping.)

Download (one-off)
------------------

The XLM-R embedding model is fetched automatically by ``transformers`` on
first use. The classifier heads must be downloaded explicitly:

.. code-block:: python

    from huggingface_hub import hf_hub_download
    for fname in ("deu_Latn.pt", "fra_Latn.pt", "ita_Latn.pt",
                  "spa_Latn.pt", "jpn_Jpan.pt"):
        hf_hub_download(
            repo_id="epfml/FineWeb-HQ-Classifiers",
            filename=fname,
            local_dir="models/FineWeb-HQ-Classifiers",
        )

This module **does not** trigger the download. Missing files raise a clear
``FileNotFoundError`` with the snippet above.

Calibration
-----------
Identical to the English path — :func:`...english_fasttext.calibrate_threshold`
is reused via the runner.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .text_assembly import build_scoring_text


# ── language → classifier filename ──────────────────────────────────────────
DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME: Dict[str, str] = {
    "de": "deu_Latn.pt",
    "fr": "fra_Latn.pt",
    "it": "ita_Latn.pt",
    "es": "spa_Latn.pt",
    "ja": "jpn_Jpan.pt",
    "en": "eng_Latn.pt",
    "pt": "por_Latn.pt",
    "nl": "nld_Latn.pt",
    "pl": "pol_Latn.pt",
    "ru": "rus_Cyrl.pt",
    "zh": "zho_Hans.pt",
}

DEFAULT_EMBEDDING_MODEL_ID = "FacebookAI/xlm-roberta-base"
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_CLASSIFIER_HIDDEN_DIM = 256
DEFAULT_MAX_TOKENS = 512


_DOWNLOAD_HINT = (
    "Download the FineWeb-HQ classifier heads with:\n"
    "  from huggingface_hub import hf_hub_download\n"
    "  for fname in ('deu_Latn.pt','fra_Latn.pt','ita_Latn.pt','spa_Latn.pt','jpn_Jpan.pt'):\n"
    "      hf_hub_download(\n"
    "          repo_id='epfml/FineWeb-HQ-Classifiers',\n"
    "          filename=fname,\n"
    "          local_dir='models/FineWeb-HQ-Classifiers',\n"
    "      )"
)


# ── classifier head ─────────────────────────────────────────────────────────


def _binary_classifier_module(
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_dim: int = DEFAULT_CLASSIFIER_HIDDEN_DIM,
):
    """Construct the small MLP head used by FineWeb-HQ classifiers.

    Architecture: ``Linear(emb→hidden) → ReLU → Dropout(0.2) → Linear(hidden→1)``.
    Returns a ``torch.nn.Module`` in eval mode (no parameters loaded yet).
    """
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_dim, 1),
    )


def load_classifier_head(
    state_dict_path: str | Path,
    *,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_dim: int = DEFAULT_CLASSIFIER_HIDDEN_DIM,
):
    """Load one FineWeb-HQ classifier head from disk.

    The state-dict has the standard PyTorch ``Sequential`` layout because
    the original training code wrapped the head in a single ``Sequential``
    named ``classifier``. We mirror that here so ``load_state_dict`` is a
    drop-in match.
    """
    import torch

    p = Path(state_dict_path)
    if not p.exists():
        raise FileNotFoundError(
            f"FineWeb-HQ classifier head not found at: {p}\n{_DOWNLOAD_HINT}"
        )

    # Wrap in the same {"classifier": Sequential(...)} container the
    # original training script used so the saved state_dict keys match.
    head = torch.nn.Module()
    head.classifier = _binary_classifier_module(embedding_dim, hidden_dim)
    state = torch.load(str(p), map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.eval()
    return head


def _try_load_embedding_model(model_id: str = DEFAULT_EMBEDDING_MODEL_ID):
    """Load XLM-R tokenizer + base model. Triggers HF auto-download once."""
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "transformers is not installed. Run: pip install transformers torch"
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    embed_model = AutoModel.from_pretrained(model_id)
    embed_model.eval()
    return tokenizer, embed_model


# ── pooling + scoring ───────────────────────────────────────────────────────


def _mean_pool(hidden_states, attention_mask):
    """Length-normalised mean pool that respects the attention mask.

    ``hidden_states`` has shape ``[B, T, H]``; ``attention_mask`` is
    ``[B, T]`` with 1 for valid tokens. The pooled output is ``[B, H]``.
    """
    import torch

    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _score_texts_with_objects(
    texts: Sequence[str],
    *,
    tokenizer,
    embed_model,
    classifier,
    device,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[float]:
    """Run one batch of texts through XLM-R + classifier head.

    All inputs are pre-loaded; the caller is responsible for moving
    tokeniser / model / classifier to the right device.
    """
    import torch
    import torch.nn.functional as F

    if not texts:
        return []
    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
        pooled = _mean_pool(
            outputs.last_hidden_state.float(), inputs["attention_mask"]
        )
        logits = classifier(pooled.cpu())
        scores = F.sigmoid(logits).squeeze(-1).tolist()
    if isinstance(scores, float):
        return [float(scores)]
    return [float(s) for s in scores]


# ── public scorer object ────────────────────────────────────────────────────


@dataclass
class MultilingualFineWebHqScorer:
    """Stage 2B multilingual quality scorer.

    Parameters
    ----------
    language
        ISO 639-1 code of the language we are scoring (``"de"``,
        ``"fr"``, ...). Used to pick the classifier head filename via
        :data:`DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME` unless the caller
        passes ``classifier_path`` directly.
    classifiers_dir
        Directory containing the per-language classifier heads
        (``deu_Latn.pt`` etc.). Required.
    classifier_path
        Optional explicit path to the classifier head; if set, overrides
        the lookup-by-language and ``classifiers_dir``.
    embedding_model_id
        HF id of the embedding model. Default
        ``"FacebookAI/xlm-roberta-base"``. The first call triggers an
        auto-download by the ``transformers`` library.
    device
        ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` / ... Default ``"cpu"``.
    batch_size
        Batch size for the embedding+classifier pass. Default 8 (matches
        the legacy multilingual script).
    max_tokens
        Max XLM-R tokens per input. Default 512.

    Lifecycle
    ---------
    The scorer lazily loads the embedding model and classifier head on
    first use; subsequent ``score_texts`` calls reuse them. The objects
    stay on the device until garbage collection.
    """

    language: str
    classifiers_dir: str | Path | None = None
    classifier_path: str | Path | None = None
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    device: str = "cpu"
    batch_size: int = 8
    max_tokens: int = DEFAULT_MAX_TOKENS
    name: str = field(init=False)

    # internal lazy-loaded handles
    _tokenizer: Optional[Any] = field(default=None, init=False, repr=False)
    _embed_model: Optional[Any] = field(default=None, init=False, repr=False)
    _classifier: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"fineweb_hq_{self.language}"

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_classifier_path(self) -> Path:
        if self.classifier_path:
            return Path(self.classifier_path)
        if not self.classifiers_dir:
            raise ValueError(
                "MultilingualFineWebHqScorer needs either `classifier_path` "
                "or `classifiers_dir` to locate the per-language head."
            )
        fname = DEFAULT_LANGUAGE_TO_CLASSIFIER_FILENAME.get(self.language.lower())
        if not fname:
            raise ValueError(
                f"No default FineWeb-HQ classifier filename for language "
                f"{self.language!r}; pass `classifier_path` explicitly."
            )
        return Path(self.classifiers_dir) / fname

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None or self._embed_model is None:
            self._tokenizer, self._embed_model = _try_load_embedding_model(
                self.embedding_model_id
            )
            import torch  # local import keeps torch optional at module load

            self._embed_model = self._embed_model.to(torch.device(self.device))
        if self._classifier is None:
            self._classifier = load_classifier_head(self._resolve_classifier_path())

    # ── public API ───────────────────────────────────────────────────────

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        """Score a sequence of plain-text strings."""
        if not texts:
            return []
        self._ensure_loaded()
        import torch

        device = torch.device(self.device)
        out: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            out.extend(
                _score_texts_with_objects(
                    chunk,
                    tokenizer=self._tokenizer,
                    embed_model=self._embed_model,
                    classifier=self._classifier,
                    device=device,
                    max_tokens=self.max_tokens,
                )
            )
        return out

    def score_rows(self, rows: Sequence[dict]) -> List[float]:
        """Score SFT rows via :func:`text_assembly.build_scoring_text`."""
        return self.score_texts([build_scoring_text(r) for r in rows])

    def calibrate(
        self,
        scores: Sequence[float],
        *,
        base_threshold: float,
        max_drop_rate: float,
    ):
        """Reuse the English calibrator (the calibration policy is identical)."""
        from .english_fasttext import calibrate_threshold

        return calibrate_threshold(
            scores, base_threshold=base_threshold, max_drop_rate=max_drop_rate
        )
