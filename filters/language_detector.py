"""Language detection using `facebook/fasttext-language-identification`.

This module provides a cached, process-wide language detector built on top of
the Facebook/Meta fastText LID model (218 languages, BCP-47 + script tags).

Model path (default):
    Master_Thesis/models/fasttext/language_detector/model.bin

Download once via:
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin",
        local_dir="Master_Thesis/models/fasttext",
    )

Label format:  __label__<iso639_3>_<script>
Examples:
    __label__eng_Latn  -> English
    __label__deu_Latn  -> German
    __label__fra_Latn  -> French
    __label__zho_Hans  -> Chinese (Simplified)
    __label__zho_Hant  -> Chinese (Traditional)
    __label__jpn_Jpan  -> Japanese
    __label__arb_Arab  -> Arabic
    __label__spa_Latn  -> Spanish

We normalise these to short ISO 639-1 codes where possible (en, de, fr, zh, ja,
ar, es, …) so downstream code can use simple string comparisons.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

# ── ISO 639-3 + script → ISO 639-1 short code mapping ────────────────────────
# Only the labels we care about / are likely to appear in SFT datasets.
_LABEL_TO_SHORT: dict[str, str] = {
    "eng_Latn": "en",
    "deu_Latn": "de",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "por_Latn": "pt",
    "ita_Latn": "it",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "rus_Cyrl": "ru",
    "ukr_Cyrl": "uk",
    "ces_Latn": "cs",
    "slk_Latn": "sk",
    "ron_Latn": "ro",
    "hun_Latn": "hu",
    "fin_Latn": "fi",
    "swe_Latn": "sv",
    "nor_Latn": "no",
    "dan_Latn": "da",
    "tur_Latn": "tr",
    "vie_Latn": "vi",
    "ind_Latn": "id",
    "msa_Latn": "ms",
    "tha_Thai": "th",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "ara_Arab": "ar",
    "arb_Arab": "ar",
    "heb_Hebr": "he",
    "hin_Deva": "hi",
    "ben_Beng": "bn",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "kan_Knda": "kn",
    "mar_Deva": "mr",
    "urd_Arab": "ur",
    "fas_Arab": "fa",
    "swh_Latn": "sw",
    "cat_Latn": "ca",
    "ell_Grek": "el",
    "bul_Cyrl": "bg",
    "hrv_Latn": "hr",
    "srp_Cyrl": "sr",
    "slv_Latn": "sl",
    "lit_Latn": "lt",
    "lav_Latn": "lv",
    "est_Latn": "et",
    # CJK variants → normalise all to "zh"
    "yue_Hant": "zh",   # Cantonese (Traditional)
    "wuu_Hans": "zh",   # Wu / Shanghainese (Simplified)
    "wuu_Hant": "zh",   # Wu / Shanghainese (Traditional)
    "hak_Hant": "zh",   # Hakka
    "nan_Hant": "zh",   # Min Nan / Hokkien
    # Additional common languages
    "por_Latn": "pt",   # already above, alias
    "nob_Latn": "no",   # Norwegian Bokmål
    "nno_Latn": "no",   # Norwegian Nynorsk
    "afr_Latn": "af",   # Afrikaans
    "glg_Latn": "gl",   # Galician
    "eus_Latn": "eu",   # Basque
    "isl_Latn": "is",   # Icelandic
    "mkd_Cyrl": "mk",   # Macedonian
    "bel_Cyrl": "be",   # Belarusian
    "kaz_Cyrl": "kk",   # Kazakh
    "uzb_Latn": "uz",   # Uzbek
    "aze_Latn": "az",   # Azerbaijani
    "tgl_Latn": "tl",   # Filipino/Tagalog
    "mya_Mymr": "my",   # Burmese
    "khm_Khmr": "km",   # Khmer
    "sin_Sinh": "si",   # Sinhala
    "nep_Deva": "ne",   # Nepali
    "som_Latn": "so",   # Somali
    "amh_Ethi": "am",   # Amharic
    "hau_Latn": "ha",   # Hausa
    "yor_Latn": "yo",   # Yoruba
    "ibo_Latn": "ig",   # Igbo
    "zul_Latn": "zu",   # Zulu
}

# Default model path (relative to repo root; override via MODEL_PATH constant below)
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]  # Master_Thesis/
    / "models"
    / "fasttext"
    / "language_detector"
    / "model.bin"
)

MODEL_PATH: str = str(_DEFAULT_MODEL_PATH)


@lru_cache(maxsize=4)
def _load_model(model_path: str):
    """Load and cache the fastText model (once per process)."""
    try:
        import fasttext  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "fasttext is not installed. Run: pip install fasttext-wheel"
        ) from e
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"FastText language-ID model not found at: {model_path}\n"
            "Download it with:\n"
            "  from huggingface_hub import hf_hub_download\n"
            "  hf_hub_download('facebook/fasttext-language-identification', "
            "'model.bin', local_dir='Master_Thesis/models/fasttext')"
        )
    return fasttext.load_model(str(p))


def _clean(text: str) -> str:
    """Collapse whitespace and strip newlines (fastText expects single-line input)."""
    return " ".join((text or "").strip().split())


def label_to_short(label: str) -> str:
    """Convert a raw fastText label (e.g. '__label__eng_Latn') to a short code ('en').

    Falls back to the iso639_3 part (first 3 chars of the lang tag) if not in
    the known mapping.
    """
    raw = label.replace("__label__", "")          # e.g. "eng_Latn"
    if raw in _LABEL_TO_SHORT:
        return _LABEL_TO_SHORT[raw]
    # Fallback: use first part before underscore, truncated to 2 chars
    parts = raw.split("_")
    return parts[0][:2].lower()


def detect_language(
    text: str,
    *,
    model_path: str = MODEL_PATH,
    min_chars: int = 5,
    default_language: str = "en",
) -> str:
    """Detect the dominant language of `text`.

    Returns a short ISO 639-1 code (e.g. "en", "de", "zh").
    Falls back to `default_language` when text is too short or detection fails.

    Args:
        text:             Input text (prompt + response, or just prompt).
        model_path:       Path to the facebook/fasttext-language-identification model.bin.
        min_chars:        Minimum character count to attempt detection (default 5).
                          Set low because CJK text is very dense (17 chars = a full sentence).
        default_language: Fallback code when detection is skipped or fails.
    """
    cleaned = _clean(text)
    if len(cleaned) < min_chars:
        return default_language
    try:
        model = _load_model(model_path)
        labels, _probs = model.predict(cleaned, k=1)
        return label_to_short(labels[0])
    except Exception:
        return default_language


def detect_language_with_confidence(
    text: str,
    *,
    model_path: str = MODEL_PATH,
    k: int = 3,
    min_chars: int = 5,
    default_language: str = "en",
) -> Tuple[str, float]:
    """Detect language and return (short_code, confidence).

    Confidence is the raw fastText probability for the top prediction [0, 1].
    Returns (default_language, 0.0) when text is too short or detection fails.
    """
    cleaned = _clean(text)
    if len(cleaned) < min_chars:
        return default_language, 0.0
    try:
        model = _load_model(model_path)
        labels, probs = model.predict(cleaned, k=k)
        return label_to_short(labels[0]), float(probs[0])
    except Exception:
        return default_language, 0.0


def detect_language_top_k(
    text: str,
    *,
    model_path: str = MODEL_PATH,
    k: int = 3,
    min_chars: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-k (short_code, prob) predictions sorted by probability descending."""
    cleaned = _clean(text)
    if len(cleaned) < min_chars:
        return []
    try:
        model = _load_model(model_path)
        labels, probs = model.predict(cleaned, k=k)
        return [(label_to_short(l), float(p)) for l, p in zip(labels, probs)]
    except Exception:
        return []


def is_mixed_language(
    text: str,
    *,
    model_path: str = MODEL_PATH,
    min_chars: int = 20,
    confidence_threshold: float = 0.75,
    default_keep: bool = True,
) -> bool:
    """Return True if the text appears to be a language mix (should be dropped).

    Strategy: if the top fastText prediction has confidence below
    `confidence_threshold`, the model is uncertain → text is likely
    mixed-language or noisy.  We only apply this for texts longer than
    `min_chars` to avoid false-positives on short strings.

    Args:
        text:                 Input text to test.
        model_path:           Path to the 218-language fastText model.bin.
        min_chars:            Minimum character count before the check is applied.
                              Shorter texts are considered "not mixed" by default.
        confidence_threshold: Minimum confidence score for the dominant language
                              to be considered "clean" (not mixed). Default 0.75.
        default_keep:         What to return when text is too short to judge.
                              True = don't drop short texts (default).

    Returns:
        True  → text is mixed-language or unrecognisable → should be dropped.
        False → text is confidently single-language → keep.
    """
    cleaned = _clean(text)
    if len(cleaned) < min_chars:
        return not default_keep
    try:
        model = _load_model(model_path)
        _labels, probs = model.predict(cleaned, k=1)
        top_confidence = float(probs[0])
        return top_confidence < confidence_threshold
    except Exception:
        return False
