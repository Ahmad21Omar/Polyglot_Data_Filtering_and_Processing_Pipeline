"""Generic Stage 2A filter & format pipeline.

Usage::

    from Filtering_Pipeline.stage_2a.filter_and_format import (
        FilterConfig, english_only_preset, multilingual_preset, run,
    )
    from Filtering_Pipeline.stage_2a.filter_and_format.adapters.dolci_think_sft_7b import (
        DolciThinkSft7BAdapter,
    )

    run(
        adapter=DolciThinkSft7BAdapter(),
        cfg=english_only_preset(),
        input_dataset_dir="/.../allenai__Dolci-Think-SFT-7B/train",
        output_dir="/.../sft_dolci_v0",
        split="train",
    )

Every per-dataset adapter is a small file under ``adapters/``. Adding a new
source dataset only requires writing one new adapter; the pipeline, schema,
and filters are shared.
"""

from .adapter import DatasetAdapter  # noqa: F401
from .config import (  # noqa: F401
    FilterConfig,
    english_only_preset,
    multilingual_preset,
)
from .output_schema import (  # noqa: F401
    KEPT_COLUMNS,
    KeptRowFields,
    build_dropped_row,
    build_kept_row,
    empty_row,
    stable_sha256,
)
from .pipeline import map_row  # noqa: F401
from .runner import run  # noqa: F401

__all__ = [
    "DatasetAdapter",
    "FilterConfig",
    "english_only_preset",
    "multilingual_preset",
    "KEPT_COLUMNS",
    "KeptRowFields",
    "empty_row",
    "stable_sha256",
    "build_dropped_row",
    "build_kept_row",
    "map_row",
    "run",
]
