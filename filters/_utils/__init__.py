"""Cross-cutting filter utilities that don't fit into a single Stage 2A phase.

These helpers are imported on-demand by per-dataset adapters:

- :mod:`Filtering_Pipeline.filters._utils.code_task` — structural validation for
  code-task datasets (``problem`` / ``tests`` / ``solutions`` fields). Used by
  the few datasets that ship raw competitive-programming rows; not part of any
  Sankey phase but called inside Phase 2A.1 by the relevant adapter.

- :mod:`Filtering_Pipeline.filters._utils.fasttext_classifier` — generic FastText
  binary classifier wrapper (training + scoring). Used by the Phase 2B quality
  filtering stage; lives here because it is a shared low-level utility.
"""
