"""Per-dataset adapters for the generic Stage 2A pipeline.

Each adapter is a small subclass of
:class:`Filtering_Pipeline.stage_2a.filter_and_format.adapter.DatasetAdapter` that
encapsulates the dataset-specific bits (message extraction, subsource label,
domain / language inference, ...). The pipeline, schema, filters, and CLI are
shared across all of them — adding a new source dataset only requires writing
one new adapter file in this folder.
"""
