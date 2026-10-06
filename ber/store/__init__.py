"""
ber.store — Parquet Storage Schemas & Writers
===============================================
Defines the on-disk schemas for all BER artefacts and provides helpers to
write/read them without loading full datasets into RAM.

Schemas:
    records         — raw records with dataset_id, record_id, raw fields
    labels          — ground-truth entity assignments and their provenance
    pairs_labels    — optional explicit positive/negative pair judgements
    candidates      — retrieval output with channel_mask, score, rank metadata

See docs/schemas.md for the full schema reference.
"""
from ber.store.schemas import (  # noqa: F401
    RECORDS_SCHEMA,
    LABELS_SCHEMA,
    PAIRS_LABELS_SCHEMA,
    CANDIDATES_SCHEMA,
    write_records,
    write_labels,
    write_candidates,
)
