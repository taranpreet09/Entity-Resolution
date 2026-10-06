"""
ber.store.schemas — Parquet Schema Definitions and Writer Helpers
=================================================================
All on-disk schemas are defined here as PyArrow schemas for portability.
DuckDB is used for reading; PyArrow is used for writing.

Schemas
-------
RECORDS_SCHEMA
    One row per source record.  Raw fields are stored as-is (strings).

LABELS_SCHEMA
    Ground-truth entity assignments.  One row per (record_id, entity_id) pair.

PAIRS_LABELS_SCHEMA
    Optional: explicit pairwise judgements (positive or negative).

CANDIDATES_SCHEMA
    Retrieval output.  One row per (query_id, cand_id) candidate pair,
    with channel provenance, score, and ranking metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

RECORDS_SCHEMA = pa.schema(
    [
        # Identity
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        # Canonical version applied to produce this record view.
        # NULL until the canonicalisation stage has run.
        pa.field("canon_version", pa.string(), nullable=True),
        # Raw fields — stored verbatim as a JSON string so the schema is
        # dataset-agnostic.  Downstream stages parse specific keys.
        pa.field("raw_fields", pa.string(), nullable=False),  # JSON
        # Ingestion bookkeeping
        pa.field("source_file", pa.string(), nullable=True),
        pa.field("source_row", pa.int64(), nullable=True),
    ]
)

LABELS_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        # How was this label derived?
        # manual             — human-annotated
        # identifier_derived — two records share an authoritative identifier
        #                       (e.g. same Companies House number)
        # algorithmic        — produced by a clustering algorithm
        # synthetic          — generated artificially for testing
        pa.field(
            "label_provenance",
            pa.dictionary(pa.int8(), pa.string()),
            nullable=False,
        ),
        pa.field("dataset_id", pa.string(), nullable=True),
        pa.field("split", pa.string(), nullable=True),  # train/val/test
    ]
)

PAIRS_LABELS_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("cand_id", pa.string(), nullable=False),
        # 1 = match, 0 = non-match, -1 = unknown/unlabelled
        pa.field("label", pa.int8(), nullable=False),
        pa.field("label_provenance", pa.string(), nullable=True),
        pa.field("dataset_id", pa.string(), nullable=True),
        pa.field("split", pa.string(), nullable=True),
    ]
)

CANDIDATES_SCHEMA = pa.schema(
    [
        # Pair identity — table is sorted by (query_id, cand_id).
        # Never rely on dict ordering for alignment; always join explicitly.
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("cand_id", pa.string(), nullable=False),
        # Provenance bitmask: bit i is set if channel i contributed this pair.
        # Channel-to-bit mapping is stored in the experiment config.
        pa.field("channel_mask", pa.uint32(), nullable=False),
        # Score from the primary retrieval channel (highest score if multiple).
        pa.field("score", pa.float16(), nullable=False),
        # Rank within the candidate set for this query_id (1-indexed).
        pa.field("rank", pa.uint16(), nullable=False),
        # Rank of the query in the candidate's result list (for symmetric eval).
        pa.field("reverse_rank", pa.uint16(), nullable=False),
        # Retrieval round (for multi-round / iterative retrieval).
        pa.field("round", pa.uint8(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Writer helpers
# ---------------------------------------------------------------------------

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_records(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    append: bool = False,
) -> None:
    """Write a list of record dicts to a Parquet file.

    Parameters
    ----------
    records:
        Each dict must have keys: dataset_id, record_id, raw_fields.
        canon_version, source_file, source_row are optional (default None).
    path:
        Output Parquet file path.
    append:
        If True, append to an existing file; otherwise overwrite.
    """
    import json as _json

    path = Path(path)
    _ensure_parent(path)

    n = len(records)
    table = pa.table(
        {
            "dataset_id": [r["dataset_id"] for r in records],
            "record_id": [r["record_id"] for r in records],
            "canon_version": [r.get("canon_version") for r in records],
            "raw_fields": [
                r["raw_fields"]
                if isinstance(r["raw_fields"], str)
                else _json.dumps(r["raw_fields"])
                for r in records
            ],
            "source_file": [r.get("source_file") for r in records],
            "source_row": pa.array(
                [r.get("source_row") for r in records], type=pa.int64()
            ),
        },
        schema=RECORDS_SCHEMA,
    )

    if append and path.exists():
        existing = pq.read_table(path, schema=RECORDS_SCHEMA)
        table = pa.concat_tables([existing, table])

    pq.write_table(table, path, compression="snappy")


def write_labels(
    labels: list[dict[str, Any]],
    path: str | Path,
    *,
    append: bool = False,
) -> None:
    """Write a list of label dicts to a Parquet file."""
    path = Path(path)
    _ensure_parent(path)

    provenance_values = [r["label_provenance"] for r in labels]
    valid = {"manual", "identifier_derived", "algorithmic", "synthetic"}
    for v in provenance_values:
        if v not in valid:
            raise ValueError(
                f"Invalid label_provenance '{v}'. Must be one of {valid}."
            )

    table = pa.table(
        {
            "record_id": [r["record_id"] for r in labels],
            "entity_id": [r["entity_id"] for r in labels],
            "label_provenance": pa.array(
                provenance_values,
                type=pa.dictionary(pa.int8(), pa.string()),
            ),
            "dataset_id": [r.get("dataset_id") for r in labels],
            "split": [r.get("split") for r in labels],
        },
        schema=LABELS_SCHEMA,
    )

    if append and path.exists():
        existing = pq.read_table(path, schema=LABELS_SCHEMA)
        table = pa.concat_tables([existing, table])

    pq.write_table(table, path, compression="snappy")


def write_candidates(
    candidates: list[dict[str, Any]],
    path: str | Path,
    *,
    append: bool = False,
) -> None:
    """Write a list of candidate dicts to a Parquet file.

    Candidates are sorted by (query_id, cand_id) before writing.
    """
    path = Path(path)
    _ensure_parent(path)

    # Sort by (query_id, cand_id) — explicit, never rely on insertion order.
    candidates = sorted(candidates, key=lambda r: (r["query_id"], r["cand_id"]))

    table = pa.table(
        {
            "query_id": [r["query_id"] for r in candidates],
            "cand_id": [r["cand_id"] for r in candidates],
            "channel_mask": pa.array(
                [r["channel_mask"] for r in candidates], type=pa.uint32()
            ),
            "score": pa.array(
                [r["score"] for r in candidates], type=pa.float16()
            ),
            "rank": pa.array(
                [r["rank"] for r in candidates], type=pa.uint16()
            ),
            "reverse_rank": pa.array(
                [r.get("reverse_rank", 0) for r in candidates], type=pa.uint16()
            ),
            "round": pa.array(
                [r.get("round", 1) for r in candidates], type=pa.uint8()
            ),
            "dataset_id": [r.get("dataset_id") for r in candidates],
        },
        schema=CANDIDATES_SCHEMA,
    )

    if append and path.exists():
        existing = pq.read_table(path, schema=CANDIDATES_SCHEMA)
        table = pa.concat_tables([existing, table])

    pq.write_table(table, path, compression="snappy")
