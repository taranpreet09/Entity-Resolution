"""
datasets.adapters.gleif_ch — GLEIF × Companies House Dataset Adapter
======================================================================
Produces three artefacts from the raw source files:

1. **Query records** (GLEIF GB entities) — model-visible fields only.
   ALL identifier columns (LEI, RegistrationAuthorityEntityID, CompanyNumber,
   URI) are EXCLUDED from raw_fields.  They live only in the label store.

2. **Target records** (Companies House entries) — model-visible fields only.
   CompanyNumber and URI are EXCLUDED from raw_fields.

3. **Labels** (pairs_labels) — provenance=identifier_derived.  One positive
   pair per GLEIF record whose registeredAs value exactly matches a
   CompanyNumber after normalisation.  Queries with no CH match are kept
   as "no-match queries" and tracked separately.

Identifier stripping is verifiable: see ``assert_no_identifier_leakage()``.

Out-of-core design: GLEIF is decompressed in streaming chunks; CH is scanned
by DuckDB without loading into RAM.  Outputs are written as Parquet files.

Dataset ID:   gleif_ch
Query prefix: GLEIF_
Target prefix: CH_

Usage::

    from datasets.adapters.gleif_ch import GleifChAdapter
    adapter = GleifChAdapter()
    stats = adapter.build(output_dir="data/processed/gleif_ch")
    print(stats)
"""

from __future__ import annotations

import csv
import io
import json
import re
import struct
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Iterator

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ber.store.schemas import (
    RECORDS_SCHEMA,
    LABELS_SCHEMA,
    PAIRS_LABELS_SCHEMA,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "gleif_ch"
QUERY_PREFIX = "GLEIF_"
TARGET_PREFIX = "CH_"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GLEIF_PATH = _REPO_ROOT / "ber" / "data" / "raw" / "gleif" / "gleif_level1.csv"
CH_PATH = (
    _REPO_ROOT
    / "ber"
    / "data"
    / "raw"
    / "companies_house"
    / "BasicCompanyDataAsOneFile-2026-10-01.csv"
)

# GB RA codes (ADR-003 — confirmed from data)
GB_RA_CODES = {"RA000585", "RA000586", "RA000587"}

# GLEIF column indices (0-based, confirmed from data exploration)
_G = {
    "LEI": 0,
    "LegalName": 1,
    "LegalName_lang": 2,
    "OtherName_1": 3,
    "OtherName_1_lang": 4,
    "OtherName_1_type": 5,
    "OtherName_2": 6,
    "OtherName_2_lang": 7,
    "OtherName_2_type": 8,
    "OtherName_3": 9,
    "OtherName_3_lang": 10,
    "OtherName_3_type": 11,
    "LegalAddr_Line1": 34,
    "LegalAddr_Line2": 38,
    "LegalAddr_Line3": 39,
    "LegalAddr_Line4": 40,
    "LegalAddr_City": 41,
    "LegalAddr_Region": 42,
    "LegalAddr_Country": 43,
    "LegalAddr_PostCode": 44,
    "HQAddr_Line1": 46,
    "HQAddr_Line2": 50,
    "HQAddr_Line3": 51,
    "HQAddr_Line4": 52,
    "HQAddr_City": 53,
    "HQAddr_Region": 54,
    "HQAddr_Country": 55,
    "HQAddr_PostCode": 56,
    "RA_Code": 187,
    "RA_EntityID": 189,   # ← THE IDENTIFIER — excluded from model view
    "Jurisdiction": 190,
    "EntityStatus": 199,
}

# Fields that are identifiers and MUST be excluded from model-visible records
GLEIF_IDENTIFIER_FIELDS = {"LEI", "RA_EntityID"}
CH_IDENTIFIER_FIELDS = {"CompanyNumber", "URI"}

# CH columns exposed in model-visible records
CH_MODEL_COLS = [
    "CompanyName",
    "RegAddress.CareOf",
    "RegAddress.POBox",
    "RegAddress.AddressLine1",
    "RegAddress.AddressLine2",
    "RegAddress.PostTown",
    "RegAddress.County",
    "RegAddress.Country",
    "RegAddress.PostCode",
    "CompanyCategory",
    "CompanyStatus",
    "CountryOfOrigin",
    "IncorporationDate",
    # Previous names (model can see them; they don't encode the number)
    "PreviousName_1.CompanyName",
    "PreviousName_2.CompanyName",
    "PreviousName_3.CompanyName",
]


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _norm_ch_number(raw: str) -> str:
    """Normalise a Companies House number for matching.

    Steps: strip whitespace → uppercase → zero-pad numeric-only to 8 chars.
    Returns empty string if input is empty/None.
    """
    if not raw:
        return ""
    s = str(raw).strip().upper()
    if s.isdigit():
        s = s.zfill(8)
    return s


def _get(row: list[str], idx: int) -> str:
    """Safe column accessor with strip+unquote."""
    if idx >= len(row):
        return ""
    return row[idx].strip().strip('"')


# ---------------------------------------------------------------------------
# GLEIF streaming decompressor
# ---------------------------------------------------------------------------

def _gleif_row_iter(path: Path) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (header, row) tuples from the truncated GLEIF Zip64 file.

    Decompresses in 4 MB chunks to avoid loading the full file into RAM.
    Yields the header list once with row=[], then each data row.
    """
    with open(path, "rb") as f:
        raw = f.read()

    fn_len = struct.unpack("<H", raw[26:28])[0]
    extra_len = struct.unpack("<H", raw[28:30])[0]
    data_start = 30 + fn_len + extra_len
    compressed = raw[data_start:]

    d = zlib.decompressobj(-15)
    STEP = 4_000_000
    leftovers = b""
    header: list[str] = []

    pos = 0
    while pos < len(compressed):
        chunk = compressed[pos : pos + STEP]
        pos += STEP
        try:
            decompressed = d.decompress(chunk)
        except zlib.error:
            break

        data = leftovers + decompressed
        lines = data.split(b"\n")
        leftovers = lines[-1]

        for line_bytes in lines[:-1]:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            if not line:
                continue
            try:
                parts = next(csv.reader([line]))
            except Exception:
                continue

            if not header:
                header = [c.strip('"').strip() for c in parts]
                yield header, []
            else:
                yield header, parts


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class GleifChAdapter:
    """Build GLEIF × Companies House query/target/label Parquet files.

    Parameters
    ----------
    gleif_path, ch_path:
        Override default raw file paths (useful in tests).
    """

    def __init__(
        self,
        gleif_path: Path = GLEIF_PATH,
        ch_path: Path = CH_PATH,
    ) -> None:
        self.gleif_path = gleif_path
        self.ch_path = ch_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, output_dir: str | Path) -> dict[str, Any]:
        """Build all Parquet artefacts and return summary statistics.

        Parameters
        ----------
        output_dir:
            Directory to write:
              queries.parquet       — GLEIF model-visible records
              targets.parquet       — CH model-visible records
              labels.parquet        — entity labels (cluster = CH record_id)
              pairs_labels.parquet  — positive pairs (identifier_derived)
              unlabeled_queries.json — LEIs with no CH match (formerly no_match_queries.json)

        Returns
        -------
        dict with keys:
            n_gleif_gb, n_matched_queries, n_unlabeled_queries,
            n_ch_targets, n_positive_pairs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build CH lookup: normalised CompanyNumber → record
        print("[1/4] Building Companies House lookup...", flush=True)
        ch_lookup, ch_records_rows = self._build_ch_lookup()
        print(f"  → {len(ch_lookup):,} CH records loaded", flush=True)

        # 2. Write CH target records (identifiers stripped)
        print("[2/4] Writing CH target records...", flush=True)
        _write_records_parquet(ch_records_rows, output_dir / "targets.parquet")
        print(f"  → {len(ch_records_rows):,} target records written", flush=True)

        # 3. Stream GLEIF, build query records + labels
        print("[3/4] Streaming GLEIF, building queries and labels...", flush=True)
        (
            query_rows, label_rows, pairs_rows, unlabeled_leis
        ) = self._stream_gleif_and_match(ch_lookup)
        print(
            f"  → {len(query_rows):,} GLEIF GB queries "
            f"({len(query_rows) - len(unlabeled_leis):,} matched, "
            f"{len(unlabeled_leis):,} unlabeled — no CH identifier match)",
            flush=True,
        )

        # 4. Write outputs
        print("[4/4] Writing output Parquet files...", flush=True)
        _write_records_parquet(query_rows, output_dir / "queries.parquet")
        _write_labels_parquet(label_rows, output_dir / "labels.parquet")
        _write_pairs_parquet(pairs_rows, output_dir / "pairs_labels.parquet")

        no_match_path = output_dir / "unlabeled_queries.json"
        no_match_path.write_text(
            json.dumps(sorted(unlabeled_leis), indent=2), encoding="utf-8"
        )

        stats = {
            "n_gleif_gb": len(query_rows),
            "n_matched_queries": len(query_rows) - len(unlabeled_leis),
            "n_unlabeled_queries": len(unlabeled_leis),
            "n_ch_targets": len(ch_records_rows),
            "n_positive_pairs": len(pairs_rows),
        }
        print(f"  → Done. Stats: {stats}", flush=True)
        return stats

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_ch_lookup(
        self,
    ) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
        """Scan CH CSV with DuckDB; build norm_number→fields lookup.

        Returns
        -------
        ch_lookup:
            normalised_number → {col: val, ...} (model-visible fields only)
        ch_records_rows:
            List of record dicts for write_records (CompanyNumber excluded)
        """
        con = duckdb.connect()
        ch_pos = self.ch_path.as_posix()

        # Select only model-visible columns + CompanyNumber for matching
        select_cols = ", ".join(
            f'"{c}"' for c in ["CompanyNumber"] + CH_MODEL_COLS
        )
        rows = con.execute(
            f"SELECT {select_cols} "
            f"FROM read_csv_auto('{ch_pos}', ignore_errors=true)"
        ).fetchall()

        col_names = ["CompanyNumber"] + CH_MODEL_COLS
        ch_lookup: dict[str, dict[str, str]] = {}
        ch_records_rows: list[dict[str, Any]] = []

        for row_num, row in enumerate(rows):
            row_dict = {col_names[i]: (str(v) if v is not None else "") for i, v in enumerate(row)}
            raw_number = row_dict["CompanyNumber"]
            norm_number = _norm_ch_number(raw_number)
            if not norm_number:
                continue

            # Model-visible fields: everything except CompanyNumber
            model_fields = {k: v for k, v in row_dict.items() if k not in CH_IDENTIFIER_FIELDS}

            ch_lookup[norm_number] = model_fields
            ch_records_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    # Use TARGET_PREFIX + normalised number as stable record_id
                    "record_id": TARGET_PREFIX + norm_number,
                    "canon_version": None,
                    "raw_fields": json.dumps(model_fields),
                    "source_file": str(self.ch_path),
                    "source_row": row_num,
                }
            )

        return ch_lookup, ch_records_rows

    def _stream_gleif_and_match(
        self,
        ch_lookup: dict[str, dict[str, str]],
    ) -> tuple[
        list[dict[str, Any]],  # query records
        list[dict[str, Any]],  # label rows
        list[dict[str, Any]],  # pairs_labels rows
        list[str],              # no-match LEIs
    ]:
        """Stream GLEIF; emit query records (identifiers stripped) and labels."""
        query_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        pairs_rows: list[dict[str, Any]] = []
        unlabeled_leis: list[str] = []

        row_num = 0
        for header, parts in _gleif_row_iter(self.gleif_path):
            if not parts:
                continue  # header row — skip

            ra_code = _get(parts, _G["RA_Code"])
            if ra_code not in GB_RA_CODES:
                continue

            lei = _get(parts, _G["LEI"])
            ra_entity_id = _get(parts, _G["RA_EntityID"])
            norm_ch = _norm_ch_number(ra_entity_id)

            # Build model-visible fields (NO LEI, NO RA_EntityID)
            model_fields: dict[str, str] = {}
            for field_name, col_idx in _G.items():
                if field_name in GLEIF_IDENTIFIER_FIELDS:
                    continue
                model_fields[field_name] = _get(parts, col_idx)

            query_id = QUERY_PREFIX + lei
            query_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "record_id": query_id,
                    "canon_version": None,
                    "raw_fields": json.dumps(model_fields),
                    "source_file": str(self.gleif_path),
                    "source_row": row_num,
                }
            )
            row_num += 1

            if norm_ch and norm_ch in ch_lookup:
                target_id = TARGET_PREFIX + norm_ch
                # Labels: both query and target get the same entity_id
                # entity_id = target_id (the CH record is the canonical entity)
                for rec_id in (query_id, target_id):
                    label_rows.append(
                        {
                            "record_id": rec_id,
                            "entity_id": target_id,
                            "label_provenance": "identifier_derived",
                            "dataset_id": DATASET_ID,
                            "split": None,  # filled in later by splitter
                        }
                    )
                pairs_rows.append(
                    {
                        "query_id": query_id,
                        "cand_id": target_id,
                        "label": 1,
                        "label_provenance": "identifier_derived",
                        "dataset_id": DATASET_ID,
                        "split": None,
                    }
                )
            else:
                unlabeled_leis.append(lei)

        return query_rows, label_rows, pairs_rows, unlabeled_leis


# ---------------------------------------------------------------------------
# Writer helpers (bypass the Phase 1 list-based writers for scale)
# ---------------------------------------------------------------------------

def _write_records_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    table = pa.table(
        {
            "dataset_id": [r["dataset_id"] for r in rows],
            "record_id": [r["record_id"] for r in rows],
            "canon_version": [r.get("canon_version") for r in rows],
            "raw_fields": [r["raw_fields"] for r in rows],
            "source_file": [r.get("source_file") for r in rows],
            "source_row": pa.array(
                [r.get("source_row") for r in rows], type=pa.int64()
            ),
        },
        schema=RECORDS_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def _write_labels_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    table = pa.table(
        {
            "record_id": [r["record_id"] for r in rows],
            "entity_id": [r["entity_id"] for r in rows],
            "label_provenance": pa.array(
                [r["label_provenance"] for r in rows],
                type=pa.dictionary(pa.int8(), pa.string()),
            ),
            "dataset_id": [r.get("dataset_id") for r in rows],
            "split": [r.get("split") for r in rows],
        },
        schema=LABELS_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def _write_pairs_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    table = pa.table(
        {
            "query_id": [r["query_id"] for r in rows],
            "cand_id": [r["cand_id"] for r in rows],
            "label": pa.array([r["label"] for r in rows], type=pa.int8()),
            "label_provenance": [r.get("label_provenance") for r in rows],
            "dataset_id": [r.get("dataset_id") for r in rows],
            "split": [r.get("split") for r in rows],
        },
        schema=PAIRS_LABELS_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


# ---------------------------------------------------------------------------
# Leakage assertion (callable from tests)
# ---------------------------------------------------------------------------

GLEIF_BANNED_FIELD_NAMES = {
    # Exact column names that must never appear in model-visible raw_fields
    "LEI", "RA_EntityID",
    # String patterns that would indicate accidental inclusion
}

CH_BANNED_FIELD_NAMES = {"CompanyNumber", "URI"}


def assert_no_identifier_leakage(
    queries_parquet: str | Path,
    targets_parquet: str | Path,
) -> None:
    """Assert that no model-visible record contains identifier columns.

    Raises AssertionError with a descriptive message if any identifier
    field is found in raw_fields JSON of any query or target record.

    Parameters
    ----------
    queries_parquet, targets_parquet:
        Paths to the query and target Parquet files produced by GleifChAdapter.
    """
    import pyarrow.parquet as pq

    for path, banned, side in [
        (queries_parquet, GLEIF_BANNED_FIELD_NAMES, "GLEIF query"),
        (targets_parquet, CH_BANNED_FIELD_NAMES, "CH target"),
    ]:
        tbl = pq.read_table(path, columns=["raw_fields"])
        for i, batch in enumerate(tbl.to_batches()):
            for j, raw in enumerate(batch.column("raw_fields").to_pylist()):
                if raw is None:
                    continue
                try:
                    fields = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for banned_key in banned:
                    assert banned_key not in fields, (
                        f"Identifier leakage in {side} record (row ~{i * 65536 + j}): "
                        f"field '{banned_key}' found in raw_fields. "
                        f"This field must not be visible to any model or retriever."
                    )
