"""
datasets.adapters.foursquare — Foursquare Location Matching Adapter
=====================================================================
Single-table deduplication mode: every record is both a query and a target.
A true pair = two records sharing the same ``point_of_interest`` value.

Data source: Foursquare Location Matching dataset (Kaggle competition),
preprocessing_data/train0.csv + preprocessing_data/train1.csv.

IMPORTANT CAVEATS (documented per task B2):
  1. Labels are SYNTHETIC (competition-generated simulated pairs).
     Provenance is recorded as "synthetic" in the label store.
     This is an assumption — see docs/decisions.md.
  2. ``point_of_interest`` is NEVER exposed in model-visible raw_fields.
     It lives only in the label store.
  3. The pair file (haversine_train*.csv) is pre-blocked and must NOT be
     used for retrieval evaluation — it is a candidate subset, not the
     full label set.
  4. All names in this dataset are already lowercased / transliterated to
     ASCII by the competition's preprocessing pipeline. Non-Latin script
     detection on these names reflects post-preprocessing state.

Dataset ID: foursquare
Model-visible columns: id, name, latitude, longitude, address, city,
  state, zip, country, url, phone, categories
Excluded (labels only): point_of_interest, rad_latitude, rad_longitude,
  location_key, cv_flag

Usage::
    from datasets.adapters.foursquare import FoursquareAdapter
    adapter = FoursquareAdapter()
    stats = adapter.build(output_dir="data/processed/foursquare")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ber.store.schemas import RECORDS_SCHEMA, LABELS_SCHEMA, PAIRS_LABELS_SCHEMA

# ---------------------------------------------------------------------------
# Blocking key helpers (also used by tests and the experiment script)
# ---------------------------------------------------------------------------

def extract_url_domain(url: str) -> str:
    """Extract registrable domain from a URL string."""
    if not url:
        return ""
    url = url.lower().strip()
    for prefix in ("https://", "http://", "//"):
        if url.startswith(prefix):
            url = url[len(prefix):]
    url = url.split("/")[0].split("?")[0]
    if url.startswith("www."):
        url = url[4:]
    return url if "." in url else ""


def digits_only(s: str) -> str:
    """Keep only digit characters from string."""
    return "".join(c for c in s if c.isdigit())


def geo_cell(lat: str, lon: str) -> str:
    """Round lat/lon to 2 decimal places → ~1km geo cell key."""
    try:
        la = round(float(lat), 2)
        lo = round(float(lon), 2)
        return f"{la:.2f}|{lo:.2f}"
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "foursquare"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = (
    _REPO_ROOT / "ber" / "data" / "raw" / "foursquare_kaggle" / "preprocessing_data"
)
SOURCE_FILES = ["train0.csv", "train1.csv"]

# Columns exposed in model-visible raw_fields (point_of_interest excluded)
MODEL_COLS = [
    "id", "name", "latitude", "longitude",
    "address", "city", "state", "zip", "country",
    "url", "phone", "categories",
]

# Columns that must NEVER appear in model-visible fields
POI_IDENTIFIER_FIELDS = {"point_of_interest", "rad_latitude", "rad_longitude",
                          "location_key", "cv_flag"}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FoursquareAdapter:
    """Build Foursquare dedup records/labels/pairs Parquet artefacts.

    Parameters
    ----------
    data_dir:
        Directory containing train0.csv and train1.csv.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

    def build(self, output_dir: str | Path) -> dict[str, Any]:
        """Read source CSVs and write Parquet artefacts.

        Returns
        -------
        dict with keys:
            n_records, n_poi_clusters, n_singleton_clusters,
            n_positive_pairs, n_unlabeled_records (singletons)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("[foursquare] Reading source CSVs...", flush=True)
        records: list[dict[str, Any]] = []
        poi_to_ids: dict[str, list[str]] = {}

        for src_file in SOURCE_FILES:
            path = self.data_dir / src_file
            if not path.exists():
                raise FileNotFoundError(f"Source file not found: {path}")
            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                for row_num, row in enumerate(csv.DictReader(fh)):
                    record_id = row["id"]
                    poi = row.get("point_of_interest", "")

                    model_fields = {
                        col: row.get(col, "") for col in MODEL_COLS
                    }
                    records.append({
                        "dataset_id": DATASET_ID,
                        "record_id": record_id,
                        "canon_version": None,
                        "raw_fields": json.dumps(model_fields),
                        "source_file": str(path),
                        "source_row": row_num,
                    })
                    if poi:
                        poi_to_ids.setdefault(poi, []).append(record_id)

        print(f"  {len(records):,} records, {len(poi_to_ids):,} POI clusters", flush=True)

        # Build positive pairs (all ordered pairs within each cluster)
        # For dedup: query_id < cand_id (lexicographic) to avoid duplicates
        pairs: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        singleton_record_ids: set[str] = set()

        for poi, ids in poi_to_ids.items():
            if len(ids) == 1:
                singleton_record_ids.add(ids[0])
                continue
            # Labels: all records in cluster share entity_id = poi
            for rid in ids:
                label_rows.append({
                    "record_id": rid,
                    "entity_id": poi,
                    "label_provenance": "synthetic",
                    "dataset_id": DATASET_ID,
                    "split": None,
                })
            # Positive pairs
            ids_sorted = sorted(ids)
            for i, a in enumerate(ids_sorted):
                for b in ids_sorted[i + 1:]:
                    pairs.append({
                        "query_id": a,
                        "cand_id": b,
                        "label": 1,
                        "label_provenance": "synthetic",
                        "dataset_id": DATASET_ID,
                        "split": None,
                    })

        # Singletons get their own entity_id (record_id itself)
        for rid in singleton_record_ids:
            label_rows.append({
                "record_id": rid,
                "entity_id": rid,   # singleton: entity = self
                "label_provenance": "synthetic",
                "dataset_id": DATASET_ID,
                "split": None,
            })

        print(f"  {len(pairs):,} positive pairs", flush=True)
        print(f"  {len(singleton_record_ids):,} singleton records", flush=True)

        # Write artefacts
        print("[foursquare] Writing Parquet files...", flush=True)
        _write_records(records, output_dir / "records.parquet")
        _write_labels(label_rows, output_dir / "labels.parquet")
        _write_pairs(pairs, output_dir / "pairs_labels.parquet")

        # Save singleton list
        (output_dir / "singleton_record_ids.json").write_text(
            json.dumps(sorted(singleton_record_ids), indent=2)
        )

        stats = {
            "n_records": len(records),
            "n_poi_clusters": len(poi_to_ids),
            "n_singleton_clusters": len(singleton_record_ids),
            "n_non_singleton_clusters": len(poi_to_ids) - len(singleton_record_ids),
            "n_positive_pairs": len(pairs),
            "n_unlabeled_records": len(singleton_record_ids),
        }
        (output_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"  Done. Stats: {stats}", flush=True)
        return stats


# ---------------------------------------------------------------------------
# Leakage assertion
# ---------------------------------------------------------------------------

def assert_no_poi_leakage(records_parquet: str | Path) -> None:
    """Assert that no model-visible record contains POI or derived columns.

    Raises AssertionError if any banned field is found in raw_fields JSON.
    """
    tbl = pq.read_table(records_parquet, columns=["raw_fields"])
    for batch in tbl.to_batches():
        for rf in batch.column("raw_fields").to_pylist():
            if rf is None:
                continue
            try:
                fields = json.loads(rf)
            except Exception:
                continue
            for banned in POI_IDENTIFIER_FIELDS:
                assert banned not in fields, (
                    f"POI leakage: field '{banned}' found in model-visible "
                    f"raw_fields. It must only appear in the label store."
                )


# ---------------------------------------------------------------------------
# Writer helpers
# ---------------------------------------------------------------------------

def _write_records(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.table({
        "dataset_id": [r["dataset_id"] for r in rows],
        "record_id": [r["record_id"] for r in rows],
        "canon_version": [r.get("canon_version") for r in rows],
        "raw_fields": [r["raw_fields"] for r in rows],
        "source_file": [r.get("source_file") for r in rows],
        "source_row": pa.array([r.get("source_row") for r in rows], type=pa.int64()),
    }, schema=RECORDS_SCHEMA)
    pq.write_table(table, path, compression="snappy")


def _write_labels(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.table({
        "record_id": [r["record_id"] for r in rows],
        "entity_id": [r["entity_id"] for r in rows],
        "label_provenance": pa.array(
            [r["label_provenance"] for r in rows],
            type=pa.dictionary(pa.int8(), pa.string()),
        ),
        "dataset_id": [r.get("dataset_id") for r in rows],
        "split": [r.get("split") for r in rows],
    }, schema=LABELS_SCHEMA)
    pq.write_table(table, path, compression="snappy")


def _write_pairs(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.table({
        "query_id": [r["query_id"] for r in rows],
        "cand_id": [r["cand_id"] for r in rows],
        "label": pa.array([r["label"] for r in rows], type=pa.int8()),
        "label_provenance": [r.get("label_provenance") for r in rows],
        "dataset_id": [r.get("dataset_id") for r in rows],
        "split": [r.get("split") for r in rows],
    }, schema=PAIRS_LABELS_SCHEMA)
    pq.write_table(table, path, compression="snappy")
