"""
datasets.adapters.fodors_zagats — Adapter for the Fodors-Zagats/Products Dataset
==================================================================================
Reads tableA.csv, tableB.csv and the pair-label files (train/valid/test)
without loading full files into RAM.

The dataset ID is ``fodors_zagats`` (preserves supplied folder name).

**Important:** The pairs in this dataset were derived from a blocking step.
They are NOT a random sample of all positive pairs.  Retrieval recall cannot
be measured from this data; see ADR-004 in docs/decisions.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

DATASET_ID = "fodors_zagats"
# Data lives at ber/data/raw/Fodors-Zagats relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _REPO_ROOT / "ber" / "data" / "raw" / "Fodors-Zagats"


def iter_records(
    split: str | None = None,
    data_dir: Path = DATA_DIR,
) -> Iterator[dict]:
    """Yield record dicts in the BER records schema format.

    Both tableA and tableB are yielded with record_id = "A_{id}" / "B_{id}".

    Parameters
    ----------
    split:
        Ignored for record iteration (records themselves have no split);
        pass ``None`` to get all records.
    data_dir:
        Override the data directory (useful in tests).
    """
    for table_name, prefix in [("tableA.csv", "A"), ("tableB.csv", "B")]:
        path = data_dir / table_name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row_num, row in enumerate(reader):
                raw_id = row.get("id", str(row_num))
                yield {
                    "dataset_id": DATASET_ID,
                    "record_id": f"{prefix}_{raw_id}",
                    "canon_version": None,
                    "raw_fields": json.dumps(dict(row)),
                    "source_file": str(path),
                    "source_row": row_num,
                }


def iter_pairs(
    split: str = "train",
    data_dir: Path = DATA_DIR,
) -> Iterator[dict]:
    """Yield pair-label dicts for the given split.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    data_dir:
        Override the data directory.

    Yields
    ------
    dict with keys: query_id, cand_id, label, label_provenance, dataset_id, split
    """
    fname_map = {"train": "train_set.csv", "val": "valid_set.csv", "test": "test_set.csv"}
    fname = fname_map.get(split, f"{split}_set.csv")
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label_raw = str(row.get("label", "1")).strip()
            label = 1 if label_raw == "1" else 0

            left_id = str(row.get("left_id", "")).strip()
            right_id = str(row.get("right_id", "")).strip()

            yield {
                "query_id": f"A_{left_id}",
                "cand_id": f"B_{right_id}",
                "label": label,
                "label_provenance": "manual",
                "dataset_id": DATASET_ID,
                "split": split,
            }


def load_all_pairs(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load pairs from all splits combined."""
    pairs: list[dict] = []
    for split in ("train", "val", "test"):
        try:
            pairs.extend(iter_pairs(split=split, data_dir=data_dir))
        except FileNotFoundError:
            pass
    return pairs
