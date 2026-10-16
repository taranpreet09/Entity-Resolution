"""
Tests for datasets.splits.splitter.EntityDisjointSplitter
==========================================================
Verifies:
  - No entity straddles splits
  - No duplicate group straddles splits
  - Ratios are approximately respected
  - Determinism: same seed → same split
  - Different seed → (usually) different split
  - Edge cases: single entity, all same entity
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber_datasets.splits.splitter import EntityDisjointSplitter, _norm_key_hash


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_toy_dataset(n: int = 30, n_entities: int = 10, seed: int = 0):
    """Create a toy dataset with n records and n_entities distinct entities."""
    import random
    rng = random.Random(seed)
    record_ids = [f"r{i}" for i in range(n)]
    entity_ids = [f"e{rng.randint(0, n_entities - 1)}" for _ in range(n)]
    return record_ids, entity_ids


def _check_entity_disjoint(assignments: dict[str, str], entity_ids: list[str]) -> None:
    """Assert no entity straddles splits."""
    entity_to_splits: dict[str, set[str]] = defaultdict(set)
    for rid, split in assignments.items():
        idx = list(assignments.keys()).index(rid)
        entity_to_splits[entity_ids[idx]].add(split)
    for eid, splits in entity_to_splits.items():
        assert len(splits) == 1, (
            f"Entity {eid!r} appears in multiple splits: {splits}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEntityDisjointSplitter:

    def test_no_entity_straddles_splits(self):
        record_ids, entity_ids = _make_toy_dataset(n=60, n_entities=15)
        splitter = EntityDisjointSplitter(seed=42)
        splits = splitter.split(record_ids, entity_ids)

        _check_entity_disjoint(splits, entity_ids)

    def test_all_records_assigned(self):
        record_ids, entity_ids = _make_toy_dataset(n=50)
        splitter = EntityDisjointSplitter(seed=42)
        splits = splitter.split(record_ids, entity_ids)

        assert set(splits.keys()) == set(record_ids)
        assert all(v in ("train", "val", "test") for v in splits.values())

    def test_ratios_approximately_respected(self):
        # With 300 records / 100 entities, ratios should be close to 70/15/15
        record_ids, entity_ids = _make_toy_dataset(n=300, n_entities=100, seed=7)
        splitter = EntityDisjointSplitter(
            ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42
        )
        splits = splitter.split(record_ids, entity_ids)
        from collections import Counter
        counts = Counter(splits.values())
        total = sum(counts.values())
        # Allow ±20 percentage points due to entity-level granularity
        assert abs(counts["train"] / total - 0.7) < 0.2
        assert abs(counts["val"] / total - 0.15) < 0.2
        assert abs(counts["test"] / total - 0.15) < 0.2

    def test_deterministic_same_seed(self):
        record_ids, entity_ids = _make_toy_dataset(n=40, n_entities=12)
        splitter = EntityDisjointSplitter(seed=99)
        splits1 = splitter.split(record_ids, entity_ids)
        splits2 = splitter.split(record_ids, entity_ids)
        assert splits1 == splits2

    def test_different_seeds_differ(self):
        record_ids, entity_ids = _make_toy_dataset(n=100, n_entities=30)
        s1 = EntityDisjointSplitter(seed=1).split(record_ids, entity_ids)
        s2 = EntityDisjointSplitter(seed=2).split(record_ids, entity_ids)
        # Very unlikely to be identical with 30 entities
        assert s1 != s2

    def test_duplicate_group_does_not_straddle(self):
        """Two records with the same (name, address) hash must end up together."""
        record_ids = ["r0", "r1", "r2", "r3", "r4"]
        entity_ids = ["e0", "e1", "e2", "e3", "e4"]  # all different entities
        # r0 and r1 share the same normalised key → must be in same split
        norm_keys = [
            ("acme ltd", "london"),
            ("acme ltd", "london"),   # duplicate of r0
            ("beta inc", "paris"),
            ("gamma co", ""),
            ("delta llp", "berlin"),
        ]
        splitter = EntityDisjointSplitter(
            ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42
        )
        splits = splitter.split(record_ids, entity_ids, norm_keys=norm_keys)
        assert splits["r0"] == splits["r1"], (
            "Duplicate group r0/r1 straddles splits"
        )

    def test_single_entity_all_same_split(self):
        record_ids = [f"r{i}" for i in range(10)]
        entity_ids = ["e0"] * 10
        splitter = EntityDisjointSplitter(seed=42)
        splits = splitter.split(record_ids, entity_ids)
        assert len(set(splits.values())) == 1

    def test_ratio_sum_not_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            EntityDisjointSplitter(ratios={"train": 0.5, "val": 0.3})

    def test_mismatched_lengths_raise(self):
        splitter = EntityDisjointSplitter()
        with pytest.raises(ValueError):
            splitter.split(["r0", "r1"], ["e0"])

    def test_brand_split_no_entity_straddle(self):
        """brand_split must also respect entity disjointness."""
        record_ids = [f"r{i}" for i in range(20)]
        entity_ids = [f"e{i % 5}" for i in range(20)]
        norm_names = [f"brand_{i % 3}" for i in range(20)]
        splitter = EntityDisjointSplitter(seed=42)
        splits = splitter.brand_split(record_ids, norm_names, entity_ids)
        _check_entity_disjoint(splits, entity_ids)

    def test_norm_key_hash_stable(self):
        h1 = _norm_key_hash("  Acme Ltd  ", "London")
        h2 = _norm_key_hash("acme ltd", "london")
        assert h1 == h2

    def test_empty_dataset(self):
        splitter = EntityDisjointSplitter(seed=42)
        result = splitter.split([], [])
        assert result == {}
