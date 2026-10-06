"""
datasets.splits.splitter — Entity-Disjoint Deterministic Splitter
==================================================================
Guarantees that no entity cluster (connected component of the match graph)
straddles train / validation / test splits.

Additionally, records sharing the same normalised-(name, address) hash are
assigned to the same split as their cluster, so near-duplicates never end up
on opposite sides of a split boundary.

A secondary ``brand_split`` mode groups by normalised-name cluster only,
allowing measurement of brand-name memorisation independently of the primary
entity split.

Usage::

    from datasets.splits.splitter import EntityDisjointSplitter

    splitter = EntityDisjointSplitter(
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        seed=42,
    )
    splits = splitter.split(
        record_ids=["r1", "r2", "r3", "r4"],
        entity_ids=["e1", "e1", "e2", "e3"],
        norm_keys=[("acme ltd", "london"), ("acme ltd", "london"),
                   ("beta inc", "paris"), ("gamma co", "")],
    )
    # splits == {"r1": "train", "r2": "train", "r3": "val", "r4": "test"}
    # (exact assignment is seed-dependent; entities never straddle splits)
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Sequence


def _norm_key_hash(name: str, address: str) -> str:
    """Stable hash for a (normalised_name, normalised_address) pair."""
    key = f"{name.lower().strip()}|{address.lower().strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class EntityDisjointSplitter:
    """Deterministic entity-disjoint splitter.

    Parameters
    ----------
    ratios:
        Dict mapping split name → fraction.  Must sum to 1.0 (±1e-9).
    seed:
        Integer random seed.  The same seed always produces the same split.
    """

    def __init__(
        self,
        ratios: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        if ratios is None:
            ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}.")
        self.ratios = ratios
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        record_ids: Sequence[str],
        entity_ids: Sequence[str],
        norm_keys: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, str]:
        """Assign each record to a split.

        Parameters
        ----------
        record_ids:
            Stable record identifiers (strings).
        entity_ids:
            Cluster / entity ID for each record.  Two records with the same
            entity_id will always land in the same split.
        norm_keys:
            Optional list of ``(normalised_name, normalised_address)`` tuples,
            one per record.  Records sharing the same pair hash are co-assigned.
            Pass ``None`` to skip duplicate-group enforcement.

        Returns
        -------
        dict[str, str]
            Mapping from record_id → split name.
        """
        if len(record_ids) != len(entity_ids):
            raise ValueError("record_ids and entity_ids must have the same length.")
        if norm_keys is not None and len(norm_keys) != len(record_ids):
            raise ValueError("norm_keys must have the same length as record_ids.")

        # 1. Build entity → set of record IDs
        entity_to_records: dict[str, set[str]] = defaultdict(set)
        for rid, eid in zip(record_ids, entity_ids):
            entity_to_records[eid].add(rid)

        # 2. Duplicate-group merging: union-find over entities that share a
        #    normalised (name, address) key.
        if norm_keys is not None:
            key_hash_to_entities: dict[str, set[str]] = defaultdict(set)
            for rid, eid, nk in zip(record_ids, entity_ids, norm_keys):
                h = _norm_key_hash(nk[0], nk[1])
                key_hash_to_entities[h].add(eid)

            # Merge entity sets that share a duplicate-group key
            parent = {eid: eid for eid in entity_to_records}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: str, b: str) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for eids in key_hash_to_entities.values():
                eids_list = list(eids)
                for i in range(1, len(eids_list)):
                    union(eids_list[0], eids_list[i])

            # Rebuild entity_to_records using canonical roots
            merged: dict[str, set[str]] = defaultdict(set)
            for eid, records in entity_to_records.items():
                merged[find(eid)].update(records)
            entity_to_records = merged

        # 3. Deterministic shuffle of component IDs, then bin-pack by ratio.
        component_ids = sorted(entity_to_records.keys())
        rng = random.Random(self.seed)
        rng.shuffle(component_ids)

        split_names = list(self.ratios.keys())
        thresholds: list[float] = []
        cumulative = 0.0
        for name in split_names:
            cumulative += self.ratios[name]
            thresholds.append(cumulative)

        n_components = len(component_ids)
        component_to_split: dict[str, str] = {}
        for i, comp_id in enumerate(component_ids):
            frac = (i + 0.5) / max(n_components, 1)
            for split_name, threshold in zip(split_names, thresholds):
                if frac <= threshold:
                    component_to_split[comp_id] = split_name
                    break
            else:
                component_to_split[comp_id] = split_names[-1]

        # 4. Map records to splits via their component
        if norm_keys is not None:
            # Use the merged component lookup
            record_to_component: dict[str, str] = {}
            for comp_id, records in entity_to_records.items():
                for rid in records:
                    record_to_component[rid] = comp_id
            return {
                rid: component_to_split[record_to_component[rid]]
                for rid in record_ids
            }
        else:
            entity_to_split = {
                eid: component_to_split[eid]
                for eid in entity_to_records
            }
            result = {}
            for rid, eid in zip(record_ids, entity_ids):
                result[rid] = entity_to_split[eid]
            return result

    def brand_split(
        self,
        record_ids: Sequence[str],
        norm_names: Sequence[str],
        entity_ids: Sequence[str],
    ) -> dict[str, str]:
        """Secondary split grouped by normalised name cluster.

        Groups records by normalised name ONLY (ignoring address).  Records
        sharing a name cluster land in the same split.  Entity membership is
        still respected; if two name-clusters correspond to the same entity
        they are co-assigned.

        Parameters
        ----------
        record_ids, entity_ids:
            As in :meth:`split`.
        norm_names:
            Normalised legal/trade name for each record.

        Returns
        -------
        dict[str, str]
            Mapping from record_id → split name.
        """
        # Treat each unique normalised name as the "address" empty → each name
        # group hash is distinct; entity merging still applies.
        norm_keys = [(name, "") for name in norm_names]
        return self.split(record_ids, entity_ids, norm_keys=norm_keys)
