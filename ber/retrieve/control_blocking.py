"""
ber.retrieve.control_blocking — Control Baseline Exact-Key Blocker
====================================================================
A generic, config-driven blocking component that imitates the "old pipeline"
of exact-key matching with a bucket-size gate.

Four blocking keys are defined (all use the control canonicalisation):

    K1  full_canon_name
        Canonical legal name of GLEIF entity vs CompanyName.
        Simple, high-precision, misses abbreviations and spelling variants.

    K2  first_token_plus_outward
        first_token(name) + "|" + outward_code(postcode)
        Catches name-prefix matches within the same postal district.

    K3  first3_plus_district
        first_3_chars(name) + "|" + postcode_district
        Fuzzier than K2; catches truncated names in the same district.

    K4  outward_plus_first_token_target
        outward_code + "|" + first_token(CompanyName)
        Postcode-primary; used when the query name is noisy but address is clean.

Bucket-size gate G: buckets with more than G candidates are dropped entirely
(all pairs from that bucket are discarded).  This is the "size gate" that
caps candidate set size at the cost of recall.

Memory note
-----------
The 5.7 M Companies House records are ~13 GB when loaded as Python dicts.
Use ``index_targets_from_duckdb()`` instead of ``index_targets()``:
it streams only CompanyName + PostCode from DuckDB (≈ 300 MB) and never
materialises the full JSON blob.  The index stores only
``{key_value → [record_id, ...]}`` strings (≈ 250 MB total).

Usage::

    from ber.retrieve.control_blocking import ControlBlocker, BlockingConfig

    config = BlockingConfig(gate_size=50, keys=["K1", "K2", "K3", "K4"])
    blocker = ControlBlocker(config)
    # Fast, out-of-core — use this for real data:
    blocker.index_targets_from_duckdb(ch_csv_path)
    result = blocker.retrieve(query_records)

    # result.candidates: list of CandidateRow dicts
    # result.stats: per-key stats dict
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ber.canon.normalise import (
    canonicalise,
    first_token,
    first_n_chars,
    outward_code,
    postcode_district,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_KEY_NAMES = ["K1", "K2", "K3", "K4"]

# Channel bitmask: bit i corresponds to KEY_NAMES[i]
KEY_BIT = {name: (1 << i) for i, name in enumerate(ALL_KEY_NAMES)}


@dataclass
class BlockingConfig:
    """Configuration for the control blocker.

    Parameters
    ----------
    gate_size:
        Maximum bucket size.  Buckets larger than this are dropped.
        None means no gate (all buckets retained regardless of size).
    keys:
        Which keys to activate.  Defaults to all four.
    """
    gate_size: int | None = 50
    keys: list[str] = field(default_factory=lambda: list(ALL_KEY_NAMES))

    def __post_init__(self) -> None:
        unknown = set(self.keys) - set(ALL_KEY_NAMES)
        if unknown:
            raise ValueError(f"Unknown blocking key(s): {unknown}. "
                             f"Valid: {ALL_KEY_NAMES}")


# ---------------------------------------------------------------------------
# Key extraction helpers
# ---------------------------------------------------------------------------

def _query_keys(raw_fields: dict[str, str]) -> dict[str, str]:
    """Compute all blocking key values for a GLEIF query record."""
    name = raw_fields.get("LegalName", "")
    # Use legal address postcode; fall back to HQ postcode
    postcode = raw_fields.get("LegalAddr_PostCode", "") or raw_fields.get("HQAddr_PostCode", "")

    canon_name = canonicalise(name)
    tok = first_token(name)
    f3 = first_n_chars(name, 3)
    out = outward_code(postcode)
    dist = postcode_district(postcode)

    return {
        "K1": canon_name,
        "K2": f"{tok}|{out}" if tok and out else "",
        "K3": f"{f3}|{dist}" if f3 and dist else "",
        "K4": f"{out}|{tok}" if out and tok else "",
    }


def _target_keys(raw_fields: dict[str, str]) -> dict[str, str]:
    """Compute all blocking key values for a CH target record."""
    name = raw_fields.get("CompanyName", "")
    postcode = raw_fields.get("RegAddress.PostCode", "")

    canon_name = canonicalise(name)
    tok = first_token(name)
    f3 = first_n_chars(name, 3)
    out = outward_code(postcode)
    dist = postcode_district(postcode)

    return {
        "K1": canon_name,
        "K2": f"{tok}|{out}" if tok and out else "",
        "K3": f"{f3}|{dist}" if f3 and dist else "",
        "K4": f"{out}|{tok}" if out and tok else "",
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BlockingResult:
    """Output of a single retrieval pass."""
    candidates: list[dict[str, Any]]   # list of CandidateRow dicts
    stats: dict[str, Any]              # per-key and summary stats
    size_gated_pairs: set[tuple[str, str]]   # pairs dropped by gate
    retrieved_pairs: set[tuple[str, str]]    # pairs that survived


# ---------------------------------------------------------------------------
# Main blocker
# ---------------------------------------------------------------------------

class ControlBlocker:
    """Config-driven exact-key blocker with bucket-size gate.

    Parameters
    ----------
    config:
        BlockingConfig instance.
    """

    def __init__(self, config: BlockingConfig | None = None) -> None:
        self.config = config or BlockingConfig()
        # key_name → bucket_value → list of target_ids
        self._index: dict[str, dict[str, list[str]]] = {
            k: defaultdict(list) for k in self.config.keys
        }
        # NOTE: _target_raw is NOT populated by index_targets_from_duckdb to
        # avoid the ~13 GB memory footprint.  It is populated by index_targets()
        # only (used in tests with small synthetic data).
        self._target_raw: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_targets(self, target_records: list[dict[str, Any]]) -> None:
        """Build the blocking index from target (CH) records.

        FOR TESTS / SMALL DATA ONLY.  For 5.7 M Companies House records use
        ``index_targets_from_duckdb()`` to avoid loading ~13 GB into RAM.

        Parameters
        ----------
        target_records:
            Each dict must have keys: record_id, raw_fields (JSON string or dict).
        """
        for rec in target_records:
            tid = rec["record_id"]
            raw = rec["raw_fields"]
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            self._target_raw[tid] = raw

            keys = _target_keys(raw)
            for key_name in self.config.keys:
                val = keys.get(key_name, "")
                if val:
                    self._index[key_name][val].append(tid)

    def index_targets_from_duckdb(
        self,
        ch_csv_path: str,
        batch_size: int = 200_000,
        verbose: bool = True,
    ) -> None:
        """Build the blocking index by streaming CompanyName+PostCode from DuckDB.

        Only extracts the two columns needed to compute blocking keys
        (CompanyName, RegAddress.PostCode) plus the normalised CompanyNumber
        used as record_id.  Never loads the full 55-column JSON blob, keeping
        peak RAM under ~1 GB even for 5.7 M records.

        Parameters
        ----------
        ch_csv_path:
            Path to the raw Companies House CSV.
        batch_size:
            Number of rows to fetch per DuckDB batch.
        verbose:
            Print progress every 1 M rows.
        """
        import duckdb as _ddb
        con = _ddb.connect()
        pos_path = str(ch_csv_path).replace("\\", "/")

        total = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{pos_path}', ignore_errors=true)"
        ).fetchone()[0]
        if verbose:
            print(f"  Indexing {total:,} CH records from DuckDB (batch={batch_size:,})...",
                  flush=True)

        offset = 0
        n_indexed = 0
        while True:
            rows = con.execute(
                f"SELECT \"CompanyNumber\", \"CompanyName\", \"RegAddress.PostCode\" "
                f"FROM read_csv_auto('{pos_path}', ignore_errors=true) "
                f"LIMIT {batch_size} OFFSET {offset}"
            ).fetchall()
            if not rows:
                break
            for raw_number, company_name, postcode in rows:
                from datasets.adapters.gleif_ch import _norm_ch_number  # local import
                norm_number = _norm_ch_number(str(raw_number) if raw_number else "")
                if not norm_number:
                    continue
                tid = "CH_" + norm_number
                raw_fields = {
                    "CompanyName": company_name or "",
                    "RegAddress.PostCode": postcode or "",
                }
                keys = _target_keys(raw_fields)
                for key_name in self.config.keys:
                    val = keys.get(key_name, "")
                    if val:
                        self._index[key_name][val].append(tid)
                n_indexed += 1
            offset += batch_size
            if verbose and offset % 1_000_000 < batch_size:
                print(f"  ... {offset:,} / {total:,} rows processed", flush=True)

        if verbose:
            print(f"  Index built: {n_indexed:,} CH records indexed.", flush=True)

    def with_gate(self, gate_size: int | None) -> "ControlBlocker":
        """Return a new ControlBlocker sharing the same index but a different gate.

        This lets you build the index once and sweep gate sizes cheaply.
        The index data structures are shared (not copied) — do not modify them.

        Parameters
        ----------
        gate_size:
            The new gate size (None = no gate).
        """
        new_config = BlockingConfig(gate_size=gate_size, keys=list(self.config.keys))
        sibling = ControlBlocker(new_config)
        # Share the index (read-only from here on)
        sibling._index = self._index
        sibling._target_raw = self._target_raw
        return sibling

    def index_stats(self) -> dict[str, Any]:
        """Return summary statistics about the built index."""
        stats: dict[str, Any] = {}
        for key_name in self.config.keys:
            buckets = self._index[key_name]
            sizes = [len(v) for v in buckets.values()]
            if not sizes:
                stats[key_name] = {"n_buckets": 0}
                continue
            sizes_sorted = sorted(sizes)
            n_gated = sum(
                1 for s in sizes_sorted
                if self.config.gate_size is not None and s > self.config.gate_size
            )
            stats[key_name] = {
                "n_buckets": len(sizes),
                "n_gated_buckets": n_gated,
                "bucket_size_min": sizes_sorted[0],
                "bucket_size_p50": _percentile(sizes_sorted, 50),
                "bucket_size_p95": _percentile(sizes_sorted, 95),
                "bucket_size_max": sizes_sorted[-1],
            }
        return stats

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_records: list[dict[str, Any]],
        dataset_id: str = "gleif_ch",
    ) -> BlockingResult:
        """Retrieve candidates for all query records.

        For each query, unions results across all active keys (respecting
        the gate), deduplicates by (query_id, cand_id), and records the
        channel_mask bitmask.

        Parameters
        ----------
        query_records:
            Each dict must have: record_id, raw_fields (JSON string or dict).
        dataset_id:
            Stored in each candidate row.

        Returns
        -------
        BlockingResult
        """
        # Per-key stats accumulators
        key_stats: dict[str, dict[str, int]] = {
            k: {
                "n_queries_with_any_hit": 0,
                "n_pairs_before_gate": 0,
                "n_pairs_gated": 0,
                "n_pairs_contributed": 0,
            }
            for k in self.config.keys
        }

        # (query_id, cand_id) → channel_mask  (uint32)
        pair_mask: dict[tuple[str, str], int] = {}
        size_gated_pairs: set[tuple[str, str]] = set()
        # Per-key: which pairs did this key uniquely produce?
        key_pairs: dict[str, set[tuple[str, str]]] = {k: set() for k in self.config.keys}

        for rec in query_records:
            qid = rec["record_id"]
            raw = rec["raw_fields"]
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}

            q_keys = _query_keys(raw)

            for key_name in self.config.keys:
                bit = KEY_BIT[key_name]
                val = q_keys.get(key_name, "")
                if not val:
                    continue

                bucket = self._index[key_name].get(val, [])
                if not bucket:
                    continue

                key_stats[key_name]["n_pairs_before_gate"] += len(bucket)

                if (
                    self.config.gate_size is not None
                    and len(bucket) > self.config.gate_size
                ):
                    # Gate: drop the whole bucket
                    key_stats[key_name]["n_pairs_gated"] += len(bucket)
                    for tid in bucket:
                        size_gated_pairs.add((qid, tid))
                    continue

                key_stats[key_name]["n_queries_with_any_hit"] += 1
                for tid in bucket:
                    pair = (qid, tid)
                    pair_mask[pair] = pair_mask.get(pair, 0) | bit
                    key_stats[key_name]["n_pairs_contributed"] += 1
                    key_pairs[key_name].add(pair)

        # Remove size-gated pairs that were also retrieved via another key
        # (size_gated = gated in ALL keys it appeared in, not any)
        actual_gated = size_gated_pairs - set(pair_mask.keys())

        # Assign ranks per query (arbitrary but deterministic: sorted by cand_id)
        # and build candidate rows
        from collections import defaultdict as _dd
        query_cands: dict[str, list[tuple[str, int]]] = _dd(list)
        for (qid, tid), mask in pair_mask.items():
            query_cands[qid].append((tid, mask))

        candidates: list[dict[str, Any]] = []
        for qid, cands in query_cands.items():
            # Sort by cand_id for determinism
            cands_sorted = sorted(cands, key=lambda x: x[0])
            for rank, (tid, mask) in enumerate(cands_sorted, start=1):
                candidates.append(
                    {
                        "query_id": qid,
                        "cand_id": tid,
                        "channel_mask": mask,
                        "score": float(bin(mask).count("1")),  # key-count as score
                        "rank": rank,
                        "reverse_rank": 0,
                        "round": 1,
                        "dataset_id": dataset_id,
                    }
                )

        # Per-key unique contribution (pairs ONLY from that key)
        per_key_unique: dict[str, int] = {}
        for k in self.config.keys:
            other_keys_pairs: set[tuple[str, str]] = set()
            for k2 in self.config.keys:
                if k2 != k:
                    other_keys_pairs |= key_pairs[k2]
            per_key_unique[k] = len(key_pairs[k] - other_keys_pairs)

        total_before_gate = sum(s["n_pairs_before_gate"] for s in key_stats.values())
        total_gated = sum(s["n_pairs_gated"] for s in key_stats.values())

        stats: dict[str, Any] = {
            "gate_size": self.config.gate_size,
            "n_queries": len(query_records),
            "n_candidate_pairs": len(pair_mask),
            "n_size_gated_pairs": len(actual_gated),
            "total_pairs_before_gate": total_before_gate,
            "per_key": {
                k: {
                    **key_stats[k],
                    "n_unique_contribution": per_key_unique[k],
                    "n_pairs_in_key": len(key_pairs[k]),
                }
                for k in self.config.keys
            },
        }

        return BlockingResult(
            candidates=candidates,
            stats=stats,
            size_gated_pairs=actual_gated,
            retrieved_pairs=set(pair_mask.keys()),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[int], pct: float) -> float:
    if not sorted_data:
        return float("nan")
    idx = (len(sorted_data) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] * (1 - (idx - lo)) + sorted_data[hi] * (idx - lo)
