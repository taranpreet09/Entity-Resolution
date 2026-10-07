"""
ber.eval.metrics_dedup — Additional Metrics for Single-Table Deduplication
===========================================================================
Supplements ber.eval.metrics for the deduplication setting where:
  - Queries and targets are from the same table
  - True pairs are unordered (a,b) = (b,a)
  - Clusters can have >2 members (partial recall is meaningful)
  - Singleton records (no duplicate) generate false-positive candidates
    ("singleton burden")

All standard metrics (recall@K, zero_candidate_rate, etc.) remain in
ber.eval.metrics and apply here with query-side entity IDs as the unit.

New metrics defined here:

pair_recall(retrieved_pairs, true_pairs) -> float
    Fraction of true UNORDERED pairs in the retrieved set.
    retrieved_pairs is treated as ordered (query→cand); both (a,b) and
    (b,a) count as retrieving the pair {a,b}.

singleton_burden(candidate_map, singleton_ids) -> dict
    Candidates generated for singleton queries (queries with no true
    duplicate).  These are guaranteed false positives at the retrieval
    stage.  Returns total count and per-query distribution.
"""

from __future__ import annotations

import statistics
from typing import Any

from ber.eval.metrics import CandidateMap, TrueTargetsMap


def pair_recall(
    retrieved_pairs: set[tuple[str, str]],
    true_pairs: list[tuple[str, str]],
) -> float:
    """Fraction of true UNORDERED pairs recovered by retrieval.

    A true pair {a, b} is recovered if either (a, b) or (b, a) is in
    ``retrieved_pairs``.

    Parameters
    ----------
    retrieved_pairs:
        Set of (query_id, cand_id) tuples as produced by the blocker.
    true_pairs:
        List of canonical true pairs.  Each pair is stored once (a < b
        lexicographically in the adapter), but both orderings are checked.

    Returns
    -------
    float
        Pair recall in [0, 1].  Returns NaN if true_pairs is empty.
    """
    if not true_pairs:
        return float("nan")
    # Normalise: canonical form is min_id < max_id
    true_set = {(min(a, b), max(a, b)) for a, b in true_pairs}
    retr_normalised = {(min(a, b), max(a, b)) for a, b in retrieved_pairs}
    return len(true_set & retr_normalised) / len(true_set)


def singleton_burden(
    candidate_map: CandidateMap,
    singleton_ids: set[str],
) -> dict[str, Any]:
    """Candidates generated for singleton queries.

    Singletons have no true duplicate — any candidate they retrieve is a
    false positive at the retrieval stage.

    Parameters
    ----------
    candidate_map:
        Standard CandidateMap from build_candidate_map().
    singleton_ids:
        Set of record IDs that are singletons (no true duplicate).

    Returns
    -------
    dict with keys:
        n_singleton_queries_with_candidates: int
        total_singleton_candidates: int
        cpq_singleton_median: float
        cpq_singleton_p95: float
        cpq_singleton_mean: float
    """
    counts = sorted(
        len(candidate_map.get(sid, []))
        for sid in singleton_ids
        if candidate_map.get(sid)
    )
    if not counts:
        return {
            "n_singleton_queries_with_candidates": 0,
            "total_singleton_candidates": 0,
            "cpq_singleton_median": 0.0,
            "cpq_singleton_p95": 0.0,
            "cpq_singleton_mean": 0.0,
        }

    def pct(data, p):
        idx = (len(data) - 1) * p / 100.0
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        return data[lo] * (1 - (idx - lo)) + data[hi] * (idx - lo)

    return {
        "n_singleton_queries_with_candidates": len(counts),
        "total_singleton_candidates": sum(counts),
        "cpq_singleton_median": round(pct(counts, 50), 2),
        "cpq_singleton_p95": round(pct(counts, 95), 2),
        "cpq_singleton_mean": round(statistics.mean(counts), 2),
    }
