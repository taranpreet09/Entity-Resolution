"""
ber.eval.metrics — Retrieval Metrics with Bootstrap Confidence Intervals
=========================================================================
All metrics are computed over ENTITIES (unique query-side entity IDs), not
over individual pairs.  This avoids inflating confidence by counting the same
entity multiple times when it has many true targets.

Metrics
-------
recall_at_k(candidates, labels, k) -> float
    Fraction of queries with at least one true target in the top-k candidates.

recall_at_budget(candidates, labels, budget) -> float
    Fraction of queries with at least one true target within `budget` total
    retrieved candidates (across all channels).

zero_candidate_rate(candidates, labels) -> float
    Fraction of queries that have at least one true target but retrieved
    zero candidates at all.  A query is "zero-candidate" when its query_id
    does not appear in the candidates table at all.

partial_recall_rate(candidates, labels, k) -> float
    Fraction of queries for which SOME but not ALL true targets appear in
    the top-k candidates.

candidates_per_query_distribution(candidates) -> dict
    Summary statistics (min, p25, median, p75, p95, max, mean) of the
    number of candidates retrieved per query.

bootstrap_ci(
    candidates, labels, metric_fn, n_boot, k, seed
) -> (point_estimate, lower_ci, upper_ci)
    Paired bootstrap over entities.  Resamples query-side entities with
    replacement and recomputes the metric.

compare_systems(
    cands_a, cands_b, labels, metric_fn, n_boot, k, seed
) -> dict
    Compare two systems on the same entities with a paired bootstrap test.
    Returns point estimates for both, the difference, and a two-sided p-value.

Definitions
-----------
* A **query** is a unique query_id in the candidates table OR in the labels
  table (if a query has true targets but retrieved zero candidates, it still
  counts in the denominator).

* **Zero-candidate rate**: among queries that have at least one true target,
  the fraction for which no candidates at all were retrieved.  Formally:

      zero_cand_rate = |{q : true_targets(q) ≠ ∅ ∧ retrieved(q) = ∅}|
                       ─────────────────────────────────────────────────
                            |{q : true_targets(q) ≠ ∅}|

* **recall@K**: for each query q that has true targets,
      hit(q, K) = 1  if  |top_K_candidates(q) ∩ true_targets(q)| ≥ 1
                 0  otherwise
      recall@K = mean(hit(q, K))   over all q with true_targets(q) ≠ ∅

* **recall@budget**: same as recall@K but `budget` is the total number of
  candidates across ALL queries divided by the number of queries — i.e. the
  per-query candidate cap is `budget`.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Maps query_id -> list of (cand_id, rank, score)
CandidateMap = dict[str, list[tuple[str, int, float]]]
# Maps query_id -> set of true cand_ids
TrueTargetsMap = dict[str, set[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_candidate_map(
    candidates: list[dict[str, Any]],
) -> CandidateMap:
    """Build a {query_id: [(cand_id, rank, score), ...]} map from a flat list.

    Parameters
    ----------
    candidates:
        Each dict must have keys: query_id, cand_id, rank, score.
    """
    cm: CandidateMap = {}
    for c in candidates:
        qid = c["query_id"]
        if qid not in cm:
            cm[qid] = []
        cm[qid].append((c["cand_id"], int(c["rank"]), float(c["score"])))
    # Sort by rank within each query
    for qid in cm:
        cm[qid].sort(key=lambda x: x[1])
    return cm


def build_true_targets_map(
    labels: list[dict[str, Any]],
    query_dataset_id: str | None = None,
    cand_dataset_id: str | None = None,
) -> TrueTargetsMap:
    """Build a {query_id: {cand_id, ...}} map from pair labels.

    Parameters
    ----------
    labels:
        Each dict must have keys: query_id, cand_id, label.
        Only rows with label == 1 are treated as true targets.
    query_dataset_id, cand_dataset_id:
        Optional dataset filters.
    """
    tm: TrueTargetsMap = {}
    for row in labels:
        if int(row.get("label", 1)) != 1:
            continue
        qid = row["query_id"]
        cid = row["cand_id"]
        if qid not in tm:
            tm[qid] = set()
        tm[qid].add(cid)
    return tm


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(
    candidate_map: CandidateMap,
    true_targets: TrueTargetsMap,
    k: int,
) -> float:
    """Recall@K: fraction of queries with a true target in the top-k candidates.

    Only queries that have at least one true target count in the denominator.
    """
    queries_with_targets = [q for q in true_targets if true_targets[q]]
    if not queries_with_targets:
        return float("nan")

    hits = 0
    for q in queries_with_targets:
        top_k_cands = {c for c, r, _ in candidate_map.get(q, []) if r <= k}
        if top_k_cands & true_targets[q]:
            hits += 1
    return hits / len(queries_with_targets)


def recall_at_budget(
    candidate_map: CandidateMap,
    true_targets: TrueTargetsMap,
    budget: int,
) -> float:
    """Recall@budget: same as recall@K with K = budget (per-query cap)."""
    return recall_at_k(candidate_map, true_targets, k=budget)


def zero_candidate_rate(
    candidate_map: CandidateMap,
    true_targets: TrueTargetsMap,
) -> float:
    """Fraction of queries with true targets that retrieved zero candidates.

    See module docstring for the precise definition.
    """
    queries_with_targets = [q for q in true_targets if true_targets[q]]
    if not queries_with_targets:
        return float("nan")

    zero_cand = sum(
        1 for q in queries_with_targets if not candidate_map.get(q)
    )
    return zero_cand / len(queries_with_targets)


def partial_recall_rate(
    candidate_map: CandidateMap,
    true_targets: TrueTargetsMap,
    k: int,
) -> float:
    """Fraction of queries with SOME but not ALL true targets in top-k.

    Only queries with ≥2 true targets can be partial; queries with exactly
    one true target are either full-hits or full-misses.
    """
    queries_with_targets = [q for q in true_targets if true_targets[q]]
    if not queries_with_targets:
        return float("nan")

    partial = 0
    for q in queries_with_targets:
        top_k_cands = {c for c, r, _ in candidate_map.get(q, []) if r <= k}
        n_hit = len(top_k_cands & true_targets[q])
        n_total = len(true_targets[q])
        if 0 < n_hit < n_total:
            partial += 1
    return partial / len(queries_with_targets)


def candidates_per_query_distribution(
    candidate_map: CandidateMap,
) -> dict[str, float]:
    """Summary statistics for number of candidates retrieved per query."""
    counts = sorted(len(v) for v in candidate_map.values())
    if not counts:
        return {k: float("nan") for k in ("min", "p25", "median", "p75", "p95", "max", "mean")}

    def percentile(data: list[int], pct: float) -> float:
        if not data:
            return float("nan")
        idx = (len(data) - 1) * pct / 100
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        frac = idx - lo
        return data[lo] * (1 - frac) + data[hi] * frac

    return {
        "min": float(counts[0]),
        "p25": percentile(counts, 25),
        "median": percentile(counts, 50),
        "p75": percentile(counts, 75),
        "p95": percentile(counts, 95),
        "max": float(counts[-1]),
        "mean": statistics.mean(counts),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    candidate_map: CandidateMap,
    true_targets: TrueTargetsMap,
    metric_fn: Callable[[CandidateMap, TrueTargetsMap], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap confidence interval over entities (query IDs).

    Parameters
    ----------
    candidate_map, true_targets:
        As built by :func:`build_candidate_map` / :func:`build_true_targets_map`.
    metric_fn:
        A function ``(CandidateMap, TrueTargetsMap) -> float``.
        Use ``functools.partial`` to bind extra arguments (e.g. k).
    n_boot:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (point_estimate, lower_95_ci, upper_95_ci)
    """
    all_query_ids = sorted(
        {q for q in true_targets if true_targets[q]}
    )
    if not all_query_ids:
        nan = float("nan")
        return nan, nan, nan

    point_estimate = metric_fn(candidate_map, true_targets)

    rng = random.Random(seed)
    boot_scores: list[float] = []

    for _ in range(n_boot):
        sample_qids = [rng.choice(all_query_ids) for _ in all_query_ids]
        boot_cm = {q: candidate_map.get(q, []) for q in sample_qids}
        boot_tt = {q: true_targets[q] for q in sample_qids}
        score = metric_fn(boot_cm, boot_tt)
        if score == score:  # skip NaN
            boot_scores.append(score)

    if not boot_scores:
        nan = float("nan")
        return point_estimate, nan, nan

    boot_scores.sort()
    lo_idx = max(0, int(0.025 * len(boot_scores)))
    hi_idx = min(len(boot_scores) - 1, int(0.975 * len(boot_scores)))
    return point_estimate, boot_scores[lo_idx], boot_scores[hi_idx]


def compare_systems(
    cands_a: CandidateMap,
    cands_b: CandidateMap,
    true_targets: TrueTargetsMap,
    metric_fn: Callable[[CandidateMap, TrueTargetsMap], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two retrieval systems on the same entities with paired bootstrap.

    Computes the metric for system A and B, then resamples (with replacement)
    the shared query-entity population to build a null distribution of
    A_score − B_score.  The two-sided p-value is the fraction of resamples
    where |delta| ≥ |observed delta|.

    Parameters
    ----------
    cands_a, cands_b:
        Candidate maps for the two systems.  Must cover the same query IDs.
    true_targets:
        Shared ground truth.
    metric_fn:
        Metric function taking (CandidateMap, TrueTargetsMap) -> float.
    n_boot:
        Number of bootstrap resamples.
    seed:
        Random seed.

    Returns
    -------
    dict with keys:
        score_a, score_b, delta (a - b), p_value, n_queries, n_boot
    """
    all_query_ids = sorted(
        {q for q in true_targets if true_targets[q]}
    )
    if not all_query_ids:
        return {
            "score_a": float("nan"), "score_b": float("nan"),
            "delta": float("nan"), "p_value": float("nan"),
            "n_queries": 0, "n_boot": n_boot,
        }

    score_a = metric_fn(cands_a, true_targets)
    score_b = metric_fn(cands_b, true_targets)
    observed_delta = score_a - score_b

    rng = random.Random(seed)
    extreme_count = 0

    for _ in range(n_boot):
        sample_qids = [rng.choice(all_query_ids) for _ in all_query_ids]
        boot_tt = {q: true_targets[q] for q in sample_qids}
        boot_cm_a = {q: cands_a.get(q, []) for q in sample_qids}
        boot_cm_b = {q: cands_b.get(q, []) for q in sample_qids}
        sa = metric_fn(boot_cm_a, boot_tt)
        sb = metric_fn(boot_cm_b, boot_tt)
        if (sa - sb) != (sa - sb):  # NaN check
            continue
        # Centre around observed delta for the null distribution
        boot_delta = (sa - sb) - observed_delta
        if abs(boot_delta) >= abs(observed_delta):
            extreme_count += 1

    p_value = extreme_count / n_boot

    return {
        "score_a": score_a,
        "score_b": score_b,
        "delta": observed_delta,
        "p_value": p_value,
        "n_queries": len(all_query_ids),
        "n_boot": n_boot,
    }
