"""
Tests for ber.eval.metrics
===========================
Covers all metric functions and bootstrap utilities.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber.eval.metrics import (
    build_candidate_map,
    build_true_targets_map,
    bootstrap_ci,
    candidates_per_query_distribution,
    compare_systems,
    partial_recall_rate,
    recall_at_budget,
    recall_at_k,
    zero_candidate_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_scenario():
    """
    3 queries, each with 1 true target.
    q1 → c1 retrieved at rank 1 (hit)
    q2 → c2 retrieved at rank 5 (hit within k=10, miss within k=3)
    q3 → c3 NOT retrieved (miss)
    """
    candidates = [
        {"query_id": "q1", "cand_id": "c1", "rank": 1, "score": 1.0},
        {"query_id": "q1", "cand_id": "cx", "rank": 2, "score": 0.5},
        {"query_id": "q2", "cand_id": "cy", "rank": 1, "score": 0.9},
        {"query_id": "q2", "cand_id": "c2", "rank": 5, "score": 0.6},
    ]
    labels = [
        {"query_id": "q1", "cand_id": "c1", "label": 1},
        {"query_id": "q2", "cand_id": "c2", "label": 1},
        {"query_id": "q3", "cand_id": "c3", "label": 1},
    ]
    return (
        build_candidate_map(candidates),
        build_true_targets_map(labels),
    )


def _multi_target_scenario():
    """
    q1 has 3 true targets: c1 (retrieved rank 1), c2 (retrieved rank 2),
    c3 NOT retrieved → partial recall.
    """
    candidates = [
        {"query_id": "q1", "cand_id": "c1", "rank": 1, "score": 1.0},
        {"query_id": "q1", "cand_id": "c2", "rank": 2, "score": 0.9},
    ]
    labels = [
        {"query_id": "q1", "cand_id": "c1", "label": 1},
        {"query_id": "q1", "cand_id": "c2", "label": 1},
        {"query_id": "q1", "cand_id": "c3", "label": 1},
    ]
    return (
        build_candidate_map(candidates),
        build_true_targets_map(labels),
    )


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:

    def test_perfect_recall(self):
        cm = build_candidate_map([
            {"query_id": "q1", "cand_id": "c1", "rank": 1, "score": 1.0},
            {"query_id": "q2", "cand_id": "c2", "rank": 1, "score": 1.0},
        ])
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 1},
            {"query_id": "q2", "cand_id": "c2", "label": 1},
        ])
        assert recall_at_k(cm, tt, k=1) == pytest.approx(1.0)

    def test_zero_recall(self):
        cm = build_candidate_map([])
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 1},
        ])
        assert recall_at_k(cm, tt, k=10) == pytest.approx(0.0)

    def test_partial_recall(self):
        cm, tt = _simple_scenario()
        # q1 hit (rank 1 ≤ 10), q2 hit (rank 5 ≤ 10), q3 miss → 2/3
        assert recall_at_k(cm, tt, k=10) == pytest.approx(2 / 3)

    def test_k_cutoff_matters(self):
        cm, tt = _simple_scenario()
        # k=3: q1 hit (rank 1), q2 miss (rank 5 > 3), q3 miss → 1/3
        assert recall_at_k(cm, tt, k=3) == pytest.approx(1 / 3)

    def test_no_targets_returns_nan(self):
        cm = build_candidate_map([])
        tt = build_true_targets_map([])
        result = recall_at_k(cm, tt, k=10)
        assert math.isnan(result)

    def test_non_match_labels_excluded(self):
        cm = build_candidate_map([
            {"query_id": "q1", "cand_id": "c1", "rank": 1, "score": 1.0},
        ])
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 0},  # non-match
        ])
        # After filtering label==0, no true targets → NaN
        assert math.isnan(recall_at_k(cm, tt, k=10))


# ---------------------------------------------------------------------------
# zero_candidate_rate
# ---------------------------------------------------------------------------

class TestZeroCandidateRate:

    def test_no_zero_candidates(self):
        cm = build_candidate_map([
            {"query_id": "q1", "cand_id": "c1", "rank": 1, "score": 1.0},
            {"query_id": "q2", "cand_id": "c2", "rank": 1, "score": 1.0},
        ])
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 1},
            {"query_id": "q2", "cand_id": "c2", "label": 1},
        ])
        assert zero_candidate_rate(cm, tt) == pytest.approx(0.0)

    def test_all_zero_candidates(self):
        cm = build_candidate_map([])
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 1},
            {"query_id": "q2", "cand_id": "c2", "label": 1},
        ])
        assert zero_candidate_rate(cm, tt) == pytest.approx(1.0)

    def test_one_third_zero(self):
        cm, tt = _simple_scenario()
        # q3 retrieved nothing → 1/3
        assert zero_candidate_rate(cm, tt) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# partial_recall_rate
# ---------------------------------------------------------------------------

class TestPartialRecallRate:

    def test_partial_hit(self):
        cm, tt = _multi_target_scenario()
        # q1: 2/3 targets retrieved → partial
        rate = partial_recall_rate(cm, tt, k=10)
        assert rate == pytest.approx(1.0)

    def test_no_partial_when_all_hit(self):
        cm = build_candidate_map([
            {"query_id": "q1", "cand_id": "c1", "rank": 1, "score": 1.0},
            {"query_id": "q1", "cand_id": "c2", "rank": 2, "score": 0.9},
        ])
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 1},
            {"query_id": "q1", "cand_id": "c2", "label": 1},
        ])
        assert partial_recall_rate(cm, tt, k=10) == pytest.approx(0.0)

    def test_no_partial_when_all_miss(self):
        cm = build_candidate_map([])
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 1},
            {"query_id": "q1", "cand_id": "c2", "label": 1},
        ])
        assert partial_recall_rate(cm, tt, k=10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# candidates_per_query_distribution
# ---------------------------------------------------------------------------

class TestCandidatesDistribution:

    def test_basic_stats(self):
        cm = build_candidate_map([
            {"query_id": "q1", "cand_id": f"c{i}", "rank": i, "score": 1.0}
            for i in range(1, 6)  # 5 candidates for q1
        ] + [
            {"query_id": "q2", "cand_id": "c1", "rank": 1, "score": 1.0},
        ])
        stats = candidates_per_query_distribution(cm)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["median"] == pytest.approx(3.0)

    def test_empty_candidates(self):
        cm = build_candidate_map([])
        stats = candidates_per_query_distribution(cm)
        assert all(math.isnan(v) for v in stats.values())


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:

    def test_ci_covers_point_estimate(self):
        cm = build_candidate_map([
            {"query_id": f"q{i}", "cand_id": f"c{i}", "rank": 1, "score": 1.0}
            for i in range(20)
        ])
        tt = build_true_targets_map([
            {"query_id": f"q{i}", "cand_id": f"c{i}", "label": 1}
            for i in range(20)
        ])
        metric_fn = functools.partial(recall_at_k, k=5)
        pt, lo, hi = bootstrap_ci(cm, tt, metric_fn, n_boot=200, seed=42)
        assert lo <= pt <= hi

    def test_ci_deterministic(self):
        cm = build_candidate_map([
            {"query_id": f"q{i}", "cand_id": f"c{i}", "rank": 1, "score": 1.0}
            for i in range(10)
        ])
        tt = build_true_targets_map([
            {"query_id": f"q{i}", "cand_id": f"c{i}", "label": 1}
            for i in range(10)
        ])
        metric_fn = functools.partial(recall_at_k, k=5)
        r1 = bootstrap_ci(cm, tt, metric_fn, n_boot=100, seed=7)
        r2 = bootstrap_ci(cm, tt, metric_fn, n_boot=100, seed=7)
        assert r1 == r2

    def test_ci_empty_targets_returns_nan(self):
        cm = build_candidate_map([])
        tt = build_true_targets_map([])
        metric_fn = functools.partial(recall_at_k, k=5)
        pt, lo, hi = bootstrap_ci(cm, tt, metric_fn, n_boot=50, seed=42)
        assert math.isnan(pt)
        assert math.isnan(lo)
        assert math.isnan(hi)


# ---------------------------------------------------------------------------
# compare_systems
# ---------------------------------------------------------------------------

class TestCompareSystems:

    def _make_cands(self, query_ids, hit_ids):
        """Create a candidate map where each query_id hits hit_ids."""
        rows = []
        for i, qid in enumerate(query_ids):
            if qid in hit_ids:
                rows.append({"query_id": qid, "cand_id": f"c_{qid}", "rank": 1, "score": 1.0})
        return build_candidate_map(rows)

    def test_identical_systems_delta_zero(self):
        qids = [f"q{i}" for i in range(20)]
        tt = build_true_targets_map([
            {"query_id": q, "cand_id": f"c_{q}", "label": 1} for q in qids
        ])
        cm = self._make_cands(qids, set(qids))
        metric_fn = functools.partial(recall_at_k, k=5)
        result = compare_systems(cm, cm, tt, metric_fn, n_boot=100, seed=42)
        assert result["delta"] == pytest.approx(0.0)

    def test_better_system_positive_delta(self):
        qids = [f"q{i}" for i in range(20)]
        tt = build_true_targets_map([
            {"query_id": q, "cand_id": f"c_{q}", "label": 1} for q in qids
        ])
        cm_a = self._make_cands(qids, set(qids))        # 100% recall
        cm_b = self._make_cands(qids, set(qids[:10]))   # 50% recall
        metric_fn = functools.partial(recall_at_k, k=5)
        result = compare_systems(cm_a, cm_b, tt, metric_fn, n_boot=200, seed=42)
        assert result["delta"] > 0
        assert result["score_a"] > result["score_b"]

    def test_returns_required_keys(self):
        qids = ["q1", "q2"]
        tt = build_true_targets_map([
            {"query_id": "q1", "cand_id": "c1", "label": 1},
            {"query_id": "q2", "cand_id": "c2", "label": 1},
        ])
        cm = build_candidate_map([
            {"query_id": "q1", "cand_id": "c1", "rank": 1, "score": 1.0},
        ])
        metric_fn = functools.partial(recall_at_k, k=5)
        result = compare_systems(cm, cm, tt, metric_fn, n_boot=50, seed=42)
        for key in ("score_a", "score_b", "delta", "p_value", "n_queries", "n_boot"):
            assert key in result
