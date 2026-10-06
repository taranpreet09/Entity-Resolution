"""
Tests for experiment reproducibility
======================================
Verifies that running the sanity experiment twice with the same seed
produces identical metrics.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _metrics_equal(m1: dict, m2: dict) -> bool:
    """Compare two metrics dicts, treating NaN == NaN as equal."""
    if set(m1.keys()) != set(m2.keys()):
        return False
    for k in m1:
        v1, v2 = m1[k], m2[k]
        # Both NaN
        if isinstance(v1, float) and isinstance(v2, float):
            if math.isnan(v1) and math.isnan(v2):
                continue
            if not math.isclose(v1, v2, rel_tol=1e-9):
                return False
        elif v1 != v2:
            return False
    return True


class TestSanityReproducibility:
    """Running the sanity experiment twice must give identical metrics."""

    def test_identical_metrics_on_two_runs(self, tmp_path):
        """
        Import and run the experiment function twice with the same seed.
        The returned metrics dicts must be identical (excluding run_id and
        timestamp, which are not part of the metrics dict).
        NaN values in both dicts are treated as equal.
        """
        import experiments.sanity_exact_name as sanity

        # Patch both the module-level LEDGER_PATH and the run_dir base
        original_ledger_path = sanity.LEDGER_PATH
        temp_ledger = tmp_path / "runs.parquet"
        sanity.LEDGER_PATH = temp_ledger

        try:
            metrics1 = sanity.run_experiment(seed=42, k=10)
            metrics2 = sanity.run_experiment(seed=42, k=10)
        finally:
            sanity.LEDGER_PATH = original_ledger_path

        # Remove run-specific / non-deterministic keys before comparison
        EXCLUDE = {"note"}
        m1 = {k: v for k, v in metrics1.items() if k not in EXCLUDE}
        m2 = {k: v for k, v in metrics2.items() if k not in EXCLUDE}

        assert _metrics_equal(m1, m2), (
            f"Metrics differ between runs:\n"
            f"Run 1: {m1}\n"
            f"Run 2: {m2}"
        )

    def test_different_seed_same_core_metrics(self, tmp_path):
        """Core metrics (recall, zero_cand_rate) must be equal across seeds
        because they depend only on the candidates, not the bootstrap seed."""
        import experiments.sanity_exact_name as sanity

        original_ledger_path = sanity.LEDGER_PATH
        temp_ledger = tmp_path / "runs2.parquet"
        sanity.LEDGER_PATH = temp_ledger

        try:
            m1 = sanity.run_experiment(seed=42, k=10)
            m2 = sanity.run_experiment(seed=99, k=10)
        finally:
            sanity.LEDGER_PATH = original_ledger_path

        # Core metrics do not depend on seed
        def eq(a, b):
            if a is None and b is None:
                return True
            if isinstance(a, float) and isinstance(b, float):
                return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b)
            return a == b

        assert eq(m1["recall@10"], m2["recall@10"]), "recall@10 differs across seeds"
        assert eq(m1["zero_candidate_rate"], m2["zero_candidate_rate"])
        assert eq(m1["n_candidates"], m2["n_candidates"])
        assert eq(m1["n_queries_with_targets"], m2["n_queries_with_targets"])
