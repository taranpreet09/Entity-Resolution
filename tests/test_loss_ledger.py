"""
Tests for ber.eval.loss_ledger.LossLedger
==========================================
Hand-built toy cases covering every stage, including all sub-types.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber.eval.loss_ledger import LossLedger, LossStage, RETRIEVAL_SUBTYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ledger(pairs: list[tuple[str, str]]) -> LossLedger:
    ll = LossLedger()
    ll.add_true_pairs(pairs)
    return ll


# ---------------------------------------------------------------------------
# Stage assignment tests
# ---------------------------------------------------------------------------

class TestStageCoverage:
    """One hand-built toy for every distinct stage / sub-type."""

    def test_input_loss(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_input_loss("q1", "c1", reason="name field empty")
        counts = ll.stage_counts()
        assert counts["INPUT"] == 1
        assert counts["RETRIEVAL"] == 0

    def test_canon_loss(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_canon_loss("q1", "c1", reason="canonical name empty after stripping")
        counts = ll.stage_counts()
        assert counts["CANONICALIZATION"] == 1

    def test_retrieval_loss_no_subtype(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_retrieval_loss("q1", "c1")
        assert ll.stage_counts()["RETRIEVAL"] == 1

    def test_retrieval_loss_beyond_k(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_retrieval_loss("q1", "c1", subtype="beyond_k", k=10)
        rec = ll._pairs[("q1", "c1")]
        assert rec.loss_stage == LossStage.RETRIEVAL
        assert rec.subtype == "beyond_k"
        assert rec.extra["k"] == 10

    def test_retrieval_loss_below_floor(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_retrieval_loss("q1", "c1", subtype="below_floor", score=0.05)
        rec = ll._pairs[("q1", "c1")]
        assert rec.subtype == "below_floor"
        assert rec.extra["score"] == pytest.approx(0.05)

    def test_retrieval_loss_size_gated(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_retrieval_loss("q1", "c1", subtype="size_gated")
        assert ll._pairs[("q1", "c1")].subtype == "size_gated"

    def test_budget_prune_loss(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_budget_prune_loss("q1", "c1")
        assert ll.stage_counts()["BUDGET_PRUNE"] == 1

    def test_features_loss_stub(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_features_loss("q1", "c1")
        assert ll.stage_counts()["FEATURES"] == 1

    def test_ranker_loss_stub(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_ranker_loss("q1", "c1", score=0.3, threshold=0.5)
        rec = ll._pairs[("q1", "c1")]
        assert rec.loss_stage == LossStage.RANKER
        assert rec.extra["score"] == pytest.approx(0.3)

    def test_decision_loss_stub(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_decision_loss("q1", "c1")
        assert ll.stage_counts()["DECISION"] == 1

    def test_recovered(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_recovered("q1", "c1")
        assert ll.stage_counts()["NONE"] == 1
        assert ll.stage_counts()["RETRIEVAL"] == 0


class TestEarliestStageWins:
    """Confirm that the earliest stage always wins, not the last call."""

    def test_input_beats_retrieval(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_retrieval_loss("q1", "c1")
        ll.mark_input_loss("q1", "c1")   # earlier stage → should win
        assert ll._pairs[("q1", "c1")].loss_stage == LossStage.INPUT

    def test_retrieval_does_not_overwrite_input(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_input_loss("q1", "c1")
        ll.mark_retrieval_loss("q1", "c1")  # later stage → should NOT win
        assert ll._pairs[("q1", "c1")].loss_stage == LossStage.INPUT

    def test_budget_prune_does_not_overwrite_canon(self):
        ll = _make_ledger([("q1", "c1")])
        ll.mark_canon_loss("q1", "c1")
        ll.mark_budget_prune_loss("q1", "c1")
        assert ll._pairs[("q1", "c1")].loss_stage == LossStage.CANONICALIZATION

    def test_decision_beats_ranker_if_called_first(self):
        # RANKER < DECISION in the taxonomy; RANKER should win
        ll = _make_ledger([("q1", "c1")])
        ll.mark_decision_loss("q1", "c1")
        ll.mark_ranker_loss("q1", "c1")
        assert ll._pairs[("q1", "c1")].loss_stage == LossStage.RANKER


class TestBulkApplication:

    def test_apply_candidates_marks_unretrieved(self):
        pairs = [("q1", "c1"), ("q2", "c2"), ("q3", "c3")]
        ll = _make_ledger(pairs)
        retrieved = {("q1", "c1"), ("q2", "c2")}  # q3/c3 not retrieved
        ll.apply_candidates(retrieved)
        # q3 should still be RETRIEVAL (default)
        assert ll._pairs[("q3", "c3")].loss_stage == LossStage.RETRIEVAL
        # q1, q2 are still RETRIEVAL until explicitly recovered
        assert ll._pairs[("q1", "c1")].loss_stage == LossStage.RETRIEVAL

    def test_apply_budget_prune(self):
        ll = _make_ledger([("q1", "c1"), ("q2", "c2")])
        # q1/c1 was retrieved (RETRIEVAL stage), then pruned
        ll.apply_budget_prune({("q1", "c1")})
        assert ll._pairs[("q1", "c1")].loss_stage == LossStage.BUDGET_PRUNE
        assert ll._pairs[("q2", "c2")].loss_stage == LossStage.RETRIEVAL


class TestReporting:

    def test_stage_counts_all_zeros_initially_except_default(self):
        ll = LossLedger()
        ll.add_true_pairs([("q1", "c1"), ("q2", "c2")])
        counts = ll.stage_counts()
        assert counts["RETRIEVAL"] == 2  # default
        assert counts["INPUT"] == 0

    def test_to_csv_creates_file(self, tmp_path):
        ll = _make_ledger([("q1", "c1"), ("q2", "c2")])
        ll.mark_input_loss("q1", "c1", reason="missing")
        out = tmp_path / "ledger.csv"
        ll.to_csv(out)
        assert out.exists()
        lines = out.read_text().splitlines()
        assert lines[0].startswith("query_id,cand_id,loss_stage")
        assert len(lines) == 3  # header + 2 rows

    def test_to_stage_table_has_all_stages(self):
        ll = _make_ledger([("q1", "c1")])
        table = ll.to_stage_table()
        stage_names = {row["stage"] for row in table}
        for s in LossStage:
            assert s.value in stage_names

    def test_len(self):
        ll = _make_ledger([("q1", "c1"), ("q2", "c2"), ("q3", "c3")])
        assert len(ll) == 3

    def test_invalid_retrieval_subtype_raises(self):
        ll = _make_ledger([("q1", "c1")])
        with pytest.raises(ValueError, match="Invalid retrieval subtype"):
            ll.mark_retrieval_loss("q1", "c1", subtype="bogus_type")
