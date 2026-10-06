"""
Tests for ber.retrieve.control_blocking.ControlBlocker
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber.retrieve.control_blocking import (
    ControlBlocker, BlockingConfig, BlockingResult,
    ALL_KEY_NAMES, KEY_BIT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_target(record_id: str, company_name: str, postcode: str) -> dict:
    return {
        "record_id": record_id,
        "raw_fields": json.dumps({
            "CompanyName": company_name,
            "RegAddress.PostCode": postcode,
        }),
    }


def _make_query(record_id: str, legal_name: str, postcode: str) -> dict:
    return {
        "record_id": record_id,
        "raw_fields": json.dumps({
            "LegalName": legal_name,
            "LegalAddr_PostCode": postcode,
        }),
    }


TARGETS = [
    _make_target("CH_01", "ACME LIMITED", "EC1A 1BB"),
    _make_target("CH_02", "BETA CORPORATION", "M1 1AE"),
    _make_target("CH_03", "ACME HOLDINGS", "EC1A 2CC"),
    _make_target("CH_04", "GAMMA SERVICES", "SW1A 2AA"),
    _make_target("CH_05", "DELTA PARTNERS", "EC1A 3DD"),
]

QUERY_ACME = _make_query("GLEIF_01", "ACME LIMITED", "EC1A 1BB")
QUERY_BETA = _make_query("GLEIF_02", "BETA CORPORATION", "M1 1AE")
QUERY_UNKNOWN = _make_query("GLEIF_03", "ZETA UNKNOWN", "ZZ99 9ZZ")


# ---------------------------------------------------------------------------
# Basic retrieval tests
# ---------------------------------------------------------------------------

class TestControlBlocker:

    def test_exact_k1_match(self):
        blocker = ControlBlocker(BlockingConfig(gate_size=None, keys=["K1"]))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        cand_ids = {c["cand_id"] for c in result.candidates}
        assert "CH_01" in cand_ids  # exact name match

    def test_k2_matches_same_outward(self):
        """K2 (first_token|outward) should match ACME queries to EC1A targets."""
        blocker = ControlBlocker(BlockingConfig(gate_size=None, keys=["K2"]))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        cand_ids = {c["cand_id"] for c in result.candidates}
        # ACME LIMITED EC1A1BB and ACME HOLDINGS EC1A2CC share K2="acme|EC1A"
        assert "CH_01" in cand_ids
        assert "CH_03" in cand_ids

    def test_no_match_query(self):
        blocker = ControlBlocker(BlockingConfig(gate_size=None))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_UNKNOWN])
        assert result.candidates == []

    def test_channel_mask_multi_key(self):
        """A pair matching K1 and K2 should have both bits set."""
        blocker = ControlBlocker(BlockingConfig(gate_size=None, keys=["K1", "K2"]))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        # CH_01 should match K1 (exact name) and K2 (same first_token+outward)
        pair = next(c for c in result.candidates if c["cand_id"] == "CH_01")
        expected_mask = KEY_BIT["K1"] | KEY_BIT["K2"]
        assert pair["channel_mask"] == expected_mask

    def test_gate_drops_large_bucket(self):
        """Gate G=1 should drop any bucket with >1 member."""
        # EC1A 1BB targets: CH_01, CH_03 (K2 key "acme|EC1A" has 2 entries for ACME query)
        blocker = ControlBlocker(BlockingConfig(gate_size=1, keys=["K2"]))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        # bucket size=2 > gate=1, so gated
        cand_ids = {c["cand_id"] for c in result.candidates}
        assert "CH_01" not in cand_ids
        assert "CH_03" not in cand_ids

    def test_gate_keeps_small_bucket(self):
        """Gate G=10 should keep all small buckets."""
        blocker = ControlBlocker(BlockingConfig(gate_size=10, keys=["K1"]))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        cand_ids = {c["cand_id"] for c in result.candidates}
        assert "CH_01" in cand_ids

    def test_gate_none_keeps_everything(self):
        """No gate should keep all candidates."""
        blocker = ControlBlocker(BlockingConfig(gate_size=None))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        # Should get candidates (K1, K2, K3, K4 all active)
        assert len(result.candidates) >= 1

    def test_stats_structure(self):
        blocker = ControlBlocker(BlockingConfig(gate_size=50))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME, QUERY_BETA])
        assert "n_candidate_pairs" in result.stats
        assert "per_key" in result.stats
        for k in ["K1", "K2", "K3", "K4"]:
            assert k in result.stats["per_key"]

    def test_candidates_sorted_by_cand_id(self):
        """Candidates for a query must be sorted by cand_id (determinism)."""
        blocker = ControlBlocker(BlockingConfig(gate_size=None, keys=["K2"]))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        acme_cands = [c for c in result.candidates if c["query_id"] == "GLEIF_01"]
        cand_ids = [c["cand_id"] for c in acme_cands]
        assert cand_ids == sorted(cand_ids)

    def test_ranks_contiguous_per_query(self):
        blocker = ControlBlocker(BlockingConfig(gate_size=None, keys=["K2"]))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME])
        acme_ranks = sorted(c["rank"] for c in result.candidates if c["query_id"] == "GLEIF_01")
        assert acme_ranks == list(range(1, len(acme_ranks) + 1))

    def test_unique_contribution_per_key(self):
        """per_key unique contribution must be non-negative and ≤ total."""
        blocker = ControlBlocker(BlockingConfig(gate_size=None))
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([QUERY_ACME, QUERY_BETA])
        total = result.stats["n_candidate_pairs"]
        for k in ALL_KEY_NAMES:
            uc = result.stats["per_key"][k]["n_unique_contribution"]
            assert uc >= 0
            assert uc <= total

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unknown blocking key"):
            BlockingConfig(keys=["K1", "K9"])

    def test_empty_targets(self):
        blocker = ControlBlocker(BlockingConfig())
        blocker.index_targets([])
        result = blocker.retrieve([QUERY_ACME])
        assert result.candidates == []

    def test_empty_queries(self):
        blocker = ControlBlocker(BlockingConfig())
        blocker.index_targets(TARGETS)
        result = blocker.retrieve([])
        assert result.candidates == []

    def test_reproducible(self):
        """Same input produces same candidates."""
        blocker1 = ControlBlocker(BlockingConfig(gate_size=50))
        blocker2 = ControlBlocker(BlockingConfig(gate_size=50))
        for b in (blocker1, blocker2):
            b.index_targets(TARGETS)
        r1 = blocker1.retrieve([QUERY_ACME, QUERY_BETA])
        r2 = blocker2.retrieve([QUERY_ACME, QUERY_BETA])
        pairs1 = sorted((c["query_id"], c["cand_id"]) for c in r1.candidates)
        pairs2 = sorted((c["query_id"], c["cand_id"]) for c in r2.candidates)
        assert pairs1 == pairs2
