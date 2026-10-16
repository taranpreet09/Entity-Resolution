"""
Tests for datasets.adapters.foursquare
  - No POI leakage in model-visible fields
  - FoursquareBlocker basic retrieval
  - pair_recall metric
  - singleton_burden metric
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber_datasets.adapters.foursquare import (
    FoursquareAdapter, assert_no_poi_leakage,
    POI_IDENTIFIER_FIELDS, MODEL_COLS,
    extract_url_domain, digits_only, geo_cell,
)
from ber.eval.metrics_dedup import pair_recall, singleton_burden
from ber.eval.metrics import build_candidate_map


# ── POI leakage tests ──────────────────────────────────────────────────────

class TestNoPoiLeakage:
    def test_poi_identifier_fields_defined(self):
        assert "point_of_interest" in POI_IDENTIFIER_FIELDS
        assert "cv_flag" in POI_IDENTIFIER_FIELDS
        assert "location_key" in POI_IDENTIFIER_FIELDS

    def test_model_cols_exclude_poi(self):
        for banned in POI_IDENTIFIER_FIELDS:
            assert banned not in MODEL_COLS

    def test_leakage_detection_raises(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        from ber.store.schemas import RECORDS_SCHEMA
        bad = json.dumps({"name": "test", "point_of_interest": "poi_123"})
        tbl = pa.table({
            "dataset_id": ["foursquare"], "record_id": ["R1"],
            "canon_version": [None], "raw_fields": [bad],
            "source_file": [None], "source_row": pa.array([0], type=pa.int64()),
        }, schema=RECORDS_SCHEMA)
        p = tmp_path / "r.parquet"
        pq.write_table(tbl, p)
        with pytest.raises(AssertionError, match="point_of_interest"):
            assert_no_poi_leakage(p)

    def test_clean_records_pass(self, tmp_path):
        import pyarrow as pa, pyarrow.parquet as pq
        from ber.store.schemas import RECORDS_SCHEMA
        ok = json.dumps({"name": "test", "latitude": "1.0", "country": "US"})
        tbl = pa.table({
            "dataset_id": ["foursquare"], "record_id": ["R1"],
            "canon_version": [None], "raw_fields": [ok],
            "source_file": [None], "source_row": pa.array([0], type=pa.int64()),
        }, schema=RECORDS_SCHEMA)
        p = tmp_path / "r.parquet"
        pq.write_table(tbl, p)
        assert_no_poi_leakage(p)  # should not raise


# ── Key extraction helpers ─────────────────────────────────────────────────

class TestFoursquareKeyHelpers:
    def test_extract_url_domain_http(self):
        assert extract_url_domain("http://www.starbucks.com/menu") == "starbucks.com"

    def test_extract_url_domain_https(self):
        assert extract_url_domain("https://starbucks.com") == "starbucks.com"

    def test_extract_url_domain_empty(self):
        assert extract_url_domain("") == ""

    def test_extract_url_domain_no_dot(self):
        assert extract_url_domain("localhost") == ""

    def test_digits_only(self):
        assert digits_only("+1 (555) 123-4567") == "15551234567"

    def test_digits_only_empty(self):
        assert digits_only("") == ""

    def test_geo_cell_rounds(self):
        assert geo_cell("51.5074", "-0.1278") == "51.51|-0.13"

    def test_geo_cell_invalid(self):
        assert geo_cell("", "") == ""


# ── FoursquareBlocker — import lazily to avoid loading experiment module at collection time ──

def _get_blocker():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.run_foursquare_control import FoursquareBlocker
    return FoursquareBlocker


# ── FoursquareBlocker ─────────────────────────────────────────────────────

class TestFoursquareBlocker:
    def _make_rec(self, rid, name, country="US", lat="40.0", lon="-74.0",
                  phone="", url="", zipcode=""):
        return {"record_id": rid, "raw_fields": json.dumps({
            "name": name, "country": country, "latitude": lat, "longitude": lon,
            "phone": phone, "url": url, "zip": zipcode,
        })}

    def test_k1_exact_name_country(self):
        FoursquareBlocker = _get_blocker()
        bl = FoursquareBlocker(gate_size=None, keys=["K1"])
        recs = [
            self._make_rec("A", "Starbucks", "US"),
            self._make_rec("B", "Starbucks", "US"),
            self._make_rec("C", "McDonalds", "US"),
        ]
        bl.index_records(recs)
        cands, _, _, retr = bl.retrieve([recs[0]])
        assert ("A", "B") in retr or ("B", "A") in retr
        assert not any(c["cand_id"] == "A" for c in cands)  # no self-pair

    def test_no_self_pair(self):
        FoursquareBlocker = _get_blocker()
        bl = FoursquareBlocker(gate_size=None, keys=["K1"])
        rec = self._make_rec("A", "Starbucks", "US")
        bl.index_records([rec])
        cands, _, _, _ = bl.retrieve([rec])
        assert all(c["cand_id"] != "A" for c in cands)

    def test_gate_drops_large_bucket(self):
        FoursquareBlocker = _get_blocker()
        # Make 5 records with same name+country
        recs = [self._make_rec(f"R{i}", "Starbucks", "US") for i in range(5)]
        bl = FoursquareBlocker(gate_size=2, keys=["K1"])
        bl.index_records(recs)
        cands, _, gated, retr = bl.retrieve([recs[0]])
        # bucket size = 4 (excluding self) > gate=2 → all gated
        assert len(retr) == 0

    def test_feasibility_estimate(self):
        FoursquareBlocker = _get_blocker()
        from experiments.run_foursquare_control import PAIR_COUNT_LIMIT
        recs = [self._make_rec(f"R{i}", "Starbucks", "US") for i in range(100)]
        bl = FoursquareBlocker(gate_size=None, keys=["K1"])
        bl.index_records(recs)
        est = bl.estimate_pair_count()
        assert "estimated_total_upper_bound" in est
        assert "feasible" in est
        # 100 records in same bucket → 100*99/2 = 4950 pairs
        assert est["per_key"]["K1"] == 100 * 99 // 2


# ── pair_recall ───────────────────────────────────────────────────────────

class TestPairRecall:
    def test_perfect(self):
        retr = {("A","B"), ("C","D")}
        true = [("A","B"), ("C","D")]
        assert pair_recall(retr, true) == pytest.approx(1.0)

    def test_reverse_order_counts(self):
        # True pair is (A,B); retrieved as (B,A)
        retr = {("B","A")}
        true = [("A","B")]
        assert pair_recall(retr, true) == pytest.approx(1.0)

    def test_zero(self):
        retr = set()
        true = [("A","B")]
        assert pair_recall(retr, true) == pytest.approx(0.0)

    def test_partial(self):
        retr = {("A","B")}
        true = [("A","B"), ("C","D")]
        assert pair_recall(retr, true) == pytest.approx(0.5)

    def test_empty_true_returns_nan(self):
        import math
        assert math.isnan(pair_recall(set(), []))


# ── singleton_burden ──────────────────────────────────────────────────────

class TestSingletonBurden:
    def test_basic(self):
        cands = [{"query_id": "S1", "cand_id": f"X{i}", "rank": i, "score": 1.0}
                 for i in range(1, 6)]
        cm = build_candidate_map(cands)
        sb = singleton_burden(cm, {"S1", "S2"})
        assert sb["n_singleton_queries_with_candidates"] == 1
        assert sb["total_singleton_candidates"] == 5

    def test_empty(self):
        sb = singleton_burden({}, {"S1"})
        assert sb["total_singleton_candidates"] == 0
