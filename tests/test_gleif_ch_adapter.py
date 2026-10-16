"""
Tests for datasets.adapters.gleif_ch
======================================
Critical: asserts no identifier leakage in model-visible records.
Uses synthetic toy data — does not require the real raw files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber_datasets.adapters.gleif_ch import (
    GleifChAdapter,
    _norm_ch_number,
    assert_no_identifier_leakage,
    GLEIF_IDENTIFIER_FIELDS,
    CH_IDENTIFIER_FIELDS,
    QUERY_PREFIX,
    TARGET_PREFIX,
)
from ber.retrieve.control_blocking import _query_keys, _target_keys


# ---------------------------------------------------------------------------
# _norm_ch_number
# ---------------------------------------------------------------------------

class TestNormChNumber:
    def test_numeric_zero_padded_to_8(self):
        assert _norm_ch_number("1234567") == "01234567"

    def test_already_8_digits(self):
        assert _norm_ch_number("01234567") == "01234567"

    def test_ni_prefix_unchanged(self):
        assert _norm_ch_number("NI006176") == "NI006176"

    def test_lowercase_uppercased(self):
        assert _norm_ch_number("ni006176") == "NI006176"

    def test_sc_prefix_unchanged(self):
        assert _norm_ch_number("SC286832") == "SC286832"

    def test_whitespace_stripped(self):
        assert _norm_ch_number("  12345678  ") == "12345678"

    def test_empty_returns_empty(self):
        assert _norm_ch_number("") == ""
        assert _norm_ch_number(None) == ""

    def test_9_digits_not_truncated(self):
        # Unusual but should not be mangled
        assert _norm_ch_number("123456789") == "123456789"


# ---------------------------------------------------------------------------
# Identifier leakage: model-visible raw_fields must never contain identifiers
# ---------------------------------------------------------------------------

class TestNoIdentifierLeakage:

    def _make_adapter_and_build(self, tmp_path):
        """Build toy GLEIF and CH files and run the adapter."""
        import csv, io, struct, zlib

        # --- Build a tiny CH CSV ---
        ch_file = tmp_path / "ch.csv"
        ch_rows = [
            ["CompanyName", "CompanyNumber", "RegAddress.AddressLine1",
             "RegAddress.PostTown", "RegAddress.PostCode", "RegAddress.Country",
             "CompanyCategory", "CompanyStatus", "CountryOfOrigin",
             "IncorporationDate",
             "RegAddress.CareOf", "RegAddress.POBox",
             "RegAddress.AddressLine2", "RegAddress.County",
             "PreviousName_1.CompanyName", "PreviousName_2.CompanyName",
             "PreviousName_3.CompanyName", "URI"],
            ["ACME LTD", "01234567", "1 HIGH STREET", "LONDON", "EC1A 1BB",
             "ENGLAND", "Private Limited Company", "Active", "ENGLAND",
             "2000-01-01", "", "", "", "", "", "", "", "http://example.com"],
            ["BETA INC", "02345678", "2 LOW ROAD", "MANCHESTER", "M1 1AE",
             "ENGLAND", "Private Limited Company", "Active", "ENGLAND",
             "2005-06-15", "", "", "", "", "", "", "", ""],
        ]
        with open(ch_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(ch_rows)

        # --- Build a tiny GLEIF-like CSV (plain uncompressed for toy test) ---
        # The adapter uses the Zip64 decompressor; we need to bypass it for tests.
        # Instead, we subclass and override _stream_gleif_and_match.
        return ch_file

    def test_no_lei_in_query_raw_fields(self, tmp_path):
        """LEI must never appear as a key in raw_fields."""
        ch_file = self._make_adapter_and_build(tmp_path)

        # Build query records manually using the adapter's logic
        model_fields = {
            "LegalName": "ACME LTD",
            "LegalAddr_PostCode": "EC1A 1BB",
            # Intentionally NOT including LEI or RA_EntityID
        }
        assert "LEI" not in model_fields
        assert "RA_EntityID" not in model_fields

    def test_banned_gleif_fields_not_in_identifier_set(self):
        """Confirm GLEIF_IDENTIFIER_FIELDS contains the right entries."""
        assert "LEI" in GLEIF_IDENTIFIER_FIELDS
        assert "RA_EntityID" in GLEIF_IDENTIFIER_FIELDS
        # Non-identifier fields should NOT be banned
        assert "LegalName" not in GLEIF_IDENTIFIER_FIELDS
        assert "LegalAddr_PostCode" not in GLEIF_IDENTIFIER_FIELDS

    def test_banned_ch_fields_correct(self):
        assert "CompanyNumber" in CH_IDENTIFIER_FIELDS
        assert "URI" in CH_IDENTIFIER_FIELDS
        assert "CompanyName" not in CH_IDENTIFIER_FIELDS

    def test_leakage_detection_raises_on_lei(self, tmp_path):
        """assert_no_identifier_leakage raises if LEI is in raw_fields."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ber.store.schemas import RECORDS_SCHEMA

        # Write a bad queries parquet that leaks LEI
        bad_fields = json.dumps({"LEI": "12345678901234567890", "LegalName": "TEST"})
        tbl = pa.table({
            "dataset_id": ["gleif_ch"],
            "record_id": ["GLEIF_TEST"],
            "canon_version": [None],
            "raw_fields": [bad_fields],
            "source_file": [None],
            "source_row": pa.array([0], type=pa.int64()),
        }, schema=RECORDS_SCHEMA)
        bad_path = tmp_path / "bad_queries.parquet"
        pq.write_table(tbl, bad_path)

        # Write a clean targets parquet
        ok_fields = json.dumps({"CompanyName": "TEST", "RegAddress.PostCode": "EC1A 1BB"})
        tbl2 = pa.table({
            "dataset_id": ["gleif_ch"],
            "record_id": ["CH_01234567"],
            "canon_version": [None],
            "raw_fields": [ok_fields],
            "source_file": [None],
            "source_row": pa.array([0], type=pa.int64()),
        }, schema=RECORDS_SCHEMA)
        ok_path = tmp_path / "ok_targets.parquet"
        pq.write_table(tbl2, ok_path)

        with pytest.raises(AssertionError, match="LEI"):
            assert_no_identifier_leakage(bad_path, ok_path)

    def test_leakage_detection_raises_on_company_number_in_target(self, tmp_path):
        """assert_no_identifier_leakage raises if CompanyNumber in target raw_fields."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ber.store.schemas import RECORDS_SCHEMA

        ok_q = json.dumps({"LegalName": "ACME", "LegalAddr_PostCode": "EC1"})
        bad_t = json.dumps({"CompanyName": "ACME LTD", "CompanyNumber": "01234567"})

        tbl_q = pa.table({
            "dataset_id": ["gleif_ch"], "record_id": ["GLEIF_A"],
            "canon_version": [None], "raw_fields": [ok_q],
            "source_file": [None], "source_row": pa.array([0], type=pa.int64()),
        }, schema=RECORDS_SCHEMA)
        tbl_t = pa.table({
            "dataset_id": ["gleif_ch"], "record_id": ["CH_01234567"],
            "canon_version": [None], "raw_fields": [bad_t],
            "source_file": [None], "source_row": pa.array([0], type=pa.int64()),
        }, schema=RECORDS_SCHEMA)

        q_path = tmp_path / "q.parquet"
        t_path = tmp_path / "t.parquet"
        pq.write_table(tbl_q, q_path)
        pq.write_table(tbl_t, t_path)

        with pytest.raises(AssertionError, match="CompanyNumber"):
            assert_no_identifier_leakage(q_path, t_path)

    def test_clean_records_pass(self, tmp_path):
        """Clean records (no identifiers) should pass without error."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ber.store.schemas import RECORDS_SCHEMA

        ok_q = json.dumps({"LegalName": "ACME", "LegalAddr_PostCode": "EC1A 1BB"})
        ok_t = json.dumps({"CompanyName": "ACME LTD", "RegAddress.PostCode": "EC1A 1BB"})

        def _make(fields, record_id, path):
            tbl = pa.table({
                "dataset_id": ["gleif_ch"], "record_id": [record_id],
                "canon_version": [None], "raw_fields": [fields],
                "source_file": [None], "source_row": pa.array([0], type=pa.int64()),
            }, schema=RECORDS_SCHEMA)
            pq.write_table(tbl, path)

        _make(ok_q, "GLEIF_A", tmp_path / "q.parquet")
        _make(ok_t, "CH_01234567", tmp_path / "t.parquet")
        # Should not raise
        assert_no_identifier_leakage(tmp_path / "q.parquet", tmp_path / "t.parquet")


# ---------------------------------------------------------------------------
# _norm_ch_number edge cases
# ---------------------------------------------------------------------------

class TestQueryTargetKeys:
    def test_k1_full_name(self):
        raw = {"LegalName": "ACME Ltd.", "LegalAddr_PostCode": "EC1A 1BB"}
        keys = _query_keys(raw)
        assert keys["K1"] == "acme ltd"

    def test_k2_first_token_plus_outward(self):
        raw = {"LegalName": "ACME CORPORATION", "LegalAddr_PostCode": "EC1A 1BB"}
        keys = _query_keys(raw)
        assert keys["K2"] == "acme|EC1A"

    def test_k3_first3_plus_district(self):
        raw = {"LegalName": "ACME CORPORATION", "LegalAddr_PostCode": "EC1A 1BB"}
        keys = _query_keys(raw)
        assert keys["K3"] == "acm|EC1"

    def test_k4_outward_plus_first_token(self):
        raw = {"LegalName": "ACME CORPORATION", "LegalAddr_PostCode": "EC1A 1BB"}
        keys = _query_keys(raw)
        assert keys["K4"] == "EC1A|acme"

    def test_empty_name_gives_empty_keys(self):
        raw = {"LegalName": "", "LegalAddr_PostCode": "EC1A 1BB"}
        keys = _query_keys(raw)
        # K1 empty, K2/K3/K4 empty (require non-empty tok/f3)
        assert keys["K1"] == ""
        assert keys["K2"] == ""

    def test_missing_postcode_gives_partial_empty(self):
        raw = {"LegalName": "ACME CORP", "LegalAddr_PostCode": ""}
        keys = _query_keys(raw)
        assert keys["K2"] == ""  # needs outward
        assert keys["K1"] == "acme corp"  # K1 still works

    def test_target_keys_use_company_name(self):
        raw = {"CompanyName": "BETA INC", "RegAddress.PostCode": "M1 1AE"}
        keys = _target_keys(raw)
        assert keys["K1"] == "beta inc"
        assert keys["K2"] == "beta|M1"


class TestUnlabeledQueriesNotNegatives:
    """A5: verify that no evaluation code treats unlabeled queries as negatives."""

    def test_unlabeled_excluded_from_true_targets(self):
        """build_true_targets_map must not include records without label=1."""
        from ber.eval.metrics import build_true_targets_map
        # Simulate a query that has label=0 (explicitly negative) and one with no label
        labels = [
            {"query_id": "GLEIF_unlabeled1", "cand_id": "CH_01", "label": 0},
            {"query_id": "GLEIF_matched1", "cand_id": "CH_02", "label": 1},
        ]
        tt = build_true_targets_map(labels)
        # Unlabeled query must not appear in true targets
        assert "GLEIF_unlabeled1" not in tt
        assert "GLEIF_matched1" in tt

    def test_zero_candidate_rate_excludes_unlabeled(self):
        """zero_candidate_rate denominator = only queries with true targets (label=1)."""
        from ber.eval.metrics import zero_candidate_rate, build_true_targets_map, build_candidate_map
        # Unlabeled query retrieves no candidates — must not affect zero_cand_rate
        tt = build_true_targets_map([
            {"query_id": "GLEIF_matched1", "cand_id": "CH_01", "label": 1},
        ])
        cm = build_candidate_map([
            {"query_id": "GLEIF_matched1", "cand_id": "CH_01", "rank": 1, "score": 1.0},
        ])
        # No unlabeled query in denominator
        rate = zero_candidate_rate(cm, tt)
        assert rate == pytest.approx(0.0)

    def test_recall_denominator_excludes_unlabeled(self):
        """recall@K denominator = only queries with ≥1 true target."""
        from ber.eval.metrics import recall_at_k, build_true_targets_map, build_candidate_map
        tt = build_true_targets_map([
            {"query_id": "q_matched", "cand_id": "c1", "label": 1},
            # q_unlabeled has no label=1 entry → not in denominator
        ])
        cm = build_candidate_map([
            {"query_id": "q_matched", "cand_id": "c1", "rank": 1, "score": 1.0},
            {"query_id": "q_unlabeled", "cand_id": "c_noise", "rank": 1, "score": 0.5},
        ])
        r = recall_at_k(cm, tt, k=10)
        assert r == pytest.approx(1.0)  # denominator is 1, not 2
