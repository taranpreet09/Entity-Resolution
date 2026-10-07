"""
Tests for ber.canon.normalise
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber.canon.normalise import (
    canonicalise, outward_code, postcode_district,
    first_token, first_n_chars, canon_stats,
)


class TestCanonicalize:
    def test_lowercase(self):
        assert canonicalise("ACME LTD") == "acme ltd"

    def test_punctuation_stripped(self):
        assert canonicalise("ACME Ltd.") == "acme ltd"

    def test_nfkd_accent_dropped(self):
        # café → cafe (accent dropped via ASCII-ignore)
        assert canonicalise("café") == "cafe"

    def test_ligature_expanded(self):
        # NFKD expands ligatures: ﬁ → fi
        assert canonicalise("ﬁnance") == "finance"

    def test_ampersand_stripped(self):
        assert canonicalise("Smith & Jones") == "smith jones"

    def test_multiple_spaces_collapsed(self):
        assert canonicalise("foo   bar") == "foo bar"

    def test_leading_trailing_stripped(self):
        assert canonicalise("  acme  ") == "acme"

    def test_none_returns_empty(self):
        assert canonicalise(None) == ""

    def test_empty_returns_empty(self):
        assert canonicalise("") == ""

    def test_non_ascii_only_returns_empty(self):
        # Pure Chinese characters → all dropped by ASCII-ignore
        assert canonicalise("中文") == ""

    def test_digits_kept(self):
        assert canonicalise("2024 Holdings") == "2024 holdings"

    def test_hyphen_becomes_space(self):
        assert canonicalise("Smith-Jones") == "smith jones"

    def test_comma_stripped(self):
        assert canonicalise("Jones, Ltd") == "jones ltd"


class TestOutwardCode:
    def test_standard_with_space(self):
        assert outward_code("SW1A 2AA") == "SW1A"

    def test_without_space(self):
        assert outward_code("EC1A1BB") == "EC1A"

    def test_lowercase(self):
        assert outward_code("sw1a 2aa") == "SW1A"

    def test_single_letter_district(self):
        assert outward_code("M1 1AE") == "M1"

    def test_two_letter_district(self):
        assert outward_code("CR2 6XH") == "CR2"

    def test_empty_returns_empty(self):
        assert outward_code("") == ""
        assert outward_code(None) == ""

    def test_invalid_returns_empty(self):
        assert outward_code("INVALID") == ""

    def test_short_truncated_postcode(self):
        assert outward_code("EC1A") == "EC1A"


class TestPostcodeDistrict:
    def test_standard(self):
        assert postcode_district("SW1A 2AA") == "SW1"

    def test_single_digit(self):
        assert postcode_district("M1 1AE") == "M1"

    def test_empty(self):
        assert postcode_district("") == ""

    def test_two_letter_area(self):
        assert postcode_district("CR2 6XH") == "CR2"


class TestFirstToken:
    def test_basic(self):
        assert first_token("ACME LIMITED") == "acme"

    def test_single_word(self):
        assert first_token("ACME") == "acme"

    def test_empty(self):
        assert first_token("") == ""

    def test_with_punctuation(self):
        # punctuation stripped before tokenising
        assert first_token("!ACME LTD") == "acme"


class TestFirstNChars:
    def test_basic(self):
        assert first_n_chars("ACME CORP", 3) == "acm"

    def test_shorter_than_n(self):
        assert first_n_chars("AB", 3) == "ab"

    def test_empty(self):
        assert first_n_chars("", 3) == ""


class TestCanonStats:
    def test_counts_empty_after_canon(self):
        records = [
            {"name": "中文"},      # non-ASCII → empty after canon
            {"name": "ACME LTD"},  # stays non-empty
            {"name": ""},          # raw empty → not counted
        ]
        result = canon_stats(records, fields=["name"])
        assert result["name"]["raw_non_empty"] == 2  # "中文" and "ACME LTD" are non-empty raw
        assert result["name"]["canon_empty_from_non_empty"] == 1  # only "中文"

    def test_raw_fields_json_lookup(self):
        records = [
            {"raw_fields": '{"CompanyName": "ACME", "addr": ""}'},
            {"raw_fields": '{"CompanyName": "", "addr": "中文"}'},
        ]
        result = canon_stats(records, fields=["CompanyName", "addr"])
        assert result["CompanyName"]["raw_non_empty"] == 1
        assert result["addr"]["canon_empty_from_non_empty"] == 1


class TestStripLegalSuffixes:
    """Tests for ber.canon.normalise.strip_legal_suffixes."""

    def test_strip_limited(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("ACME LIMITED") == "acme"

    def test_strip_ltd(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("ACME LTD") == "acme"

    def test_strip_plc(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("BARCLAYS PLC") == "barclays"

    def test_strip_llp(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("SMITH JONES LLP") == "smith jones"

    def test_ampersand_normalised(self):
        from ber.canon.normalise import strip_legal_suffixes
        # & → and; then trailing co and ltd both stripped
        assert strip_legal_suffixes("BARCLAYS & CO LTD") == "barclays"

    def test_ampersand_mid_name_kept(self):
        from ber.canon.normalise import strip_legal_suffixes
        # After canonicalise(), '&' becomes a space (punct-stripped), not 'and'.
        # So "SMITH & JONES LTD" → canon "smith jones ltd" → strip "ltd" → "smith jones"
        # The and_aliases substitution only matters when the raw text contains
        # the word "and" (not "&", which is already gone by canon time).
        result = strip_legal_suffixes("SMITH & JONES LTD")
        assert result == "smith jones"

    def test_multiword_suffix(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("ACME LIMITED LIABILITY PARTNERSHIP") == "acme"

    def test_whole_name_is_suffix_returns_empty(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("LTD") == ""

    def test_no_suffix(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("NATIONAL GRID") == "national grid"

    def test_empty_input(self):
        from ber.canon.normalise import strip_legal_suffixes
        assert strip_legal_suffixes("") == ""
        assert strip_legal_suffixes(None) == ""

    def test_idempotent(self):
        from ber.canon.normalise import strip_legal_suffixes
        s = strip_legal_suffixes("ACME LIMITED")
        assert strip_legal_suffixes(s) == s  # no suffix to strip


class TestControlBlockerV2:
    """Tests that v2 key extraction and with_gate() version propagation work."""

    def test_v2_k4v2_fires_on_suffix_variant(self):
        import json
        from ber.retrieve.control_blocking import ControlBlocker, BlockingConfig, ALL_KEY_NAMES_V2
        # Target: "ACME LIMITED"; Query: "ACME LTD" — differ only in suffix
        # K1 misses (canon names differ: "acme limited" vs "acme ltd")
        # K4v2 hits (both strip to "acme")
        cfg = BlockingConfig(gate_size=None, keys=["K4v2"], version="v2")
        blocker = ControlBlocker(cfg)
        blocker.index_targets([
            {"record_id": "CH_01", "raw_fields": json.dumps(
                {"CompanyName": "ACME LIMITED", "RegAddress.PostCode": "EC1A 1BB"})},
        ])
        result = blocker.retrieve([
            {"record_id": "GLEIF_01", "raw_fields": json.dumps(
                {"LegalName": "ACME LTD", "LegalAddr_PostCode": "EC1A 1BB"})},
        ])
        cand_ids = {c["cand_id"] for c in result.candidates}
        assert "CH_01" in cand_ids

    def test_v2_k4v2_does_not_fire_on_different_name(self):
        import json
        from ber.retrieve.control_blocking import ControlBlocker, BlockingConfig
        cfg = BlockingConfig(gate_size=None, keys=["K4v2"], version="v2")
        blocker = ControlBlocker(cfg)
        blocker.index_targets([
            {"record_id": "CH_99", "raw_fields": json.dumps(
                {"CompanyName": "ZETA HOLDINGS PLC", "RegAddress.PostCode": "M1 1AE"})},
        ])
        result = blocker.retrieve([
            {"record_id": "GLEIF_01", "raw_fields": json.dumps(
                {"LegalName": "ACME LTD", "LegalAddr_PostCode": "EC1A 1BB"})},
        ])
        assert result.candidates == []

    def test_with_gate_preserves_version(self):
        from ber.retrieve.control_blocking import ControlBlocker, BlockingConfig, ALL_KEY_NAMES_V2
        cfg = BlockingConfig(gate_size=None, keys=ALL_KEY_NAMES_V2, version="v2")
        blocker = ControlBlocker(cfg)
        g50 = blocker.with_gate(50)
        assert g50.config.version == "v2"
        assert g50.config.keys == ALL_KEY_NAMES_V2

    def test_v1_k4_is_swapped_k2(self):
        """Confirm v1 K4 = outward|first_token is structurally identical to K2 = first_token|outward."""
        import json
        from ber.retrieve.control_blocking import _query_keys, _target_keys
        raw = {"LegalName": "ACME CORP", "LegalAddr_PostCode": "EC1A 1BB"}
        qk = _query_keys(raw)
        # K2 = first_token|outward, K4 = outward|first_token
        # They should differ in value but produce the same bucket cardinality on the target side
        assert qk["K2"] == "acme|EC1A"
        assert qk["K4"] == "EC1A|acme"
        # For a target, same structure
        tr = {"CompanyName": "ACME CORP", "RegAddress.PostCode": "EC1A 1BB"}
        tk = _target_keys(tr)
        assert tk["K2"] == "acme|EC1A"
        assert tk["K4"] == "EC1A|acme"
