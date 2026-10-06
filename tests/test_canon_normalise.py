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
