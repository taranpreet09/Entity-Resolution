"""
tests.test_idf — Unit Tests for Channel G3 IDF Lexical Retriever
================================================================
"""

import math
import pytest

from ber.retrieve.idf import IDFRetriever, IDFConfig, infer_country_from_coords


def test_infer_country_from_coords():
    # New York / US
    assert infer_country_from_coords(40.71, -74.00) == "US"
    # London / GB
    assert infer_country_from_coords(51.50, -0.12) == "GB"
    # Brussels / BE
    assert infer_country_from_coords(50.85, 4.35) == "BE"
    # Invalid / NaN
    assert infer_country_from_coords(float("nan"), 0.0) == ""
    assert infer_country_from_coords(None, None) == ""


def test_idf_indexer_and_scorer():
    records = [
        {"id": "doc1", "name": "Starbucks Coffee", "country": "US"},
        {"id": "doc2", "name": "Starbucks Reserve", "country": "US"},
        {"id": "doc3", "name": "Peet's Coffee", "country": "US"},
        {"id": "doc4", "name": "Blue Bottle", "country": "US"},
    ]

    retriever = IDFRetriever(IDFConfig(k_candidates=5, df_cap=None))
    retriever.build_index(records)

    assert retriever.total_docs == 4
    # 'starbucks' appears in 2 docs, 'coffee' in 2 docs, 'reserve' in 1 doc
    assert retriever.df_table[retriever.token_to_id["starbucks"]] == 2
    assert retriever.df_table[retriever.token_to_id["coffee"]] == 2
    assert retriever.df_table[retriever.token_to_id["reserve"]] == 1

    # Overlap scores
    sc_exact = retriever.score_pair("Starbucks Coffee", "Starbucks Coffee")
    sc_partial = retriever.score_pair("Starbucks Coffee", "Starbucks Reserve")
    sc_none = retriever.score_pair("Starbucks Coffee", "Blue Bottle")

    assert sc_exact > sc_partial > 0.0
    assert sc_none == 0.0

    # Length normalization
    norm_retriever = IDFRetriever(IDFConfig(length_normalized=True))
    norm_retriever.build_index(records)
    norm_sc = norm_retriever.score_pair("Starbucks Coffee", "Starbucks Reserve")
    assert 0.0 < norm_sc < sc_partial


def test_exclude_self_and_top_k():
    records = [
        {"id": "q1", "name": "Acme Tools Store"},
        {"id": "c1", "name": "Acme Tools"},
        {"id": "c2", "name": "Acme Hardware"},
        {"id": "c3", "name": "Tools Depot"},
    ]

    retriever = IDFRetriever(IDFConfig(k_candidates=2, df_cap=None))
    retriever.build_index(records)

    results = retriever.retrieve_single(records[0])  # query q1
    cand_ids = [cid for cid, _ in results]

    # Must NOT include q1 (self-match excluded)
    assert "q1" not in cand_ids
    # Top-K capped at 2
    assert len(cand_ids) <= 2
    # c1 ("Acme Tools") matches two tokens, should be top candidate
    assert cand_ids[0] == "c1"


def test_df_cap_skips_common_tokens():
    # 'venue' appears in all 5 docs; 'alpha' in 2 docs
    records = [
        {"id": "1", "name": "Alpha Venue"},
        {"id": "2", "name": "Alpha Venue"},
        {"id": "3", "name": "Beta Venue"},
        {"id": "4", "name": "Gamma Venue"},
        {"id": "5", "name": "Delta Venue"},
    ]

    # Cap at df <= 3 -> 'venue' (df=5) is skipped
    cfg = IDFConfig(k_candidates=10, df_cap=3)
    retriever = IDFRetriever(cfg)
    retriever.build_index(records)

    # Query with only 'venue' should get no candidates
    res_venue = retriever.retrieve_single({"id": "q_v", "name": "Super Venue"})
    assert len(res_venue) == 0

    # Query with 'Alpha Venue' should match doc 1 and doc 2 via 'Alpha'
    res_alpha = retriever.retrieve_single({"id": "q_a", "name": "Alpha Venue"})
    cand_ids = {cid for cid, _ in res_alpha}
    assert cand_ids == {"1", "2"}


def test_country_filter():
    records = [
        {"id": "us1", "name": "Target Store", "country": "US"},
        {"id": "us2", "name": "Target Express", "country": "US"},
        {"id": "ca1", "name": "Target Canada", "country": "CA"},
    ]

    # Without country filter
    retriever_no_filter = IDFRetriever(IDFConfig(k_candidates=10, country_filter=False))
    retriever_no_filter.build_index(records)
    cands_all = [cid for cid, _ in retriever_no_filter.retrieve_single(records[0])]
    assert "us2" in cands_all
    assert "ca1" in cands_all

    # With country filter
    retriever_filter = IDFRetriever(IDFConfig(k_candidates=10, country_filter=True))
    retriever_filter.build_index(records)
    cands_filtered = [cid for cid, _ in retriever_filter.retrieve_single(records[0])]
    assert "us2" in cands_filtered
    assert "ca1" not in cands_filtered


def test_batch_retrieval():
    records = [
        {"id": "a", "name": "Bakery Delight"},
        {"id": "b", "name": "Delightful Bakery"},
        {"id": "c", "name": "Pizza Corner"},
    ]

    retriever = IDFRetriever(IDFConfig(k_candidates=5))
    retriever.build_index(records)

    cand_map, undir_pairs, total_slots = retriever.retrieve_batch(records)
    assert len(cand_map) == 3
    assert ("a", "b") in undir_pairs or ("b", "a") in undir_pairs
    assert total_slots > 0

