"""
scratch/diagnose_step3.py — Phase 4b Step 3 Identical-Name Miss Diagnostics & Train Fix
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ber.canon.normalise import canonicalise
from ber.retrieve.idf import IDFRetriever, IDFConfig, infer_country_from_coords

def classify_name(name_a: str, name_b: str) -> str:
    ca = canonicalise(name_a); cb = canonicalise(name_b)
    if not ca or not cb: return "empty"
    if ca == cb: return "identical"
    toks_a = set(ca.split()); toks_b = set(cb.split())
    if toks_a == toks_b: return "token_reorder"
    union = toks_a | toks_b
    jacc = len(toks_a & toks_b) / len(union) if union else 0.0
    if jacc >= 0.5: return "partial_overlap"
    return "different"

def main():
    print("=== Phase 4b Step 3: Identical-Name Miss Diagnostics on Train Sample ===", flush=True)

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    record_map = {r["id"]: r for r in all_records}

    # 100k Train sample (seed 42)
    rng = random.Random(42)
    all_indices = list(range(len(df)))
    train_sample_indices = sorted(rng.sample(all_indices, 100_000))
    train_sample_records = [all_records[i] for i in train_sample_indices]
    train_sample_ids = {r["id"] for r in train_sample_records}

    # True pairs in train sample
    poi_groups = defaultdict(list)
    for r in train_sample_records:
        poi = r["point_of_interest"]
        if poi: poi_groups[poi].append(r)

    train_true_pairs = []
    for poi, rows in poi_groups.items():
        if len(rows) > 1:
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    train_true_pairs.append((rows[i], rows[j]))

    print(f"Total true pairs on train sample: {len(train_true_pairs):,}", flush=True)

    # Identical name pairs
    identical_pairs = [
        (ra, rb) for ra, rb in train_true_pairs
        if classify_name(ra["name"], rb["name"]) == "identical"
    ]
    print(f"Total identical-name true pairs: {len(identical_pairs):,} ({len(identical_pairs)/len(train_true_pairs)*100:.2f}%)", flush=True)

    # Build standard G3 (K=5, df_cap=5000, country_filter=True)
    g3 = IDFRetriever(IDFConfig(k_candidates=5, df_cap=5000, country_filter=True))
    g3.build_index(all_records)
    effective_df_cap = g3.get_effective_df_cap()

    # Document frequencies across full corpus
    token_doc_counts = Counter()
    for r in all_records:
        cn = canonicalise(r["name"])
        if cn:
            token_doc_counts.update(set(cn.split()))

    print("\n--- Top 20 Tokens by Document Frequency in Full Table ---", flush=True)
    for rank, (tok, cnt) in enumerate(token_doc_counts.most_common(20), 1):
        pct = cnt / len(all_records) * 100.0
        capped = cnt > effective_df_cap
        print(f"  {rank:2d}. {tok:15s}: {cnt:6d} docs ({pct:5.2f}%) {'[CAPPED]' if capped else ''}")

    # Inspect "venue" specifically
    venue_cnt = token_doc_counts.get("venue", 0)
    print(f"\nInspection of 'venue': DF = {venue_cnt:,} ({venue_cnt/len(all_records)*100:.2f}% of corpus). Effective DF cap = {effective_df_cap:,}. Is capped: {venue_cnt > effective_df_cap}")

    # Run retrieval for identical-name query endpoints
    ident_q_records = list({r["id"]: r for ra, rb in identical_pairs for r in (ra, rb)}.values())
    cands_ident, undir_ident, _ = g3.retrieve_batch(ident_q_records, k=5, country_filter=True)

    # Categorize identical-name misses
    missed_identical = []
    found_identical = []
    for ra, rb in identical_pairs:
        p_tuple = (min(ra["id"], rb["id"]), max(ra["id"], rb["id"]))
        if p_tuple in undir_ident:
            found_identical.append((ra, rb))
        else:
            missed_identical.append((ra, rb))

    print(f"\nIdentical-Name True Pairs: Found = {len(found_identical):,} ({len(found_identical)/len(identical_pairs)*100:.2f}%), Missed = {len(missed_identical):,} ({len(missed_identical)/len(identical_pairs)*100:.2f}%)")

    # Diagnose reasons for misses:
    # (a) All tokens above df cap
    # (b) Outranked by >K ties (report max tie size)
    # (c) Country filter excluded
    # (d) Other
    cat_counts = Counter()
    tokens_in_all_capped = Counter()
    max_tie_size_seen = 0
    tie_sizes = []

    for ra, rb in missed_identical:
        cn_a = canonicalise(ra["name"])
        cn_b = canonicalise(rb["name"])
        toks_a = cn_a.split() if cn_a else []
        toks_b = cn_b.split() if cn_b else []
        shared_toks = set(toks_a) & set(toks_b)

        # Check (c) Country filter excluded
        country_a = ra.get("country", "")
        country_b = rb.get("country", "")
        is_diff_country = (country_a and country_b and country_a != country_b)

        # Check if all tokens capped
        all_capped = (len(shared_toks) == 0) or all(token_doc_counts[t] > effective_df_cap for t in shared_toks)

        if is_diff_country:
            cat_counts["(c) country_filter_excluded"] += 1
        elif all_capped:
            cat_counts["(a) all_tokens_above_df_cap"] += 1
            tokens_in_all_capped.update(shared_toks)
        else:
            # Check candidate list and ties
            # Score target b from query a
            score_b = g3.score_pair(ra["name"], rb["name"])
            # What was the lowest score in top-K candidates for ra and rb?
            # Check how many candidates had score >= score_b
            cands_a = cands_ident.get(ra["id"], [])
            cands_b = cands_ident.get(rb["id"], [])
            
            # Count candidates in posting list that share tokens
            # Retrieve all raw candidates for ra without K limit
            q_toks = [t for t in toks_a if token_doc_counts[t] <= effective_df_cap]
            pool_candidates = set()
            for t in q_toks:
                if t in g3.token_to_id:
                    tid = g3.token_to_id[t]
                    pool_candidates.update(g3.postings[tid])
            
            # How many pool candidates share the exact same score?
            tie_size = len(pool_candidates)
            if tie_size > max_tie_size_seen:
                max_tie_size_seen = tie_size
            tie_sizes.append(tie_size)

            if len(cands_a) >= 5 or len(cands_b) >= 5:
                cat_counts["(b) outranked_by_gt_k_ties"] += 1
            else:
                cat_counts["(d) other"] += 1

    print("\n--- Categorization of Identical-Name Misses (K=5, Country ON) ---")
    for cat, cnt in sorted(cat_counts.items()):
        pct = cnt / len(missed_identical) * 100.0
        print(f"  {cat:32s}: {cnt:5d} ({pct:5.2f}%)")

    print(f"\nMax tie size seen in category (b): {max_tie_size_seen:,} candidates.")
    if tie_sizes:
        print(f"Tie sizes in (b): Median = {np.median(tie_sizes):.1f}, P95 = {np.percentile(tie_sizes, 95):.1f}, Max = {np.max(tie_sizes)}")

    print("\n--- Top Tokens in Category (a) [All Tokens Above DF Cap] ---")
    for tok, cnt in tokens_in_all_capped.most_common(15):
        df_val = token_doc_counts[tok]
        print(f"  {tok:15s}: in {cnt:4d} missed pairs (Full corpus DF = {df_val:,} docs, cap = {effective_df_cap})")

    # -------------------------------------------------------------
    # TEST ONE TRAIN FIX:
    # Fix hypothesis: If all tokens of a venue are above the df_cap,
    # don't drop ALL of them to return 0 candidates! Instead, retain the
    # RAREST token (relative df cap / fallback rarest token) so short/generic
    # identical names (like "Venue 1 Bistro" or "Panda Express") still retrieve candidates.
    # OR test tie-break by IDF-weighted char overlap.
    # Let's test the "Fallback Rarest Token" fix (or length-adjusted df cap):
    # If all tokens are capped, keep the min-DF token for that query!
    # -------------------------------------------------------------
    print("\n=== Testing ONE Train Fix: Fallback Rarest Token for All-Capped Queries ===", flush=True)

    class EnhancedIDFRetriever(IDFRetriever):
        def _get_query_tokens(self, query_record: dict) -> list[int]:
            name = canonicalise(query_record.get("name", ""))
            if not name:
                return []
            tokens = name.split()
            valid_ids = []
            min_df = float("inf")
            min_df_tid = None
            
            for t in tokens:
                if t in self.token_to_id:
                    tid = self.token_to_id[t]
                    df_t = len(self.postings[tid])
                    if df_t <= self.effective_df_cap:
                        valid_ids.append(tid)
                    if df_t < min_df:
                        min_df = df_t
                        min_df_tid = tid

            # Fix: If ALL tokens exceeded df cap, keep the rarest token
            if not valid_ids and min_df_tid is not None:
                valid_ids.append(min_df_tid)

            return valid_ids

    enhanced_g3 = EnhancedIDFRetriever(IDFConfig(k_candidates=5, df_cap=5000, country_filter=True))
    enhanced_g3.build_index(all_records)

    # Evaluate on a 10,000 query train sample to measure recall gain vs candidate increase
    test_q_records = train_sample_records[:10_000]
    test_q_rids = [r["id"] for r in test_q_records]
    test_q_set = set(test_q_rids)
    test_true_pairs = [
        (ra, rb) for ra, rb in train_true_pairs
        if ra["id"] in test_q_set and rb["id"] in test_q_set
    ]
    test_true_pair_set = {(min(ra["id"], rb["id"]), max(ra["id"], rb["id"])) for ra, rb in test_true_pairs}

    # Baseline G3 on 10k test sample
    cands_base, undir_base, tot_c_base = g3.retrieve_batch(test_q_records, k=5, country_filter=True)
    base_hits = len(test_true_pair_set & undir_base)
    base_recall = base_hits / len(test_true_pair_set) if test_true_pair_set else 0.0

    # Enhanced G3 on 10k test sample
    cands_enh, undir_enh, tot_c_enh = enhanced_g3.retrieve_batch(test_q_records, k=5, country_filter=True)
    enh_hits = len(test_true_pair_set & undir_enh)
    enh_recall = enh_hits / len(test_true_pair_set) if test_true_pair_set else 0.0

    cand_increase_pct = (tot_c_enh - tot_c_base) / max(1, tot_c_base) * 100.0
    recall_gain_pp = (enh_recall - base_recall) * 100.0

    print(f"Baseline G3 (K=5): Pair Recall = {base_recall:.4f} ({base_hits}/{len(test_true_pair_set)}), Candidates = {tot_c_base:,}")
    print(f"Enhanced G3 (K=5): Pair Recall = {enh_recall:.4f} ({enh_hits}/{len(test_true_pair_set)}), Candidates = {tot_c_enh:,}")
    print(f"Delta: {recall_gain_pp:+.2f} pp recall, Candidate Increase = {cand_increase_pct:+.2f}% (Threshold: <= +10.0%)")

    keep_fix = (enh_recall > base_recall) and (cand_increase_pct <= 10.0)
    print(f"Keep Fix Decision: {'KEEP' if keep_fix else 'REJECT'} (Recall raised without >10% candidate increase: {keep_fix})")

    # Save diagnostics
    diag_out = _REPO / "experiments" / "step3_diagnostics.json"
    diag_data = {
        "identical_pairs_total": len(identical_pairs),
        "found_identical": len(found_identical),
        "missed_identical": len(missed_identical),
        "categorization": dict(cat_counts),
        "top_tokens_in_category_a": dict(tokens_in_all_capped.most_common(20)),
        "venue_inspection": {
            "df": venue_cnt,
            "df_pct": venue_cnt / len(all_records) * 100.0,
            "capped": venue_cnt > effective_df_cap,
        },
        "max_tie_size_in_category_b": max_tie_size_seen,
        "fix_evaluated": "fallback_rarest_token_if_all_capped",
        "fix_result": {
            "base_recall": base_recall,
            "enhanced_recall": enh_recall,
            "recall_gain_pp": recall_gain_pp,
            "cand_increase_pct": cand_increase_pct,
            "decision": "KEEP" if keep_fix else "REJECT"
        }
    }
    diag_out.write_text(json.dumps(diag_data, indent=2))
    print(f"Diagnostics saved to {diag_out}", flush=True)

if __name__ == "__main__":
    main()
