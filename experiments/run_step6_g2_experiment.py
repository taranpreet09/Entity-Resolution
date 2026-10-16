"""
scratch/run_step6_g2_experiment.py — Phase 4b Step 6: G2 Cost Experiment & Pre-Registration Evaluation
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
from sklearn.neighbors import BallTree

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ber.canon.normalise import canonicalise
from ber.retrieve.geo_blocking import EARTH_RADIUS_METERS
from ber.retrieve.idf import IDFRetriever, IDFConfig, infer_country_from_coords
from ber_datasets.splits.splitter import EntityDisjointSplitter

def char_3gram_jaccard(s1: str, s2: str) -> float:
    c1 = canonicalise(s1); c2 = canonicalise(s2)
    if not c1 or not c2: return 0.0
    if len(c1) < 3 or len(c2) < 3:
        return 1.0 if c1 == c2 else 0.0
    g1 = {c1[i:i+3] for i in range(len(c1) - 2)}
    g2 = {c2[i:i+3] for i in range(len(c2) - 2)}
    u = g1 | g2
    return len(g1 & g2) / len(u) if u else 0.0

def main():
    print("=== Step 6: G2 Cost Experiment (K_pool=200, R=5km, 6 Features) ===", flush=True)
    t_start = time.time()

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    rid_to_idx = {r["id"]: i for i, r in enumerate(all_records)}

    # Build G3 index for IDF scoring & retrieval (K=5, Country ON)
    print("Building G3 Index...", flush=True)
    g3 = IDFRetriever(IDFConfig(k_candidates=5, df_cap=5000, country_filter=True))
    g3.build_index(all_records)

    # BallTree
    print("Building BallTree...", flush=True)
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    geo_tree = BallTree(coords_rad, metric="haversine")
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS

    # Splits
    splitter = EntityDisjointSplitter(ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
    splits = splitter.split(rids, [r["point_of_interest"] for r in all_records])

    # 100k Train sample (seed 42)
    rng = random.Random(42)
    all_indices = list(range(len(df)))
    train_sample_indices = sorted(rng.sample(all_indices, 100_000))
    train_sample_records = [all_records[i] for i in train_sample_indices]
    train_sample_rids = [r["id"] for r in train_sample_records]

    # Val split records
    val_rids = [rid for rid, s in splits.items() if s == "val"]
    val_records = [all_records[rid_to_idx[rid]] for rid in val_rids]
    val_indices = [rid_to_idx[rid] for rid in val_rids]

    # Ground truth
    # Train true pairs
    poi_g_tr = defaultdict(list)
    for r in train_sample_records:
        poi = r["point_of_interest"]
        if poi: poi_g_tr[poi].append(r["id"])
    tr_true_pairs = set()
    for ids in poi_g_tr.values():
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                tr_true_pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    tr_denom = len(tr_true_pairs)
    print(f"Train true pairs: {tr_denom:,}", flush=True)

    # Val true pairs
    poi_g_val = defaultdict(list)
    for r in val_records:
        poi = r["point_of_interest"]
        if poi: poi_g_val[poi].append(r["id"])
    val_true_pairs = set()
    for ids in poi_g_val.values():
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                val_true_pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    val_denom = len(val_true_pairs)
    print(f"Val true pairs: {val_denom:,}", flush=True)

    # Feature weights
    w1, w2, w3, w4, w5, w6 = 2.0, 1.5, 1.0, 1.0, 0.5, 0.5

    def run_g2_pipeline(query_records, query_indices, qids_list, is_train=True):
        q_coords_rad = coords_rad[query_indices]
        N_q = len(query_records)
        print(f"Running G2 pool + rerank on {N_q:,} queries ({'TRAIN' if is_train else 'VAL'})...", flush=True)
        
        # G3 candidates
        t_g3_0 = time.time()
        g3_cands, g3_undir, _ = g3.retrieve_batch(query_records, k=5, country_filter=True)
        print(f"  G3 retrieval took {time.time()-t_g3_0:.2f}s", flush=True)

        # Baseline G1 (K=50) candidates
        t_g1_0 = time.time()
        g1_50_cands = defaultdict(list)
        g1_50_undir = set()
        tot_g1_cands = 0
        batch_size = 5000
        
        # We also need pool K_pool = 200
        g2_reranked_by_q = {}
        t_pool_tot = 0.0
        t_rerank_tot = 0.0

        for b_start in range(0, N_q, batch_size):
            b_end = min(b_start + batch_size, N_q)
            b_coords = q_coords_rad[b_start:b_end]
            b_qids = qids_list[b_start:b_end]
            b_qrecs = query_records[b_start:b_end]

            t0_b = time.time()
            dists, inds = geo_tree.query(b_coords, k=201, return_distance=True)
            t_pool_tot += time.time() - t0_b

            t0_rr = time.time()
            for q_rec, row_inds, row_dists in zip(b_qrecs, inds, dists):
                qid = q_rec["id"]
                q_name = q_rec["name"]
                q_country = q_rec["country"]
                
                scored_cands = []
                g1_cnt = 0
                for cand_idx, d_rad in zip(row_inds, row_dists):
                    cand_id = rids[cand_idx]
                    if cand_id == qid: continue
                    if d_rad > r5km_rad: continue

                    # G1 baseline collection (top 50)
                    if g1_cnt < 50:
                        g1_50_cands[qid].append(cand_id)
                        g1_50_undir.add((min(qid, cand_id), max(qid, cand_id)))
                        tot_g1_cands += 1
                        g1_cnt += 1

                    # G2 scoring for all pool candidates up to 200
                    c_rec = all_records[cand_idx]
                    d_m = d_rad * EARTH_RADIUS_METERS

                    f1 = g3.score_pair(q_name, c_rec["name"])
                    f2 = char_3gram_jaccard(q_name, c_rec["name"])
                    f3 = 0.0
                    f4 = 1.0 if (q_country and c_rec["country"] and q_country == c_rec["country"]) else 0.0
                    f5 = 0.0
                    f6 = -math.log(1.0 + d_m)

                    score = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5 + w6 * f6
                    scored_cands.append((score, cand_id))

                scored_cands.sort(key=lambda x: x[0], reverse=True)
                g2_reranked_by_q[qid] = [c[1] for c in scored_cands]
            t_rerank_tot += time.time() - t0_rr

        print(f"  Pool query took {t_pool_tot:.2f}s, 6-feature scoring took {t_rerank_tot:.2f}s.", flush=True)

        # Baseline G1(K=50) + G3
        baseline_union_undir = g1_50_undir | g3_undir
        baseline_union_cands_cnt = sum(len(set(g1_50_cands[q]) | set(g3_cands.get(q, []))) for q in qids_list)
        
        results_by_m = {}
        for m in [3, 5, 10]:
            g2_m_undir = set()
            g2_m_cands_cnt = 0
            union_m_cands_cnt = 0
            union_m_undir = set(g3_undir)

            for q in qids_list:
                top_m = g2_reranked_by_q.get(q, [])[:m]
                g2_m_cands_cnt += len(top_m)
                for c in top_m:
                    g2_m_undir.add((min(q, c), max(q, c)))
                    union_m_undir.add((min(q, c), max(q, c)))
                union_m_cands_cnt += len(set(top_m) | set(g3_cands.get(q, [])))

            results_by_m[m] = {
                "g2_m_undir": g2_m_undir,
                "g2_m_cands": g2_m_cands_cnt,
                "union_m_undir": union_m_undir,
                "union_m_cands": union_m_cands_cnt,
            }

        return {
            "g1_50_undir": g1_50_undir,
            "g1_50_cands": tot_g1_cands,
            "baseline_union_undir": baseline_union_undir,
            "baseline_union_cands": baseline_union_cands_cnt,
            "g3_undir": g3_undir,
            "results_by_m": results_by_m,
        }

    # 1. RUN ON TRAIN SAMPLE TO CHOOSE m
    print("\n--- Step 6.1: Evaluating G2(m) + G3 on TRAIN SAMPLE ---", flush=True)
    tr_res = run_g2_pipeline(train_sample_records, train_sample_indices, train_sample_rids, is_train=True)
    
    base_tr_hits = len(tr_true_pairs & tr_res["baseline_union_undir"])
    base_tr_rec = base_tr_hits / tr_denom
    base_g1_cands = tr_res["g1_50_cands"]
    print(f"\nBaseline G1(K=50) + G3 on Train: {base_tr_hits:,}/{tr_denom:,} ({base_tr_rec*100:.2f}%), G1 Cands = {base_g1_cands:,}, Union Cands = {tr_res['baseline_union_cands']:,}")

    m_eval = {}
    chosen_m = None
    success_criteria_met = False

    print("\nSweep over m in {3, 5, 10} on Train:")
    header = f"{'m':3s} | {'Union Cands':12s} | {'% of G1 Cands':15s} | {'True Pairs':10s} | {'Pair Recall':12s} | {'Delta vs Base':14s} | Success Criteria"
    print(header)
    print("-" * len(header))

    for m in [3, 5, 10]:
        m_data = tr_res["results_by_m"][m]
        u_hits = len(tr_true_pairs & m_data["union_m_undir"])
        u_rec = u_hits / tr_denom
        delta_pp = (u_rec - base_tr_rec) * 100.0
        g2_cand_pct = (m_data["g2_m_cands"] / base_g1_cands) * 100.0

        # Success criterion:
        # G2(m) + G3 within 1.0 pp of G1 + G3 with <= 25% of G1 candidates,
        # OR higher recall at equal candidates.
        is_success = (delta_pp >= -1.0) and (g2_cand_pct <= 25.0)

        m_eval[m] = {
            "m": m,
            "union_candidates": m_data["union_m_cands"],
            "g2_candidates": m_data["g2_m_cands"],
            "pct_of_g1_cands": g2_cand_pct,
            "true_pairs": u_hits,
            "pair_recall": u_rec,
            "delta_pp": delta_pp,
            "success": is_success,
        }

        print(f"{m:3d} | {m_data['union_m_cands']:12,d} | {g2_cand_pct:14.1f}% | {u_hits:6,d}/{tr_denom:,d} | {u_rec*100:10.2f}% | {delta_pp:+12.2f} pp | {'PASSED' if is_success else 'FAILED'}")

    # Choose m on train
    # Best m that passes success criteria, or highest recall if none passes
    passed_ms = [m for m, d in m_eval.items() if d["success"]]
    if passed_ms:
        chosen_m = max(passed_ms) # Pick largest passing m for max recall
        success_criteria_met = True
    else:
        # If none pass, pick m=5 as standard balanced setting
        chosen_m = 5
        success_criteria_met = False

    print(f"\nCHOSEN m ON TRAIN: m={chosen_m} (Success Criteria: {'MET' if success_criteria_met else 'NOT MET'})", flush=True)

    # 2. EVALUATE CHOSEN m ONCE ON VAL
    print(f"\n--- Step 6.2: Evaluating Chosen m={chosen_m} ONCE on VAL SPLIT ---", flush=True)
    val_res = run_g2_pipeline(val_records, val_indices, val_rids, is_train=False)

    base_val_hits = len(val_true_pairs & val_res["baseline_union_undir"])
    base_val_rec = base_val_hits / val_denom
    base_val_g1_cands = val_res["g1_50_cands"]

    val_m_data = val_res["results_by_m"][chosen_m]
    val_u_hits = len(val_true_pairs & val_m_data["union_m_undir"])
    val_u_rec = val_u_hits / val_denom
    val_delta_pp = (val_u_rec - base_val_rec) * 100.0
    val_cand_pct = (val_m_data["g2_m_cands"] / base_val_g1_cands) * 100.0
    val_success = (val_delta_pp >= -1.0) and (val_cand_pct <= 25.0)

    print("\nValidation Evaluation Results:")
    print(f"  Baseline G1(K=50) + G3 : {base_val_hits:,} / {val_denom:,} ({base_val_rec*100:.2f}%), G1 Cands = {base_val_g1_cands:,}, Union Cands = {val_res['baseline_union_cands']:,}")
    print(f"  G2(m={chosen_m}) + G3        : {val_u_hits:,} / {val_denom:,} ({val_u_rec*100:.2f}%), G2 Cands = {val_m_data['g2_m_cands']:,} ({val_cand_pct:.1f}% of G1), Union Cands = {val_m_data['union_m_cands']:,}")
    print(f"  Delta on Validation   : {val_delta_pp:+.2f} pp")
    print(f"  Validation Success    : {'SUCCESS' if val_success else 'FAILED'}")

    # Save results
    out_step6 = _REPO / "experiments" / "step6_g2_cost_results.json"
    out_data = {
        "train_eval": m_eval,
        "chosen_m": chosen_m,
        "success_criteria_met_on_train": success_criteria_met,
        "val_eval": {
            "chosen_m": chosen_m,
            "baseline_true_pairs": base_val_hits,
            "baseline_pair_recall": base_val_rec,
            "baseline_g1_candidates": base_val_g1_cands,
            "baseline_union_candidates": val_res["baseline_union_cands"],
            "g2_true_pairs": val_u_hits,
            "g2_pair_recall": val_u_rec,
            "g2_candidates": val_m_data["g2_m_cands"],
            "g2_pct_of_g1_cands": val_cand_pct,
            "union_candidates": val_m_data["union_m_cands"],
            "delta_pp": val_delta_pp,
            "success": val_success,
        },
        "total_runtime_s": time.time() - t_start,
    }
    out_step6.write_text(json.dumps(out_data, indent=2))
    print(f"Saved Step 6 results to {out_step6}", flush=True)

if __name__ == "__main__":
    main()

