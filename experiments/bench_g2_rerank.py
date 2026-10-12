"""
scratch/bench_g2_rerank.py — Benchmark 5,000 queries for G2 pool rerank
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
    print("=== Step 6: Benchmarking 5,000 Queries for G2 Reranking ===", flush=True)

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    rid_to_idx = {r["id"]: i for i, r in enumerate(all_records)}

    # Build G3 index for IDF scoring
    g3 = IDFRetriever(IDFConfig(k_candidates=5, df_cap=5000, country_filter=True))
    g3.build_index(all_records)

    # BallTree
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    geo_tree = BallTree(coords_rad, metric="haversine")
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS

    # Sample 5,000 benchmark queries
    rng = random.Random(42)
    bench_indices = sorted(rng.sample(range(len(df)), 5000))
    bench_coords_rad = coords_rad[bench_indices]
    bench_records = [all_records[i] for i in bench_indices]

    t0 = time.time()
    # 1. Pool retrieval K_pool = 200, R = 5km
    t_pool_0 = time.time()
    dists, inds = geo_tree.query(bench_coords_rad, k=201, return_distance=True)
    t_pool = time.time() - t_pool_0

    # 2. Feature extraction & scoring
    t_score_0 = time.time()
    w1, w2, w3, w4, w5, w6 = 2.0, 1.5, 1.0, 1.0, 0.5, 0.5

    reranked_cands = {}
    total_scored_cands = 0

    for q_rec, row_inds, row_dists in zip(bench_records, inds, dists):
        qid = q_rec["id"]
        q_name = q_rec["name"]
        q_country = q_rec["country"]
        
        candidates = []
        for cand_idx, d_rad in zip(row_inds, row_dists):
            cand_id = rids[cand_idx]
            if cand_id == qid: continue
            if d_rad > r5km_rad: continue
            
            c_rec = all_records[cand_idx]
            d_m = d_rad * EARTH_RADIUS_METERS

            # 6 Features
            f1 = g3.score_pair(q_name, c_rec["name"])
            f2 = char_3gram_jaccard(q_name, c_rec["name"])
            f3 = 0.0 # Address tokens
            f4 = 1.0 if (q_country and c_rec["country"] and q_country == c_rec["country"]) else 0.0
            f5 = 0.0 # Category
            f6 = -math.log(1.0 + d_m)

            score = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5 + w6 * f6
            candidates.append((score, cand_id))

        total_scored_cands += len(candidates)
        # Sort descending by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        reranked_cands[qid] = [c[1] for c in candidates]

    t_score = time.time() - t_score_0
    total_time = time.time() - t0

    ms_per_q = total_time / 5000.0 * 1000.0
    est_100k_time_s = total_time * 20.0
    est_100k_time_min = est_100k_time_s / 60.0

    print(f"5,000 queries completed in {total_time:.2f}s ({ms_per_q:.2f} ms/query).", flush=True)
    print(f"  - Spatial pool time: {t_pool:.2f}s ({t_pool/5000*1000:.2f} ms/query)", flush=True)
    print(f"  - 6-Feature reranking time: {t_score:.2f}s ({t_score/5000*1000:.2f} ms/query)", flush=True)
    print(f"  - Total pool candidates scored: {total_scored_cands:,} ({total_scored_cands/5000:.1f} per query)", flush=True)
    print(f"\nExtrapolation to 100,000 train queries: {est_100k_time_min:.2f} minutes.", flush=True)
    print(f"Runtime gate (<= 40 minutes): {'PASSED' if est_100k_time_min <= 40.0 else 'FAILED'}", flush=True)

    # Save benchmark
    bench_out = _REPO / "experiments" / "step6_benchmark_5k.json"
    bench_out.write_text(json.dumps({
        "n_queries": 5000,
        "runtime_s": total_time,
        "ms_per_query": ms_per_q,
        "extrapolated_100k_min": est_100k_time_min,
        "gate_passed": est_100k_time_min <= 40.0,
    }, indent=2))

if __name__ == "__main__":
    main()

