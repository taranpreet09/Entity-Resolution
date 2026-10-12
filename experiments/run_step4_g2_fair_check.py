"""
experiments/run_step4_g2_fair_check.py — Phase 4c Step 4: G2 Fair Check at Matched Candidates
"""
from __future__ import annotations

import json
import math
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

def char_3gram_jaccard(s1: str, s2: str) -> float:
    c1 = canonicalise(s1); c2 = canonicalise(s2)
    if not c1 or not c2: return 0.0
    if len(c1) < 3 or len(c2) < 3:
        return 1.0 if c1 == c2 else 0.0
    g1 = {c1[i:i+3] for i in range(len(c1) - 2)}
    g2 = {c2[i:i+3] for i in range(len(c2) - 2)}
    u = g1 | g2
    return len(g1 & g2) / len(u) if u else 0.0

def token_jaccard(s1: str, s2: str) -> float:
    c1 = canonicalise(s1); c2 = canonicalise(s2)
    if not c1 or not c2: return 0.0
    t1 = set(c1.split()); t2 = set(c2.split())
    u = t1 | t2
    return len(t1 & t2) / len(u) if u else 0.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(math.isnan(x) for x in (lat1, lon1, lat2, lon2)):
        return float("nan")
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c

def main():
    print("=== Phase 4c Step 4: G2 Fair Check on 20k Train Subsample ===", flush=True)
    t0 = time.time()

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    print(f"Loaded {len(df):,} records in {time.time()-t0:.2f}s", flush=True)

    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    rid_to_idx = {r["id"]: i for i, r in enumerate(all_records)}

    # BallTree over all 150k records
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    geo_tree = BallTree(coords_rad, metric="haversine")
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS

    # G3 index
    print("Building G3 IDF index...", flush=True)
    g3 = IDFRetriever(IDFConfig(k_candidates=20, df_cap=5000, country_filter=True))
    g3.build_index(all_records)

    # 100k Train sample (seed 42) & 20k Subsample
    rng = random.Random(42)
    all_indices = list(range(len(df)))
    train_sample_indices = sorted(rng.sample(all_indices, 100_000))
    sub_q_indices = train_sample_indices[:20_000]
    sub_q_records = [all_records[i] for i in sub_q_indices]
    sub_q_rids = [r["id"] for r in sub_q_records]
    sub_q_set = set(sub_q_rids)

    # True pairs on 100k train sample
    poi_g_tr = defaultdict(list)
    for idx in train_sample_indices:
        r = all_records[idx]
        poi = r["point_of_interest"]
        if poi: poi_g_tr[poi].append(r)
    
    true_pairs_train = []
    for rows in poi_g_tr.values():
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                true_pairs_train.append((rows[i], rows[j]))

    # True pairs with at least one endpoint in 20k subsample
    sub_true_pairs = [
        (ra, rb) for ra, rb in true_pairs_train
        if ra["id"] in sub_q_set or rb["id"] in sub_q_set
    ]
    sub_canonical_pairs = {(min(ra["id"], rb["id"]), max(ra["id"], rb["id"])): (ra, rb) for ra, rb in sub_true_pairs}
    sub_denom = len(sub_canonical_pairs)
    print(f"20k Subsample true pairs (undirected): {sub_denom:,}", flush=True)

    # Pre-classify all true pairs for breakdowns
    pair_meta = {}
    for (aid, bid), (ra, rb) in sub_canonical_pairs.items():
        n_cls = classify_name(ra["name"], rb["name"])
        dist = haversine_distance(ra["latitude"], ra["longitude"], rb["latitude"], rb["longitude"])
        if math.isnan(dist):
            d_bin = "unknown"
        elif dist < 1000.0:
            d_bin = "<1km"
        elif dist <= 5000.0:
            d_bin = "1-5km"
        else:
            d_bin = ">5km"
        is_renamed_within_5km = (n_cls == "different" and dist <= 5000.0)
        pair_meta[(aid, bid)] = {
            "name_class": n_cls,
            "dist_bin": d_bin,
            "dist_m": dist,
            "is_renamed_5km": is_renamed_within_5km,
        }

    # Retrieve G3 candidates for the 20k queries (Country Filter ON, K=5)
    print("Retrieving G3 candidates (K=5, Country Filter ON)...", flush=True)
    g3_cands, g3_undir, g3_tot_cands = g3.retrieve_batch(sub_q_records, k=5, country_filter=True)
    print(f"G3 alone: hits = {len(set(sub_canonical_pairs.keys()) & g3_undir):,}/{sub_denom:,}, cands = {g3_tot_cands:,}", flush=True)

    # BallTree queries for 20k queries: K_pool = 200
    print("Querying BallTree (K_pool=200, R=5km) for 20k queries...", flush=True)
    sub_coords_rad = coords_rad[sub_q_indices]
    batch_size = 5000
    dists_list, inds_list = [], []
    for b_start in range(0, 20_000, batch_size):
        b_end = min(b_start + batch_size, 20_000)
        dists, inds = geo_tree.query(sub_coords_rad[b_start:b_end], k=201, return_distance=True)
        dists_list.append(dists)
        inds_list.append(inds)
    all_dists = np.vstack(dists_list)
    all_inds = np.vstack(inds_list)

    # Feature weights for G2: (w1..w6) = (2.0, 1.5, 1.0, 1.0, 0.5, 0.5)
    w1, w2, w3, w4, w5, w6 = 2.0, 1.5, 1.0, 1.0, 0.5, 0.5

    # Storage for candidates
    g1_cands_by_k = {4: defaultdict(list), 12: defaultdict(list), 50: defaultdict(list)}
    g1_undir_by_k = {4: set(), 12: set(), 50: set()}
    g2_cands_by_m = {3: defaultdict(list), 5: defaultdict(list), 10: defaultdict(list)}
    g2_undir_by_m = {3: set(), 5: set(), 10: set()}

    print("Extracting G1 and G2 reranked candidates...", flush=True)
    t_start_rerank = time.time()
    for q_idx_local, q_rec in enumerate(sub_q_records):
        qid = q_rec["id"]
        q_name = q_rec["name"]
        q_country = q_rec["country"]
        row_inds = all_inds[q_idx_local]
        row_dists = all_dists[q_idx_local]

        # Valid candidates within 5km
        g1_count = 0
        scored_cands = []
        for cand_idx, d_rad in zip(row_inds, row_dists):
            cand_id = rids[cand_idx]
            if cand_id == qid:
                continue
            if d_rad > r5km_rad:
                continue

            # G1 collections
            if g1_count < 4:
                g1_cands_by_k[4][qid].append(cand_id)
                g1_undir_by_k[4].add((min(qid, cand_id), max(qid, cand_id)))
            if g1_count < 12:
                g1_cands_by_k[12][qid].append(cand_id)
                g1_undir_by_k[12].add((min(qid, cand_id), max(qid, cand_id)))
            if g1_count < 50:
                g1_cands_by_k[50][qid].append(cand_id)
                g1_undir_by_k[50].add((min(qid, cand_id), max(qid, cand_id)))
            g1_count += 1

            # G2 6-feature scoring
            c_rec = all_records[cand_idx]
            d_m = d_rad * EARTH_RADIUS_METERS
            f1 = g3.score_pair(q_name, c_rec["name"])
            f2 = char_3gram_jaccard(q_name, c_rec["name"])
            f3 = token_jaccard(q_name, c_rec["name"])
            f4 = 1.0 if (q_country and c_rec["country"] and q_country == c_rec["country"]) else 0.0
            f5 = 0.0  # category match
            f6 = -math.log(1.0 + d_m)

            score = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5 + w6*f6
            scored_cands.append((score, cand_id))

        # Sort G2 by score descending
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        for m in [3, 5, 10]:
            top_m = [c[1] for c in scored_cands[:m]]
            g2_cands_by_m[m][qid] = top_m
            for c in top_m:
                g2_undir_by_m[m].add((min(qid, c), max(qid, c)))

    print(f"Candidate extraction done in {time.time()-t_start_rerank:.2f}s", flush=True)

    # Compute metrics for standalone and union with G3
    def evaluate_config(name, undir_cands, cand_map_by_q):
        tot_cands = sum(len(cand_map_by_q[q]) for q in sub_q_rids)
        hits = set(sub_canonical_pairs.keys()) & undir_cands
        rec = len(hits) / sub_denom

        # Breakdown by name class
        by_name = defaultdict(lambda: {"total": 0, "hits": 0})
        # Breakdown by dist bin
        by_dist = defaultdict(lambda: {"total": 0, "hits": 0})
        # Renamed within 5km
        renamed_5km_total = 0
        renamed_5km_hits = 0

        for pair, meta in pair_meta.items():
            n_cls = meta["name_class"]
            d_bin = meta["dist_bin"]
            by_name[n_cls]["total"] += 1
            by_dist[d_bin]["total"] += 1
            if meta["is_renamed_5km"]:
                renamed_5km_total += 1

            if pair in hits:
                by_name[n_cls]["hits"] += 1
                by_dist[d_bin]["hits"] += 1
                if meta["is_renamed_5km"]:
                    renamed_5km_hits += 1

        name_rec = {k: v["hits"] / v["total"] for k, v in by_name.items()}
        dist_rec = {k: v["hits"] / v["total"] for k, v in by_dist.items()}
        renamed_rec = renamed_5km_hits / renamed_5km_total if renamed_5km_total else 0.0

        return {
            "name": name,
            "total_cands": tot_cands,
            "cpq": tot_cands / len(sub_q_rids),
            "hits": len(hits),
            "recall": rec,
            "name_recall": name_rec,
            "name_counts": {k: f"{v['hits']}/{v['total']}" for k, v in by_name.items()},
            "dist_recall": dist_rec,
            "dist_counts": {k: f"{v['hits']}/{v['total']}" for k, v in by_dist.items()},
            "renamed_5km_hits": renamed_5km_hits,
            "renamed_5km_total": renamed_5km_total,
            "renamed_5km_recall": renamed_rec,
            "hits_set": hits,
        }

    # Evaluate all configurations
    configs = {}
    # Standalone
    for k in [4, 12]:
        configs[f"G1 (K={k})"] = evaluate_config(f"G1 (K={k})", g1_undir_by_k[k], g1_cands_by_k[k])
    for m in [3, 5, 10]:
        configs[f"G2 (m={m})"] = evaluate_config(f"G2 (m={m})", g2_undir_by_m[m], g2_cands_by_m[m])

    # Union with G3
    for k in [4, 12]:
        union_undir = g1_undir_by_k[k] | g3_undir
        union_map = {q: list(set(g1_cands_by_k[k][q]) | set(g3_cands.get(q, []))) for q in sub_q_rids}
        configs[f"G1 (K={k}) + G3"] = evaluate_config(f"G1 (K={k}) + G3", union_undir, union_map)

    for m in [3, 5, 10]:
        union_undir = g2_undir_by_m[m] | g3_undir
        union_map = {q: list(set(g2_cands_by_m[m][q]) | set(g3_cands.get(q, []))) for q in sub_q_rids}
        configs[f"G2 (m={m}) + G3"] = evaluate_config(f"G2 (m={m}) + G3", union_undir, union_map)

    # Paired Bootstrap CI (1,000 resamples over true pairs)
    print("\nComputing Paired Bootstrap CIs (1,000 resamples)...", flush=True)
    np_rng = np.random.RandomState(42)
    pairs_list = list(sub_canonical_pairs.keys())
    N_pairs = len(pairs_list)

    comparisons = [
        ("G1 (K=4) + G3", "G2 (m=3) + G3"),
        ("G1 (K=4) + G3", "G2 (m=5) + G3"),
        ("G1 (K=12) + G3", "G2 (m=10) + G3"),
    ]

    boot_results = {}
    for cfg_a, cfg_b in comparisons:
        hits_a = np.array([1 if p in configs[cfg_a]["hits_set"] else 0 for p in pairs_list])
        hits_b = np.array([1 if p in configs[cfg_b]["hits_set"] else 0 for p in pairs_list])
        diff = hits_a - hits_b
        observed_delta = float(np.mean(diff))

        # Bootstrap resamples
        boot_diffs = []
        for _ in range(1000):
            sample_idx = np_rng.choice(N_pairs, size=N_pairs, replace=True)
            boot_diffs.append(float(np.mean(diff[sample_idx])))
        
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        p_val = float(np.mean(np.array(boot_diffs) <= 0.0))

        boot_results[f"{cfg_a} vs {cfg_b}"] = {
            "observed_delta_pp": observed_delta * 100.0,
            "ci_lower_pp": ci_lower * 100.0,
            "ci_upper_pp": ci_upper * 100.0,
            "p_value": p_val,
        }
        print(f"  {cfg_a} vs {cfg_b}: Delta = {observed_delta*100.0:+.2f} pp, 95% CI: [{ci_lower*100.0:+.2f} pp, {ci_upper*100.0:+.2f} pp], p={p_val:.4f}")

    # Summary table
    print("\n--- Summary Table: Matched Candidates on 20k Subsample ---")
    print(f"{'Configuration':<25} {'Cands':>8} {'CPQ':>6} {'Hits':>6} {'Recall':>8} {'Renamed<=5km':>12}")
    print("-" * 72)
    for cname in ["G1 (K=4) + G3", "G2 (m=3) + G3", "G2 (m=5) + G3", "G1 (K=12) + G3", "G2 (m=10) + G3"]:
        cfg = configs[cname]
        print(f"{cname:<25} {cfg['total_cands']:>8,} {cfg['cpq']:>6.1f} {cfg['hits']:>6,} {cfg['recall']*100:>7.2f}% {cfg['renamed_5km_recall']*100:>11.2f}%")

    # Clean hits_set before serialization
    for k in configs:
        del configs[k]["hits_set"]

    out_data = {
        "subsample_n": 20_000,
        "true_pair_denom": sub_denom,
        "configs": configs,
        "paired_bootstrap_cis": boot_results,
    }
    out_path = _REPO / "experiments" / "step4_g2_fair_check_results.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved Step 4 results to {out_path}", flush=True)

if __name__ == "__main__":
    main()

