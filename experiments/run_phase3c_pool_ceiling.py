"""
experiments/run_phase3c_pool_ceiling.py — Phase 3c: Pool Ceiling & True Pair Distance Diagnostics
===================================================================================================
Executes:
  STEP 2: Train-sample diagnostics (fixed 100k train sample, seed 42)
    (a) Distance distribution of ALL true pairs across bins:
        <50m, <200m, <1km, <5km, <10km, >=10km, missing coords
        by name class (identical, token_reorder, partial_overlap, different, empty)
        loaded via DuckDB.
    (b) G1 (K=50, R=5km) recovery rate by name class and distance bin.
        Evaluate if "geo recovers renamed venues" is supported or unsupported.

  STEP 3: Pool ceiling sweep (K_pool in {50, 100, 200}, R=5km)
    Streaming in query batches on a fixed 20k-query subsample (full index).
    Reports: pair recall (primary), query recall, candidates, dense vs sparse,
             name class recovery, runtime, peak RSS.
    GATE: Pair Recall(K_pool=200) - Pair Recall(K_pool=50) >= +3.0 pp.
          Paired bootstrap CI on the difference.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ber.canon.normalise import canonicalise
from ber.eval.metrics_dedup import pair_recall, directed_slot_recall
from ber.retrieve.geo_blocking import EARTH_RADIUS_METERS

_STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "at", "by", "for", "with", "de", "la", "le"}

def classify_name(name_a: str, name_b: str) -> str:
    ca = canonicalise(name_a); cb = canonicalise(name_b)
    if not ca or not cb:
        return "empty"
    if ca == cb:
        return "identical"
    toks_a = set(ca.split()); toks_b = set(cb.split())
    if toks_a == toks_b:
        return "token_reorder"
    filt_a = toks_a - _STOPWORDS; filt_b = toks_b - _STOPWORDS
    if filt_a and filt_a == filt_b:
        return "token_reorder"
    union = toks_a | toks_b
    jacc = len(toks_a & toks_b) / len(union) if union else 0.0
    if jacc >= 0.5:
        return "partial_overlap"
    return "different"

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(math.isnan(x) for x in (lat1, lon1, lat2, lon2)):
        return float("nan")
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c

def get_distance_bin(d: float) -> str:
    if math.isnan(d):
        return "missing_coords"
    if d < 50.0:
        return "<50m"
    elif d < 200.0:
        return "<200m"
    elif d < 1000.0:
        return "<1km"
    elif d < 5000.0:
        return "<5km"
    elif d < 10000.0:
        return "<10km"
    else:
        return ">=10km"

DISTANCE_BINS = ["<50m", "<200m", "<1km", "<5km", "<10km", ">=10km", "missing_coords"]
NAME_CLASSES = ["identical", "token_reorder", "partial_overlap", "different", "empty"]

def main():
    print("=== Phase 3c: Pool Ceiling & True Pair Distance Diagnostics ===", flush=True)
    t_start = time.time()
    tracemalloc.start()

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    if not coord_path.exists():
        raise FileNotFoundError(f"Missing {coord_path}")

    # Load via DuckDB
    con = duckdb.connect()
    print("Loading id, name, latitude, longitude, point_of_interest via DuckDB...", flush=True)
    df = con.execute(f"""
        SELECT id, name, latitude, longitude, point_of_interest
        FROM '{coord_path.as_posix()}'
    """).fetchdf()

    print(f"Loaded {len(df):,} records via DuckDB.", flush=True)

    # 100k Train sample (seed 42)
    rng = random.Random(42)
    all_indices = list(range(len(df)))
    sample_indices = sorted(rng.sample(all_indices, min(100_000, len(df))))
    sample_df = df.iloc[sample_indices].copy().reset_index(drop=True)
    sample_ids = set(sample_df["id"])
    print(f"Sampled {len(sample_df):,} train queries (seed 42).", flush=True)

    # Fast true pairs lookup using lists of dicts
    sample_records = sample_df.to_dict(orient="records")
    poi_groups = defaultdict(list)
    for rec in sample_records:
        poi = rec["point_of_interest"]
        if poi:
            poi_groups[poi].append(rec)

    true_pairs_sample = []
    for poi, rows in poi_groups.items():
        if len(rows) > 1:
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    true_pairs_sample.append((rows[i], rows[j]))

    print(f"Total true unordered pairs on 100k train sample: {len(true_pairs_sample):,}", flush=True)

    # STEP 2 (a): Distance distribution of ALL true pairs by name class
    print("\n=== STEP 2 (a): Distance Distribution of ALL True Pairs by Name Class ===", flush=True)
    crosstab_counts = defaultdict(lambda: defaultdict(int))
    for ra, rb in true_pairs_sample:
        d = haversine_distance(ra["latitude"], ra["longitude"], rb["latitude"], rb["longitude"])
        dbin = get_distance_bin(d)
        ncat = classify_name(ra["name"], rb["name"])
        crosstab_counts[ncat][dbin] += 1

    print("\nDistance Bin Distribution Across Name Classes (Counts):", flush=True)
    header = f"{'Name Class':20s} | " + " | ".join(f"{b:>8s}" for b in DISTANCE_BINS) + " | Total"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for ncat in NAME_CLASSES:
        row_str = f"{ncat:20s} | "
        row_tot = 0
        for dbin in DISTANCE_BINS:
            cnt = crosstab_counts[ncat][dbin]
            row_tot += cnt
            row_str += f"{cnt:8d} | "
        row_str += f"{row_tot:8d}"
        print(row_str, flush=True)

    tot_line = f"{'Total':20s} | "
    grand_tot = 0
    for dbin in DISTANCE_BINS:
        col_tot = sum(crosstab_counts[nc][dbin] for nc in NAME_CLASSES)
        grand_tot += col_tot
        tot_line += f"{col_tot:8d} | "
    tot_line += f"{grand_tot:8d}"
    print("-" * len(header), flush=True)
    print(tot_line, flush=True)

    # Index coordinates in BallTree
    print("\nIndexing coordinates in BallTree (Haversine metric)...", flush=True)
    valid_coords = df.dropna(subset=["latitude", "longitude"]).copy()
    coords_rad = np.radians(valid_coords[["latitude", "longitude"]].values)
    rids = valid_coords["id"].tolist()
    rid_to_idx = {rid: i for i, rid in enumerate(rids)}

    tree = BallTree(coords_rad, metric="haversine")
    print(f"Indexed {len(valid_coords):,} coordinates into BallTree.", flush=True)

    # Fixed 20k-query subsample for pool ceiling evaluation
    subsample_n = 20_000
    sub_q_indices = sample_indices[:subsample_n]
    sub_q_rids = [rids[i] for i in sub_q_indices]
    sub_q_set = set(sub_q_rids)

    # Density calculation on the 20k queries
    print("Computing local density on subsample queries (1km radius count)...", flush=True)
    r1km_rad = 1000.0 / EARTH_RADIUS_METERS
    sub_q_coords_rad = coords_rad[sub_q_indices]
    counts_1km = tree.query_radius(sub_q_coords_rad, r=r1km_rad, count_only=True)
    p90_threshold = float(np.percentile(counts_1km, 90))
    dense_q_set = {sub_q_rids[i] for i, cnt in enumerate(counts_1km) if cnt >= p90_threshold}
    print(f"Local density P90 threshold: {p90_threshold:.1f} records within 1km. Dense queries: {len(dense_q_set):,}", flush=True)

    sub_true_pairs_sample = [
        (ra, rb) for ra, rb in true_pairs_sample
        if ra["id"] in sub_q_set or rb["id"] in sub_q_set
    ]
    sub_canonical_true = [(min(ra["id"], rb["id"]), max(ra["id"], rb["id"])) for ra, rb in sub_true_pairs_sample]
    sub_canonical_set = set(sub_canonical_true)
    print(f"Subsample true unordered pairs: {len(sub_canonical_set):,}", flush=True)

    sub_targets_by_q = defaultdict(set)
    for a, b in sub_canonical_true:
        if a in sub_q_set: sub_targets_by_q[a].add(b)
        if b in sub_q_set: sub_targets_by_q[b].add(a)

    sub_q_with_targets = [q for q in sub_q_rids if q in sub_targets_by_q]
    print(f"Subsample answerable queries: {len(sub_q_with_targets):,}", flush=True)

    # STEP 3 & STEP 2 (b): Run retrieval for K_pool in {50, 100, 200} with R = 5km
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS
    results_by_k = {}
    BATCH = 5000

    for k_val in [50, 100, 200]:
        t0 = time.time()
        retrieved_pairs_undir = set()
        cands_by_q = defaultdict(set)
        total_candidates = 0

        # Query tree.query directly in C++
        k_query = min(k_val + 1, len(coords_rad))
        for b_start in range(0, len(sub_q_indices), BATCH):
            b_coords = sub_q_coords_rad[b_start:b_start + BATCH]
            b_qids = sub_q_rids[b_start:b_start + BATCH]

            dists, inds = tree.query(b_coords, k=k_query, return_distance=True)
            for q_id, row_inds, row_dists in zip(b_qids, inds, dists):
                cnt = 0
                for cand_idx, d_rad in zip(row_inds, row_dists):
                    cand_id = rids[cand_idx]
                    if cand_id == q_id:
                        continue
                    if d_rad > r5km_rad:
                        continue
                    retrieved_pairs_undir.add((min(q_id, cand_id), max(q_id, cand_id)))
                    cands_by_q[q_id].add(cand_id)
                    cnt += 1
                    if cnt >= k_val:
                        break
                total_candidates += cnt

        # Primary pair recall
        p_hits = len(sub_canonical_set & retrieved_pairs_undir)
        p_recall = p_hits / len(sub_canonical_set) if sub_canonical_set else 0.0

        # Query recall
        q_hits = sum(1 for q in sub_q_with_targets if (cands_by_q[q] & sub_targets_by_q[q]))
        q_recall = q_hits / len(sub_q_with_targets) if sub_q_with_targets else 0.0

        # Dense vs sparse
        dense_hits, dense_tot = 0, 0
        sparse_hits, sparse_tot = 0, 0
        for ra, rb in sub_true_pairs_sample:
            p_tuple = (min(ra["id"], rb["id"]), max(ra["id"], rb["id"]))
            is_dense = (ra["id"] in dense_q_set) or (rb["id"] in dense_q_set)
            hit = p_tuple in retrieved_pairs_undir
            if is_dense:
                dense_tot += 1
                if hit: dense_hits += 1
            else:
                sparse_tot += 1
                if hit: sparse_hits += 1

        dense_recall = dense_hits / dense_tot if dense_tot else 0.0
        sparse_recall = sparse_hits / sparse_tot if sparse_tot else 0.0

        # Recovery by name class
        name_hits = Counter(); name_tots = Counter()
        for ra, rb in sub_true_pairs_sample:
            p_tuple = (min(ra["id"], rb["id"]), max(ra["id"], rb["id"]))
            ncat = classify_name(ra["name"], rb["name"])
            name_tots[ncat] += 1
            if p_tuple in retrieved_pairs_undir:
                name_hits[ncat] += 1

        rec_time = time.time() - t0
        _, peak_mem = tracemalloc.get_traced_memory()

        results_by_k[k_val] = {
            "pair_recall": p_recall,
            "pair_hits": p_hits,
            "query_recall": q_recall,
            "query_hits": q_hits,
            "total_candidates": total_candidates,
            "cpq_mean": total_candidates / len(sub_q_indices),
            "dense_pair_recall": dense_recall,
            "sparse_pair_recall": sparse_recall,
            "name_recovery": {nc: (name_hits[nc] / name_tots[nc] if name_tots[nc] else 0.0) for nc in NAME_CLASSES},
            "runtime_s": rec_time,
            "peak_rss_mb": peak_mem / (1024 * 1024),
            "retrieved_undir": retrieved_pairs_undir,
        }

        print(f"K_pool={k_val:3d}: Pair Recall = {p_recall:.4f} ({p_hits:,}/{len(sub_canonical_set):,}), "
              f"Query Recall = {q_recall:.4f}, Cands = {total_candidates:,}, "
              f"Dense = {dense_recall:.4f}, Sparse = {sparse_recall:.4f}, Time = {rec_time:.2f}s, Peak Mem = {peak_mem/(1024*1024):.1f}MB", flush=True)

    # STEP 2 (b): G1 (K=50) recovery rate table
    print("\n=== STEP 2 (b): G1 (K=50, R=5km) Recovery Rate by Name Class & Distance Bin ===", flush=True)
    g1_retr = results_by_k[50]["retrieved_undir"]
    bin_name_hits = defaultdict(lambda: defaultdict(int))
    bin_name_tots = defaultdict(lambda: defaultdict(int))

    for ra, rb in sub_true_pairs_sample:
        p_tuple = (min(ra["id"], rb["id"]), max(ra["id"], rb["id"]))
        d = haversine_distance(ra["latitude"], ra["longitude"], rb["latitude"], rb["longitude"])
        dbin = get_distance_bin(d)
        ncat = classify_name(ra["name"], rb["name"])
        bin_name_tots[ncat][dbin] += 1
        if p_tuple in g1_retr:
            bin_name_hits[ncat][dbin] += 1

    bheader = f"{'Name Class':20s} | " + " | ".join(f"{b:>8s}" for b in DISTANCE_BINS) + " | Overall"
    print(bheader, flush=True)
    print("-" * len(bheader), flush=True)
    diff_under_5km_hits = 0; diff_under_5km_tots = 0
    for ncat in NAME_CLASSES:
        row_str = f"{ncat:20s} | "
        tot_h = sum(bin_name_hits[ncat][b] for b in DISTANCE_BINS)
        tot_t = sum(bin_name_tots[ncat][b] for b in DISTANCE_BINS)
        for dbin in DISTANCE_BINS:
            h = bin_name_hits[ncat][dbin]; t = bin_name_tots[ncat][dbin]
            rate = 100.0 * h / t if t else 0.0
            row_str += f"{rate:7.1f}% | "
            if ncat == "different" and dbin in ("<50m", "<200m", "<1km", "<5km"):
                diff_under_5km_hits += h; diff_under_5km_tots += t
        tot_rate = 100.0 * tot_h / tot_t if tot_t else 0.0
        row_str += f"{tot_rate:7.1f}%"
        print(row_str, flush=True)

    diff_geo_rate = 100.0 * diff_under_5km_hits / diff_under_5km_tots if diff_under_5km_tots else 0.0
    print(f"\nRenamed venues ('different' name <= 5 km) recovery rate: {diff_geo_rate:.2f}% ({diff_under_5km_hits:,} / {diff_under_5km_tots:,})", flush=True)
    is_renamed_supported = diff_geo_rate >= 50.0
    print(f"Conclusion: 'geo recovers renamed venues' claim is {'SUPPORTED' if is_renamed_supported else 'UNSUPPORTED'} by the number ({diff_geo_rate:.1f}%).", flush=True)

    # STEP 3: GATE Evaluation
    print("\n=== STEP 3: GATE Evaluation ===", flush=True)
    p50 = results_by_k[50]["pair_recall"]
    p200 = results_by_k[200]["pair_recall"]
    delta_pp = (p200 - p50) * 100.0

    n_boot = 500
    boot_deltas = []
    pair_hits_50 = np.array([1 if p in results_by_k[50]["retrieved_undir"] else 0 for p in sub_canonical_true])
    pair_hits_200 = np.array([1 if p in results_by_k[200]["retrieved_undir"] else 0 for p in sub_canonical_true])

    boot_rng = np.random.default_rng(42)
    N_pairs = len(pair_hits_50)
    for _ in range(n_boot):
        sample_idx = boot_rng.integers(0, N_pairs, size=N_pairs)
        b_p50 = pair_hits_50[sample_idx].mean()
        b_p200 = pair_hits_200[sample_idx].mean()
        boot_deltas.append(b_p200 - b_p50)

    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))

    print(f"Pair Recall at K_pool=50:  {p50:.4f}", flush=True)
    print(f"Pair Recall at K_pool=100: {results_by_k[100]['pair_recall']:.4f}", flush=True)
    print(f"Pair Recall at K_pool=200: {p200:.4f}", flush=True)
    print(f"Delta (K_pool=200 - K_pool=50): {delta_pp:+.2f} pp", flush=True)
    print(f"95% Paired Bootstrap CI on Delta: [{ci_lo*100:+.2f} pp, {ci_hi*100:+.2f} pp]", flush=True)

    gate_passed = delta_pp >= 3.0
    gate_decision = "G2 rerank justified" if gate_passed else "G2 not justified"
    print(f"\nGATE RESULT: {gate_decision} (Threshold: >= +3.00 pp; Observed: {delta_pp:+.2f} pp)", flush=True)

    out_json = _REPO / "experiments" / "phase3c_ceiling_results.json"
    summary_data = {
        "step2_crosstab": {nc: dict(crosstab_counts[nc]) for nc in NAME_CLASSES},
        "step2_g1_recovery_rate": {nc: {dbin: (100.0 * bin_name_hits[nc][dbin] / bin_name_tots[nc][dbin] if bin_name_tots[nc][dbin] else 0.0) for dbin in DISTANCE_BINS} for nc in NAME_CLASSES},
        "renamed_venues_rate_under_5km": diff_geo_rate,
        "renamed_venues_claim_supported": is_renamed_supported,
        "step3_pool_ceiling": {
            str(k): {
                "pair_recall": results_by_k[k]["pair_recall"],
                "query_recall": results_by_k[k]["query_recall"],
                "total_candidates": results_by_k[k]["total_candidates"],
                "cpq_mean": results_by_k[k]["cpq_mean"],
                "dense_pair_recall": results_by_k[k]["dense_pair_recall"],
                "sparse_pair_recall": results_by_k[k]["sparse_pair_recall"],
                "name_recovery": results_by_k[k]["name_recovery"],
                "runtime_s": results_by_k[k]["runtime_s"],
                "peak_rss_mb": results_by_k[k]["peak_rss_mb"],
            }
            for k in [50, 100, 200]
        },
        "gate_delta_pp": delta_pp,
        "gate_ci_lo_pp": ci_lo * 100.0,
        "gate_ci_hi_pp": ci_hi * 100.0,
        "gate_passed": gate_passed,
        "gate_decision": gate_decision,
    }
    out_json.write_text(json.dumps(summary_data, indent=2))
    print(f"\nSaved execution metrics to {out_json}.", flush=True)
    print(f"Total runtime: {time.time() - t_start:.2f}s.", flush=True)

if __name__ == "__main__":
    main()
