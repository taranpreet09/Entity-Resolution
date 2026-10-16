"""
experiments/run_step3_tiebreak.py — Phase 4c Step 3: G3 Tie-Break Fix on Train & Val
"""
from __future__ import annotations

import json
import math
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
from ber_datasets.splits.splitter import EntityDisjointSplitter

EARTH_RADIUS_METERS = 6371000.0

def main():
    print("=== Phase 4c Step 3: G3 Tie-Break Fix (Train 100k & Val) ===", flush=True)
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

    # Pre-extract attributes for fast lookup
    doc_canon_name = [canonicalise(r["name"]) for r in all_records]
    doc_zip = [str(r.get("zip") or "").strip().lower() for r in all_records]
    doc_city = [str(r.get("city") or "").strip().lower() for r in all_records]
    doc_country = [str(r.get("country") or "").strip().upper() for r in all_records]
    doc_lat = np.array([float(r.get("latitude", np.nan)) for r in all_records], dtype=np.float64)
    doc_lon = np.array([float(r.get("longitude", np.nan)) for r in all_records], dtype=np.float64)
    doc_lat_rad = np.radians(doc_lat)
    doc_lon_rad = np.radians(doc_lon)
    doc_cos_lat = np.cos(doc_lat_rad)
    doc_lens = np.array([len(cn.split()) if cn else 1 for cn in doc_canon_name], dtype=np.float64)
    avgdl = float(np.mean(doc_lens))

    # Splits
    splitter = EntityDisjointSplitter(ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
    splits = splitter.split(rids, [r["point_of_interest"] for r in all_records])
    
    val_rids = [rid for rid, s in splits.items() if s == "val"]
    val_records = [all_records[rid_to_idx[rid]] for rid in val_rids]

    # Fixed 100k Train sample (seed 42)
    rng = random.Random(42)
    all_indices = list(range(len(df)))
    train_sample_indices = sorted(rng.sample(all_indices, 100_000))
    train_sample_records = [all_records[i] for i in train_sample_indices]
    train_sample_rids = [r["id"] for r in train_sample_records]

    # Ground truth
    # Train true pairs (54,375)
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
    print(f"Train 100k true pairs: {tr_denom:,}", flush=True)

    # Val true pairs (17,916)
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

    # Build G3 index
    print("Building G3 IDF index...", flush=True)
    g3 = IDFRetriever(IDFConfig(k_candidates=20, df_cap=5000, country_filter=True))
    g3.build_index(all_records)
    effective_df_cap = g3.get_effective_df_cap()
    print(f"Indexed {g3.total_docs:,} docs. Vocab: {len(g3.token_to_id):,}, df_cap: {effective_df_cap}", flush=True)

    # Retrieval runner for different tie-break variants
    # Mode:
    # 0: 'baseline' (raw IDF score, insertion/default tie-break)
    # 1: 'deterministic' (raw IDF score + exact name + zip/city + country + haversine + id)
    # 2: 'bm25' (BM25 length-normalized score, k1=1.2, b=0.75)
    # 3: 'bm25_deterministic' (BM25 score + tie-break)

    def retrieve_with_mode(query_records, k=5, mode="baseline", country_filter=True):
        t_start = time.time()
        undir_pairs = set()
        total_cands = 0

        k1_bm25 = 1.2
        b_bm25 = 0.75

        for q_rec in query_records:
            qid = q_rec["id"]
            q_idx = rid_to_idx[qid]
            q_name = q_rec["name"]
            q_canon = doc_canon_name[q_idx]
            if not q_canon:
                continue

            q_country = doc_country[q_idx] if country_filter else ""
            q_zip = doc_zip[q_idx]
            q_city = doc_city[q_idx]
            q_lat_rad = doc_lat_rad[q_idx]
            q_lon_rad = doc_lon_rad[q_idx]
            q_cos = doc_cos_lat[q_idx]

            toks = q_canon.split()
            active_tids = [
                g3.token_to_id[t]
                for t in set(toks)
                if t in g3.token_to_id and g3.df_table[g3.token_to_id[t]] <= effective_df_cap
            ]
            if not active_tids:
                continue

            # Accumulate candidate scores
            cand_scores = defaultdict(float)
            for tid in active_tids:
                idf_val = g3.idf_table[tid]
                for doc_idx in g3.postings[tid]:
                    if doc_idx == q_idx:
                        continue
                    if country_filter and q_country:
                        if doc_country[doc_idx] and doc_country[doc_idx] != q_country:
                            continue
                    cand_scores[doc_idx] += idf_val

            if not cand_scores:
                continue

            # Candidate ranking
            if mode == "baseline":
                # Baseline uses raw IDF, sorted descending by score
                # In Python sort: stable sort by (-score) preserves default doc order
                sorted_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:k]
                c_indices = [c[0] for c in sorted_cands]

            elif mode == "deterministic":
                # (i) Exact name -> same zip/city -> same country -> nearer Haversine distance -> candidate ID
                cand_list = []
                for doc_idx, sc in cand_scores.items():
                    # 1. exact name
                    exact_name = 1 if doc_canon_name[doc_idx] == q_canon else 0
                    # 2. same zip or city
                    same_loc = 1 if ((q_zip and doc_zip[doc_idx] and q_zip == doc_zip[doc_idx]) or
                                     (q_city and doc_city[doc_idx] and q_city == doc_city[doc_idx])) else 0
                    # 3. same country
                    same_ctry = 1 if (q_country and doc_country[doc_idx] and q_country == doc_country[doc_idx]) else 0
                    # 4. haversine distance
                    c_lat_rad = doc_lat_rad[doc_idx]
                    c_lon_rad = doc_lon_rad[doc_idx]
                    if math.isnan(c_lat_rad) or math.isnan(q_lat_rad):
                        dist = 1e9
                    else:
                        dlat = c_lat_rad - q_lat_rad
                        dlon = c_lon_rad - q_lon_rad
                        a = math.sin(dlat * 0.5)**2 + q_cos * doc_cos_lat[doc_idx] * math.sin(dlon * 0.5)**2
                        dist = EARTH_RADIUS_METERS * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
                    cand_id = rids[doc_idx]
                    # Sort key: (-sc, -exact_name, -same_loc, -same_ctry, dist, cand_id)
                    cand_list.append((doc_idx, -sc, -exact_name, -same_loc, -same_ctry, dist, cand_id))

                cand_list.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5], x[6]))
                c_indices = [x[0] for x in cand_list[:k]]

            elif mode == "bm25":
                # (ii) BM25 length normalization
                cand_list = []
                for doc_idx, sc in cand_scores.items():
                    c_len = doc_lens[doc_idx]
                    denom = 1.0 + k1_bm25 * (1.0 - b_bm25 + b_bm25 * (c_len / avgdl))
                    bm25_sc = sc * (k1_bm25 + 1.0) / denom
                    cand_list.append((doc_idx, -bm25_sc))
                cand_list.sort(key=lambda x: x[1])
                c_indices = [x[0] for x in cand_list[:k]]

            elif mode == "bm25_deterministic":
                # Combined BM25 + deterministic tie break
                cand_list = []
                for doc_idx, sc in cand_scores.items():
                    c_len = doc_lens[doc_idx]
                    denom = 1.0 + k1_bm25 * (1.0 - b_bm25 + b_bm25 * (c_len / avgdl))
                    bm25_sc = sc * (k1_bm25 + 1.0) / denom

                    exact_name = 1 if doc_canon_name[doc_idx] == q_canon else 0
                    same_loc = 1 if ((q_zip and doc_zip[doc_idx] and q_zip == doc_zip[doc_idx]) or
                                     (q_city and doc_city[doc_idx] and q_city == doc_city[doc_idx])) else 0
                    same_ctry = 1 if (q_country and doc_country[doc_idx] and q_country == doc_country[doc_idx]) else 0
                    c_lat_rad = doc_lat_rad[doc_idx]
                    c_lon_rad = doc_lon_rad[doc_idx]
                    if math.isnan(c_lat_rad) or math.isnan(q_lat_rad):
                        dist = 1e9
                    else:
                        dlat = c_lat_rad - q_lat_rad
                        dlon = c_lon_rad - q_lon_rad
                        a = math.sin(dlat * 0.5)**2 + q_cos * doc_cos_lat[doc_idx] * math.sin(dlon * 0.5)**2
                        dist = EARTH_RADIUS_METERS * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
                    cand_id = rids[doc_idx]
                    cand_list.append((doc_idx, -bm25_sc, -exact_name, -same_loc, -same_ctry, dist, cand_id))

                cand_list.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5], x[6]))
                c_indices = [x[0] for x in cand_list[:k]]

            for doc_idx in c_indices:
                cand_id = rids[doc_idx]
                undir_pairs.add((min(qid, cand_id), max(qid, cand_id)))
            total_cands += len(c_indices)

        elapsed = time.time() - t_start
        return undir_pairs, total_cands, elapsed

    # Run sweeps on 100k Train sample
    print("\n--- Evaluating Tie-Break Variants on 100k Train Sample ---", flush=True)
    results = {}
    for k in [5, 20]:
        print(f"\n>> Evaluating K={k} (Country Filter ON):", flush=True)
        # 1. Baseline
        p_base, c_base, t_base = retrieve_with_mode(train_sample_records, k=k, mode="baseline", country_filter=True)
        hits_base = len(tr_true_pairs & p_base)
        rec_base = hits_base / tr_denom
        print(f"  Baseline:            recall={rec_base:.4f} ({hits_base:,}/{tr_denom:,}), cands={c_base:,}, time={t_base:.2f}s", flush=True)

        # 2. Fix (i) Deterministic tie-breaker
        p_det, c_det, t_det = retrieve_with_mode(train_sample_records, k=k, mode="deterministic", country_filter=True)
        hits_det = len(tr_true_pairs & p_det)
        rec_det = hits_det / tr_denom
        gain_det_pp = (rec_det - rec_base) * 100.0
        print(f"  (i) Deterministic:   recall={rec_det:.4f} ({hits_det:,}/{tr_denom:,}), cands={c_det:,}, gain={gain_det_pp:+.2f} pp, time={t_det:.2f}s", flush=True)

        # 3. Fix (ii) BM25 length norm
        p_bm25, c_bm25, t_bm25 = retrieve_with_mode(train_sample_records, k=k, mode="bm25", country_filter=True)
        hits_bm25 = len(tr_true_pairs & p_bm25)
        rec_bm25 = hits_bm25 / tr_denom
        gain_bm25_pp = (rec_bm25 - rec_base) * 100.0
        print(f"  (ii) BM25 norm:      recall={rec_bm25:.4f} ({hits_bm25:,}/{tr_denom:,}), cands={c_bm25:,}, gain={gain_bm25_pp:+.2f} pp, time={t_bm25:.2f}s", flush=True)

        # 4. BM25 + Deterministic
        p_both, c_both, t_both = retrieve_with_mode(train_sample_records, k=k, mode="bm25_deterministic", country_filter=True)
        hits_both = len(tr_true_pairs & p_both)
        rec_both = hits_both / tr_denom
        gain_both_pp = (rec_both - rec_base) * 100.0
        print(f"  BM25 + Det:          recall={rec_both:.4f} ({hits_both:,}/{tr_denom:,}), cands={c_both:,}, gain={gain_both_pp:+.2f} pp, time={t_both:.2f}s", flush=True)

        results[f"train_k{k}"] = {
            "baseline": {"recall": rec_base, "hits": hits_base, "cands": c_base, "time": t_base},
            "deterministic": {"recall": rec_det, "hits": hits_det, "cands": c_det, "gain_pp": gain_det_pp, "time": t_det},
            "bm25": {"recall": rec_bm25, "hits": hits_bm25, "cands": c_bm25, "gain_pp": gain_bm25_pp, "time": t_bm25},
            "bm25_deterministic": {"recall": rec_both, "hits": hits_both, "cands": c_both, "gain_pp": gain_both_pp, "time": t_both},
        }

    # Evaluate Winner on Val
    print("\n--- Evaluating Winner on Validation Split (22,373 queries, 17,916 pairs) ---", flush=True)
    for k in [5, 20]:
        print(f"\n>> Val evaluation K={k} (Country Filter ON):", flush=True)
        p_val_base, c_val_base, t_val_base = retrieve_with_mode(val_records, k=k, mode="baseline", country_filter=True)
        hits_val_base = len(val_true_pairs & p_val_base)
        rec_val_base = hits_val_base / val_denom
        print(f"  Val Baseline:        recall={rec_val_base:.4f} ({hits_val_base:,}/{val_denom:,}), cands={c_val_base:,}, time={t_val_base:.2f}s", flush=True)

        p_val_det, c_val_det, t_val_det = retrieve_with_mode(val_records, k=k, mode="deterministic", country_filter=True)
        hits_val_det = len(val_true_pairs & p_val_det)
        rec_val_det = hits_val_det / val_denom
        gain_val_det_pp = (rec_val_det - rec_val_base) * 100.0
        print(f"  Val Deterministic:   recall={rec_val_det:.4f} ({hits_val_det:,}/{val_denom:,}), cands={c_val_det:,}, gain={gain_val_det_pp:+.2f} pp, time={t_val_det:.2f}s", flush=True)

        results[f"val_k{k}"] = {
            "baseline": {"recall": rec_val_base, "hits": hits_val_base, "cands": c_val_base, "time": t_val_base},
            "deterministic": {"recall": rec_val_det, "hits": hits_val_det, "cands": c_val_det, "gain_pp": gain_val_det_pp, "time": t_val_det},
        }

    out_file = _REPO / "experiments" / "step3_tiebreak_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
