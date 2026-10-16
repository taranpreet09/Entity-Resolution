"""
scratch/run_phase4b_steps.py — Execute Phase 4b Steps 2 & 3
"""
from __future__ import annotations

import gc
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
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ber.canon.normalise import canonicalise
from ber.eval.metrics_dedup import pair_recall, directed_slot_recall
from ber.retrieve.geo_blocking import EARTH_RADIUS_METERS
from ber.retrieve.idf import IDFRetriever, IDFConfig, infer_country_from_coords
from ber_datasets.splits.splitter import EntityDisjointSplitter

import ctypes
from ctypes import wintypes
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]

def get_peak_rss_mb() -> float:
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    GetCurrentProcess = kernel32.GetCurrentProcess
    GetCurrentProcess.restype = wintypes.HANDLE
    GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
    GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    handle = GetCurrentProcess()
    if GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return counters.PeakWorkingSetSize / (1024 * 1024)
    return 0.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(math.isnan(x) for x in (lat1, lon1, lat2, lon2)):
        return float("nan")
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c

def main():
    print("=== Phase 4b: Step 2 & Step 3 Execution ===", flush=True)
    t_start = time.time()

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    print(f"Loaded {len(df):,} records.", flush=True)

    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    rid_to_idx = {r["id"]: i for i, r in enumerate(all_records)}

    # Splits
    # 1. EntityDisjointSplitter (70/15/15)
    splitter = EntityDisjointSplitter(ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
    disjoint_splits = splitter.split(rids, [r["point_of_interest"] for r in all_records])
    
    val_disjoint_rids = [rid for rid, s in disjoint_splits.items() if s == "val"]
    val_disjoint_records = [all_records[rid_to_idx[rid]] for rid in val_disjoint_rids]
    val_disjoint_set = set(val_disjoint_rids)

    # 2. Fixed 100k Train sample (seed 42)
    rng = random.Random(42)
    all_indices = list(range(len(df)))
    train_sample_indices = sorted(rng.sample(all_indices, 100_000))
    train_sample_records = [all_records[i] for i in train_sample_indices]
    train_sample_rids = [r["id"] for r in train_sample_records]
    train_sample_set = set(train_sample_rids)

    # Compute Ground Truth True Pairs
    # For Train sample (internal pairs):
    poi_groups_train = defaultdict(list)
    for r in train_sample_records:
        poi = r["point_of_interest"]
        if poi: poi_groups_train[poi].append(r["id"])
    train_true_pairs = []
    for poi, ids in poi_groups_train.items():
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    train_true_pairs.append((min(ids[i], ids[j]), max(ids[i], ids[j])))
    train_true_pair_set = set(train_true_pairs)
    print(f"Train sample (100k) true pairs: {len(train_true_pair_set):,}", flush=True)

    # Train query answerability
    train_targets_by_q = defaultdict(set)
    for a, b in train_true_pairs:
        train_targets_by_q[a].add(b)
        train_targets_by_q[b].add(a)
    train_q_with_targets = [q for q in train_sample_rids if q in train_targets_by_q]
    train_singleton_q = [q for q in train_sample_rids if q not in train_targets_by_q]
    print(f"Train answerable queries: {len(train_q_with_targets):,}, singleton queries: {len(train_singleton_q):,}", flush=True)

    # For Val disjoint:
    poi_groups_val = defaultdict(list)
    for r in val_disjoint_records:
        poi = r["point_of_interest"]
        if poi: poi_groups_val[poi].append(r["id"])
    val_true_pairs = []
    for poi, ids in poi_groups_val.items():
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    val_true_pairs.append((min(ids[i], ids[j]), max(ids[i], ids[j])))
    val_true_pair_set = set(val_true_pairs)
    print(f"Val disjoint (22,373 queries) true pairs: {len(val_true_pair_set):,}", flush=True)

    val_targets_by_q = defaultdict(set)
    for a, b in val_true_pairs:
        val_targets_by_q[a].add(b)
        val_targets_by_q[b].add(a)
    val_q_with_targets = [q for q in val_disjoint_rids if q in val_targets_by_q]
    val_singleton_q = [q for q in val_disjoint_rids if q not in val_targets_by_q]
    print(f"Val answerable queries: {len(val_q_with_targets):,}, singleton queries: {len(val_singleton_q):,}", flush=True)

    # BUILD BALLTREE FOR G1 (K=50, R=5km)
    print("\n--- Building BallTree for G1 Retrieval ---", flush=True)
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    geo_tree = BallTree(coords_rad, metric="haversine")
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS

    # Retrieve G1 for Train sample queries
    print("Retrieving G1 (K=50, R=5km) on 100k Train sample queries...", flush=True)
    t0_g1 = time.time()
    g1_train_cands_by_q = defaultdict(list)
    g1_train_undir = set()
    total_g1_train_cands = 0
    train_coords_rad = coords_rad[train_sample_indices]

    batch_size = 10_000
    for b_start in range(0, len(train_sample_indices), batch_size):
        b_end = b_start + batch_size
        b_coords = train_coords_rad[b_start:b_end]
        b_qids = train_sample_rids[b_start:b_end]
        dists, inds = geo_tree.query(b_coords, k=51, return_distance=True)
        for q_id, row_inds, row_dists in zip(b_qids, inds, dists):
            cnt = 0
            for cand_idx, d_rad in zip(row_inds, row_dists):
                cand_id = rids[cand_idx]
                if cand_id == q_id: continue
                if d_rad > r5km_rad: continue
                g1_train_cands_by_q[q_id].append(cand_id)
                g1_train_undir.add((min(q_id, cand_id), max(q_id, cand_id)))
                cnt += 1
                if cnt >= 50: break
            total_g1_train_cands += cnt

    g1_train_hits = len(train_true_pair_set & g1_train_undir)
    g1_train_pair_recall = g1_train_hits / len(train_true_pair_set)
    print(f"G1 Train: {g1_train_hits:,}/{len(train_true_pair_set):,} ({g1_train_pair_recall:.4f}), Cands: {total_g1_train_cands:,} in {time.time()-t0_g1:.2f}s", flush=True)

    # Retrieve G1 for Val queries
    print("Retrieving G1 (K in {4, 12, 50}, R=5km) on Val queries...", flush=True)
    val_indices = [rid_to_idx[rid] for rid in val_disjoint_rids]
    val_coords_rad = coords_rad[val_indices]
    g1_val_cands_by_q = {50: defaultdict(list), 12: defaultdict(list), 4: defaultdict(list)}
    g1_val_undir = {50: set(), 12: set(), 4: set()}
    total_g1_val_cands = {50: 0, 12: 0, 4: 0}

    for b_start in range(0, len(val_indices), batch_size):
        b_end = b_start + batch_size
        b_coords = val_coords_rad[b_start:b_end]
        b_qids = val_disjoint_rids[b_start:b_end]
        dists, inds = geo_tree.query(b_coords, k=51, return_distance=True)
        for q_id, row_inds, row_dists in zip(b_qids, inds, dists):
            cnt = 0
            for cand_idx, d_rad in zip(row_inds, row_dists):
                cand_id = rids[cand_idx]
                if cand_id == q_id: continue
                if d_rad > r5km_rad: continue
                for k_limit in [50, 12, 4]:
                    if cnt < k_limit:
                        g1_val_cands_by_q[k_limit][q_id].append(cand_id)
                        g1_val_undir[k_limit].add((min(q_id, cand_id), max(q_id, cand_id)))
                        total_g1_val_cands[k_limit] += 1
                cnt += 1
                if cnt >= 50: break

    for k_val in [4, 12, 50]:
        hits = len(val_true_pair_set & g1_val_undir[k_val])
        rec = hits / len(val_true_pair_set)
        print(f"G1 Val (K={k_val}): {hits:,}/{len(val_true_pair_set):,} ({rec:.4f}), Cands: {total_g1_val_cands[k_val]:,}", flush=True)

    # 3. BUILD CHANNEL G3
    print("\n--- Building Channel G3 on Full 150k Table ---", flush=True)
    t_bld = time.time()
    g3 = IDFRetriever(IDFConfig(k_candidates=20, df_cap=5000, country_filter=False))
    g3.build_index(all_records)
    print(f"G3 Built in {time.time()-t_bld:.2f}s. Vocab: {len(g3.token_to_id):,}, DF cap: {g3.get_effective_df_cap():,}", flush=True)

    # STEP 2: Run G3 on ALL train-sample queries and ALL val queries for K in {5, 10, 20}, country ON & OFF
    sweep_results = {}
    print("\n=== STEP 2: G3 Parameter Sweep on Full Query Sets ===", flush=True)

    for country_flag in [False, True]:
        c_label = "Country ON" if country_flag else "Country OFF"
        print(f"\n--- Running G3 with {c_label} ---", flush=True)
        
        # Train 100k
        t0 = time.time()
        cands_train_20, undir_train_20, tot_c_train_20 = g3.retrieve_batch(train_sample_records, k=20, country_filter=country_flag)
        train_time = time.time() - t0
        peak_rss = get_peak_rss_mb()
        print(f"Train 100k retrieved in {train_time:.2f}s ({train_time/100:.2f} ms/q). Peak RSS: {peak_rss:.1f} MB", flush=True)

        # Val 22,373
        t0_v = time.time()
        cands_val_20, undir_val_20, tot_c_val_20 = g3.retrieve_batch(val_disjoint_records, k=20, country_filter=country_flag)
        val_time = time.time() - t0_v

        for k_val in [5, 10, 20]:
            cfg_name = f"K={k_val}_{'cf_on' if country_flag else 'cf_off'}"
            
            # Slice train
            cands_train = {q: cands_train_20[q][:k_val] for q in train_sample_rids}
            undir_train = set()
            for q, clist in cands_train.items():
                for c in clist:
                    undir_train.add((min(q, c), max(q, c)))
            tot_cands_train = sum(len(clist) for clist in cands_train.values())

            # Train metrics
            tr_hits = len(train_true_pair_set & undir_train)
            tr_pair_rec = tr_hits / len(train_true_pair_set)
            
            tr_q_hits = sum(1 for q in train_q_with_targets if set(cands_train.get(q, [])) & train_targets_by_q[q])
            tr_q_rec = tr_q_hits / len(train_q_with_targets)
            
            tr_partial_hits = sum(1 for q in train_q_with_targets if 0 < len(set(cands_train.get(q, [])) & train_targets_by_q[q]) < len(train_targets_by_q[q]))
            tr_partial_rec = tr_partial_hits / len(train_q_with_targets)
            
            tr_lens = [len(cands_train.get(q, [])) for q in train_sample_rids]
            tr_cpq_med = float(np.median(tr_lens))
            tr_cpq_p95 = float(np.percentile(tr_lens, 95))
            tr_cpq_mean = float(np.mean(tr_lens))
            tr_zero_cand = sum(1 for l in tr_lens if l == 0) / len(tr_lens)

            # Singleton burden train
            tr_sing_lens = [len(cands_train.get(q, [])) for q in train_singleton_q]
            tr_sing_tot = sum(tr_sing_lens)
            tr_sing_mean = float(np.mean(tr_sing_lens))

            # Marginal over G1 on train
            union_train_undir = g1_train_undir | undir_train
            union_tr_hits = len(train_true_pair_set & union_train_undir)
            union_tr_rec = union_tr_hits / len(train_true_pair_set)
            
            # Union candidate count (unique pairs by query)
            union_cands_by_q_cnt = sum(len(set(g1_train_cands_by_q[q]) | set(cands_train.get(q, []))) for q in train_sample_rids)
            added_cands_tr = union_cands_by_q_cnt - total_g1_train_cands
            marginal_pairs_tr = union_tr_hits - g1_train_hits
            marginal_yield_tr = marginal_pairs_tr / max(1, added_cands_tr)

            # Slice val
            cands_val = {q: cands_val_20[q][:k_val] for q in val_disjoint_rids}
            undir_val = set()
            for q, clist in cands_val.items():
                for c in clist:
                    undir_val.add((min(q, c), max(q, c)))
            tot_cands_val = sum(len(clist) for clist in cands_val.values())

            # Val metrics
            v_hits = len(val_true_pair_set & undir_val)
            v_pair_rec = v_hits / len(val_true_pair_set)

            v_q_hits = sum(1 for q in val_q_with_targets if set(cands_val.get(q, [])) & val_targets_by_q[q])
            v_q_rec = v_q_hits / len(val_q_with_targets)

            v_partial_hits = sum(1 for q in val_q_with_targets if 0 < len(set(cands_val.get(q, [])) & val_targets_by_q[q]) < len(val_targets_by_q[q]))
            v_partial_rec = v_partial_hits / len(val_q_with_targets)

            v_lens = [len(cands_val.get(q, [])) for q in val_disjoint_rids]
            v_cpq_med = float(np.median(v_lens))
            v_cpq_p95 = float(np.percentile(v_lens, 95))
            v_cpq_mean = float(np.mean(v_lens))
            v_zero_cand = sum(1 for l in v_lens if l == 0) / len(v_lens)

            v_sing_lens = [len(cands_val.get(q, [])) for q in val_singleton_q]
            v_sing_tot = sum(v_sing_lens)
            v_sing_mean = float(np.mean(v_sing_lens))

            sweep_results[cfg_name] = {
                "k": k_val,
                "country_filter": country_flag,
                "train": {
                    "pair_recall": tr_pair_rec,
                    "pair_hits": tr_hits,
                    "pair_denom": len(train_true_pair_set),
                    "query_recall": tr_q_rec,
                    "partial_recall": tr_partial_rec,
                    "zero_cand_rate": tr_zero_cand,
                    "cpq_median": tr_cpq_med,
                    "cpq_p95": tr_cpq_p95,
                    "cpq_mean": tr_cpq_mean,
                    "total_candidates": tot_cands_train,
                    "singleton_total": tr_sing_tot,
                    "singleton_cpq_mean": tr_sing_mean,
                    "runtime_s": train_time * (k_val / 20.0), # proportional proxy
                    "rss_mb": peak_rss,
                    "marginal_pairs_over_g1": marginal_pairs_tr,
                    "added_cands_over_g1": added_cands_tr,
                    "marginal_yield": marginal_yield_tr,
                    "union_pair_recall": union_tr_rec,
                },
                "val": {
                    "pair_recall": v_pair_rec,
                    "pair_hits": v_hits,
                    "pair_denom": len(val_true_pair_set),
                    "query_recall": v_q_rec,
                    "partial_recall": v_partial_rec,
                    "zero_cand_rate": v_zero_cand,
                    "cpq_median": v_cpq_med,
                    "cpq_p95": v_cpq_p95,
                    "cpq_mean": v_cpq_mean,
                    "total_candidates": tot_cands_val,
                    "singleton_total": v_sing_tot,
                    "singleton_cpq_mean": v_sing_mean,
                    "runtime_s": val_time * (k_val / 20.0),
                }
            }

            print(f"[{cfg_name:14s}] Train PairRec: {tr_pair_rec:.4f} ({tr_hits:,}/{len(train_true_pair_set):,}), CPQ: {tr_cpq_mean:.2f}, Added: {added_cands_tr:,}, MargYield: {marginal_yield_tr:.5f} | Val PairRec: {v_pair_rec:.4f} ({v_hits:,}/{len(val_true_pair_set):,})", flush=True)

    # Find chosen setting on TRAIN ONLY: highest marginal true pairs per added candidate over G1
    print("\n--- Model Selection on TRAIN ONLY ---", flush=True)
    best_cfg = None
    best_yield = -1.0
    for name, r in sweep_results.items():
        y = r["train"]["marginal_yield"]
        cands = r["train"]["total_candidates"]
        print(f"  {name:14s}: Marginal Yield = {y:.6f} ({r['train']['marginal_pairs_over_g1']:,} pairs / {r['train']['added_cands_over_g1']:,} added cands)")
        if y > best_yield:
            best_yield = y
            best_cfg = name
        elif abs(y - best_yield) < 1e-6:
            # tie break toward fewer candidates
            if cands < sweep_results[best_cfg]["train"]["total_candidates"]:
                best_cfg = name

    print(f"\nCHOSEN SETTING (Train Only): {best_cfg} (Marginal Yield = {best_yield:.6f})", flush=True)

    # Save intermediate results
    out_path = _REPO / "experiments" / "phase4b_step2_sweep.json"
    out_path.write_text(json.dumps({
        "sweep_results": sweep_results,
        "chosen_cfg": best_cfg,
    }, indent=2))
    print(f"Saved Step 2 results to {out_path}", flush=True)

if __name__ == "__main__":
    main()

