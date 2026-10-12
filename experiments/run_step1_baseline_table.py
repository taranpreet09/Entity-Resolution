"""
experiments/run_step1_baseline_table.py — Phase 4c Step 1: Recomputed Baselines & Slots Analysis
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
from datasets.splits.splitter import EntityDisjointSplitter

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

def main():
    print("=== Phase 4c Step 1: Baseline Reruns on Canonical 150k Table ===", flush=True)
    t0_start = time.time()

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    rid_to_idx = {r["id"]: i for i, r in enumerate(all_records)}

    # Entity-disjoint split (70/15/15, seed 42)
    splitter = EntityDisjointSplitter(ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
    splits = splitter.split(rids, [r["point_of_interest"] for r in all_records])

    val_rids = [rid for rid, s in splits.items() if s == "val"]
    val_records = [all_records[rid_to_idx[rid]] for rid in val_rids]
    val_set = set(val_rids)

    # True pairs on validation
    poi_g = defaultdict(list)
    for r in val_records:
        poi = r["point_of_interest"]
        if poi: poi_g[poi].append(r["id"])

    val_true_pairs = []
    for ids in poi_g.values():
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                val_true_pairs.append((min(ids[i], ids[j]), max(ids[i], ids[j])))
    val_denom_pairs = len(val_true_pairs)
    val_denom_slots = val_denom_pairs * 2 # exactly 2 slots per true pair

    print(f"Validation queries: {len(val_records):,}, true pairs: {val_denom_pairs:,}, true slots: {val_denom_slots:,}", flush=True)

    # BallTree for G1
    print("\nIndexing BallTree...", flush=True)
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    geo_tree = BallTree(coords_rad, metric="haversine")
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS

    val_indices = [rid_to_idx[rid] for rid in val_rids]
    val_coords_rad = coords_rad[val_indices]

    # Retrieve G1 for K in {4, 12, 50}
    print("Retrieving G1 (K in {4, 12, 50})...", flush=True)
    t0_g1 = time.time()
    g1_cands_by_q = {4: defaultdict(list), 12: defaultdict(list), 50: defaultdict(list)}
    batch_size = 5000
    for b_start in range(0, len(val_indices), batch_size):
        b_end = min(b_start + batch_size, len(val_indices))
        b_coords = val_coords_rad[b_start:b_end]
        b_qids = val_rids[b_start:b_end]
        dists, inds = geo_tree.query(b_coords, k=51, return_distance=True)
        for qid, row_inds, row_dists in zip(b_qids, inds, dists):
            cnt = 0
            for cand_idx, d_rad in zip(row_inds, row_dists):
                cid = rids[cand_idx]
                if cid == qid: continue
                if d_rad > r5km_rad: continue
                for k_limit in [50, 12, 4]:
                    if cnt < k_limit:
                        g1_cands_by_q[k_limit][qid].append(cid)
                cnt += 1
                if cnt >= 50: break
    g1_time = time.time() - t0_g1

    # Control G=50
    print("Constructing Control G=50 Blocker...", flush=True)
    t0_ctrl = time.time()
    ctrl_k1 = defaultdict(list)
    ctrl_k2 = defaultdict(list)
    for r in all_records:
        rid = r["id"]
        c_name = canonicalise(r["name"])
        country = r.get("country", "")
        if c_name and country:
            ctrl_k1[f"{c_name}|{country}"].append(rid)
        lat, lon = r["latitude"], r["longitude"]
        if not math.isnan(lat) and not math.isnan(lon):
            cell_key = f"{int(math.floor(lat / 0.05))}|{int(math.floor(lon / 0.05))}"
            ctrl_k2[cell_key].append(rid)

    ctrl_g50_cands_by_q = defaultdict(list)
    gate_limit = 50
    for r in val_records:
        qid = r["id"]
        c_name = canonicalise(r["name"])
        country = r.get("country", "")
        lat, lon = r["latitude"], r["longitude"]
        c_set = set()
        if c_name and country:
            b1 = ctrl_k1.get(f"{c_name}|{country}", [])
            if len(b1) - 1 <= gate_limit:
                for cid in b1:
                    if cid != qid: c_set.add(cid)
        if not math.isnan(lat) and not math.isnan(lon):
            cell_key = f"{int(math.floor(lat / 0.05))}|{int(math.floor(lon / 0.05))}"
            b2 = ctrl_k2.get(cell_key, [])
            if len(b2) - 1 <= gate_limit:
                for cid in b2:
                    if cid != qid: c_set.add(cid)
        ctrl_g50_cands_by_q[qid] = list(c_set)
    ctrl_time = time.time() - t0_ctrl

    # G3 Index & Retrieval (K=5 and K=20, Country ON)
    print("Building G3 Index & Retrieving (K=5, K=20, Country ON)...", flush=True)
    t0_g3_bld = time.time()
    g3 = IDFRetriever(IDFConfig(k_candidates=20, df_cap=5000, country_filter=True))
    g3.build_index(all_records)
    g3_bld_time = time.time() - t0_g3_bld

    t0_g3_ret = time.time()
    g3_20_cands_by_q, _, _ = g3.retrieve_batch(val_records, k=20, country_filter=True)
    g3_time = time.time() - t0_g3_ret
    g3_5_cands_by_q = {q: clist[:5] for q, clist in g3_20_cands_by_q.items()}

    # Compute directed slots and pairs accurately
    def evaluate_candidates(name: str, cand_dict: dict[str, list[str]], runtime_s: float):
        # 1. Total candidates
        tot_cands = sum(len(clist) for clist in cand_dict.values())
        
        # 2. Directed candidates map
        cand_map = {q: set(clist) for q, clist in cand_dict.items()}

        # 3. Directed slots & undirected pairs
        bidi_pairs = 0
        unidi_pairs = 0
        missed_pairs = 0

        for a, b in val_true_pairs:
            a_finds_b = b in cand_map.get(a, set())
            b_finds_a = a in cand_map.get(b, set())

            if a_finds_b and b_finds_a:
                bidi_pairs += 1
            elif a_finds_b or b_finds_a:
                unidi_pairs += 1
            else:
                missed_pairs += 1

        undir_pairs_found = bidi_pairs + unidi_pairs
        directed_slots_found = (bidi_pairs * 2) + unidi_pairs

        pair_rec = undir_pairs_found / val_denom_pairs
        slot_rec = directed_slots_found / val_denom_slots

        # Assertions
        assert round(pair_rec * val_denom_pairs) == undir_pairs_found
        assert round(slot_rec * val_denom_slots) == directed_slots_found

        return {
            "name": name,
            "candidates": tot_cands,
            "undir_pairs_found": undir_pairs_found,
            "pair_recall": pair_rec,
            "pair_denominator": val_denom_pairs,
            "directed_slots_found": directed_slots_found,
            "slot_recall": slot_rec,
            "slot_denominator": val_denom_slots,
            "bidirectional_pairs": bidi_pairs,
            "unidirectional_pairs": unidi_pairs,
            "slots_vs_2x_pairs_diff": (2 * undir_pairs_found) - directed_slots_found,
            "runtime_s": runtime_s,
            "rss_mb": get_peak_rss_mb(),
        }

    configs = {
        "Control G=50": (ctrl_g50_cands_by_q, ctrl_time),
        "G1 (K=4, R=5km)": (g1_cands_by_q[4], g1_time * (4/50)),
        "G1 (K=12, R=5km)": (g1_cands_by_q[12], g1_time * (12/50)),
        "G1 (K=50, R=5km)": (g1_cands_by_q[50], g1_time),
        "G3 (K=5, Country ON)": (g3_5_cands_by_q, g3_time * (5/20)),
        "G3 (K=20, Country ON)": (g3_20_cands_by_q, g3_time),
    }

    # Also compute unions
    union_ctrl_g1_cands = {q: list(set(ctrl_g50_cands_by_q[q]) | set(g1_cands_by_q[50][q])) for q in val_rids}
    union_g1_g3_k5 = {q: list(set(g1_cands_by_q[50][q]) | set(g3_5_cands_by_q[q])) for q in val_rids}
    union_ctrl_g1_g3_k5 = {q: list(set(ctrl_g50_cands_by_q[q]) | set(g1_cands_by_q[50][q]) | set(g3_5_cands_by_q[q])) for q in val_rids}

    configs["Control G=50 + G1 (K=50)"] = (union_ctrl_g1_cands, ctrl_time + g1_time)
    configs["G1 (K=50) + G3 (K=5, Country ON)"] = (union_g1_g3_k5, g1_time + g3_time*(5/20))
    configs["Control G=50 + G1 + G3 (K=5)"] = (union_ctrl_g1_g3_k5, ctrl_time + g1_time + g3_time*(5/20))

    results = {}
    print("\n--- Canonical 150k Baseline Table (Val Split, N=17,916 pairs, N=35,832 slots) ---", flush=True)
    header = f"{'Configuration':34s} | {'Candidates':12s} | {'Pairs Found':11s} | {'Pair Recall':11s} | {'Slots Found':11s} | {'Slot Recall':11s} | {'Asymm Diff':10s} | {'Time (s)':8s}"
    print(header)
    print("-" * len(header))

    for cfg_name, (c_dict, rtime) in configs.items():
        res = evaluate_candidates(cfg_name, c_dict, rtime)
        results[cfg_name] = res
        print(f"{cfg_name:34s} | {res['candidates']:12,d} | {res['undir_pairs_found']:5,d}/{val_denom_pairs:,d} | {res['pair_recall']*100:9.2f}% | {res['directed_slots_found']:5,d}/{val_denom_slots:,d} | {res['slot_recall']*100:9.2f}% | {res['slots_vs_2x_pairs_diff']:10d} | {res['runtime_s']:8.2f}s")

    out_file = _REPO / "experiments" / "step1_canonical_baseline.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nStep 1 Baseline Results saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()

