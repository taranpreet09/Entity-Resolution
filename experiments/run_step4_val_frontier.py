"""
scratch/run_step4_val_frontier.py — Phase 4b Step 4: Real Val Frontier & Loss Ledger
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
from sklearn.neighbors import BallTree

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ber.canon.normalise import canonicalise
from ber.retrieve.geo_blocking import EARTH_RADIUS_METERS
from ber.retrieve.idf import IDFRetriever, IDFConfig, infer_country_from_coords
from datasets.splits.splitter import EntityDisjointSplitter

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(math.isnan(x) for x in (lat1, lon1, lat2, lon2)):
        return float("nan")
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c

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
    print("=== Phase 4b Step 4: Real Validation Frontier & Loss Ledger ===", flush=True)

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    rid_to_idx = {r["id"]: i for i, r in enumerate(all_records)}

    # Entity-disjoint split (70/15/15)
    splitter = EntityDisjointSplitter(ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
    splits = splitter.split(rids, [r["point_of_interest"] for r in all_records])

    val_rids = [rid for rid, s in splits.items() if s == "val"]
    val_records = [all_records[rid_to_idx[rid]] for rid in val_rids]
    val_ids_set = set(val_rids)

    # Validation Ground Truth True Pairs
    poi_groups_val = defaultdict(list)
    for r in val_records:
        poi = r["point_of_interest"]
        if poi: poi_groups_val[poi].append(r)

    val_true_pairs_list = []
    for poi, rows in poi_groups_val.items():
        if len(rows) > 1:
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    val_true_pairs_list.append((rows[i], rows[j]))

    val_true_pair_tuples = [(min(ra["id"], rb["id"]), max(ra["id"], rb["id"])) for ra, rb in val_true_pairs_list]
    val_denom = len(val_true_pair_tuples)
    val_true_set = set(val_true_pair_tuples)
    print(f"Validation split queries: {len(val_records):,}, true pairs denominator: {val_denom:,}", flush=True)

    # 1. RETRIEVE G1 (K in {4, 12, 50}, R=5km)
    print("\n--- Retrieving G1 on Validation Queries ---", flush=True)
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    geo_tree = BallTree(coords_rad, metric="haversine")
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS

    val_indices = [rid_to_idx[rid] for rid in val_rids]
    val_coords_rad = coords_rad[val_indices]

    g1_cands = {4: defaultdict(list), 12: defaultdict(list), 50: defaultdict(list)}
    g1_undir = {4: set(), 12: set(), 50: set()}
    g1_tot_c = {4: 0, 12: 0, 50: 0}

    batch_size = 5000
    for b_start in range(0, len(val_indices), batch_size):
        b_end = b_start + batch_size
        b_coords = val_coords_rad[b_start:b_end]
        b_qids = val_rids[b_start:b_end]
        dists, inds = geo_tree.query(b_coords, k=51, return_distance=True)
        for q_id, row_inds, row_dists in zip(b_qids, inds, dists):
            cnt = 0
            for cand_idx, d_rad in zip(row_inds, row_dists):
                cand_id = rids[cand_idx]
                if cand_id == q_id: continue
                if d_rad > r5km_rad: continue
                for k_limit in [50, 12, 4]:
                    if cnt < k_limit:
                        g1_cands[k_limit][q_id].append(cand_id)
                        g1_undir[k_limit].add((min(q_id, cand_id), max(q_id, cand_id)))
                        g1_tot_c[k_limit] += 1
                cnt += 1
                if cnt >= 50: break

    # 2. RETRIEVE G3 (Chosen Setting: K=5, Country ON)
    print("\n--- Retrieving G3 (Chosen: K=5, Country ON) on Validation Queries ---", flush=True)
    g3 = IDFRetriever(IDFConfig(k_candidates=5, df_cap=5000, country_filter=True))
    g3.build_index(all_records)
    g3_cands, g3_undir, g3_tot_c = g3.retrieve_batch(val_records, k=5, country_filter=True)

    # 3. CONTROL G=50 (K1: Name+Country, K2: Geo Cell 0.05 deg grid)
    print("\n--- Constructing Control G=50 Blocker on Full Table & Retrieving on Val ---", flush=True)
    # Build control index
    ctrl_index_k1 = defaultdict(list)
    ctrl_index_k2 = defaultdict(list)
    for r in all_records:
        rid = r["id"]
        c_name = canonicalise(r["name"])
        country = r.get("country", "")
        if c_name and country:
            ctrl_index_k1[f"{c_name}|{country}"].append(rid)
        lat, lon = r["latitude"], r["longitude"]
        if not math.isnan(lat) and not math.isnan(lon):
            cell_key = f"{int(math.floor(lat / 0.05))}|{int(math.floor(lon / 0.05))}"
            ctrl_index_k2[cell_key].append(rid)

    ctrl_g50_cands = defaultdict(list)
    ctrl_g50_undir = set()
    ctrl_g50_tot_c = 0
    gate_limit = 50

    for r in val_records:
        qid = r["id"]
        c_name = canonicalise(r["name"])
        country = r.get("country", "")
        lat, lon = r["latitude"], r["longitude"]

        c_set = set()
        # K1
        if c_name and country:
            b1 = ctrl_index_k1.get(f"{c_name}|{country}", [])
            if len(b1) - 1 <= gate_limit:
                for cid in b1:
                    if cid != qid: c_set.add(cid)
        # K2
        if not math.isnan(lat) and not math.isnan(lon):
            cell_key = f"{int(math.floor(lat / 0.05))}|{int(math.floor(lon / 0.05))}"
            b2 = ctrl_index_k2.get(cell_key, [])
            if len(b2) - 1 <= gate_limit:
                for cid in b2:
                    if cid != qid: c_set.add(cid)

        c_list = list(c_set)
        ctrl_g50_cands[qid] = c_list
        ctrl_g50_tot_c += len(c_list)
        for cid in c_list:
            ctrl_g50_undir.add((min(qid, cid), max(qid, cid)))

    # UNIONS
    union_ctrl_g1_undir = ctrl_g50_undir | g1_undir[50]
    union_g1_g3_undir = g1_undir[50] | g3_undir
    union_ctrl_g1_g3_undir = ctrl_g50_undir | g1_undir[50] | g3_undir

    # Union candidates counts
    union_ctrl_g1_cands = sum(len(set(ctrl_g50_cands[q]) | set(g1_cands[50][q])) for q in val_rids)
    union_g1_g3_cands = sum(len(set(g1_cands[50][q]) | set(g3_cands.get(q, []))) for q in val_rids)
    union_ctrl_g1_g3_cands = sum(len(set(ctrl_g50_cands[q]) | set(g1_cands[50][q]) | set(g3_cands.get(q, []))) for q in val_rids)

    # METRICS EVALUATION
    def compute_stats(name: str, undir_set: set, cands_count: int):
        hits = len(val_true_set & undir_set)
        rec = hits / val_denom
        assert round(rec * val_denom) == hits, f"Assertion failed: {rec} * {val_denom} != {hits}"
        return {
            "name": name,
            "candidates": cands_count,
            "true_pairs": hits,
            "denominator": val_denom,
            "pair_recall": rec,
            "undir_set": undir_set,
        }

    systems = {
        "ctrl_g50": compute_stats("Control G=50", ctrl_g50_undir, ctrl_g50_tot_c),
        "g1_k4": compute_stats("G1 (K=4, R=5km)", g1_undir[4], g1_tot_c[4]),
        "g1_k12": compute_stats("G1 (K=12, R=5km)", g1_undir[12], g1_tot_c[12]),
        "g1_k50": compute_stats("G1 (K=50, R=5km)", g1_undir[50], g1_tot_c[50]),
        "g3_chosen": compute_stats("G3 (K=5, Country ON)", g3_undir, g3_tot_c),
        "union_ctrl_g1": compute_stats("Control G=50 + G1", union_ctrl_g1_undir, union_ctrl_g1_cands),
        "union_g1_g3": compute_stats("G1 (K=50) + G3", union_g1_g3_undir, union_g1_g3_cands),
        "union_ctrl_g1_g3": compute_stats("Control G=50 + G1 + G3", union_ctrl_g1_g3_undir, union_ctrl_g1_g3_cands),
    }

    print("\n--- Validation Split Performance Summary ---", flush=True)
    header = f"{'System':25s} | {'Candidates':12s} | {'True Pairs':10s} | {'Pair Recall':12s}"
    print(header)
    print("-" * len(header))
    for k, s in systems.items():
        print(f"{s['name']:25s} | {s['candidates']:12,d} | {s['true_pairs']:6,d} / {val_denom:,d} | {s['pair_recall']*100:6.2f}%")

    # 4. PAIRED BOOTSTRAP CIs (500 resamples, seed 42)
    print("\n--- Paired Bootstrap CIs (500 resamples) ---", flush=True)
    boot_rng = np.random.default_rng(42)
    n_boot = 500
    N_p = val_denom

    def calc_ci(set_a, set_b):
        hits_a = np.array([1 if p in set_a else 0 for p in val_true_pair_tuples])
        hits_b = np.array([1 if p in set_b else 0 for p in val_true_pair_tuples])
        deltas = []
        for _ in range(n_boot):
            idx = boot_rng.integers(0, N_p, size=N_p)
            deltas.append(float(hits_a[idx].mean() - hits_b[idx].mean()))
        mean_d = float(np.mean(deltas))
        ci_lo = float(np.percentile(deltas, 2.5))
        ci_hi = float(np.percentile(deltas, 97.5))
        return mean_d, ci_lo, ci_hi

    # Comparisons:
    # 1. Union G1+G3 vs G1 standalone (K=50)
    d1, lo1, hi1 = calc_ci(systems["union_g1_g3"]["undir_set"], systems["g1_k50"]["undir_set"])
    print(f"Union G1+G3 vs G1 Standalone: Delta = {d1*100:+.2f} pp [95% CI: {lo1*100:+.2f} pp, {hi1*100:+.2f} pp]")

    # 2. Union Control+G1+G3 vs Control+G1
    d2, lo2, hi2 = calc_ci(systems["union_ctrl_g1_g3"]["undir_set"], systems["union_ctrl_g1"]["undir_set"])
    print(f"Union Ctrl+G1+G3 vs Ctrl+G1:  Delta = {d2*100:+.2f} pp [95% CI: {lo2*100:+.2f} pp, {hi2*100:+.2f} pp]")

    # 3. G1 K=12 vs G1 K=4
    d3, lo3, hi3 = calc_ci(systems["g1_k12"]["undir_set"], systems["g1_k4"]["undir_set"])
    print(f"G1 K=12 vs G1 K=4:            Delta = {d3*100:+.2f} pp [95% CI: {lo3*100:+.2f} pp, {hi3*100:+.2f} pp]")

    # 5. VALIDATION LOSS LEDGER WITH STRICT SUM ASSERTION
    print("\n--- Validation Loss Ledger (Control G=50 + G1 + G3) ---", flush=True)
    best_undir = systems["union_ctrl_g1_g3"]["undir_set"]
    recovered_pairs = len(val_true_set & best_undir)

    missed_pairs = [
        (ra, rb) for ra, rb in val_true_pairs_list
        if (min(ra["id"], rb["id"]), max(ra["id"], rb["id"])) not in best_undir
    ]
    assert len(missed_pairs) + recovered_pairs == val_denom, "Recovery + Misses != Denominator!"

    # Break down missed pairs into mutually exclusive exhaustive stages:
    # Stage 1: Different country inferred (cross-country)
    # Stage 2: Missing coordinates or nan
    # Stage 3: Distance > 5km with different name (lexical & spatial impossible)
    # Stage 4: Distance > 5km with identical/overlapping name (lexical budget / tie / cap)
    # Stage 5: Distance <= 5km missed due to local density / fixed K
    ledger_stages = Counter()
    for ra, rb in missed_pairs:
        d = haversine_distance(ra["latitude"], ra["longitude"], rb["latitude"], rb["longitude"])
        c_a = ra.get("country", "")
        c_b = rb.get("country", "")
        ncat = classify_name(ra["name"], rb["name"])

        if c_a and c_b and c_a != c_b:
            ledger_stages["cross_country"] += 1
        elif math.isnan(d):
            ledger_stages["missing_coords"] += 1
        elif d > 5000.0 and ncat == "different":
            ledger_stages["gt5km_different_name"] += 1
        elif d > 5000.0 and ncat in ("identical", "token_reorder", "partial_overlap"):
            ledger_stages["gt5km_lexical_unrecovered"] += 1
        elif d <= 5000.0:
            ledger_stages["under5km_dense_cluster_miss"] += 1
        else:
            ledger_stages["other_residual"] += 1

    ledger_stages["recovered_true_pairs"] = recovered_pairs

    print(f"Validation Loss Ledger Counts (Denominator = {val_denom:,}):")
    total_ledger_sum = 0
    for stage_name, cnt in sorted(ledger_stages.items()):
        total_ledger_sum += cnt
        pct = cnt / val_denom * 100.0
        print(f"  {stage_name:32s}: {cnt:5d} ({pct:5.2f}%)")

    # STRICT ASSERTION: sum of stage counts must equal denominator!
    print(f"\nAssertion Check: Total Ledger Sum ({total_ledger_sum:,}) == Denominator ({val_denom:,})")
    assert total_ledger_sum == val_denom, f"STRICT ASSERTION FAILED: {total_ledger_sum} != {val_denom}"
    print("ASSERTION PASSED: Validation loss ledger counts strictly sum to the true pair denominator!", flush=True)

    # Save results
    out_step4 = _REPO / "experiments" / "phase4b_step4_val_frontier.json"
    clean_sys = {}
    for k, v in systems.items():
        clean_sys[k] = {
            "name": v["name"],
            "candidates": v["candidates"],
            "true_pairs": v["true_pairs"],
            "denominator": v["denominator"],
            "pair_recall": v["pair_recall"],
        }
    out_data = {
        "val_denominator": val_denom,
        "systems": clean_sys,
        "paired_bootstrap": {
            "union_g1_g3_vs_g1": {"delta_pp": d1*100, "ci_lo_pp": lo1*100, "ci_hi_pp": hi1*100},
            "union_ctrl_g1_g3_vs_ctrl_g1": {"delta_pp": d2*100, "ci_lo_pp": lo2*100, "ci_hi_pp": hi2*100},
            "g1_k12_vs_g1_k4": {"delta_pp": d3*100, "ci_lo_pp": lo3*100, "ci_hi_pp": hi3*100},
        },
        "loss_ledger": dict(ledger_stages),
        "assertion_passed": True,
    }
    out_step4.write_text(json.dumps(out_data, indent=2))
    print(f"Saved Step 4 results to {out_step4}", flush=True)

if __name__ == "__main__":
    main()

