"""
experiments/run_step5_freeze_retrieval.py — Phase 4c Step 5: Freeze Retrieval v1 & Candidate Store
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
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
from ber.retrieve.geo_blocking import EARTH_RADIUS_METERS
from ber.retrieve.idf import IDFRetriever, IDFConfig, infer_country_from_coords
from ber.store.schemas import CANDIDATES_SCHEMA
from datasets.splits.splitter import EntityDisjointSplitter

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(math.isnan(x) for x in (lat1, lon1, lat2, lon2)):
        return float("nan")
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c

def main():
    print("=== Phase 4c Step 5: Freeze Retrieval v1 + Candidate Store ===", flush=True)
    t0_all = time.time()

    coord_path = _REPO / "ber" / "data" / "raw" / "foursquare_kaggle" / "coordinates.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT id, name, latitude, longitude, point_of_interest FROM '{coord_path.as_posix()}'").fetchdf()
    print(f"Loaded {len(df):,} records.", flush=True)

    all_records = df.to_dict(orient="records")
    for r in all_records:
        r["country"] = infer_country_from_coords(r["latitude"], r["longitude"])
    rids = [r["id"] for r in all_records]
    rid_to_idx = {r["id"]: i for i, r in enumerate(all_records)}

    # Attributes
    doc_canon_name = [canonicalise(r["name"]) for r in all_records]
    doc_country = [str(r.get("country") or "").strip().upper() for r in all_records]
    doc_lat = np.array([float(r.get("latitude", np.nan)) for r in all_records], dtype=np.float64)
    doc_lon = np.array([float(r.get("longitude", np.nan)) for r in all_records], dtype=np.float64)
    doc_lat_rad = np.radians(doc_lat)
    doc_lon_rad = np.radians(doc_lon)
    doc_cos_lat = np.cos(doc_lat_rad)

    # BallTree for G1
    print("Building BallTree index...", flush=True)
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    geo_tree = BallTree(coords_rad, metric="haversine")
    r5km_rad = 5000.0 / EARTH_RADIUS_METERS

    # G3 IDF Index
    print("Building G3 IDF index...", flush=True)
    g3 = IDFRetriever(IDFConfig(k_candidates=20, df_cap=5000, country_filter=True))
    g3.build_index(all_records)
    effective_df_cap = g3.get_effective_df_cap()

    # Control Blocker tables
    print("Building Control G=50 blocker tables...", flush=True)
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
    gate_limit = 50

    # Splits
    splitter = EntityDisjointSplitter(ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)
    splits = splitter.split(rids, [r["point_of_interest"] for r in all_records])

    # Val split
    val_rids = [rid for rid, s in splits.items() if s == "val"]
    val_records = [all_records[rid_to_idx[rid]] for rid in val_rids]
    val_indices = [rid_to_idx[rid] for rid in val_rids]

    # Fixed 100k Train sample (seed 42)
    rng = random.Random(42)
    all_indices = list(range(len(df)))
    train_sample_indices = sorted(rng.sample(all_indices, 100_000))
    train_sample_records = [all_records[i] for i in train_sample_indices]
    train_sample_rids = [r["id"] for r in train_sample_records]

    # Ground truth true pairs
    # Train
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
    print(f"Train sample true pairs: {tr_denom:,}", flush=True)

    # Val
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

    # Retrieval runner for frozen channels:
    # bit 0 (1): Control G=50
    # bit 1 (2): G1 K_pool=100, R=5km
    # bit 2 (4): G3 Country ON K=20 with deterministic tie-break
    def generate_candidates(query_records, query_indices, qids_list, label=""):
        print(f"\n--- Generating Retrieval v1 Candidates for {label} ({len(qids_list):,} queries) ---", flush=True)
        t_gen_0 = time.time()
        
        # 1. Channel 0: Control G=50
        t0_c = time.time()
        cands_ctrl = defaultdict(dict) # qid -> cid -> score
        for r in query_records:
            qid = r["id"]
            c_name = canonicalise(r["name"])
            country = r.get("country", "")
            lat, lon = r["latitude"], r["longitude"]
            if c_name and country:
                b1 = ctrl_k1.get(f"{c_name}|{country}", [])
                if len(b1) - 1 <= gate_limit:
                    for cid in b1:
                        if cid != qid:
                            cands_ctrl[qid][cid] = 1.0
            if not math.isnan(lat) and not math.isnan(lon):
                cell_key = f"{int(math.floor(lat / 0.05))}|{int(math.floor(lon / 0.05))}"
                b2 = ctrl_k2.get(cell_key, [])
                if len(b2) - 1 <= gate_limit:
                    for cid in b2:
                        if cid != qid:
                            cands_ctrl[qid][cid] = 1.0
        print(f"  Channel 0 (Control G=50) completed in {time.time()-t0_c:.2f}s", flush=True)

        # 2. Channel 1: G1 K_pool=100, R=5km
        t0_g1 = time.time()
        cands_g1 = defaultdict(dict) # qid -> cid -> score
        q_coords_rad = coords_rad[query_indices]
        batch_size = 5000
        for b_start in range(0, len(query_indices), batch_size):
            b_end = min(b_start + batch_size, len(query_indices))
            b_coords = q_coords_rad[b_start:b_end]
            b_qids = qids_list[b_start:b_end]
            dists, inds = geo_tree.query(b_coords, k=101, return_distance=True)
            for qid, row_inds, row_dists in zip(b_qids, inds, dists):
                cnt = 0
                for cand_idx, d_rad in zip(row_inds, row_dists):
                    cid = rids[cand_idx]
                    if cid == qid: continue
                    if d_rad > r5km_rad: continue
                    d_m = d_rad * EARTH_RADIUS_METERS
                    # Proximity score in [0, 1]
                    prox_score = max(0.0, 1.0 - (d_m / 5000.0))
                    cands_g1[qid][cid] = prox_score
                    cnt += 1
                    if cnt >= 100: break
        print(f"  Channel 1 (G1 K=100) completed in {time.time()-t0_g1:.2f}s", flush=True)

        # 3. Channel 2: G3 Country ON K=20 with deterministic tie-break
        t0_g3 = time.time()
        cands_g3 = defaultdict(dict) # qid -> cid -> score
        for q_rec in query_records:
            qid = q_rec["id"]
            q_idx = rid_to_idx[qid]
            q_canon = doc_canon_name[q_idx]
            if not q_canon:
                continue

            q_country = doc_country[q_idx]
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

            cand_scores = defaultdict(float)
            for tid in active_tids:
                idf_val = g3.idf_table[tid]
                for doc_idx in g3.postings[tid]:
                    if doc_idx == q_idx:
                        continue
                    if q_country and doc_country[doc_idx] and doc_country[doc_idx] != q_country:
                        continue
                    cand_scores[doc_idx] += idf_val

            if not cand_scores:
                continue

            # Deterministic tie break
            cand_list = []
            for doc_idx, sc in cand_scores.items():
                exact_name = 1 if doc_canon_name[doc_idx] == q_canon else 0
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
                cand_list.append((doc_idx, -sc, -exact_name, -same_ctry, dist, cand_id))

            cand_list.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5]))
            for item in cand_list[:20]:
                cid = item[5]
                cands_g3[qid][cid] = -item[1] # raw IDF score

        print(f"  Channel 2 (G3 K=20 Deterministic) completed in {time.time()-t0_g3:.2f}s", flush=True)

        # Merge candidate sets with bitmask
        print("  Merging candidates into unified candidate pool...", flush=True)
        merged_cands_by_q = {}
        for qid in qids_list:
            cands_set = set(cands_ctrl.get(qid, {}).keys()) | set(cands_g1.get(qid, {}).keys()) | set(cands_g3.get(qid, {}).keys())
            if not cands_set:
                merged_cands_by_q[qid] = []
                continue

            q_cand_items = []
            for cid in cands_set:
                mask = 0
                if cid in cands_ctrl.get(qid, {}):
                    mask |= 1
                if cid in cands_g1.get(qid, {}):
                    mask |= 2
                if cid in cands_g3.get(qid, {}):
                    mask |= 4

                # Primary score: G3 IDF score if available, else G1 proximity score, else 1.0
                if mask & 4:
                    score = cands_g3[qid][cid]
                elif mask & 2:
                    score = cands_g1[qid][cid]
                else:
                    score = cands_ctrl[qid][cid]

                popcnt = bin(mask).count("1")
                q_cand_items.append({
                    "cand_id": cid,
                    "channel_mask": mask,
                    "score": score,
                    "popcount": popcnt,
                })

            # Sort within query candidate set:
            # Multi-channel consensus first (popcount desc), then score desc, then cand_id
            q_cand_items.sort(key=lambda x: (-x["popcount"], -x["score"], x["cand_id"]))
            merged_cands_by_q[qid] = q_cand_items

        print(f"  Finished {label} in {time.time()-t_gen_0:.2f}s", flush=True)
        return merged_cands_by_q

    # Run for Train Sample (100k) and Val (22k)
    train_cands_by_q = generate_candidates(train_sample_records, train_sample_indices, train_sample_rids, label="TRAIN")
    val_cands_by_q = generate_candidates(val_records, val_indices, val_rids, label="VAL")

    # Evaluate Train Curve: Recall vs per-query cap
    print("\n--- Train Recall vs Per-Query Cap Curve (Evaluating Cap C) ---", flush=True)
    caps = [10, 20, 30, 40, 50, 60, 75, 100, 150, 9999]
    cap_curve = {}
    uncapped_train_pairs = set()

    for qid, items in train_cands_by_q.items():
        for item in items:
            cid = item["cand_id"]
            uncapped_train_pairs.add((min(qid, cid), max(qid, cid)))
    uncapped_tr_hits = len(tr_true_pairs & uncapped_train_pairs)
    uncapped_tr_rec = uncapped_tr_hits / tr_denom
    print(f"Uncapped Train Pool: Recall = {uncapped_tr_rec*100:.2f}% ({uncapped_tr_hits:,}/{tr_denom:,}), Total Cands = {sum(len(v) for v in train_cands_by_q.values()):,}")

    for c in caps:
        c_label = f"cap_{c}" if c < 9999 else "uncapped"
        c_pairs = set()
        tot_c = 0
        for qid, items in train_cands_by_q.items():
            capped_items = items[:c]
            tot_c += len(capped_items)
            for item in capped_items:
                cid = item["cand_id"]
                c_pairs.add((min(qid, cid), max(qid, cid)))
        hits = len(tr_true_pairs & c_pairs)
        rec = hits / tr_denom
        budget_prune_loss = uncapped_tr_hits - hits
        pct_preserved = (rec / uncapped_tr_rec) * 100.0 if uncapped_tr_rec else 100.0
        print(f"  Cap C={c:4d}: Recall={rec*100:.2f}% ({hits:,}/{tr_denom:,}), Preserved={pct_preserved:.2f}%, Pruned={budget_prune_loss:,} pairs, CPQ={tot_c/len(train_sample_rids):.1f}")
        cap_curve[c_label] = {
            "cap": c if c < 9999 else None,
            "recall": rec,
            "hits": hits,
            "total_candidates": tot_c,
            "cpq": tot_c / len(train_sample_rids),
            "budget_prune_pairs": budget_prune_loss,
            "pct_preserved": pct_preserved,
        }

    # Selected Cap from Train Curve:
    # Cap C=100 preserves 99.85% of pool recall with manageable CPQ; Cap C=75 preserves 99.64%.
    # Let's check C=100 as the frozen retrieval v1 pool cap.
    chosen_cap = 100
    print(f"\nFrozen Retrieval v1 Cap Selected: C={chosen_cap} (preserves {cap_curve[f'cap_{chosen_cap}']['pct_preserved']:.2f}% of pool recall)")

    # Measure BUDGET_PRUNE on Val
    print("\n--- Validation Split Evaluation & Loss Ledger ---", flush=True)
    uncapped_val_pairs = set()
    for qid, items in val_cands_by_q.items():
        for item in items:
            cid = item["cand_id"]
            uncapped_val_pairs.add((min(qid, cid), max(qid, cid)))
    uncapped_val_hits = len(val_true_pairs & uncapped_val_pairs)
    uncapped_val_rec = uncapped_val_hits / val_denom

    # Capped Val pairs
    capped_val_pairs = set()
    val_tot_cands = 0
    for qid, items in val_cands_by_q.items():
        capped_items = items[:chosen_cap]
        val_tot_cands += len(capped_items)
        for item in capped_items:
            cid = item["cand_id"]
            capped_val_pairs.add((min(qid, cid), max(qid, cid)))
    val_hits = len(val_true_pairs & capped_val_pairs)
    val_rec = val_hits / val_denom
    val_budget_prune = uncapped_val_hits - val_hits

    print(f"Validation Uncapped Pool: Recall = {uncapped_val_rec*100:.2f}% ({uncapped_val_hits:,}/{val_denom:,})")
    print(f"Validation Capped Pool (C={chosen_cap}): Recall = {val_rec*100:.2f}% ({val_hits:,}/{val_denom:,}), Total Cands = {val_tot_cands:,}, CPQ = {val_tot_cands/len(val_rids):.1f}")
    print(f"Validation BUDGET_PRUNE Loss: {val_budget_prune:,} pairs ({val_budget_prune/val_denom*100:.2f} pp)")

    # Validation Loss Ledger Breakdown
    # Categorize all 17,916 true pairs strictly
    loss_ledger = defaultdict(int)
    for pair in val_true_pairs:
        aid, bid = pair
        ra, rb = all_records[rid_to_idx[aid]], all_records[rid_to_idx[bid]]
        if pair in capped_val_pairs:
            loss_ledger["RECOVERED_IN_STORE"] += 1
        elif pair in uncapped_val_pairs:
            # Was in unpruned candidate pool, but dropped by C=100 cap
            loss_ledger["BUDGET_PRUNE"] += 1
        else:
            # Missed by retrieval channels
            dist = haversine_distance(ra["latitude"], ra["longitude"], rb["latitude"], rb["longitude"])
            ca = doc_canon_name[rid_to_idx[aid]]
            cb = doc_canon_name[rid_to_idx[bid]]
            has_tok_overlap = bool(ca and cb and (set(ca.split()) & set(cb.split())))

            if not math.isnan(dist) and dist > 5000.0:
                if not has_tok_overlap:
                    loss_ledger["NO_GEO_AND_NO_LEXICAL_OVERLAP_GT_5KM"] += 1
                else:
                    loss_ledger["LEXICAL_PRESENT_BUT_GT_5KM_DROPPED_BY_G3_K20"] += 1
            elif not math.isnan(dist) and dist <= 5000.0:
                # Within 5km, but dropped because dense cluster exceeded K_pool=100
                loss_ledger["LOCAL_DENSITY_EXCEEDED_K100_LE_5KM"] += 1
            else:
                loss_ledger["UNINDEXED_OR_MISSING_COORDINATES"] += 1

    print("\n--- Validation Loss Ledger (Strict Sum to Denominator) ---")
    total_ledger = sum(loss_ledger.values())
    for category, count in sorted(loss_ledger.items(), key=lambda x: -x[1]):
        pct = count / val_denom * 100.0
        print(f"  {category:<50}: {count:>6,} pairs ({pct:5.2f}%)")
    print(f"  {'TOTAL':<50}: {total_ledger:>6,} pairs (Denominator: {val_denom:,})")

    # STRICT ASSERTION
    assert total_ledger == val_denom, f"Loss ledger sum ({total_ledger}) != denominator ({val_denom})"
    print("ASSERTION PASSED: Loss ledger strictly sums to 17,916!", flush=True)

    # Build and Write Parquet Candidate Store
    # For both Train Sample and Val:
    # Schema: query_id (str), cand_id (str), channel_mask (uint32), score (float16), rank (uint16), reverse_rank (uint16), round (uint8)
    def write_candidate_store(cands_by_q, qids_list, out_path, cap=chosen_cap):
        print(f"\nConstructing Parquet candidate store for {out_path.name}...", flush=True)
        # Compute reverse rank lookup:
        # cand_to_q_rank[cid][qid] = rank
        cand_to_q_rank = defaultdict(dict)
        for qid in qids_list:
            items = cands_by_q.get(qid, [])[:cap]
            for rk, item in enumerate(items, 1):
                cid = item["cand_id"]
                cand_to_q_rank[cid][qid] = rk

        records = []
        for qid in qids_list:
            items = cands_by_q.get(qid, [])[:cap]
            for rk, item in enumerate(items, 1):
                cid = item["cand_id"]
                rev_rk = cand_to_q_rank.get(qid, {}).get(cid, 0)
                records.append({
                    "query_id": qid,
                    "cand_id": cid,
                    "channel_mask": int(item["channel_mask"]),
                    "score": float(item["score"]),
                    "rank": int(rk),
                    "reverse_rank": int(rev_rk),
                    "round": 1,
                    "dataset_id": "foursquare_kaggle",
                })

        # Sort explicitly by (query_id, cand_id)
        records.sort(key=lambda x: (x["query_id"], x["cand_id"]))
        print(f"  Writing {len(records):,} rows to {out_path}...", flush=True)

        tbl = pa.Table.from_pylist(records, schema=CANDIDATES_SCHEMA)
        pq.write_table(tbl, out_path, compression="zstd")
        print(f"  Saved {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)

    val_store_path = _REPO / "experiments" / "candidates_frozen_val.parquet"
    write_candidate_store(val_cands_by_q, val_rids, val_store_path, cap=chosen_cap)

    train_store_path = _REPO / "experiments" / "candidates_frozen_train.parquet"
    write_candidate_store(train_cands_by_q, train_sample_rids, train_store_path, cap=chosen_cap)

    # Config file: configs/retrieval_frozen.yaml
    config_dict = {
        "version": "1.0",
        "date": "2026-10-12",
        "description": "BER Frozen Retrieval v1 Configuration",
        "corpus": {
            "canonical_file": "ber/data/raw/foursquare_kaggle/coordinates.parquet",
            "total_records": len(df),
            "val_denominator_pairs": val_denom,
            "val_denominator_slots": val_denom * 2,
        },
        "channels": {
            "bit_0": {
                "name": "control_g50",
                "bitmask": 1,
                "gate": 50,
                "keys": ["canon_name|country", "lat_lon_grid_0.05deg"],
            },
            "bit_1": {
                "name": "geo_g1",
                "bitmask": 2,
                "radius_meters": 5000.0,
                "k_pool": 100,
                "metric": "haversine",
            },
            "bit_2": {
                "name": "idf_g3",
                "bitmask": 4,
                "k_candidates": 20,
                "df_cap": 5000,
                "country_filter": True,
                "tie_break": "deterministic (exact_name -> same_country -> haversine -> id)",
            },
        },
        "post_processing": {
            "per_query_cap": chosen_cap,
            "sorting": ["popcount desc", "primary_score desc", "cand_id asc"],
        },
        "validation_performance": {
            "pair_recall": val_rec,
            "pair_hits": val_hits,
            "total_candidates": val_tot_cands,
            "cpq": val_tot_cands / len(val_rids),
            "budget_prune_loss_pairs": val_budget_prune,
            "budget_prune_loss_pp": (val_budget_prune / val_denom) * 100.0,
        },
    }

    def dump_yaml(data, indent=0):
        lines = []
        prefix = "  " * indent
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    lines.append(dump_yaml(v, indent + 1))
                else:
                    if isinstance(v, bool):
                        val_str = "true" if v else "false"
                    elif v is None:
                        val_str = "null"
                    elif isinstance(v, str) and (":" in v or " " in v or "#" in v):
                        val_str = f'"{v}"'
                    else:
                        val_str = str(v)
                    lines.append(f"{prefix}{k}: {val_str}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(dump_yaml(item, indent + 1))
                else:
                    item_str = f'"{item}"' if isinstance(item, str) and (" " in item or ":" in item) else str(item)
                    lines.append(f"{prefix}- {item_str}")
        return "\n".join(lines)

    config_yaml_str = dump_yaml(config_dict)
    config_hash = hashlib.sha256(config_yaml_str.encode("utf-8")).hexdigest()
    config_dict["config_hash"] = config_hash
    config_yaml_str = f"# config_hash: {config_hash}\n" + dump_yaml(config_dict)

    cfg_out_path = _REPO / "configs" / "retrieval_frozen.yaml"
    cfg_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_out_path, "w", encoding="utf-8") as f:
        f.write(config_yaml_str + "\n")
    print(f"\nSaved frozen config to {cfg_out_path} with SHA256: {config_hash}", flush=True)

    summary_out = _REPO / "experiments" / "step5_freeze_results.json"
    summary_data = {
        "config_hash": config_hash,
        "cap_curve": cap_curve,
        "chosen_cap": chosen_cap,
        "val_performance": {
            "hits": val_hits,
            "recall": val_rec,
            "total_candidates": val_tot_cands,
            "cpq": val_tot_cands / len(val_rids),
            "budget_prune_loss": val_budget_prune,
        },
        "loss_ledger": dict(loss_ledger),
    }
    summary_out.write_text(json.dumps(summary_data, indent=2))
    print(f"Saved Step 5 summary to {summary_out}", flush=True)

if __name__ == "__main__":
    main()
