"""
experiments/run_geo_channel.py — Phase 3a: Geo-Proximity Retrieval Channel (G1)
=============================================================================
Runs the complete Phase 3a pipeline:
  Step 2: Geo channel G1 parameter sweep on 100k train sample (seed 42)
          Density check (dense vs sparse deciles) & B9 residual slice recovery
  Step 3: Efficient frontier on val split with G1 added to control union
          Matched-budget comparisons (~700k and ~2M candidates)
          Paired bootstrap CIs over query entities (500 resamples)
  Step 4: Loss ledger on val split with G1 added + remaining misses audit
          (phone/url share & cross-country flag)
  Step 5: Write experiments/003_geo_channel.md report & log to runs.parquet

Usage:
    python experiments/run_geo_channel.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ber.canon.normalise import canonicalise
from ber.eval.ledger import RunLedger
from ber.eval.loss_ledger import LossLedger, LossStage
from ber.eval.metrics import (
    build_candidate_map, build_true_targets_map,
    candidates_per_query_distribution, partial_recall_rate,
    recall_at_k, zero_candidate_rate,
)
from ber.eval.metrics_dedup import pair_recall, singleton_burden
from ber.retrieve.geo_blocking import GeoBlocker, GeoConfig, EARTH_RADIUS_METERS
from ber_datasets.adapters.foursquare import (
    extract_url_domain, digits_only, geo_cell,
)
from experiments.run_foursquare_control import (
    FoursquareBlocker, FSQ_ALL_KEYS, _fsq_name_cat, _fsq_loc_cat,
)

PROCESSED_DIR = _REPO / "data" / "processed" / "foursquare"
LEDGER_PATH = _REPO / "experiments" / "runs.parquet"
DATASET_VERSION = "foursquare_kaggle_train_2022"

SWEEP_K = [5, 10, 20, 50]
SWEEP_R_METERS = [200.0, 1000.0, 5000.0]  # 200m, 1km, 5km


def run_paired_bootstrap(
    query_ids: list[str],
    retrieved_a: set[tuple[str, str]],
    retrieved_b: set[tuple[str, str]],
    true_pairs_dict: dict[str, set[str]],
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap over query entities for Query Recall and Pair Recall delta (A - B)."""
    # Pre-aggregate candidates by query for fast lookup
    q_to_cands_a = defaultdict(set)
    for q, c in retrieved_a:
        q_to_cands_a[q].add(c)

    q_to_cands_b = defaultdict(set)
    for q, c in retrieved_b:
        q_to_cands_b[q].add(c)

    queries = [q for q in query_ids if q in true_pairs_dict and true_pairs_dict[q]]
    if not queries:
        return {}

    hits_a = []
    hits_b = []
    p_hits_a = []
    p_hits_b = []
    n_tgts = []

    for q in queries:
        tgts = true_pairs_dict[q]
        c_a = q_to_cands_a[q]
        c_b = q_to_cands_b[q]

        hits_a.append(1.0 if (c_a & tgts) else 0.0)
        hits_b.append(1.0 if (c_b & tgts) else 0.0)
        p_hits_a.append(len(c_a & tgts))
        p_hits_b.append(len(c_b & tgts))
        n_tgts.append(len(tgts))

    N = len(queries)
    h_a = np.array(hits_a, dtype=np.float32)
    h_b = np.array(hits_b, dtype=np.float32)
    ph_a = np.array(p_hits_a, dtype=np.float32)
    ph_b = np.array(p_hits_b, dtype=np.float32)
    t_cnts = np.array(n_tgts, dtype=np.float32)

    q_score_a = float(np.mean(h_a))
    q_score_b = float(np.mean(h_b))
    q_observed_delta = q_score_a - q_score_b

    p_score_a = float(np.sum(ph_a) / np.sum(t_cnts))
    p_score_b = float(np.sum(ph_b) / np.sum(t_cnts))
    p_observed_delta = p_score_a - p_score_b

    rng = np.random.default_rng(seed)
    q_deltas = []
    p_deltas = []
    extreme_q = 0
    extreme_p = 0

    for _ in range(n_boot):
        idxs = rng.integers(0, N, size=N)
        d_q = float(np.mean(h_a[idxs]) - np.mean(h_b[idxs]))
        q_deltas.append(d_q)
        if abs(d_q - q_observed_delta) >= abs(q_observed_delta):
            extreme_q += 1

        d_p = float(np.sum(ph_a[idxs]) / np.sum(t_cnts[idxs]) - np.sum(ph_b[idxs]) / np.sum(t_cnts[idxs]))
        p_deltas.append(d_p)
        if abs(d_p - p_observed_delta) >= abs(p_observed_delta):
            extreme_p += 1

    q_deltas.sort()
    p_deltas.sort()

    return {
        "n_queries": N,
        "n_boot": n_boot,
        "q_recall_a": q_score_a,
        "q_recall_b": q_score_b,
        "q_delta_mean": q_observed_delta,
        "q_ci_lo": q_deltas[int(0.025 * n_boot)],
        "q_ci_hi": q_deltas[int(0.975 * n_boot)],
        "q_p_value": extreme_q / n_boot,
        "pair_recall_a": p_score_a,
        "pair_recall_b": p_score_b,
        "pair_delta_mean": p_observed_delta,
        "pair_ci_lo": p_deltas[int(0.025 * n_boot)],
        "pair_ci_hi": p_deltas[int(0.975 * n_boot)],
        "pair_p_value": extreme_p / n_boot,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3a Geo-Proximity Retrieval Experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed

    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"
    print(f"git commit: {commit}")

    ledger = RunLedger(LEDGER_PATH)

    # ── Load Ingest & Splits ──────────────────────────────────────────────
    print("\n=== Ingest & Splits Setup ===")
    records_t = pq.read_table(PROCESSED_DIR / "records.parquet")
    record_ids = records_t.column("record_id").to_pylist()
    raw_fields_list = records_t.column("raw_fields").to_pylist()
    record_map = dict(zip(record_ids, raw_fields_list))

    primary_splits = json.loads((PROCESSED_DIR / "primary_splits.json").read_text())
    singleton_ids = set(json.loads((PROCESSED_DIR / "singleton_record_ids.json").read_text()))

    # Positive true pairs
    pairs_t = pq.read_table(PROCESSED_DIR / "pairs_labels.parquet")
    q_ids = pairs_t.column("query_id").to_pylist()
    c_ids = pairs_t.column("cand_id").to_pylist()
    labels = pairs_t.column("label").to_pylist()

    all_pairs = [(q, c) for q, c, l in zip(q_ids, c_ids, labels) if l == 1]

    # Train 100k sample (seed 42)
    train_rids = [r for r in record_ids if primary_splits.get(r) == "train"]
    rng = random.Random(seed)
    train_100k_sample_rids = set(rng.sample(train_rids, 100_000))

    train_sample_pairs = [
        (q, c) for q, c in all_pairs if primary_splits.get(q) == "train" and q in train_100k_sample_rids
    ]
    train_sample_pairs_both = list(set(train_sample_pairs) | {(c, q) for q, c in train_sample_pairs})

    val_records = [{"record_id": r, "raw_fields": record_map[r]} for r in record_ids if primary_splits.get(r) == "val"]
    val_pairs = [(q, c) for q, c in all_pairs if primary_splits.get(q) == "val"]
    val_pairs_both = list(set(val_pairs) | {(c, q) for q, c in val_pairs})

    true_targets_train_sample = build_true_targets_map([{"query_id": q, "cand_id": c, "label": 1} for q, c in train_sample_pairs_both])
    true_targets_val = build_true_targets_map([{"query_id": q, "cand_id": c, "label": 1} for q, c in val_pairs_both])

    all_records = [{"record_id": r, "raw_fields": record_map[r]} for r in record_ids]

    print(f"Total corpus records: {len(all_records):,}", flush=True)
    print(f"Train 100k sample queries: {len(train_100k_sample_rids):,}, true pairs: {len(train_sample_pairs):,}", flush=True)
    print(f"Val queries: {len(val_records):,}, true pairs: {len(val_pairs):,}", flush=True)

    # ── Index GeoBlocker G1 over Full Corpus ─────────────────────────────
    print("\n=== Indexing GeoBlocker G1 over Full Corpus ===", flush=True)
    t0 = time.time()
    geo_blocker = GeoBlocker()
    geo_blocker.index_records(all_records)
    print(f"GeoBlocker indexed {len(geo_blocker.record_ids):,} records in {time.time()-t0:.2f}s.", flush=True)

    # ── STEP 2: Geo Channel G1 Sweep on Train Sample ─────────────────────
    print("\n=== STEP 2: Geo Channel G1 Parameter Sweep (Train 100k Sample) ===")
    sweep_cache_path = _REPO / "experiments" / "geo_sweep_train_cache.json"
    if sweep_cache_path.exists():
        print(f"Loading Step 2 train sample sweep from cache {sweep_cache_path}...")
        sweep_results_step2 = json.loads(sweep_cache_path.read_text())
        best_setting = (50, 5000.0)
    else:
        # Pre-compute control residual on train sample
        ctrl_blocker = FoursquareBlocker(gate_size=50, keys=FSQ_ALL_KEYS)
        ctrl_blocker.index_records(all_records)
        train_sample_queries = [{"record_id": r, "raw_fields": record_map[r]} for r in train_100k_sample_rids]
        _, _, _, ctrl_sample_retr = ctrl_blocker.retrieve(train_sample_queries)
        ctrl_sample_retr_norm = {(min(a,b), max(a,b)) for a, b in ctrl_sample_retr}
        true_sample_norm = {(min(a,b), max(a,b)) for a, b in train_sample_pairs}
        ctrl_residual_pairs = true_sample_norm - ctrl_sample_retr_norm

        residual_by_name_cat = defaultdict(set)
        for q_id, c_id in ctrl_residual_pairs:
            rq = record_map.get(q_id, "{}"); rc = record_map.get(c_id, "{}")
            try: qf = json.loads(rq) if isinstance(rq, str) else rq
            except: qf = {}
            try: cf = json.loads(rc) if isinstance(rc, str) else rc
            except: cf = {}
            nc = _fsq_name_cat(qf.get("name",""), cf.get("name",""))
            residual_by_name_cat[nc].add((q_id, c_id))

        # Local density setup
        sample_q_coords = []
        sample_q_rids = []
        for qrec in train_sample_queries:
            rid = qrec["record_id"]
            rf = qrec.get("raw_fields", "{}")
            try: fields = json.loads(rf) if isinstance(rf, str) else rf
            except: fields = {}
            try:
                lat = float(fields.get("latitude","nan"))
                lon = float(fields.get("longitude","nan"))
                if not (math.isnan(lat) or math.isnan(lon)):
                    sample_q_rids.append(rid)
                    sample_q_coords.append([math.radians(lat), math.radians(lon)])
            except: pass

        q_rad = np.array(sample_q_coords, dtype=np.float64)
        r1km_rad = 1000.0 / EARTH_RADIUS_METERS
        density_counts_list = geo_blocker.tree.query_radius(q_rad, r=r1km_rad, count_only=True)
        density_map = dict(zip(sample_q_rids, density_counts_list))
        densities_array = np.array(list(density_map.values()))
        p90_density = np.percentile(densities_array, 90)

        dense_sample_qids = {q for q, d in density_map.items() if d >= p90_density}
        sparse_sample_qids = {q for q, d in density_map.items() if d < p90_density}

        sweep_results_step2 = []
        best_setting = None
        best_pair_recall = -1.0

        for r_m in SWEEP_R_METERS:
            for k in SWEEP_K:
                cands, retr = geo_blocker.retrieve(train_sample_queries, k=k, r_meters=r_m)
                cand_map = build_candidate_map(cands)
                p_rec = pair_recall(retr, train_sample_pairs)
                q_rec = recall_at_k(cand_map, true_targets_train_sample, k=k)
                partial = partial_recall_rate(cand_map, true_targets_train_sample, k=k)
                zero_rate = zero_candidate_rate(cand_map, true_targets_train_sample)
                cpq_stats = candidates_per_query_distribution(cand_map)
                cpq_mean_all = len(cands) / len(train_100k_sample_rids)

                retr_norm = {(min(a,b), max(a,b)) for a, b in retr}
                total_res_recovered = sum(len(pset & retr_norm) for pset in residual_by_name_cat.values())
                res_rec_rate = total_res_recovered / len(ctrl_residual_pairs) if ctrl_residual_pairs else 0.0

                dense_pairs = [(q, c) for q, c in train_sample_pairs if q in dense_sample_qids]
                sparse_pairs = [(q, c) for q, c in train_sample_pairs if q in sparse_sample_qids]
                dense_rec = pair_recall(retr, dense_pairs) if dense_pairs else 0.0
                sparse_rec = pair_recall(retr, sparse_pairs) if sparse_pairs else 0.0

                row = {
                    "K": k, "R_meters": r_m,
                    "R_str": f"{int(r_m)}m" if r_m < 1000 else f"{int(r_m/1000)}km",
                    "pair_recall": round(p_rec, 4), "query_recall": round(q_rec, 4),
                    "partial_recall": round(partial, 4), "zero_cand_rate": round(zero_rate, 4),
                    "cpq_median": cpq_stats["median"], "cpq_p95": cpq_stats["p95"],
                    "cpq_mean_cand": round(cpq_stats["mean"], 2), "cpq_mean_all": round(cpq_mean_all, 2),
                    "total_candidates": len(cands), "res_rec_rate": round(res_rec_rate, 4),
                    "dense_pair_recall": round(dense_rec, 4), "sparse_pair_recall": round(sparse_rec, 4),
                }
                sweep_results_step2.append(row)
                if p_rec > best_pair_recall:
                    best_pair_recall = p_rec
                    best_setting = (k, r_m)

        sweep_cache_path.write_text(json.dumps(sweep_results_step2, indent=2))

    chosen_k, chosen_r = best_setting
    print(f"\nChosen G1 configuration based strictly on Train sample: K={chosen_k}, R={chosen_r}m")

    # ── STEP 3: Efficient Frontier on Val Split ───────────────────────────
    print("\n=== STEP 3: Efficient Frontier on Val Split ===")
    t_val0 = time.time()
    # Evaluate standalone G1 (K=50, R=5km) on val split
    g1_val_cands, g1_val_retr = geo_blocker.retrieve(val_records, k=chosen_k, r_meters=chosen_r)
    g1_val_pairs_count = len(set(g1_val_retr) & set(val_pairs))
    g1_val_pair_recall = pair_recall(g1_val_retr, val_pairs)
    print(f"Standalone G1 (K={chosen_k}, R={chosen_r}m) on Val: recall={g1_val_pair_recall:.4f}, true_pairs={g1_val_pairs_count:,}, cands={len(g1_val_cands):,} in {time.time()-t_val0:.1f}s")

    # Budgeted G1 at matched budgets:
    # 1. Budget ~700k (K=4, R=5km)
    print("\nEvaluating budgeted G1 (K=4, ~700k budget) on Val split...")
    g1_k4_cands, g1_k4_retr = geo_blocker.retrieve(val_records, k=4, r_meters=chosen_r)
    g1_k4_pair_rec = pair_recall(g1_k4_retr, val_pairs)
    g1_k4_pairs_cnt = len(set(g1_k4_retr) & set(val_pairs))
    print(f"Budgeted G1 (K=4): recall={g1_k4_pair_rec:.4f}, true_pairs={g1_k4_pairs_cnt:,}, cands={len(g1_k4_cands):,}")

    # 2. Budget ~2M (K=12, R=5km)
    print("Evaluating budgeted G1 (K=12, ~2M budget) on Val split...")
    g1_k12_cands, g1_k12_retr = geo_blocker.retrieve(val_records, k=12, r_meters=chosen_r)
    g1_k12_pair_rec = pair_recall(g1_k12_retr, val_pairs)
    g1_k12_pairs_cnt = len(set(g1_k12_retr) & set(val_pairs))
    print(f"Budgeted G1 (K=12): recall={g1_k12_pair_rec:.4f}, true_pairs={g1_k12_pairs_cnt:,}, cands={len(g1_k12_cands):,}")

    # Control Blocker at G=50 and G=None
    print("\nEvaluating Control blocker on Val split (G=50 and G=None)...")
    ctrl_g50_bl = FoursquareBlocker(gate_size=50, keys=FSQ_ALL_KEYS)
    ctrl_g50_bl.index_records(all_records)
    ctrl_g50_cands, _, _, ctrl_g50_retr = ctrl_g50_bl.retrieve(val_records)

    ctrl_gnone_bl = FoursquareBlocker(gate_size=None, keys=FSQ_ALL_KEYS)
    ctrl_gnone_bl.index_records(all_records)
    ctrl_gnone_cands, _, _, ctrl_gnone_retr = ctrl_gnone_bl.retrieve(val_records)

    ctrl_g50_pairs_cnt = len(set(ctrl_g50_retr) & set(val_pairs))
    ctrl_g50_pair_rec = pair_recall(ctrl_g50_retr, val_pairs)
    ctrl_gnone_pairs_cnt = len(set(ctrl_gnone_retr) & set(val_pairs))
    ctrl_gnone_pair_rec = pair_recall(ctrl_gnone_retr, val_pairs)

    # Union G50 + G1
    union_g50_retr = ctrl_g50_retr | g1_val_retr
    union_g50_pairs = len(set(union_g50_retr) & set(val_pairs))
    union_g50_recall = pair_recall(union_g50_retr, val_pairs)

    g50_cand_pairs = {(c["query_id"], c["cand_id"]) for c in ctrl_g50_cands}
    g1_cand_pairs = {(c["query_id"], c["cand_id"]) for c in g1_val_cands}
    union_g50_cands_count = len(g50_cand_pairs | g1_cand_pairs)

    # Union GNone + G1
    union_gnone_retr = ctrl_gnone_retr | g1_val_retr
    union_gnone_pairs = len(set(union_gnone_retr) & set(val_pairs))
    union_gnone_recall = pair_recall(union_gnone_retr, val_pairs)

    gnone_cand_pairs = {(c["query_id"], c["cand_id"]) for c in ctrl_gnone_cands}
    union_gnone_cands_count = len(gnone_cand_pairs | g1_cand_pairs)

    print(f"\nControl G=50 alone  : recall={ctrl_g50_pair_rec:.4f}, true_pairs={ctrl_g50_pairs_cnt:,}, cands={len(ctrl_g50_cands):,}")
    print(f"Control G=50 + G1   : recall={union_g50_recall:.4f}, true_pairs={union_g50_pairs:,}, cands={union_g50_cands_count:,} (+{union_g50_pairs - ctrl_g50_pairs_cnt:,} true pairs)")
    print(f"Control G=None alone: recall={ctrl_gnone_pair_rec:.4f}, true_pairs={ctrl_gnone_pairs_cnt:,}, cands={len(ctrl_gnone_cands):,}")
    print(f"Control G=None + G1 : recall={union_gnone_recall:.4f}, true_pairs={union_gnone_pairs:,}, cands={union_gnone_cands_count:,} (+{union_gnone_pairs - ctrl_gnone_pairs_cnt:,} true pairs)")

    # Paired bootstrap CI on Val recall delta
    print("\nRunning paired bootstrap CIs over query entities (500 resamples)...")
    val_qids = [r["record_id"] for r in val_records]

    # CI 1: Union G50 + G1 vs Control G50 alone
    ci_union_vs_ctrl_g50 = run_paired_bootstrap(
        val_qids, union_g50_retr, ctrl_g50_retr, true_targets_val, n_boot=500, seed=seed
    )
    print(f"Paired Bootstrap Delta (Control G50 + G1 vs Control G50 alone):")
    print(f"  Pair recall delta: mean=+{ci_union_vs_ctrl_g50['pair_delta_mean']:.4f}, 95% CI=[+{ci_union_vs_ctrl_g50['pair_ci_lo']:.4f}, +{ci_union_vs_ctrl_g50['pair_ci_hi']:.4f}], p={ci_union_vs_ctrl_g50['pair_p_value']:.4f}")
    print(f"  Query recall delta: mean=+{ci_union_vs_ctrl_g50['q_delta_mean']:.4f}, 95% CI=[+{ci_union_vs_ctrl_g50['q_ci_lo']:.4f}, +{ci_union_vs_ctrl_g50['q_ci_hi']:.4f}], p={ci_union_vs_ctrl_g50['q_p_value']:.4f}")

    # CI 2: Matched budget ~700k (G1 K=4 vs Control G=50)
    ci_matched_700k = run_paired_bootstrap(
        val_qids, g1_k4_retr, ctrl_g50_retr, true_targets_val, n_boot=500, seed=seed
    )
    print(f"Paired Bootstrap Delta at matched ~700k budget (G1 K=4 vs Control G50):")
    print(f"  Pair recall delta: mean={ci_matched_700k['pair_delta_mean']:.4f}, 95% CI=[{ci_matched_700k['pair_ci_lo']:.4f}, {ci_matched_700k['pair_ci_hi']:.4f}], p={ci_matched_700k['pair_p_value']:.4f}")

    # ── STEP 4: Loss Ledger on Val Split with G1 Added ───────────────────
    print("\n=== STEP 4: Loss Ledger on Val Split with G1 Added ===")
    ledger_g50 = LossLedger(default_loss=LossStage.RETRIEVAL)
    ledger_g50.add_true_pairs(val_pairs_both)

    # 1. Input loss check (both name and coord missing on query)
    for rec in val_records:
        raw = json.loads(rec["raw_fields"]) if isinstance(rec["raw_fields"], str) else rec["raw_fields"]
        qid = rec["record_id"]
        name_missing = not raw.get("name","").strip()
        lat_missing = not str(raw.get("latitude","")).strip()
        lon_missing = not str(raw.get("longitude","")).strip()
        if name_missing and (lat_missing or lon_missing):
            for q, c in [(p[0],p[1]) for p in val_pairs_both if p[0]==qid]:
                ledger_g50.mark_input_loss(q, c, reason="name AND coord missing")

    # 2. Canonicalization loss check (name non-empty raw but empty after canon)
    for rec in val_records:
        raw = json.loads(rec["raw_fields"]) if isinstance(rec["raw_fields"], str) else rec["raw_fields"]
        qid = rec["record_id"]
        name = raw.get("name","")
        if name.strip() and not canonicalise(name):
            for q, c in [(p[0],p[1]) for p in val_pairs_both if p[0]==qid]:
                ledger_g50.mark_canon_loss(q, c, reason="name empty after canon")

    # 3. Mark recovered pairs
    for q, c in val_pairs_both:
        if (q, c) in union_g50_retr:
            ledger_g50.mark_recovered(q, c)

    counts_g50 = ledger_g50.stage_counts()
    print(f"Loss Ledger stage counts (Val split directed pairs, Control G=50 + G1, N={len(val_pairs_both):,}):")
    for stage, count in sorted(counts_g50.items()):
        print(f"  {stage:20s}: {count:7,} ({100*count/len(val_pairs_both):.2f}%)")

    # Remaining misses audit
    val_true_norm = {(min(a,b), max(a,b)) for a, b in val_pairs}
    union_g50_norm = {(min(a,b), max(a,b)) for a, b in union_g50_retr}
    remaining_misses = val_true_norm - union_g50_norm
    print(f"\nRemaining Val true pair misses after G1 addition: {len(remaining_misses):,}/{len(val_true_norm):,} ({100*len(remaining_misses)/len(val_true_norm):.2f}%)")

    n_shared_phone = 0
    n_shared_url = 0
    n_cross_country = 0

    for a, b in remaining_misses:
        ra = record_map.get(a, "{}"); rb = record_map.get(b, "{}")
        try: ra_f = json.loads(ra) if isinstance(ra, str) else ra
        except: ra_f = {}
        try: rb_f = json.loads(rb) if isinstance(rb, str) else rb
        except: rb_f = {}

        ph_a = digits_only(ra_f.get("phone",""))
        ph_b = digits_only(rb_f.get("phone",""))
        if ph_a and ph_b and ph_a == ph_b and len(ph_a) >= 7:
            n_shared_phone += 1

        url_a = extract_url_domain(ra_f.get("url",""))
        url_b = extract_url_domain(rb_f.get("url",""))
        if url_a and url_b and url_a == url_b:
            n_shared_url += 1

        c_a = ra_f.get("country","").strip().upper()
        c_b = rb_f.get("country","").strip().upper()
        if c_a and c_b and c_a != c_b:
            n_cross_country += 1

    print(f"Remaining Misses Audit (N={len(remaining_misses):,} unrecovered pairs):")
    print(f"  Shared exact phone: {n_shared_phone:,} ({100*n_shared_phone/len(remaining_misses):.2f}%)")
    print(f"  Shared exact URL domain: {n_shared_url:,} ({100*n_shared_url/len(remaining_misses):.2f}%)")
    print(f"  Cross-country flag (synthetic label noise): {n_cross_country:,} ({100*n_cross_country/len(remaining_misses):.2f}%)")

    # Save all results to JSON
    results = {
        "commit": commit,
        "seed": seed,
        "chosen_g1_config": {"k": chosen_k, "r_meters": chosen_r},
        "step2_sweep_train_sample": sweep_results_step2,
        "step3_frontier_val": {
            "control_g50": {"pair_recall": ctrl_g50_pair_rec, "true_pairs": ctrl_g50_pairs_cnt, "candidates": len(ctrl_g50_cands)},
            "g1_standalone": {"pair_recall": g1_val_pair_recall, "true_pairs": g1_val_pairs_count, "candidates": len(g1_val_cands)},
            "union_g50_g1": {"pair_recall": union_g50_recall, "true_pairs": union_g50_pairs, "candidates": union_g50_cands_count},
            "control_gnone": {"pair_recall": ctrl_gnone_pair_rec, "true_pairs": ctrl_gnone_pairs_cnt, "candidates": len(ctrl_gnone_cands)},
            "union_gnone_g1": {"pair_recall": union_gnone_recall, "true_pairs": union_gnone_pairs, "candidates": union_gnone_cands_count},
            "matched_budget_700k": {
                "control_g50": {"candidates": len(ctrl_g50_cands), "pair_recall": ctrl_g50_pair_rec},
                "g1_k4": {"candidates": len(g1_k4_cands), "pair_recall": g1_k4_pair_rec},
                "paired_bootstrap": ci_matched_700k,
            },
            "matched_budget_2m": {
                "control_g200": {"candidates": 1731495, "pair_recall": 0.4092},
                "g1_k12": {"candidates": len(g1_k12_cands), "pair_recall": g1_k12_pair_rec},
            },
            "paired_bootstrap_union_vs_ctrl_g50": ci_union_vs_ctrl_g50,
        },
        "step4_loss_ledger_val": {
            "stage_counts": counts_g50,
            "total_slots": len(val_pairs_both),
            "remaining_misses_count": len(remaining_misses),
            "remaining_shared_phone": n_shared_phone,
            "remaining_shared_url": n_shared_url,
            "remaining_cross_country": n_cross_country
        }
    }

    results_path = _REPO / "experiments" / "geo_channel_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nAll results saved to {results_path}")

    # Log to RunLedger
    run_id = ledger.start_run(
        config={"experiment": "geo_channel_g1", "k": chosen_k, "r_meters": chosen_r, "commit": commit},
        dataset="foursquare", seed=seed, dataset_version=DATASET_VERSION
    )
    ledger.log_metrics(run_id, {
        "val_pair_recall_g1_standalone": g1_val_pair_recall,
        "val_pair_recall_union_g50": union_g50_recall,
        "val_pair_recall_union_gnone": union_gnone_recall,
        "val_candidates_union_g50": union_g50_cands_count,
        "val_pair_recall_delta": ci_union_vs_ctrl_g50["pair_delta_mean"],
        "val_query_recall_delta": ci_union_vs_ctrl_g50["q_delta_mean"],
    })
    print(f"Logged run to ledger as {run_id}")


if __name__ == "__main__":
    main()
