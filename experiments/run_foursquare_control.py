"""
experiments/run_foursquare_control.py — Foursquare Control Baseline
====================================================================
Runs the full Part B pipeline:
  B1  Data verification (inline; doc written separately)
  B2  Ingest via FoursquareAdapter (cached)
  B3  Entity-disjoint splits by POI
  B4  Canon stats (NFKD->ASCII pipeline on pre-processed text)
  B5  Control blocker sweep G in {10,50,200,1000,none}
  B6  Metrics (pair recall, query recall, partial recall, cpq, singleton)
  B7  Efficient frontier
  B8  Loss ledger at G=50 and largest feasible G
  B9  Residual diagnosis on TRAIN queries (cross-tab name x location)

Usage:
    python experiments/run_foursquare_control.py [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pyarrow.parquet as pq

from ber.canon.normalise import canonicalise, first_token, first_n_chars, outward_code
from ber.eval.ledger import RunLedger
from ber.eval.loss_ledger import LossLedger, LossStage
from ber.eval.metrics import (
    bootstrap_ci, build_candidate_map, build_true_targets_map,
    candidates_per_query_distribution, partial_recall_rate,
    recall_at_k, zero_candidate_rate,
)
from ber.eval.metrics_dedup import pair_recall, singleton_burden
from ber.retrieve.control_blocking import BlockingConfig, ControlBlocker
from ber.store.schemas import write_candidates
from ber_datasets.adapters.foursquare import (
    FoursquareAdapter, assert_no_poi_leakage,
    extract_url_domain, digits_only, geo_cell,
)

# Local aliases matching original private names used in this file
_extract_url_domain = extract_url_domain
_digits_only = digits_only
_geo_cell = geo_cell
from ber_datasets.splits.splitter import EntityDisjointSplitter

PROCESSED_DIR = _REPO / "data" / "processed" / "foursquare"
FSQ_DATA_DIR = (_REPO / "ber" / "data" / "raw" / "foursquare_kaggle"
                / "preprocessing_data")
LEDGER_PATH = _REPO / "experiments" / "runs.parquet"
DATASET_VERSION = "foursquare_kaggle_train_2022"
GATE_SWEEP = [10, 50, 200, 1000, None]
PAIR_COUNT_LIMIT = 500_000_000  # abort estimate if above this


# ── helpers ────────────────────────────────────────────────────────────────

def _pairs(path: Path) -> list[tuple[str, str]]:
    t = pq.read_table(path, columns=["query_id", "cand_id", "label"])
    return [(q, c) for q, c, l in zip(
        t.column("query_id").to_pylist(),
        t.column("cand_id").to_pylist(),
        t.column("label").to_pylist()) if l == 1]

def _f(rec: dict) -> dict:
    rf = rec.get("raw_fields", "{}")
    if isinstance(rf, str):
        try: return json.loads(rf)
        except: return {}
    return rf or {}

def _pct(d, p):
    if not d: return float("nan")
    idx = (len(d)-1)*p/100
    lo, hi = int(idx), min(int(idx)+1, len(d)-1)
    return d[lo]*(1-(idx-lo)) + d[hi]*(idx-lo)

# ── Foursquare blocking keys ───────────────────────────────────────────────

# All 5 Foursquare keys operate on model-visible fields only.
# K1 name+country, K2 geo cell, K3 phone digits, K4 url domain, K5 name3+zip

def _extract_url_domain(url: str) -> str:
    """Extract registrable domain from a URL string."""
    if not url:
        return ""
    url = url.lower().strip()
    # Strip protocol
    for prefix in ("https://", "http://", "//"):
        if url.startswith(prefix):
            url = url[len(prefix):]
    # Take up to first /
    url = url.split("/")[0].split("?")[0]
    # Remove www.
    if url.startswith("www."):
        url = url[4:]
    # Keep only if it looks like a domain (has a dot)
    return url if "." in url else ""

def _digits_only(s: str) -> str:
    """Keep only digit characters."""
    return "".join(c for c in s if c.isdigit())

def _geo_cell(lat: str, lon: str) -> str:
    """Round lat/lon to 2 decimal places → ~1km cell."""
    try:
        la = round(float(lat), 2)
        lo = round(float(lon), 2)
        return f"{la:.2f}|{lo:.2f}"
    except Exception:
        return ""

def _fsq_record_keys(raw: dict) -> dict[str, str]:
    """Compute all 5 Foursquare blocking key values."""
    name = raw.get("name", "")
    country = raw.get("country", "").strip().upper()
    lat = raw.get("latitude", "")
    lon = raw.get("longitude", "")
    phone = raw.get("phone", "")
    url = raw.get("url", "")
    zipcode = raw.get("zip", "").strip()

    canon_name = canonicalise(name)
    tok = first_token(name)
    f3 = first_n_chars(name, 3)
    digits_phone = _digits_only(phone)
    domain = _extract_url_domain(url)
    cell = _geo_cell(lat, lon)

    return {
        "K1": f"{canon_name}|{country}" if canon_name and country else "",
        "K2": f"{tok}|{cell}" if tok and cell else "",
        "K3": digits_phone if len(digits_phone) >= 7 else "",  # require min 7 digits
        "K4": domain,
        "K5": f"{f3}|{canonicalise(zipcode)}" if f3 and zipcode.strip() else "",
    }

FSQ_ALL_KEYS = ["K1", "K2", "K3", "K4", "K5"]


class FoursquareBlocker:
    """Exact-key blocker for Foursquare dedup.

    Uses the same ControlBlocker infrastructure but with Foursquare-specific
    key extraction and a feasibility check before running.
    """

    def __init__(self, gate_size: int | None = 50,
                 keys: list[str] | None = None) -> None:
        self.gate_size = gate_size
        self.keys = keys or list(FSQ_ALL_KEYS)
        # key → bucket_value → [record_ids]
        self._index: dict[str, dict[str, list[str]]] = {
            k: defaultdict(list) for k in self.keys
        }

    def index_records(self, records: list[dict]) -> None:
        """Index all records (both query and target side — same table)."""
        for rec in records:
            rid = rec["record_id"]
            raw = _f(rec)
            keys = _fsq_record_keys(raw)
            for k in self.keys:
                val = keys.get(k, "")
                if val:
                    self._index[k][val].append(rid)

    def estimate_pair_count(self) -> dict[str, Any]:
        """Estimate total candidate pairs as sum_b b*(b-1)/2 for each key."""
        total = 0
        per_key = {}
        for k in self.keys:
            key_total = sum(
                n * (n - 1) // 2
                for n in (len(v) for v in self._index[k].values())
                if (self.gate_size is None or n <= self.gate_size)
            )
            per_key[k] = key_total
            total += key_total  # union is ≤ sum; this is upper bound
        return {"estimated_total_upper_bound": total, "per_key": per_key,
                "feasible": total <= PAIR_COUNT_LIMIT}

    def with_gate(self, gate_size: int | None) -> "FoursquareBlocker":
        sibling = FoursquareBlocker(gate_size=gate_size, keys=list(self.keys))
        sibling._index = self._index
        return sibling

    def retrieve(self, query_records: list[dict]) -> tuple[list[dict], dict, set, set]:
        """Retrieve candidates.  Excludes self-pairs (query == cand)."""
        pair_mask: dict[tuple[str, str], int] = {}
        size_gated: set[tuple[str, str]] = set()
        key_stats = {k: {"n_pairs_before": 0, "n_gated": 0,
                          "n_contributed": 0} for k in self.keys}
        key_pairs: dict[str, set] = {k: set() for k in self.keys}

        for rec in query_records:
            qid = rec["record_id"]
            raw = _f(rec)
            q_keys = _fsq_record_keys(raw)

            for k in self.keys:
                bit = (1 << FSQ_ALL_KEYS.index(k))
                val = q_keys.get(k, "")
                if not val:
                    continue
                bucket = self._index[k].get(val, [])
                # Exclude self
                bucket_filtered = [tid for tid in bucket if tid != qid]
                if not bucket_filtered:
                    continue

                key_stats[k]["n_pairs_before"] += len(bucket_filtered)

                if self.gate_size is not None and len(bucket_filtered) > self.gate_size:
                    key_stats[k]["n_gated"] += len(bucket_filtered)
                    for tid in bucket_filtered:
                        size_gated.add((qid, tid))
                    continue

                for tid in bucket_filtered:
                    pair = (qid, tid)
                    pair_mask[pair] = pair_mask.get(pair, 0) | bit
                    key_stats[k]["n_contributed"] += 1
                    key_pairs[k].add(pair)

        actual_gated = size_gated - set(pair_mask.keys())

        # Per-key unique candidate pairs
        per_key_unique = {}
        for k in self.keys:
            other = set()
            for k2 in self.keys:
                if k2 != k:
                    other |= key_pairs[k2]
            per_key_unique[k] = len(key_pairs[k] - other)

        # Build candidate rows (sorted by query_id, cand_id)
        query_cands: dict[str, list] = defaultdict(list)
        for (qid, tid), mask in pair_mask.items():
            query_cands[qid].append((tid, mask))

        candidates = []
        for qid, cands in query_cands.items():
            for rank, (tid, mask) in enumerate(sorted(cands), start=1):
                candidates.append({
                    "query_id": qid, "cand_id": tid,
                    "channel_mask": mask,
                    "score": float(bin(mask).count("1")),
                    "rank": rank, "reverse_rank": 0, "round": 1,
                    "dataset_id": "foursquare",
                })

        stats = {
            "gate_size": self.gate_size,
            "n_queries": len(query_records),
            "n_candidate_pairs": len(pair_mask),
            "n_size_gated_pairs": len(actual_gated),
            "per_key": {k: {**key_stats[k],
                             "n_unique_cand_pairs": per_key_unique[k],
                             "n_pairs_in_key": len(key_pairs[k])}
                        for k in self.keys},
        }
        return candidates, stats, actual_gated, set(pair_mask.keys())


# ── B9 residual taxonomy ──────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(min(1, math.sqrt(a)))

_FSQ_STOP = {"the", "of", "and", "a", "an", "for", "in", "at", "to", "&"}

def _fsq_name_cat(q_name, t_name):
    qc = canonicalise(q_name); tc = canonicalise(t_name)
    if not qc or not tc:
        return "name_empty_after_canon"
    if qc == tc:
        return "identical"
    q_toks = set(qc.split()); t_toks = set(tc.split())
    if q_toks == t_toks:
        return "token_reorder"
    if (q_toks - _FSQ_STOP) == (t_toks - _FSQ_STOP) and (q_toks - _FSQ_STOP):
        return "the_stopword_only"
    u = q_toks | t_toks
    jacc = len(q_toks & t_toks) / len(u) if u else 0.0
    return "partial_overlap" if jacc >= 0.5 else "different"

def _fsq_loc_cat(q_raw, t_raw):
    try:
        q_lat, q_lon = float(q_raw.get("latitude","")), float(q_raw.get("longitude",""))
        t_lat, t_lon = float(t_raw.get("latitude","")), float(t_raw.get("longitude",""))
        dist_km = _haversine_km(q_lat, q_lon, t_lat, t_lon)
    except Exception:
        dist_km = None

    q_country = q_raw.get("country","").strip().upper()
    t_country = t_raw.get("country","").strip().upper()
    same_country = (q_country == t_country and q_country != "")

    q_phone = _digits_only(q_raw.get("phone",""))
    t_phone = _digits_only(t_raw.get("phone",""))
    q_domain = _extract_url_domain(q_raw.get("url",""))
    t_domain = _extract_url_domain(t_raw.get("url",""))
    same_phone = bool(q_phone and t_phone and q_phone == t_phone and len(q_phone) >= 7)
    same_url = bool(q_domain and t_domain and q_domain == t_domain)

    if dist_km is None:
        dist_cat = "unknown"
    elif dist_km < 0.1:
        dist_cat = "<100m"
    elif dist_km < 1.0:
        dist_cat = "<1km"
    elif dist_km < 10.0:
        dist_cat = "<10km"
    else:
        dist_cat = ">=10km"

    loc_cat = dist_cat if same_country else f"diff_country_{dist_cat}"
    return loc_cat, same_phone, same_url, dist_km


# ── Main pipeline ──────────────────────────────────────────────────────────

def main(seed: int = 42) -> dict:
    import subprocess
    try:
        commit = subprocess.run(
            ["git","rev-parse","HEAD"], capture_output=True, text=True,
            cwd=str(_REPO), timeout=5).stdout.strip()
    except Exception:
        commit = "untracked"
    print(f"git commit: {commit}", flush=True)

    ledger = RunLedger(LEDGER_PATH)

    # ── B2: Ingest ────────────────────────────────────────────────────────
    print("\n=== B2: Ingest ===", flush=True)
    records_path = PROCESSED_DIR / "records.parquet"
    if records_path.exists():
        print("  Using cached ingest.", flush=True)
        ingest_stats = json.loads((PROCESSED_DIR / "ingest_stats.json").read_text())
    else:
        adapter = FoursquareAdapter(FSQ_DATA_DIR)
        ingest_stats = adapter.build(PROCESSED_DIR)

    # Verify no POI leakage
    assert_no_poi_leakage(PROCESSED_DIR / "records.parquet")
    print(f"  Leakage check: PASSED", flush=True)
    print(f"  {ingest_stats}", flush=True)

    # ── B3: Splits ────────────────────────────────────────────────────────
    print("\n=== B3: Splits ===", flush=True)
    primary_path = PROCESSED_DIR / "primary_splits.json"
    brand_path = PROCESSED_DIR / "brand_splits.json"

    all_pairs = _pairs(PROCESSED_DIR / "pairs_labels.parquet")

    t = pq.read_table(PROCESSED_DIR / "records.parquet",
                      columns=["record_id", "raw_fields"])
    record_ids = t.column("record_id").to_pylist()
    raw_fields_list = t.column("raw_fields").to_pylist()
    record_map = {rid: rf for rid, rf in zip(record_ids, raw_fields_list)}

    if primary_path.exists():
        print("  Using cached splits.", flush=True)
        primary_splits = json.loads(primary_path.read_text())
        brand_splits = json.loads(brand_path.read_text())
    else:
        print("  Building entity-disjoint splits by POI...", flush=True)
        labels_t = pq.read_table(PROCESSED_DIR / "labels.parquet",
                                  columns=["record_id", "entity_id"])
        entity_map = {r: e for r, e in zip(
            labels_t.column("record_id").to_pylist(),
            labels_t.column("entity_id").to_pylist())}
        entity_ids_list = [entity_map.get(r, r) for r in record_ids]

        # norm_keys for dup-guard
        norm_keys = []
        norm_names = []
        for rf in raw_fields_list:
            try: fields = json.loads(rf)
            except: fields = {}
            nc = canonicalise(fields.get("name",""))
            city = canonicalise(fields.get("city",""))
            norm_keys.append((nc, city))
            norm_names.append(nc)

        splitter = EntityDisjointSplitter(
            ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=seed)
        primary_splits = splitter.split(record_ids, entity_ids_list, norm_keys=norm_keys)
        brand_splits = splitter.brand_split(record_ids, norm_names, entity_ids_list)
        primary_path.write_text(json.dumps(primary_splits))
        brand_path.write_text(json.dumps(brand_splits))

    split_counts = Counter(primary_splits.values())
    brand_counts = Counter(brand_splits.values())

    # Leakage check
    labels_t = pq.read_table(PROCESSED_DIR / "labels.parquet",
                              columns=["record_id", "entity_id"])
    entity_map = {r: e for r, e in zip(
        labels_t.column("record_id").to_pylist(),
        labels_t.column("entity_id").to_pylist())}
    entity_to_splits: dict[str, set] = defaultdict(set)
    for rid in record_ids:
        eid = entity_map.get(rid, rid)
        entity_to_splits[eid].add(primary_splits.get(rid, "?"))
    n_leaking = sum(1 for s in entity_to_splits.values() if len(s) > 1)
    assert n_leaking == 0, f"Split leakage: {n_leaking}"
    print(f"  Primary: {dict(split_counts)}", flush=True)
    print(f"  Brand:   {dict(brand_counts)}", flush=True)
    print(f"  Leakage: 0 (PASS)", flush=True)

    # ── B4: Canon stats ───────────────────────────────────────────────────
    print("\n=== B4: Canon stats ===", flush=True)
    name_empty = 0
    name_total = 0
    country_empty: Counter = Counter()
    country_total: Counter = Counter()
    for rf in raw_fields_list:
        try: fields = json.loads(rf)
        except: fields = {}
        name = fields.get("name","")
        country = fields.get("country","")
        name_total += 1
        country_total[country] += 1
        if name.strip() and not canonicalise(name):
            name_empty += 1
            country_empty[country] += 1
    print(f"  name empty after canon: {name_empty} / {name_total} = {100*name_empty/name_total:.2f}%")

    # ── B5: Build index over FULL corpus ─────────────────────────────────
    print("\n=== B5: Build index (full corpus) ===", flush=True)
    print("  Index built from ALL records (full corpus) — not split-restricted.", flush=True)
    print("  This ensures bucket sizes reflect true corpus density (no leakage:", flush=True)
    print("  no labels are used during index construction).", flush=True)

    all_records = [{"record_id": rid, "raw_fields": rf}
                   for rid, rf in zip(record_ids, raw_fields_list)]

    base_blocker = FoursquareBlocker(gate_size=None, keys=FSQ_ALL_KEYS)
    base_blocker.index_records(all_records)
    print(f"  Index built for {len(all_records):,} records.", flush=True)

    # Singletons
    singleton_ids = set(json.loads(
        (PROCESSED_DIR / "singleton_record_ids.json").read_text()))

    # ── Get val and train queries ──────────────────────────────────────────
    val_records = [{"record_id": r, "raw_fields": record_map[r]}
                   for r in record_ids if primary_splits.get(r) == "val"]
    train_records = [{"record_id": r, "raw_fields": record_map[r]}
                     for r in record_ids if primary_splits.get(r) == "train"]

    # Val true pairs (query in val split)
    val_pair_set = {(q, c) for q, c in all_pairs if primary_splits.get(q) == "val"}
    val_true_pairs = list(val_pair_set)
    # Also need reverse for pair recall
    val_true_pairs_both = list(val_pair_set | {(c, q) for q, c in val_pair_set})

    # Train true pairs (for residual diagnosis in B9)
    train_pair_set = {(q, c) for q, c in all_pairs if primary_splits.get(q) == "train"}

    true_targets_val = build_true_targets_map(
        [{"query_id": q, "cand_id": c, "label": 1} for q, c in val_true_pairs_both])

    print(f"  Val records: {len(val_records):,}, val true pairs: {len(val_pair_set):,}", flush=True)
    print(f"  Train records: {len(train_records):,}, train true pairs: {len(train_pair_set):,}", flush=True)

    # ── G-sweep on val ───────────────────────────────────────────────────
    print("\n=== B5-B6: G-sweep (val split) ===", flush=True)
    sweep_run_id = ledger.start_run(
        config={"experiment": "foursquare_control_sweep",
                "keys": FSQ_ALL_KEYS, "dataset_version": DATASET_VERSION,
                "seed": seed, "commit": commit,
                "note": "Index built from full corpus; eval on val split only"},
        dataset="foursquare", seed=seed, dataset_version=DATASET_VERSION,
    )
    sweep_dir = _REPO / "experiments" / sweep_run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for g in GATE_SWEEP:
        # Feasibility check before running
        est = base_blocker.with_gate(g).estimate_pair_count()
        if not est["feasible"]:
            print(f"  G={str(g):<5} INFEASIBLE: estimated {est['estimated_total_upper_bound']:,} pairs > {PAIR_COUNT_LIMIT:,}", flush=True)
            sweep_results.append({"gate_size": g, "feasible": False,
                                   "estimated_pairs": est["estimated_total_upper_bound"]})
            continue

        bl = base_blocker.with_gate(g)
        cands, stats, gated, retrieved = bl.retrieve(val_records)

        cand_map = build_candidate_map(cands)
        max_rank = max((c["rank"] for c in cands), default=0)
        k_b = max(max_rank, 1)

        q_recall = recall_at_k(cand_map, true_targets_val, k=k_b)
        p_recall = pair_recall(retrieved, val_true_pairs)
        partial = partial_recall_rate(cand_map, true_targets_val, k=k_b)
        z_rate = zero_candidate_rate(cand_map, true_targets_val)
        cpq = candidates_per_query_distribution(cand_map)
        sb = singleton_burden(cand_map, singleton_ids & {r["record_id"] for r in val_records})

        fn = functools.partial(recall_at_k, k=k_b)
        pt, lo, hi = bootstrap_ci(cand_map, true_targets_val, fn, n_boot=500, seed=seed)

        # cpq over all val queries (including zero-cand)
        n_val_q = len(val_records)
        mean_cpq_all = len(cands) / max(n_val_q, 1)

        print(f"  G={str(g):<5} q_recall={q_recall:.4f} [{lo:.4f},{hi:.4f}]"
              f" pair_recall={p_recall:.4f} cands={len(cands):,}"
              f" zero={z_rate:.4f} cpq_med={cpq['median']:.1f}", flush=True)

        if cands:
            write_candidates(cands, sweep_dir / f"candidates_G{g}.parquet")

        sweep_results.append({
            "gate_size": g, "feasible": True,
            "n_candidates": len(cands),
            "query_recall": round(q_recall, 6),
            "pair_recall": round(p_recall, 6),
            "partial_recall_rate": round(partial, 6) if partial == partial else None,
            "zero_candidate_rate": round(z_rate, 6),
            "recall_ci_lo": round(lo, 6), "recall_ci_hi": round(hi, 6),
            "cpq_median": cpq["median"], "cpq_p95": cpq["p95"], "cpq_mean": cpq["mean"],
            "cpq_mean_all_val_queries": round(mean_cpq_all, 2),
            "n_val_queries": n_val_q,
            "n_queries_with_targets": len(true_targets_val),
            "singleton_burden": sb,
            "per_key_stats": stats["per_key"],
        })

    ledger.log_metrics(sweep_run_id, {
        "sweep_results": sweep_results,
        "ingest_stats": ingest_stats,
        "split_stats": {"primary": dict(split_counts), "brand": dict(brand_counts)},
        "canon_stats": {"name_empty_after_canon": name_empty, "name_total": name_total},
    })

    # ── B7: Efficient frontier ────────────────────────────────────────────
    print("\n=== B7: Efficient frontier ===", flush=True)
    true_set = {(min(a,b), max(a,b)) for a, b in val_true_pairs}
    n_pairs = len(true_set)

    frontier = []
    for g in [50, None]:
        selected = []
        current_retrieved_norm: set = set()
        current_cands = 0
        remaining = list(FSQ_ALL_KEYS)

        while remaining:
            best_k = None; best_marg_true = -1; best_marg_cands = 0
            best_retr = None; best_cands = 0

            for k in remaining:
                test_keys = selected + [k]
                bl = FoursquareBlocker(gate_size=g, keys=test_keys)
                bl._index = {k2: base_blocker._index.get(k2, {}) for k2 in test_keys}
                cands_t, _, _, retr_t = bl.retrieve(val_records)
                retr_norm = {(min(a,b),max(a,b)) for a,b in retr_t}
                marg = len((retr_norm & true_set) - current_retrieved_norm)
                if marg > best_marg_true:
                    best_k = k; best_marg_true = marg
                    best_retr = retr_norm; best_cands = len(cands_t)

            selected.append(best_k)
            remaining.remove(best_k)
            current_retrieved_norm = best_retr & true_set
            current_cands = best_cands
            cum_pairs = len(current_retrieved_norm)

            frontier.append({
                "G": g, "key_added": best_k,
                "keys_active": list(selected),
                "cumulative_pair_recall": round(cum_pairs / max(n_pairs, 1), 6),
                "cumulative_pairs": cum_pairs,
                "cumulative_candidates": current_cands,
                "marginal_pairs": best_marg_true,
            })
            print(f"  G={str(g):<5} +{best_k}: pairs={cum_pairs}/{n_pairs} "
                  f"recall={cum_pairs/max(n_pairs,1):.4f} cands={current_cands:,}", flush=True)

    # ── B8: Loss ledger at G=50 ───────────────────────────────────────────
    print("\n=== B8: Loss ledger G=50 ===", flush=True)
    _run_loss_ledger(base_blocker, val_records, val_true_pairs_both,
                     gate_size=50, sweep_dir=sweep_dir, suffix="G50")

    # Find largest feasible G for loss ledger
    largest_feasible = max(
        (r["gate_size"] for r in sweep_results if r.get("feasible")),
        key=lambda g: g if g is not None else float("inf"),
        default=None,
    )
    if largest_feasible != 50:
        print(f"\n=== B8: Loss ledger G={largest_feasible} ===", flush=True)
        _run_loss_ledger(base_blocker, val_records, val_true_pairs_both,
                         gate_size=largest_feasible, sweep_dir=sweep_dir,
                         suffix=f"G{largest_feasible}")

    # ── B9: Residual on TRAIN split ───────────────────────────────────────
    print("\n=== B9: Residual taxonomy (train split, G=50) ===", flush=True)
    print("  Note: using G=50 for train retrieval to avoid OOM (800k queries x 1.1M corpus).", flush=True)
    tax = _residual_taxonomy(base_blocker, train_records, train_pair_set,
                              record_map, seed=seed, gate_for_residual=50)
    print(f"  Train unrecovered: {tax['n_unrecovered']}", flush=True)
    for cat, cnt in sorted(tax["category_counts"].items()):
        print(f"    {cat}: {cnt}", flush=True)

    tax_run_id = ledger.start_run(
        config={"experiment": "foursquare_residual_train", "commit": commit, "seed": seed},
        dataset="foursquare", seed=seed, dataset_version=DATASET_VERSION,
    )
    ledger.log_metrics(tax_run_id, {k: v for k, v in tax.items() if k != "examples"})

    # Save all results
    out_path = _REPO / "experiments" / "foursquare_results.json"
    out_path.write_text(json.dumps({
        "commit": commit, "ingest_stats": ingest_stats,
        "split_stats": {"primary": dict(split_counts)},
        "canon_stats": {"name_empty": name_empty, "name_total": name_total},
        "sweep_results": sweep_results,
        "frontier": frontier,
        "taxonomy_train": {k: v for k, v in tax.items() if k != "examples"},
        "taxonomy_examples": {k: v for k, v in tax["examples"].items()},
    }, indent=2, default=str))
    print(f"\nResults saved to {out_path}", flush=True)

    return {
        "sweep_results": sweep_results, "frontier": frontier,
        "taxonomy": tax, "ingest_stats": ingest_stats,
    }


def _run_loss_ledger(base_blocker, query_records, true_pairs_both, gate_size, sweep_dir, suffix):
    bl = base_blocker.with_gate(gate_size)
    cands, _, gated, retrieved = bl.retrieve(query_records)
    ll = LossLedger(default_loss=LossStage.RETRIEVAL)
    ll.add_true_pairs(true_pairs_both)

    # INPUT: both name and coordinates missing
    for rec in query_records:
        raw = _f(rec)
        qid = rec["record_id"]
        name_missing = not raw.get("name","").strip()
        lat_missing = not raw.get("latitude","").strip()
        lon_missing = not raw.get("longitude","").strip()
        if name_missing and (lat_missing or lon_missing):
            for q, c in [(p[0],p[1]) for p in true_pairs_both if p[0]==qid]:
                ll.mark_input_loss(q, c, reason="name AND coord missing")

    # CANONICALIZATION: name non-empty raw but empty after canon
    for rec in query_records:
        raw = _f(rec)
        qid = rec["record_id"]
        name = raw.get("name","")
        if name.strip() and not canonicalise(name):
            for q, c in [(p[0],p[1]) for p in true_pairs_both if p[0]==qid]:
                ll.mark_canon_loss(q, c, reason="name empty after canon")

    ll.apply_candidates(retrieved)
    for q, c in true_pairs_both:
        if (q, c) in gated:
            rec = ll._pairs.get((q, c))
            if rec and rec.loss_stage == LossStage.RETRIEVAL:
                ll.mark_retrieval_loss(q, c, subtype="size_gated",
                                       reason=f"bucket > G={gate_size}")
    for q, c in true_pairs_both:
        if (q, c) in retrieved:
            ll.mark_recovered(q, c)

    ll.to_csv(sweep_dir / f"loss_ledger_{suffix}.csv")
    stage_counts = ll.stage_counts()
    print(f"  Loss ledger {suffix}: {stage_counts}", flush=True)
    return stage_counts


def _residual_taxonomy(base_blocker, train_records, train_pair_set,
                        record_map, seed=42, gate_for_residual=50):
    """Run residual taxonomy on train-split queries.

    Uses gate_for_residual (default 50) instead of G=None to avoid OOM.
    At G=None the candidate set can reach ~40M pairs for 800k train queries
    against the full 1.1M-record corpus, exceeding 16 GB RAM.
    The residual is defined as true pairs NOT recovered at gate_for_residual.
    """
    bl = base_blocker.with_gate(gate_for_residual)

    # Stream in batches of 10k to limit peak RAM
    BATCH = 10_000
    retrieved_norm: set = set()
    train_list = list(train_records)
    for i in range(0, len(train_list), BATCH):
        batch = train_list[i:i + BATCH]
        _, _, _, retr = bl.retrieve(batch)
        retrieved_norm.update((min(a,b),max(a,b)) for a,b in retr)
        if i % 100_000 == 0 and i > 0:
            print(f"  B9 retrieved {i:,}/{len(train_list):,} train queries...", flush=True)

    true_norm = {(min(a,b),max(a,b)) for a,b in train_pair_set}
    unrecovered_norm = true_norm - retrieved_norm

    cats = defaultdict(list)
    n_shared_phone = 0; n_shared_url = 0
    dist_kms = []

    for a, b in unrecovered_norm:
        ra = record_map.get(a, "{}"); rb = record_map.get(b, "{}")
        try: ra_f = json.loads(ra) if isinstance(ra, str) else ra
        except: ra_f = {}
        try: rb_f = json.loads(rb) if isinstance(rb, str) else rb
        except: rb_f = {}

        name_a = ra_f.get("name",""); name_b = rb_f.get("name","")
        name_cat = _fsq_name_cat(name_a, name_b)
        loc_cat, same_ph, same_url, dist_km = _fsq_loc_cat(ra_f, rb_f)
        if same_ph: n_shared_phone += 1
        if same_url: n_shared_url += 1
        if dist_km is not None: dist_kms.append(dist_km)
        cats[(name_cat, loc_cat)].append(
            (name_a, name_b, ra_f.get("country",""), rb_f.get("country",""),
             ra_f.get("latitude",""), ra_f.get("longitude",""),
             rb_f.get("latitude",""), rb_f.get("longitude",""))
        )

    rng = random.Random(seed)
    examples = {}
    for (nc, lc), items in cats.items():
        sample = rng.sample(items, min(5, len(items)))
        examples[f"{nc}|{lc}"] = [
            {"name_q": i[0], "name_t": i[1],
             "country_q": i[2], "country_t": i[3],
             "lat_q": i[4], "lon_q": i[5], "lat_t": i[6], "lon_t": i[7]}
            for i in sample
        ]

    def avg(lst): return round(sum(lst)/len(lst),3) if lst else float("nan")
    def med(lst):
        if not lst: return float("nan")
        s=sorted(lst); m=len(s)//2
        return round(s[m] if len(s)%2 else (s[m-1]+s[m])/2,3)

    return {
        "gate_for_residual": gate_for_residual,
        "n_unrecovered": len(unrecovered_norm),
        "n_shared_phone": n_shared_phone,
        "n_shared_url": n_shared_url,
        "dist_km_mean": avg(dist_kms),
        "dist_km_median": med(dist_kms),
        "category_counts": {f"{k[0]}|{k[1]}": len(v) for k,v in sorted(cats.items())},
        "examples": examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b9-only", action="store_true",
                        help="Only run B9 residual taxonomy (use when B2-B8 already completed)")
    args = parser.parse_args()
    if args.b9_only:
        # Minimal run: load cached data, build index, run B9 only
        import json as _json
        from pathlib import Path as _Path
        _REPO2 = _Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(_REPO2))

        PROCESSED_DIR2 = _REPO2 / "data" / "processed" / "foursquare"
        import pyarrow.parquet as pq2
        primary_splits2 = _json.loads((_REPO2/"data/processed/foursquare/primary_splits.json").read_text())
        t2 = pq2.read_table(PROCESSED_DIR2/"records.parquet", columns=["record_id","raw_fields"])
        record_ids2 = t2.column("record_id").to_pylist()
        raw_fields2 = t2.column("raw_fields").to_pylist()
        record_map2 = {r: f for r, f in zip(record_ids2, raw_fields2)}

        all_pairs2 = [(q,c) for q,c,l in zip(
            pq2.read_table(PROCESSED_DIR2/"pairs_labels.parquet", columns=["query_id","cand_id","label"]).column("query_id").to_pylist(),
            pq2.read_table(PROCESSED_DIR2/"pairs_labels.parquet", columns=["query_id","cand_id","label"]).column("cand_id").to_pylist(),
            pq2.read_table(PROCESSED_DIR2/"pairs_labels.parquet", columns=["query_id","cand_id","label"]).column("label").to_pylist()) if l==1]
        train_pairs2 = {(q,c) for q,c in all_pairs2 if primary_splits2.get(q)=="train"}
        train_recs2 = [{"record_id": r, "raw_fields": record_map2[r]} for r in record_ids2 if primary_splits2.get(r)=="train"]

        all_recs2 = [{"record_id": r, "raw_fields": f} for r,f in zip(record_ids2, raw_fields2)]
        print("Building index for B9...", flush=True)
        base_bl2 = FoursquareBlocker(gate_size=None, keys=FSQ_ALL_KEYS)
        base_bl2.index_records(all_recs2)

        print(f"Running B9 on {len(train_recs2):,} train queries, {len(train_pairs2):,} pairs...", flush=True)
        tax2 = _residual_taxonomy(base_bl2, train_recs2, train_pairs2, record_map2, seed=args.seed, gate_for_residual=50)
        print(f"  Unrecovered at G=50: {tax2['n_unrecovered']}", flush=True)
        for cat, cnt in sorted(tax2["category_counts"].items()):
            print(f"    {cat}: {cnt}", flush=True)

        out2 = _REPO2 / "experiments" / "foursquare_b9_results.json"
        out2.write_text(_json.dumps({k: v for k, v in tax2.items() if k != "examples"}, indent=2, default=str))
        examples_out = _REPO2 / "experiments" / "foursquare_b9_examples.json"
        examples_out.write_text(_json.dumps(tax2["examples"], indent=2, default=str))
        print(f"Saved to {out2}", flush=True)
    else:
        main(seed=args.seed)
