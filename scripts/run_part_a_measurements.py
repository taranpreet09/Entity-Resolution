"""
scripts/run_part_a_measurements.py — Phase 2c Part A measurements
=================================================================
Produces all numbers needed for the 001b addendum:
  A2  Cross-tab residual taxonomy (v2, G=none) with 5 examples per cell
  A3  cpq median/p95/mean at each G for v2 (both denominators)
  A4  Efficient frontier: K4v2 → +K1 → +K2 → +K3 (order by marginal true pairs/cand)
  A5  Verify unlabeled-query counts by RA code (renaming done in code)
  A6  LEI prefix (first 4 chars) distribution across full file and GB subset

Usage:
    python scripts/run_part_a_measurements.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import struct
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pyarrow.parquet as pq

from ber.canon.normalise import canonicalise, strip_legal_suffixes, outward_code
from ber.eval.ledger import RunLedger
from ber.eval.metrics import (
    bootstrap_ci, build_candidate_map, build_true_targets_map,
    candidates_per_query_distribution, recall_at_k,
)
from ber.retrieve.control_blocking import BlockingConfig, ControlBlocker, ALL_KEY_NAMES_V2

PROCESSED_DIR = _REPO / "data" / "processed" / "gleif_ch"
CH_CSV = (_REPO / "ber" / "data" / "raw" / "companies_house"
          / "BasicCompanyDataAsOneFile-2026-10-01.csv")
GLEIF_CSV = _REPO / "ber" / "data" / "raw" / "gleif" / "gleif_level1.csv"
LEDGER_PATH = _REPO / "experiments" / "runs.parquet"
DATASET_VERSION = "gleif_2026-10-05_ch_2026-10-01"
GATE_SWEEP = [10, 50, 200, 1000, None]


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_splits():
    return json.loads((PROCESSED_DIR / "primary_splits.json").read_text())

def _pairs(path):
    t = pq.read_table(path, columns=["query_id", "cand_id", "label"])
    return [(q, c) for q, c, l in zip(t.column("query_id").to_pylist(),
                                       t.column("cand_id").to_pylist(),
                                       t.column("label").to_pylist()) if l == 1]

def _load_queries(splits, split="val"):
    t = pq.read_table(PROCESSED_DIR/"queries.parquet", columns=["record_id","raw_fields"])
    return [{"record_id": r, "raw_fields": f}
            for r, f in zip(t.column("record_id").to_pylist(),
                            t.column("raw_fields").to_pylist())
            if splits.get(r) == split]

def _fields(rec_or_rf):
    rf = rec_or_rf if isinstance(rec_or_rf, str) else rec_or_rf.get("raw_fields", "{}")
    if isinstance(rf, str):
        try: return json.loads(rf)
        except: return {}
    return rf or {}

def _fmt(v):
    if v is None: return "none"
    if isinstance(v, float):
        if v != v: return "nan"
        return f"{v:.4f}"
    return str(v)

_STOP = {"the", "of", "and", "a", "an", "for", "in", "at", "to"}

def _classify_pair(q_name, t_name, q_post, t_post):
    """Priority: legal_suffix > THE/stopword > reorder > partial_overlap > different.
       Separate from postcode dimension."""
    qc = canonicalise(q_name); tc = canonicalise(t_name)
    qs = strip_legal_suffixes(q_name); ts = strip_legal_suffixes(t_name)
    q_toks = set(qc.split()); t_toks = set(tc.split())

    # Name dimension
    if qs and ts and qs == ts and qc != tc:
        name_cat = "legal_suffix_only"
    elif qc == tc:
        name_cat = "identical"  # shouldn't appear in residual
    elif q_toks == t_toks:
        name_cat = "token_reorder"
    elif (q_toks - _STOP) == (t_toks - _STOP) and (q_toks - _STOP):
        name_cat = "the_stopword_only"
    else:
        u = q_toks | t_toks
        jacc = len(q_toks & t_toks) / len(u) if u else 0.0
        name_cat = "partial_overlap" if jacc >= 0.5 else "different"

    # Postcode dimension
    q_out = outward_code(q_post); t_out = outward_code(t_post)
    q_full = q_post.strip().upper().replace(" ", "")
    t_full = t_post.strip().upper().replace(" ", "")

    if not q_full or not q_out or not t_full or not t_out:
        post_cat = "missing_malformed"
    elif q_full == t_full:
        post_cat = "same_full_postcode"
    elif q_out == t_out:
        post_cat = "same_outward_only"
    else:
        post_cat = "different_outward"

    return name_cat, post_cat


# ── A2: cross-tab taxonomy ──────────────────────────────────────────────────

def run_a2(base_v2, val_query_records, val_true_pairs, query_map, target_map,
           seed=42):
    print("[A2] Cross-tab taxonomy (v2, G=None)...", flush=True)

    res = base_v2.with_gate(None).retrieve(val_query_records)
    retrieved = res.retrieved_pairs
    unrecovered = [(q, c) for q, c in val_true_pairs if (q, c) not in retrieved]

    print(f"  Unrecovered: {len(unrecovered)}", flush=True)

    # Build cross-tab
    crosstab = defaultdict(list)
    for q, c in unrecovered:
        qf = query_map.get(q, {}); tf = target_map.get(c, {})
        q_name = qf.get("LegalName", ""); t_name = tf.get("CompanyName", "")
        q_post = qf.get("LegalAddr_PostCode","") or qf.get("HQAddr_PostCode","")
        t_post = tf.get("RegAddress.PostCode","")
        name_cat, post_cat = _classify_pair(q_name, t_name, q_post, t_post)
        crosstab[(name_cat, post_cat)].append(
            (q_name, t_name, q_post, t_post)
        )

    # Bucket-size distribution for same_full_postcode pairs
    import duckdb
    con = duckdb.connect()
    pos = CH_CSV.as_posix()
    ch_postcode_buckets = {}
    same_full_pairs = [
        pairs for (nc, pc), pairs in crosstab.items() if pc == "same_full_postcode"
    ]
    if same_full_pairs:
        all_same = [p for pairs in same_full_pairs for p in pairs]
        unique_postcodes = set()
        for q_name, t_name, q_post, t_post in all_same:
            nq = q_post.strip().upper().replace(" ","")
            if nq: unique_postcodes.add(nq)
        if unique_postcodes:
            pc_list = "'" + "','".join(list(unique_postcodes)[:100]) + "'"
            try:
                rows = con.execute(
                    f'SELECT UPPER(REPLACE("RegAddress.PostCode"," ","")), COUNT(*) '
                    f'FROM read_csv_auto(\'{pos}\', ignore_errors=true) '
                    f'WHERE UPPER(REPLACE("RegAddress.PostCode"," ","")) IN ({pc_list}) '
                    f'GROUP BY 1'
                ).fetchall()
                ch_postcode_buckets = {r[0]: r[1] for r in rows if r[0]}
            except Exception as e:
                print(f"  Bucket query failed: {e}")

    # Random examples (5 per cell)
    rng = random.Random(seed)
    examples = {}
    for (nc, pc), items in crosstab.items():
        sample = rng.sample(items, min(5, len(items)))
        examples[(nc, pc)] = [
            {"query_name": i[0], "target_name": i[1],
             "query_postcode": i[2], "target_postcode": i[3]}
            for i in sample
        ]

    return {
        "n_unrecovered": len(unrecovered),
        "crosstab_counts": {f"{k[0]}|{k[1]}": len(v) for k, v in sorted(crosstab.items())},
        "crosstab_examples": {f"{k[0]}|{k[1]}": v for k, v in examples.items()},
        "same_full_postcode_bucket_sizes": ch_postcode_buckets,
    }


# ── A3: cpq distribution for v2 ──────────────────────────────────────────────

def run_a3(base_v2, val_query_records, val_true_pairs, seed=42):
    print("[A3] cpq distribution for v2...", flush=True)
    true_targets = build_true_targets_map(
        [{"query_id": q, "cand_id": c, "label": 1} for q, c in val_true_pairs])
    n_queries_with_targets = len(true_targets)
    n_all_val = len(val_query_records)

    results = {}
    for g in GATE_SWEEP:
        bl = base_v2.with_gate(g)
        res = bl.retrieve(val_query_records)
        cands = res.candidates
        cm = build_candidate_map(cands)
        # cpq over all queries (including zero-cand)
        cpq_all = {qid: len(cm.get(qid, [])) for rec in val_query_records
                   for qid in [rec["record_id"]]}
        counts_all = sorted(cpq_all.values())
        # cpq over queries with ≥1 candidate
        counts_pos = sorted(v for v in counts_all if v > 0)

        def pct(data, p):
            if not data: return float("nan")
            idx = (len(data)-1)*p/100
            lo, hi = int(idx), min(int(idx)+1, len(data)-1)
            return data[lo]*(1-(idx-lo)) + data[hi]*(idx-lo)

        results[str(g)] = {
            "n_candidates": len(cands),
            "n_queries_all_val": n_all_val,
            "n_queries_with_targets": n_queries_with_targets,
            "n_queries_with_any_cand": len(counts_pos),
            # Over queries with ≥1 candidate (standard cpq denominator)
            "cpq_median_over_cand_queries": round(pct(counts_pos, 50), 1) if counts_pos else float("nan"),
            "cpq_p95_over_cand_queries": round(pct(counts_pos, 95), 1) if counts_pos else float("nan"),
            "cpq_mean_over_cand_queries": round(sum(counts_pos)/len(counts_pos), 2) if counts_pos else float("nan"),
            # Over all val queries (including zero-cand)
            "cpq_mean_over_all_val": round(sum(counts_all)/len(counts_all), 2) if counts_all else float("nan"),
        }
        print(f"  G={str(g):<5} med={results[str(g)]['cpq_median_over_cand_queries']} "
              f"p95={results[str(g)]['cpq_p95_over_cand_queries']} "
              f"mean(cand)={results[str(g)]['cpq_mean_over_cand_queries']} "
              f"mean(all)={results[str(g)]['cpq_mean_over_all_val']}", flush=True)
    return results


# ── A4: efficient frontier ────────────────────────────────────────────────────

def run_a4(base_v2, val_query_records, val_true_pairs, seed=42):
    """Greedy frontier: start with K4v2, add keys in order of marginal true pairs per candidate."""
    print("[A4] Efficient frontier (v2)...", flush=True)
    true_set = set(val_true_pairs)
    n_true = len(val_true_pairs)

    key_order_to_try = ["K4v2", "K1", "K2", "K3"]  # initial order per task spec
    # We'll measure each marginal gain

    selected = []
    current_retrieved = set()
    current_cands = 0
    rows = []

    for g in [50, None]:
        selected = []
        current_retrieved = set()
        current_cands = 0
        rows_g = []
        remaining = list(key_order_to_try)

        while remaining:
            best_key = None
            best_marginal_true = -1
            best_marginal_ratio = -1
            best_new_retrieved = None
            best_new_cands = 0

            for k in remaining:
                test_keys = selected + [k]
                bl = ControlBlocker(BlockingConfig(gate_size=g, keys=test_keys, version="v2"))
                bl._index = {k2: base_v2._index.get(k2, {}) for k2 in test_keys}
                bl._target_raw = base_v2._target_raw
                res = bl.retrieve(val_query_records)
                new_retrieved = res.retrieved_pairs
                marginal_true = len((new_retrieved & true_set) - current_retrieved)
                marginal_cands = len(res.candidates) - current_cands
                ratio = marginal_true / max(marginal_cands, 1)
                if marginal_true > best_marginal_true or (
                    marginal_true == best_marginal_true and ratio > best_marginal_ratio
                ):
                    best_key = k
                    best_marginal_true = marginal_true
                    best_marginal_ratio = ratio
                    best_new_retrieved = new_retrieved
                    best_new_cands = len(res.candidates)

            selected.append(best_key)
            remaining.remove(best_key)
            current_retrieved = best_new_retrieved & true_set if best_new_retrieved else current_retrieved
            current_cands = best_new_cands
            cum_true = len(current_retrieved)
            cum_recall = round(cum_true / max(n_true, 1), 6)

            rows_g.append({
                "G": g,
                "key_added": best_key,
                "keys_active": list(selected),
                "cumulative_true_pairs": cum_true,
                "cumulative_recall": cum_recall,
                "cumulative_candidates": current_cands,
                "marginal_true_pairs": best_marginal_true,
                "marginal_candidates": best_new_cands - (current_cands - best_new_cands + (current_cands - best_new_cands)),
            })
            print(f"  G={str(g):<5} +{best_key}: "
                  f"cum_true={cum_true} recall={cum_recall:.4f} "
                  f"cands={current_cands:,} marg_true={best_marginal_true}", flush=True)
        rows.extend(rows_g)

    return rows


# ── A6: LEI prefix distribution ──────────────────────────────────────────────

def run_a6():
    print("[A6] LEI prefix distribution...", flush=True)
    with open(GLEIF_CSV, "rb") as f:
        raw = f.read()
    fn_len = struct.unpack("<H", raw[26:28])[0]
    extra_len = struct.unpack("<H", raw[28:30])[0]
    data_start = 30 + fn_len + extra_len
    compressed = raw[data_start:]

    d = zlib.decompressobj(-15)
    STEP = 5_000_000
    leftovers = b""
    header = []
    lei_idx = ra_idx = None

    all_prefixes = Counter()
    gb_prefixes = Counter()
    GB_RA = {"RA000585", "RA000586", "RA000587"}
    file_positions = []  # row numbers of GB records
    total_rows = 0
    gb_rows = 0
    gb_row_positions = []

    pos_c = 0
    while pos_c < len(compressed):
        chunk = compressed[pos_c:pos_c + STEP]
        pos_c += STEP
        try:
            dec = d.decompress(chunk)
        except zlib.error:
            break

        data = leftovers + dec
        lines = data.split(b"\n")
        leftovers = lines[-1]

        for line_bytes in lines[:-1]:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            if not line:
                continue
            try:
                parts = next(csv.reader([line]))
            except Exception:
                continue

            if not header:
                header = [c.strip('"').strip() for c in parts]
                for i, c in enumerate(header):
                    if c == "LEI": lei_idx = i
                    if c == "Entity.RegistrationAuthority.RegistrationAuthorityID": ra_idx = i
                continue

            total_rows += 1
            lei = parts[lei_idx].strip('"').strip() if lei_idx and lei_idx < len(parts) else ""
            ra = parts[ra_idx].strip('"').strip() if ra_idx and ra_idx < len(parts) else ""

            if lei:
                prefix = lei[:4]
                all_prefixes[prefix] += 1
                if ra in GB_RA:
                    gb_prefixes[prefix] += 1
                    gb_rows += 1
                    gb_row_positions.append(total_rows)

    # File position stats for GB records
    if gb_row_positions:
        min_pos = min(gb_row_positions)
        max_pos = max(gb_row_positions)
        median_pos = sorted(gb_row_positions)[len(gb_row_positions) // 2]
    else:
        min_pos = max_pos = median_pos = 0

    return {
        "total_rows_in_file": total_rows,
        "n_distinct_lei_prefixes": len(all_prefixes),
        "top_20_lei_prefixes_all": dict(all_prefixes.most_common(20)),
        "n_gb_records": gb_rows,
        "n_distinct_gb_prefixes": len(gb_prefixes),
        "top_20_lei_prefixes_gb": dict(gb_prefixes.most_common(20)),
        "gb_first_row_position": min_pos,
        "gb_last_row_position": max_pos,
        "gb_median_row_position": median_pos,
        "file_fraction_containing_gb": round((max_pos - min_pos) / max(total_rows, 1), 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(seed=42):
    import subprocess
    try:
        commit = subprocess.run(
            ["git","rev-parse","HEAD"], capture_output=True, text=True,
            cwd=str(_REPO), timeout=5).stdout.strip()
    except Exception:
        commit = "untracked"

    ledger = RunLedger(LEDGER_PATH)
    splits = _load_splits()
    all_positive_pairs = _pairs(PROCESSED_DIR / "pairs_labels.parquet")
    val_true_pairs = [(q, c) for q, c in all_positive_pairs if splits.get(q) == "val"]
    val_query_records = _load_queries(splits, "val")

    # Load query and target raw fields for val true pairs
    true_cids = {c for _, c in val_true_pairs}
    query_map = {rec["record_id"]: _fields(rec) for rec in val_query_records}
    t_tbl = pq.read_table(PROCESSED_DIR/"targets.parquet", columns=["record_id","raw_fields"])
    target_map = {rid: _fields(rf) for rid, rf in zip(
        t_tbl.column("record_id").to_pylist(),
        t_tbl.column("raw_fields").to_pylist()) if rid in true_cids}

    # Build v2 base index once
    print("Building v2 base index...", flush=True)
    base_v2 = ControlBlocker(BlockingConfig(gate_size=None, keys=ALL_KEY_NAMES_V2, version="v2"))
    base_v2.index_targets_from_duckdb(CH_CSV, batch_size=250_000, verbose=True)

    a2 = run_a2(base_v2, val_query_records, val_true_pairs, query_map, target_map, seed)
    a3 = run_a3(base_v2, val_query_records, val_true_pairs, seed)
    a4 = run_a4(base_v2, val_query_records, val_true_pairs, seed)
    a6 = run_a6()

    # Also compute paired bootstrap v2-vs-v1 delta for A1
    print("[A1] Paired bootstrap v2 - v1 delta...", flush=True)
    from ber.retrieve.control_blocking import ALL_KEY_NAMES
    base_v1 = ControlBlocker(BlockingConfig(gate_size=None, keys=ALL_KEY_NAMES, version="v1"))
    base_v1.index_targets_from_duckdb(CH_CSV, batch_size=250_000, verbose=False)

    true_targets = build_true_targets_map(
        [{"query_id": q, "cand_id": c, "label": 1} for q, c in val_true_pairs])

    def _recall_for_blocker(blocker, qrecs, tt):
        res = blocker.retrieve(qrecs)
        cm = build_candidate_map(res.candidates)
        max_rank = max((c["rank"] for c in res.candidates), default=0)
        return recall_at_k(cm, tt, k=max(max_rank, 1))

    # For paired bootstrap we need per-entity scores for v1 and v2 G=None
    res_v1 = base_v1.with_gate(None).retrieve(val_query_records)
    res_v2 = base_v2.with_gate(None).retrieve(val_query_records)
    cm_v1 = build_candidate_map(res_v1.candidates)
    cm_v2 = build_candidate_map(res_v2.candidates)
    max_rank_v1 = max((c["rank"] for c in res_v1.candidates), default=0)
    max_rank_v2 = max((c["rank"] for c in res_v2.candidates), default=0)

    import functools
    fn_v1 = functools.partial(recall_at_k, k=max(max_rank_v1, 1))
    fn_v2 = functools.partial(recall_at_k, k=max(max_rank_v2, 1))

    from ber.eval.metrics import compare_systems
    paired = compare_systems(cm_v2, cm_v1, true_targets, fn_v2,
                             n_boot=1000, seed=seed)
    print(f"  v2 recall={paired['score_a']:.6f} v1 recall={paired['score_b']:.6f} "
          f"delta={paired['delta']:.6f} p={paired['p_value']:.4f}", flush=True)

    # Log to ledger
    run_id = ledger.start_run(
        config={"experiment": "part_a_measurements", "commit": commit, "seed": seed},
        dataset="gleif_ch", seed=seed, dataset_version=DATASET_VERSION,
    )
    ledger.log_metrics(run_id, {
        "a2_crosstab": a2["crosstab_counts"],
        "a2_n_unrecovered": a2["n_unrecovered"],
        "a3_cpq": a3,
        "a4_frontier": a4,
        "a6_lei_prefix": {k: v for k, v in a6.items() if "examples" not in k},
        "a1_paired_bootstrap": paired,
    })

    # Save full results
    out = _REPO / "experiments" / "part_a_results.json"
    # Remove non-serializable examples from crosstab before saving
    a2_save = {k: v for k, v in a2.items() if k != "crosstab_examples"}
    a2_save["crosstab_examples"] = {
        k: v for k, v in a2["crosstab_examples"].items()
    }
    out.write_text(json.dumps({
        "commit": commit,
        "a1_paired_bootstrap": paired,
        "a2": a2_save,
        "a3_cpq": a3,
        "a4_frontier": a4,
        "a6": a6,
    }, indent=2, default=str))
    print(f"\nResults saved to {out}", flush=True)
    return {"a1": paired, "a2": a2, "a3": a3, "a4": a4, "a6": a6}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
