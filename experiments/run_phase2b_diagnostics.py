"""
experiments/run_phase2b_diagnostics.py — Phase 2b Diagnostics
==============================================================
Runs:
  1. control_v1 reproduction check (must be bit-identical to Exp 001)
  2. control_v2 G-sweep (K4v2 = suffix-stripped name, no postcode)
  3. Corrected per-key metrics: standalone recall, leave-one-out, unique
     TRUE pairs — for v1 and v2 at G=50 and G=none
  4. Residual taxonomy: rule-based classification of val pairs not
     retrieved by v2 at G=none
  5. No-match query diagnosis (all 12,424)
  6. GLEIF sample statistics (vs latest_lei.json metadata)

Usage::
    python experiments/run_phase2b_diagnostics.py [--seed 42]

All results written to experiments/001b_control_diagnostics.md and logged
to experiments/runs.parquet.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import random
import re
import sys
import zlib, struct
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pyarrow.parquet as pq

from ber.canon.normalise import (
    canonicalise, strip_legal_suffixes, outward_code, postcode_district, first_token
)
from ber.eval.ledger import RunLedger
from ber.eval.metrics import (
    bootstrap_ci, build_candidate_map, build_true_targets_map,
    recall_at_k, zero_candidate_rate,
)
from ber.retrieve.control_blocking import (
    BlockingConfig, ControlBlocker,
    ALL_KEY_NAMES, ALL_KEY_NAMES_V2,
)

PROCESSED_DIR = _REPO / "data" / "processed" / "gleif_ch"
CH_CSV = (_REPO / "ber" / "data" / "raw" / "companies_house"
          / "BasicCompanyDataAsOneFile-2026-10-01.csv")
GLEIF_CSV = _REPO / "ber" / "data" / "raw" / "gleif" / "gleif_level1.csv"
LEDGER_PATH = _REPO / "experiments" / "runs.parquet"
DATASET_VERSION = "gleif_2026-10-05_ch_2026-10-01"
GATE_SWEEP = [10, 50, 200, 1000, None]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pairs_from_parquet(path: Path) -> list[tuple[str, str]]:
    t = pq.read_table(path, columns=["query_id", "cand_id", "label"])
    return [(q, c) for q, c, l in zip(
        t.column("query_id").to_pylist(),
        t.column("cand_id").to_pylist(),
        t.column("label").to_pylist()) if l == 1]

def _load_splits() -> dict[str, str]:
    return json.loads((PROCESSED_DIR / "primary_splits.json").read_text())

def _load_val_queries(splits: dict[str, str]) -> list[dict]:
    t = pq.read_table(PROCESSED_DIR / "queries.parquet",
                      columns=["record_id", "raw_fields"])
    return [{"record_id": r, "raw_fields": f}
            for r, f in zip(t.column("record_id").to_pylist(),
                            t.column("raw_fields").to_pylist())
            if splits.get(r) == "val"]

def _fields(rec: dict) -> dict:
    rf = rec.get("raw_fields", "{}")
    if isinstance(rf, str):
        try: return json.loads(rf)
        except: return {}
    return rf or {}

def _fmt(v) -> str:
    if v is None: return "none"
    if isinstance(v, float):
        if v != v: return "nan"
        return f"{v:.4f}"
    return str(v)

def _bootstrap_recall(cand_map, true_tgts, k, seed=42, n_boot=500):
    fn = functools.partial(recall_at_k, k=k)
    pt, lo, hi = bootstrap_ci(cand_map, true_tgts, fn, n_boot=n_boot, seed=seed)
    return pt, lo, hi

# ─────────────────────────────────────────────────────────────────────────────
# Per-key analysis helper
# ─────────────────────────────────────────────────────────────────────────────

def per_key_analysis(
    base_blocker: ControlBlocker,
    val_query_records: list[dict],
    val_true_pairs: list[tuple[str, str]],
    gate_size: int | None,
    key_names: list[str],
    seed: int = 42,
    n_boot: int = 200,
) -> dict:
    """Compute standalone, leave-one-out and unique-TRUE-pairs for each key."""
    true_set = set(val_true_pairs)
    true_targets_map = build_true_targets_map(
        [{"query_id": q, "cand_id": c, "label": 1} for q, c in val_true_pairs])
    n_true = len(val_true_pairs)

    # Full system
    full_blocker = base_blocker.with_gate(gate_size)
    full_result = full_blocker.retrieve(val_query_records)
    full_retrieved = full_result.retrieved_pairs
    full_cands = len(full_result.candidates)

    results = {}

    for k in key_names:
        # Standalone: only this key
        solo_cfg = BlockingConfig(gate_size=gate_size, keys=[k],
                                  version=base_blocker.config.version)
        solo = ControlBlocker(solo_cfg)
        solo._index = {k: base_blocker._index.get(k, {})}
        solo._target_raw = base_blocker._target_raw
        solo_result = solo.retrieve(val_query_records)
        solo_retrieved = solo_result.retrieved_pairs
        solo_true = solo_retrieved & true_set
        solo_recall = len(solo_true) / max(n_true, 1)
        solo_cands = len(solo_result.candidates)

        # Leave-one-out: all keys except this one
        loo_keys = [k2 for k2 in key_names if k2 != k]
        loo_cfg = BlockingConfig(gate_size=gate_size, keys=loo_keys,
                                 version=base_blocker.config.version)
        loo = ControlBlocker(loo_cfg)
        loo._index = {k2: base_blocker._index.get(k2, {}) for k2 in loo_keys}
        loo._target_raw = base_blocker._target_raw
        loo_result = loo.retrieve(val_query_records)
        loo_retrieved = loo_result.retrieved_pairs
        loo_recall = len(loo_retrieved & true_set) / max(n_true, 1)
        loo_cands = len(loo_result.candidates)

        # Unique TRUE pairs: recovered by this key but NOT by any other key alone
        other_keys = [k2 for k2 in key_names if k2 != k]
        other_retrieved: set = set()
        for k2 in other_keys:
            ok_cfg = BlockingConfig(gate_size=gate_size, keys=[k2],
                                    version=base_blocker.config.version)
            ok = ControlBlocker(ok_cfg)
            ok._index = {k2: base_blocker._index.get(k2, {})}
            ok._target_raw = base_blocker._target_raw
            other_retrieved |= ok.retrieve(val_query_records).retrieved_pairs
        unique_true = solo_retrieved & true_set - (other_retrieved & true_set)

        results[k] = {
            "standalone_true_pairs": len(solo_true),
            "standalone_recall": round(solo_recall, 6),
            "standalone_candidates": solo_cands,
            "leave_one_out_recall": round(loo_recall, 6),
            "leave_one_out_candidates": loo_cands,
            "unique_true_pairs": len(unique_true),
        }

    return {
        "full_candidates": full_cands,
        "full_true_pairs_recovered": len(full_retrieved & true_set),
        "full_recall": round(len(full_retrieved & true_set) / max(n_true, 1), 6),
        "per_key": results,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Residual taxonomy
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = {"the", "of", "and", "a", "an", "for", "in", "at", "to"}

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0

def _trigrams(s: str) -> set[str]:
    return {s[i:i+3] for i in range(len(s)-2)} if len(s) >= 3 else set()

def _classify_pair(q_name: str, t_name: str, q_post: str, t_post: str) -> tuple[str, dict]:
    """
    Priority order: a > b > c > d > e_suffix > f > g > h.
    Returns (category_code, extra_data).
    """
    qc = canonicalise(q_name)
    tc = canonicalise(t_name)
    qs = strip_legal_suffixes(q_name)
    ts = strip_legal_suffixes(t_name)
    q_out = outward_code(q_post)
    t_out = outward_code(t_post)

    q_toks = set(qc.split())
    t_toks = set(tc.split())
    qs_toks = set(qs.split()) if qs else set()
    ts_toks = set(ts.split()) if ts else set()

    tok_jacc = _jaccard(q_toks, t_toks)
    tri_jacc = _jaccard(_trigrams(qc), _trigrams(tc))
    extra = {"tok_jaccard": round(tok_jacc, 3), "tri_jaccard": round(tri_jacc, 3)}

    # (a) legal-suffix variant: stripped names match, full names differ
    if qs and ts and qs == ts and qc != tc:
        return "a_legal_suffix", extra

    # (b) case/punct/spacing only: already collapsed by canonicalise, so
    # if canon forms match they'd have been retrieved — this can only fire
    # if names differ only by non-ASCII that survives; in practice skip
    if qc == tc:
        return "b_punct_case", extra

    # (c) token reordering: same tokens, different order
    if q_toks == t_toks and q_toks:
        return "c_reorder", extra

    # (d) stop-word addition/removal (one side has extra stop words)
    q_content = q_toks - _STOP_WORDS
    t_content = t_toks - _STOP_WORDS
    if q_content == t_content and q_content:
        return "d_stopword", extra

    # (f) query postcode missing
    if not q_post.strip() or not outward_code(q_post):
        return "f_postcode_missing", extra

    # (g) postcode present but different outward code
    if q_out and t_out and q_out != t_out:
        return "g_postcode_mismatch", extra

    # (e) substantial rename — catch-all with Jaccard stats
    return "e_substantial", extra


def diagnose_residual(
    val_true_pairs: list[tuple[str, str]],
    retrieved_pairs: set[tuple[str, str]],
    query_map: dict[str, dict],   # record_id → raw_fields dict
    target_map: dict[str, dict],  # record_id → raw_fields dict
    rng_seed: int = 42,
) -> dict:
    """Classify unrecovered val true pairs by taxonomy (a–h)."""
    unrecovered = [(q, c) for q, c in val_true_pairs if (q, c) not in retrieved_pairs]

    categories: dict[str, list] = defaultdict(list)
    tok_jaccards, tri_jaccards = [], []
    q_postcode_missing = 0
    same_outward = 0

    for q, c in unrecovered:
        qf = query_map.get(q, {})
        tf = target_map.get(c, {})
        q_name = qf.get("LegalName", "")
        t_name = tf.get("CompanyName", "")
        q_post = qf.get("LegalAddr_PostCode", "") or qf.get("HQAddr_PostCode", "")
        t_post = tf.get("RegAddress.PostCode", "")

        cat, extra = _classify_pair(q_name, t_name, q_post, t_post)
        categories[cat].append((q_name, t_name, q_post, t_post))
        tok_jaccards.append(extra["tok_jaccard"])
        tri_jaccards.append(extra["tri_jaccard"])

        q_out = outward_code(q_post)
        t_out = outward_code(t_post)
        if not q_post.strip() or not q_out:
            q_postcode_missing += 1
        if q_out and t_out and q_out == t_out:
            same_outward += 1

    # Random examples (10 per category, seeded)
    rng = random.Random(rng_seed)
    examples: dict[str, list] = {}
    for cat, pairs in categories.items():
        sample = rng.sample(pairs, min(10, len(pairs)))
        examples[cat] = [{"query_name": p[0], "target_name": p[1],
                          "query_postcode": p[2], "target_postcode": p[3]}
                         for p in sample]

    def _pct(n, d): return round(100 * n / max(d, 1), 2)

    n = len(unrecovered)
    counts = {cat: len(pairs) for cat, pairs in categories.items()}

    def _avg(lst): return round(sum(lst) / len(lst), 3) if lst else float("nan")
    def _med(lst):
        if not lst: return float("nan")
        s = sorted(lst)
        m = len(s) // 2
        return round(s[m] if len(s) % 2 else (s[m-1]+s[m])/2, 3)

    return {
        "n_unrecovered": n,
        "n_q_postcode_missing": q_postcode_missing,
        "n_same_outward_code": same_outward,
        "pct_q_postcode_missing": _pct(q_postcode_missing, n),
        "pct_same_outward": _pct(same_outward, n),
        "category_counts": counts,
        "tok_jaccard_mean": _avg(tok_jaccards),
        "tok_jaccard_median": _med(tok_jaccards),
        "tri_jaccard_mean": _avg(tri_jaccards),
        "tri_jaccard_median": _med(tri_jaccards),
        "examples": examples,
    }

# ─────────────────────────────────────────────────────────────────────────────
# No-match query diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_no_match(rng_seed: int = 42) -> dict:
    """Diagnose all 12,424 no-match GLEIF queries from the full query set."""
    import duckdb

    no_match_leis = set(json.loads(
        (PROCESSED_DIR / "no_match_queries.json").read_text()))

    # Load all query raw_fields for no-match LEIs
    t = pq.read_table(PROCESSED_DIR / "queries.parquet",
                      columns=["record_id", "raw_fields"])
    nm_fields: dict[str, dict] = {}
    for rid, rf in zip(t.column("record_id").to_pylist(),
                       t.column("raw_fields").to_pylist()):
        lei = rid.replace("GLEIF_", "")
        if lei in no_match_leis:
            try: nm_fields[rid] = json.loads(rf) if isinstance(rf, str) else rf
            except: nm_fields[rid] = {}

    # Counters
    ra_counter: Counter = Counter()
    id_format: Counter = Counter()  # prefix pattern
    entity_status: Counter = Counter()
    entity_category_proxy: Counter = Counter()  # from RA code

    for rid, f in nm_fields.items():
        ra = f.get("RA_Code", "")
        ra_counter[ra] += 1
        eid = f.get("RA_EntityID", "")
        status = f.get("EntityStatus", "")
        entity_status[status if status else "unknown"] += 1
        # ID format: prefix letters
        if eid:
            prefix = re.match(r"^([A-Z]+)", eid.strip().upper())
            fmt = prefix.group(1) if prefix else "NUMERIC"
            if fmt.isdigit() or not prefix:
                fmt = "NUMERIC"
            id_format[fmt] += 1
        else:
            id_format["EMPTY"] += 1

    # K1 name-collision check: does any no-match query share a canonical name
    # with a CH record under a DIFFERENT company number?
    print("  [no-match] Building K1 collision check via DuckDB...", flush=True)
    con = duckdb.connect()
    pos = CH_CSV.as_posix()
    # Get all (canon_name → count) from CH
    ch_names_rows = con.execute(
        f'SELECT "CompanyName" FROM read_csv_auto(\'{pos}\', ignore_errors=true)'
    ).fetchall()
    ch_canon_set: set[str] = set()
    for (cn,) in ch_names_rows:
        c = canonicalise(cn or "")
        if c:
            ch_canon_set.add(c)

    k1_collision = 0
    for rid, f in nm_fields.items():
        q_canon = canonicalise(f.get("LegalName", ""))
        if q_canon and q_canon in ch_canon_set:
            k1_collision += 1

    return {
        "n_no_match": len(nm_fields),
        "ra_code_counts": dict(ra_counter.most_common()),
        "entity_status_counts": dict(entity_status.most_common()),
        "id_format_counts": dict(id_format.most_common()),
        "k1_name_collision_count": k1_collision,
        "k1_name_collision_pct": round(100 * k1_collision / max(len(nm_fields), 1), 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# GLEIF sample statistics
# ─────────────────────────────────────────────────────────────────────────────

def gleif_sample_stats() -> dict:
    """Row count vs metadata; status/category/authority distributions for GB subset."""
    # latest_lei.json has the announced count
    lei_meta_path = _REPO / "ber" / "data" / "raw" / "gleif" / "latest_lei.json"
    with open(lei_meta_path) as f:
        meta = json.load(f)
    announced = meta["data"]["type"]
    full_file_meta = meta["data"].get("full_file", {}).get("csv", {})
    announced_count = full_file_meta.get("record_count", "unknown")
    file_date = meta["data"].get("publish_date", "unknown")

    # Scan the (partial) GLEIF file for GB status/authority distributions
    with open(GLEIF_CSV, "rb") as f:
        raw_bytes = f.read()
    fn_len = struct.unpack("<H", raw_bytes[26:28])[0]
    extra_len = struct.unpack("<H", raw_bytes[28:30])[0]
    data_start = 30 + fn_len + extra_len
    compressed = raw_bytes[data_start:]

    d = zlib.decompressobj(-15)
    header_cols: list[str] = []
    GB_RA = {"RA000585", "RA000586", "RA000587"}

    total_rows = 0
    status_counter: Counter = Counter()
    ra_counter: Counter = Counter()

    STEP = 5_000_000
    leftovers = b""
    pos = 0
    ra_idx = status_idx = None
    first_chunk = True

    while pos < len(compressed):
        chunk = compressed[pos:pos + STEP]
        pos += STEP
        try:
            chunk_dec = d.decompress(chunk)
        except zlib.error:
            break

        data = leftovers + chunk_dec
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

            if not header_cols:
                header_cols = [c.strip('"').strip() for c in parts]
                for i, c in enumerate(header_cols):
                    if c == "Entity.EntityStatus": status_idx = i
                    if c == "Entity.RegistrationAuthority.RegistrationAuthorityID": ra_idx = i
                continue

            total_rows += 1
            if ra_idx is not None and ra_idx < len(parts):
                ra = parts[ra_idx].strip('"').strip()
                if ra in GB_RA:
                    ra_counter[ra] += 1
                    if status_idx is not None and status_idx < len(parts):
                        status = parts[status_idx].strip('"').strip()
                        status_counter[status if status else "UNKNOWN"] += 1

    return {
        "gleif_golden_copy_date": file_date,
        "gleif_announced_record_count": announced_count,
        "gleif_file_rows_decompressed": total_rows,
        "gleif_coverage_pct": round(100 * total_rows / max(int(announced_count) if str(announced_count).isdigit() else 1, 1), 1),
        "gb_ra_code_counts_in_file": dict(ra_counter.most_common()),
        "gb_entity_status_counts": dict(status_counter.most_common()),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(seed: int = 42) -> dict:
    import subprocess
    ledger = RunLedger(LEDGER_PATH)

    # Get commit hash
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(_REPO), timeout=5
        ).stdout.strip()
    except Exception:
        commit = "untracked"
    print(f"git commit: {commit}", flush=True)

    splits = _load_splits()
    all_positive_pairs = _pairs_from_parquet(PROCESSED_DIR / "pairs_labels.parquet")
    val_true_pairs = [(q, c) for q, c in all_positive_pairs if splits.get(q) == "val"]
    val_query_records = _load_val_queries(splits)
    n_val_true = len(val_true_pairs)
    print(f"Val true pairs: {n_val_true}", flush=True)

    # ── Query and target raw fields maps (for taxonomy / per-key analysis) ──
    query_map: dict[str, dict] = {}
    for rec in val_query_records:
        query_map[rec["record_id"]] = _fields(rec)

    # Load target raw fields only for the true pair targets we need
    true_cand_ids = {c for _, c in val_true_pairs}
    t_tbl = pq.read_table(PROCESSED_DIR / "targets.parquet",
                          columns=["record_id", "raw_fields"])
    target_map: dict[str, dict] = {}
    for rid, rf in zip(t_tbl.column("record_id").to_pylist(),
                       t_tbl.column("raw_fields").to_pylist()):
        if rid in true_cand_ids:
            try: target_map[rid] = json.loads(rf) if isinstance(rf, str) else (rf or {})
            except: target_map[rid] = {}

    # ── Build SINGLE index for each version; reuse via with_gate() ──
    print("\n=== Building v1 index ===", flush=True)
    base_v1_cfg = BlockingConfig(gate_size=None, keys=ALL_KEY_NAMES, version="v1")
    base_v1 = ControlBlocker(base_v1_cfg)
    base_v1.index_targets_from_duckdb(CH_CSV, batch_size=250_000, verbose=True)

    print("\n=== Building v2 index ===", flush=True)
    base_v2_cfg = BlockingConfig(gate_size=None, keys=ALL_KEY_NAMES_V2, version="v2")
    base_v2 = ControlBlocker(base_v2_cfg)
    base_v2.index_targets_from_duckdb(CH_CSV, batch_size=250_000, verbose=True)

    # ── 1. v1 reproduction check (G-sweep, compare to Exp 001) ──
    print("\n=== v1 reproduction check ===", flush=True)
    EXP001_RECALL = {10: 0.9665, 50: 0.9675, 200: 0.9683, 1000: 0.9685, None: 0.9685}
    EXP001_CANDS = {10: 34437, 50: 130629, 200: 321275, 1000: 549355, None: 685788}

    v1_repro_run_id = ledger.start_run(
        config={"experiment": "control_v1_reproduction", "seed": seed,
                "dataset_version": DATASET_VERSION, "commit": commit},
        dataset="gleif_ch", seed=seed, dataset_version=DATASET_VERSION,
    )
    v1_sweep: list[dict] = []
    for g in GATE_SWEEP:
        bl = base_v1.with_gate(g)
        res = bl.retrieve(val_query_records)
        true_set = set(val_true_pairs)
        true_targets = build_true_targets_map(
            [{"query_id": q, "cand_id": c, "label": 1} for q, c in val_true_pairs])
        cand_map = build_candidate_map(res.candidates)
        max_rank = max((c["rank"] for c in res.candidates), default=0)
        recall = recall_at_k(cand_map, true_targets, k=max(max_rank, 1))
        n_cands = len(res.candidates)
        identical = (round(recall, 4) == EXP001_RECALL.get(g) and n_cands == EXP001_CANDS.get(g))
        print(f"  G={str(g):<5} recall={recall:.4f} cands={n_cands:,}  "
              f"{'✓ IDENTICAL' if identical else '✗ DIFFERS'}", flush=True)
        v1_sweep.append({"gate_size": g, "recall": round(recall, 6),
                         "n_candidates": n_cands, "identical_to_001": identical})
    all_identical = all(r["identical_to_001"] for r in v1_sweep)
    ledger.log_metrics(v1_repro_run_id,
                       {"v1_sweep": v1_sweep, "all_identical": all_identical,
                        "commit": commit})
    print(f"  Reproduction: {'ALL IDENTICAL ✓' if all_identical else 'DIFFERS ✗'}")

    # ── 2. v2 G-sweep ──
    print("\n=== v2 G-sweep ===", flush=True)
    v2_run_id = ledger.start_run(
        config={"experiment": "control_v2_sweep", "keys": ALL_KEY_NAMES_V2,
                "seed": seed, "dataset_version": DATASET_VERSION, "commit": commit},
        dataset="gleif_ch", seed=seed, dataset_version=DATASET_VERSION,
    )
    v2_sweep: list[dict] = []
    for g in GATE_SWEEP:
        bl = base_v2.with_gate(g)
        res = bl.retrieve(val_query_records)
        true_targets = build_true_targets_map(
            [{"query_id": q, "cand_id": c, "label": 1} for q, c in val_true_pairs])
        cand_map = build_candidate_map(res.candidates)
        max_rank = max((c["rank"] for c in res.candidates), default=0)
        recall = recall_at_k(cand_map, true_targets, k=max(max_rank, 1))
        z_rate = zero_candidate_rate(cand_map, true_targets)
        n_cands = len(res.candidates)
        recovered = len(res.retrieved_pairs & set(val_true_pairs))
        pt, lo, hi = _bootstrap_recall(cand_map, true_targets, max(max_rank, 1),
                                       seed=seed, n_boot=500)
        print(f"  G={str(g):<5} recall={recall:.4f} [{lo:.4f},{hi:.4f}]  "
              f"cands={n_cands:,}  zero={z_rate:.4f}", flush=True)
        v2_sweep.append({"gate_size": g, "recall": round(recall, 6),
                         "recall_ci_lo": round(lo, 6), "recall_ci_hi": round(hi, 6),
                         "n_candidates": n_cands, "zero_candidate_rate": round(z_rate, 6),
                         "n_true_recovered": recovered})
    ledger.log_metrics(v2_run_id, {"v2_sweep": v2_sweep})

    # ── 3. Corrected per-key metrics ──
    print("\n=== Per-key metrics v1 G=50 ===", flush=True)
    pk_v1_g50 = per_key_analysis(base_v1, val_query_records, val_true_pairs,
                                  50, ALL_KEY_NAMES, seed=seed)
    print("\n=== Per-key metrics v1 G=None ===", flush=True)
    pk_v1_gnone = per_key_analysis(base_v1, val_query_records, val_true_pairs,
                                    None, ALL_KEY_NAMES, seed=seed)
    print("\n=== Per-key metrics v2 G=50 ===", flush=True)
    pk_v2_g50 = per_key_analysis(base_v2, val_query_records, val_true_pairs,
                                  50, ALL_KEY_NAMES_V2, seed=seed)
    print("\n=== Per-key metrics v2 G=None ===", flush=True)
    pk_v2_gnone = per_key_analysis(base_v2, val_query_records, val_true_pairs,
                                    None, ALL_KEY_NAMES_V2, seed=seed)

    pk_run_id = ledger.start_run(
        config={"experiment": "per_key_corrected", "seed": seed, "commit": commit},
        dataset="gleif_ch", seed=seed, dataset_version=DATASET_VERSION,
    )
    ledger.log_metrics(pk_run_id, {
        "v1_g50": pk_v1_g50, "v1_gnone": pk_v1_gnone,
        "v2_g50": pk_v2_g50, "v2_gnone": pk_v2_gnone,
    })

    # ── 4. Residual taxonomy (v2 G=None) ──
    print("\n=== Residual taxonomy ===", flush=True)
    bl_v2_none = base_v2.with_gate(None)
    res_v2_none = bl_v2_none.retrieve(val_query_records)
    v2_none_retrieved = res_v2_none.retrieved_pairs

    taxonomy = diagnose_residual(
        val_true_pairs, v2_none_retrieved, query_map, target_map, rng_seed=seed)
    print(f"  Unrecovered: {taxonomy['n_unrecovered']}", flush=True)
    for cat, cnt in sorted(taxonomy["category_counts"].items()):
        print(f"    {cat}: {cnt}", flush=True)

    tax_run_id = ledger.start_run(
        config={"experiment": "residual_taxonomy", "seed": seed, "commit": commit},
        dataset="gleif_ch", seed=seed, dataset_version=DATASET_VERSION,
    )
    ledger.log_metrics(tax_run_id, {k: v for k, v in taxonomy.items() if k != "examples"})

    # Also v1 G=None residual for comparison
    bl_v1_none = base_v1.with_gate(None)
    res_v1_none = bl_v1_none.retrieve(val_query_records)
    v1_none_retrieved = res_v1_none.retrieved_pairs
    taxonomy_v1 = diagnose_residual(
        val_true_pairs, v1_none_retrieved, query_map, target_map, rng_seed=seed)

    # ── 5. No-match diagnosis ──
    print("\n=== No-match query diagnosis ===", flush=True)
    nm_diag = diagnose_no_match(rng_seed=seed)
    print(f"  n_no_match: {nm_diag['n_no_match']}", flush=True)
    print(f"  k1_name_collision: {nm_diag['k1_name_collision_count']} "
          f"({nm_diag['k1_name_collision_pct']:.1f}%)", flush=True)
    nm_run_id = ledger.start_run(
        config={"experiment": "no_match_diagnosis", "seed": seed, "commit": commit},
        dataset="gleif_ch", seed=seed, dataset_version=DATASET_VERSION,
    )
    ledger.log_metrics(nm_run_id, nm_diag)

    # ── 6. GLEIF sample stats ──
    print("\n=== GLEIF sample stats ===", flush=True)
    gleif_stats = gleif_sample_stats()
    print(f"  Announced: {gleif_stats['gleif_announced_record_count']:,}", flush=True)
    print(f"  In file:   {gleif_stats['gleif_file_rows_decompressed']:,}", flush=True)
    print(f"  Coverage:  {gleif_stats['gleif_coverage_pct']}%", flush=True)

    gleif_run_id = ledger.start_run(
        config={"experiment": "gleif_sample_stats", "commit": commit},
        dataset="gleif_ch", seed=-1, dataset_version=DATASET_VERSION,
    )
    ledger.log_metrics(gleif_run_id, gleif_stats)

    return {
        "commit": commit,
        "all_v1_identical": all_identical,
        "v1_sweep": v1_sweep,
        "v2_sweep": v2_sweep,
        "pk_v1_g50": pk_v1_g50,
        "pk_v1_gnone": pk_v1_gnone,
        "pk_v2_g50": pk_v2_g50,
        "pk_v2_gnone": pk_v2_gnone,
        "taxonomy_v1": taxonomy_v1,
        "taxonomy_v2": taxonomy,
        "no_match": nm_diag,
        "gleif_stats": gleif_stats,
        "n_val_true": n_val_true,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = main(seed=args.seed)

    # Save results JSON for report generation
    out_path = _REPO / "experiments" / "phase2b_results.json"
    # Exclude per-pair examples from JSON (written to report directly)
    json_safe = {k: v for k, v in results.items() if k != "taxonomy_v2"}
    json_safe["taxonomy_v2_counts"] = results["taxonomy_v2"]["category_counts"]
    json_safe["taxonomy_v2_summary"] = {k: v for k, v in results["taxonomy_v2"].items()
                                         if k != "examples"}
    out_path.write_text(json.dumps(json_safe, indent=2, default=str))
    print(f"\nResults saved to {out_path}", flush=True)
    print("Now run: python experiments/write_001b_report.py", flush=True)
