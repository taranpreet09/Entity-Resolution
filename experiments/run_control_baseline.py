"""
experiments/run_control_baseline.py — Control Baseline Pipeline
================================================================
Single-command entry point that:

  1. Ingests GLEIF × Companies House via the adapter (cached: skips if
     data/processed/gleif_ch/ already exists and --force is not set).
  2. Runs control canonicalisation and reports empty-after-canon counts.
  3. Builds entity-disjoint train/val/test splits (queries only).
  4. Builds the blocking index ONCE from the raw CH CSV via DuckDB
     (streams only CompanyName + PostCode — ~300 MB, not ~13 GB).
  5. Sweeps gate size G ∈ {10, 50, 200, 1000, None} by sharing the same
     index and varying only the gate at query time via with_gate().
  6. Evaluates on the VALIDATION split only; test split is never touched.
  7. Runs the loss ledger with full detail at G=50 and G=None.
  8. Logs all runs to the experiment ledger.
  9. Saves a control_curve.csv and per-G candidate Parquet files.

Usage::

    python experiments/run_control_baseline.py [--force-ingest] [--seed 42]

Outputs:
    data/processed/gleif_ch/         — ingested Parquet artefacts (cached)
    experiments/<run_id>/            — per-G candidates + loss ledger CSVs
    experiments/runs.parquet         — experiment ledger
    experiments/control_curve.csv    — recall vs candidates table
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import pyarrow.parquet as pq

from ber.canon.normalise import canonicalise, canon_stats
from ber.eval.ledger import RunLedger
from ber.eval.loss_ledger import LossLedger, LossStage
from ber.eval.metrics import (
    bootstrap_ci,
    build_candidate_map,
    build_true_targets_map,
    candidates_per_query_distribution,
    partial_recall_rate,
    recall_at_k,
    zero_candidate_rate,
)
from ber.retrieve.control_blocking import BlockingConfig, ControlBlocker
from ber.store.schemas import write_candidates
from ber_datasets.adapters.gleif_ch import GleifChAdapter, assert_no_identifier_leakage
from ber_datasets.splits.splitter import EntityDisjointSplitter

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROCESSED_DIR = _REPO / "data" / "processed" / "gleif_ch"
CH_CSV = (
    _REPO / "ber" / "data" / "raw" / "companies_house"
    / "BasicCompanyDataAsOneFile-2026-10-01.csv"
)
LEDGER_PATH = _REPO / "experiments" / "runs.parquet"
CURVE_PATH = _REPO / "experiments" / "control_curve.csv"

DATASET_VERSION = "gleif_2026-10-05_ch_2026-10-01"

# G sweep — the index is built once; each G is a cheap with_gate() call
GATE_SWEEP = [10, 50, 200, 1000, None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairs_from_parquet(pairs_path: Path) -> list[tuple[str, str]]:
    tbl = pq.read_table(pairs_path, columns=["query_id", "cand_id", "label"])
    return [
        (q, c)
        for q, c, lbl in zip(
            tbl.column("query_id").to_pylist(),
            tbl.column("cand_id").to_pylist(),
            tbl.column("label").to_pylist(),
        )
        if lbl == 1
    ]


def _fmt(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Step 1: Ingest (cached)
# ---------------------------------------------------------------------------

def step_ingest(force: bool = False) -> dict:
    queries_path = PROCESSED_DIR / "queries.parquet"
    if queries_path.exists() and not force:
        print("[ingest] Using cached data in data/processed/gleif_ch/", flush=True)
        stats_path = PROCESSED_DIR / "ingest_stats.json"
        if stats_path.exists():
            return json.loads(stats_path.read_text())
        nm = json.loads((PROCESSED_DIR / "no_match_queries.json").read_text())
        q_n = pq.read_table(PROCESSED_DIR / "queries.parquet").num_rows
        t_n = pq.read_table(PROCESSED_DIR / "targets.parquet").num_rows
        p_n = pq.read_table(PROCESSED_DIR / "pairs_labels.parquet").num_rows
        return {
            "n_gleif_gb": q_n, "n_matched_queries": q_n - len(nm),
            "n_no_match_queries": len(nm), "n_ch_targets": t_n,
            "n_positive_pairs": p_n,
        }
    adapter = GleifChAdapter()
    stats = adapter.build(PROCESSED_DIR)
    (PROCESSED_DIR / "ingest_stats.json").write_text(json.dumps(stats, indent=2))
    assert_no_identifier_leakage(
        PROCESSED_DIR / "queries.parquet",
        PROCESSED_DIR / "targets.parquet",
    )
    print("[ingest] Leakage check: PASSED", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Step 2: Canon stats
# ---------------------------------------------------------------------------

def step_canon_stats() -> dict:
    print("[canon] Computing empty-after-canonicalisation counts...", flush=True)
    q_tbl = pq.read_table(PROCESSED_DIR / "queries.parquet", columns=["raw_fields"])
    q_rows = [{"raw_fields": v} for v in q_tbl.column("raw_fields").to_pylist()]
    gleif_s = canon_stats(q_rows, fields=["LegalName", "LegalAddr_PostCode"])

    t_tbl = pq.read_table(PROCESSED_DIR / "targets.parquet", columns=["raw_fields"])
    t_rows = [{"raw_fields": v} for v in t_tbl.column("raw_fields").to_pylist()]
    ch_s = canon_stats(t_rows, fields=["CompanyName", "RegAddress.PostCode"])

    result = {"gleif": gleif_s, "ch": ch_s}
    for side, d in result.items():
        for field, counts in d.items():
            raw_ne = counts["raw_non_empty"]
            empty = counts["canon_empty_from_non_empty"]
            pct = 100 * empty / max(raw_ne, 1)
            print(f"  {side}.{field}: raw_non_empty={raw_ne:,} "
                  f"canon_empty={empty:,} ({pct:.2f}%)", flush=True)
    return result


# ---------------------------------------------------------------------------
# Step 3: Splits (cached)
# ---------------------------------------------------------------------------

def step_splits(seed: int = 42) -> tuple[dict[str, str], dict[str, str], dict]:
    primary_path = PROCESSED_DIR / "primary_splits.json"
    brand_path = PROCESSED_DIR / "brand_splits.json"
    if primary_path.exists() and brand_path.exists():
        print("[splits] Using cached splits.", flush=True)
        primary_splits = json.loads(primary_path.read_text())
        brand_splits = json.loads(brand_path.read_text())
        primary_counts = Counter(primary_splits.values())
        brand_counts = Counter(brand_splits.values())
    else:
        print("[splits] Building entity-disjoint splits...", flush=True)
        all_positive_pairs = _pairs_from_parquet(PROCESSED_DIR / "pairs_labels.parquet")
        q_tbl = pq.read_table(
            PROCESSED_DIR / "queries.parquet", columns=["record_id", "raw_fields"]
        )
        record_ids = q_tbl.column("record_id").to_pylist()
        raw_fields_list = q_tbl.column("raw_fields").to_pylist()
        pair_lookup = {q: c for q, c in all_positive_pairs}
        entity_ids = [pair_lookup.get(rid, rid) for rid in record_ids]
        norm_keys: list[tuple[str, str]] = []
        norm_names: list[str] = []
        for rf in raw_fields_list:
            try:
                fields = json.loads(rf) if isinstance(rf, str) else rf
            except Exception:
                fields = {}
            name = canonicalise(fields.get("LegalName", ""))
            addr = canonicalise(fields.get("LegalAddr_Line1", ""))
            norm_keys.append((name, addr))
            norm_names.append(name)
        splitter = EntityDisjointSplitter(
            ratios={"train": 0.70, "val": 0.15, "test": 0.15}, seed=seed
        )
        primary_splits = splitter.split(record_ids, entity_ids, norm_keys=norm_keys)
        brand_splits = splitter.brand_split(record_ids, norm_names, entity_ids)
        primary_path.write_text(json.dumps(primary_splits))
        brand_path.write_text(json.dumps(brand_splits))
        primary_counts = Counter(primary_splits.values())
        brand_counts = Counter(brand_splits.values())

    # Leakage check
    all_positive_pairs = _pairs_from_parquet(PROCESSED_DIR / "pairs_labels.parquet")
    pair_lookup = {q: c for q, c in all_positive_pairs}
    record_ids = list(primary_splits.keys())
    entity_ids = [pair_lookup.get(rid, rid) for rid in record_ids]
    entity_to_splits: dict[str, set[str]] = defaultdict(set)
    for rid, eid in zip(record_ids, entity_ids):
        entity_to_splits[eid].add(primary_splits[rid])
    n_leaking = sum(1 for s in entity_to_splits.values() if len(s) > 1)
    assert n_leaking == 0, f"Split leakage: {n_leaking} entities cross splits!"

    print(f"  Primary: train={primary_counts['train']:,} "
          f"val={primary_counts['val']:,} test={primary_counts['test']:,}", flush=True)
    print(f"  Brand:   train={brand_counts['train']:,} "
          f"val={brand_counts['val']:,} test={brand_counts['test']:,}", flush=True)
    print(f"  Leakage check: 0 entities cross splits (PASS)", flush=True)

    return primary_splits, brand_splits, {
        "primary_split_counts": dict(primary_counts),
        "brand_split_counts": dict(brand_counts),
        "n_leaking_entities": n_leaking,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Step 4: Build index ONCE from raw CH CSV (streaming, low memory)
# ---------------------------------------------------------------------------

def step_build_base_index() -> ControlBlocker:
    """Stream CompanyName+PostCode from CH CSV via DuckDB. ~300 MB, not ~13 GB."""
    # gate_size=None here — gate is applied per-query via with_gate()
    base_config = BlockingConfig(gate_size=None, keys=["K1", "K2", "K3", "K4"])
    blocker = ControlBlocker(base_config)
    blocker.index_targets_from_duckdb(CH_CSV, batch_size=250_000, verbose=True)
    return blocker


# ---------------------------------------------------------------------------
# Step 5: Load val queries (small — 11,682 records)
# ---------------------------------------------------------------------------

def step_load_val_queries(primary_splits: dict[str, str]) -> list[dict]:
    q_tbl = pq.read_table(
        PROCESSED_DIR / "queries.parquet", columns=["record_id", "raw_fields"]
    )
    return [
        {"record_id": rid, "raw_fields": rf}
        for rid, rf in zip(
            q_tbl.column("record_id").to_pylist(),
            q_tbl.column("raw_fields").to_pylist(),
        )
        if primary_splits.get(rid) == "val"
    ]


# ---------------------------------------------------------------------------
# Step 6: Evaluate one G setting
# ---------------------------------------------------------------------------

def step_evaluate(
    blocker_g: ControlBlocker,
    val_query_records: list[dict],
    val_true_pairs: list[tuple[str, str]],
    gate_size: int | None,
    run_dir: Path,
    empty_name_qids: set[str],
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    result = blocker_g.retrieve(val_query_records)
    candidates = result.candidates
    size_gated_pairs = result.size_gated_pairs
    retrieved_pairs = result.retrieved_pairs
    blocker_stats = result.stats

    candidate_map = build_candidate_map(candidates)
    true_targets = build_true_targets_map(
        [{"query_id": q, "cand_id": c, "label": 1} for q, c in val_true_pairs]
    )
    max_rank = max((c["rank"] for c in candidates), default=0)
    k_budget = max(max_rank, 1)

    r_at_budget = recall_at_k(candidate_map, true_targets, k=k_budget)
    z_rate = zero_candidate_rate(candidate_map, true_targets)
    p_rate = partial_recall_rate(candidate_map, true_targets, k=k_budget)
    cpq = candidates_per_query_distribution(candidate_map)
    metric_fn = functools.partial(recall_at_k, k=k_budget)
    pt, lo, hi = bootstrap_ci(
        candidate_map, true_targets, metric_fn, n_boot=n_boot, seed=seed
    )

    # Loss ledger
    ll = LossLedger(default_loss=LossStage.RETRIEVAL)
    ll.add_true_pairs(val_true_pairs)
    for q, c in val_true_pairs:
        if q in empty_name_qids:
            ll.mark_input_loss(q, c, reason="canonical LegalName is empty")
    ll.apply_candidates(retrieved_pairs)
    for q, c in val_true_pairs:
        if (q, c) in size_gated_pairs:
            rec = ll._pairs.get((q, c))
            if rec and rec.loss_stage == LossStage.RETRIEVAL:
                ll.mark_retrieval_loss(q, c, subtype="size_gated",
                                       reason=f"bucket > G={gate_size}")
    for q, c in val_true_pairs:
        if (q, c) in retrieved_pairs:
            ll.mark_recovered(q, c)

    stage_counts = ll.stage_counts()
    ll_path = run_dir / f"loss_ledger_G{gate_size}.csv"
    ll.to_csv(ll_path)

    n_size_gated = sum(
        1 for rec in ll._pairs.values()
        if rec.loss_stage == LossStage.RETRIEVAL and rec.subtype == "size_gated"
    )
    n_no_key = sum(
        1 for rec in ll._pairs.values()
        if rec.loss_stage == LossStage.RETRIEVAL and rec.subtype != "size_gated"
    )

    print(
        f"  G={str(gate_size):<5} recall={r_at_budget:.4f} "
        f"[{lo:.4f},{hi:.4f}]  cands={len(candidates):,}  "
        f"zero={z_rate:.4f}  INPUT={stage_counts['INPUT']}  "
        f"gated={n_size_gated}  no_key={n_no_key}",
        flush=True,
    )

    # Write candidate Parquet
    if candidates:
        write_candidates(candidates, run_dir / f"candidates_G{gate_size}.parquet")

    return {
        "gate_size": gate_size,
        "n_val_queries_with_targets": len(true_targets),
        "n_total_candidates": len(candidates),
        "recall_at_budget": round(r_at_budget, 6) if r_at_budget == r_at_budget else None,
        "recall_ci_lo": round(lo, 6) if lo == lo else None,
        "recall_ci_hi": round(hi, 6) if hi == hi else None,
        "zero_candidate_rate": round(z_rate, 6) if z_rate == z_rate else None,
        "partial_recall_rate": round(p_rate, 6) if p_rate == p_rate else None,
        "candidates_per_query_median": cpq["median"],
        "candidates_per_query_p95": cpq["p95"],
        "candidates_per_query_mean": cpq["mean"],
        "loss_stage_counts": stage_counts,
        "n_size_gated_true_pairs": n_size_gated,
        "n_no_shared_key_true_pairs": n_no_key,
        "per_key_stats": blocker_stats.get("per_key", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force_ingest: bool = False, seed: int = 42) -> dict:
    ledger = RunLedger(LEDGER_PATH)

    # 1. Ingest
    print("\n=== Step 1: Ingest ===", flush=True)
    ingest_stats = step_ingest(force=force_ingest)
    print(f"  {ingest_stats}", flush=True)

    # 2. Canon stats
    print("\n=== Step 2: Canon stats ===", flush=True)
    canon_result = step_canon_stats()

    # 3. Splits
    print("\n=== Step 3: Splits ===", flush=True)
    primary_splits, brand_splits, split_stats = step_splits(seed=seed)

    all_positive_pairs = _pairs_from_parquet(PROCESSED_DIR / "pairs_labels.parquet")
    val_true_pairs = [(q, c) for q, c in all_positive_pairs
                     if primary_splits.get(q) == "val"]
    print(f"  Val true pairs: {len(val_true_pairs):,}", flush=True)

    # 4. Build index ONCE (DuckDB streaming — low memory)
    print("\n=== Step 4: Build blocking index (DuckDB stream) ===", flush=True)
    base_blocker = step_build_base_index()
    idx_stats = base_blocker.index_stats()
    print(f"  Index stats: K1 buckets={idx_stats['K1']['n_buckets']:,} "
          f"K2={idx_stats['K2']['n_buckets']:,} "
          f"K3={idx_stats['K3']['n_buckets']:,} "
          f"K4={idx_stats['K4']['n_buckets']:,}", flush=True)

    # 5. Load val queries (11,682 records — fits easily in RAM)
    val_query_records = step_load_val_queries(primary_splits)
    print(f"  Val queries loaded: {len(val_query_records):,}", flush=True)

    # Pre-compute empty-name queries (used in loss ledger for every G)
    empty_name_qids: set[str] = set()
    for rec in val_query_records:
        try:
            fields = json.loads(rec["raw_fields"]) if isinstance(rec["raw_fields"], str) \
                else rec["raw_fields"]
        except Exception:
            fields = {}
        if not canonicalise(fields.get("LegalName", "")):
            empty_name_qids.add(rec["record_id"])

    # 6. G sweep
    print("\n=== Step 5: G sweep ===", flush=True)
    config_base = {
        "experiment": "control_baseline_g_sweep",
        "keys": ["K1", "K2", "K3", "K4"],
        "dataset": "gleif_ch",
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "split": "val",
        "n_val_true_pairs": len(val_true_pairs),
        "index_stats": {k: {"n_buckets": v.get("n_buckets", 0)}
                        for k, v in idx_stats.items()},
    }
    sweep_run_id = ledger.start_run(
        config={**config_base, "gate_sweep": [str(g) for g in GATE_SWEEP]},
        dataset="gleif_ch",
        seed=seed,
        dataset_version=DATASET_VERSION,
    )
    sweep_dir = _REPO / "experiments" / sweep_run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_gate_metrics: list[dict] = []
    for gate_size in GATE_SWEEP:
        # Reuse same index, just swap the gate — no re-indexing
        blocker_g = base_blocker.with_gate(gate_size)
        g_metrics = step_evaluate(
            blocker_g, val_query_records, val_true_pairs,
            gate_size, sweep_dir, empty_name_qids,
            n_boot=500, seed=seed,
        )
        all_gate_metrics.append(g_metrics)

    # 7. Log to ledger
    ledger.log_metrics(sweep_run_id, {
        "gate_sweep_results": all_gate_metrics,
        "ingest_stats": ingest_stats,
        "split_stats": split_stats,
        "canon_stats": {
            "gleif_LegalName_canon_empty_from_non_empty":
                canon_result["gleif"]["LegalName"]["canon_empty_from_non_empty"],
            "gleif_LegalName_raw_non_empty":
                canon_result["gleif"]["LegalName"]["raw_non_empty"],
            "ch_CompanyName_canon_empty_from_non_empty":
                canon_result["ch"]["CompanyName"]["canon_empty_from_non_empty"],
        },
    })

    # 8. Control curve CSV
    print(f"\n=== Saving control_curve.csv ===", flush=True)
    with open(CURVE_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "gate_size", "n_total_candidates", "candidates_per_query_median",
            "candidates_per_query_p95", "recall_at_budget",
            "recall_ci_lo", "recall_ci_hi",
            "zero_candidate_rate", "partial_recall_rate",
            "n_size_gated_true_pairs", "n_no_shared_key_true_pairs",
        ])
        writer.writeheader()
        for m in all_gate_metrics:
            writer.writerow({
                "gate_size": m["gate_size"],
                "n_total_candidates": m["n_total_candidates"],
                "candidates_per_query_median": _fmt(m["candidates_per_query_median"]),
                "candidates_per_query_p95": _fmt(m["candidates_per_query_p95"]),
                "recall_at_budget": _fmt(m["recall_at_budget"]),
                "recall_ci_lo": _fmt(m["recall_ci_lo"]),
                "recall_ci_hi": _fmt(m["recall_ci_hi"]),
                "zero_candidate_rate": _fmt(m["zero_candidate_rate"]),
                "partial_recall_rate": _fmt(m["partial_recall_rate"]),
                "n_size_gated_true_pairs": m["n_size_gated_true_pairs"],
                "n_no_shared_key_true_pairs": m["n_no_shared_key_true_pairs"],
            })

    print(f"\n=== Done. sweep_run_id={sweep_run_id} ===", flush=True)
    return {
        "sweep_run_id": sweep_run_id,
        "all_gate_metrics": all_gate_metrics,
        "ingest_stats": ingest_stats,
        "split_stats": split_stats,
        "canon_stats": canon_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BER control baseline pipeline.")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(force_ingest=args.force_ingest, seed=args.seed)
