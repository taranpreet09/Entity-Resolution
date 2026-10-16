"""
experiments/sanity_exact_name.py — Sanity Run: Exact-Name Matcher on Fodors-Zagats
====================================================================================
Proves the harness (ledger, metrics, loss ledger) runs end-to-end using a
trivial exact-title matcher on the Fodors-Zagats products dataset.

The matcher: for each query record from tableA, retrieve all tableB records
whose ``title`` field exactly matches (case-insensitively, stripped).

**IMPORTANT NOTE:** This experiment cannot measure retrieval recall in any
meaningful sense.  The Fodors-Zagats pair labels were produced by a blocking
step — they are a subset of true pairs, not a random sample.  An exact-name
matcher that retrieves 100 % of blocked pairs still tells us nothing about how
many true pairs exist outside the blocking window.  See ADR-004.

Usage::

    python experiments/sanity_exact_name.py [--seed SEED] [--k K]

Outputs:
    - A run entry in experiments/runs.parquet
    - A candidates Parquet file in experiments/<run_id>/candidates.parquet
    - A loss ledger CSV in experiments/<run_id>/loss_ledger.csv
    - Metrics printed to stdout and stored in the ledger
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ber.eval.ledger import RunLedger
from ber.eval.loss_ledger import LossLedger, LossStage
from ber.eval.metrics import (
    build_candidate_map,
    build_true_targets_map,
    candidates_per_query_distribution,
    partial_recall_rate,
    recall_at_k,
    zero_candidate_rate,
    bootstrap_ci,
)
from ber.store.schemas import write_candidates
from ber_datasets.adapters.fodors_zagats import iter_records, iter_pairs, DATASET_ID

LEDGER_PATH = Path(__file__).resolve().parent / "runs.parquet"


def run_experiment(seed: int = 42, k: int = 10) -> dict:
    """Run the sanity experiment and return the metrics dict."""

    # ------------------------------------------------------------------
    # 0. Pre-register in ledger
    # ------------------------------------------------------------------
    config = {
        "experiment": "sanity_exact_name",
        "matcher": "exact_title_case_insensitive",
        "dataset": DATASET_ID,
        "k": k,
        "seed": seed,
        "note": (
            "Trivial exact-name matcher on Fodors-Zagats products dataset. "
            "Pairs were derived from blocking; recall figures are post-blocking only."
        ),
    }
    ledger = RunLedger(LEDGER_PATH)
    run_id = ledger.start_run(
        config=config,
        dataset=DATASET_ID,
        seed=seed,
        dataset_version="2026-10-01",
    )
    print(f"Run ID: {run_id}", flush=True)
    run_dir = Path(__file__).resolve().parent / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load records from both tables
    # ------------------------------------------------------------------
    print("Loading records...", flush=True)
    table_a: dict[str, str] = {}  # record_id -> title (normalised)
    table_b: dict[str, str] = {}

    for rec in iter_records():
        raw = json.loads(rec["raw_fields"])
        title = raw.get("title", "").strip().strip('"').lower()
        if rec["record_id"].startswith("A_"):
            table_a[rec["record_id"]] = title
        else:
            table_b[rec["record_id"]] = title

    print(f"  tableA: {len(table_a):,} records", flush=True)
    print(f"  tableB: {len(table_b):,} records", flush=True)

    # ------------------------------------------------------------------
    # 2. Build exact-name index: title -> list of B record_ids
    # ------------------------------------------------------------------
    title_to_b: dict[str, list[str]] = {}
    for rid, title in table_b.items():
        if title:
            title_to_b.setdefault(title, []).append(rid)

    # ------------------------------------------------------------------
    # 3. Retrieve candidates
    # ------------------------------------------------------------------
    print("Retrieving candidates (exact name)...", flush=True)
    cand_rows: list[dict] = []
    for a_rid, a_title in table_a.items():
        if not a_title:
            continue
        matches = title_to_b.get(a_title, [])
        for rank, b_rid in enumerate(matches[:k], start=1):
            cand_rows.append({
                "query_id": a_rid,
                "cand_id": b_rid,
                "channel_mask": 1,   # channel 0 = exact_name
                "score": 1.0,
                "rank": rank,
                "reverse_rank": 1,
                "round": 1,
                "dataset_id": DATASET_ID,
            })

    print(f"  {len(cand_rows):,} candidate pairs retrieved", flush=True)

    # ------------------------------------------------------------------
    # 4. Load ground-truth pairs (all splits)
    # ------------------------------------------------------------------
    print("Loading ground-truth pairs...", flush=True)
    all_pairs: list[dict] = []
    for split in ("train", "val", "test"):
        try:
            all_pairs.extend(iter_pairs(split=split))
        except FileNotFoundError:
            pass

    positive_pairs = [(p["query_id"], p["cand_id"]) for p in all_pairs if p["label"] == 1]
    print(f"  {len(positive_pairs):,} true positive pairs", flush=True)

    # ------------------------------------------------------------------
    # 5. Build evaluation maps
    # ------------------------------------------------------------------
    candidate_map = build_candidate_map(cand_rows)
    true_targets = build_true_targets_map(
        [{"query_id": q, "cand_id": c, "label": 1} for q, c in positive_pairs]
    )

    # ------------------------------------------------------------------
    # 6. Compute metrics
    # ------------------------------------------------------------------
    print("Computing metrics...", flush=True)

    import functools
    r_at_k = recall_at_k(candidate_map, true_targets, k=k)
    r_at_budget = recall_at_k(candidate_map, true_targets, k=len(table_b))
    z_rate = zero_candidate_rate(candidate_map, true_targets)
    p_rate = partial_recall_rate(candidate_map, true_targets, k=k)
    cpq_dist = candidates_per_query_distribution(candidate_map)

    # Bootstrap CI for recall@k
    metric_fn = functools.partial(recall_at_k, k=k)
    pt, lo, hi = bootstrap_ci(
        candidate_map, true_targets, metric_fn, n_boot=1000, seed=seed
    )

    metrics = {
        f"recall@{k}": round(r_at_k, 6) if r_at_k == r_at_k else None,
        "recall@budget": round(r_at_budget, 6) if r_at_budget == r_at_budget else None,
        "zero_candidate_rate": round(z_rate, 6) if z_rate == z_rate else None,
        f"partial_recall_rate@{k}": round(p_rate, 6) if p_rate == p_rate else None,
        f"recall@{k}_ci_lo": round(lo, 6) if lo == lo else None,
        f"recall@{k}_ci_hi": round(hi, 6) if hi == hi else None,
        "candidates_per_query_median": cpq_dist["median"],
        "candidates_per_query_p95": cpq_dist["p95"],
        "n_queries_with_targets": len(true_targets),
        "n_candidates": len(cand_rows),
        "note": (
            "POST-BLOCKING RECALL ONLY. Pairs from blocking step. "
            "This figure does not measure retrieval recall on all true pairs."
        ),
    }

    # ------------------------------------------------------------------
    # 7. Build loss ledger
    # ------------------------------------------------------------------
    ll = LossLedger(default_loss=LossStage.RETRIEVAL)
    ll.add_true_pairs(positive_pairs)

    retrieved_set = {(r["query_id"], r["cand_id"]) for r in cand_rows}
    ll.apply_candidates(retrieved_set)

    # Mark recovered pairs
    for key in retrieved_set:
        if key in ll._pairs:
            ll.mark_recovered(key[0], key[1])

    stage_counts = ll.stage_counts()
    metrics["loss_stage_counts"] = stage_counts

    ll_path = run_dir / "loss_ledger.csv"
    ll.to_csv(ll_path)
    print(f"  Loss ledger written: {ll_path}", flush=True)

    # ------------------------------------------------------------------
    # 8. Write candidates Parquet
    # ------------------------------------------------------------------
    cands_path = run_dir / "candidates.parquet"
    write_candidates(cand_rows, cands_path)
    print(f"  Candidates written: {cands_path}", flush=True)

    # ------------------------------------------------------------------
    # 9. Log metrics to ledger
    # ------------------------------------------------------------------
    ledger.log_metrics(run_id, metrics)

    # ------------------------------------------------------------------
    # 10. Print summary
    # ------------------------------------------------------------------
    print("\n=== Sanity Run Results ===")
    print(f"  recall@{k}          : {r_at_k:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  recall@budget       : {r_at_budget:.4f}")
    print(f"  zero_candidate_rate : {z_rate:.4f}")
    print(f"  partial_recall@{k}  : {p_rate:.4f}")
    print(f"  candidates/query p50: {cpq_dist['median']:.1f}")
    print(f"  Loss stage counts   : {stage_counts}")
    print()
    print(
        "NOTE: These figures reflect POST-BLOCKING recall only.\n"
        "      The Fodors-Zagats pairs came from a blocking step.\n"
        "      Retrieval recall on all true pairs cannot be measured here."
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="BER sanity run: exact-name matcher.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=int, default=10, help="Recall@K cutoff.")
    args = parser.parse_args()
    run_experiment(seed=args.seed, k=args.k)


if __name__ == "__main__":
    main()
