"""
scratch/log_phase4b_run.py — Log Phase 4b runs to experiments/runs.parquet
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from ber.eval.ledger import RunLedger

def main():
    ledger = RunLedger(str(_REPO / "experiments" / "runs.parquet"))
    real_commit = "d29a52009ddc75311311ad1648289033ffd36a24"

    step2_data = json.loads((_REPO / "experiments" / "phase4b_step2_sweep.json").read_text())
    step4_data = json.loads((_REPO / "experiments" / "phase4b_step4_val_frontier.json").read_text())
    step6_data = json.loads((_REPO / "experiments" / "step6_g2_cost_results.json").read_text())

    run_config = {
        "phase": "4b",
        "title": "Retrieval Consolidation & Channel G2 Cost Experiment",
        "g3_chosen_k": 5,
        "g3_chosen_country_filter": True,
        "g2_pool_depth": 200,
        "g2_radius_m": 5000.0,
        "g2_rerank_weights": [2.0, 1.5, 1.0, 1.0, 0.5, 0.5],
        "g2_chosen_m": 5,
    }

    run_metrics = {
        "train_g3_k5_cf_on_pair_recall": step2_data["sweep_results"]["K=5_cf_on"]["train"]["pair_recall"],
        "train_g3_k5_cf_on_marginal_yield": step2_data["sweep_results"]["K=5_cf_on"]["train"]["marginal_yield"],
        "val_g1_k50_pair_recall": step4_data["systems"]["g1_k50"]["pair_recall"],
        "val_g3_k5_pair_recall": step4_data["systems"]["g3_chosen"]["pair_recall"],
        "val_union_g1_g3_pair_recall": step4_data["systems"]["union_g1_g3"]["pair_recall"],
        "val_union_ctrl_g1_g3_pair_recall": step4_data["systems"]["union_ctrl_g1_g3"]["pair_recall"],
        "val_union_gain_over_g1_pp": step4_data["paired_bootstrap"]["union_g1_g3_vs_g1"]["delta_pp"],
        "val_loss_ledger_assertion": step4_data["assertion_passed"],
        "g2_cost_experiment_val_delta_pp": step6_data["val_eval"]["delta_pp"],
        "g2_cost_experiment_success": step6_data["val_eval"]["success"],
    }

    run_id = ledger.start_run(
        config=run_config,
        dataset="foursquare",
        dataset_version="foursquare_kaggle_train_2022",
        seed=42,
    )
    ledger.log_metrics(run_id, run_metrics)
    print(f"Logged Phase 4b Run ID: {run_id}")

if __name__ == "__main__":
    main()

