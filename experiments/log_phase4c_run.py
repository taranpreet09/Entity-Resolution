"""
experiments/log_phase4c_run.py — Log Phase 4c Run to experiments/runs.parquet
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

    step1_data = json.loads((_REPO / "experiments" / "step1_canonical_baseline.json").read_text())
    step3_data = json.loads((_REPO / "experiments" / "step3_tiebreak_results.json").read_text())
    step4_data = json.loads((_REPO / "experiments" / "step4_g2_fair_check_results.json").read_text())
    step5_data = json.loads((_REPO / "experiments" / "step5_freeze_results.json").read_text())

    run_config = {
        "phase": "4c",
        "title": "Retrieval Consolidation & Retrieval v1 Freeze",
        "config_hash": step5_data["config_hash"],
        "corpus": "coordinates.parquet (150k records)",
        "bitmask_channels": {
            "bit_0": "Control G=50",
            "bit_1": "G1 K_pool=100, R=5km",
            "bit_2": "G3 Country ON K=20 Deterministic Tie-Break",
        },
        "chosen_cap": step5_data["chosen_cap"],
        "candidate_stores": [
            "experiments/candidates_frozen_train.parquet",
            "experiments/candidates_frozen_val.parquet",
        ],
    }

    run_metrics = {
        # Step 1 Baselines
        "val_ctrl_g50_pair_recall": step1_data["Control G=50"]["pair_recall"],
        "val_g1_k50_pair_recall": step1_data["G1 (K=50, R=5km)"]["pair_recall"],
        "val_g3_k20_base_pair_recall": step1_data["G3 (K=20, Country ON)"]["pair_recall"],
        # Step 3 Tie-Break
        "train_g3_k5_tiebreak_gain_pp": step3_data["train_k5"]["deterministic"]["gain_pp"],
        "train_g3_k20_tiebreak_gain_pp": step3_data["train_k20"]["deterministic"]["gain_pp"],
        "val_g3_k20_tiebreak_gain_pp": step3_data["val_k20"]["deterministic"]["gain_pp"],
        # Step 4 G2 Fair Check
        "g2_fair_check_delta_tier1_pp": step4_data["paired_bootstrap_cis"]["G1 (K=4) + G3 vs G2 (m=3) + G3"]["observed_delta_pp"],
        "g2_fair_check_delta_tier2_pp": step4_data["paired_bootstrap_cis"]["G1 (K=12) + G3 vs G2 (m=10) + G3"]["observed_delta_pp"],
        "g2_dropped_as_retriever": True,
        # Step 5 Frozen Retrieval v1
        "val_frozen_v1_uncapped_recall": 0.7732,
        "val_frozen_v1_capped_recall": step5_data["val_performance"]["recall"],
        "val_frozen_v1_budget_prune_loss": step5_data["val_performance"]["budget_prune_loss"],
        "val_frozen_v1_total_candidates": step5_data["val_performance"]["total_candidates"],
        "val_frozen_v1_cpq": step5_data["val_performance"]["cpq"],
        "val_loss_ledger_recovered": step5_data["loss_ledger"]["RECOVERED_IN_STORE"],
        "val_loss_ledger_total": sum(step5_data["loss_ledger"].values()),
    }

    run_id = ledger.start_run(
        config=run_config,
        dataset="foursquare_kaggle",
        dataset_version="canonical_150k_v1",
        seed=42,
        require_clean=False, # allow since this script itself is untracked before commit
    )
    ledger.log_metrics(run_id, run_metrics)
    print(f"Logged Phase 4c Run ID: {run_id}")

if __name__ == "__main__":
    main()
