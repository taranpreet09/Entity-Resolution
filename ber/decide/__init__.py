"""
ber.decide — Entity Decision Stage
=====================================
Responsibility: Collapse ranked candidates into entity-level decisions:
is this query record the same entity as this candidate?  Applies a calibrated
probability threshold and produces the final entity cluster assignments.

The Loss Ledger assigns a DECISION loss when a true pair survives ranking but
is blocked by the final threshold.

NOT IN SCOPE for Phase 1.  This module is a documented stub.

Public API (to be implemented):
    DecisionConfig — threshold, min_score, max_cluster_size
    make_decisions(ranked_candidates, config, calibrator) -> EntityClusters
    EntityClusters — maps record_id -> entity_id
"""
