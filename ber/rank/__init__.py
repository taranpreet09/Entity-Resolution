"""
ber.rank — Ranking Stage
==========================
Responsibility: Score each (query, candidate) pair using a trained model or
heuristic and assign a rank within the candidate set for each query.

The Loss Ledger assigns a RANKER loss when a true pair is present in the
candidate set but ranked below the threshold after scoring.

NOT IN SCOPE for Phase 1.  This module is a documented stub.

Public API (to be implemented):
    Ranker — abstract base class
    ExactMatchRanker(Ranker) — trivial implementation used in sanity run
    train_ranker(features_table, labels_table, config) -> Ranker
    rank_candidates(features_table, ranker) -> RankedCandidatesTable
"""
