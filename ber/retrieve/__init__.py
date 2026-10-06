"""
ber.retrieve — Multi-Channel Retrieval Stage
===============================================
Responsibility: Execute all registered retrieval channels for each query record,
enforce per-channel and global candidate budgets, union results across channels,
and record provenance (which channels contributed each candidate).

Budget enforcement is BY CONSTRUCTION: no channel can produce more than its
allowed quota before the union step.

NOT IN SCOPE for Phase 1 (no channels implemented).  This module is a
documented stub.

Public API (to be implemented):
    RetrievalConfig — dataclass with channel_budgets: Dict[str, int],
                      global_budget: int, score_floor: float
    retrieve_batch(queries, channels, config, ledger) -> CandidatesTable
    CandidatesTable — wraps the candidates Parquet schema (see ber.store)
"""
