"""
ber.index — Index Construction Stage
=======================================
Responsibility: Build offline lookup structures (inverted index, ANN index,
prefix trees) from the canonical representation of the *target* corpus.

Each index type is a separate channel.  Channels are registered before an
experiment starts so the Loss Ledger can enumerate them.

NOT IN SCOPE for Phase 1 (no retrieval channels are implemented yet).
This module is a documented stub.

Public API (to be implemented):
    IndexChannel — abstract base class with .build(records_parquet) and
                   .retrieve(query, k, score_floor) -> List[Candidate]
    ExactNameIndex(IndexChannel)
    TrigramIndex(IndexChannel)
    register_channel(name, channel_cls) -> None
"""
