"""
datasets.splits — Entity-Disjoint Splitting
=============================================
Provides a deterministic, seeded splitter that guarantees:
  - No entity (cluster) straddles train / validation / test splits.
  - No near-duplicate group (normalised name+address hash) straddles splits.
  - Optional secondary split grouped by normalised-name cluster, to measure
    brand memorisation separately.

See datasets/splits/splitter.py for the implementation.
"""
from datasets.splits.splitter import EntityDisjointSplitter  # noqa: F401
