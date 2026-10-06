"""
ber.features — Feature Extraction Stage
==========================================
Responsibility: Compute pairwise similarity features for (query, candidate)
pairs produced by the retrieval stage.  Features feed the ranker.

The Loss Ledger assigns a FEATURES loss if feature computation fails for a pair
that was present in the candidate set.

NOT IN SCOPE for Phase 1.  This module is a documented stub.

Public API (to be implemented):
    FeatureExtractor — abstract base class
    NameSimilarityFeatures(FeatureExtractor)
    AddressSimilarityFeatures(FeatureExtractor)
    extract_features(candidates_table, config) -> FeaturesTable
"""
