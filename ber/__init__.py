"""
BER — Business Entity Resolution
==================================
Dataset-agnostic, out-of-core pipeline for resolving business entities across
heterogeneous data sources.

Pipeline stages (in order):
    Profile → Canonicalize → Index → Retrieve → Union/Provenance →
    Rank → Calibrate → Entity decision

Every stage is instrumented by a shared LOSS LEDGER so no true pair is ever
silently dropped — it is always assigned to the earliest stage that lost it.
"""
