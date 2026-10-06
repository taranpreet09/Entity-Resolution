"""
ber.canon — Canonicalization Stage
=====================================
Responsibility: Transform raw field values into a normalised canonical view
(e.g. legal-form stripping, address parsing, phone E.164 normalisation).

The stage is instrumented by the Loss Ledger: any record whose raw field is
non-empty but whose canonical view is empty or unparseable is assigned a
CANONICALIZATION loss.

NOT IN SCOPE for Phase 1.  This module is a documented stub.

Public API (to be implemented):
    Canonicalizer — abstract base class
    NameCanonicalizer(Canonicalizer)
    AddressCanonicalizer(Canonicalizer)
    canonicalize_dataset(dataset_id, profile_report, output_dir) -> None
"""
