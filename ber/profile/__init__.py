"""
ber.profile — Data Profiling Stage
====================================
Responsibility: Scan raw records and produce per-field statistics (null rates,
cardinality, format distributions, length histograms).  Profiling results are
stored in Parquet and used downstream to decide canonicalization strategies and
to assign INPUT-stage losses when fields are entirely missing or unparseable.

NOT IN SCOPE for Phase 1.  This module is a documented stub.

Public API (to be implemented):
    profile_dataset(dataset_id, source_path, output_dir) -> ProfileReport
    ProfileReport.null_rate(field) -> float
    ProfileReport.is_parseable(record_id, field) -> bool
"""
