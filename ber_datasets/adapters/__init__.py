"""
datasets.adapters — Dataset Adapters
========================================
Responsibility: Translate raw dataset files (CSV, JSON, Parquet) into the
canonical BER record schema (see ber.store.schemas).

Each adapter is a thin reader that:
  1. Yields records without loading the whole file into RAM.
  2. Assigns stable dataset_id and record_id values.
  3. Preserves raw fields verbatim — no canonicalisation here.

Adapters currently registered:
    FodorsZagatsAdapter — products matching dataset stored in data/raw/Fodors-Zagats/
    GLEIFAdapter         — GLEIF LEI level-1 golden copy
    CompaniesHouseAdapter — Companies House basic company data

Usage::

    from ber_datasets.adapters import FodorsZagatsAdapter
    for record in FodorsZagatsAdapter(split='train').iter_records():
        ...
"""
