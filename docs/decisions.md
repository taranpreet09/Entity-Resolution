# Architecture Decisions

## ADR-001: Storage Engine — DuckDB + Parquet

**Date:** 2026-10-06  
**Status:** Accepted

### Context

The pipeline must operate out-of-core on files that exceed available RAM
(Companies House CSV: ~2.7 GB; GLEIF golden copy: ~484 MB compressed /
~3.5 GB uncompressed; full candidate tables can grow to 10s of GB).
Two options were evaluated:

| Criterion | Polars (lazy) | DuckDB + Parquet |
|---|---|---|
| Out-of-core queries | ✓ (streaming scan) | ✓ (native, proven at TB scale) |
| SQL surface area | ✗ (DSL only) | ✓ (full SQL) |
| Multi-file aggregation | Partial | ✓ (glob patterns, hive partitioning) |
| Persistent store | ✗ (ephemeral) | ✓ (`.duckdb` or Parquet) |
| Append-only ledger | Manual | ✓ (INSERT INTO parquet via COPY) |
| Joins across datasets | Awkward | ✓ (natural SQL JOIN) |
| Python ecosystem | Good | Good (Python API) |
| Parquet write | ✓ (fast, native) | ✓ (via pyarrow or COPY TO) |

**Decision:** Use **DuckDB** as the query engine and **Parquet** as the
persistent file format.  Polars is used *only* where its lazy API provides
a convenient iterator over row-groups for Python-side processing; it never
replaces DuckDB for cross-dataset joins or aggregations.

### Rationale

1. The experiment ledger needs append-only writes and SQL queries across
   many runs — DuckDB handles this without a separate database server.
2. The GLEIF × Companies House join is a straightforward SQL operation;
   implementing it in Polars DSL would be less readable and harder to audit.
3. Parquet is columnar and splittable; every stage reads only the columns it
   needs, keeping RAM usage proportional to *columns used*, not file size.
4. DuckDB's `read_csv_auto` with `buffer_size` lets us scan the 2.7 GB
   Companies House file without materialising it — we never call `pd.read_csv`
   on large files.

### Consequences

- All schema files use PyArrow schemas for portability (no DuckDB-specific types).
- Writers use `pyarrow.parquet.write_table`; readers use DuckDB `read_parquet`.
- Tests may use in-memory DuckDB (`:memory:`) for speed.

---

## ADR-002: GLEIF File Handling

**Date:** 2026-10-06  
**Status:** Accepted

The shipped file `gleif_level1.csv` has PK/Zip magic bytes but is a
**truncated Zip64 archive** — the end-of-central-directory record is absent
(no `PK\x05\x06` signature found).  `zipfile.is_zipfile()` therefore returns
`False`.  The compressed payload is nonetheless a valid deflate stream and can
be decompressed with `zlib.decompressobj(-15)` (raw deflate).

The file covers **~1,524,435 rows** out of the 3,451,660 announced in
`latest_lei.json` for the 2026-10-05 golden copy — approximately 44 % of the
full dataset.  All GLEIF × CH counts reported in `docs/data_verification.md`
are therefore lower bounds and are labelled "partial file" accordingly.

---

## ADR-003: GB Registration Authority Codes

**Date:** 2026-10-06  
**Status:** Accepted — confirmed from GLEIF data + external LEI lookups

The following RA codes were **confirmed from the actual GLEIF data file** (not
guessed):

| RA Code | Authority | Sample entity IDs |
|---|---|---|
| RA000585 | Companies House — England & Wales | `03909510`, `06345337` |
| RA000586 | Companies House — Northern Ireland | `NI006176`, `NI632198` |
| RA000587 | Companies House — Scotland | `SC286832`, `SL006108` |

No other RA codes are used for GB Companies House matching.  Records with
`Entity.LegalJurisdiction` starting with `GB` but carrying a different RA code
are valid GB entities but registered via a non-CH authority (e.g. FCA, PRA,
charity registers) and are *not* expected to match Companies House numbers.

---

## ADR-004: "Fodors-Zagats" Dataset Naming

**Date:** 2026-10-06  
**Status:** Noted

The folder `data/raw/Fodors-Zagats/` contains a **product-matching dataset**
(columns: `id, title, modelno, price, shipweight, brand, dimensions`) — not
a restaurant dataset.  The folder name appears to be a mis-labelling in the
data drop.  The system treats it as-is; the dataset ID used throughout is
`fodors_zagats` (preserving the supplied name) and the domain is noted in
adapters.

The train/valid/test split labels supplied with the dataset came out of a
blocking step — they are **not** a random sample of all positive pairs.
Therefore recall measurements on this dataset say nothing about retrieval
recall; this is documented in every experiment that uses it.

---

## ADR-005: Splitter Design

**Date:** 2026-10-06  
**Status:** Accepted

Entity-disjoint splitting proceeds in two passes:

1. **Cluster-level assignment:** connected components of the match graph are
   assigned to splits as atomic units.  A component is never split across
   train/val/test.
2. **Duplicate-group guard:** records sharing the same `(normalised_name,
   normalised_address)` hash are grouped; the whole group moves with its
   cluster.  This prevents a near-duplicate from appearing in a different split
   from its near-twin.

Seeding uses Python's `random.seed(seed)` applied to a shuffled list of
component IDs, giving a deterministic mapping for any given seed.

A secondary `brand_split` mode groups records by normalised-name cluster
(ignoring address) to measure brand memorisation separately.

---

## ADR-006: Experiment Pre-registration

**Date:** 2026-10-06  
**Status:** Accepted

**No recall or F-score claim may be published without a corresponding ledger
entry.**  Every call to `start_run()` in `ber.eval.ledger` records a
`config_hash` (SHA-256 of the serialised config), `git_commit`, `seed`, and
`dataset+version` before any metrics are logged.  This prevents post-hoc
cherry-picking of results.

---

## ADR-007: Canonical Dev Corpus for Foursquare Deduplication

**Date:** 2026-10-12  
**Status:** Accepted

### Context

Experiments 001–005 and initial validation runs evaluated recall against the full 1.14M-row Kaggle Foursquare deduplication corpus (`data/processed/foursquare/train.csv`, true-pair denominator $123{,}067$ pairs / $169{,}452$ val queries). That full processed file was an ephemeral artifact generated during earlier phases and is no longer preserved on disk.

The preserved raw coordinate dataset is `ber/data/raw/foursquare_kaggle/coordinates.parquet` (150,000 records, columns: `id`, `name`, `latitude`, `longitude`, `address`, `city`, `state`, `zip`, `country`, `url`, `phone`, `categories`, `point_of_interest`).

Using the deterministic entity-disjoint splitter (70/15/15, seed 42) from ADR-005:
- **Train split:** 105,311 records ($87{,}261$ true pairs). Fixed 100k train sample (seed 42): 100,000 records ($54{,}375$ true pairs).
- **Validation split:** 22,373 records ($17{,}916$ true pairs / $35{,}832$ directed true slots).
- **Test split:** 22,316 records ($18{,}195$ true pairs) — strictly sealed.

### Decision

Declare `ber/data/raw/foursquare_kaggle/coordinates.parquet` (150,000 records) as the **canonical development and evaluation corpus** for all BER work from Phase 4c onward.

All candidate stores, freeze gates, retrieval sweeps, loss ledgers, and final comparisons must use this canonical table and its canonical validation denominator ($17{,}916$ undirected true pairs / $35{,}832$ directed true slots).

Earlier full-val recall numbers (denominator $123{,}067$) are unpreserved and cannot be directly compared in absolute terms to numbers evaluated on the 150k canonical corpus. However, relative channel ordering, incremental-yield rules, and trade-off frontiers remain empirically consistent.

