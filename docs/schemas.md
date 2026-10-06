# Storage Schemas

All persistent artefacts are stored in **Parquet** (columnar, splittable,
schema-enforced) and queried via **DuckDB**.  PyArrow is used for writes.
No stage may load an entire large file into RAM; all reads must use
`read_parquet` with projection pushdown or row-group streaming.

See `ber/store/schemas.py` for the PyArrow schema objects.

---

## `records` — Raw Source Records

One row per source record.  Raw fields are preserved verbatim as a JSON
string; no field is modified or discarded by ingestion.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `dataset_id` | STRING | No | Stable dataset identifier, e.g. `"fodors_zagats"` |
| `record_id` | STRING | No | Stable record identifier unique within the dataset |
| `canon_version` | STRING | Yes | Identifier of the canonicalization run applied; NULL until canon stage runs |
| `raw_fields` | STRING | No | JSON object of all raw source fields, verbatim |
| `source_file` | STRING | Yes | Source file path (for auditing) |
| `source_row` | INT64 | Yes | 0-indexed row number in the source file |

**Partitioning:** by `dataset_id`.

**Invariant:** `raw_fields` is never modified after initial write.  Any
derived views (canonical, normalised) live in separate tables or columns.

---

## `labels` — Ground-Truth Entity Assignments

Maps each record to its entity cluster.  Multiple records sharing an
`entity_id` are considered co-referent.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `record_id` | STRING | No | Foreign key into `records.record_id` |
| `entity_id` | STRING | No | Cluster identifier |
| `label_provenance` | DICT(INT8→STRING) | No | How the label was derived (see below) |
| `dataset_id` | STRING | Yes | Dataset context |
| `split` | STRING | Yes | `train`, `val`, or `test` |

**label_provenance values:**

| Value | Meaning |
|---|---|
| `manual` | Human-annotated |
| `identifier_derived` | Two records share an authoritative identifier (e.g. same Companies House number) |
| `algorithmic` | Produced by a clustering algorithm on canonical features |
| `synthetic` | Artificially generated for testing purposes |

---

## `pairs_labels` — Explicit Pairwise Judgements (Optional)

Used when explicit positive/negative pair annotations are available (e.g.
crowd-sourced judgements, blocking-derived negatives).

| Column | Type | Nullable | Description |
|---|---|---|---|
| `query_id` | STRING | No | One member of the pair |
| `cand_id` | STRING | No | Other member of the pair |
| `label` | INT8 | No | `1` = match, `0` = non-match, `-1` = unknown |
| `label_provenance` | STRING | Yes | Free-text provenance note |
| `dataset_id` | STRING | Yes | Dataset context |
| `split` | STRING | Yes | `train`, `val`, or `test` |

**Note:** The Fodors-Zagats dataset supplies pairs in this format.  Its
pairs came out of a blocking step and are therefore not a random sample of
all positives.  Any recall measurement on these pairs reflects
*post-blocking* recall only; retrieval recall cannot be measured from this
data.

---

## `candidates` — Retrieval Output

One row per (query, candidate) pair produced by any retrieval channel.
The table is **sorted by `(query_id, cand_id)`** — alignment must always
be done via explicit joins, never by relying on row ordering.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `query_id` | STRING | No | Query record ID |
| `cand_id` | STRING | No | Candidate record ID |
| `channel_mask` | UINT32 | No | Bitmask; bit i = channel i contributed this pair |
| `score` | FLOAT16 | No | Highest retrieval score across contributing channels |
| `rank` | UINT16 | No | Rank of this candidate for the query (1-indexed) |
| `reverse_rank` | UINT16 | No | Rank of the query in the candidate's result list |
| `round` | UINT8 | No | Retrieval round (for multi-round / iterative retrieval; default 1) |
| `dataset_id` | STRING | Yes | Dataset context |

**Channel bitmask encoding:** the experiment config registers channels in
a fixed order.  Bit 0 = channel 0, bit 1 = channel 1, etc.  A candidate
contributed by channels 0 and 2 has `channel_mask = 0b0101 = 5`.

**Why explicit pair table?** Relying on dict ordering or positional
alignment across two arrays is a common source of silent bugs in ER systems.
Every downstream join uses `(query_id, cand_id)` as the key.

---

## File Layout

```
experiments/
  runs.parquet              ← experiment run ledger
  <run_id>/
    records/
      dataset_id=<id>/
        part-0.parquet
    labels/
      dataset_id=<id>/
        part-0.parquet
    candidates/
      dataset_id=<id>/
        part-0.parquet
    loss_ledger.csv
docs/
  data_verification.md
  schemas.md
  decisions.md
```
