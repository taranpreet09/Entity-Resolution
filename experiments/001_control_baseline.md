# Experiment 001 — Control Baseline: Exact-Key Blocking on GLEIF × Companies House

## Setup

| Field | Value |
|---|---|
| **run_id** | `1828c340-d12a-4bca-90d6-dcc5769a0a62` |
| **config_hash** | `0e024add9448ca3153bf9ff813050b10e9e91dbd099212ddf58c2087cfe7114b` |
| **git_commit** | `untracked` |
| **dataset** | `gleif_ch` |
| **dataset_version** | `gleif_2026-10-05_ch_2026-10-01` |
| **seed** | `42` |
| **evaluated split** | `val` (test split untouched) |
| **command** | `python experiments/run_control_baseline.py --seed 42` |

Results are reproducible from the single command above given the cached
`data/processed/gleif_ch/` artefacts. The index is rebuilt from the raw CH
CSV on each run (DuckDB streaming); splits are loaded from cache.

---

## Dataset

### Ingestion

| Metric | Count |
|---|---|
| GLEIF GB LEI records in file (partial — ~44 % of golden copy) | 77,812 |
| GLEIF records matched to a Companies House number | 65,388 |
| GLEIF records with no CH match (non-live / non-CH entities) | 12,424 |
| Companies House target records | 5,704,711 |
| Positive pairs (label provenance: `identifier_derived`) | 65,388 |

No-match queries (12,424) are excluded from all recall denominators. They
are tracked in `data/processed/gleif_ch/no_match_queries.json`.

Labels were derived solely from the GLEIF `RegistrationAuthorityEntityID`
field matching against a normalised `CompanyNumber`. Both identifier
columns are **excluded from every model-visible record**; they exist only
in the label store. The leakage test in `tests/test_gleif_ch_adapter.py`
asserts this at the Parquet level.

### Splits (seed=42, ratios 70/15/15)

| Split | Queries | True pairs |
|---|---|---|
| train | 54,454 | unmeasured (not evaluated) |
| **val** | **11,682** | **9,860** |
| test | 11,676 | unmeasured (kept sealed) |

Secondary (brand-cluster) split: train=54,454 / val=11,680 / test=11,678.
Leakage check: **0 entities cross splits** (assertion in pipeline).

---

## Canonicalisation

Algorithm: NFKD decomposition → ASCII-ignore → lowercase → strip
punctuation → collapse whitespace. Deliberately lossy baseline.

| Dataset | Field | Raw non-empty | Empty after canon | Rate |
|---|---|---|---|---|
| GLEIF | `LegalName` | 77,812 | **1** | 0.00 % |
| GLEIF | `LegalAddr_PostCode` | 77,798 | 0 | 0.00 % |
| CH | `CompanyName` | 5,704,711 | **0** | 0.00 % |
| CH | `RegAddress.PostCode` | 5,613,236 | 43 | 0.00 % |

Canonicalisation contributes **0 losses** to the ledger on the val set.
The single GLEIF name that canonicalises to empty is assigned INPUT loss;
no val true pairs happen to involve that query.

---

## Blocking Keys

Four exact-match keys over canonicalised fields:

| Key | Query side | Target side |
|---|---|---|
| K1 | `canon(LegalName)` | `canon(CompanyName)` |
| K2 | `first_token(LegalName) \| outward_code(postcode)` | same pattern |
| K3 | `first_3_chars(LegalName) \| postcode_district` | same pattern |
| K4 | `outward_code(postcode) \| first_token(LegalName)` | same pattern |

Bucket-size gate G: buckets with more than G target records are dropped
entirely for that query.

Index built from the raw CH CSV via DuckDB streaming (CompanyName +
PostCode columns only). Peak RAM during indexing: ~700 MB measured. The
previous approach (full JSON materialisation) required ~13 GB and caused
an OOM crash.

---

## G-Sweep Results (validation split)

All metrics computed over **entities** (unique query IDs), not pairs.
Bootstrap CI: 500 resamples, seed=42, over query-side entities.
`recall@budget` = recall at the maximum candidate rank retrieved per query
(equivalent to "did the true target appear anywhere in the candidate set?").
`partial_recall` = fraction of queries where some but not all true targets
are retrieved; all val queries have exactly one true target so this is
necessarily 0.

| G | Candidates | cpq median | cpq p95 | recall@budget | 95% CI | zero-cand rate |
|---|---|---|---|---|---|---|
| 10 | 34,437 | 2 | 9 | **0.9665** | [0.9634, 0.9693] | 0.0158 |
| 50 | 130,629 | 6 | 41 | **0.9675** | [0.9644, 0.9702] | 0.0082 |
| 200 | 321,275 | 12 | 119 | **0.9683** | [0.9653, 0.9710] | 0.0051 |
| 1000 | 549,355 | 14 | 219 | **0.9685** | [0.9654, 0.9712] | 0.0038 |
| none | 685,788 | 14 | 237 | **0.9685** | [0.9654, 0.9712] | 0.0038 |

Control curve saved to `experiments/control_curve.csv`.

Key observation: recall saturates rapidly. Going from G=10 to G=1000
adds 514,918 candidates (+1,496 %) for a recall gain of +0.0020 absolute
(+0.21 %). The recall gap between G=1000 and G=None is zero — no true pair
sits in a bucket larger than 1000.

---

## Per-Key Unique Contribution (G=50)

"Unique contribution" = candidate pairs produced **exclusively** by that
key and no other key. Pairs covered by multiple keys are not double-counted.

| Key | Pairs in key | Unique contribution |
|---|---|---|
| K1 | 9,546 | 1,959 |
| K2 | 42,622 | 0 |
| K3 | 110,831 | 85,947 |
| K4 | 42,622 | 0 |

K2 and K4 share the same bucket structure (first_token+outward vs
outward+first_token map to the same inverted index when query and target
names start with the same token) and add zero unique pairs beyond their
overlap with K3. K3 (3-char prefix + district) contributes the majority
of unique candidates because its larger, fuzzier buckets contain pairs
that K1 misses.

---

## Loss Ledger

Denominators: 9,860 val true pairs total.

### At G=50

| Stage | Count | % of total |
|---|---|---|
| INPUT | 0 | 0.00 % |
| CANONICALIZATION | 0 | 0.00 % |
| RETRIEVAL — size-gated | 9 | 0.09 % |
| RETRIEVAL — no shared key | 311 | 3.16 % |
| BUDGET_PRUNE | 0 | 0.00 % |
| FEATURES | 0 | 0.00 % |
| RANKER | 0 | 0.00 % |
| DECISION | 0 | 0.00 % |
| RECOVERED (true pair retrieved) | 9,540 | 96.75 % |

Per-pair CSV: `experiments/1828c340.../loss_ledger_G50.csv`

### At G=None (no gate)

| Stage | Count | % of total |
|---|---|---|
| INPUT | 0 | 0.00 % |
| CANONICALIZATION | 0 | 0.00 % |
| RETRIEVAL — size-gated | 0 | 0.00 % |
| RETRIEVAL — no shared key | **311** | 3.16 % |
| RECOVERED | 9,549 | 96.85 % |

The gate accounts for only 9 additional losses at G=50. The **structural
floor** is 311 pairs (3.16 %) that share no blocking key between their
GLEIF query and CH target regardless of gate size. These represent the
hard limit of this control system at any G.

### Breakdown of the 311 no-shared-key losses

The 311 pairs where no key fires are candidates for the address-based
retrieval channel pre-registered in experiment 003. Possible causes:
- GLEIF legal name differs substantially from the registered CompanyName
  (e.g. trading name vs legal name, abbreviations, language differences)
- Missing or non-UK postcode on the GLEIF record preventing K2/K3/K4
- Company registered under a holding-structure name not reflected in GLEIF

Exact diagnosis requires inspecting the 311 records; this is noted as
an observation, not a measured claim.

---

## Observations

All observations below are backed by the measured numbers above.

1. **The control ceiling is 96.85 % recall** (G=None). No gate-free exact-key
   matcher over name + postcode can exceed this with the current key set.

2. **G=50 is the practical operating point.** It recovers 96.75 % of true pairs
   at 13.3 candidates per query median (p95=41), which is 97 % of the
   maximum possible recall at 19 % of the G=None candidate volume.

3. **Size-gating contributes negligible loss at G=50** (9 pairs, 0.09 %).
   The dominant loss source is the structural no-key floor (311, 3.16 %).

4. **K2 and K4 add no unique value** beyond K1 and K3 at any G tested.
   K3 (first-3-chars + district) dominates unique coverage. A future key
   design could replace K2/K4 with a more discriminating alternative.

5. **Canonicalisation is not a bottleneck** for this dataset: only 1 GLEIF
   name and 0 CH names collapse to empty after the control pipeline.

6. **Partial recall is 0** across all G values, because every val query has
   exactly one true target (1:1 matching by identifier derivation).

---

## Files

| Artefact | Path |
|---|---|
| Experiment run ledger | `experiments/runs.parquet` |
| Control curve | `experiments/control_curve.csv` |
| G=10 candidates | `experiments/1828c340-.../candidates_G10.parquet` |
| G=50 candidates | `experiments/1828c340-.../candidates_G50.parquet` |
| G=200 candidates | `experiments/1828c340-.../candidates_G200.parquet` |
| G=1000 candidates | `experiments/1828c340-.../candidates_G1000.parquet` |
| G=None candidates | `experiments/1828c340-.../candidates_GNone.parquet` |
| Loss ledger G=50 | `experiments/1828c340-.../loss_ledger_G50.csv` |
| Loss ledger G=None | `experiments/1828c340-.../loss_ledger_GNone.csv` |
| Processed queries | `data/processed/gleif_ch/queries.parquet` |
| Processed targets | `data/processed/gleif_ch/targets.parquet` |
| Positive pairs | `data/processed/gleif_ch/pairs_labels.parquet` |
| Primary splits | `data/processed/gleif_ch/primary_splits.json` |
| Brand splits | `data/processed/gleif_ch/brand_splits.json` |
| No-match queries | `data/processed/gleif_ch/no_match_queries.json` |


---

## Errata

*Added 2026-10-07 as part of Phase 2b. The body of this document is unchanged.*

### E1 — K4 is a swapped duplicate of K2 (structural defect)

**What the document says:** "K4: `outward_code(postcode) | first_token(LegalName)`"
and "K2 and K4 add no unique value beyond K1 and K3 at any G tested."

**What is actually true:** K4 in control_v1 is `outward|first_token` while K2 is
`first_token|outward`. Because both query and target use the same pattern, the bucket
a query hits under K4 is exactly the set of targets whose first_token and outward_code
match — the same set K2 finds. The keys are structurally symmetric; their values differ
(e.g. `"EC1A|acme"` vs `"acme|EC1A"`) so they hash to different buckets and appear to
produce different index entries, but the intersection with true pairs is identical. This
is why K2 and K4 both show 0 unique candidate pairs and 0 unique true pairs in every G
sweep. The claim in Observation 4 that "K2 and K4 add no unique value" is correct in
its conclusion but the stated reason ("same bucket structure") was imprecise — the real
reason is that both keys index the same (first_token, outward) joint distribution with
swapped positions, which recovers the same true pairs.

**Fix:** control_v2 replaces K4 with K4v2 = `strip_legal_suffixes(LegalName)` (no
postcode), which recovers pairs where names differ only in legal-form suffix. See
`experiments/001b_control_diagnostics.md` for v2 results.

### E2 — Observation 2: "13.3 candidates per query median" is wrong

**What the document says:** "G=50 is the practical operating point. It recovers 96.75 %
of true pairs at 13.3 candidates per query median (p95=41)."

**What is actually true:** The measured cpq median at G=50 is **6**, not 13.3. The
figure 13.3 was the **mean** (total candidates ÷ queries with ≥1 candidate =
130,629 ÷ ~9,800). The median and mean differ substantially because the distribution is
right-skewed (most queries get 1–6 candidates; a small number get many more via K3).
Both statistics are meaningful; neither alone is sufficient. Corrected: cpq median=6,
mean≈13.3 (computed over queries with ≥1 candidate), p95=41.

**Note on denominator for mean:** the denominator for cpq mean/median is queries that
received at least one candidate, not all queries with true targets. See
`docs/metrics.md` for the precise definition.

### E3 — Observation 1: recall ceiling claim

**What the document says:** "The control ceiling is 96.85 % recall (G=None). No
gate-free exact-key matcher over name + postcode can exceed this with the current key
set."

**What is actually true:** 96.85 % is the v1 ceiling with keys K1–K4. The observation
is correct for v1 but should be labelled "control_v1 ceiling." control_v2 with K4v2
achieves a different (measured) ceiling; see `001b_control_diagnostics.md`.

### E4 — git_commit was "untracked"

The original run was executed before the repository was initialised. The git commit
recorded in the ledger for run `1828c340-d12a-4bca-90d6-dcc5769a0a62` is `"untracked"`.
The repository was initialised on 2026-10-07 and the code at that point was committed as
`c805f600e4caa2a2c61be72b5bdc2fd808a71e99`. The v1 reproduction check (run
`phase2b_v1_repro`) confirmed bit-identical metrics with that commit; the errata do not
affect the numerical results of Experiment 001.

### E5 — partial_recall "necessarily 0" explanation was incomplete

The document states partial recall "is necessarily 0" because every val query has
exactly one true target. This is correct but should be labelled "N/A" per the metric
definition in `docs/metrics.md` rather than reported as a substantive result of 0.0.
A value of 0.0 with one true target per query is a tautology, not a measurement.
