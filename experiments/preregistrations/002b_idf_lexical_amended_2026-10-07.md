# Pre-registration 002b — IDF-Weighted Lexical Retrieval vs Control Baseline
## (Amended 2026-10-07 — supersedes 002_idf_lexical.md)

**Status:** Pre-registered — NOT run
**Pre-registered:** 2026-10-07
**Supersedes:** `experiments/preregistrations/002_idf_lexical.md`
**Depends on:** Experiments 001 (control_v1), 001b (control_v2, residual taxonomy)

---

## Changes from 002_idf_lexical.md

1. **Baseline is now control_v2** (K1, K2, K3, K4v2) rather than control_v1.
   The v1 K4 defect (duplicate of K2) has been corrected.
2. **"Name-missing slice" renamed to "name-differs slice"**: this dataset has no
   queries with a completely missing name (1 in 77,812; none in val). The relevant
   slice is queries where the name-based keys fire but the true pair is not recovered
   — i.e. the residual after v2 at G=None. See taxonomy in 001b.
3. **Primary comparison budget clarified**: comparison is at 1–15 cpq median (the
   operating range of control_v2 at G=10 to G=50), not at total candidate counts.
4. **Minimum detectable effect stated**: differences below ~0.003 absolute (0.3 pp)
   cannot be claimed at 95% confidence given the val set size (~9,800 queries with
   true targets). The CI half-width of both the control and any new system must
   be reported; overlapping CIs are not a claim of improvement.
5. **Cross-dataset requirement added**: conclusions about IDF vs control require
   replication on the Foursquare dataset (or the next dataset added). A result on
   GLEIF×CH alone is preliminary.

---

## Hypothesis

An IDF-weighted lexical retrieval channel (BM25 or TF-IDF cosine over canonical
name tokens) will achieve higher recall@budget than **control_v2 at matched
candidate count**, recovering a measurable fraction of the "name-differs slice"
defined in 001b.

---

## Method

1. Build a sparse inverted BM25 index (k1=1.5, b=0.75) over `canon(CompanyName)`
   tokens for all CH targets.
2. Retrieve top-K for each GLEIF val query using `canon(LegalName)` as the query
   string. K is chosen so that total val candidates ≈ control_v2 G=50 total.
3. Evaluate on **val split only**. Test split sealed.
4. Report recall@budget, zero-candidate rate, cpq distribution (median, mean,
   p95 — all three, with denominators), loss ledger stage counts.
5. Compare to control_v2 G=50 using `compare_systems()`: paired bootstrap, 1,000
   resamples, seed=42. Report point estimates, delta, 95% CI for delta, p-value.

---

## Success criterion

**Success:** recall@budget (IDF) > recall@budget (control_v2 G=50), with the 95%
bootstrap CI for delta strictly above zero AND the delta ≥ 0.003 absolute.

**Failure:** delta < 0.003, or CI straddles zero.

---

## Name-differs slice analysis

For the residual pairs from 001b taxonomy, report: how many does the IDF channel
recover per category (a–h)? The IDF channel is expected to help most for categories
(c) token reordering, (d) stop-word addition, and (e) substantial rename with high
token Jaccard. It is NOT expected to help categories (f) postcode missing alone or
(a) legal-suffix variant (which K4v2 already handles).

---

## Cross-dataset requirement

This experiment will be run on GLEIF×CH and on the Foursquare dataset (or the next
dataset added to the system). A result on GLEIF×CH alone is labelled "preliminary."
Conclusions about whether IDF beats the control require concordance across at least
two datasets with different recall ceilings.

---

## What this experiment does NOT measure

- Retrieval on the test split (sealed)
- Semantic similarity beyond lexical overlap
- No-match queries (excluded from denominator)
- Behaviour on datasets with multi-target queries (partial recall N/A here)
