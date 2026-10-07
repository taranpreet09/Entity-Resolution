# Pre-registration 003b — Address-Based Retrieval on the Name-Differs Slice
## (Amended 2026-10-07 — supersedes 003_address_retrieval.md)

**Status:** Pre-registered — NOT run
**Pre-registered:** 2026-10-07
**Supersedes:** `experiments/preregistrations/003_address_retrieval.md`
**Depends on:** Experiments 001b (residual taxonomy, postcode analysis)

---

## Changes from 003_address_retrieval.md

1. **"Name-missing slice" does not exist** in GLEIF×CH: only 1 of 77,812 GLEIF
   records has a blank LegalName; none appear in the val set. The original framing
   was wrong.
2. **Redefined target slice = "name-differs slice"**: the 001b taxonomy (control_v2
   G=None residual) classifies unrecovered pairs by name mismatch pattern. This
   experiment targets specifically the subset where a postcode signal exists and the
   name mismatch is too large for K4v2 to bridge.
3. **Minimum effect threshold added** (same as 002b): Δrecall ≥ 0.003 absolute
   required to claim improvement.
4. **Cross-dataset requirement**: results must replicate on at least one additional
   dataset.
5. **Sequencing clarified**: run AFTER 002b. If 002b recovers ≥ 80% of the residual
   slice (as measured in 001b), address retrieval may have negligible remaining scope
   and should be re-evaluated before running.

---

## Motivation

The 001b residual taxonomy quantifies, for each unrecovered pair, whether the query
has a usable postcode and whether the postcard outward code matches the target. For
pairs where:
- query postcode is present AND
- query outward code matches target outward code AND
- the name mismatch is not bridgeable by K4v2 or IDF,

an address-primary blocking key may recover the pair at low candidate cost.

---

## Hypothesis

For the subset of name-differs-slice pairs where both query and target share the
same outward postcode code and the query postcode is present, an address-primary
key (postcode + first name token) will recover a measurable fraction at ≤ 20
candidates per query median (to avoid flooding the candidate set with postcode
bucket members).

---

## Method

1. Define the address channel keys:
   - **A1**: `outward_code(postcode) | first_token(name)` — same as K2/K4 but
     applied as a separate channel so its contribution is isolated.
   - **A2**: `canon(full_postcode)` — exact normalised postcode (no name component).
   Apply gate G_addr ∈ {5, 20, 100} on address buckets.

2. Build address index from CH `RegAddress.PostCode` + `CompanyName`.
   (Requires loading PostCode column from CH CSV — already available in the
   DuckDB streaming path.)

3. Union with control_v2 G=50. Evaluate the **combined system** on val.
   Compare to control_v2 G=50 alone via paired bootstrap (1,000 resamples, seed=42).

4. Report specifically:
   - How many of the 001b name-differs residual pairs are recovered?
   - cpq distribution delta (does adding address inflate candidate count?)
   - Zero-candidate rate delta

---

## Success criterion

**Success (primary):** Combined system recall@budget > control_v2 G=50 alone, with
95% CI for delta strictly above zero AND Δ ≥ 0.003.

**Success (secondary, slice):** ≥ 30% of the 001b residual pairs where
`postcode_present AND same_outward` are recovered by the address channel alone.

**Failure:** Either primary criterion not met, OR cpq median of the combined system
is > 2× cpq median of control_v2 G=50 (address channel is flooding candidates).

---

## Cross-dataset requirement

Results on GLEIF×CH are preliminary. Conclusions require replication on at least
one additional dataset.

---

## What this experiment does NOT measure

- Test split (sealed)
- No-match queries (excluded from denominator)
- Name retrieval for queries without a valid UK postcode
