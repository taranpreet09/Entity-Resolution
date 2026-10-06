# Pre-registration 003 — Address-Based Retrieval on the Name-Missing Slice

**Status:** Pre-registered — NOT run  
**Pre-registered:** 2026-10-07  
**Depends on:** Experiment 001 (control baseline loss ledger), Experiment 002 results

---

## Hypothesis

For the 311 true pairs that the control system never retrieves (no shared
blocking key at any G), a substantial fraction have a usable address signal:
the GLEIF `LegalAddr_PostCode` or `LegalAddr_Line1` matches or is close to
the CH `RegAddress.PostCode` or `RegAddress.AddressLine1`. An address-primary
retrieval channel targeting this slice will recover a measurable subset of
these 311 losses without degrading recall on pairs the control already handles.

---

## Motivation from experiment 001

The 311 no-shared-key losses are the structural floor of the control system.
They cannot be recovered by any gate-size adjustment. Diagnosis suggests two
main causes (unmeasured — to be confirmed):
1. Name mismatch: GLEIF legal name diverges from CH registered name beyond
   what token-prefix blocking can bridge.
2. Postcode mismatch or absence on the GLEIF side, preventing K2/K3/K4.

Address retrieval targets cause 2: when the name keys fail, the postcode
and address line may still uniquely or near-uniquely identify the entity.

---

## Method

1. Define the **name-missing slice**: the subset of val queries whose true
   pair appears in the G=None loss ledger as RETRIEVAL / no-shared-key
   (i.e. the 311 pairs identified in experiment 001). Evaluate this channel
   on the full val set but measure its unique contribution on this slice.

2. Build an address channel with two sub-keys:
   - **A1:** `outward_code(postcode) | first_token(address_line_1)`
     (query: GLEIF LegalAddr; target: CH RegAddress)
   - **A2:** `canon(postcode_full)`
     (exact normalised postcode match, no name component)

3. Apply a gate G_addr (sweep: 5, 20, 100) on address buckets to prevent
   dense urban postcodes from flooding the candidate set.

4. Union results with the control at G=50. Evaluate the **combined system**
   against experiment 001's control-only results.

5. Run `compare_systems()` (paired bootstrap, 1,000 resamples, seed=42)
   between combined system and control-only on the full val set.

---

## Success criterion

**Success (primary):** The combined system (control G=50 + address channel)
recovers ≥ 50 of the 311 no-key losses identified in experiment 001,
with recall@budget for the combined system strictly higher than control
G=50 alone (bootstrap CI for delta > 0).

**Success (secondary):** No regression — the combined system does not drop
any pairs that the control at G=50 recovered.

**Failure:** Fewer than 50 of the 311 no-key pairs recovered, or the delta
CI straddles zero, or address buckets are too dense to be useful (majority
of A2 buckets gated even at G_addr=100).

---

## Metrics to report

- For the name-missing slice (311 pairs): count recovered by address channel
- recall@budget for control-only vs combined (with bootstrap CI)
- zero-candidate rate change
- cpq distribution change (check candidate inflation from dense postcodes)
- loss ledger: INPUT / CANON / RETRIEVAL counts before and after
- A1 vs A2 unique contribution

---

## What this experiment does NOT measure

- Retrieval quality on the test split (sealed)
- Address retrieval for queries where the control already succeeds
  (these are in the denominator but are not the focus)
- Performance on no-match queries (excluded from denominator)
- Whether address signal generalises to other datasets

---

## Sequencing note

This experiment should run after experiment 002. If 002 shows that IDF
lexical retrieval already recovers ≥ 250 of the 311 no-key pairs at matched
budget, experiment 003 may not be necessary and should be re-evaluated.
