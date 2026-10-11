# Pre-registration 004 — Geo-Proximity Retrieval Channel (G1)
## Phase 3a — Haversine Distance Retrieval

**Status:** Executed (Phase 3a)  
**Date:** 2026-10-07  
**Author:** BER System  
**Depends on:** Experiment 002 (`experiments/002_foursquare_control.md`)  

---

## 1. Background & Motivation

In Experiment 002 (Foursquare control baseline), the control keys K1–K5 achieved an ungated ($G=\text{None}$) pair recall ceiling of only **41.24 %** (50,754 / 123,067 val true pairs). Analysis of the B9 residual on the train split revealed that **48.7 %** (226,285 pairs) of unrecovered true pairs at $G=50$ are different-name entities located within 10 km in the same country.

Because names vary drastically across sources (due to abbreviations, language variants, parent brand vs venue name), surface-form blocking alone cannot bridge this gap. Geo-proximity retrieval directly targets spatial co-location.

---

## 2. Hypothesis

A metric geo-proximity channel G1 retrieving the $K$ nearest neighbors within radius $R$ meters (using haversine distance over indexed coordinates) will:
1. Substantially increase pair recall when combined with the surface-form control baseline.
2. Recover a large fraction of the B9 residual slice (especially different-name pairs).
3. Outperform surface-form control at matched candidate budgets.

---

## 3. Pre-Registered Architecture & Parameters

### Channel G1 Specification:
- **Indexing:** All corpus records indexed by (latitude, longitude) in radians using an out-of-core compatible BallTree with haversine metric.
- **Query:** Each query record queries the BallTree.
- **Candidates:** $K$ nearest neighbors within radius $R$ meters. Excludes self-match ($q \neq c$).
- **No country filter required** (pure metric distance).

### Parameter Sweep (Train Sample Only):
- **Training sample:** Fixed random 100,000-query train sample (seed 42).
- **Sweep grid:** $K \in \{5, 10, 20, 50\} \times R \in \{200\text{ m}, 1000\text{ m}, 5000\text{ m}\}$ (12 settings).
- **Selection rule:** Select $(K, R)$ maximizing pair recall on the train sample. Val is not tuned on.

---

## 4. Evaluation & Comparators (Val Split)

1. **Standalone G1:** Evaluated at the train-chosen $(K, R)$ on the val split.
2. **Control + G1 Union:** Evaluated at $G=50$ and $G=\text{None}$.
3. **Matched-Budget Comparisons:** Evaluated at ~700k and ~2M candidate budgets against Control ($G=50$ and $G=200$).
4. **Statistical Rigor:** Paired bootstrap over query entities ($N_{\text{boot}} = 500$, seed 42) for recall delta ($A - B$). Differences below the 95% bootstrap CI width are labeled unconfirmed.

---

## 5. Dual-Dataset Requirement & Pre-Registered Deviation

> [!WARNING] Pre-Registered Deviation
> Project rules require retrieval channels to be evaluated across both benchmark datasets (GLEIF × Companies House and Foursquare).
> 
> **Dataset Limitation:** The GLEIF × Companies House dataset contains **no geographic coordinates** (latitude/longitude), only UK postal codes.
> 
> **Deviation Resolution:** The dual-dataset requirement for spatial proximity retrieval cannot be satisfied via coordinate-based BallTree indexing on GLEIF × CH. Instead, it will be satisfied by a **postcode-based proximity channel** (e.g. full postcode, outward code, or postcode-district graph adjacency).
> 
> **Status:** Implementation of the postcode-based spatial variant on GLEIF × CH is **deferred to Phase 3b**. Phase 3a reports coordinate-based geo-proximity retrieval on Foursquare exclusively.
