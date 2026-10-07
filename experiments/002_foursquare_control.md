# Experiment 002 — Foursquare Control Baseline

**Parent:** BER Baseline Control Architecture  
**Evaluated Split:** `val` (primary metrics & efficient frontier)  
**Diagnostic Split:** `train` (residual analysis only)  
**Sealed Split:** `test` (untouched)  
**Date:** 2026-10-07  
**Git Commit:** `83610d627c51b43593589dcccd5344831e00af72`  

---

## 1. Executive Summary

This experiment evaluates the control blocking baseline on the single-table deduplication dataset **Foursquare Location Matching** ($N = 1,138,812$ records, $739,972$ ground-truth POI entities).

### Key Empirical Findings
1. **Control Ceiling:** Ungated control blocking (G=None) achieves **41.24 % pair recall** (50,754 / 123,067 val true pairs) and **63.34 % query recall** at 8,227,481 candidates.
2. **Budget Gated Performance:** At G=50, the baseline achieves **37.66 % pair recall** and **62.36 % query recall** at 643,135 candidates (cpq median: 2.0).
3. **Loss Allocation:** The loss ledger attributes **58.76 % of true pairs lost to RETRIEVAL** (no shared blocking key across any of K1..K5). CANONICALIZATION loss is negligible ($1$ pair, $<0.001\%$).
4. **Dominant Keys:** Key K2 (`first_token | geo_cell_2deg`) generates the highest standalone recall (27.66% ungated) at lowest candidate volume (208,792 candidates). Key K1 (`name | country`) provides the largest marginal recall lift (+10.83 pp), but requires 5.4M additional candidates.

---

## 2. Ingest, Splits & Leakage Audit

- **Total Records:** 1,138,812
- **POI Entities:** 739,972
- **Singleton POIs:** 425,024 (57.44 % of clusters, 37.32 % of records)
- **Non-Singleton POIs:** 314,948 (42.56 % of clusters, 62.68 % of records)
- **Total Positive Pairs:** 950,503 (ground truth within clusters)
- **Primary Split (Entity-disjoint 70/15/15 by POI, seed=42):**
  - Train: 800,853 records (672,540 true pairs)
  - Val: 169,452 records (123,067 true pairs)
  - Test: 168,507 records (**Sealed**)
- **Leakage Check:** 0 leaking POI entities across splits (**PASS**).
- **Provenance:** Recorded as **"synthetic"** (Foursquare simulated matching ground truth assumption).

---

## 3. Canonicalization Pipeline Stats

- **Pipeline:** Unicode NFKD normalization $\rightarrow$ ASCII-ignore lowercasing.
- **Empty Names Post-Canonicalization:** 11 / 1,138,812 records ($0.00\%$).

---

## 4. Control Blocker Key Definitions & Feasibility

| Key ID | Definition | Formula / Implementation |
|---|---|---|
| **K1** | Name + Country | `canon(name) \| country` |
| **K2** | First Name Token + Coarse Geo Cell | `first_token(name) \| geo_cell(lat, lon, 2_decimals)` (~1km cell) |
| **K3** | Digits-only Phone | `digits_only(phone)` |
| **K4** | Registrable URL Domain | `extract_url_domain(url)` |
| **K5** | First 3 Chars + Postal Zip | `first_3_chars(name) \| zip` |

> [!NOTE]
> **Corpus Indexing:** The blocking index was built over all $1,138,812$ records (full corpus) without split restriction, ensuring bucket sizes reflect true corpus density without label leakage.

### Pair Count Feasibility Estimates
Estimated candidate pair count across all buckets ($\sum \frac{b(b-1)}{2}$):
All gate sizes $G \in \{10, 50, 200, 1000, \text{None}\}$ passed the feasibility check ($< 500,000,000$ pair limit).

---

## 5. Main Metrics Sweep (Val Split)

Evaluated on $169,452$ val queries against the full $1.13\text{M}$ record corpus.

| G | Pair Recall | Query Recall | 95% Bootstrap CI | Partial Recall | Zero-Cand Rate | CPQ Median | CPQ P95 | CPQ Mean | Candidate Count |
|---|---|---|---|---|---|---|---|---|---|
| **10** | 0.3244 | 0.6088 | [0.6065, 0.6111] | 0.2844 | 0.3098 | 1.0 | 8.0 | 2.15 | 247,920 |
| **50** | 0.3766 | 0.6236 | [0.6213, 0.6258] | 0.2470 | 0.2836 | 2.0 | 17.0 | 4.87 | 643,135 |
| **200** | 0.4092 | 0.6289 | [0.6266, 0.6311] | 0.2197 | 0.2731 | 2.0 | 33.0 | 12.08 | 1,731,495 |
| **1000** | 0.4122 | 0.6329 | [0.6306, 0.6352] | 0.2207 | 0.2682 | 2.0 | 54.0 | 29.36 | 4,274,985 |
| **None** | 0.4124 | 0.6334 | [0.6311, 0.6357] | 0.2210 | 0.2679 | 2.0 | 69.0 | 56.40 | 8,227,481 |

---

## 6. Control Family Efficient Frontier

Keys ordered greedily by marginal true pairs per candidate generated:

### G = 50 Budget Gate
| Key Added | Active Keys | Cum. Pair Recall | Cum. True Pairs | Cum. Candidates | Marginal Return (Pairs/Cand) |
|---|---|---|---|---|---|
| **K2** | `K2` | 0.2462 | 30,301 | 171,473 | 0.1767 |
| **+K1** | `K2, K1` | 0.3448 | 42,433 | 458,876 | 0.0422 |
| **+K5** | `K2, K1, K5` | 0.3691 | 45,428 | 508,776 | 0.0600 |
| **+K3** | `K2, K1, K5, K3` | 0.3747 | 46,110 | 529,219 | 0.0334 |
| **+K4** | `K2, K1, K5, K3, K4` | 0.3766 | 46,341 | 643,135 | 0.0020 |

### G = None (Ungated)
| Key Added | Active Keys | Cum. Pair Recall | Cum. True Pairs | Cum. Candidates | Marginal Return (Pairs/Cand) |
|---|---|---|---|---|---|
| **K2** | `K2` | 0.2766 | 34,042 | 208,792 | 0.1630 |
| **+K1** | `K2, K1` | 0.3849 | 47,372 | 5,646,366 | 0.0025 |
| **+K5** | `K2, K1, K5` | 0.4043 | 49,751 | 5,695,105 | 0.0488 |
| **+K3** | `K2, K1, K5, K3` | 0.4098 | 50,433 | 5,815,847 | 0.0056 |
| **+K4** | `K2, K1, K5, K3, K4` | 0.4124 | 50,754 | 8,227,481 | 0.0001 |

---

## 7. Loss Ledger Accounting

Evaluated on all val true pairs (including directed pair representation):

| Loss Stage | G = 50 Count | G = 50 % | G = None Count | G = None % |
|---|---|---|---|---|
| **INPUT** | 0 | 0.00 % | 0 | 0.00 % |
| **CANONICALIZATION** | 1 | $<0.01\%$ | 1 | $<0.01\%$ |
| **RETRIEVAL** | 153,451 | 62.35 % | 144,625 | 58.76 % |
| **BUDGET_PRUNE** | 0 | 0.00 % | 0 | 0.00 % |
| **FEATURES** | 0 | 0.00 % | 0 | 0.00 % |
| **RANKER** | 0 | 0.00 % | 0 | 0.00 % |
| **DECISION** | 0 | 0.00 % | 0 | 0.00 % |
| **RECOVERED (NONE)** | 92,682 | 37.65 % | 101,508 | 41.24 % |
| **Total True Pairs Evaluated** | **246,134** | **100.00 %** | **246,134** | **100.00 %** |

---

## 8. Train Residual Diagnosis (Train Split)

Evaluated strictly on `train` queries ($N_{pairs} = 672,540$).

### Distance & Attribute Cross-Tabulation (Residual True Pairs)
- **Haversine Distance $< 100\text{ m}$:** 48.2 % of unrecovered pairs (differing name spelling/transliteration).
- **Haversine Distance $< 1\text{ km}$:** 24.1 % of unrecovered pairs.
- **Haversine Distance $< 10\text{ km}$:** 15.3 % of unrecovered pairs.
- **Haversine Distance $\ge 10\text{ km}$:** 12.4 % of unrecovered pairs.
- **Exact Phone Match in Residual:** 4,128 unrecovered true pairs share an identical digits-only phone number (missed by geo/name keys due to coordinate distance).
- **Exact URL Domain Match in Residual:** 2,914 unrecovered true pairs share an identical registrable URL domain.

---

## 9. Pre-Registrations for Phase 3 Retrieval Experiments

1. **Dual Dataset Prerequisite:** Retrieval channels (IDF lexical, address/geo, character n-gram) will be run on **BOTH** GLEIF x CH and Foursquare datasets. Conclusions require consistency across both datasets.
2. **Bootstrap CI Threshold:** Performance differences below the 95% bootstrap CI width will not be claimed as significant.
3. **Matched Budget Comparison:** Primary comparison will evaluate recall at matched candidate budgets against the control efficient frontier and measure recovery on the residual slice.
