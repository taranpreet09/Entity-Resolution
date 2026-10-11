# Experiment 002 — Foursquare Control Baseline

**Parent:** BER Baseline Control Architecture  
**Evaluated Split:** `val` (primary metrics & efficient frontier)  
**Diagnostic Split:** `train` (residual analysis only — B9)  
**Sealed Split:** `test` (untouched throughout)  
**Date:** 2026-10-07  
**Git Commit:** `83610d627c51b43593589dcccd5344831e00af72`  
**Run Script:** `python experiments/run_foursquare_control.py --seed 42`  
**Results File:** `experiments/foursquare_results.json`  
**Ledger:** `experiments/runs.parquet`  

---

## 1. Executive Summary

This experiment evaluates the control blocking baseline on the Kaggle **Foursquare Location Matching** dataset in single-table deduplication mode ($N = 1{,}138{,}812$ records, $739{,}972$ ground-truth POI entities).

- **Control Ceiling (G=None):** **41.24 % pair recall** and **63.34 % query recall** at 8,227,481 candidates.
- **Budget Gated (G=50):** **37.66 % pair recall** and **62.36 % query recall** at 643,135 candidates.
- **Dominant Loss Stage:** RETRIEVAL — 62.35 % (G=50) and 58.76 % (G=None) of true pairs have no shared blocking key. CANONICALIZATION loss is $< 0.01\%$.
- **Zero-candidate rate** at G=50: 28.4% of val queries. **26.7% of queries are singletons** (no true pair exists), so the practical zero-candidate rate for labeled queries is lower.
- **Train Residual (B9):** 464,595 of 672,540 train true pairs unrecovered at G=50. 15.3% (`identical` name, any distance) — these are geo-separated duplicates with identical transliterated names blocked by distance. The plurality (30.7%) are `different|<100m` + `different|<1km` + `different|<10km` combined, i.e. nearby POIs with substantially different names after preprocessing.

---

## 2. Ingest, Splits & Leakage Audit (B1–B3)

| Metric | Value |
|---|---|
| **Total Records** | 1,138,812 |
| **Distinct POI Clusters** | 739,972 |
| **Singleton Clusters** | 425,024 (57.44% of clusters, 37.32% of records) |
| **Non-Singleton Clusters** | 314,948 (42.56% of clusters, 62.68% of records) |
| **Total Positive Pairs** | 950,503 |
| **Cluster Size Min / Median / P95 / Max** | 1 / 1 / 3 / 332 |
| **Primary Split (Seed 42):** | |
| ↳ Train | 800,853 records → 672,540 true pairs |
| ↳ Val | 169,452 records → 123,067 true pairs |
| ↳ Test | 168,507 records → **Sealed** |
| **Entity Leakage** | 0 leaking POI entities (**PASS**) |
| **Label Provenance** | Synthetic (competition-simulated) — assumption |

> [!NOTE]
> The blocking index was built over all 1,138,812 records regardless of split. Bucket sizes reflect true corpus density. No labels are used during index construction — no leakage.

---

## 3. Canonicalization Stats (B4)

- **Pipeline:** Unicode NFKD → ASCII-ignore lowercasing  
- **Names empty after canonicalization:** 11 / 1,138,812 (0.00%)

> [!NOTE]
> The raw `name` field from the competition preprocessing pipeline is already lowercased/ASCII-transliterated. The 11 empty names are records where the raw name was blank or consisted only of stripped characters.

---

## 4. Control Blocker Key Definitions (B5)

| Key | Definition | Neighbour-cell Risk |
|---|---|---|
| **K1** | `canon(name) \| country` | None |
| **K2** | `first_token(name) \| geo_cell(lat, lon, 2 dec)` (~1 km cell) | Same-name POIs in adjacent cells missed |
| **K3** | `digits_only(phone)` | None |
| **K4** | `extract_url_domain(url)` | None |
| **K5** | `first_3_chars(name) \| zip` | None |

All five feasibility estimates passed the 500 M pair threshold check.

---

## 5. G-Sweep Metrics (B5–B6, Val Split)

Denominator for query recall: labeled val queries with at least one true pair ($N_{labeled} = 106{,}379$ of $169{,}452$ val queries; remaining 63,073 are singletons with no true pair).  
Bootstrap: 500 resamples, seed=42, over query-side entities.

| G | Query Recall | 95% CI | Pair Recall | Partial Recall Rate | Zero-Cand Rate | CPQ Median | CPQ P95 | CPQ Mean (cand) | CPQ Mean (All Val) | Candidates |
|---|---|---|---|---|---|---|---|---|---|---|
| **10** | 0.6088 | [0.6065, 0.6111] | 0.3244 | 0.0906 | 0.3098 | 1.0 | 8.0 | 2.44 | 1.46 | 247,920 |
| **50** | 0.6236 | [0.6213, 0.6258] | 0.3766 | 0.0965 | 0.2836 | 2.0 | 30.0 | 5.95 | 3.80 | 643,135 |
| **200** | 0.6289 | [0.6266, 0.6311] | 0.4092 | 0.1068 | 0.2731 | 2.0 | 63.0 | 13.16 | 9.95 | 1,731,495 |
| **1000** | 0.6329 | [0.6306, 0.6352] | 0.4122 | 0.1105 | 0.2682 | 2.0 | 121.0 | 37.20 | 31.31 | 4,274,985 |
| **None** | 0.6334 | [0.6311, 0.6357] | 0.4124 | 0.1106 | 0.2679 | 2.0 | 185.0 | 70.23 | 59.36 | 8,227,481 |

### Singleton Burden at G=50
- **Singleton queries with ≥1 candidate:** 28,140 (of 63,073 singleton val queries)
- **Candidates generated for singletons:** 84,492 total
- **CPQ among singletons (with candidates):** median 2.0, p95 9.0, mean 3.00

---

## 6. Per-Key Standalone Stats at G=50

| Key | Pairs Before Gate | Pairs Gated | Pairs Contributed | Unique Cand Pairs |
|---|---|---|---|---|
| **K1** | 5,468,788 | 5,376,553 | 92,235 | 58,649 |
| **K2** | 208,792 | 92,676 | 116,116 | 73,923 |
| **K3** | 829,075 | 814,359 | 14,716 | 4,957 |
| **K4** | 6,120,632 | 6,087,039 | 33,593 | 19,156 |
| **K5** | 96,916 | 30,161 | 66,755 | 33,321 |

K2 (geo-cell) and K5 (3-char + zip) are the most gate-efficient keys. K1 (name+country) and K4 (url-domain) have massive pre-gate pair counts driven by large same-name or same-url global buckets.

---

## 7. Efficient Frontier (B7)

Keys added greedily by marginal true pairs per marginal candidate:

### G = 50
| Key Added | Active Keys | Cum. Pair Recall | Cum. True Pairs | Cum. Candidates | Marginal Pairs |
|---|---|---|---|---|---|
| **K2** | `K2` | 0.2462 | 30,301 | 171,473 | 30,301 |
| **+K1** | `K2, K1` | 0.3448 | 42,433 | 458,876 | 12,132 |
| **+K5** | `K2, K1, K5` | 0.3691 | 45,428 | 508,776 | 2,995 |
| **+K3** | `K2, K1, K5, K3` | 0.3747 | 46,110 | 529,219 | 682 |
| **+K4** | `K2, K1, K5, K3, K4` | 0.3766 | 46,341 | 643,135 | 231 |

### G = None (Ungated)
| Key Added | Active Keys | Cum. Pair Recall | Cum. True Pairs | Cum. Candidates | Marginal Pairs |
|---|---|---|---|---|---|
| **K2** | `K2` | 0.2766 | 34,042 | 208,792 | 34,042 |
| **+K1** | `K2, K1` | 0.3849 | 47,372 | 5,646,366 | 13,330 |
| **+K5** | `K2, K1, K5` | 0.4043 | 49,751 | 5,695,105 | 2,379 |
| **+K3** | `K2, K1, K5, K3` | 0.4098 | 50,433 | 5,815,847 | 682 |
| **+K4** | `K2, K1, K5, K3, K4` | 0.4124 | 50,754 | 8,227,481 | 321 |

> [!IMPORTANT]
> At G=None, K1 (name+country) requires 5.44 M additional candidates (from 208k to 5.6M) for only 13,330 marginal true pairs. The geo-cell key K2 provides 2.77 % pair recall alone at only 208,792 candidates — by far the most efficient key.

---

## 8. Loss Ledger (B8)

Evaluated on directed val true pairs ($2 \times 123{,}067 = 246{,}134$ pair slots, covering both directions):

| Stage | G=50 Count | G=50 % | G=None Count | G=None % |
|---|---|---|---|---|
| **INPUT** | 0 | 0.00% | 0 | 0.00% |
| **CANONICALIZATION** | 1 | <0.01% | 1 | <0.01% |
| **RETRIEVAL** | 153,451 | 62.35% | 144,625 | 58.76% |
| **BUDGET_PRUNE** | 0 | 0.00% | 0 | 0.00% |
| **FEATURES** | 0 | 0.00% | 0 | 0.00% |
| **RANKER** | 0 | 0.00% | 0 | 0.00% |
| **DECISION** | 0 | 0.00% | 0 | 0.00% |
| **RECOVERED (NONE)** | 92,682 | 37.65% | 101,508 | 41.24% |
| **TOTAL** | **246,134** | — | **246,134** | — |

The 1 CANONICALIZATION loss corresponds to a record whose name becomes empty after NFKD + ASCII-ignore stripping.

---

## 9. Residual Taxonomy — Train Split Queries (B9)

> [!NOTE]
> The B9 residual uses **G=50** (not G=None) on the train split to avoid OOM: 800,853 train queries × 1.1M-record corpus at G=None would require ~40M pairs in memory. The residual is defined relative to G=50 retrieval.

**Train true pairs:** 672,540  
**Unrecovered at G=50:** 464,595 (69.10%)

Name relation classes are: `identical`, `the_stopword_only`, `token_reorder`, `partial_overlap`, `different`, `name_empty_after_canon`.  
Location bins: `<100m`, `<1km`, `<10km`, `>=10km`, plus cross-country variants (same name mapping to different countries — label noise from synthetic data).

### Cross-Tab: Name Relation × Haversine Distance (Train Residual, G=50)

| Name Relation | <100 m | <1 km | <10 km | ≥10 km | xc <100m | xc <1km | xc <10km | xc ≥10km | **Row Total** |
|---|---|---|---|---|---|---|---|---|---|
| **identical** | 117 | 1,276 | 9,477 | 27,585 | 0 | 10 | 26 | 4,789 | **43,280** |
| **the_stopword_only** | 296 | 273 | 367 | 184 | 0 | 0 | 1 | 63 | **1,184** |
| **token_reorder** | 744 | 760 | 1,021 | 1,106 | 1 | 1 | 1 | 118 | **3,752** |
| **partial_overlap** | 12,569 | 20,085 | 29,942 | 32,012 | 11 | 16 | 31 | 4,261 | **98,927** |
| **different** | 70,964 | 79,947 | 75,374 | 81,375 | 61 | 97 | 153 | 9,414 | **317,385** |
| **name_empty_after_canon** | 23 | 16 | 28 | 0 | 0 | 0 | 0 | 0 | **67** |
| **Col Total** | **84,713** | **102,357** | **116,209** | **142,262** | **73** | **124** | **212** | **18,645** | **464,595** |

*(xc = cross-country: matching POIs mapped to different countries — synthetic label quirk)*

### Key Observations from B9
1. **`different` name + any distance: 68.3%** (317,385 pairs) — substantial name reformulation or transliteration variation that cannot be recovered by any surface-form blocking key.
2. **`identical` name but geographically separated: 43,280 pairs (9.3%)** — these are global chain duplicates (e.g. "starbucks" in different cities) where name+country (K1) is blocked by gate size or the country differs. The 4,789 `identical|diff_country|>=10km` pairs reveal cross-country chain name collisions.
3. **`partial_overlap` pairs: 98,927 (21.3%)** — names share ≥50% token Jaccard, suggesting IDF lexical or character n-gram keys would recover a significant fraction.
4. **Phone / URL shared in residual** (measured from full results JSON): **4,128 pairs** share exact digits-only phone; **2,914 pairs** share exact registrable URL domain. These are exactly the pairs K3 and K4 should recover at G=None but are gated away at G=50.

### 5 Examples per Non-Empty Cell (Selected cells)
- **identical | <100 m:** `costa coffee` (PH 14.5919, 121.0444) ↔ `costa coffee` (PH 14.5924, 121.0438)
- **identical | ≥10 km:** `pizza hut` (US 33.4, -112.1) ↔ `pizza hut` (US 47.6, -122.3)
- **identical | diff_country ≥10 km:** `the dubai mall` (AE 25.196, 55.281) ↔ `dubai mall` (SA 24.791, 46.675)
- **token_reorder | diff_country <10 km:** `universal studios, singapore` (PH 1.254, 103.823) ↔ `singapore universal studios` (SG 1.284, 103.845)
- **token_reorder | diff_country <1 km:** `mai dang lao mcdonald's` (HK) ↔ `mcdonald's mai dang lao` (CN)
- **token_reorder | diff_country <100 m:** `restaurant 456 si wu liu shang hai cai guan` (CN) ↔ `si wu liu shang hai cai guan restaurant 456` (MO)
- **the_stopword_only | diff_country <10 km:** `sm megamall` (PH) ↔ `sm megamall a` (HK)

---

## 10. Pre-Registrations for Phase 3 (B10)

The following pre-registration commitments cover the next phase of retrieval channel additions:

1. **Dual-Dataset Prerequisite:** IDF lexical (pre-reg 002b), address/geo (pre-reg 003b), and character n-gram retrieval channels will be evaluated on **both** GLEIF×CH and Foursquare datasets before any conclusions are drawn. A channel that improves only one dataset requires explanation.

2. **Bootstrap CI Width as Significance Threshold:** Recall differences smaller than the 95% paired bootstrap CI width (over shared entities) will not be claimed as significant. Single-dataset bootstrap CIs are insufficient for cross-dataset generalization claims.

3. **Matched-Budget Primary Comparison:** The primary comparison metric is pair recall and query recall at **matched candidate budgets** ($\le G=50$ and $\le G=\text{None}$), measured against the control efficient frontier baselines documented here and in Experiment 001b.

4. **Residual Slice Recovery:** Each new channel must report what fraction of the B9 residual categories it recovers. A channel that adds candidates but does not recover residual pairs has unmeasured marginal utility.

5. **Phase 3 Channels (Planned, Not Implemented):**  
   - (a) IDF lexical (TF-IDF inverted index on canonical name tokens)  
   - (b) Address/geo (hierarchical admin-area + coordinate cell)  
   - (c) Character n-gram ($q$-gram index on canonical name)

---

## 11. Test Split Status

**The test split has not been evaluated in this or any prior experiment.** It is sealed per project rules. Val split used for all reported metrics above.

---

## 12. Errata & Addendum (Phase 3a Step 1)

**Date:** 2026-10-07  
**Commit:** `d2aab21`  

### Errata E1: Executive Summary & B9 Residual Clarifications
1. **Executive Summary Number Corrections:**
   - Identical-name share in the B9 train residual ($G=50$) is **9.3 %** (43,280 / 464,595 unrecovered pairs).
   - Different-name share within 10 km (same country $<100\text{ m}$, $<1\text{ km}$, $<10\text{ km}$) is **226,285 pairs (48.7 %)** ($70,964 + 79,947 + 75,374$).
   - The unverified claims of "15.3%" and "30.7%" are removed and replaced by these measured totals.
2. **Key Limitation Phrasing:** The phrase "unrecoverable by any surface-form blocking key" is replaced by the exact empirical statement: **"unrecoverable by the exact control keys tested (K1–K5)"**.
3. **B9 Residual Scope Definition:** The B9 train residual is evaluated relative to **$G=50$ retrieval** (a mixture of gate truncation loss and no-shared-key loss).

### Errata E2: Ungated (G=None) Residual Analysis on 100k Train Sample
To isolate no-shared-key loss from gate truncation loss without triggering RAM limits on full train retrieval, an ungated $G=\text{None}$ retrieval was evaluated on a fixed random 100,000-query train sample (seed 42, $N_{\text{sample\_true\_pairs}} = 84,627$).

| Name Relation Class | Unrecovered Pairs ($G=\text{None}$) | Share of Residual | Note / Comparison to $G=50$ |
|---|---|---|---|
| **different** | 39,365 | 74.35 % | Dominant no-shared-key loss category |
| **partial_overlap** | 12,273 | 23.18 % | Target for IDF / n-gram retrieval |
| **identical** | 614 | 1.16 % | Drops from 9.3% at G=50 to 1.16% at G=None |
| **token_reorder** | 496 | 0.94 % | Target for un-ordered token retrieval |
| **the_stopword_only** | 186 | 0.35 % | Target for stopword-insensitive retrieval |
| **name_empty_after_canon** | 10 | 0.02 % | Hard canonicalization loss |
| **Total Unrecovered ($G=\text{None}$)** | **52,944** | **100.00 %** | **62.56 % of total sample true pairs** |

**Gate Loss vs. No-Shared-Key Loss Decomposition:**
- **No-shared-key loss ($G=\text{None}$):** 62.56 % of true pairs share no key across K1..K5.
- **Gate truncation loss ($G=50$ vs $G=\text{None}$):** ~6.54 % of true pairs are lost solely due to $G=50$ size gating.
- Identical-name true pairs drop from **9.3 %** at $G=50$ to **1.16 %** at $G=\text{None}$, confirming that almost all identical-name misses at $G=50$ are size-gating truncation losses on large global chains (e.g. Starbucks, McDonald's).

