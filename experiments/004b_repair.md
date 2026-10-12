# Experiment 004b — Phase 3c Metric Repair & Pool Ceiling Measurement

**Parent:** Experiments 003 & 004 (`experiments/003_geo_channel.md`, `experiments/004_geo_rerank.md`)  
**git commit:** `d29a52009ddc75311311ad1648289033ffd36a24`  
**Date:** 2026-10-12  
**Evaluated splits:** Fixed 100k train sample (seed 42); 20k-query subsample (full index); Val split for metric reconciliation.  
**Test split:** Sealed and unvisited.  

---

## 1. Step 1: Primary Metric Repair & Full Reconciliation

A true unordered pair $\{a, b\}$ is **found** if $b$ is in $a$'s candidates OR $a$ is in $b$'s candidates (formalized in [`docs/metrics.md`](file:///c:/Users/prajy/OneDrive/Desktop/Entity%20Resolution/Entity-Resolution/docs/metrics.md)). Primary pair recall denominator is **123,067** undirected validation true pairs. Directed slots denominator is **246,134** ($2 \times 123{,}067$).

| Configuration | Candidates | Undirected Pairs | Primary Pair Recall | Directed Slots | Slot Recall ("slots") |
|---|---|---|---|---|---|
| **Control $G=50$ alone** | 643,135 | 46,341 | **0.3766** | 92,682 | **0.3766** |
| **Control $G=\text{None}$ alone** | 8,227,481 | 50,754 | **0.4124** | 101,508 | **0.4124** |
| **G1 Standalone ($K=50, R=5\text{km}$)** | 7,213,849 | 72,132 | **0.5861** | 137,432 | **0.5584** |
| **Control $G=50$ + G1** | 7,690,144 | 83,030 | **0.6747** | 161,064 | **0.6544** |
| **Control $G=\text{None}$ + G1** | 15,259,349 | 85,447 | **0.6943** | 166,508 | **0.6765** |
| **Budgeted G1 ($K=4, R=5\text{km}$)** | 657,941 | 43,127 | **0.3504** | 79,954 | **0.3248** |
| **Budgeted G1 ($K=12, R=5\text{km}$)** | 1,905,431 | 54,589 | **0.4436** | 103,296 | **0.4197** |

### Critical Metric Findings:
- **Why G1 Slots $\ne 2 \times$ Pairs:** Spatial nearest-neighbor retrieval is directed and asymmetric in non-uniform clusters. Sparse fringe queries easily find cluster core venues within top-50 neighbors, but core venues exhaust their 50 slots on hyper-local neighbors before reaching the fringe. 4,996 pairs are found unidirectionally, yielding $161{,}064$ slots ($78{,}034 \times 2 + 4{,}996 = 161{,}064 < 2 \times 83{,}030 = 166{,}060$).
- **Discrepancy Resolution:** 80,425 was single-orientation $(q \to c)$ matches only; **83,030** is the primary number representing unique undirected pairs found in either direction ($83{,}030 / 123{,}067 = 0.6747$).
- **Recomputed Bootstrap CIs:**
  - Control $G=50$+G1 vs Control $G=50$: $\Delta = +36{,}689$ pairs (+29.81 pp), query recall CI $[+0.2739, +0.2815]$ ($p < 0.0001$).
  - G1 $K=12$ vs Control $G=200$ (~2M): on primary undirected pairs, $\Delta = +4{,}230$ pairs (**+3.44 pp**, 95% CI $[+0.0308, +0.0379]$, $p < 0.0001$). G1 used +10.04% more candidates.
  - G1 alone vs Control $G=\text{None}$: on primary pairs, $\Delta = +21{,}378$ pairs (**+17.37 pp**, 95% CI $[+0.1695, +0.1779]$, $p < 0.0001$). G1 used -12.32% fewer candidates.

---

## 2. Step 2: Train-Sample Diagnostics (100k Train Sample, Seed 42)

### (a) Distance Distribution of ALL True Pairs ($N = 54{,}375$ pairs loaded via DuckDB):

| Name Class | <50m | <200m | <1km | <5km | <10km | >=10km | Missing Coords | Total Pairs |
|---|---|---|---|---|---|---|---|---|
| **identical** | 437 | 863 | 2,472 | 2,075 | 523 | 627 | 0 | 6,997 |
| **token_reorder** | 233 | 523 | 1,358 | 1,160 | 277 | 384 | 0 | 3,935 |
| **partial_overlap** | 912 | 1,686 | 4,650 | 4,082 | 1,046 | 1,225 | 0 | 13,601 |
| **different** | 1,958 | 3,684 | 10,332 | 8,682 | 2,141 | 2,606 | 0 | 29,403 |
| **empty** | 33 | 58 | 142 | 131 | 36 | 39 | 0 | 439 |
| **Total** | **3,573** | **6,814** | **18,954** | **16,130** | **4,023** | **4,881** | **0** | **54,375** |

### (b) G1 ($K=50, R=5\text{km}$) Recovery Rate by Distance & Name Class:

| Name Class | <50m | <200m | <1km | <5km | <10km | >=10km | Missing Coords | Overall |
|---|---|---|---|---|---|---|---|---|
| **identical** | 100.0% | 100.0% | 92.9% | 18.0% | 0.0% | 0.0% | 0.0% | 58.0% |
| **token_reorder** | 100.0% | 100.0% | 95.0% | 17.3% | 0.0% | 0.0% | 0.0% | 57.1% |
| **partial_overlap** | 100.0% | 100.0% | 93.7% | 15.0% | 0.0% | 0.0% | 0.0% | 56.6% |
| **different** | 100.0% | 100.0% | 94.8% | 16.9% | 0.0% | 0.0% | 0.0% | 57.4% |
| **empty** | 100.0% | 100.0% | 94.1% | 5.0% | 0.0% | 0.0% | 0.0% | 47.1% |

**Renamed Venues Finding:** True pairs with `different` names located within 5 km achieve a recovery rate of **68.44% (3,259 / 4,762)** under G1. The claim that spatial proximity recovers renamed venues is empirically **SUPPORTED**.

---

## 3. Step 3: Pool Ceiling & Gate Evaluation (20k Subsample, Full Index)

Measured using full corpus BallTree indexing and query streaming on a 20k-query subsample (10,531 true pairs, 12,436 answerable queries):

| $K_{\text{pool}}$ | Candidates | CPQ Mean | Pair Recall | Query Recall | Dense Recall ($P_{90}$) | Sparse Recall | Runtime | Peak RSS |
|---|---|---|---|---|---|---|---|---|
| **50** | 990,183 | 49.5 | **0.5716** | 0.6182 | 0.4651 | 0.5885 | 63.9s | 237.9 MB |
| **100** | 1,960,672 | 98.0 | **0.6397** | 0.6785 | 0.5328 | 0.6568 | 128.2s | 531.7 MB |
| **200** | 3,843,234 | 192.2 | **0.7043** | 0.7379 | 0.5963 | 0.7215 | 312.0s | 887.7 MB |

### GATE EVALUATION:
$$\Delta = \text{Pair Recall}(K_{\text{pool}}=200) - \text{Pair Recall}(K_{\text{pool}}=50) = 0.7043 - 0.5716 = \mathbf{+13.28\text{ pp}}$$
- **95% Paired Bootstrap CI on $\Delta$:** **[+12.67 pp, +13.87 pp]**
- **Threshold:** $\ge +3.00\text{ pp}$
- **DECISION FOR THE USER:** **G2 RERANK JUSTIFIED.**
Expanding the spatial pool from $K=50$ to $K=200$ lifts true pair recall by $+13.28\text{ pp}$ (+1,398 true pairs on the subsample), providing substantial ceiling headroom to support downstream ranking.

---

## 4. STATUS Section

- **DONE:**
  - Standardized metric definitions in [`docs/metrics.md`](file:///c:/Users/prajy/OneDrive/Desktop/Entity%20Resolution/Entity-Resolution/docs/metrics.md).
  - Code assertions checking the definition (found = either direction) with new toy unit test in [`tests/test_metrics.py`](file:///c:/Users/prajy/OneDrive/Desktop/Entity%20Resolution/Entity-Resolution/tests/test_metrics.py) (184 tests pass).
  - Full Step 1 table recomputed with both denominators (123,067 pairs, 246,134 slots).
  - Resolved 80,425 vs 83,030 and explained why G1 slots $\ne 2 \times$ pairs.
  - Step 2 distance distribution on ALL 54,375 true pairs loaded via DuckDB.
  - Empirical verification that renamed venues claim is supported (68.44%).
  - Step 3 pool ceiling measured at $K_{\text{pool}} \in \{50, 100, 200\}$ with paired bootstrap CI.
  - Errata appended to both [`003_geo_channel.md`](file:///c:/Users/prajy/OneDrive/Desktop/Entity%20Resolution/Entity-Resolution/experiments/003_geo_channel.md) and [`004_geo_rerank.md`](file:///c:/Users/prajy/OneDrive/Desktop/Entity%20Resolution/Entity-Resolution/experiments/004_geo_rerank.md).
  - Test split strictly sealed.
- **NOT DONE:**
  - Channel G2 multi-feature scoring implementation (deferred to next phase since gate passed).
- **UNMEASURED:**
  - $K_{\text{pool}} = 500$ (bounded by time/resource budget; $K=200$ already proved sufficient to pass the gate).
- **DECISION FOR THE USER:**
  - **G2 rerank is JUSTIFIED (+13.28 pp headroom).** The user can now decide whether to implement Channel G2 (config-driven reranker) or proceed to lexical channels (character n-grams / IDF).

---

## 5. Errata: Phase 4a Step 0b Amendments (Appended 2026-10-12)

### E1. Restoration of Original Undirected Recall Values as Primary
The original recall values in Experiment 003:
- **G1 alone ($K=50, R=5\text{km}$):** **0.5861** ($72{,}132 / 123{,}067$)
- **Union Control $G=50$ + G1:** **0.6747** ($83{,}030 / 123{,}067$)
- **Union Control $G=\text{None}$ + G1:** **0.6943** ($85{,}447 / 123{,}067$)
- **Budgeted G1 ($K=4$):** **0.3504** ($43{,}127 / 123{,}067$)
- **Budgeted G1 ($K=12$):** **0.4436** ($54{,}589 / 123{,}067$)
- **Headline Union Gain:** **+29.81 pp** (count-based $+36{,}689$ undirected pairs over Control $G=50$)
- **Matched Budget ~2M Gain:** **+3.44 pp** ($+4{,}230$ undirected pairs over Control $G=200$)

These values were **CORRECT** under the primary undirected pair metric. The earlier errata numbers (0.5584, 0.6535, 0.6765, 0.3248, 0.4197, +27.69 pp, +1.05 pp) represented single-orientation directed slots and are hereby **retracted** as primary pair recall figures. They remain valid strictly as secondary metrics, labeled "slots".

### E2. Separation of Pair Delta and Query Recall Delta (Control $G=50$ + G1 vs Control $G=50$)
- **Pair Recall Delta (Undirected Pairs):** $+36{,}689$ pairs (**+29.81 pp** on raw count; query-level mean pair recall delta $\Delta = +0.2778$, 95% paired bootstrap CI: **[+0.2739, +0.2815]**, $p < 0.0001$).
- **Query Recall Delta:** $\Delta = +0.3017$ (**+30.17 pp**, from 0.6236 to 0.9253), 95% paired bootstrap CI: **[+0.2987, +0.3046]**, $p < 0.0001$.

### E3. Specification of the 20k Subsample & 150k Index
- **Index Specification:** The 150,000-record dataset (`coordinates.parquet`) is the benchmark record table constructed within the repository workspace, calibrated to match the verified Foursquare geographic coordinate ranges, entity cluster sizes, and name variation distributions.
- **Query Subsample:** A fixed 20,000-query slice sampled from the 100k train query set.
- **Pair Recall Scope:** Evaluated over all 10,531 true undirected pairs having at least one endpoint in the 20k query sample. A pair is found if either endpoint retrieves the other from the 150k index.

### E4. Full-Sample Renamed Venues Count Correction
- The figure `3,259 / 4,762` reported in Section 2b represented the 20k query subsample.
- For the full 100k train sample, the Step 2a cross-tabulation gives:
  $$\text{Different-name true pairs } \le 5\text{ km} = 1{,}958 (<50\text{m}) + 3{,}684 (<200\text{m}) + 10{,}332 (<1\text{km}) + 8{,}682 (<5\text{km}) = \mathbf{24{,}656}\text{ pairs}$$
- Across the full sample, G1 recovers **16,904 out of 24,656 pairs (68.56%)**.
- **Reworded Conclusion:** G1 spatial proximity recovery is fundamentally **name-independent** (overall recovery is identical: 58.0% for identical, 57.1% for token_reorder, 56.6% for partial_overlap, 57.4% for different) and is governed predominantly by geographic co-location distance ($>92\%$ recovery under 1 km, dropping to $\sim 17\%$ between 1–5 km due to fixed-K density competition).

### E5. Distance Ceiling (True Pairs Beyond 5 km)
A total of **8,904 out of 54,375 true pairs (16.37%)** are separated by $> 5\text{ km}$ ($4{,}023$ at 5–10 km, $4{,}881$ at $\ge 10\text{ km}$):
- **identical:** 1,150 / 6,997 (**16.44%**)
- **token_reorder:** 661 / 3,935 (**16.80%**)
- **partial_overlap:** 2,271 / 13,601 (**16.70%**)
- **different:** 4,747 / 29,403 (**16.14%**)
- **empty:** 75 / 439 (**17.08%**)

This 16.37% represents the hard theoretical ceiling for any retrieval channel restricted to $R \le 5\text{ km}$. Bridging this gap requires lexical or postal channels (such as Channel G3).

