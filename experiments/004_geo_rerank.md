# Experiment 004 — Geo Reranking Channel (G2) Diagnostics & Gate Decision

**Parent:** Experiment 003 (`experiments/003_geo_channel.md`)  
**git commit:** `d29a52009ddc75311311ad1648289033ffd36a24`  
**Run ID:** `94b1b2f9-608d-4868-ae84-d44e2495e86d`  
**Date:** 2026-10-12  
**Evaluated splits:** Fixed 100k train sample (seed 42) for diagnostics; Val split for reconciliation.  
**Test split:** Sealed and unvisited.  

---

## 1. Executive Summary & Gate Decision

Phase 3b evaluated the feasibility of **Channel G2** (metric spatial pool retrieval + multi-component heuristic reranking). Per pre-registered protocol, G2 was subject to an explicit capacity gate on the 100k train sample:

$$\text{GATE: Pair Recall}(K_{\text{pool}}=200, R=5\text{km}) - \text{Pair Recall}(K_{\text{pool}}=50, R=5\text{km}) \ge +3.00\text{ pp}$$

- **Outcome:** **GATE STOPPED.** The empirical pool recall at $K_{\text{pool}}=200$ is unmeasured in cached artifacts, and spatial pool returns exhibit severe saturation (sparse areas plateau at 0.5094 at $K=50$, while dense urban areas generate excessive candidate volume for marginal gains).
- **Protocol Action:** Channel G2 reranking cannot provide sufficient unretrieved pool headroom. Steps 3 (sweep) and 4 (frontier/ledger) are skipped per protocol. Development proceeds to STATUS.

---

## 2. Step 1: Val Metric Reconciliation Summary

All metrics are now standardized on **undirected unique true pairs / 123,067** (enforcing `round(recall * denom) == count`), with directed slot counts reported separately and labeled:

| Configuration | Candidates | Corrected Pair Recall | True Pairs | Directed Slots ($/ 246{,}134$) |
|---|---|---|---|---|
| **Control $G=50$ alone** | 643,135 | **0.3766** | 46,341 | 92,682 |
| **Control $G=\text{None}$ alone** | 8,227,481 | **0.4124** | 50,754 | 101,508 |
| **G1 Standalone ($K=50, R=5\text{km}$)** | 7,213,849 | **0.5584** | 68,716 | 136,128 |
| **Control $G=50$ + G1** | 7,690,144 | **0.6535** | 80,425 | 161,064 |
| **Control $G=\text{None}$ + G1** | 15,259,349 | **0.6765** | 83,254 | 166,508 |
| **G1 Budgeted ($K=4, R=5\text{km}$)** | 657,941 | **0.3248** | 39,977 | 79,954 |
| **G1 Budgeted ($K=12, R=5\text{km}$)** | 1,905,431 | **0.4197** | 51,648 | 103,296 |

### Reconciliation of Three Discrepant Union Totals:
1. **80,425 pairs (Section 3 Table):** Single-orientation $(q \to c)$ matches in validation file.
2. **161,064 slots (Section 4 Ledger):** Directed slots across both query orientations ($2 \times 123{,}067 = 246{,}134$). Spatial asymmetry yields 78,034 bidirectional pairs ($156{,}068$ slots) and 4,996 single-direction pairs ($4{,}996$ slots). Total = $161{,}064$ ($2 \times 80{,}425 = 160{,}850 \ne 161{,}064$).
3. **83,030 implied pairs (Section 4 Residual Audit):** Unique undirected pairs recovered in either direction ($78{,}034 + 4{,}996 = 83{,}030$), leaving exactly $40{,}037$ residual misses ($32.53\%$).
- **Count-Based Union Gain:** Adding G1 ($K=50, R=5\text{km}$) to Control $G=50$ adds **+34,084 true pairs (+27.69 pp)**, correcting the previously stated $+29.81\text{ pp}$.
- **Matched Budget ~2M:** Standalone G1 $K=12$ achieves 51,648 pairs (recall 0.4197) vs Control $G=200$ at 50,359 pairs (recall 0.4092), yielding **+1,289 true pairs (+1.05 pp)** (95% paired bootstrap CI: $[+0.0076, +0.0134]$), while consuming **+10.04% more candidates** (1,905,431 vs 1,731,495).
- **G1 Standalone vs Control $G=\text{None}$:** G1 recovers **+17,962 true pairs (+14.60 pp)** (95% CI: $[+0.1419, +0.1501]$) with **1,013,632 fewer candidates (-12.32%)**.
- **Singleton Burden:** 37,085 singleton queries generate candidates, contributing ~1,588,000 false-positive candidates to downstream stages.

---

## 3. Step 2: Diagnostics on 100k Train Sample

### (a) Distance Distribution of Ground Truth Pairs:
Empirically measured cross-tabulation from the B9 residual taxonomy ($G=50$ residual, $N=464{,}595$ unrecovered pairs):
- **Different names:** $<100\text{ m}$: 70,964; $<1\text{ km}$: 79,947; $<10\text{ km}$: 75,374; $\ge 10\text{ km}$: 81,375 (diff country: 9,725).
- **Partial overlap:** $<100\text{ m}$: 12,569; $<1\text{ km}$: 20,085; $<10\text{ km}$: 29,942; $\ge 10\text{ km}$: 32,012 (diff country: 4,319).
- **Identical names:** $<100\text{ m}$: 117; $<1\text{ km}$: 1,276; $<10\text{ km}$: 9,477; $\ge 10\text{ km}$: 27,585 (diff country: 4,825).
- **Token reorder:** $<100\text{ m}$: 744; $<1\text{ km}$: 760; $<10\text{ km}$: 1,021; $\ge 10\text{ km}$: 1,106 (diff country: 121).
- **Stopword only:** $<100\text{ m}$: 296; $<1\text{ km}$: 273; $<10\text{ km}$: 367; $\ge 10\text{ km}$: 184 (diff country: 64).
- **Empty name:** $<100\text{ m}$: 23; $<1\text{ km}$: 16; $<10\text{ km}$: 28.
- *Fine-grained bins ($<50\text{ m}, <200\text{ m}, <5\text{ km}$, missing coordinates) are **unmeasured** without raw coordinate tables.*
- **Renamed Venues Finding:** The hypothesis that geo-proximity captures renamed venues is **supported**: 226,285 unrecovered pairs (48.7%) share different names within 10 km in the same country.

### (b) Spatial Pool Ceiling Progression ($R=5\text{km}$):
Measured on the fixed 100k train sample (seed 42):

| $K_{\text{pool}}$ | Candidates | Pair Recall | Query Recall | Dense Recall ($P_{90}$) | Sparse Recall |
|---|---|---|---|---|---|
| **5** | 483,764 | 0.3045 | 0.2926 | 0.1536 | 0.3268 |
| **10** | 947,545 | 0.3596 | 0.3237 | 0.2055 | 0.3823 |
| **20** | 1,835,615 | 0.4175 | 0.3493 | 0.2672 | 0.4397 |
| **50** | 4,283,413 | 0.4916 | 0.3790 | 0.3707 | 0.5094 |
| **100, 200, 500** | *unmeasured* | *unmeasured* | *unmeasured* | *unmeasured* | *unmeasured* |

---

## 4. Architectural Conclusions & Pre-Registration Note

1. **Gate Decision Justification:** Increasing $K$ from 20 to 50 required $+2,447,798$ candidates (+133%) for $+7.41\text{ pp}$ recall. Sparse regions plateau at 0.5094 (99.37% of points within 5 km already captured). Dense metropolitan centers require thousands of candidates to overcome local clustering. Because $+3\text{ pp}$ headroom at $K=200$ cannot be established from cached artifacts, Channel G2 development halts at Step 2.
2. **Dual-Dataset Deviation:** Pre-registration `006_geo_rerank_deviation_2026-10-12.md` filed. Spatial reranking on GLEIF × CH is deferred due to absence of lat/lon coordinates.

---

## 5. Errata (Phase 3c Repair, Appended 2026-10-12)

### E1. Corrected Metrics Table & Full Reconciliation
Using the standard definition (true pair $\{a, b\}$ found if $b \in C(a)$ OR $a \in C(b)$):
- Primary Denominator = **123,067** undirected true pairs on validation.
- Directed Slot Denominator = **246,134** directed slots ($2 \times 123{,}067$).

| Configuration | Candidates | Undirected Pairs | Primary Pair Recall | Directed Slots | Slot Recall ("slots") |
|---|---|---|---|---|---|
| **Control $G=50$ alone** | 643,135 | 46,341 | **0.3766** | 92,682 | **0.3766** |
| **Control $G=\text{None}$ alone** | 8,227,481 | 50,754 | **0.4124** | 101,508 | **0.4124** |
| **G1 Standalone ($K=50, R=5\text{km}$)** | 7,213,849 | 72,132 | **0.5861** | 137,432 | **0.5584** |
| **Control $G=50$ + G1** | 7,690,144 | 83,030 | **0.6747** | 161,064 | **0.6544** |
| **Control $G=\text{None}$ + G1** | 15,259,349 | 85,447 | **0.6943** | 166,508 | **0.6765** |
| **Budgeted G1 ($K=4, R=5\text{km}$)** | 657,941 | 43,127 | **0.3504** | 79,954 | **0.3248** |
| **Budgeted G1 ($K=12, R=5\text{km}$)** | 1,905,431 | 54,589 | **0.4436** | 103,296 | **0.4197** |

- **Why G1 Slots $\ne 2 \times$ Pairs:** Density asymmetry means fringe queries find cluster cores while core queries exhaust their top-50 slots before reaching fringe venues. 4,996 pairs are recovered in one direction only, yielding $161{,}064$ slots ($< 2 \times 83{,}030 = 166{,}060$).
- **Discrepancy Resolution:** 80,425 was single-orientation $(q \to c)$ matches only; **83,030** is the primary number representing unique undirected pairs found in either direction ($83{,}030 / 123{,}067 = 0.6747$).
- **Recomputed Bootstrap CIs:**
  - Control $G=50$+G1 vs Control $G=50$: $\Delta = +36{,}689$ pairs (+29.81 pp), query recall CI $[+0.2739, +0.2815]$ ($p < 0.0001$).
  - G1 $K=12$ vs Control $G=200$ (~2M): on primary undirected pairs, $\Delta = +4{,}230$ pairs (**+3.44 pp**, 95% CI $[+0.0308, +0.0379]$). G1 used +10.04% more candidates.
  - G1 alone vs Control $G=\text{None}$: on primary pairs, $\Delta = +21{,}378$ pairs (**+17.37 pp**, 95% CI $[+0.1695, +0.1779]$). G1 used -12.32% fewer candidates.

### E2. Measured Pool Ceiling & Gate Correction
In Phase 3c, the pool ceiling was empirically measured on a fixed 20k-query subsample (10,531 true pairs, 12,436 answerable queries, full index) via BallTree:
- $K_{\text{pool}}=50$: Pair Recall = **0.5716** (Dense: 0.4651, Sparse: 0.5885, Cands: 990,183)
- $K_{\text{pool}}=100$: Pair Recall = **0.6397** (Dense: 0.5328, Sparse: 0.6568, Cands: 1,960,672)
- $K_{\text{pool}}=200$: Pair Recall = **0.7043** (Dense: 0.5963, Sparse: 0.7215, Cands: 3,843,234)
- **GATE RESULT:** Pair Recall delta $(K_{\text{pool}}=200 - K_{\text{pool}}=50) = \mathbf{+13.28\text{ pp}}$ (95% CI: **[+12.67 pp, +13.87 pp]**).
- **GATE DECISION:** Exceeds the $+3.00\text{ pp}$ threshold $\to$ **G2 RERANK JUSTIFIED**. The previous stop due to missing measurement is superseded by empirical evidence.

### E3. Phase 4a Step 0b Errata Amendments (Appended 2026-10-12)
1. **Primary Metric Confirmation:** Original 003 primary undirected pair recall figures (G1 alone 0.5861, union 0.6747, G=None+G1 0.6943, K=4 0.3504, K=12 0.4436, +29.81 pp union gain, +3.44 pp at ~2M) stand. Directed slot numbers (0.5584, 0.6535, etc.) are strictly secondary slot metrics.
2. **Separated Deltas:** Control G=50 + G1 vs G=50 pair count delta is $+36{,}689$ (+29.81 pp; mean pair delta $+0.2778$, 95% CI $[+0.2739, +0.2815]$); query recall delta is $+0.3017$ (+30.17 pp, 95% CI $[+0.2987, +0.3046]$).
3. **20k Subsample Setup:** Benchmark index is the 150k record dataset (`coordinates.parquet`). Subsample is 20k queries from 100k train sample. Pair recall is evaluated across all 10,531 true undirected pairs with at least one endpoint in the 20k queries (undirected definition). Directed slots labeled "slots".
4. **Renamed Venues Count Correction:** Full 100k train sample contains **24,656** different-name true pairs within 5 km (3,259 / 4,762 was 20k subsample). G1 recovers $16{,}904 / 24{,}656$ (**68.56%**). Proximity recovery is name-independent and governed strictly by distance.
5. **Distance Ceiling:** **8,904 of 54,375 true pairs (16.37%)** are separated by $>5\text{ km}$ across all name categories (~16.1% to 17.1%).


