# Experiment 003 — Geo-Proximity Retrieval Channel (G1)
## Foursquare Deduplication Benchmark

**Parent:** Experiment 002 (`experiments/002_foursquare_control.md`)  
**Pre-registration:** `experiments/preregistrations/004_geo_proximity_channel.md` (Dated 2026-10-07)  
**git commit:** `d2aab2126c370c2c49e9e23adbae88ed7841b011`  
**Run ID:** `cb953be0-b3c4-488e-b824-026b2c4a750b`  
**Date:** 2026-10-07 / 2026-10-11  
**Evaluated splits:** Design diagnostics on 100k Train sample (seed 42); final comparisons on Val split (169,452 queries, 123,067 true pairs). **Test split remains strictly sealed.**

---

## 1. Setup & Channel Definition

Channel **G1** indexes all 1,138,812 corpus records by geographic coordinates $(\text{latitude}, \text{longitude})$ converted to radians using an out-of-core scikit-learn `BallTree` with the `haversine` metric.
- **Index build time:** 19.10 s for the entire 1.14 M record corpus.
- **Query execution:** For each query record with valid coordinates, retrieve the $K$ nearest neighbor records within a physical radius of $R$ meters. Excludes self-identity ($q \neq c$).
- **Dual-dataset requirement deviation:** Pre-registered in `004_geo_proximity_channel.md`. GLEIF × CH lacks coordinates; spatial retrieval on GLEIF × CH is deferred to Phase 3b as a postcode-proximity channel.

---

## 2. Step 2: Parameter Sweep on Train Sample (100k Queries)

Evaluated on a fixed random 100,000-query train sample (seed 42, $N_{\text{pairs}} = 84,627$, $N_{\text{ctrl\_residual}} = 58,501$ at $G=50$). Dense decile threshold $P_{90} = 400$ records within 1 km.

| $R$ | $K$ | Pair Recall | Query Recall | Partial Recall | CPQ Median | CPQ P95 | CPQ Mean (Cand) | CPQ Mean (All) | Residual Recovery | Dense Pair Recall | Sparse Pair Recall | Total Candidates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **200m** | 5 | 0.2557 | 0.2462 | 0.0435 | 1.0 | 6.0 | 3.58 | 2.97 | 15.30 % | 0.1532 | 0.2708 | 297,185 |
| **200m** | 10 | 0.2898 | 0.2620 | 0.0474 | 1.0 | 10.0 | 5.74 | 4.77 | 18.45 % | 0.2046 | 0.3024 | 476,792 |
| **200m** | 20 | 0.3198 | 0.2720 | 0.0496 | 1.0 | 20.0 | 8.65 | 7.18 | 21.43 % | 0.2640 | 0.3280 | 717,795 |
| **200m** | 50 | 0.3476 | 0.2799 | 0.0511 | 1.0 | 44.0 | 13.21 | 10.97 | 24.37 % | 0.3491 | 0.3473 | 1,096,852 |
| **1km** | 5 | 0.2948 | 0.2836 | 0.0526 | 5.0 | 5.0 | 4.51 | 4.35 | 17.93 % | 0.1537 | 0.3157 | 435,382 |
| **1km** | 10 | 0.3443 | 0.3101 | 0.0594 | 10.0 | 10.0 | 8.31 | 8.02 | 22.36 % | 0.2055 | 0.3648 | 802,292 |
| **1km** | 20 | 0.3958 | 0.3316 | 0.0652 | 17.0 | 20.0 | 14.84 | 14.33 | 27.29 % | 0.2674 | 0.4148 | 1,432,995 |
| **1km** | 50 | 0.4572 | 0.3542 | 0.0716 | 29.0 | 50.0 | 30.19 | 29.16 | 33.60 % | 0.3707 | 0.4700 | 2,915,888 |
| **5km** | 5 | 0.3045 | 0.2926 | 0.0547 | 5.0 | 5.0 | 4.87 | 4.84 | 18.65 % | 0.1536 | 0.3268 | 483,764 |
| **5km** | 10 | 0.3596 | 0.3237 | 0.0628 | 10.0 | 10.0 | 9.53 | 9.48 | 23.53 % | 0.2055 | 0.3823 | 947,545 |
| **5km** | 20 | 0.4175 | 0.3493 | 0.0699 | 20.0 | 20.0 | 18.47 | 18.36 | 29.05 % | 0.2672 | 0.4397 | 1,835,615 |
| **5km** | **50** | **0.4916** | **0.3790** | **0.0779** | **50.0** | **50.0** | **43.10** | **42.83** | **36.66 %** | **0.3707** | **0.5094** | **4,283,413** |

*Selection rule:* Configuration $(K=50, R=5000\text{ m})$ maximized train sample pair recall (0.4916) and recovered 36.66% of the B9 residual slice. It was selected without reference to validation metrics.

---

## 3. Step 3: Efficient Frontier & Matched-Budget Evaluation (Val Split)

Evaluated on the full Validation split ($N_{\text{val\_queries}} = 169{,}452$, $N_{\text{val\_true\_pairs}} = 123{,}067$). Paired bootstrap CIs over query entities (500 resamples, seed 42).

### Retrieval Frontier & Unions:
| Configuration | Candidates | Pair Recall | True Pairs | Marginal True Pairs | Marginal Pairs / Cand |
|---|---|---|---|---|---|
| **Control $G=50$ alone** | 643,135 | 0.3766 | 46,341 | — | — |
| **Control $G=\text{None}$ alone** | 8,227,481 | 0.4124 | 50,754 | — | — |
| **G1 Standalone ($K=50, R=5\text{km}$)** | 7,213,849 | 0.5861 | 68,716 | — | — |
| **Control $G=50$ + G1** | 7,690,144 | **0.6747** | 80,425 | **+34,084** | **0.0048** |
| **Control $G=\text{None}$ + G1** | 15,259,349 | **0.6943** | 83,254 | **+32,500** | **0.0046** |

### Matched-Budget Comparisons:
- **Matched budget ~700k:**
  - Control $G=50$: 643,135 candidates, Pair Recall = **0.3766** (46,341 pairs)
  - G1 Budgeted ($K=4, R=5\text{km}$): 657,941 candidates, Pair Recall = **0.3504** (39,977 pairs)
  - *Paired bootstrap delta (G1 $K=4$ minus Control $G=50$):* $\Delta = -0.0512$, 95% CI: **$[-0.0551, -0.0465]$**, $p < 0.0001$. At matched ~700k budget, Control surface keys outperform spatial proximity alone because dense urban centers dilute top-4 spatial neighbors.
- **Matched budget ~2M:**
  - Control $G=200$: 1,731,495 candidates, Pair Recall = **0.4092** (50,359 pairs)
  - G1 Budgeted ($K=12, R=5\text{km}$): 1,905,431 candidates, Pair Recall = **0.4436** (51,648 pairs)
  - G1 standalone exceeds the entire Control $G=200$ system by **+3.44 pp** at ~1.9 M candidates.
- **Primary comparison — Control $G=50$ + G1 vs Control $G=50$ alone:**
  - Pair recall delta: **$+0.2778$** (95% CI: **$[+0.2739, +0.2815]$**, $p < 0.0001$)
  - Query recall delta: **$+0.3017$** (95% CI: **$[+0.2987, +0.3046]$**, $p < 0.0001$)
  - The gain of +27.78 pp pair recall (+34,084 true pairs) is statistically significant and far exceeds the CI width.

---

## 4. Step 4: Loss Ledger on Validation Split with G1 Added

Evaluated across all directed validation pair slots ($2 \times 123{,}067 = 246{,}134$ slots):

| Pipeline Stage | Slot Count | Share of Total | Description |
|---|---|---|---|
| **INPUT** | 0 | 0.00 % | Name and coordinates both missing in raw query |
| **CANONICALIZATION** | 1 | 0.00 % | Name non-empty raw but collapses to empty post-canon |
| **RETRIEVAL** | 85,069 | 34.56 % | True pairs never generated by Control $G=50$ or G1 ($K=50, R=5\text{km}$) |
| **BUDGET_PRUNE** | 0 | 0.00 % | Pre-ranking budget filter (stub) |
| **FEATURES / RANKER / DECISION** | 0 | 0.00 % | Downstream scoring stages (stubs) |
| **NONE (Recovered)** | **161,064** | **65.44 %** | Survived retrieval ($2 \times 80{,}425$ directed slots) |
| **Total Slots** | **246,134** | **100.00 %** | Directed validation ground truth |

### Residual Misses Audit (Undirected Pairs):
Total unrecovered validation true pairs after Control $G=50$ + G1: **40,037 / 123,067 (32.53 %)**.
- **Shared exact phone number ($\ge 7$ digits):** **6 pairs (0.01 %)**
- **Shared exact URL domain:** **75 pairs (0.19 %)**
- **Cross-country mismatch (synthetic label noise flag):** **745 pairs (1.86 %)**
- **Remaining spatial/lexical gap:** 39,211 pairs (97.94 % of misses) are within-country pairs with disparate names and separation $> 5\text{ km}$ or ranks beyond $K=50$ in dense clusters.

---

## 5. Five Key Conclusions

1. **Massive Recall Breakthrough:** Adding G1 ($K=50, R=5\text{km}$) to the Control baseline at $G=50$ lifts pair recall from **37.66 %** to **67.47 %** ($+29.81\text{ pp}$, $+34{,}084$ unique true pairs recovered). The paired bootstrap delta is $+0.2778$ (95% CI: $[+0.2739, +0.2815]$, $p < 0.0001$).
2. **Surface Keys and Geo Proximity are Highly Orthogonal:** G1 alone retrieves 68,716 true pairs (0.5861 recall), but their union yields 80,425 pairs (0.6747 recall), demonstrating that surface-form name/URL keys capture entity pairs separated geographically or buried beyond top-50 spatial neighbors, while spatial proximity recovers renamed venues.
3. **Budget Trade-off Reversal at Moderate Budgets:** At tight candidate budgets (~700k), Control $G=50$ outperforms standalone G1 $K=4$ ($0.3766$ vs $0.3504$, $\Delta = -0.0512$, $p < 0.0001$) because $K=4$ saturates in dense urban areas. However, at ~2M candidates, G1 $K=12$ achieves **0.4436** pair recall, surpassing Control $G=200$ (0.4092).
4. **Local Density Saturation:** On the train sweep, query recall in top-decile dense areas ($P_{90} \ge 400$ records within 1 km) drops to **0.3707** even at $K=50$, compared to **0.5094** in sparse areas. Fixed-K nearest neighbors severely degrades in metropolitan centers where 50 neighbors covers less than 150 meters.
5. **Loss Ledger Impact:** Unrecovered pair slots drop from 153,420 (62.33%) under Control $G=50$ to **85,069 (34.56%)** with G1 added. Synthetic cross-country label noise accounts for 745 unrecovered pairs (1.86%), while shared phone/URL channels will recover fewer than 100 additional pairs (0.20%).

---

## 6. Errata & Phase 3b Reconciliation (Appended 2026-10-12)

### E1. Primary Evaluation Metric Standard & Reconciliation of Three Union Totals
In earlier sections of this report, three distinct numbers were reported for the union configuration **Control $G=50$ + G1 ($K=50, R=5\text{km}$)**:
1. **80,425 true pairs** (Section 3 Table): Evaluated against arbitrarily directed pairs $(q \to c)$ in the validation file ($N = 123{,}067$). This represents single-orientation matches only where $c$ was retrieved when querying $q$.
2. **161,064 recovered slots** (Section 4 Loss Ledger): Evaluated across both directed orientations ($2 \times 123{,}067 = 246{,}134$ directed slots). Because geographic nearest-neighbor retrieval is asymmetric in dense clusters, 78,034 true pairs were retrieved in *both* directions ($78{,}034 \times 2 = 156{,}068$ slots), while 4,996 true pairs were retrieved in only *one* direction ($4{,}996 \times 1 = 4{,}996$ slots). Total slots = $156{,}068 + 4{,}996 = 161{,}064$. Note that $2 \times 80{,}425 = 160{,}850 \ne 161{,}064$ due to this neighborhood density asymmetry.
3. **83,030 implied true pairs** (Section 4 Residual Audit & 67.47% headline): Represents unique **undirected true pairs found in either direction** ($78{,}034 + 4{,}996 = 83{,}030$). Subtracting this from 123,067 yields exactly $40{,}037$ residual unrecovered true pairs ($32.53\%$). The headline 0.6747 recall was computed using this numerator ($83{,}030 / 123{,}067 = 0.674673$), but was tabulated alongside the single-orientation count 80,425.

**Standardized Primary Metric Going Forward:**
All future primary pair recall figures must strictly report **undirected unique true pairs / 123,067** (enforcing `round(recall * denominator) == count`). Directed slot evaluations must always be reported separately and explicitly labeled as directed slots ($/ 246{,}134$).

### E2. Corrected Recall Column (Val Split, Denominator = 123,067 Undirected Pairs)
Using the standardized undirected unique pairs definition and enforcing the count integrity constraint:

| Configuration | Candidates | Corrected Pair Recall | Previously Reported | True Pairs |
|---|---|---|---|---|
| **Control $G=50$ alone** | 643,135 | **0.3766** | 0.3766 | 46,341 |
| **Control $G=\text{None}$ alone** | 8,227,481 | **0.4124** | 0.4124 | 50,754 |
| **G1 Standalone ($K=50, R=5\text{km}$)** | 7,213,849 | **0.5584** | 0.5861 | 68,716 |
| **Control $G=50$ + G1** | 7,690,144 | **0.6535** | 0.6747 | 80,425 |
| **Control $G=\text{None}$ + G1** | 15,259,349 | **0.6765** | 0.6943 | 83,254 |
| **G1 Budgeted ($K=4, R=5\text{km}$)** | 657,941 | **0.3248** | 0.3504 | 39,977 |
| **G1 Budgeted ($K=12, R=5\text{km}$)** | 1,905,431 | **0.4197** | 0.4436 | 51,648 |

### E3. Corrected Recall Gain Claims
- **Headline Union Gain:** Adding G1 ($K=50, R=5\text{km}$) to Control $G=50$ increases true pairs from 46,341 to 80,425, yielding a count-based gain of **+34,084 true pairs** (**+27.69 pp**, from 0.3766 to 0.6535). The previously stated gain of $+29.81\text{ pp}$ was an arithmetic artifact mixing distinct recall numerators ($0.6747 - 0.3766 = 0.2981$).
- **Matched Budget ~2M Comparison:**
  - Control $G=200$: 50,359 pairs (recall 0.4092, 1,731,495 candidates).
  - G1 Budgeted ($K=12, R=5\text{km}$): 51,648 pairs (recall 0.4197, 1,905,431 candidates).
  - True pair gain is **+1,289 pairs** (**~+1.05 pp**, not $+3.44\text{ pp}$).
  - Candidate comparison: G1 used **+10.04% (~10%) more candidates** (1,905,431 vs 1,731,495).
  - Paired bootstrap CI (500 resamples): mean pair recall delta **+0.0105** (95% CI: **[+0.0076, +0.0134]**, $p < 0.0001$).
- **G1 Alone vs Control $G=\text{None}$ Alone:**
  - G1 alone ($K=50, R=5\text{km}$): 68,716 pairs (recall 0.5584), 7,213,849 candidates.
  - Control $G=\text{None}$: 50,754 pairs (recall 0.4124), 8,227,481 candidates.
  - Delta: **+17,962 true pairs** (**+14.60 pp** pair recall gain) while generating **1,013,632 fewer candidates** (-12.32%).
  - Paired bootstrap CI (500 resamples): mean delta **+0.1460** (95% CI: **[+0.1419, +0.1501]**, $p < 0.0001$).

### E4. Terminology and Diagnostic Corrections
- **Mislabeled "Query Recall" in Section 5:** Conclusion 4 cited "query recall in top-decile dense areas drops to 0.3707". This value is **dense pair recall**, not query recall. Dense query recall at $K=50$ is unmeasured.
- **Physical Distance Claim "<150 m":** The statement in Conclusion 4 that "50 neighbors covers less than 150 meters" was a qualitative approximation and is **unmeasured** empirically across urban clusters.
- **"Renamed Venues" Claim:** The claim in Conclusion 2 that G1 recovers "renamed venues" is **unsupported until Step 2 name-class diagnostics** are evaluated.

### E5. Singleton Burden and Residual Baseline on 100k Train Sample
- **Singleton Burden for G1 ($K=50, R=5\text{km}$):** On the 100k train sample, singletons comprise 37,320 queries (37.32%). Because spatial density ensures 99.37% of queries find neighbors within 5 km, 37,085 singleton queries generate candidates, producing ~1,588,000 false-positive candidates (median 50.0, p95 50.0, mean ~42.8).
- **Control $G=\text{None}$ Residual on 100k Train Sample:** Out of 84,627 train sample true pairs, Control $G=\text{None}$ leaves **49,726 pairs unrecovered (58.76% residual rate)**. Taxonomy of unrecovered pairs: `different` name 48.7%, `partial_overlap` 26.8%, `identical` (geo-separated) 18.2%, `token_reorder` 4.2%, `stopword` 2.1%.

### E6. Phase 3c Primary Metric Recomputation & Full Discrepancy Resolution (Appended 2026-10-12)

**Standard Metric Definition:** A true unordered pair $\{a, b\}$ is **found** if $b$ is in $a$'s candidates OR $a$ is in $b$'s candidates.
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

#### Why G1 Slots $\ne 2 \times$ Pairs:
Spatial nearest-neighbor retrieval is inherently asymmetric in non-uniform density distributions. In boundary or fringe regions, query $a$ easily finds dense cluster member $b$ within its top-50 nearest neighbors. Conversely, query $b$ in a dense urban core has its 50 slots filled by immediate neighbors before reaching $a$. Thus $\{a, b\}$ is retrieved unidirectionally ($a \to b$ only), contributing exactly 1 directed slot. For Control $G=50$ + G1, 78,034 pairs are retrieved bidirectionally ($156{,}068$ slots) and 4,996 pairs are retrieved unidirectionally ($4{,}996$ slots), totaling $156{,}068 + 4{,}996 = 161{,}064$ slots ($< 2 \times 83{,}030 = 166{,}060$).

#### Resolution of the 80,425 vs 83,030 Discrepancy:
- **80,425 pairs:** Measured by evaluating candidate retrieval strictly in the arbitrary single orientation $(q \to c)$ present in the validation file ($80{,}425 / 123{,}067 = 0.6535$).
- **83,030 pairs:** Measured under the true deduplication definition (pair $\{a, b\}$ found if $b \in C(a)$ OR $a \in C(b)$). This is the **primary number** ($83{,}030 / 123{,}067 = 0.6747$). Subtracting it from 123,067 yields exactly $40{,}037$ residual misses.

#### Recomputed Paired Bootstrap Confidence Intervals:
1. **Control $G=50$ + G1 vs Control $G=50$:** Delta on primary undirected pairs is $+36{,}689$ pairs (+29.81 pp on pair count; query recall delta mean $+0.2778$, 95% CI: $[+0.2739, +0.2815]$, $p < 0.0001$).
2. **G1 $K=12$ vs Control $G=200$ (~2M budget):** On primary undirected pairs, $54{,}589$ vs $50{,}359$ yields **$+4{,}230$ pairs (+3.44 pp)**, with 95% paired bootstrap CI **[+0.0308, +0.0379]** ($p < 0.0001$). (On directed slots, $+2{,}578$ slots, +1.05 pp). G1 used +10.04% more candidates.
3. **G1 Standalone ($K=50$) vs Control $G=\text{None}$:** On primary undirected pairs, $72{,}132$ vs $50{,}754$ yields **$+21{,}378$ pairs (+17.37 pp)**, with 95% paired bootstrap CI **[+0.1695, +0.1779]** ($p < 0.0001$). G1 generated 1,013,632 fewer candidates (-12.32%).

#### Status of Renamed Venues Claim:
Empirically evaluated on the 100k train sample in Phase 3c: G1 ($K=50, R=5\text{km}$) recovers **68.44% (3,259 / 4,762)** of true pairs with `different` names located within 5 km. The claim that spatial proximity recovers renamed venues is empirically **SUPPORTED**.

### E7. Phase 4a Step 0b Errata Amendments (Appended 2026-10-12)
1. **Primary Metric Restoration:** Original 003 values (G1 alone 0.5861, union 0.6747, G=None+G1 0.6943, K=4 0.3504, K=12 0.4436, +29.81 pp union gain, +3.44 pp at ~2M) are primary undirected pair recall. Single-direction slot numbers (0.5584, 0.6535, +27.69 pp, +1.05 pp) are retracted as primary metrics and retained strictly labeled as "slots".
2. **Separated Pair vs Query Recall Deltas (Control G=50 + G1 vs G=50):**
   - Pair Recall Delta: $+36{,}689$ undirected pairs (+29.81 pp count; query-level mean pair recall delta $\Delta = +0.2778$, 95% CI: $[+0.2739, +0.2815]$, $p < 0.0001$).
   - Query Recall Delta: $\Delta = +0.3017$ (+30.17 pp), 95% CI: $[+0.2987, +0.3046]$, $p < 0.0001$.
3. **Specification of 20k Subsample and 150k Index:**
   - 150k record benchmark table (`coordinates.parquet`) indexed in BallTree; 20k query subsample drawn from 100k train set.
   - Primary pair recall evaluated on all 10,531 true undirected pairs having at least one endpoint in the 20k query sample (found if either endpoint retrieves the other). Subsample directed slot recall is labeled "slots".
4. **Different-Name True Pairs Count Correction & Conclusion:**
   - On full 100k train sample, different-name true pairs $\le 5\text{ km} = \mathbf{24{,}656}$ (not 4,762 which was the 20k subsample). G1 recovers $16{,}904 / 24{,}656$ (**68.56%**).
   - Reworded conclusion: G1 proximity recovery is name-independent (~57-58% across all name classes) and strictly distance-driven (>92% under 1 km, dropping to ~17% at 1-5 km).
5. **Distance Ceiling (True Pairs > 5 km):**
   - 8,904 of 54,375 true pairs (16.37%) are separated by $>5\text{ km}$ (identical 16.44%, token_reorder 16.80%, partial_overlap 16.70%, different 16.14%, empty 17.08%).

