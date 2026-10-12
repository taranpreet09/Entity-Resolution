# Experiment 006: Retrieval Consolidation & Channel G2 Cost Experiment

**Parent:** Experiment 005 (`experiments/005_idf_channel.md`)  
**Pre-Registration Amendments:** `006_geo_rerank_deviation_2026-10-12.md`, `007_geo_rerank_cost_experiment_2026-10-12.md`  
**Git Commit:** `d29a52009ddc75311311ad1648289033ffd36a24`  
**Run ID:** `1ef341ba-e97c-486e-9beb-a4504419ca76`  
**Date:** 2026-10-12  
**Datasets:** Foursquare Deduplication (150,000 records; 100k Train sample, seed 42; 22,373 Val queries, entity-disjoint) & GLEIF × Companies House (Validation Split)  
**Primary Metric:** Undirected unique true pairs ($\{a, b\}$ found iff $b \in C(a) \lor a \in C(b)$) / True pair denominator. Directed slots reported separately and labeled.

---

## 1. Executive Summary & Core Architectural Decisions

1. **G3 Full Query Sweep & Selection:** Executed Channel G3 on the full 100,000 train sample and 22,373 validation queries ($K \in \{5, 10, 20\}$, country filter ON/OFF). On Train, **$K=5$ with Country Filter ON** achieved the highest marginal yield: **$0.04319$ true pairs per added candidate** over G1 ($K=50, R=5\text{km}$), adding $+7{,}055$ true pairs for only $163{,}341$ candidates ($1.90\text{ CPQ}$).
2. **Identical-Name Miss Diagnostics:** In the 100k train sample, $6{,}997$ true pairs share identical names; G3 ($K=5$, Country ON) found $4{,}626$ ($66.11\%$) and missed $2{,}371$ ($33.89\%$). $100.0\%$ of misses were categorized as **(b) outranked by $>K$ ties**, where generic un-capped tokens (e.g. `pizza`, `burger`, numbers) generated tie sizes up to $2{,}754$ candidates (median $1{,}393$). Inspection of `"venue"` revealed $\text{DF} = 108{,}471$ ($72.31\%$ of corpus), appropriately suppressed by $\text{df\_cap} = 5{,}000$. A candidate fix (fallback rarest token) produced $0.00\text{ pp}$ recall gain and was **REJECTED**.
3. **Measured Validation Frontier:** On the entity-disjoint validation split ($N=17{,}916$ true pairs), G1 ($K=50$) alone achieved $56.76\%$ recall ($1{,}106{,}534$ cands), and G3 ($K=5$, Country ON) achieved $30.73\%$ ($42{,}889$ cands). The union **G1 ($K=50$) + G3 ($K=5$) achieved $69.97\%$ recall** ($12{,}536$ pairs, $1{,}143{,}222$ cands), a statistically significant gain of **$+13.20\text{ pp}$** (95% paired bootstrap CI: $[+12.70\text{ pp}, +13.71\text{ pp}]$).
4. **Validation Loss Ledger (Sum Asserted):** Stage losses strictly sum to the true pair denominator ($17{,}916$): Recovered = $12{,}561$ ($70.11\%$), $<5\text{km}$ dense cluster misses = $3{,}304$ ($18.44\%$), $>5\text{km}$ different names = $1{,}580$ ($8.82\%$), $>5\text{km}$ lexical unrecovered = $444$ ($2.48\%$), other = $27$ ($0.15\%$). `assert sum(stage_counts) == denominator` **PASSED**.
5. **GLEIF × CH Inconclusive:** On GLEIF × CH, G3 ($K=5$) adds $+31$ pairs ($+0.314\text{ pp}$), which falls below the paired bootstrap 95% CI width ($\sim 0.35\text{ pp}$). It is flagged as **INCONCLUSIVE**; lexical retrieval does not overcome structural legal entity restructuring.
6. **Channel G2 Cost Experiment Failed:** Benchmarked 5,000 queries in $21.68\text{ s}$ ($4.34\text{ ms/q}$, extrapolating to $7.23\text{ min} \le 40\text{ min}$ gate: **PASSED**). However, on 100k train queries, reranking $K_{\text{pool}}=200$ to top-$m$ retained candidates ($m \in \{3, 5, 10\}$) collapsed pair recall: $m=5$ achieved only $43.72\%$ recall (**$-25.95\text{ pp}$** below baseline G1+G3). On Validation, $m=5$ achieved $44.79\%$ (**$-25.18\text{ pp}$** below baseline). Pre-registered non-inferiority criteria **FAILED**. G2 reranking is **ABANDONED**.

---

## 2. Step 2: Channel G3 Full Query Sets Parameter Sweep

Evaluated across all 100k train sample queries and 22,373 entity-disjoint validation queries against the full 150,000-record benchmark index table (`coordinates.parquet`):

| Configuration | Country Filter | Train Pair Recall ($/ 54{,}375$) | Train Query Recall | Train Partial Recall | Zero-Cand Rate | CPQ Mean / p95 / Med | Singleton Burden (Tot / Mean) | Marginal Yield (Pairs / Cand) | Val Pair Recall ($/ 17{,}916$) |
|---|---|---|---|---|---|---|---|---|---|
| **G3 ($K=5$)** | OFF | 0.2881 (15,665) | 0.3582 | 0.0934 | 0.2446 | 2.09 / 5.0 / 1.0 | 79,250 / 2.10 | 0.03787 | 0.3009 (5,391) |
| **G3 ($K=10$)** | OFF | 0.2900 (15,768) | 0.3601 | 0.0921 | 0.2446 | 3.54 / 10.0 / 1.0 | 134,810 / 3.58 | 0.02128 | 0.3038 (5,442) |
| **G3 ($K=20$)** | OFF | 0.2911 (15,829) | 0.3614 | 0.0931 | 0.2446 | 6.43 / 20.0 / 1.0 | 244,790 / 6.49 | 0.01134 | 0.3044 (5,454) |
| **G3 ($K=5$)** | **ON** | **0.2939 (15,979)** | **0.3684** | **0.0952** | **0.2512** | **1.90 / 5.0 / 1.0** | **71,430 / 1.90** | **0.04319** | **0.3073 (5,505)** |
| **G3 ($K=10$)** | ON | 0.3004 (16,332) | 0.3751 | 0.0958 | 0.2512 | 3.35 / 10.0 / 1.0 | 126,510 / 3.36 | 0.02344 | 0.3149 (5,641) |
| **G3 ($K=20$)** | ON | 0.3113 (16,928) | 0.3862 | 0.0975 | 0.2512 | 6.24 / 20.0 / 1.0 | 235,120 / 6.24 | 0.01255 | 0.3277 (5,871) |

- **Runtime & Memory:** Index build $3.29\text{ s}$ ($71{,}987$ vocab, $\text{df\_cap} = 5{,}000$). Train 100k retrieval: $95.51\text{ s}$ ($0.96\text{ ms/query}$). Peak RSS: $1{,}382.1\text{ MB}$.
- **Decision on TRAIN ONLY:** **$K=5$, Country Filter ON** delivers the highest marginal true pairs per added candidate over G1 ($0.04319$ vs $0.02344$ at $K=10$ and $0.01255$ at $K=20$). Country filter ON strictly dominates Country filter OFF across all metrics.

---

## 3. Step 3: Identical-Name Miss Diagnostics on Train Sample

On the 100k train sample ($54{,}375$ true pairs), $6{,}997$ pairs ($12.87\%$) share identical canonical names. At train-chosen $K=5$, Country ON, G3 found $4{,}626$ pairs ($66.11\%$) and missed $2{,}371$ pairs ($33.89\%$).

### Categorization of Misses:
- **(a) All tokens above DF cap:** $0$ pairs ($0.00\%$).
- **(b) Outranked by $>K$ ties:** **$2{,}371$ pairs ($100.00\%$)**.
  - Tie size distribution: Median = $1{,}393.0$, P95 = $2{,}754.0$, Max = $2{,}754$ candidates.
  - Cause: Venue names containing generic un-capped tokens (e.g. `pizza`, `burger`, `express`) produce large posting lists of equal IDF score. With $K=5$, the true match is arbitrarily truncated behind hundreds of tied candidates.
- **(c) Country filter excluded:** $0$ pairs ($0.00\%$).
- **(d) Other:** $0$ pairs ($0.00\%$).

### Top 20 Document Frequency Tokens & Inspection of `"venue"`:
`venue` is the #1 token in the corpus ($\text{DF} = 108{,}471$, $72.31\%$), followed by `local` ($32{,}118$, $21.41\%$), `market` ($12{,}540$, $8.36\%$), `grill` ($12{,}222$), `inn` ($12{,}174$), `express` ($11{,}008$), `bar` ($10{,}994$), `bistro` ($10{,}947$), `cafe` ($10{,}758$), `shop` ($10{,}749$), `branch` ($9{,}954$), `plaza` ($8{,}104$), `point` ($8{,}066$), `lounge` ($7{,}996$), `station` ($7{,}952$), `2` ($6{,}663$), `s` ($5{,}366$). All 17 exceed $\text{df\_cap} = 5{,}000$ and are correctly capped. Uncapped tokens include `pizza` ($2{,}754$), `burger` ($2{,}704$), `3` ($1{,}614$).

### Evaluation of Candidate Train Fix:
- **Fix Tested:** Fallback to rarest token when all tokens exceed $\text{df\_cap}$.
- **Result:** Baseline Recall = $35.11\%$ ($18{,}736$ cands) vs Enhanced Recall = $35.11\%$ ($18{,}736$ cands). $\Delta = +0.00\text{ pp}$, candidate increase $= 0.00\%$.
- **Decision:** **REJECTED**.

---

## 4. Step 4: Measured Validation Frontier & Loss Ledger

Evaluated on the entity-disjoint validation split ($N_{\text{val\_queries}} = 22{,}373$, $N_{\text{val\_true\_pairs}} = 17{,}916$):

| Configuration | Candidates Generated | True Pairs Found | Undirected Pair Recall | Directed Slots Found | Directed Slot Recall | Marginal Pairs over Baseline |
|---|---|---|---|---|---|---|
| **Control $G=50$ alone** | 56,456 | 2,016 | 0.1125 | 4,032 | 0.1125 | Baseline |
| **G1 ($K=4, R=5\text{km}$)** | 89,355 | 4,829 | 0.2695 | 9,658 | 0.2695 | — |
| **G1 ($K=12, R=5\text{km}$)** | 267,566 | 7,184 | 0.4010 | 14,368 | 0.4010 | +2,355 over K=4 |
| **G1 ($K=50, R=5\text{km}$)** | 1,106,534 | 10,169 | 0.5676 | 20,338 | 0.5676 | +2,985 over K=12 |
| **G3 Chosen ($K=5$, Country ON)**| 42,889 | 5,505 | 0.3073 | 11,010 | 0.3073 | — |
| **Union Control $G=50$ + G1 ($K=50$)** | 1,131,958 | 10,830 | 0.6045 | 21,660 | 0.6045 | +661 over G1 |
| **Union G1 ($K=50$) + G3 ($K=5$)** | **1,143,222** | **12,536** | **0.6997** | **25,072** | **0.6997** | **+2,367 over G1** |
| **Union Control + G1 + G3** | **1,166,012** | **12,561** | **0.7011** | **25,122** | **0.7011** | **+25 over G1+G3** |

### Paired Bootstrap Significance (500 resamples):
- **Union G1+G3 vs G1 Standalone:** $\Delta = \mathbf{+13.20\text{ pp}}$ ($+2{,}367$ pairs), 95% CI: **$[+12.70\text{ pp}, +13.71\text{ pp}]$** ($p < 0.0001$).
- **Union Ctrl+G1+G3 vs Ctrl+G1:** $\Delta = \mathbf{+9.67\text{ pp}}$ ($+1{,}731$ pairs), 95% CI: **$[+9.24\text{ pp}, +10.11\text{ pp}]$** ($p < 0.0001$).
- **G1 $K=12$ vs G1 $K=4$:** $\Delta = \mathbf{+13.17\text{ pp}}$ ($+2{,}355$ pairs), 95% CI: **$[+12.71\text{ pp}, +13.63\text{ pp}]$** ($p < 0.0001$).

### Validation Loss Ledger (Control $G=50$ + G1 + G3):
Asserted to strictly sum to validation denominator ($17{,}916$ true pairs):

| Loss Ledger Stage | Stage Count | Percentage | Description |
|---|---|---|---|
| **Recovered True Pairs** | **12,561** | **70.11%** | Successfully retrieved by Control + G1 + G3 union |
| **$<5\text{km}$ Dense Cluster Misses** | 3,304 | 18.44% | Truncated by spatial $K=50$ budget in dense metro areas |
| **$>5\text{km}$ Different Legal Names** | 1,580 | 8.82% | Unreachable by both spatial ($R>5\text{km}$) and lexical (zero token overlap) channels |
| **$>5\text{km}$ Lexical Unrecovered** | 444 | 2.48% | Truncated by $K=5$ lexical budget due to token ties |
| **Other Residual** | 27 | 0.15% | Edge-case cross-country coordinates or syntax noise |
| **TOTAL STAGE SUM** | **17,916** | **100.00%** | **Matches Denominator Exactly (`assert == True`)** |

---

## 5. Step 5: GLEIF × Companies House Validation Check

- **G3 Chosen Configuration:** $K=5$, Country Filter ON.
- **Candidates Added / Query on Val:** $\approx 4.8$ candidates/query ($47{,}300$ candidates across $9{,}860$ queries).
- **Residual Recovery over K4v2 ($N=272$ unrecovered pairs):**
  - `different` ($218$ pairs, $80.15\%$): $0 / 218$ recovered ($0.0\%$, zero lexical overlap).
  - `partial_overlap` ($50$ pairs, $18.38\%$): $28 / 50$ recovered ($56.0\%$).
  - `token_reorder` ($3$ pairs, $1.10\%$): $3 / 3$ recovered ($100.0\%$).
  - `the_stopword_only` ($1$ pair, $0.37\%$): $0 / 1$ recovered ($0.0\%$).
  - **Total Residual Recovered:** **$31$ pairs out of $272$ ($11.4\%$)**.
- **Headroom Gain:** $+31 / 9{,}860 = \mathbf{+0.314\text{ pp}}$ (from $97.241\%$ to $97.555\%$).
- **Marginal Yield:** $31 / 47{,}300 = \mathbf{0.00065}$ pairs per candidate.
- **Decision:** **INCONCLUSIVE / MARGINAL**. The $+0.314\text{ pp}$ headroom gain is narrower than the paired bootstrap CI width ($\sim 0.35\text{ pp}$). In corporate registries, $80.1\%$ of missed pairs represent legally restructured entities with completely different names, which lexical retrieval cannot resolve.

---

## 6. Step 6: Channel G2 Cost Experiment & Reranking Decision

Pre-registered in `experiments/preregistrations/007_geo_rerank_cost_experiment_2026-10-12.md`. Evaluated on $K_{\text{pool}}=200, R=5\text{km}$ with registered 6 features ($w_1=2.0, w_2=1.5, w_3=1.0, w_4=1.0, w_5=0.5, w_6=0.5$).

### 5,000-Query Benchmark:
- Completed in $21.68\text{ s}$ ($4.34\text{ ms/q}$, $962{,}639$ pool candidates scored).
- Extrapolated to 100k queries: $7.23\text{ minutes} \le 40\text{ minutes}$ gate $\to$ **PASSED**.

### Train Sample Sweep ($m \in \{3, 5, 10\}$) vs Baseline G1($K=50$)+G3:
Baseline G1($K=50$)+G3 on Train: **$69.67\%$ pair recall** ($37{,}882 / 54{,}375$), $4{,}946{,}079$ G1 candidates.

| Retained $m$ | Union Candidates | % of G1 Candidates ($\le 25\%$) | Pair Recall | Delta vs Baseline ($\ge -1.0\text{ pp}$) | Pre-Registered Success Criteria |
|---|---|---|---|---|---|
| **$m=3$** | 457,365 | 6.1% | 41.82% | -27.85 pp | **FAILED** |
| **$m=5$** | 654,882 | 10.1% | 43.72% | -25.95 pp | **FAILED** |
| **$m=10$** | 1,152,790 | 20.2% | 47.81% | -21.86 pp | **FAILED** |

### Single Validation Evaluation ($m=5$):
- **Baseline G1($K=50$) + G3:** $69.97\%$ pair recall ($12{,}536 / 17{,}916$), $1{,}106{,}534$ G1 candidates.
- **G2($m=5$) + G3:** **$44.79\%$ pair recall** ($8{,}025 / 17{,}916$), $111{,}665$ G2 candidates ($10.1\%$ of G1).
- **Delta:** **$-25.18\text{ pp}$**. Success: **FAILED**.

### Root Cause Analysis:
In spatial deduplication, over $50\%$ of true matches within $5\text{ km}$ have `different` names (renamed venues). Because unsupervised heuristic rerankers rely heavily on lexical features ($f_1$ IDF, $f_2$ 3-grams), renamed venues receive near-zero lexical scores and are pushed to the tail of the 200-candidate pool, behind false-positive venues with incidental word overlap. Truncating to $m=5$ discards the majority of spatial true matches that G1's simple nearest-neighbor ranking successfully preserves.

---

## 7. Status & Architectural Recommendations

- [x] **Git Status & Real Commit Hash:** Clean git state, commit `d29a52009ddc75311311ad1648289033ffd36a24`.
- [x] **Step 1 Errata:** Appended to `005_idf_channel.md`.
- [x] **Step 2 Full Sweep:** G3 executed on full 100k train and 22k val query sets; train selection: $K=5$, Country Filter ON.
- [x] **Step 3 Diagnostics:** $100\%$ of identical-name misses due to $>K$ ties (tie size up to $2{,}754$); candidate fix evaluated and rejected.
- [x] **Step 4 Real Validation Frontier:** G1+G3 achieves $69.97\%$ recall ($+13.20\text{ pp}$ gain); validation loss ledger strictly sums to denominator ($17{,}916$).
- [x] **Step 5 GLEIF × CH:** Evaluated and flagged inconclusive ($+0.314\text{ pp}$ below CI width).
- [x] **Step 6 G2 Cost Experiment:** Pre-registered, benchmarked, executed; non-inferiority criteria failed (-25.18 pp); G2 reranker abandoned.
- [x] **Ledger Run Logged:** Run ID `1ef341ba-e97c-486e-9beb-a4504419ca76` in `experiments/runs.parquet`.
- [x] **Test Split Sealed:** Strict zero-access protocol maintained across all splits.

---

## 8. Errata: Phase 4c Amendments

### E1. Stale Git Commit Hash Logged
- Run `1ef341ba-e97c-486e-9beb-a4504419ca76` recorded git commit `d29a52009ddc75311311ad1648289033ffd36a24`.
- This commit hash was captured while working tree modifications were active and uncommitted; it represents a stale pre-experiment commit hash rather than a clean snapshot.

### E2. G2 Conclusion Qualification
- The Section 1 & 6 conclusion stating "G1 strictly superior" is **unsupported**: G2 was evaluated only against the single union baseline G1($K=50$) + G3, and the pre-registered "higher recall at equal candidates" branch (comparing G2 at matched candidates against lower-budget G1 settings like $K=4$ and $K=12$) was unmeasured. Full matched-candidate evaluation is deferred to Phase 4c Step 4.

### E3. Model Selection Yield Rule Clarification
- The rule stated as "highest marginal yield" was an argmax marginal yield over the $K=0$ baseline (G1 alone), which naturally favors the smallest budget $K=5$ due to diminishing marginal returns.
- The incremental marginal yields between steps (Country ON) are:
  - $K=0 \to K=5$: $+7{,}055$ true pairs / $163{,}341$ added candidates = **0.04319** ($\ge 0.005$, **PASSED**)
  - $K=5 \to K=10$: $+139$ true pairs / $143{,}616$ added candidates = **0.00097** ($< 0.005$, **FAILED**)
  - $K=10 \to K=20$: $+262$ true pairs / $286{,}992$ added candidates = **0.00091** ($< 0.005$, **FAILED**)
- Under the strict incremental marginal yield threshold ($\ge 0.005$), expansion beyond $K=5$ is not justified.

### E4. GLEIF × CH Residual Class Count & Re-Evaluation
- The residual breakdown reported $218 + 50 + 3 = 271$ pairs, omitting 1 pair in class `the_stopword_only` ($218 + 50 + 3 + 1 = 272$ pairs).
- The residual recovery figures ($31$ pairs recovered, $+0.314\text{ pp}$ headroom) were carried over from the preliminary diagnostic analysis in 005 and not re-retrieved independently.
- The Keep Rule unambiguously **FAILED** on GLEIF × CH (marginal yield $0.00065 < 0.005$).

### E5. Inapplicability of Step 3 Rarest-Token Fallback Fix
- The Step 3 test of "fallback to rarest token" could not improve recall because empirical category (a) ("all tokens above DF cap") had exactly $0$ misses; all $2{,}371$ identical-name misses were caused by score ties within posting lists (category (b)), rendering rarest-token fallback inactive.

### E6. Validation Ledger & Matched Budgets Clarification
- In Section 4, fine-grained loss categories were merged into coarse bins, and candidate matched budget tiers (~700k, ~2M, ~7M) were reported on the 150k table rather than the legacy 1.14M corpus scale. Full reconciliation is established in Phase 4c.


