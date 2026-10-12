# Experiment 007 — Retrieval Consolidation & Retrieval v1 Freeze
## Foursquare Deduplication Benchmark

**Parent:** Experiment 006 (`experiments/006_retrieval_consolidation.md`)  
**Date:** 2026-10-12  
**Active Branch:** `phase-4c`  
**Configuration File:** `configs/retrieval_frozen.yaml`  
**Configuration SHA-256:** `17971dce7cad701786ce2317fc9a1c201880d98975bca78164e309ea25351601`  
**Evaluated Splits:** Canonical 150k table (`coordinates.parquet`). Design & selection strictly on fixed 100k Train sample (seed 42, $N=54{,}375$ pairs) and 20k train subsample ($N=10{,}531$ pairs); final verification once on Validation split ($N=17{,}916$ pairs / $35{,}832$ slots). **Test split strictly sealed.**

---

## 1. Executive Summary & Freeze Declaration

Retrieval v1 is officially **frozen** as a multi-channel candidate generation pool combining three complementary channels with deterministic tie-breaking and per-query budget capping:

$$\text{Bitmask } \mathbf{M} = (\text{Control } G=50) \;\lor\; (\text{G1 Spatial } K_{\text{pool}}=100, R=5\text{km}) \;\lor\; (\text{G3 IDF } K=20, \text{Country ON, Det Tie-Break})$$

- **Frozen Candidate Stores:**
  - `experiments/candidates_frozen_train.parquet` (9,817,035 rows, 57.95 MB, 100k queries, $C=100$).
  - `experiments/candidates_frozen_val.parquet` (2,195,396 rows, 11.89 MB, 22,373 queries, $C=100$).
- **Validation Pool Ceiling:** **$77.13\%$ pair recall** ($13{,}818 / 17{,}916$), generating $98.1$ CPQ ($2{,}195{,}396$ total candidate slots).
- **Budget Prune Impact:** Per-query cap $C=100$ prunes only $34$ validation true pairs (**$0.19\text{ pp}$ loss**) while bounding candidate generation to $\le 100$ items per query.
- **Architectural Verdict on G2 Reranker:** G2 is **permanently dropped as a retrieval component**; its 6 scoring features are preserved as candidate-level features for the learned ranker phase.

---

## 2. Step 1: Canonical Baseline Reconciliation & Asymmetry Analysis

Per ADR-007, `coordinates.parquet` (150,000 records) is the canonical dev corpus. Baseline retrieval was recomputed across all channels with identical denominators ($N=17{,}916$ pairs, $N=35{,}832$ directed slots):

| Retrieval Channel | Candidates | Undir Pairs | Pair Recall | Dir Slots | Slot Recall | Asymmetry Diff ($2P - S$) | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|
| **Control $G=50$** | 56,456 | 2,016 | 11.25 % | 4,032 | 11.25 % | **0** | 0.65 s | 445 MB |
| **G1 ($K=4, R=5\text{km}$)** | 89,355 | 4,829 | 26.95 % | 8,770 | 24.48 % | **888** | 0.28 s | 445 MB |
| **G1 ($K=12, R=5\text{km}$)** | 267,566 | 7,184 | 40.10 % | 13,638 | 38.06 % | **730** | 0.85 s | 456 MB |
| **G1 ($K=50, R=5\text{km}$)** | 1,106,534 | 10,169 | 56.76 % | 19,956 | 55.69 % | **382** | 3.53 s | 486 MB |
| **G3 ($K=5$, Country ON)** | 42,889 | 5,505 | 30.73 % | 10,912 | 30.45 % | **98** | 1.29 s | 486 MB |
| **G3 ($K=20$, Country ON)** | 140,967 | 5,871 | 32.77 % | 11,697 | 32.64 % | **45** | 5.15 s | 486 MB |
| **Control + G1 ($K=50$)** | 1,131,958 | 10,830 | 60.45 % | 21,330 | 59.53 % | **330** | 4.18 s | 486 MB |
| **G1 ($K=50$) + G3 ($K=5$)** | 1,143,222 | 12,536 | 69.97 % | 24,771 | 69.13 % | **301** | 4.82 s | 486 MB |
| **Control + G1 + G3 ($K=5$)** | 1,166,012 | 12,561 | 70.11 % | 24,835 | 69.31 % | **287** | 5.47 s | 487 MB |

### Mathematical Explanation of Asymmetry:
- **Control $G=50$:** Uses symmetric blocking keys ($a \in \text{bucket} \iff b \in \text{bucket}$), guaranteeing $a \in C(b) \iff b \in C(a)$. Every retrieved pair is bidirectional, yielding exactly $\text{Slots} = 2 \times \text{Pairs}$ (diff $= 0$).
- **G1 & G3 (Directed kNN):** Truncation capacity is fixed per query endpoint ($K$). Under non-uniform entity cluster density, query $a$ in a sparse peripheral area has few local neighbors and readily includes core venue $b$ within its top-$K$. Conversely, core query $b$ exhausts its $K$ slots on hyper-local immediate neighbors before reaching peripheral venue $a$. Consequently, $\{a, b\}$ is retrieved unidirectionally ($a \to b$ only), causing $\text{Slots} < 2 \times \text{Pairs}$. As $K$ expands ($4 \to 12 \to 50$), unidirectional pairs decrease ($888 \to 730 \to 382$), converging toward symmetric recovery.

---

## 3. Step 3: G3 Deterministic Tie-Break Evaluation

Diagnosed in 006, 100% of G3 identical-name misses were caused by $>K$ score ties within common token posting lists (tie sizes up to 2,754). We tested two ranking interventions on the 100k Train sample:
1. **Deterministic Tie-Break (i):** `IDF score desc -> exact canonical name desc -> same country desc -> nearer Haversine distance asc -> candidate ID asc`.
2. **BM25 Length Normalization (ii):** Document length normalization ($k_1=1.2, b=0.75, \text{avgdl}=2.51$).

| Evaluation Split / Budget | Channel / Variant | Undir Pairs | Recall | Delta vs Base | Candidates | Decision Rule |
|---|---|---|---|---|---|---|
| **Train 100k ($K=5$)** | Baseline G3 | 15,979 | 29.39 % | — | 190,160 | — |
| | (i) Deterministic Tie-Break | 18,583 | **34.18 %** | **+4.79 pp** | 190,160 | **ACCEPTED ($\ge 0.5\text{ pp}$)** |
| | (ii) BM25 Length Norm | 15,973 | 29.38 % | -0.01 pp | 190,160 | REJECTED ($< 0.5\text{ pp}$) |
| | BM25 + Deterministic | 20,831 | 38.31 % | +8.92 pp | 190,160 | Explored |
| **Train 100k ($K=20$)** | Baseline G3 | 16,928 | 31.13 % | — | 623,868 | — |
| | (i) Deterministic Tie-Break | 22,360 | **41.12 %** | **+9.99 pp** | 623,868 | **ACCEPTED ($\ge 0.5\text{ pp}$)** |
| | (ii) BM25 Length Norm | 17,045 | 31.35 % | +0.22 pp | 623,868 | REJECTED ($< 0.5\text{ pp}$) |
| **Validation ($K=5$)** | Baseline G3 | 5,505 | 30.73 % | — | 42,889 | — |
| | (i) Deterministic Tie-Break | 6,346 | **35.42 %** | **+4.69 pp** | 42,889 | Confirmed on Val |
| **Validation ($K=20$)** | Baseline G3 | 5,871 | 32.77 % | — | 140,967 | — |
| | (i) Deterministic Tie-Break | 7,562 | **42.21 %** | **+9.44 pp** | 140,967 | Confirmed on Val |

**Outcome:** Deterministic tie-breaking delivers a massive $+4.79\text{ pp}$ gain at $K=5$ and $+9.99\text{ pp}$ at $K=20$ on Train (confirmed on Val with $+4.69\text{ pp}$ and $+9.44\text{ pp}$) at **zero added candidate cost**. It is permanently integrated into Channel G3.

---

## 4. Step 4: G2 Fair Check at Matched Candidates

On the 20k Train subsample ($N=10{,}531$ true pairs), G2 reranking ($K_{\text{pool}}=200, R=5\text{km}$, 6 features) was benchmarked against G1 at matched candidate budgets, each unioned with Channel G3:

| Tier | Configuration | Total Cands | CPQ | Hits | Recall | Renamed $\le 5\text{km}$ | Paired Bootstrap 95% CI vs G2 | $p$-value |
|---|---|---|---|---|---|---|---|---|
| **Tier 1 (~100k)** | **G1 ($K=4$) + G3** | 115,217 | 5.8 | 5,358 | **50.88 %** | **33.20 %** | vs G2 ($m=3$): **[+7.31 pp, +8.93 pp]** | $< 0.0001$ |
| | G2 ($m=3$) + G3 | 90,909 | 4.5 | 4,503 | 42.76 % | 3.86 % | — | — |
| | G2 ($m=5$) + G3 | 130,402 | 6.5 | 4,698 | 44.61 % | 6.59 % | vs G1 ($K=4$): **[-7.12 pp, -5.46 pp]** | $< 0.0001$ |
| **Tier 2 (~250k)** | **G1 ($K=12$) + G3** | 273,313 | 13.7 | 6,268 | **59.52 %** | **48.40 %** | vs G2 ($m=10$): **[+9.72 pp, +11.43 pp]** | $< 0.0001$ |
| | G2 ($m=10$) + G3 | 230,027 | 11.5 | 5,156 | 48.96 % | 15.88 % | — | — |

### Breakdown by Name Class & Distance:
- **Renamed Venues ($\le 5\text{km}$, `different` name):** G1 recovers **33.20%** at $K=4$ and **48.40%** at $K=12$, whereas G2 recovers only **3.86%** at $m=3$, **6.59%** at $m=5$, and **15.88%** at $m=10$.
- **Failure Mode of G2:** Unsupervised lexical weighting severely penalizes renamed venues (which lack lexical overlap), pushing them past rank $m$ behind spurious lexical collisions. Simple spatial nearest-neighbor retrieval (G1) is name-agnostic and strictly preserves these true matches.
- **Decision Rule Application:** G1 strictly dominates G2 across all matched candidate budgets ($\Delta = +6.27\text{ pp}$ to $+10.56\text{ pp}$, $p < 0.0001$). Under the pre-registered decision rule, **G2 is dropped as a retrieval component**.

---

## 5. Step 5: Freeze Retrieval v1 Pool, Cap Curve, & Loss Ledger

### Train Budget-Prune Curve (100k Train Sample):
Evaluated across per-query candidate caps $C \in [10, 150]$:
- Uncapped Pool ($C=\infty$): $77.63\%$ recall ($42{,}211 / 54{,}375$), $10{,}379{,}437$ candidates ($103.8$ CPQ).
- Cap $C=50$: $73.37\%$ recall ($39{,}897$ hits, $94.52\%$ preserved, $2{,}314$ pairs pruned, $49.6$ CPQ).
- Cap $C=75$: $75.76\%$ recall ($41{,}193$ hits, $97.59\%$ preserved, $1{,}018$ pairs pruned, $74.0$ CPQ).
- **Cap $C=100$ (FROZEN):** **$77.43\%$ recall** ($42{,}103$ hits, **$99.74\%$ preserved**, $108$ pairs pruned, $98.2$ CPQ).

### Validation Split Accounting ($N=17{,}916$ true pairs):
- Uncapped Multi-Channel Union: $77.32\%$ pair recall ($13{,}852 / 17{,}916$).
- Capped Candidate Store ($C=100$): **$77.13\%$ pair recall** ($13{,}818 / 17{,}916$), $2{,}195{,}396$ candidates ($98.1$ CPQ).
- `BUDGET_PRUNE` Loss: **34 pairs (0.19 pp)**.

### Strict Validation Loss Ledger:
Every true pair in the validation set is uniquely categorized, strictly summing to $17{,}916$:

$$\begin{aligned}
\text{Total Pairs} &= \mathbf{17{,}916} \\
\text{Recovered in Candidate Store} &= 13{,}818 \quad (77.13\%) \\
\text{Local Density Exceeded } K_{\text{pool}}=100 \; (\le 5\text{km}) &= 2{,}109 \quad (11.77\%) \\
\text{Lexical Present but } >5\text{km Dropped by G3 } K=20 &= 1{,}469 \quad (8.20\%) \\
\text{No Geo and No Lexical Overlap } (>5\text{km}) &= 486 \quad (2.71\%) \\
\text{Budget Prune Loss } (C=100) &= 34 \quad (0.19\%) \\
\hline
\mathbf{\sum \text{Ledger Categories}} &= \mathbf{17{,}916} \quad (\textbf{100.00\% accountancy})
\end{aligned}$$

---

## 6. Pre-Registration for Ranker Phase (Phase 5)

**Pre-Registration Date:** 2026-10-12  
**Target Goal:** Train pairwise/listwise ranking models to rerank candidate pairs from the frozen candidate store into high-precision deduplication decisions.

1. **Fixed Input Store:**
   - Training candidate pool: `experiments/candidates_frozen_train.parquet` (9,817,035 pairs).
   - Validation candidate pool: `experiments/candidates_frozen_val.parquet` (2,195,396 pairs).
   - Test split: **Strictly sealed.**
2. **Model Architectures:**
   - **Baseline Floor:** `LogisticRegression` on normalized features with class weighting.
   - **Primary Ranker:** `LightGBM` / `XGBoost` (`objective: lambdarank` or binary logloss with group queries).
3. **Core Feature Space (14 Features):**
   - *Retrieval Provenance:* `channel_mask` bits (Control, G1, G3), multi-channel consensus count (`popcount`), `score`, `rank`, `reverse_rank`.
   - *String / Lexical Sim:* Token Jaccard, character 3-gram Jaccard, normalized Levenshtein ratio, token containment ratio, length difference.
   - *Spatial & Geographic:* Haversine distance in meters, $\ln(1 + \text{dist}_m)$, country exact match, latitude/longitude deltas.
4. **Primary Evaluation Metrics:**
   - Mean Average Precision (MAP), Precision@1, Precision@3.
   - Global pairwise F1 @ optimal threshold calibrated on Train sample.
   - Non-inferiority gate: Primary Ranker must exceed the Logistic floor by $\ge +3.00\text{ pp}$ F1 on Validation before test evaluation.
