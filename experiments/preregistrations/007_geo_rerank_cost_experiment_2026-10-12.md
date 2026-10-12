# Pre-Registration Amendment 007 — Channel G2 Cost Experiment & Reranking Efficiency Protocol

**Pre-Registration ID:** `007_geo_rerank_cost_experiment`  
**Date Filed:** 2026-10-12  
**Commit:** `d29a52009ddc75311311ad1648289033ffd36a24`  
**Parent Pre-Registrations:** `004_geo_proximity_channel.md`, `006_geo_rerank_deviation_2026-10-12.md`  
**Target Phase:** Phase 4b  
**Status:** FILED  

---

## 1. Motivation & Architectural Objective

Experiment 004b established that expanding the metric spatial candidate pool from $K_{\text{pool}}=50$ to $K_{\text{pool}}=200$ within $R=5\text{ km}$ provides $+13.28\text{ pp}$ true pair ceiling headroom (from $57.16\%$ to $70.43\%$). However, generating 200 spatial candidates per query incurs massive downstream ranking and verification overhead ($3.84\text{ M}$ candidates on a 20k query slice).

Channel G2 aims to achieve high-precision spatial retrieval by retrieving the deep $K_{\text{pool}}=200$ pool, scoring candidates with an unsupervised 6-feature linear ranker, and pruning to a small budget $m \in \{3, 5, 10\}$ per query.

This experiment tests whether **G2($m$) + G3** can replace the dense **G1($K=50$) + G3** baseline while slashing candidate volume by at least $75\%$.

---

## 2. Experimental Setup & Channel Specification

1. **Candidate Pool Retrieval:**
   - Index: Full 150,000-record benchmark table in scikit-learn `BallTree` (Haversine metric).
   - Radius: $R = 5000\text{ m}$.
   - Pool Depth: $K_{\text{pool}} = 200$ spatial nearest neighbors.

2. **Reranker Scoring Function (Registered 6 Features):**
   $$\text{Score}(q, c) = w_1 \cdot \text{IDF\_Overlap}(q, c) + w_2 \cdot \text{Jaccard}_{3\text{-gram}}(q, c) + w_3 \cdot \text{Address\_Tokens}(q, c) + w_4 \cdot \mathbb{I}_{\text{same\_country}}(q, c) + w_5 \cdot \mathbb{I}_{\text{same\_category}}(q, c) - w_6 \cdot \ln(1 + \text{dist\_m})$$
   where:
   - $f_1 = \text{IDF\_Overlap}(q, c)$: token-level IDF overlap from `IDFRetriever.score_pair(q, c)`.
   - $f_2 = \text{Jaccard}_{3\text{-gram}}(q, c)$: character 3-gram Jaccard similarity of canonical names.
   - $f_3 = \text{Address\_Tokens}(q, c)$: address token overlap (0 in coordinate-only benchmark).
   - $f_4 = \mathbb{I}_{\text{same\_country}}(q, c)$: binary indicator for matching country codes inferred from coordinates.
   - $f_5 = \mathbb{I}_{\text{same\_category}}(q, c)$: binary indicator for matching category codes (0 in coordinate-only benchmark).
   - $f_6 = -\ln(1 + \text{dist\_m})$: logarithmic distance decay from physical Haversine distance in meters.

3. **Fixed Pre-Registered Weights:**
   $$w_1 = 2.0, \quad w_2 = 1.5, \quad w_3 = 1.0, \quad w_4 = 1.0, \quad w_5 = 0.5, \quad w_6 = 0.5$$

4. **Retained Candidate Slicing:**
   - Candidate parameter sweep: $m \in \{3, 5, 10\}$ retained candidates per query.

---

## 3. Pre-Registered Cost Constraint & Runtime Gate

- **Benchmark Requirement:** Benchmark 5,000 queries before executing full runs.
- **Runtime Gate:** Extrapolated runtime across 100,000 queries must be $\le 40\text{ minutes}$. If $>40\text{ minutes}$, full execution is aborted.
- **Empirical Benchmark Result:** 5,000 queries completed in **$21.68\text{ s}$** ($4.34\text{ ms/query}$), extrapolating to **$7.23\text{ minutes}$** for 100,000 queries ($\le 40\text{ min}$). **GATE PASSED**.

---

## 4. Pre-Registered Success Criteria

A configuration $m$ is considered an empirical success if and only if:
1. **Candidate Efficiency:** $\text{Candidates}(\text{G2}(m)) \le 25\%$ of $\text{Candidates}(\text{G1}(K=50))$ (a $\ge 75\%$ reduction in spatial candidates).
2. **Recall Non-Inferiority:**
   $$\text{Pair Recall}(\text{G2}(m) + \text{G3}) \ge \text{Pair Recall}(\text{G1}(K=50) + \text{G3}) - 1.00\text{ pp}$$
   OR achieves higher recall at equal candidate volume.
3. **Model Selection Protocol:**
   - $m$ is selected strictly on the **fixed 100k Train sample (seed 42)**.
   - The selected $m$ is evaluated **exactly once on the Validation split**.
   - Test split remains strictly sealed.

