# Pre-Registration 006 — Geo Reranking Channel (G2) & Dual-Dataset Deviation

**Pre-Registration ID:** `006_geo_rerank_deviation`  
**Date Filed:** 2026-10-12  
**Commit:** `d29a52009ddc75311311ad1648289033ffd36a24`  
**Parent Pre-Registrations:** `004_geo_proximity_channel.md`, `005_phase3_channels_both_datasets_2026-10-07.md`  
**Status:** FILED (Phase 3b)  

---

## 1. Context & Channel G2 Specification

Channel G2 was designed as a metric geo-proximity candidate pool retriever followed by a multi-component heuristic reranker on the Foursquare deduplication dataset:
1. **Candidate Pool:** $K_{\text{pool}}$ nearest spatial neighbors within radius $R = 5000\text{ m}$ (haversine metric).
2. **Scoring Components (V1 name-only; V2 name+context):**
   - IDF-weighted token overlap of canonical names (full corpus token statistics)
   - Character 3-gram Jaccard similarity of canonical names
   - Address token overlap
   - Exact postal code / city match flags
   - Category equivalence flag
   - Log-distance spatial decay term $\ln(1 + \text{dist\_meters})$
3. **Budget Control:** Retain top-$m$ candidates per query record ($m \in \{2, 3, 5, 10\}$) subject to an optional score floor.

---

## 2. Gate Condition (Phase 3b Step 2)

Per project protocol, Channel G2 development is gated by spatial pool headroom:
$$\text{GATE: If Pair Recall}(K_{\text{pool}}=200, R=5\text{km}) - \text{Pair Recall}(K_{\text{pool}}=50, R=5\text{km}) < +3.00\text{ pp}, \text{ STOP.}$$

If the pool ceiling does not provide at least $+3.00\text{ pp}$ additional true pair headroom over the baseline $K=50$ pool (0.4916 on the 100k train sample), downstream reranking cannot recover enough unretrieved ground truth to justify the candidate explosion and feature computation overhead.

---

## 3. Dual-Dataset Requirement & Documented Deviation

> [!WARNING] Pre-Registered Deviation: Spatial Rerank Deferred on GLEIF × Companies House
> Under general BER project rules, all retrieval channels must be evaluated on both GLEIF × Companies House and Foursquare before architectural adoption.
>
> **Deviation Documented:**
> - The GLEIF × Companies House linkage dataset lacks coordinate fields (latitude and longitude).
> - Spatial proximity on GLEIF × Companies House relies solely on postal code hierarchy (full postcode, outward code, and postcode district).
> - Channel G2 (coordinate pool retrieval + multi-feature rerank) has **NOT been run on GLEIF × Companies House**.
> - Spatial reranking for GLEIF × Companies House is explicitly **deferred** to a future phase evaluating postal-graph distance or external geocoded centroids.

