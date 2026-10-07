# Pre-Registration: Phase 3 Retrieval Channels (Both Datasets)

**Pre-Registration ID:** 005_phase3_channels_both_datasets  
**Date Filed:** 2026-10-07  
**Status:** PLANNED — Not yet implemented

## Channels Planned

(a) **IDF Lexical:** TF-IDF inverted index on canonical name tokens. Applied to both GLEIF×CH (query: GLEIF LegalName vs target: CH CompanyName) and Foursquare (self-join dedup mode).

(b) **Address / Geo:** Hierarchical admin-area key (city/state bucket) plus coordinate-cell key. Applied to both datasets where geographic data is available (Foursquare has lat/lon; GLEIF/CH has postcode).

(c) **Character N-gram:** Character-level $q$-gram ($q \in \{3, 4\}$) blocking index on canonical name. Applied to both datasets.

## Protocol Commitments

1. **Dual-Dataset Rule:** No channel is considered validated unless it shows consistent behavior on both GLEIF×CH and Foursquare. Inconsistencies require explanation.
2. **Paired Bootstrap:** Performance comparisons use a **paired bootstrap over shared entities** (not separate CIs). A difference is only claimed if the 95% paired bootstrap CI excludes zero.
3. **Matched-Budget Comparison:** Primary comparison metric is recall at matched candidate budget against the respective control efficient frontier (see Exp 001b and 002).
4. **Residual Recovery:** Each channel must report what fraction of the B9 train-split residual categories it recovers on each dataset.
5. **No Feature Engineering or Learning:** These channels extend the retrieval stage only. No ranker, no embedding model, no feature engineering in Phase 3.
6. **Feasibility Gate:** If any channel's pair-count estimate exceeds 500 M pairs at the target gate, it is marked infeasible and not run.
7. **No Test Set Touching:** Test splits on both datasets remain sealed until the end of the full experiment sequence.

## Null Hypothesis

The control family efficient frontier (K1–K5 for Foursquare; K1–K4v2 for GLEIF×CH) cannot be significantly improved by any single retrieval channel under a matched candidate budget.

## Acceptance Gate for Phase 3 Completion

- All three channels implemented, tested, and run on both datasets.
- Results logged to `experiments/runs.parquet` with real git commit hashes.
- Experiment reports written with measured recall numbers at each G level.
- No unmeasured claims.
