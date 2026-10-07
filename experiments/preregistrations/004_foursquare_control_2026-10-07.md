# Pre-Registration: Foursquare Control Baseline Experiment

**Pre-Registration ID:** 004_foursquare_control  
**Date Filed:** 2026-10-07  
**Experiment:** Foursquare Control Baseline (Exp 002)  
**Status:** Completed — see `experiments/002_foursquare_control.md`

## Hypotheses (Filed Before Running)

1. The 5-key control blocker (K1–K5) will achieve meaningful pair recall at budget G=50.
2. Geo-cell key K2 will be the most candidate-efficient key (highest marginal true pairs per candidate generated).
3. At G=None, K1 (name+country) will generate substantially more candidates than K2 (geo-cell) due to global chain name collisions.
4. The majority of true pairs unrecovered at G=50 will be in the RETRIEVAL loss stage.
5. At least 50% of unrecovered train pairs will have `different` canonical name relation.

## Design Commitments

- Control canonicalization only (NFKD → ASCII-ignore), no learned transformations.
- Index built over full corpus; bucket density reflects full data distribution.
- Val split is evaluation-only; train is diagnostic-only for B9.
- Test split sealed throughout.

## Acceptance Criteria

All measured — results documented in `experiments/002_foursquare_control.md`.
