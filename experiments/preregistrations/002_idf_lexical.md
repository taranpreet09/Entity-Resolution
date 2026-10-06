# Pre-registration 002 — IDF-Weighted Lexical Retrieval vs Control Baseline

**Status:** Pre-registered — NOT run  
**Pre-registered:** 2026-10-07  
**Depends on:** Experiment 001 (control baseline at G=50)

---

## Hypothesis

An IDF-weighted lexical retrieval channel (e.g. BM25 or TF-IDF cosine over
a sparse inverted index of canonical name tokens) will achieve higher
recall@budget than the control baseline at a matched total candidate count.

Specifically: at a candidate budget equal to the control's G=50 output
(130,629 total val candidates, median 6 per query), the IDF channel will
recover more of the 311 no-shared-key losses by tolerating partial name
overlap.

---

## Method

1. Build a sparse inverted index over canonical `LegalName` tokens for all
   CH targets.
2. For each GLEIF query, retrieve the top-K CH targets by BM25 score
   (Okapi BM25, k1=1.5, b=0.75 — standard parameters). K is set so that
   total val candidates ≈ 130,629 (the G=50 control budget).
3. Evaluate on the **val split only** (same 11,682 queries, 9,860 true pairs
   as experiment 001). Test split remains sealed.
4. Report recall@budget, zero-candidate rate, cpq distribution, and
   per-pair loss ledger against the control.
5. Compare systems using `ber.eval.metrics.compare_systems()` (paired
   bootstrap, 1,000 resamples, seed=42) to compute the delta and p-value.

---

## Success criterion

**Success:** recall@budget (IDF) > recall@budget (control G=50), with
95 % bootstrap CI for the delta strictly above zero.

**Failure:** recall@budget (IDF) ≤ recall@budget (control G=50), or the
CI for the delta straddles zero.

In either outcome, the loss ledger must show which of the 311 no-shared-key
pairs (if any) the IDF channel recovers and which it still misses.

---

## Failure analysis required if criterion not met

If IDF does not beat the control at matched budget, investigate whether:
- The IDF gain on partial-name pairs is offset by false positives inflating
  the per-query candidate count and pushing true targets beyond the budget.
- The 311 no-key losses are structurally absent from the BM25 ranked list
  (e.g. because the GLEIF name and CH name share zero tokens after
  canonicalisation — in which case experiment 003 is the correct next step).

---

## Metrics to report

- recall@budget (IDF vs control, with bootstrap CI and p-value)
- zero-candidate rate
- cpq distribution (median, p95)
- loss ledger stage counts
- count of no-key pairs recovered vs still missed
- count of control-recovered pairs now missed (regression check)

---

## What this experiment does NOT measure

- Retrieval quality on the test split (sealed)
- Semantic similarity beyond lexical token overlap
- Performance on no-match queries (excluded from denominator)
