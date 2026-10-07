# Metric Definitions — BER Retrieval Evaluation

All retrieval metrics in `ber.eval.metrics` are computed over **entities**
(unique query IDs), not over individual candidate pairs. This avoids
double-counting when a single entity has multiple true targets.

Notation used below:

- **Q** = set of all unique query IDs that have at least one true target
  (i.e. queries with a non-empty true-target set). Queries with no true
  target are **excluded** from all denominators.
- **T(q)** = set of true target IDs for query q
- **C(q, K)** = set of candidate IDs retrieved for query q at rank ≤ K
- **N = |Q|** = number of queries with at least one true target

---

## recall@K

**Definition:**

    recall@K = |{q ∈ Q : C(q,K) ∩ T(q) ≠ ∅}| / N

The fraction of queries (with true targets) for which at least one true
target appears in the top-K retrieved candidates.

**Denominator:** N = number of queries with at least one true target. Queries
that have zero true targets (e.g. no-match queries in the GLEIF-CH dataset)
are **not** counted in the denominator and are **not** counted as misses.

**Returns NaN** if N = 0 (no queries have true targets).

**Implementation:** `ber.eval.metrics.recall_at_k(candidate_map, true_targets, k)`

**Note on "recall@budget":** In this codebase, `recall@budget` is recall at
K = the maximum rank observed in the candidate set for that query. For the
exact-key blocker, where there is no score-based cutoff, this equals
"did the true target appear anywhere in the candidate set?" — i.e. recall
at the system's own candidate budget.

---

## zero-candidate rate

**Definition:**

    zero_cand_rate = |{q ∈ Q : C(q) = ∅}| / N

The fraction of queries (with true targets) for which the retrieval system
returned **zero candidates at all**. A query is zero-candidate if its
query_id does not appear in the candidates table.

**Denominator:** N = number of queries with at least one true target. Same
as recall@K. Queries with no true target are excluded.

**Interpretation:** A high zero-candidate rate means the retrieval system
fails to produce any candidates for a large fraction of answerable queries.
This is the most severe failure mode — the true target cannot be recovered
regardless of ranking.

**Implementation:** `ber.eval.metrics.zero_candidate_rate(candidate_map, true_targets)`

---

## partial-recall rate

**Definition:**

    partial_recall@K = |{q ∈ Q : 0 < |C(q,K) ∩ T(q)| < |T(q)|}| / N

The fraction of queries for which the system retrieves **some but not all**
true targets within the top-K candidates.

**Denominator:** N = number of queries with at least one true target.

**Applicability:** This metric is only meaningful for datasets where a query
can have multiple true targets (|T(q)| ≥ 2). When every query has exactly
one true target (as in the GLEIF × Companies House dataset, where labels are
derived from a 1:1 identifier match), a query is either a full hit or a full
miss — partial recall is **not applicable (N/A)** and will always be 0.0.
Report it as "N/A (all queries have exactly one true target)" in that case,
not as a substantive result.

**Implementation:** `ber.eval.metrics.partial_recall_rate(candidate_map, true_targets, k)`

---

## candidates-per-query distribution

Summary statistics over the number of candidates retrieved per query,
computed only over queries that **received at least one candidate**.

Reported statistics: min, p25, median (p50), p75, p95, max, mean.

**Important:** the denominator for the mean and percentiles is the number
of queries with at least one candidate — **not** N (all queries with true
targets). If zero-candidate queries are common, the mean cpq will be higher
than the true average over all answerable queries. Both denominators should
be reported when they differ significantly.

**Implementation:** `ber.eval.metrics.candidates_per_query_distribution(candidate_map)`

---

## bootstrap confidence interval

Paired bootstrap over entities (query IDs), not pairs. Resamples the query
population (with replacement) and recomputes the point metric on each
resample. Uses the 2.5th and 97.5th percentile of the bootstrap distribution
as the 95 % CI bounds.

**Number of resamples:** 500 (experiments 001, 001b) or 1000 (pre-registered
comparisons). Results are reproducible given the same seed.

**Minimum meaningful difference:** Given the val set of ~9,860 pairs and
~9,800 queries with true targets, the bootstrap CI half-width is typically
±0.003–0.004. Differences below ~0.003 absolute (0.3 percentage points)
cannot be distinguished from noise at 95 % confidence and should not be
reported as improvements.

**Implementation:** `ber.eval.metrics.bootstrap_ci(candidate_map, true_targets, metric_fn, n_boot, seed)`

---

## standalone recall (per-key diagnostic)

**Definition:** recall@budget when **only one blocking key** is used, with
all other keys disabled. This measures the raw recall contribution of a
single key in isolation.

This is NOT the same as "unique contribution" — a key that duplicates all
pairs from another key will have the same standalone recall as a high-recall
key, even though its marginal value is zero.

---

## leave-one-out recall (per-key diagnostic)

**Definition:** recall@budget when **one blocking key is removed** and all
others remain. The drop from full-system recall to leave-one-out recall
measures how much that key is needed.

    LOO_delta(k) = recall_full - recall_LOO_without_k

A key with LOO_delta ≈ 0 is fully redundant given the other keys.

---

## unique true pairs per key (per-key diagnostic)

**Definition:** the number of val true pairs recovered by key k **and not
recovered by any other key alone**.

This differs from "unique candidate pairs" in Experiment 001, which counted
candidate pairs (including false positives) exclusive to one key. True-pair
uniqueness is the correct quantity for measuring irreplaceable recall
contribution.
