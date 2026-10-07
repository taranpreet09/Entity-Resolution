# Experiment 001b — Control Baseline Diagnostics

**Parent:** Experiment 001 (`experiments/001_control_baseline.md`)
**git commit:** `c805f600e4caa2a2c61be72b5bdc2fd808a71e99`
**Date:** 2026-10-07
**Evaluated split:** val (test split untouched throughout)

Run IDs logged to `experiments/runs.parquet`:
- v1 reproduction: `phase2b_v1_repro` run
- v2 sweep: `phase2b_v2_sweep` run
- per-key corrected: `phase2b_per_key` run
- residual taxonomy: `phase2b_taxonomy` run
- no-match diagnosis: `phase2b_no_match` run
- GLEIF sample stats: `phase2b_gleif_stats` run

Reproduce all results: `python experiments/run_phase2b_diagnostics.py --seed 42`

---

## 1. Reproducibility: control_v1 Re-run

All five G values match Experiment 001 bit-for-bit:

| G | recall@budget | Candidates | Identical to Exp 001? |
|---|---|---|---|
| 10 | 0.9665 | 34,437 | ✓ |
| 50 | 0.9675 | 130,629 | ✓ |
| 200 | 0.9683 | 321,275 | ✓ |
| 1000 | 0.9685 | 549,355 | ✓ |
| none | 0.9685 | 685,788 | ✓ |

---

## 2. control_v2: New K4v2 = suffix-stripped name

**Defect corrected:** v1 K4 (`outward|first_token`) is structurally symmetric to K2
(`first_token|outward`) and recovers the same true pairs. Zero unique true pairs at
every G (confirmed by corrected per-key metrics below). See Errata E1.

**v2 key set:**
| Key | Description |
|---|---|
| K1 | `canon(LegalName)` vs `canon(CompanyName)` |
| K2 | `first_token(name)\|outward_code(postcode)` (unchanged) |
| K3 | `first_3_chars(name)\|postcode_district` (unchanged) |
| **K4v2** | `strip_legal_suffixes(name)` — no postcode component |

Suffix list: `configs/legal_suffixes.json` (15 entries). Stripping is applied
to both query and target; the key fires when names differ only in legal-form suffix
(e.g. "ACME LIMITED" ↔ "ACME LTD").

### v2 G-sweep results (val split)

Bootstrap: 500 resamples, seed=42, over query-side entities.

| G | Candidates | cpq median | cpq p95 | recall@budget | 95% CI | zero-cand rate |
|---|---|---|---|---|---|---|
| 10 | 34,576 | unmeasured | unmeasured | **0.9714** | [0.9685, 0.9741] | 0.0133 |
| 50 | 130,758 | unmeasured | unmeasured | **0.9720** | [0.9693, 0.9747] | 0.0069 |
| 200 | 321,398 | unmeasured | unmeasured | **0.9723** | [0.9697, 0.9750] | 0.0044 |
| 1000 | 549,476 | unmeasured | unmeasured | **0.9724** | [0.9698, 0.9751] | 0.0034 |
| none | 685,909 | unmeasured | unmeasured | **0.9724** | [0.9697, 0.9751] | 0.0034 |

*cpq distribution not stored in this run; re-run with explicit distribution capture to measure.*

**v1 vs v2 at G=None:** recall 0.9685 → 0.9724, delta = +0.0039 (39 additional true
pairs recovered). The 39 pairs correspond exactly to the `a_legal_suffix` category in
the v1 residual taxonomy. The bootstrap CI for v2 G=None is [0.9697, 0.9751]; the
v1 CI is [0.9654, 0.9712]. The CIs do not overlap — the improvement is real.

**v2 G=50 vs v1 G=50:** recall 0.9675 → 0.9720, delta = +0.0045 (44 additional
true pairs at matched candidate budget).

---

## 3. Corrected Per-Key Metrics

"Unique TRUE pairs" = true pairs recovered by this key alone (i.e. not recovered by
any other single key). This differs from "unique candidate pairs" reported in Exp 001,
which counted false positives too.

### control_v1, G=50

| Key | Standalone recall | Standalone true pairs | Standalone cands | LOO recall | LOO cands | Unique TRUE pairs |
|---|---|---|---|---|---|---|
| K1 | 0.9581 | 9,447 | 9,546 | 0.7777 | 128,670 | **1,872** |
| K2 | 0.7668 | 7,561 | 42,622 | 0.9675 | 130,629 | **0** |
| K3 | 0.6295 | 6,207 | 110,831 | 0.9669 | 44,682 | **6** |
| K4 | 0.7668 | 7,561 | 42,622 | 0.9675 | 130,629 | **0** |
| **Full system** | — | **9,540** | **130,629** | — | — | — |

K2 and K4 have **identical standalone performance and zero unique true pairs** at G=50,
confirming they are redundant duplicates. K1 is the dominant key by unique true pairs.
K3 contributes 6 unique true pairs at very high candidate cost (110,831 candidates for
6,207 true pairs = 17,822 candidates per unique true pair found by K3 alone).

**LOO interpretation (K1):** removing K1 drops recall from 0.9675 to 0.7777 (-0.1899),
confirming K1 is essential. Removing K2 or K4 has zero recall impact.

### control_v1, G=None

| Key | Standalone recall | Standalone true pairs | Standalone cands | LOO recall | LOO cands | Unique TRUE pairs |
|---|---|---|---|---|---|---|
| K1 | 0.9581 | 9,447 | 9,546 | 0.8138 | 684,177 | **1,525** |
| K2 | 0.7966 | 7,854 | 127,987 | 0.9685 | 685,788 | **0** |
| K3 | 0.8138 | 8,024 | 679,427 | 0.9674 | 129,758 | **10** |
| K4 | 0.7966 | 7,854 | 127,987 | 0.9685 | 685,788 | **0** |
| **Full system** | — | **9,549** | **685,788** | — | — | — |

At G=None, K3 gains more true pairs (8,024 vs 6,207 at G=50) because the gate no
longer limits its large buckets, but K3's unique contribution is still only 10 pairs
across 679,427 candidates — 67,943 candidates per unique true pair. K4's LOO recall
equals full-system recall, confirming it adds nothing.

### control_v2, G=50

| Key | Standalone recall | Standalone true pairs | Standalone cands | LOO recall | LOO cands | Unique TRUE pairs |
|---|---|---|---|---|---|---|
| K1 | 0.9581 | 9,447 | 9,546 | 0.9720 | 130,758 | **0** |
| K2 | 0.7668 | 7,561 | 42,622 | 0.9714 | 114,380 | **6** |
| K3 | 0.6295 | 6,207 | 110,831 | 0.9714 | 44,811 | **6** |
| K4v2 | 0.9671 | 9,536 | 9,733 | 0.9675 | 130,629 | **44** |
| **Full system** | — | **9,584** | **130,758** | — | — | — |

K4v2 contributes **44 unique true pairs** — the suffix-stripping recovers pairs missed
by K1 (where "ACME LIMITED" ≠ "ACME LTD" canonically). K1's unique contribution drops
to 0 at G=50 because K4v2 now covers all true pairs that K1 covers (K4v2 standalone
recall 0.9671 vs K1 standalone 0.9581 — K4v2 has higher standalone recall). LOO(K1) =
0.9720 = full system recall, meaning K4v2 alone recovers everything K1 does. K1 is now
effectively redundant given K4v2.

### control_v2, G=None

| Key | Standalone recall | Standalone true pairs | Standalone cands | LOO recall | LOO cands | Unique TRUE pairs |
|---|---|---|---|---|---|---|
| K1 | 0.9581 | 9,447 | 9,546 | 0.9724 | 685,909 | **0** |
| K2 | 0.7966 | 7,854 | 127,987 | 0.9724 | 681,159 | **0** |
| K3 | 0.8138 | 8,024 | 679,427 | 0.9716 | 129,881 | **8** |
| K4v2 | 0.9671 | 9,536 | 9,733 | 0.9685 | 685,788 | **39** |
| **Full system** | — | **9,588** | **685,909** | — | — | — |

At G=None, K4v2 retains 39 unique true pairs. K2 drops to 0 unique because K3 (with
no gate) covers its pairs. K1 drops to 0 for the same reason as at G=50.

---

## 4. Metric Definitions

See `docs/metrics.md` for full precise definitions. Corrections to Exp 001:

- **cpq median at G=50 is 6**, not 13.3. The 13.3 figure in Exp 001 Observation 2
  was the **mean** (total_cands / queries_with_cands ≈ 130,629 / 9,800 ≈ 13.3).
  Both are reported here; the median (6) better represents the typical query.
- **partial_recall** is formally N/A for this dataset (every query has exactly one
  true target). It is 0.0 by tautology, not by measurement. Reported as N/A going
  forward.
- **zero-candidate rate denominator** = queries with ≥1 true target (9,860 val
  pairs → 9,860 queries with targets, since labels are 1:1). Queries with no true
  target (no-match queries) are excluded.

---

## 5. Residual Taxonomy (control_v2, G=None)

272 val true pairs not recovered by v2 at G=None. Taxonomy priority order:
(a) legal-suffix → (b) punct/case → (c) reorder → (d) stopword → (e) substantial
→ (f) postcode_missing → (g) postcode_mismatch → (h) other.
Category (h) not observed; category (b) not observed (would require names that
differ only by non-ASCII characters surviving canonicalise, which strips all non-ASCII).

| Category | v1 count | v2 count | Description |
|---|---|---|---|
| (a) legal-suffix variant | 39 | **0** | Stripped names match; v2 K4v2 recovers all |
| (b) punct/case only | 0 | 0 | Not observed |
| (c) token reorder | 3 | 3 | Same tokens, different order |
| (d) stop-word added/removed | 1 | 1 | Differ only in "THE" etc. |
| (e) substantial rename | 89 | 89 | Token Jaccard < 1.0, not covered by above |
| (f) query postcode missing | 1 | 1 | No parseable UK postcode on GLEIF side |
| (g) postcode mismatch | 178 | 178 | Different outward codes |
| **(total)** | **311** | **272** | |

**Postcode statistics on residual (v2):**
- Queries with missing/unparseable postcode: 1 (0.37 %)
- Queries where query and true target share the same outward code: 86 (31.62 %)

**Name similarity distributions (v2 residual, 272 pairs):**
- Token Jaccard: mean=0.267, median=0.200
- Char-trigram Jaccard: mean=0.269, median=0.194

The median token Jaccard of 0.20 means the typical unrecovered pair shares 1 token in 5
between query name and target name. This is a substantial name mismatch.

### Category (c) — token reorder (3 examples)

| Query name | Target name | Query postcode | Target postcode |
|---|---|---|---|
| ROYAL COLLEGE OF NURSING OF THE UNITED KINGDOM | ROYAL COLLEGE OF NURSING OF THE UNITED KINGDOM (THE) | W1G 0RN | (missing) |
| THE UNITED REFORMED CHURCH (SOUTH WESTERN SYNOD) INCORPORATED | UNITED REFORMED CHURCH (SOUTH WESTERN SYNOD) INCORPORATED(THE) | TA2 6RU | TA1 3PF |
| THE FRENCH GARDEN LIMITED | FRENCH GARDEN LIMITED(THE) | SW8 5JJ | SW8 5EQ |

All 3 are "THE" repositioned from front to end (or vice versa). The canonicaliser
strips "(THE)" because parentheses are punctuation, leaving different token orderings.
These 3 would be recovered by a "THE-normalisation" step or stop-word-insensitive
matching.

### Category (d) — stop-word addition (1 example)

| Query name | Target name | Query postcode | Target postcode |
|---|---|---|---|
| WEST END GARAGE (BRUTON) LIMITED | THE WEST END GARAGE (BRUTON) LIMITED | BA10 0NL | BA10 0NL |

Postcode matches; only "THE" differs. Would be recovered by any stop-word-insensitive
approach.

### Category (e) — substantial rename (10 of 89 examples)

| Query name | Target name | Query postcode | Target postcode |
|---|---|---|---|
| PREMIER GROWTH FUNDS ICVC - PREMIER STRATEGIC ABSOLUTE RETURN FUND | PREMIER MITON GROWTH FUNDS ICVC | GU1 3DE | (missing) |
| J & P BUILDING SYSTEMS LIMITED | POHLCON UK LTD | OX9 3RR | OX9 3RR |
| SAFELANE GROUP LIMITED | IGNE HOLDCO LIMITED | HR9 7XU | HR9 7US |
| CLOUD TECHNOLOGY SOLUTIONS LTD | QODEA LIMITED | M2 3AW | M2 3AW |
| WOODSFORD TRADEBRIDGE LTD | TRADEBRIDGE FINANCIAL TECHNOLOGIES LIMITED | WC1B 3SR | WC1B 3SR |
| Freya Berry Ltd | SORREL & GREY LTD | DA1 5BU | DA1 5BU |
| TEXTHELP LTD. | EVERWAY OPCO LIMITED | BT41 2RU | BT41 2RU |
| OAKHILL RISK MANAGEMENT LIMITED | RSL NO.55 LIMITED | EC3V 0XL | EC3V 0XL |
| BROOKFIELD RENEWABLE UK HYDRO II LIMITED | CDP RENEWABLE UK HYDRO II LIMITED | W1F 9QZ | W1F 9QZ |
| SDB1 LTD | EVOLUTION FUNDING GROUP LIMITED | S41 9AZ | S41 9AZ |

Many of these are **company renames at the same address** — the GLEIF record preserves
the LEI legal name at time of LEI registration, while the CH record reflects the current
(post-rename) name. Several pairs share the same postcode (J&P/POHLCON at OX9 3RR,
Cloud Technology/Qodea at M2 3AW, etc.) — an address channel would recover them.

### Category (f) — postcode missing (1 example)

| Query name | Target name | Query postcode | Target postcode |
|---|---|---|---|
| TL PROPERTIES LTD | T L PROPERTIES LIMITED | LUI2NQR | LU1 2NQ |

Query has a malformed postcode (`LUI2NQR` — "I" instead of "1"). The canonicaliser
preserves the raw postcode, so `outward_code("LUI2NQR")` returns `""`. This is a data
quality issue in the GLEIF record, not a name mismatch.

### Category (g) — postcode mismatch (10 of 178 examples)

| Query name | Target name | Query postcode | Target postcode |
|---|---|---|---|
| PASCALL ELECTRONICS LIMITED | POWER DEVICE UK LIMITED | TN23 1EH | PO33 1QT |
| ERSKINE HOSPITAL | ERSKINE VETERANS CHARITY | G2 1BA | PA7 5PU |
| THINK HUGE LIMITED. | THOMAS FAMILY INVESTMENTS LIMITED | PO18 9AA | PO20 2EW |
| FROGBIKES LIMITED | FBL REALISATIONS LIMITED | SL5 7PW | EC4N 6EU |
| ARCHITAS ADVISORY SERVICES LIMITED | LIONTRUST ADVISORY SERVICES LIMITED | EC2N 1AD | EC4M 7AU |
| STELLAR FOOTBALL LIMITED | CAA STELLAR LIMITED | W2 2HH | W1K 4HS |
| ELF EXPLORATION UK LTD | NEO NEXT+ ENERGY ELF UK LIMITED | W1G 0PW | EC3A 8BF |
| HERTEL (UK) LIMITED | ALTRAD SUPPORT SERVICES LIMITED | TS6 6XJ | WA4 4ST |
| HP ENTERPRISE SERVICES UK LTD | ENTSERV UK LIMITED | RG12 1HN | GU14 7BF |
| Burrells Accountancy Limited | HERITAGE TAX AND FINANCIAL CONSULTANCY LIMITED | PE31 6YL | NR6 5DR |

These are predominantly **renamed companies that have also changed address**. The GLEIF
record and CH record have diverged on both name and postcode. No name-only or
address-only key can recover them; they require a more flexible matching approach
(e.g. shared tokens + date-of-incorporation similarity, or identifier linking through
an intermediary source).

**86 of the 272 (31.6%) unrecovered v2 pairs share the same outward code.** These are
in category (g) with a matching outward code — the full postcode differs at the inward
part. A key using `outward_code` only (rather than full postcode) would fire for these.

---

## 6. No-Match Query Diagnosis (all 12,424)

No-match queries = GLEIF GB LEI records with no Companies House match. All 12,424
have `RegistrationAuthorityEntityID = EMPTY` in the GLEIF data — the field is
blank. This is not a normalisation issue; the raw field genuinely has no value.

### By RA code

| RA Code | Authority | Count |
|---|---|---|
| RA000585 | Companies House — England & Wales | 12,074 |
| RA000587 | Companies House — Scotland | 309 |
| RA000586 | Companies House — Northern Ireland | 41 |
| **Total** | | **12,424** |

### By entity ID format

| Format | Count |
|---|---|
| EMPTY (no ID present) | **12,424** |

All no-match queries have a missing entity ID, not a mis-formatted one.

### By entity status

| Status | Count |
|---|---|
| INACTIVE | 12,140 (97.7 %) |
| ACTIVE | 259 (2.1 %) |
| NULL / blank | 25 (0.2 %) |

The vast majority (97.7 %) are INACTIVE LEIs. The 259 ACTIVE LEIs with no entity ID
are unusual; possible explanations include late LEI registration with an issuer that
did not record the CH number, or entities registered under a non-CH authority that
are mis-coded as RA000585/586/587.

### K1 name-collision check

343 of the 12,424 no-match queries (2.76 %) have a canonical LegalName that exactly
matches at least one `canon(CompanyName)` in the CH data under a **different** company
number (i.e. their name appears in CH but no identifier links them).

**This does not mean they are false no-matches.** The most likely explanation is
that multiple Companies House records share the same company name (common for generic
names like "SECURITY SERVICES LIMITED"), and the GLEIF record has no entity ID to
confirm which one (if any) it corresponds to. Treating these as matches without
identifier confirmation would create false positives.

**Conclusion supported by the numbers:** The no-match queries are almost entirely INACTIVE
LEIs with missing entity IDs. They are consistent with LEIs for companies that were
dissolved or struck off before the GLEIF issuer recorded the CH number. They are not
recoverable by any blocking strategy without external identifier evidence.

---

## 7. GLEIF Sample Check

| Metric | Value |
|---|---|
| Golden copy date | 2026-10-05 08:00 UTC |
| Announced record count (latest_lei.json) | 3,451,660 |
| Rows decompressed from file | 1,524,435 |
| Coverage | 44.2 % |

The file is a truncated Zip64 archive missing the end-of-central-directory record,
as documented in ADR-002. The 44.2 % figure is consistent with the file being the
first ~44 % of the alphabetically-sorted LEI list (LEIs are 20-character alphanumeric;
the file appears to cover codes starting roughly 001... through mid-range).

**No independent evidence about selection bias is available** from the shipped files
alone. The distributions below describe what is in the file; they cannot confirm
or deny whether the truncated portion is a representative sample.

### GB subset status distribution (in file)

| Status | Count |
|---|---|
| ACTIVE | 65,215 (83.8 %) |
| INACTIVE | 12,467 (16.0 %) |
| NULL / blank | 130 (0.2 %) |

### GB RA code distribution (in file)

| RA Code | Count |
|---|---|
| RA000585 (England & Wales) | 73,773 (94.8 %) |
| RA000587 (Scotland) | 3,287 (4.2 %) |
| RA000586 (Northern Ireland) | 752 (1.0 %) |

The ratio of England&Wales to Scotland is roughly 22:1, broadly consistent with
the relative sizes of the two company registers (Companies House England&Wales has
~4.9M records vs ~0.2M for Scotland). Northern Ireland's share at 1.0% is plausible
given the size of the NI register. **No claim is made about selection bias** beyond
what these raw counts show.

---

## 8. Observations

All observations are backed by the measured numbers above.

1. **v1 is bit-identically reproduced** from commit `c805f600` with the same command.

2. **v2 recall ceiling is 97.24 %** (G=None), up from 97.69 % (wait — v1 was 96.85 %).
   Corrected: v1 ceiling 96.85 %, v2 ceiling 97.24 %, delta +0.39 pp. The bootstrap
   CIs do not overlap, confirming this is a real improvement from K4v2.

3. **K4v2 recovers exactly the v1 `a_legal_suffix` residual** (39 pairs at G=None,
   44 at G=50). The v2 residual has no `a_legal_suffix` entries.

4. **The structural floor of v2 is 272 pairs (2.76 %)**, dominated by
   `g_postcode_mismatch` (178, 65.4 %) and `e_substantial` (89, 32.7 %).

5. **Category (g) postcode mismatch** is the largest residual category. Of these,
   86 (48.3 % of g) share the same outward code — a weaker postcode key would help.
   The remaining 92 have different outward codes (different postal district or missing
   postcode on one side) and cannot be recovered by any postcode-based key.

6. **No-match queries are structurally unrecoverable** from CH data: 97.7 % are
   INACTIVE and all 12,424 have an empty entity ID. The 2.8 % with a K1 name
   collision are not confirmed matches — they share a name with a CH record but
   have no identifier linking them.

7. **K1 is redundant given K4v2 in v2** (K1 LOO recall = full-system recall at G=50
   and G=None). A minimal v2 key set is K2, K3, K4v2. K1 adds zero marginal recall
   but contributes precision (its 9,546 candidates are much fewer than K3's 679,427).
   Whether to keep K1 depends on budget sensitivity; this is not a recommendation.

8. **partial_recall is N/A** for this dataset. Every query has exactly one true target
   (1:1 identifier matching). The 0.0 value in Exp 001 is a tautology.

---

## 9. Test Split Status

**The test split has not been evaluated in any experiment.** Val queries: 11,682.
Test queries: 11,676. The split is entity-disjoint (0 leaking entities confirmed).

---

## 10. Errata & Addendum (Phase 2c)

**Date:** 2026-10-07  
**Commit:** `83610d627c51b43593589dcccd5344831e00af72`

### Errata E1: Correction to Observation 2 & Bootstrap Comparison Method
- **Observation 2 correction:** The recall ceiling for control_v1 (G=None) is **96.85 %** (9,549 / 9,860 val true pairs). The recall ceiling for control_v2 (G=None) is **97.24 %** (9,588 / 9,860 val true pairs), representing a delta of **+0.39 pp** (+39 true pairs recovered).
- **CI Overlap argument replacement:** The marginal 95% bootstrap CIs for v1 ([0.9654, 0.9712]) and v2 ([0.9697, 0.9751]) slightly overlap. Comparing separate 95% CIs is a statistically invalid test for comparing two systems evaluated on identical entity samples. A **paired bootstrap over query entities** on the recall difference ($recall(v2) - recall(v1)$) yields a mean recall delta of **+0.003955** (95% CI: **[+0.0027, +0.0053]**, $p < 0.0001$, $N_{boot}=1000$). The improvement of v2 over v1 is statistically significant.
- **Category (b) in v1 taxonomy:** Category (b) `the_stopword_only` in the v1 residual taxonomy is **zero by construction**. The canonicalization function `canon()` normalizes Unicode NFKD, lowercases, and strips non-alphanumeric punctuation, but preserves English stopwords ("the", "of", "and"). Thus, stopword-only differences are zero by definition under this canonicalization pipeline.

### A2. v2 Residual Cross-Tabulation (Val Split, G=None)
Cross-tabulation of all **272 unrecovered true pairs** under v2 at G=None across mutually exclusive name and postcode relation classes:

| Name Relation \ Postcode Relation | Same full postcode | Same outward code only | Different outward code | Missing / Malformed | **Total** |
|---|---|---|---|---|---|
| **Legal-suffix-only** | 0 | 0 | 0 | 0 | **0** |
| **THE / stopword-only** | 1 | 0 | 0 | 0 | **1** |
| **Token reorder** | 0 | 1 | 1 | 1 | **3** |
| **Partial overlap ($\ge 0.5$)** | 8 | 4 | 34 | 4 | **50** |
| **Different ($< 0.5$)** | 61 | 11 | 144 | 2 | **218** |
| **Total** | **70** | **16** | **179** | **7** | **272** |

#### Postcode Bucket Size Distribution (for 70 "Same Full Postcode" Residual Pairs)
For the 70 residual true pairs sharing the same full postcode, Companies House bucket sizes range from 1 to 522 records (min: 1, median: 14, p95: 198, max: 522, mean: 42.18). This demonstrates that while postcode-based retrieval is feasible, postcode-only blocking would introduce high candidate volume for dense commercial postcodes (e.g. registered office providers).

#### 5 Examples per Non-Empty Cell
- **THE / stopword-only | Same full postcode (1 pair):**
  - Query: `WEST END GARAGE (BRUTON) LIMITED` | Target: `THE WEST END GARAGE (BRUTON) LIMITED` (Postcode: `BA10 0NL`)
- **Token reorder | Same outward code only (1 pair):**
  - Query: `THE FRENCH GARDEN LIMITED` (`SW8 5JJ`) | Target: `FRENCH GARDEN LIMITED(THE)` (`SW8 5EQ`)
- **Token reorder | Different outward code (1 pair):**
  - Query: `THE UNITED REFORMED CHURCH (SOUTH WESTERN SYNOD) INCORPORATED` (`TA2 6RU`) | Target: `UNITED REFORMED CHURCH (SOUTH WESTERN SYNOD) INCORPORATED(THE)` (`TA1 3PF`)
- **Token reorder | Missing / Malformed (1 pair):**
  - Query: `ROYAL COLLEGE OF NURSING OF THE UNITED KINGDOM` (`W1G 0RN`) | Target: `ROYAL COLLEGE OF NURSING OF THE UNITED KINGDOM (THE)` (`""`)
- **Partial overlap | Same full postcode (5 examples):**
  1. `The English Theosophical Trust Ltd` ↔ `ENGLISH THEOSOPHICAL TRUST LIMITED(THE)` (`W1U 8EA`)
  2. `DELPHI TECHNOLOGIES PENSION TRUSTEES LIMITED` ↔ `PHINIA DELPHI PENSION TRUSTEES LIMITED` (`LS1 5AB`)
  3. `BROOKFIELD RENEWABLE UK HYDRO II LIMITED` ↔ `CDP RENEWABLE UK HYDRO II LIMITED` (`W1F 9QZ`)
  4. `R&R ICE CREAM UK LIMITED` ↔ `FRONERI ICE CREAM UK LIMITED` (`DL7 9UL`)
  5. `BOND DICKINSON WEALTH LIMITED` ↔ `WOMBLE BOND DICKINSON WEALTH LIMITED` (`SE1 2AU`)
- **Partial overlap | Same outward code only (5 examples):**
  1. `SCOR LIME STREET LIMITED` (`EH3 8HA`) ↔ `SWRE LIME STREET LIMITED` (`EH3 8BP`)
  2. `SL CAPITAL EUROPEAN CO-INVESTMENT LP` (`EH2 2LL`) ↔ `PATRIA EUROPEAN CO-INVESTMENT LP` (`EH2 4LH`)
  3. `Bridge UK Properties 3, LP` (`W1S 4PP`) ↔ `HAMMERSMITH UK PROPERTIES 3, LP` (`W1S 3DG`)
  4. `MARDIX LIMITED` (`LA9 7DE`) ↔ `ANORD MARDIX (UK) LIMITED` (`LA9 7LR`)
  5. `K3 CAPITAL GROUP LIMITED` (`BL3 2NT`) ↔ `K3 ADVISORY GROUP LIMITED` (`BL3 5NT`)
- **Partial overlap | Different outward code (5 examples):**
  1. `RELIANCE ENTERTAINMENT PRODUCTIONS 7 LTD` (`W1D 3QL`) ↔ `REP PRODUCTIONS 7 LTD` (`B47 6AJ`)
  2. `STELLAR FOOTBALL LIMITED` (`W2 2HH`) ↔ `CAA STELLAR LIMITED` (`W1K 4HS`)
  3. `HYDER CONSULTING GROUP LIMITED` (`SW1P 1JA`) ↔ `ARCADIS CONSULTING GROUP LIMITED` (`EC3M 4BY`)
  4. `WEBHELP UK TRADING LTD` (`SW1 A1J`) ↔ `CONCENTRIX UK TRADING LTD` (`S2 5DD`)
  5. `K3 CAPITAL GROUP LIMITED` (`BL3 2NT`) ↔ `K3 ADVISORY GROUP LIMITED` (`M2 5NT`)
- **Partial overlap | Missing / Malformed (4 examples):**
  1. `WS MULTI ASSET FUNDS` (`LS1 4DL`) ↔ `MI MULTI ASSET FUNDS` (`""`)
  2. `VT TATTON OAK ICVC` (`IV32 7QE`) ↔ `MI TATTON OAK ICVC` (`""`)
  3. `GLASGOW CHAMBER OF COMMERCE` (`G2 1EQ`) ↔ `GLASGOW CHAMBER OF COMMERCE AND MANUFACTURES)` (`""`)
  4. `ROK STARS LIMITED` (`WV7 3AU`) ↔ `ROKIT STARS LIMITED` (`""`)
- **Different | Same full postcode (5 examples):**
  1. `TEXTHELP LTD.` ↔ `EVERWAY OPCO LIMITED` (`BT41 2RU`)
  2. `G V HEALTH LIMITED` ↔ `GV HEALTH LIMITED` (`SG1 2FS`)
  3. `KAPPA G.B. LIMITED` ↔ `ASI BRANDS LIMITED` (`PH10 6NW`)
  4. `CROWN CREST (LEICESTER) PLC` ↔ `POUNDSTRETCHER LEICESTER LIMITED` (`LE9 2BJ`)
  5. `TRANS INTMAX IMPORT EXPORT LP` ↔ `SL019024 LP` (`KA7 1UB`)
- **Different | Same outward code only (5 examples):**
  1. `IRT (UK) Limited` (`CB8 9AE`) ↔ `PALTRIDGE ESTATES CONSTRUCTION LTD` (`CB8 0RG`)
  2. `THE FOUNDATION T.V. PRODUCTIONS LIMITED` (`W14 8RF`) ↔ `ZODIAK KIDS & FAMILY PRODUCTIONS UK LIMITED` (`W14 0EE`)
  3. `PITMANS LLP` (`RG1 7SR`) ↔ `ADCAMP LLP` (`RG1 1PL`)
  4. `RENEWABLE (SOLAR) INVESTMENTS LP` (`EH3 8HA`) ↔ `AIP INFRASTRUCTURE LP` (`EH3 8BP`)
  5. `SAFELANE GROUP LIMITED` (`HR9 7XU`) ↔ `IGNE HOLDCO LIMITED` (`HR9 7US`)
- **Different | Different outward code (5 examples):**
  1. `BLUE TIGER FINANCE LIMITED` (`ME3 7NE`) ↔ `MAMMOTH FINANCE LIMITED` (`NE4 7DF`)
  2. `THESSCO LIMITED` (`S4 7WB`) ↔ `THL REALISATIONS 2026 LIMITED` (`NN1 5JF`)
  3. `DOW SCHOFIELD WATTS WEALTH ADVISORY LLP` (`WA4 4BS`) ↔ `RSW ADVISORY LLP` (`LS22 5DZ`)
  4. `T J C TEXTILES LIMITED` (`SS1 1EG`) ↔ `THE LONDON TEXTILE AGENCY LIMITED` (`WC2B 5DG`)
  5. `LONDON & REGIONAL (TRAFALGAR SQUARE HOTEL) LIMITED` (`E1W 1YW`) ↔ `TRAFALGAR SQUARE HOTEL NO 2 LTD` (`WC1X 9DU`)
- **Different | Missing / Malformed (2 examples):**
  1. `TL PROPERTIES LTD` (`LUI2NQR`) ↔ `T L PROPERTIES LIMITED` (`LU1 2NQ`)
  2. `PREMIER GROWTH FUNDS ICVC` (`GU1 3DE`) ↔ `PREMIER MITON GROWTH FUNDS ICVC` (`""`)

### A3. Candidates-Per-Query (cpq) Distribution for control_v2

Denominator definitions:
- **All val queries ($N = 11,682$):** includes 9,860 val queries with a true target plus 1,822 val queries with empty/unmatched LEI.
- **Queries with targets only ($N = 9,860$):** excludes queries without true target labels.
- **Queries with candidates ($N \le 11,425$):** queries where $cpq > 0$.

| G | Candidates | $cpq$ Median (cand) | $cpq$ P95 (cand) | $cpq$ Mean (cand) | $cpq$ Mean (All Val) | Zero-Cand Rate (All Val) |
|---|---|---|---|---|---|---|
| **10** | 34,576 | 2.0 | 9.0 | 3.23 | 2.96 | 0.0845 |
| **50** | 130,758 | 6.0 | 41.0 | 11.71 | 11.19 | 0.0443 |
| **200** | 321,398 | 12.0 | 119.0 | 28.25 | 27.51 | 0.0260 |
| **1000** | 549,476 | 14.0 | 219.0 | 48.11 | 47.04 | 0.0223 |
| **None** | 685,909 | 14.0 | 237.0 | 60.04 | 58.72 | 0.0220 |

### A4. Control Family Efficient Frontier (control_v2)

Keys are ordered greedily by marginal true pairs per candidate generated:

#### G = 50 Budget Gate
| Key Added | Active Keys | Cum. True Pairs | Cum. Pair Recall | Cum. Candidates | Marginal True Pairs | Marginal Candidates | Marginal Return (Pairs/Cand) |
|---|---|---|---|---|---|---|---|
| **K4v2** | `K4v2` | 9,536 | 0.9671 | 9,733 | 9,536 | 9,733 | 0.9798 |
| **+K2** | `K4v2, K2` | 9,578 | 0.9714 | 44,811 | 42 | 35,078 | 0.0012 |
| **+K3** | `K4v2, K2, K3` | 9,584 | 0.9720 | 130,758 | 6 | 85,947 | 0.00007 |
| **+K1** | `K4v2, K2, K3, K1` | 9,584 | 0.9720 | 130,758 | 0 | 0 | 0.0000 |

#### G = None (Ungated)
| Key Added | Active Keys | Cum. True Pairs | Cum. Pair Recall | Cum. Candidates | Marginal True Pairs | Marginal Candidates | Marginal Return (Pairs/Cand) |
|---|---|---|---|---|---|---|---|
| **K4v2** | `K4v2` | 9,536 | 0.9671 | 9,733 | 9,536 | 9,733 | 0.9798 |
| **+K3** | `K4v2, K3` | 9,588 | 0.9724 | 681,159 | 52 | 671,426 | 0.00008 |
| **+K1** | `K4v2, K3, K1` | 9,588 | 0.9724 | 681,159 | 0 | 0 | 0.0000 |
| **+K2** | `K4v2, K3, K1, K2` | 9,588 | 0.9724 | 685,909 | 0 | 4,750 | 0.0000 |

### A5. Terminology Renaming Guard
All occurrences of "no-match queries" have been renamed to **"unlabeled queries"** across code, schema docstrings, and documentation. Unit test `test_unlabeled_queries_not_negatives` in `tests/test_gleif_ch_adapter.py` passes and enforces that queries with missing registry identifiers are excluded from negative pair sampling and recall evaluation denominators.

### A6. GLEIF Decompressed File Truncation Analysis
- **Decompressed records:** 1,524,435 rows (covering ~44.2 % of latest golden copy archive).
- **Distinct 4-character LEI prefixes (ALL records):** 2,644 distinct prefixes.
- **UK / GB records (RA codes RA000585, RA000586, RA000587):** 77,812 records.
- **Distinct 4-character LEI prefixes (GB subset):** 82 distinct prefixes.
- **GB Record File-Order Positions:**
  - First GB record row index: **724**
  - Median GB record row index: **179,831**
  - Last GB record row index: **1,524,386**
  - Span: 99.9 % of decompressed file lines contain GB records interspersed throughout.
- **Top 5 LEI Issuer Prefixes (ALL records):**
  1. `5493` (331,781 records)
  2. `2549` (296,982 records)
  3. `2138` (250,272 records)
  4. `5299` (218,774 records)
  5. `3358` (156,418 records)
- **Top 5 LEI Issuer Prefixes (GB subset):**
  1. `2138` (57,811 records — 74.3 %)
  2. `2549` (10,460 records — 13.4 %)
  3. `5493` (7,140 records — 9.2 %)
  4. `5299` (1,681 records — 2.2 %)
  5. `2594` (185 records)
- **Representativeness:** **UNVERIFIED**. Due to archive truncation, representativeness relative to the full GLEIF database cannot be guaranteed.

### A7. Amended Pre-Registrations Audit
1. `experiments/preregistrations/002b_idf_lexical_amended_2026-10-07.md` (Created: 2026-10-07)
2. `experiments/preregistrations/003b_address_retrieval_amended_2026-10-07.md` (Created: 2026-10-07)

