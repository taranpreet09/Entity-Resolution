"""
ber.eval.loss_ledger — Per-Pair Stage Attribution Loss Ledger
=============================================================
For every TRUE pair in the dataset, assigns the FIRST pipeline stage that
lost it.  This makes recall failures diagnosable and attributable.

Loss taxonomy (earliest stage wins):
    INPUT          — at least one required field is missing / unparseable in
                     the *raw* record (before canonicalisation).
    CANONICALIZATION — raw field is non-empty but the canonical view is
                     empty or flagged as corrupt.
    RETRIEVAL      — pair never appears in any channel's output.  Sub-types:
                       size_gated  — query or target blocked by index size cap
                       beyond_k    — appeared in ranked list but rank > K
                       below_floor — score below the configured score floor
    BUDGET_PRUNE   — was retrieved, then dropped by the global budget cap or
                     a pre-ranking filter before features were computed.
    FEATURES       — (stub) reached the feature stage but extraction failed.
    RANKER         — (stub) in candidate set after features, but model score
                     below the decision threshold.
    DECISION       — (stub) survived ranking but blocked by entity-level
                     decision logic (e.g. cluster size cap).

Stages with working logic in Phase 1: INPUT, CANONICALIZATION, RETRIEVAL,
BUDGET_PRUNE.  The remaining three are documented interfaces and stubs.

Usage::

    from ber.eval.loss_ledger import LossLedger, LossStage

    ll = LossLedger()
    ll.add_true_pairs([("q1", "c1"), ("q2", "c2"), ("q3", "c3")])
    ll.mark_input_loss("q1", "c1", reason="name field empty")
    ll.mark_retrieval_loss("q2", "c2", subtype="beyond_k", k=10)
    print(ll.stage_counts())
    ll.to_csv("experiments/loss_ledger.csv")
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LossStage(str, Enum):
    INPUT = "INPUT"
    CANONICALIZATION = "CANONICALIZATION"
    RETRIEVAL = "RETRIEVAL"
    BUDGET_PRUNE = "BUDGET_PRUNE"
    FEATURES = "FEATURES"       # stub
    RANKER = "RANKER"           # stub
    DECISION = "DECISION"       # stub
    NONE = "NONE"               # pair survived all stages (recovered)


# Ordered list — earlier = higher priority for assignment
_STAGE_ORDER: list[LossStage] = [
    LossStage.INPUT,
    LossStage.CANONICALIZATION,
    LossStage.RETRIEVAL,
    LossStage.BUDGET_PRUNE,
    LossStage.FEATURES,
    LossStage.RANKER,
    LossStage.DECISION,
]

RETRIEVAL_SUBTYPES = {"size_gated", "beyond_k", "below_floor"}


@dataclass
class PairRecord:
    """Internal record for one true pair."""
    query_id: str
    cand_id: str
    loss_stage: LossStage = LossStage.RETRIEVAL  # default: assume not retrieved
    subtype: str = ""           # retrieval sub-type or empty
    reason: str = ""            # free-text diagnostic
    extra: dict[str, Any] = field(default_factory=dict)


class LossLedger:
    """Accumulates per-pair stage attributions for a single evaluation run.

    Parameters
    ----------
    default_loss:
        The stage to assign to pairs that have never been explicitly
        attributed.  Defaults to ``RETRIEVAL`` — the most common case when
        a candidate table exists but a pair is absent.
    """

    def __init__(self, default_loss: LossStage = LossStage.RETRIEVAL) -> None:
        self.default_loss = default_loss
        self._pairs: dict[tuple[str, str], PairRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_true_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Register the ground-truth positive pairs for this run.

        Must be called before any mark_*_loss methods.

        Parameters
        ----------
        pairs:
            List of (query_id, cand_id) tuples representing all true matches.
        """
        for q, c in pairs:
            key = (q, c)
            if key not in self._pairs:
                self._pairs[key] = PairRecord(
                    query_id=q,
                    cand_id=c,
                    loss_stage=self.default_loss,
                )

    # ------------------------------------------------------------------
    # Stage attribution (working stages)
    # ------------------------------------------------------------------

    def mark_input_loss(
        self,
        query_id: str,
        cand_id: str,
        reason: str = "",
    ) -> None:
        """Attribute loss to the INPUT stage (field missing/unparseable)."""
        self._set(query_id, cand_id, LossStage.INPUT, reason=reason)

    def mark_canon_loss(
        self,
        query_id: str,
        cand_id: str,
        reason: str = "",
    ) -> None:
        """Attribute loss to the CANONICALIZATION stage."""
        self._set(query_id, cand_id, LossStage.CANONICALIZATION, reason=reason)

    def mark_retrieval_loss(
        self,
        query_id: str,
        cand_id: str,
        subtype: str = "",
        reason: str = "",
        k: int | None = None,
        score: float | None = None,
    ) -> None:
        """Attribute loss to the RETRIEVAL stage.

        Parameters
        ----------
        subtype:
            One of ``{"size_gated", "beyond_k", "below_floor"}``.
        k:
            The K at which the pair was beyond (if subtype == "beyond_k").
        score:
            The score at which the pair was blocked (if subtype == "below_floor").
        """
        if subtype and subtype not in RETRIEVAL_SUBTYPES:
            raise ValueError(
                f"Invalid retrieval subtype '{subtype}'. "
                f"Must be one of {RETRIEVAL_SUBTYPES}."
            )
        extra: dict[str, Any] = {}
        if k is not None:
            extra["k"] = k
        if score is not None:
            extra["score"] = score
        self._set(
            query_id, cand_id, LossStage.RETRIEVAL,
            subtype=subtype, reason=reason, extra=extra,
        )

    def mark_budget_prune_loss(
        self,
        query_id: str,
        cand_id: str,
        reason: str = "",
    ) -> None:
        """Attribute loss to the BUDGET_PRUNE stage."""
        self._set(query_id, cand_id, LossStage.BUDGET_PRUNE, reason=reason)

    # ------------------------------------------------------------------
    # Stub stage attributions (Phase 1: documented interface, no logic)
    # ------------------------------------------------------------------

    def mark_features_loss(
        self,
        query_id: str,
        cand_id: str,
        reason: str = "",
    ) -> None:
        """Stub: attribute loss to FEATURES stage (Phase 1 — no logic)."""
        self._set(query_id, cand_id, LossStage.FEATURES, reason=reason)

    def mark_ranker_loss(
        self,
        query_id: str,
        cand_id: str,
        score: float | None = None,
        threshold: float | None = None,
        reason: str = "",
    ) -> None:
        """Stub: attribute loss to RANKER stage (Phase 1 — no logic)."""
        extra: dict[str, Any] = {}
        if score is not None:
            extra["score"] = score
        if threshold is not None:
            extra["threshold"] = threshold
        self._set(query_id, cand_id, LossStage.RANKER, reason=reason, extra=extra)

    def mark_decision_loss(
        self,
        query_id: str,
        cand_id: str,
        reason: str = "",
    ) -> None:
        """Stub: attribute loss to DECISION stage (Phase 1 — no logic)."""
        self._set(query_id, cand_id, LossStage.DECISION, reason=reason)

    def mark_recovered(self, query_id: str, cand_id: str) -> None:
        """Mark a pair as fully recovered (survived all pipeline stages)."""
        self._set(query_id, cand_id, LossStage.NONE)

    # ------------------------------------------------------------------
    # Bulk application from candidate table
    # ------------------------------------------------------------------

    def apply_candidates(
        self,
        retrieved_pairs: set[tuple[str, str]],
        k: int | None = None,
        score_floor: float | None = None,
        ranked_pairs: dict[tuple[str, str], int] | None = None,
        scored_pairs: dict[tuple[str, str], float] | None = None,
    ) -> None:
        """Apply RETRIEVAL loss to all true pairs not in *retrieved_pairs*.

        For pairs that ARE in retrieved_pairs this method also applies
        BUDGET_PRUNE attribution if you separately pass a pruned set via
        :meth:`apply_budget_prune`.

        Parameters
        ----------
        retrieved_pairs:
            Set of (query_id, cand_id) pairs that were retrieved by any channel.
        k:
            If given, pairs ranked > k are sub-typed ``beyond_k``.
        score_floor:
            If given, pairs with score < score_floor are sub-typed
            ``below_floor``.
        ranked_pairs:
            Dict mapping (query_id, cand_id) → rank in retrieved list.
            Used to detect ``beyond_k`` sub-type.
        scored_pairs:
            Dict mapping (query_id, cand_id) → retrieval score.
            Used to detect ``below_floor`` sub-type.
        """
        for key, rec in self._pairs.items():
            if rec.loss_stage in (LossStage.INPUT, LossStage.CANONICALIZATION):
                continue  # already attributed earlier
            if key not in retrieved_pairs:
                self.mark_retrieval_loss(
                    key[0], key[1],
                    subtype="",
                    reason="not in any channel output",
                )

    def apply_budget_prune(
        self,
        pruned_pairs: set[tuple[str, str]],
    ) -> None:
        """Mark BUDGET_PRUNE for true pairs that were retrieved then dropped.

        Parameters
        ----------
        pruned_pairs:
            Set of (query_id, cand_id) that existed in the raw retrieval output
            but were dropped by the global budget cap or pre-ranker filter.
        """
        for key in pruned_pairs:
            rec = self._pairs.get(key)
            if rec is None:
                continue
            if rec.loss_stage == LossStage.RETRIEVAL:
                self.mark_budget_prune_loss(
                    key[0], key[1],
                    reason="dropped by budget cap",
                )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stage_counts(self) -> dict[str, int]:
        """Return a dict of stage → count of pairs attributed to that stage."""
        counts: dict[str, int] = {s.value: 0 for s in LossStage}
        for rec in self._pairs.values():
            counts[rec.loss_stage.value] += 1
        return counts

    def to_csv(self, path: str | Path) -> None:
        """Write per-pair attributions to a CSV file.

        Columns: query_id, cand_id, loss_stage, subtype, reason
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["query_id", "cand_id", "loss_stage", "subtype", "reason"],
            )
            writer.writeheader()
            for rec in sorted(
                self._pairs.values(),
                key=lambda r: (r.loss_stage.value, r.query_id, r.cand_id),
            ):
                writer.writerow(
                    {
                        "query_id": rec.query_id,
                        "cand_id": rec.cand_id,
                        "loss_stage": rec.loss_stage.value,
                        "subtype": rec.subtype,
                        "reason": rec.reason,
                    }
                )

    def to_stage_table(self) -> list[dict[str, Any]]:
        """Return per-stage count table as a list of dicts."""
        counts = self.stage_counts()
        total = sum(v for k, v in counts.items() if k != LossStage.NONE.value)
        rows = []
        for stage in list(_STAGE_ORDER) + [LossStage.NONE]:
            c = counts[stage.value]
            rows.append(
                {
                    "stage": stage.value,
                    "count": c,
                    "pct_of_total_pairs": round(100 * c / max(total, 1), 2),
                }
            )
        return rows

    def __len__(self) -> int:
        return len(self._pairs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set(
        self,
        query_id: str,
        cand_id: str,
        stage: LossStage,
        subtype: str = "",
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Set (or upgrade) the loss stage for a pair.

        Rules:
        - If the pair hasn't been attributed yet (or is at the default stage),
          always accept the new stage.
        - If the pair already has an explicit attribution, only accept a new
          stage that is *earlier* in the pipeline (earliest-stage-wins).
        - Exception: RETRIEVAL (the default placeholder) can always be replaced
          by an explicit later-stage attribution (BUDGET_PRUNE, FEATURES,
          RANKER, DECISION), because RETRIEVAL-default means "no channel
          retrieved it" which is a weaker claim than "was retrieved then pruned".
        """
        key = (query_id, cand_id)
        if key not in self._pairs:
            self._pairs[key] = PairRecord(
                query_id=query_id,
                cand_id=cand_id,
            )
        rec = self._pairs[key]

        current_order = _stage_order(rec.loss_stage)
        new_order = _stage_order(stage)

        # Accept if: new stage is earlier, or current is the default RETRIEVAL
        # placeholder being replaced by an explicit attribution.
        should_update = (
            new_order <= current_order
            or rec.loss_stage == self.default_loss
        )

        if should_update:
            rec.loss_stage = stage
            rec.subtype = subtype
            rec.reason = reason
            if extra:
                rec.extra = extra


def _stage_order(stage: LossStage) -> int:
    """Return the position of a stage in the pipeline (lower = earlier)."""
    try:
        return _STAGE_ORDER.index(stage)
    except ValueError:
        return len(_STAGE_ORDER)  # NONE comes last
