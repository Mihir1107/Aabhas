"""
The rule-based decision layer (Part 5.2 / 5.3).

Six tiers, not a binary. A system that can only say PASS or FAIL forces every
ambiguous part into a wrong bucket, and two of these tiers are the ones that
make the system defensible rather than merely aggressive:

    MEASUREMENT_INVALID  the data-quality gate failed; this is not a verdict
                         about the part at all
    FIXTURE_SUSPECT      the anomaly clusters by board position, so the likely
                         cause is the chamber or the socket, not the component

Fusion is rule-based, not learned. That is deliberate: explainability is a
scored criterion, and a rule the inspector can read beats a weighting they
cannot. **The model never overrides a hard engineering limit in either
direction** -- it cannot loosen a datasheet limit, and it cannot reject a part
the limits pass without saying which statistical rule fired and why.

Thresholds are operating points, not learned parameters, and each is named
where it is used so the whole ladder is auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ["PASS", "WATCH", "REVIEW", "REJECT", "MEASUREMENT_INVALID",
         "FIXTURE_SUSPECT"]

ACTIONS = {
    "PASS": "Release to next operation. No action.",
    "WATCH": ("Release, but tag the component ID for trend monitoring at the "
              "next screen. No rework."),
    "REVIEW": ("Hold. Repeat the measurement to rule out test error, then refer "
               "to a reliability engineer with this report."),
    "REJECT": ("Hold and quarantine. Refer for failure analysis; do not ship "
               "pending engineering disposition."),
    "MEASUREMENT_INVALID": ("Do not disposition on this data. Re-test the "
                            "component; check chamber log and tester "
                            "calibration for the affected checkpoint."),
    "FIXTURE_SUSPECT": ("Do NOT reject the component on this evidence. Raise a "
                        "fixture/chamber investigation for the affected board "
                        "and re-test these positions in a different socket."),
}


class DecisionPolicy:
    """Thresholds are quantiles of the GOOD population on held-out lots, so
    'REVIEW' means a fixed yield-loss budget rather than an arbitrary number."""

    def __init__(self, ev_obj, review_budget: float = 0.02,
                 watch_budget: float = 0.07, seed_lots: str = "val"):
        self.e = ev_obj
        lot = ev_obj.lot
        cal = lot.isin(getattr(ev_obj.split, seed_lots))
        good = ~ev_obj.y
        s = ev_obj.scores
        self.detector_cols = ["L1_dynamic_MAD", "L2_both", "L3a_MCD", "L3b_Q",
                              "L3b_T2", "L3c_kNN", "L4b_LOF", "L4d_AutoEnc"]
        self.fused_col = "CUM_C4"
        ref = s.loc[(cal & good).to_numpy()]
        self.thr_review = {c: float(np.nanquantile(ref[c], 1 - review_budget))
                           for c in self.detector_cols + [self.fused_col]}
        self.thr_watch = {c: float(np.nanquantile(ref[c], 1 - watch_budget))
                          for c in self.detector_cols + [self.fused_col]}
        self.review_budget = review_budget
        self.watch_budget = watch_budget
        # Computed once. `fired()` is called for every part in a lot and this
        # median is over the whole 120k-row score frame; recomputing it per call
        # made a 500-part worklist take minutes instead of seconds. The value
        # does not depend on the component, so the result is identical.
        self._median = self.e.scores[self.detector_cols].median()

    def fired(self, cid: str, level: str = "review") -> dict[str, float]:
        """Which detectors fired, and by how much, in a normalised space.

        Normalisation is 'how far past its own threshold', so 1.0 means exactly
        at the operating point and 2.0 means twice as far past it as the
        threshold is from the good-part median. Without this the raw scores are
        not comparable to each other and 'which detector drove the decision' is
        unanswerable.
        """
        thr = self.thr_review if level == "review" else self.thr_watch
        med = self._median
        out = {}
        for c in self.detector_cols:
            v = float(self.e.scores.loc[cid, c])
            span = thr[c] - float(med[c])
            out[c] = (v - float(med[c])) / span if span > 0 else 0.0
        return out

    def decide(self, cid: str, flagged_pool: pd.Series | None = None) -> dict:
        e = self.e
        dq = e.data_quality(cid)
        row = {"component_id": cid, "tier": None, "reasons": [],
               "detector_strength": self.fired(cid)}

        # 1. data-quality gate comes first, before any model
        if not dq["valid"] and not dq["pulled_failed"]:
            row["tier"] = "MEASUREMENT_INVALID"
            row["reasons"].append(
                f"{dq['n_missing_equipment']} checkpoint(s) lost to equipment or "
                "chamber fault; the record is incomplete for a non-part reason")
            return row

        # 2. hard engineering limit. The model never overrides this.
        if bool(e.flags.loc[cid, "L0"]):
            row["tier"] = "REJECT"
            row["reasons"].append("datasheet limit violated at a measured checkpoint")
            return row

        # 3. a part that failed hard and was pulled is already disposed of
        if dq["pulled_failed"]:
            row["tier"] = "REJECT"
            row["reasons"].append(
                "component failed during burn-in and was removed from the oven")
            return row

        fused = float(e.scores.loc[cid, self.fused_col])
        strong = fused >= self.thr_review[self.fused_col]
        weak = fused >= self.thr_watch[self.fused_col]

        # 4. spatial gate: is this the chamber rather than the component?
        if (strong or weak) and flagged_pool is not None:
            sp = e.spatial_check(cid, flagged_pool)
            if sp["clustered"]:
                row["tier"] = "FIXTURE_SUSPECT"
                row["spatial"] = sp
                row["reasons"].append(
                    f"{sp['n_flagged_on_board']} of {sp['n_on_board']} components on "
                    f"board {sp['board_id']} are flagged (expected "
                    f"{sp['expected_flagged']:.1f}, p={sp['p_value']:.1e}); the "
                    "anomaly tracks board position, not the component")
                return row

        # 5. Module B forecast against a lot-derived safe limit
        mb_breach = []
        for p in e.params:
            u = float(e.mb_upper.loc[cid, p]) if cid in e.mb_upper.index else np.nan
            ls = float(e.l_safe[p].get(cid, np.nan))
            if np.isfinite(u) and np.isfinite(ls) and u > ls:
                mb_breach.append((p, u, ls))
        if mb_breach:
            p, u, ls = mb_breach[0]
            row["reasons"].append(
                f"forecast 95% upper bound on {p} at 168 h is {u:.3f}, above the "
                f"lot-derived safe limit {ls:.3f}")

        if strong:
            row["tier"] = "REVIEW"
        elif weak or mb_breach:
            row["tier"] = "WATCH"
        else:
            row["tier"] = "PASS"
        return row
