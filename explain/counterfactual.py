"""
Counterfactual: the smallest single-measurement change that clears the
rule-based evidence.

    "The rule-based evidence would no longer flag this part if leakage at
     24 h had been at or below 18.2 nA."

Directly actionable, and the one explanation an inspector can sanity check
against the physical part.

SCOPE, stated plainly. The counterfactual is computed against the RULE LAYER --
the lot-relative robust z (6 sigma) and the robust Mahalanobis D^2 against the
training-lot reference -- because those can be recomputed exactly from the raw
measurements. The fitted L2-L4 detectors behind the fused score are not
re-evaluated (that needs the model objects), so a counterfactual says "the
rules would stop firing", not "the tier would become PASS".

VALIDATION. The candidate value is found by bisection, then applied, and the
WHOLE record is re-checked: every checkpoint must clear both rules. If another
checkpoint still fires, a one-value change cannot clear the part and the
counterfactual is discarded rather than reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

from explain.rules import UNITS


class CounterfactualEngine:
    """Re-scores a single perturbed part against the frozen lot reference.

    Only the detectors that can be recomputed exactly and cheaply from the raw
    measurements are used: the lot-relative robust z (Module A L1/L2 space) and
    the robust Mahalanobis distance (L3a). The fitted L4 models are NOT
    re-evaluated, because that would require the model objects this session is
    not allowed to refit. The counterfactual is therefore reported against the
    RULE-BASED evidence, which is exactly the part of the system an inspector is
    being asked to trust, and that scope limit is stated on the report page.
    """

    def __init__(self, e, policy):
        self.e = e
        self.policy = policy
        self._cache: dict = {}

    def _lot_ref(self, lot: str, t: float):
        key = (lot, t)
        if key in self._cache:
            return self._cache[key]
        m = self.e.ds.meas
        s = m[(m["lot_id"] == lot) & (m["checkpoint_h"] == t)
              & (m["measurement_status"] == "MEASURED")]
        X = s[self.e.params].to_numpy(float)
        mcd = MinCovDet(support_fraction=0.9, random_state=0).fit(X)
        inv = np.linalg.inv(mcd.covariance_)
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - med), axis=0)
        # same label-free reference the rule text quotes (training-lot goods)
        p99 = self.e.d2_reference(t)
        out = (mcd.location_, inv, med, mad, p99)
        self._cache[key] = out
        return out

    def evidence_score(self, cid: str, param: str, t: float, value: float) -> dict:
        """Recompute the rule-based evidence for one part with one measurement
        replaced. Everything else about the part is held fixed."""
        e = self.e
        lot = e.lot[cid]
        loc, inv, med, mad, p99 = self._lot_ref(lot, t)
        row = np.array([e.wide.loc[cid, (p, t)] for p in e.params], float)
        j = e.params.index(param)
        row[j] = value
        if not np.isfinite(row).all():
            return {"ok": False}
        z = (row - med) / np.maximum(mad, 1e-12)
        d = row - loc
        d2 = float(d @ inv @ d)
        # drift z at 24 h uses the 0 h value, which the counterfactual may move
        return {"ok": True, "max_abs_z": float(np.max(np.abs(z))),
                "d2": d2, "d2_p99": p99, "z": z}

    def flips(self, cid: str, param: str, t: float, value: float,
              z_thr: float, d2_thr: float) -> bool:
        r = self.evidence_score(cid, param, t, value)
        if not r["ok"]:
            return False
        return (r["max_abs_z"] < z_thr) and (r["d2"] < d2_thr)

    def find(self, cid: str, z_thr: float = 6.0) -> dict | None:
        """Search every (parameter, checkpoint) for the smallest single change
        that clears both the univariate and the joint rule, then VALIDATE it."""
        e = self.e
        lot = e.lot[cid]
        end = max(e.checkpoints)
        best = None
        drivers = []                          # checkpoints where the rules fire
        for t in e.checkpoints:
            try:
                loc, inv, med, mad, p99 = self._lot_ref(lot, t)
            except Exception:
                continue
            base = self.evidence_score(cid, e.params[0], t,
                                       e.wide.loc[cid, (e.params[0], t)])
            if not base["ok"]:
                continue
            if base["max_abs_z"] < z_thr and base["d2"] < p99:
                continue                      # this checkpoint is not the driver
            drivers.append(t)
            for j, p in enumerate(e.params):
                cur = float(e.wide.loc[cid, (p, t)])
                if not np.isfinite(cur):
                    continue
                target = float(med[j])
                lo, hi = (target, cur) if cur > target else (cur, target)
                if not self.flips(cid, p, t, target, z_thr, p99):
                    continue
                for _ in range(40):           # bisect for the SMALLEST change
                    mid = 0.5 * (lo + hi)
                    if self.flips(cid, p, t, mid, z_thr, p99):
                        lo, hi = (mid, hi) if cur > target else (lo, mid)
                    else:
                        lo, hi = (lo, mid) if cur > target else (mid, hi)
                need = lo if cur > target else hi
                delta = abs(cur - need)
                rel = delta / max(abs(mad[j]), 1e-12)
                if best is None or rel < best["delta_sigma"]:
                    # VALIDATE: apply it and confirm the decision actually flips
                    validated = self.flips(cid, p, t, need, z_thr, p99)
                    best = {"parameter": p, "checkpoint_h": t, "current": cur,
                            "required": need, "delta": delta,
                            "delta_sigma": rel, "unit": UNITS[p],
                            "direction": "at or below" if cur > target else "at or above",
                            "validated": bool(validated)}
        # VALIDATE AGAINST THE WHOLE RECORD, not just the edited checkpoint.
        # A single-value change can only clear the rule layer if that
        # checkpoint is the ONLY one on which the rules fire; if two
        # checkpoints drive the flag, editing one leaves the part flagged and
        # the counterfactual would be a false promise. v1.1 re-checked only
        # the edited checkpoint, which made its 100% validity a tautology.
        if best is not None:
            best["validated"] = bool(best["validated"]
                                     and drivers == [best["checkpoint_h"]])
        if best is not None and not best["validated"]:
            return None
        return best

    def text(self, cid: str, tier: str, cf: dict | None) -> str:
        if cf is None:
            return ("No single-parameter change clears the rules; the evidence is "
                    "distributed across several measurements.")
        return (f"The rule-based evidence (6 robust sigma univariate and the "
                f"joint-D2 reference) would no longer flag this part if "
                f"{cf['parameter']} at {cf['checkpoint_h']:.0f} h had been "
                f"{cf['direction']} {cf['required']:.3f} {cf['unit']} "
                f"(measured {cf['current']:.3f} {cf['unit']}, a change of "
                f"{cf['delta']:.3f} {cf['unit']} = {cf['delta_sigma']:.1f} robust sigma).")
