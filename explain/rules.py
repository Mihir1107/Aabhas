"""
Layer 1: the deterministic rule text.

Highest value per unit of effort in the whole system. Costs nothing at
inference, is always available, cannot fail, and is more convincing to an
inspector than any attribution method -- because every number carries the
comparison it was made against.

The rule for every line: a bare "19.5 sigma" is not auditable. "19.5 robust
sigma above the lot median (median 10.1 uA, sigma 1.8 uA, n=142, MAD
estimator)" is: the inspector can recompute it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modulea import features as ftmod
from explain import mechanisms as mech
from explain.decision import ACTIONS

UNITS = {"iddq_ua": "uA", "leakage_na": "nA", "prop_delay_ns": "ns",
         "supply_current_ma": "mA", "vth_shift_mv": "mV"}

# Which evidence line a primary reason is drawn from. The primary reason must
# be whichever detector actually drove the decision, never a fixed string.
REASON_TEXT = {
    "ABSOLUTE_LIMIT": "datasheet limit violation",
    "LEVEL": "elevated level relative to lot peers",
    "DRIFT_RATE": "abnormal drift rate relative to lot peers",
    "STEP": "discrete step between checkpoints",
    "JOINT": "joint-distribution anomaly across parameters (univariately normal)",
    "FORECAST": "forecast 168 h upper bound exceeds the lot-derived safe limit",
    "SPATIAL": "anomaly clusters by board position, not by component",
    "DATA_QUALITY": "measurement record incomplete for a non-part reason",
    "NONE": "no rule fired",
}


def _fmt(v, u, nd=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f} {u}"


class PartExplanation:
    """All layers for one component, assembled once and rendered many ways."""

    def __init__(self, e, policy, cid: str, flagged_pool: pd.Series):
        self.e = e
        self.cid = cid
        self.policy = policy
        self.dec = policy.decide(cid, flagged_pool)
        self.tier = self.dec["tier"]
        self.lot = e.lot[cid]
        self.meta = e.ds.meas[e.ds.meas["component_id"] == cid].iloc[0]
        self.dq = e.data_quality(cid)
        self.stats0 = e.lot_stats(self.lot, 0.0)
        self.statsE = e.lot_stats(self.lot, max(e.checkpoints))
        self.d2, self.contrib, self.d2_p99 = e.maha_contributions(cid)
        self.spatial = e.spatial_check(cid, flagged_pool)
        self.lines, self.signals = self._evidence()
        self.primary, self.secondary = self._reasons()
        self.hypotheses = mech.hypothesise(self.signals)

    # ---------------- evidence lines ----------------
    def _evidence(self):
        e, cid = self.e, self.cid
        L, S = [], {}
        end = max(e.checkpoints)

        # absolute limit
        worst_p, worst_frac = None, -np.inf
        for p in e.params:
            v = e.wide.loc[cid, (p, end)]
            lo, hi = e.limits[p]
            if np.isfinite(v):
                frac = (v - lo) / (hi - lo)
                if frac > worst_frac:
                    worst_frac, worst_p = frac, p
        if worst_p:
            v = e.wide.loc[cid, (worst_p, end)]
            lo, hi = e.limits[worst_p]
            ok = lo <= v <= hi
            L.append(("Absolute limit",
                      f"{'PASS' if ok else 'FAIL'} ({v:.2f} {UNITS[worst_p]} vs "
                      f"{hi:.1f} {UNITS[worst_p]} max, {worst_p})"))
            S["abs_violation"] = not ok

        # lot-relative level, worst parameter at the last measured checkpoint
        best_p, best_z = None, 0.0
        for p in e.params:
            v = e.wide.loc[cid, (p, end)]
            st = self.statsE.loc[p]
            if np.isfinite(v) and np.isfinite(st["sigma_mad"]) and st["sigma_mad"] > 0:
                z = (v - st["median"]) / st["sigma_mad"]
                if abs(z) > abs(best_z):
                    best_z, best_p = z, p
        if best_p:
            st = self.statsE.loc[best_p]
            u = UNITS[best_p]
            L.append(("Lot-relative level",
                      f"{best_z:+.1f} robust sigma vs lot median on {best_p} "
                      f"(median {st['median']:.2f} {u}, sigma {st['sigma_mad']:.2f} {u}, "
                      f"n={int(st['n'])}, MAD estimator; AEC-Q001 IQR/1.35 gives "
                      f"{st['sigma_iqr']:.2f} {u})"))
            S["level_sigma"] = float(best_z)
            S["level_param"] = best_p

        # drift 0 -> 24 h, against the lot's own median drift
        hd = e.healthy_lot_drift(self.lot)
        dp, dz = None, 0.0
        for p in e.params:
            zc = f"z__{p}__d1"
            if zc in e.ztraj.columns:
                z = e.ztraj.loc[cid, zc]
                if np.isfinite(z) and abs(z) > abs(dz):
                    dz, dp = float(z), p
        if dp:
            v0 = e.wide.loc[cid, (dp, 0.0)]
            v24 = e.wide.loc[cid, (dp, 24.0)]
            u = UNITS[dp]
            # A percentage change is only meaningful for a RATIO-SCALE quantity.
            # vth_shift_mv is centred on zero, so "-901.9%" is arithmetic noise,
            # not information. This is the third place this same bug has
            # appeared (L2 features, Module B r1, and here); the guard lives in
            # modulea.features so there is one definition of it.
            spec = {q["name"]: q for q in e.ds.cfg["parameters"]}[dp]
            ratio_ok = ftmod.is_ratio_scale(spec)
            pct_s = (f" ({100 * (v24 - v0) / abs(v0):+.1f}%)"
                     if ratio_ok and np.isfinite(v0) and v0 != 0 else "")
            lot_s = (f" ({100 * hd[dp] / abs(self.stats0.loc[dp, 'median']):+.1f}%)"
                     if ratio_ok else "")
            L.append(("Drift 0h to 24h",
                      f"{v24 - v0:+.3f} {u}{pct_s} on {dp}; lot median drift "
                      f"{hd[dp]:+.3f} {u}{lot_s}, part is {dz:+.1f} robust "
                      "sigma on the lot's drift distribution"))
            S["drift_sigma"] = dz
            S["drift_param"] = dp

        # drift rate and multiple of lot median
        if dp:
            s1 = (e.wide.loc[cid, (dp, 24.0)] - e.wide.loc[cid, (dp, 0.0)]) / 24.0
            lot_s1 = hd[dp] / 24.0
            mult = s1 / lot_s1 if lot_s1 not in (0, np.nan) else np.nan
            L.append(("Drift rate",
                      f"{s1:.4f} {UNITS[dp]}/h (lot median {lot_s1:.4f} "
                      f"{UNITS[dp]}/h, {mult:.1f}x)"))
            S["drift_multiple"] = float(mult) if np.isfinite(mult) else np.nan

        # step / max interval jump
        jp, jz = None, 0.0
        for p in e.params:
            zc = f"z__{p}__max_interval_jump"
            if zc in e.ztraj.columns:
                z = e.ztraj.loc[cid, zc]
                if np.isfinite(z) and z > jz:
                    jz, jp = float(z), p
        if jp:
            # A STEP is one interval moving while the others stay normal. A
            # STEEP DRIFTER moves in every interval. Both produce a large
            # max-interval-jump, so the jump magnitude alone cannot tell them
            # apart -- and calling a drifter a "step" is a wrong primary reason
            # even though the part is correctly flagged. The discriminator is
            # the CONCENTRATION of the movement: ratio of the largest interval
            # change to the median of the others.
            d = np.array([abs(e.wide.loc[cid, (jp, e.checkpoints[i + 1])]
                              - e.wide.loc[cid, (jp, e.checkpoints[i])])
                          for i in range(len(e.checkpoints) - 1)], dtype=float)
            fin = d[np.isfinite(d)]
            conc = (float(np.max(fin) / max(np.median(np.sort(fin)[:-1]), 1e-12))
                    if len(fin) >= 3 else np.nan)
            S["jump_concentration"] = conc
            L.append(("Max interval jump",
                      f"{jz:+.1f} robust sigma on {jp}; largest interval change is "
                      f"{conc:.1f}x the median of the others "
                      f"({'concentrated -> step' if np.isfinite(conc) and conc > 3.0 else 'spread -> progressive drift'})"))
            S["max_jump_z"] = jz
            S["step_detected"] = bool(jz > 6.0 and np.isfinite(conc) and conc > 3.0)

        # curvature: is the drift accelerating
        cz = 0.0
        for p in ("leakage_na", "iddq_ua"):
            zc = f"z__{p}__curvature"
            if zc in e.ztraj.columns:
                z = e.ztraj.loc[cid, zc]
                if np.isfinite(z) and abs(z) > abs(cz):
                    cz = float(z)
        S["curvature_z"] = cz
        S["accelerating_leakage"] = cz > 3.0

        # Module B forecast
        for p in e.params:
            if cid in e.mb_point.index and np.isfinite(e.mb_point.loc[cid, p]):
                pt = float(e.mb_point.loc[cid, p])
                ub = float(e.mb_upper.loc[cid, p])
                ls = float(e.l_safe[p].get(cid, np.nan))
                if p == (dp or e.params[0]):
                    u = UNITS[p]
                    L.append(("Predicted 168h", f"{pt:.3f} {u} ({p}, from 0h and 24h only)"))
                    L.append(("95% upper bound",
                              f"{ub:.3f} {u} -> {'EXCEEDS' if ub > ls else 'within'} "
                              f"lot-derived safe limit {ls:.3f} {u}"))
                    S["forecast_breach"] = bool(ub > ls)
                    break

        # joint anomaly
        if np.isfinite(self.d2):
            top = self.contrib.abs().sort_values(ascending=False)
            share = 100 * top.iloc[0] / max(self.contrib.abs().sum(), 1e-9)
            L.append(("Joint anomaly",
                      f"Mahalanobis D2 = {self.d2:.1f} (good-part 99th pct = "
                      f"{self.d2_p99:.1f}); largest contribution {top.index[0]} "
                      f"at {share:.0f}% of the total"))
            S["maha_d2"] = float(self.d2)
            S["maha_p99"] = float(self.d2_p99)
            S["joint_top_param"] = top.index[0]
            # "Only visible jointly" has to mean the univariate AND trajectory
            # views both look normal. A steep drifter also has a huge D^2, but
            # that is a CONSEQUENCE of its drift, not independent evidence, and
            # reporting JOINT as its primary reason would be a wrong
            # explanation for a correctly flagged part.
            S["joint_only"] = bool(self.d2 > self.d2_p99
                                   and abs(S.get("level_sigma", 0)) < 4.0
                                   and abs(S.get("drift_sigma", 0)) < 4.0
                                   and not S.get("step_detected", False))

        # correlation sign: leaky AND slow is off the process locus
        zl = e.zlvl.get(f"z__leakage_na__t{int(end)}")
        zd = e.zlvl.get(f"z__prop_delay_ns__t{int(end)}")
        if zl is not None and zd is not None:
            a, b = float(zl.get(cid, np.nan)), float(zd.get(cid, np.nan))
            if np.isfinite(a) and np.isfinite(b):
                inverted = a > 1.0 and b > 1.0
                S["correlation_inverted"] = bool(inverted)
                L.append(("Leakage/delay sign",
                          f"leakage {a:+.1f} sigma, delay {b:+.1f} sigma -- "
                          + ("INVERTED: leaky and slow, off the process locus"
                             if inverted else
                             "consistent with the lot's process locus")))

        # delay drift with stable current
        zdd = e.ztraj.get("z__prop_delay_ns__total_drift")
        zid = e.ztraj.get("z__iddq_ua__total_drift")
        if zdd is not None and zid is not None:
            a, b = float(zdd.get(cid, np.nan)), float(zid.get(cid, np.nan))
            S["delay_drift_stable_current"] = bool(
                np.isfinite(a) and np.isfinite(b) and a > 4.0 and abs(b) < 2.0)

        # benign corner / static level outlier
        S["benign_corner"] = bool(S.get("level_sigma", 0) > 2.0
                                  and abs(S.get("drift_sigma", 0)) < 2.0
                                  and not S.get("correlation_inverted", False)
                                  and np.isfinite(self.d2) and self.d2 < self.d2_p99)
        S["static_level_outlier"] = bool(S.get("level_sigma", 0) > 6.0
                                         and abs(S.get("drift_sigma", 0)) < 3.0)

        # spatial
        sp = self.spatial
        L.append(("Spatial check",
                  (f"{sp['n_flagged_on_board']} of {sp['n_on_board']} components "
                   f"on board {sp['board_id']} flagged (expected "
                   f"{sp['expected_flagged']:.1f}, p={sp['p_value']:.1e}) -- "
                   "CLUSTERED, suspect fixture")
                  if sp["clustered"] else
                  f"no neighbour clustering on board {sp['board_id']} "
                  f"({sp['n_flagged_on_board']}/{sp['n_on_board']} flagged), "
                  "component-specific"))
        S["spatial_clustered"] = sp["clustered"]
        S["n_flagged_on_board"] = sp["n_flagged_on_board"]
        S["board_id"] = sp["board_id"]

        # data quality
        L.append(("Data quality",
                  "all checkpoints valid"
                  if self.dq["valid"] and not self.dq["pulled_failed"] else
                  ("component pulled after hard failure; no 168 h reading"
                   if self.dq["pulled_failed"] else
                   f"{self.dq['n_missing_equipment']} checkpoint(s) lost to "
                   "equipment/chamber fault")))
        return L, S

    # ---------------- primary / secondary reason ----------------
    def _reasons(self):
        """The primary reason is whichever rule actually drove the decision.

        Order matters and follows the decision policy, not a template: a
        datasheet violation outranks everything, spatial clustering outranks a
        component verdict, and a joint-only anomaly is reported as JOINT even
        when a level rule also fired weakly, because JOINT is what a univariate
        rule would have missed.
        """
        S = self.signals
        cand = []
        if S.get("abs_violation"):
            cand.append(("ABSOLUTE_LIMIT", 100.0))
        if S.get("spatial_clustered"):
            cand.append(("SPATIAL", 90.0))
        if not self.dq["valid"] and not self.dq["pulled_failed"]:
            cand.append(("DATA_QUALITY", 95.0))
        if S.get("step_detected"):
            cand.append(("STEP", 10.0 + S.get("max_jump_z", 0)))
        if S.get("joint_only"):
            cand.append(("JOINT", 8.0 + min(S.get("maha_d2", 0)
                                            / max(S.get("maha_p99", 1), 1), 3.0)))
        elif S.get("correlation_inverted"):
            cand.append(("JOINT", 7.5))
        if abs(S.get("drift_sigma", 0)) > 4.0 or S.get("drift_multiple", 0) > 3.0:
            cand.append(("DRIFT_RATE", 6.0 + abs(S.get("drift_sigma", 0)) / 3.0))
        if abs(S.get("level_sigma", 0)) > 4.0:
            cand.append(("LEVEL", 5.0 + abs(S.get("level_sigma", 0)) / 4.0))
        if S.get("forecast_breach"):
            cand.append(("FORECAST", 4.0))
        if not cand:
            return "NONE", None
        cand.sort(key=lambda t: -t[1])
        return cand[0][0], (cand[1][0] if len(cand) > 1 else None)

    # ---------------- rendering ----------------
    def rule_text(self) -> str:
        w = max(len(k) for k, _ in self.lines) + 2
        out = [f"Component: {self.cid}     Lot: {self.lot}     "
               f"Board: {self.meta['board_id']}  Socket: "
               f"(r{int(self.meta['socket_row'])}, c{int(self.meta['socket_col'])})",
               f"Decision: {self.tier}", ""]
        for k, v in self.lines:
            out.append(f"{k + ':':<{w}} {v}")
        out.append("")
        out.append(f"{'Primary reason:':<{w}} {REASON_TEXT[self.primary]}")
        if self.secondary:
            out.append(f"{'Secondary:':<{w}} {REASON_TEXT[self.secondary]}")
        if self.hypotheses:
            h = self.hypotheses[0]
            out.append(f"{'Mechanism (HYPOTHESIS):':<{w}} {h['hypothesis']} "
                       f"[{h['confidence']}]")
        out.append(f"{'Recommended action:':<{w}} {ACTIONS[self.tier]}")
        return "\n".join(out)
