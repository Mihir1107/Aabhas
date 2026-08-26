"""
L2 trajectory features, and the lot-relative robust z-scores of them.

Computing drift features correctly is fussier than it looks. Three traps were
found while validating the generator's beta estimator, and every one of them
applies to any drift statistic computed here:

  T1  Difference PER PART, then aggregate. Aggregating first and differencing
      the aggregates -- median(x_t) - median(x_0) -- is biased, because v0 and
      lambda are separate random effects and the median of a sum is not the sum
      of medians.
  T2  Fit or aggregate WITHIN a lot. beta carries a per-lot jitter, so pooling
      lots means fitting a mixture of power laws, which is not itself a power
      law and biases the exponent low by ~0.02.
  T3  Aggregate with the MEAN, not the median, when estimating a drift
      magnitude from clean parts. lambda is log-normal so drift is
      right-skewed; symmetric measurement noise pulls a median down harder at
      24 h (where drift is comparable to the noise) than at 168 h, which
      flattens a log-log slope.

`fit_beta_within_lot` below is the corrected estimator, kept here so anything
downstream reuses it rather than reimplementing the bug.

Note on the z-scores: those deliberately use median/MAD, not the mean, because
their job is robustness against the contamination we are hunting (T3 does not
apply -- it concerns unbiased estimation from clean parts, not robust
standardisation). Within a lot this is a constant offset and cannot reorder
parts; across lots it introduces a small inconsistency wherever a lot's
signal-to-noise differs, which is noted in the report.

Censoring is never imputed here. A part pulled at 96 h has no d3, s3,
total_drift or relative_drift, and those stay NaN with an explicit `censored`
flag alongside. What is computable from the surviving intervals is computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9

BASE_FEATURES = ["d1", "d2", "d3", "s1", "s2", "s3", "r1", "curvature",
                 "total_drift", "relative_drift", "max_interval_jump",
                 "monotonic", "sign_changes"]


def fit_beta_within_lot(wide: pd.DataFrame, param: str, lot_of: pd.Series,
                        checkpoints: list[float]) -> pd.Series:
    """Saturating exponent beta per lot, with T1/T2/T3 all handled.

    drift_i(t) = x_i(t) - x_i(t0)      per part           (T1)
    mean over parts within one lot                        (T3, within T2)
    log(mean drift) = log(lambda) + beta * log(t)         fitted per lot
    """
    ts = [t for t in checkpoints if t > 0]
    t0 = min(checkpoints)
    x = np.log(np.array(ts))
    d = pd.DataFrame({t: wide[(param, t)] - wide[(param, t0)] for t in ts})
    out = {}
    for lot, idx in lot_of.groupby(lot_of).groups.items():
        med = d.reindex(idx).mean(axis=0).to_numpy()
        if not np.isfinite(med).all() or (med <= 0).any():
            out[lot] = np.nan
            continue
        out[lot] = float(np.polyfit(x, np.log(med), 1)[0])
    return pd.Series(out, name=f"beta_{param}")


def wide_frame(ds) -> pd.DataFrame:
    """component_id x (param, checkpoint), NaN where not measured."""
    m = ds.meas.copy()
    m.loc[m["measurement_status"] != "MEASURED", ds.params] = np.nan
    w = m.pivot_table(index="component_id", columns="checkpoint_h",
                      values=ds.params, dropna=False)
    return w.reindex(ds.gt["component_id"].to_numpy())


def is_ratio_scale(spec: dict) -> bool:
    """Whether a relative (percentage) drift is meaningful for this parameter.

    Only for a ratio-scale quantity with a true zero and a non-zero typical
    value. `vth_shift_mv` is centred on zero and takes both signs -- it is
    interval-scale -- so "percent change in vth_shift" is not a physical
    quantity, and computing it divides by ~0. That is what produced
    max|z| = 107,888 on ordinary good parts and destroyed PR-AUC for L2.
    """
    return spec["direction"] == "higher_is_worse" and abs(spec["v0_mean"]) > 1e-9


def trajectory_features(ds, wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw (un-standardised) trajectory features, one row per component."""
    w = wide_frame(ds) if wide is None else wide
    t = ds.checkpoints
    spec = {q["name"]: q for q in ds.cfg["parameters"]}
    out = {}
    for p in ds.params:
        v = [w[(p, tt)] for tt in t]
        d1, d2, d3 = v[1] - v[0], v[2] - v[1], v[3] - v[2]
        dt = [t[1] - t[0], t[2] - t[1], t[3] - t[2]]
        s1, s2, s3 = d1 / dt[0], d2 / dt[1], d3 / dt[2]
        D = pd.concat([d1, d2, d3], axis=1)
        out[f"{p}__d1"], out[f"{p}__d2"], out[f"{p}__d3"] = d1, d2, d3
        out[f"{p}__s1"], out[f"{p}__s2"], out[f"{p}__s3"] = s1, s2, s3
        out[f"{p}__curvature"] = s2 - s1
        out[f"{p}__total_drift"] = v[3] - v[0]
        if is_ratio_scale(spec[p]):
            # epsilon tied to the parameter's own scale, not a bare 1e-9. A
            # floor of 1e-9 is no protection for a quantity whose natural
            # magnitude is single-digit millivolts.
            eps = 0.05 * abs(spec[p]["v0_mean"])
            out[f"{p}__r1"] = d1 / (v[0].abs() + eps)
            out[f"{p}__relative_drift"] = (v[3] - v[0]) / (v[0].abs() + eps)
        # computable from whatever intervals survive censoring
        out[f"{p}__max_interval_jump"] = D.abs().max(axis=1, skipna=True)
        sg = np.sign(D)
        n_obs = D.notna().sum(axis=1)
        pos = (sg > 0).sum(axis=1)
        neg = (sg < 0).sum(axis=1)
        out[f"{p}__monotonic"] = ((pos == n_obs) | (neg == n_obs)).astype(float)
        sgn = sg.to_numpy()
        chg = np.zeros(len(D))
        for i in range(sgn.shape[1] - 1):
            a, b = sgn[:, i], sgn[:, i + 1]
            chg += ((a * b) < 0).astype(float)
        out[f"{p}__sign_changes"] = pd.Series(chg, index=D.index)
    f = pd.DataFrame(out, index=w.index)

    gt = ds.gt.set_index("component_id")
    f["censored_pulled"] = (gt["censor_status_168h"] == "PULLED_FAILED") \
        .reindex(f.index).fillna(False).astype(float)
    status = ds.meas.pivot_table(index="component_id", columns="checkpoint_h",
                                 values="measurement_status", aggfunc="first")
    f["n_missing_checkpoints"] = (status != "MEASURED").sum(axis=1).reindex(f.index) \
        .fillna(0).astype(float)
    return f


def lot_relative_z(f: pd.DataFrame, lot_of: pd.Series,
                   cols: list[str] | None = None,
                   report: dict | None = None) -> pd.DataFrame:
    """Lot-relative robust z-score, median and MAD, computed within each lot.

    This is DPAT applied to drift rather than level. A part is compared to the
    peers it was burned in with, so a lot-wide process shift (Type VI) cancels
    out instead of flagging 500 parts.
    """
    cols = cols or [c for c in f.columns
                    if c not in ("censored_pulled", "n_missing_checkpoints")]
    lot = lot_of.reindex(f.index)
    g = f[cols].groupby(lot)
    med = g.transform("median")
    mad = g.transform(lambda a: np.nanmedian(np.abs(a - np.nanmedian(a))))
    sig = 1.4826 * mad
    # A near-constant feature has MAD = 0 within almost every lot, so its
    # z-score is 0/0. `monotonic` and `sign_changes` are exactly that: almost
    # every good part is monotonic, so the lot MAD is zero and standardising
    # destroys the feature instead of scaling it. Those stay RAW -- they are
    # already lot-independent and directly interpretable -- and only the
    # continuous features are z-scored. Silently emitting NaN here would have
    # deleted the monotonicity signal that Type II is supposed to be caught by.
    degenerate = [c for c in cols
                  if float((sig[c] <= 0).mean()) > 0.5 or not np.isfinite(
                      sig[c].to_numpy()).any()]
    keep = [c for c in cols if c not in degenerate]
    if report is not None:
        report["degenerate_not_zscored"] = degenerate
    z = (f[keep] - med[keep]) / sig[keep].replace(0, np.nan)
    z.columns = [f"z__{c}" for c in keep]
    if degenerate:
        z = pd.concat([z, f[degenerate].add_prefix("raw__")], axis=1)
    return z


def build_feature_matrix(ds):
    """Returns (raw features, lot-relative z features, level z features).

    LEVEL block: the five parameters at each checkpoint, lot-relative
    z-scored. TRAJ block: the trajectory features above, lot-relative
    z-scored. Both carry NaN where censoring makes a value undefined.
    """
    w = wide_frame(ds)
    lot_of = ds.lot_of()
    traj = trajectory_features(ds, w)
    rep: dict = {}
    ztraj = lot_relative_z(traj, lot_of, report=rep)
    ztraj.attrs["degenerate_not_zscored"] = rep.get("degenerate_not_zscored", [])

    lvl = pd.DataFrame(
        {f"{p}__t{int(t)}": w[(p, t)] for p in ds.params for t in ds.checkpoints},
        index=w.index)
    zlvl = lot_relative_z(lvl, lot_of)
    return traj, ztraj, lvl, zlvl
