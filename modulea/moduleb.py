"""
Module B: drift prediction. Forecast v168 from early checkpoints, emit an upper
bound, and flag against a safety slope.

Two modes, and they are kept apart by construction rather than by discipline:

    EARLY   0 h and 24 h only. This is what the problem statement asks for and
            it is the headline number.
    MID     0 h, 24 h and 96 h. A secondary capability.

`assert_no_leak` below refuses to build an EARLY feature matrix that contains
any 96 h or 168 h derived column. Blending them is the easiest way to produce a
number that looks excellent and means nothing, so the check is mechanical.

Two guards carried over from earlier sessions, both of which already cost us
once:

  * `r1` (relative drift) is computed ONLY for ratio-scale parameters. For a
    zero-centred quantity like vth_shift_mv it divides by ~0. That bug drove
    max|z| to 107,888 on ordinary good parts and collapsed L2's PR-AUC from
    0.719 to 0.080.
  * PULLED_FAILED is never a feature. P(defective | pulled) = 1.000 in this
    dataset, so it is worth 17.1% recall for free and is pure label leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modulea import features as ft

EARLY_CHECKPOINTS = (0.0, 24.0)
MID_CHECKPOINTS = (0.0, 24.0, 96.0)
TARGET_H = 168.0

BANNED_SUBSTRINGS = ("_t96", "_t168", "v96", "v168", "d2", "d3", "s2", "s3",
                     "censor", "pulled", "defect", "severity", "true_v")


def assert_no_leak(X: pd.DataFrame, mode: str) -> None:
    """Refuse a feature matrix that can see the future (or the label)."""
    bad = []
    for c in X.columns:
        low = c.lower()
        for b in BANNED_SUBSTRINGS:
            if b in low:
                if mode == "mid" and b in ("_t96", "v96", "d2", "s2"):
                    continue
                bad.append((c, b))
                break
    if bad:
        raise AssertionError(
            f"leaking columns in mode={mode}: {bad}. Early-warning mode may see "
            "0 h and 24 h only; nothing may ever see a label-derived field.")


def build_features(ds, mode: str = "early") -> pd.DataFrame:
    """Feature matrix, one row per component.

    Contextual lot statistics (the lot median at each visible checkpoint) are
    included so the model can see whether a part is already diverging from the
    peers it was burned in with, which is the same lot-relative framing Module A
    uses. Lot medians are computed from ALL parts in the lot, which is what a
    detector would actually have; no label is consulted.
    """
    cps = list(EARLY_CHECKPOINTS if mode == "early" else MID_CHECKPOINTS)
    spec = {q["name"]: q for q in ds.cfg["parameters"]}
    m = ds.meas.copy()
    m.loc[m["measurement_status"] != "MEASURED", ds.params] = np.nan
    m = m[m["checkpoint_h"].isin(cps)]
    w = m.pivot_table(index="component_id", columns="checkpoint_h",
                      values=ds.params, dropna=False)
    w = w.reindex(ds.gt["component_id"].to_numpy())
    lot_of = ds.lot_of().reindex(w.index)

    out: dict[str, pd.Series] = {}
    for p in ds.params:
        v0 = w[(p, 0.0)]
        v24 = w[(p, 24.0)]
        d1 = v24 - v0
        out[f"{p}__v0"] = v0
        out[f"{p}__v24"] = v24
        out[f"{p}__d1"] = d1
        out[f"{p}__s1"] = d1 / 24.0
        if ft.is_ratio_scale(spec[p]):
            eps = 0.05 * abs(spec[p]["v0_mean"])
            out[f"{p}__r1"] = d1 / (v0.abs() + eps)
        for t in cps:
            med = w[(p, t)].groupby(lot_of).transform("median")
            out[f"{p}__lotmed_t{int(t)}"] = med
        if mode == "mid":
            v96 = w[(p, 96.0)]
            out[f"{p}__v96"] = v96
            out[f"{p}__d2"] = v96 - v24
            out[f"{p}__s2"] = (v96 - v24) / 72.0

    X = pd.DataFrame(out, index=w.index)
    # lot-relative robust z of v0 and d1, the Module A framing
    zcols = [c for c in X.columns if c.endswith("__v0") or c.endswith("__d1")]
    g = X[zcols].groupby(lot_of)
    med = g.transform("median")
    mad = g.transform(lambda a: np.nanmedian(np.abs(a - np.nanmedian(a))))
    z = (X[zcols] - med) / (1.4826 * mad).replace(0, np.nan)
    z.columns = [f"z__{c}" for c in zcols]
    X = pd.concat([X, z], axis=1)
    assert_no_leak(X, mode)
    return X


def targets(ds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed v168 (NaN where censored) and the LATENT v168 from ground truth.

    The latent column exists only so the survivorship bias can be quantified in
    absolute units. It is never a feature and never a training target for the
    deployable models.
    """
    m = ds.meas[ds.meas["checkpoint_h"] == TARGET_H]
    obs = m.pivot_table(index="component_id", values=ds.params, aggfunc="first")
    obs = obs.reindex(ds.gt["component_id"].to_numpy())
    st = m.set_index("component_id")["measurement_status"].reindex(obs.index)
    obs[st != "MEASURED"] = np.nan
    gt = ds.gt.set_index("component_id")
    lat = pd.DataFrame({p: gt[f"true_v168_{p}"] for p in ds.params}).reindex(obs.index)
    true = obs.copy()
    for p in ds.params:                      # truth = observed, else latent
        true[p] = obs[p].where(obs[p].notna(), lat[p])
    return obs, true


# --------------------------------------------------------------------------
# safety slopes
# --------------------------------------------------------------------------

def slope_margin_consumption(ds, X: pd.DataFrame, p: str) -> pd.Series:
    """(a) s_safe = (L - v24) / (168 - 24). Simple, explainable, naive alone."""
    hi = ds.limits[p][1]
    return (hi - X[f"{p}__v24"]) / (TARGET_H - 24.0)


def slope_lot_derived(ds, X: pd.DataFrame, p: str, healthy: pd.Series,
                      n_sigma: float = 6.0) -> pd.Series:
    """(b) median(healthy slopes) + 6 * MAD-sigma(healthy slopes), per lot.

    DPAT applied to the drift-rate distribution rather than the level, so it is
    the same statistical argument Module A makes, one derivative up.
    """
    s1 = X[f"{p}__s1"]
    lot = ds.lot_of().reindex(X.index)
    ref = s1.where(healthy)
    med = ref.groupby(lot).transform("median")
    mad = ref.groupby(lot).transform(lambda a: np.nanmedian(np.abs(a - np.nanmedian(a))))
    return med + n_sigma * 1.4826 * mad


def project_to_mission(ds, X: pd.DataFrame, p: str, beta: float,
                       mission_years: float = 15.0) -> pd.Series:
    """(c) Mission-based rule, done as a PROJECTION rather than a slope compare.

    The first version of this compared the burn-in 0->24 h slope against a
    mission-average allowed rate and flagged 46-74% of all parts. That was
    wrong, and wrong for the reason rung 1 already established: degradation is
    sub-linear, so the early slope systematically over-states the long-run rate.
    Comparing an early slope to a long-run budget is apples to oranges.

    The correct form projects with the same power law the physics implies:

        t_mission_equiv = mission_years * 8766 / AF      (stress-hours)
        v_mission       = v0 + lambda * t_equiv^beta,  lambda from d1

    and compares the projected end-of-mission VALUE against the limit. This is
    the definition that actually connects the decision to Part 1.2 of the
    research doc.
    """
    af = float(ds.cfg["derived"]["acceleration_factor"])
    t_eq = mission_years * 8766.0 / af
    lam = X[f"{p}__d1"] / (24.0 ** beta)
    return X[f"{p}__v0"] + lam * (t_eq ** beta)


def lot_safe_limit(ds, p: str, n_sigma: float = 6.0) -> pd.Series:
    """Lot-derived L_safe at 168 h: the AEC-Q001 dynamic PAT upper limit.

    Using the DATASHEET limit as L_safe makes rule (d) inert on this dataset:
    every injected defect is in spec at every checkpoint by construction, so a
    predicted bound essentially never crosses it (measured flag rate 0.00-0.03%).
    That is not a modelling failure, it is the dataset's central premise showing
    through -- and it says the useful safe limit for Module B is the same
    lot-relative limit Module A uses, not the engineering limit.

    Clipped to the datasheet limit, since a PAT limit never loosens one.
    """
    m = ds.meas[(ds.meas["measurement_status"] == "MEASURED")
                & (ds.meas["checkpoint_h"] == TARGET_H)]
    g = m.groupby("lot_id")[p]
    med = g.median()
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    lim = med + n_sigma * (q3 - q1) / 1.35
    lim = np.minimum(lim, ds.limits[p][1])
    return ds.lot_of().map(lim)


def flag_confidence_adjusted(upper: pd.Series, ds, p: str,
                             margin_frac: float = 0.0,
                             l_safe: pd.Series | None = None) -> pd.Series:
    """(d) Flag if the 95% UPPER BOUND on v168 crosses the limit.

    The decision uses the bound, not the point estimate: for a safety call the
    plausible worst case is the relevant quantity, not the expectation.
    """
    if l_safe is not None:
        return upper > l_safe.reindex(upper.index)
    lo, hi = ds.limits[p]          # limits are (lo, hi); do not swap these
    return upper > hi - (hi - lo) * margin_frac
