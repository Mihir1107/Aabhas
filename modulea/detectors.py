"""
Module A detector ladder, L0 through L4.

Every detector exposes the same two things:
    flags(...)  -> bool Series indexed by component_id (an operating point)
    score(...)  -> float Series indexed by component_id, higher = more anomalous

Detectors that are inherently binary (L0, DPAT at one fixed multiplier) also
expose a continuous score wherever a natural one exists, so that threshold-free
metrics and the recall-versus-yield-loss curve are computable. Where no natural
score exists that is stated rather than faked.

DPAT numbers are ALWAYS labelled single-parameter or union-across-parameters.
The two differ by roughly 30 points on Type Vb and conflating them is the
easiest way to publish a number that will not reproduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IQR_TO_SIGMA = 1.35        # AEC-Q001 Rev-D
MAD_TO_SIGMA = 1.4826
P99_FACTOR = 0.429858      # 1 / 2.326348
MIN_LOT_N = 30             # AEC-Q001 minimum for limit estimation
MIN_IQR_N = 20             # below this the 1.35 divisor is unreliable


# --------------------------------------------------------------------------
# robust scale estimators
# --------------------------------------------------------------------------

def sigma_mad(a: np.ndarray) -> float:
    med = np.median(a)
    return float(MAD_TO_SIGMA * np.median(np.abs(a - med)))


def sigma_iqr(a: np.ndarray) -> float:
    q1, q3 = np.percentile(a, [25, 75])
    return float((q3 - q1) / IQR_TO_SIGMA)


def sigma_classical(a: np.ndarray) -> float:
    return float(np.std(a, ddof=1))


def centre_median(a: np.ndarray) -> float:
    return float(np.median(a))


def centre_mean(a: np.ndarray) -> float:
    return float(np.mean(a))


ESTIMATORS = {
    # name: (centre_fn, sigma_fn, asymmetric?)
    "MAD": (centre_median, sigma_mad, False),
    "IQR": (centre_median, sigma_iqr, False),
    "p1p99": (centre_median, None, True),
    "classical": (centre_mean, sigma_classical, False),
}


def limits_for(a: np.ndarray, estimator: str, k: float) -> tuple[float, float]:
    """PAT limits at multiplier k, before datasheet clipping.

    The p1/p99 variant is asymmetric by construction, which is the point of it:
    AEC-Q001 permits derived methods for non-normal distributions, and Iddq and
    leakage here are log-normal. It is also the estimator with the ~1%
    breakdown point, computed FROM the tail where the defects live.
    """
    med = float(np.median(a))
    if estimator == "p1p99":
        p1, p99 = np.percentile(a, [1, 99])
        return (med - k * (med - p1) * P99_FACTOR,
                med + k * (p99 - med) * P99_FACTOR)
    centre_fn, sigma_fn, _ = ESTIMATORS[estimator]
    c, s = centre_fn(a), sigma_fn(a)
    return c - k * s, c + k * s


def robust_z(a: np.ndarray, estimator: str) -> np.ndarray:
    """Signed deviation in estimator sigmas. Thresholding |z| at k is exactly
    the PAT rule at multiplier k, so a k-sweep is a threshold sweep on this."""
    med = float(np.median(a))
    if estimator == "p1p99":
        p1, p99 = np.percentile(a, [1, 99])
        up = max((p99 - med) * P99_FACTOR, 1e-12)
        dn = max((med - p1) * P99_FACTOR, 1e-12)
        return np.where(a >= med, (a - med) / up, (a - med) / dn)
    centre_fn, sigma_fn, _ = ESTIMATORS[estimator]
    c, s = centre_fn(a), sigma_fn(a)
    return (a - c) / max(s, 1e-12)


# --------------------------------------------------------------------------
# L0: static datasheet limits
# --------------------------------------------------------------------------

def l0_static_limits(ds, use_latent_for_pulled: bool = False) -> pd.DataFrame:
    """Flag if any parameter violates its absolute datasheet limit at any
    OBSERVED checkpoint.

    By construction this should catch approximately zero genuine defects: the
    generator guarantees every injected part is inside spec at every measured
    checkpoint. A non-zero rate means the in-spec guarantee has a hole or a
    checkpoint is being read wrong, so it is reported either way.

    Parts pulled at 96 h are the documented exception. Their latent 168 h value
    IS out of spec -- that is why they were pulled -- but it lives in ground
    truth and is never visible to a detector. `use_latent_for_pulled` exposes
    that counterfactual for reporting only; the deployable L0 never sees it.
    """
    m = ds.meas[ds.meas["measurement_status"] == "MEASURED"]
    viol = pd.Series(False, index=m.index)
    excess = pd.Series(0.0, index=m.index)
    for p, (lo, hi) in ds.limits.items():
        rng = hi - lo
        over = (m[p] - hi) / rng
        under = (lo - m[p]) / rng
        viol |= (over > 0) | (under > 0)
        excess = np.maximum(excess, np.maximum(over, under))
    flag = pd.Series(viol.to_numpy(), index=m["component_id"].to_numpy()) \
        .groupby(level=0).any()
    score = pd.Series(excess.to_numpy(), index=m["component_id"].to_numpy()) \
        .groupby(level=0).max()
    idx = ds.gt["component_id"]
    out = pd.DataFrame({"flag": flag.reindex(idx).fillna(False),
                        "score": score.reindex(idx).fillna(0.0)})

    if use_latent_for_pulled:
        gt = ds.gt.set_index("component_id")
        pulled = gt.index[gt["censor_status_168h"] == "PULLED_FAILED"]
        for p, (lo, hi) in ds.limits.items():
            v = gt.loc[pulled, f"true_v168_{p}"]
            bad = ((v < lo) | (v > hi)).reindex(out.index).fillna(False)
            out.loc[bad, "flag"] = True
    return out


# --------------------------------------------------------------------------
# L1: Part Average Testing
# --------------------------------------------------------------------------

def _lot_reference(sub: pd.DataFrame, p: str, min_n: int):
    """Return the per-lot arrays used to build limits, falling back to a pooled
    peer group when a lot is too small.

    AEC-Q001 requires at least 30 parts for limit estimation and the 1.35
    divisor is unreliable below 20. Both are enforced: an undersized lot uses
    the pooled reference instead of its own statistics.
    """
    pooled = sub[p].dropna().to_numpy()
    out = {}
    for lot, g in sub.groupby("lot_id"):
        a = g[p].dropna().to_numpy()
        out[lot] = a if len(a) >= min_n else pooled
    return out, pooled


def dpat(ds, estimator: str = "MAD", mode: str = "dynamic", k: float = 6.0,
         params: list[str] | None = None, checkpoints: list[float] | None = None,
         reference_lots: list[str] | None = None,
         clip_to_datasheet: bool = True) -> pd.DataFrame:
    """AEC-Q001 Part Average Testing.

    mode="dynamic": limits recomputed per lot, per checkpoint, per parameter.
    mode="static" : limits computed once from `reference_lots` pooled and held
                    fixed, which is what a historical PAT baseline does.

    Returns flag (union across the requested parameters and checkpoints) and
    score = max |robust z| over the same set. Pass a single-element `params`
    list for a single-parameter number.

    PAT limits never loosen an engineering limit, per the standard, so they are
    clipped to the datasheet limits when `clip_to_datasheet`.
    """
    params = params or ds.params
    checkpoints = checkpoints if checkpoints is not None else ds.checkpoints
    m = ds.meas[(ds.meas["measurement_status"] == "MEASURED")
                & (ds.meas["checkpoint_h"].isin(checkpoints))]
    ref = m if reference_lots is None else m[m["lot_id"].isin(reference_lots)]

    idx = pd.Index(ds.gt["component_id"], name="component_id")
    flag = pd.Series(False, index=idx)
    score = pd.Series(0.0, index=idx)

    for p in params:
        lo_spec, hi_spec = ds.limits[p]
        for t in checkpoints:
            sub = m[m["checkpoint_h"] == t]
            rsub = ref[ref["checkpoint_h"] == t]
            if sub.empty or rsub.empty:
                continue
            if mode == "static":
                a = rsub[p].dropna().to_numpy()
                lo, hi = limits_for(a, estimator, k)
                z = np.abs(robust_z_against(sub[p].to_numpy(), a, estimator))
                lo = max(lo, lo_spec) if clip_to_datasheet else lo
                hi = min(hi, hi_spec) if clip_to_datasheet else hi
                out = (sub[p] < lo) | (sub[p] > hi)
            else:
                refs, pooled = _lot_reference(rsub, p, MIN_LOT_N)
                out = pd.Series(False, index=sub.index)
                z = np.zeros(len(sub))
                for lot, g in sub.groupby("lot_id"):
                    a = refs.get(lot, pooled)
                    lo, hi = limits_for(a, estimator, k)
                    if clip_to_datasheet:
                        lo, hi = max(lo, lo_spec), min(hi, hi_spec)
                    pos = sub.index.get_indexer(g.index)
                    z[pos] = np.abs(robust_z_against(g[p].to_numpy(), a, estimator))
                    out.loc[g.index] = (g[p] < lo) | (g[p] > hi)
            cid = sub["component_id"].to_numpy()
            hit = pd.Series(out.to_numpy(), index=cid).groupby(level=0).any()
            flag.loc[flag.index.isin(hit.index[hit])] = True
            zs = pd.Series(z, index=cid).groupby(level=0).max()
            score.loc[zs.index] = np.maximum(score.reindex(zs.index).to_numpy(),
                                             zs.to_numpy())
    return pd.DataFrame({"flag": flag, "score": score})


def robust_z_against(x: np.ndarray, ref: np.ndarray, estimator: str) -> np.ndarray:
    """|z| of x measured against a reference population's centre and scale."""
    med = float(np.median(ref))
    if estimator == "p1p99":
        p1, p99 = np.percentile(ref, [1, 99])
        up = max((p99 - med) * P99_FACTOR, 1e-12)
        dn = max((med - p1) * P99_FACTOR, 1e-12)
        return np.where(x >= med, (x - med) / up, (x - med) / dn)
    centre_fn, sigma_fn, _ = ESTIMATORS[estimator]
    c, s = centre_fn(ref), sigma_fn(ref)
    return (x - c) / max(s, 1e-12)
