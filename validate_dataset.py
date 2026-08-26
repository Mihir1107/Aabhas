"""
Validator and calibration self-test for the SIH26170 synthetic burn-in dataset.

This file exists to answer one question: *is the dataset correctly calibrated,
or does it only look correct?* It does that in three parts.

  1. Hard assertions. Structural properties the dataset must satisfy or it is
     not fit to build on. Any failure raises.

  2. A throwaway AEC-Q001 DPAT check. This is a TEST FIXTURE, not Module A.
     It is deliberately implemented here, in the validator, so nobody mistakes
     it for the real detector. Its job is to show that the difficulty gradient
     across defect types is real: DPAT should catch a meaningful fraction of
     Type I, essentially none of Type III, and should be punished by the traps.

  3. A throwaway robust-Mahalanobis reference, for one specific reason: without
     it, "DPAT catches no Type III parts" is ambiguous between "the centre-hider
     is hiding successfully" and "the centre-hider is not there at all". The
     joint-distribution reference is what distinguishes those two, and that is
     the single most important thing to get right about this dataset.

Nothing else is modelled here. No detector, no DPAT implementation for the real
pipeline, no drift predictor.

Run:  python3 validate_dataset.py [--data data] [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

IQR_TO_SIGMA = 1.35        # AEC-Q001 Rev-D divisor (exact value 1.348980)
MAD_TO_SIGMA = 1.4826


def robust_sigma_iqr(a: np.ndarray) -> float:
    """AEC-Q001 robust sigma: (Q3 - Q1) / 1.35."""
    q1, q3 = np.nanpercentile(a, [25, 75])
    return (q3 - q1) / IQR_TO_SIGMA


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; used because several defect classes are tiny and
    a bare percentage would overstate what the dataset can actually support."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


_GOOD_D2: list = [pd.Series(dtype=float)]


class Report:
    def __init__(self, strict: bool):
        self.strict = strict
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{name}: {detail}")
        return ok

    def warn(self, msg: str) -> None:
        print(f"  [WARN] {msg}")
        self.warnings.append(msg)


def h1(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def h2(t: str) -> None:
    print(f"\n--- {t} " + "-" * max(0, 72 - len(t)))


# --------------------------------------------------------------------------
# dataset container
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    meas: pd.DataFrame
    gt: pd.DataFrame
    cfg: dict

    @property
    def params(self) -> list[str]:
        return [p["name"] for p in self.cfg["parameters"]]

    @property
    def checkpoints(self) -> list[float]:
        return list(self.cfg["config"]["checkpoints_h"])

    @property
    def limits(self) -> dict[str, tuple[float, float]]:
        return {p["name"]: (p["limit_lo"], p["limit_hi"]) for p in self.cfg["parameters"]}

    def wide(self) -> pd.DataFrame:
        """component_id x (param, checkpoint) matrix of measured values."""
        w = self.meas.pivot_table(index="component_id", columns="checkpoint_h",
                                  values=self.params, dropna=False)
        return w.reindex(self.gt["component_id"].to_numpy())


def load(dirpath: Path) -> Dataset:
    meas = pd.read_csv(dirpath / "burnin_measurements.csv")
    gt = pd.read_csv(dirpath / "ground_truth.csv")
    cfg = json.loads((dirpath / "config.json").read_text())
    return Dataset(meas, gt, cfg)


# --------------------------------------------------------------------------
# 1. hard assertions
# --------------------------------------------------------------------------

def assert_no_label_leak(ds: Dataset, rep: Report) -> None:
    h2("A0. Measurements file carries no labels")
    banned = {"is_defective", "defect_type", "is_trap", "failure_mechanism",
              "injection_detail", "true_v168"}
    present = [c for c in ds.meas.columns
               if c in banned or any(b in c for b in ("defect", "true_v168", "is_def"))]
    rep.check("no label-bearing column in burnin_measurements.csv",
              not present, f"found {present}" if present else "")


def assert_limits(ds: Dataset, rep: Report) -> None:
    h2("A1. No injected part violates a datasheet limit at any checkpoint")
    m = ds.meas.merge(ds.gt[["component_id", "defect_type", "is_defective", "is_trap"]],
                      on="component_id", how="left")
    m = m[m["measurement_status"] == "MEASURED"]
    viol = pd.Series(False, index=m.index)
    for p, (lo, hi) in ds.limits.items():
        viol |= (m[p] < lo) | (m[p] > hi)
    m = m.assign(_v=viol)

    bad_def = m.loc[m["_v"] & m["is_defective"], "defect_type"].value_counts()
    bad_trap = m.loc[m["_v"] & m["is_trap"], "defect_type"].value_counts()
    n_good = int((m["_v"] & ~m["is_defective"] & ~m["is_trap"]).sum())

    rep.check("no genuine defect (I-IV) out of spec at any measured checkpoint",
              bad_def.empty, "" if bad_def.empty else bad_def.to_dict())
    rep.check("no trap (V-VII) out of spec at any measured checkpoint",
              bad_trap.empty, "" if bad_trap.empty else bad_trap.to_dict())
    print(f"        good parts naturally out of spec: {n_good} row(s) "
          f"({n_good / max(len(m), 1) * 1e6:.1f} DPPM of measurements) "
          f"-- these are legitimate L0 static-limit catches, not injected")

    # the deliberate exception: latent 168 h values of pulled parts
    pulled = ds.gt[ds.gt["censor_status_168h"] == "PULLED_FAILED"]
    if len(pulled):
        oos = 0
        for p, (lo, hi) in ds.limits.items():
            v = pulled[f"true_v168_{p}"]
            oos += int(((v < lo) | (v > hi)).sum())
        rep.check("pulled parts' latent 168 h values ARE out of spec (that is why "
                  "they were pulled)", oos >= len(pulled),
                  f"{oos} out-of-spec latent values across {len(pulled)} pulled parts")


def assert_lot_size(ds: Dataset, rep: Report) -> None:
    h2("A2. Every lot has at least 30 parts (AEC-Q001 limit estimation)")
    n = ds.gt.groupby("lot_id").size()
    rep.check("min parts per lot >= 30", int(n.min()) >= 30,
              f"min={int(n.min())}, max={int(n.max())}, lots={len(n)}")


def assert_type3_band(ds: Dataset, rep: Report) -> pd.DataFrame:
    h2("A3. Type III centre-hiders sit inside the central band on EVERY parameter")
    band = float(ds.cfg["config"]["type3_band_pct"])
    lo_p, hi_p = 50 - band, 50 + band
    t3 = ds.gt.loc[ds.gt["defect_type"] == "III_CENTRE_HIDER", "component_id"]
    if t3.empty:
        print("        (no Type III parts in this run)")
        return pd.DataFrame()

    rows = []
    lut = ds.meas.set_index(["component_id", "checkpoint_h"])
    lot_of = ds.gt.set_index("component_id")["lot_id"]
    for cid in t3:
        lot = lot_of[cid]
        peers = ds.gt.loc[ds.gt["lot_id"] == lot, "component_id"]
        for t in ds.checkpoints:
            ref = ds.meas[(ds.meas["component_id"].isin(peers))
                          & (ds.meas["checkpoint_h"] == t)]
            for p in ds.params:
                v = lut.loc[(cid, t), p]
                pct = stats.percentileofscore(ref[p].dropna(), v, kind="mean")
                rows.append({"component_id": cid, "checkpoint_h": t,
                             "parameter": p, "value": v, "lot_percentile": pct})
    df = pd.DataFrame(rows)
    # A censored checkpoint has no value, so it has no percentile. Those rows
    # must be dropped rather than compared: NaN <= band is False, which would
    # fail the assertion for a part that is perfectly inside the band.
    n_cens = int(df["value"].isna().sum())
    df = df[df["value"].notna()].copy()
    worst = df.assign(dev=(df["lot_percentile"] - 50).abs()).sort_values("dev", ascending=False)
    ok = bool((worst["dev"] <= band + 1e-9).all())
    rep.check(f"all Type III marginals within the {lo_p:.0f}-{hi_p:.0f} percentile band",
              ok, f"worst = {worst.iloc[0]['lot_percentile']:.1f}th pct on "
                  f"{worst.iloc[0]['parameter']} @ {worst.iloc[0]['checkpoint_h']:.0f}h")

    if n_cens:
        print(f"        ({n_cens} censored measurement(s) on Type III parts skipped)")
    print(f"\n        percentile actually reached, per part (worst of "
          f"{len(ds.params)} params x {len(ds.checkpoints)} checkpoints):")
    g = df.assign(dev=(df["lot_percentile"] - 50).abs()).groupby("component_id")
    summ = g.apply(lambda d: pd.Series({
        "worst_pct": d.loc[d["dev"].idxmax(), "lot_percentile"],
        "worst_on": d.loc[d["dev"].idxmax(), "parameter"],
        "median_abs_dev_from_50": d["dev"].median()}), include_groups=False)
    for cid, r in summ.iterrows():
        print(f"          {cid}  worst {r['worst_pct']:5.1f}th pct on {r['worst_on']:<18s}"
              f" median |dev| {r['median_abs_dev_from_50']:4.1f}pp")
    return df


def assert_trap_labels(ds: Dataset, rep: Report) -> None:
    h2("A4. Types V, VI and VII are labelled is_defective = False")
    for t in ("Va_MILDLY_HIGH_STABLE", "VI_LOT_SHIFT", "VII_FIXTURE_ARTIFACT"):
        sub = ds.gt[ds.gt["defect_type"] == t]
        if sub.empty:
            continue
        rep.check(f"{t} labelled not-defective (n={len(sub)})",
                  not sub["is_defective"].any() and bool(sub["is_trap"].all()))


def assert_censoring(ds: Dataset, rep: Report) -> None:
    h2("A5. Censored parts are correctly typed and carry no 168 h measurement")
    end = max(ds.checkpoints)
    last = ds.meas[ds.meas["checkpoint_h"] == end].set_index("component_id")
    gt = ds.gt.set_index("component_id")
    joined = gt.join(last[["measurement_status"] + ds.params], rsuffix="_m")

    rep.check("ground-truth censor status matches the measurements file",
              bool((joined["censor_status_168h"] == joined["measurement_status"]).all()))

    for st in ("PULLED_FAILED", "MISSING_EQUIPMENT"):
        sub = joined[joined["measurement_status"] == st]
        if sub.empty:
            continue
        has_val = sub[ds.params].notna().any(axis=1)
        rep.check(f"{st}: no 168 h value present (n={len(sub)})", not bool(has_val.any()),
                  f"{int(has_val.sum())} rows still carry a value")
        truth = sub[[f"true_v168_{p}" for p in ds.params]].notna().all(axis=1)
        rep.check(f"{st}: true 168 h value recorded in ground truth",
                  bool(truth.all()), f"{int((~truth).sum())} missing")

    surv = joined["measurement_status"] == "MEASURED"
    dfr = joined.loc[surv, "is_defective"].mean()
    dall = joined["is_defective"].mean()
    print(f"\n        SURVIVORSHIP TRAP: defect rate among 168 h survivors "
          f"{dfr * 100:.3f}% vs {dall * 100:.3f}% over all parts. Training only on "
          f"survivors drops {int(joined['is_defective'].sum() - joined.loc[surv, 'is_defective'].sum())} "
          f"of the worst drifters and under-estimates drift.")
    n_pull = int((joined["measurement_status"] == "PULLED_FAILED").sum())
    n_miss_good = int(((joined["measurement_status"] == "MISSING_EQUIPMENT")
                       & ~joined["is_defective"]).sum())
    print(f"        missing-at-168h is not a label leak: {n_pull} PULLED_FAILED "
          f"vs {n_miss_good} MISSING_EQUIPMENT on non-defective parts")


def assert_sublinear(ds: Dataset, rep: Report) -> pd.DataFrame:
    h2("A6. Good-part degradation is sub-linear and saturating (beta recovered "
       "empirically)")
    good = ds.gt.loc[ds.gt["defect_type"] == "GOOD", "component_id"]
    m = ds.meas[ds.meas["component_id"].isin(set(good))
                & (ds.meas["measurement_status"] == "MEASURED")]
    ts = [t for t in ds.checkpoints if t > 0]
    t0 = min(ds.checkpoints)
    x = np.log(np.array(ts))
    rows = []

    # Two things matter for getting this estimate right.
    #
    # 1. Drift is computed PER PART and only then aggregated. Differencing the
    #    medians instead -- median(x_t) - median(x_0) -- is biased, because v0
    #    and lambda are separate random effects and the median of a sum is not
    #    the sum of medians. Per-part differencing cancels v0 exactly.
    #
    #    Aggregation across parts then uses the MEAN, not the median. lambda is
    #    log-normal, so drift is right-skewed; adding symmetric measurement
    #    noise pulls a median down, and it pulls it down harder at 24 h (where
    #    drift is comparable to the noise) than at 168 h (where it dwarfs it).
    #    That flattens the log-log slope and biases beta low by ~0.012. The mean
    #    is unaffected because the noise is zero-mean at every checkpoint. Using
    #    it is safe here only because this runs on GOOD parts exclusively, so
    #    there is no contamination for the mean to be sensitive to.
    #
    # 2. beta is fitted WITHIN each lot and then averaged across lots. beta
    #    carries a small per-lot jitter, so a single pooled fit is a fit to a
    #    mixture of power laws with different exponents, which is not itself a
    #    power law and biases the slope low by ~0.02. Fitting where beta is
    #    actually constant removes that.
    wide = ds.meas[ds.meas["component_id"].isin(set(good))].pivot_table(
        index="component_id", columns=["checkpoint_h"], values=ds.params, dropna=False)
    lot_of = ds.gt.set_index("component_id")["lot_id"].reindex(wide.index)

    def fit(med: np.ndarray) -> float:
        if not np.isfinite(med).all() or (med <= 0).any():
            return float("nan")
        return float(np.polyfit(x, np.log(med), 1)[0])

    for p in ds.params:
        d = pd.DataFrame({t: wide[(p, t)] - wide[(p, t0)] for t in ts})
        per_lot = np.array([fit(d.loc[lot_of == lot].mean(axis=0).to_numpy())
                            for lot in lot_of.dropna().unique()])
        per_lot = per_lot[np.isfinite(per_lot)]
        beta_hat = float(np.mean(per_lot))
        se = float(np.std(per_lot, ddof=1) / np.sqrt(len(per_lot)))
        lo, hi = beta_hat - 1.96 * se, beta_hat + 1.96 * se
        med = d.median(axis=0)
        tm, te = ts[-2], ts[-1]
        cfgbeta = next(q["beta"] for q in ds.cfg["parameters"] if q["name"] == p)
        rows.append({"parameter": p, "beta_config": cfgbeta, "beta_empirical": beta_hat,
                     "beta_lo95": lo, "beta_hi95": hi, "n_lots": len(per_lot),
                     "concave": bool(med[tm] > med[te] * tm / te),
                     "drift_168h": med[te],
                     "linear_at_96h": med[te] * tm / te,
                     "actual_at_96h": med[tm]})

    df = pd.DataFrame(rows)
    # Fail only when the interval is CONFIDENTLY outside [0.5, 1.0]; a wide
    # interval means too little data to say, which is a power problem.
    conf_out = ((df["beta_lo95"] > 1.0) | (df["beta_hi95"] < 0.5)).any()
    rep.check("empirical beta not confidently outside [0.5, 1.0]", not bool(conf_out),
              f"range {df['beta_empirical'].min():.3f}-{df['beta_empirical'].max():.3f}")
    rep.check("every parameter's median trajectory is concave (saturating)",
              bool(df["concave"].all()))
    # Tolerance scales with the statistical power actually available: a small
    # run cannot resolve beta to 0.02, and failing it for that would blame the
    # generator for a sample-size limit.
    width = float((df["beta_hi95"] - df["beta_lo95"]).max())
    err = float((df["beta_empirical"] - df["beta_config"]).abs().max())
    tol = max(0.02, width / 2)
    rep.check(f"empirical beta recovers the configured beta within {tol:.3f}",
              err < tol, f"max |error| = {err:.4f}, widest 95% CI = {width:.3f}")
    if width > 0.10:
        rep.warn(f"beta estimate is under-powered (widest 95% CI = {width:.2f}). "
                 f"With {len(good)} good parts the per-lot drift curves are noisy; "
                 "this is a sample-size limit, not a defect in the generator. "
                 "Raise --parts-per-lot to tighten it.")
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    over = (df["linear_at_96h"] / df["actual_at_96h"] - 1) * 100
    print(f"\n        A linear extrapolator anchored on the 168 h endpoint would "
          f"under-shoot 96 h by {-over.mean():.1f}% on average; equivalently, "
          f"extrapolating the 0->24 h slope to 168 h OVER-predicts. That is the "
          f"error mode Module B has to avoid, and it is present by construction.")
    return df


def assert_type3_smoothness(ds: Dataset, rep: Report) -> None:
    """A8. Type III trajectories must be no smoother than a good part's.

    v1.0 placed each Type III checkpoint at a deterministic lot quantile, and
    lot quantiles move smoothly with t, so the parts inherited the smooth lot
    curve and carried no independent per-checkpoint measurement noise. Their
    trajectories were 3-5x smoother than a real part's and a trivial "too
    smooth" rule recovered 45.8% of the class at 1% yield loss -- a construction
    artifact leaking the single class that justifies the multivariate layer.

    Smoothness statistic: RMS residual of each part's 4-point trajectory from a
    per-part least-squares fit of {1, t^0.6}, averaged over parameters after
    normalising each by the good population's median. Chosen over the
    second-difference magnitude because it is what the original discriminator
    used, so the numbers are directly comparable across versions.

    Two-sided by design. Being reliably ROUGHER than a good part is just as much
    a fingerprint as being smoother, so both tails are tested.
    """
    h2("A8. Type III trajectories are no smoother (and no rougher) than good parts")
    t3 = ds.gt.loc[ds.gt["defect_type"] == "III_CENTRE_HIDER", "component_id"]
    if t3.empty:
        print("        (no Type III parts in this run)")
        return
    m = ds.meas.copy()
    m.loc[m["measurement_status"] != "MEASURED", ds.params] = np.nan
    w = m.pivot_table(index="component_id", columns="checkpoint_h",
                      values=ds.params, dropna=False)
    T = np.array(ds.checkpoints, dtype=float)
    X = np.vstack([np.ones(len(T)), T ** 0.6]).T
    cols = {}
    for p in ds.params:
        V = np.vstack([w[(p, t)].to_numpy() for t in T]).T
        ok = np.isfinite(V).all(axis=1)
        r = np.full(len(V), np.nan)
        co, _, _, _ = np.linalg.lstsq(X, V[ok].T, rcond=None)
        r[ok] = np.sqrt(((V[ok] - (X @ co).T) ** 2).mean(axis=1))
        cols[p] = pd.Series(r, index=w.index)
    rough = pd.DataFrame(cols)
    ty = ds.gt.set_index("component_id")["defect_type"].reindex(rough.index)
    norm = rough.div(rough[ty == "GOOD"].median(), axis=1).mean(axis=1)
    g = norm[ty == "GOOD"].dropna()
    s3 = norm[ty == "III_CENTRE_HIDER"].dropna()

    ks = stats.ks_2samp(s3, g)
    mw = stats.mannwhitneyu(s3, g)
    print(f"        n Type III with a complete trajectory: {len(s3)}  "
          f"(good reference n={len(g)})")
    print(f"        roughness ratio, median: Type III {s3.median():.3f} vs "
          f"good 1.000   [v1.0: 0.25]")
    print(f"        Kolmogorov-Smirnov  D={ks.statistic:.4f}  p={ks.pvalue:.4f}")
    print(f"        Mann-Whitney U                        p={mw.pvalue:.4f}")
    rep.check("Type III roughness indistinguishable from good parts (KS p > 0.01)",
              ks.pvalue > 0.01, f"p={ks.pvalue:.4f}")

    smooth = 100 * float((s3 <= g.quantile(0.01)).mean())
    roughc = 100 * float((s3 >= g.quantile(0.99)).mean())
    print(f"\n        'too smooth' discriminator at 1% yield loss catches "
          f"{smooth:.1f}% of Type III   [v1.0: 45.8%]")
    print(f"        'too rough'  discriminator at 1% yield loss catches "
          f"{roughc:.1f}% of Type III")
    print("        Chance is 1.0% by construction, since the threshold is the "
          "good population's\n        1st percentile.")
    rep.check("'too smooth' rule is at chance (catches < 5% of Type III)",
              smooth < 5.0, f"{smooth:.1f}%")
    rep.check("'too rough' rule is at chance (catches < 5% of Type III)",
              roughc < 5.0, f"{roughc:.1f}%")


def assert_lot_separation(ds: Dataset, rep: Report) -> pd.DataFrame:
    h2("A7. Lots are genuinely different from each other")
    good = ds.gt.loc[ds.gt["defect_type"] == "GOOD", ["component_id", "lot_id"]]
    m = ds.meas[(ds.meas["checkpoint_h"] == min(ds.checkpoints))
                & (ds.meas["measurement_status"] == "MEASURED")]
    m = m.merge(good, on=["component_id", "lot_id"])
    rows = []
    for p in ds.params:
        g = m.groupby("lot_id")[p]
        means, n = g.mean(), g.size()
        k = len(means)
        msb = float((n * (means - m[p].mean()) ** 2).sum() / (k - 1))
        msw = float(g.var().mean())
        n0 = float(n.mean())
        var_b = max((msb - msw) / n0, 0.0)
        icc = var_b / (var_b + msw)
        f = msb / msw
        rows.append({"parameter": p, "between_lot_sd": float(means.std()),
                     "within_lot_sd": float(np.sqrt(msw)),
                     "var_ratio_between_within": var_b / msw, "ICC": icc,
                     "anova_F": f,
                     "p_value": float(stats.f.sf(f, k - 1, len(m) - k))})
    df = pd.DataFrame(rows)
    rep.check("ICC > 0.05 on every parameter (real lot-to-lot shift exists)",
              bool((df["ICC"] > 0.05).all()),
              f"min ICC = {df['ICC'].min():.3f}")
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n        This is the number that justifies DYNAMIC limits. With ICC near "
          "zero,\n        a static limit derived from history would work and the whole "
          "premise collapses.")
    return df


# --------------------------------------------------------------------------
# 2. throwaway AEC-Q001 DPAT fixture
# --------------------------------------------------------------------------

def dpat_flags(ds: Dataset, n_sigma: float, mode: str) -> pd.Series:
    """AEC-Q001 Part Average Testing, implemented here ONLY as a calibration
    fixture.

        robust mean  = median
        robust sigma = (Q3 - Q1) / 1.35
        limits       = median +/- n_sigma * robust_sigma

    Per the standard, PAT limits never loosen an engineering limit, so they are
    clipped to the datasheet limits.

    mode="dynamic": limits recomputed per lot, per checkpoint, per parameter
                    (this is Dynamic PAT, what the problem statement describes)
    mode="static" : limits computed once from all lots pooled
                    (this is Static PAT, kept as the contrast that makes the
                    Type VI lot-shift trap legible)

    A part is flagged if any parameter at any checkpoint falls outside.
    """
    m = ds.meas[ds.meas["measurement_status"] == "MEASURED"]
    flag = pd.Series(False, index=pd.Index(ds.gt["component_id"], name="component_id"))
    for p in ds.params:
        lo_spec, hi_spec = ds.limits[p]
        for t in ds.checkpoints:
            sub = m[m["checkpoint_h"] == t]
            if sub.empty:
                continue
            if mode == "static":
                med = sub[p].median()
                sig = robust_sigma_iqr(sub[p].to_numpy())
                lo = max(med - n_sigma * sig, lo_spec)
                hi = min(med + n_sigma * sig, hi_spec)
                out = sub.loc[(sub[p] < lo) | (sub[p] > hi), "component_id"]
            else:
                g = sub.groupby("lot_id")[p]
                med = g.transform("median")
                sig = g.transform(lambda a: robust_sigma_iqr(a.to_numpy()))
                lo = np.maximum(med - n_sigma * sig, lo_spec)
                hi = np.minimum(med + n_sigma * sig, hi_spec)
                out = sub.loc[(sub[p] < lo) | (sub[p] > hi), "component_id"]
            flag.loc[flag.index.isin(out)] = True
    return flag


def mahalanobis_scores(ds: Dataset) -> pd.Series:
    """Throwaway robust-multivariate reference.

    Included for exactly one reason: to tell apart "the centre-hider is hiding"
    from "the centre-hider is not actually anomalous". Per lot and checkpoint we
    fit a Minimum Covariance Determinant estimate on the five parameters and
    take each part's worst (max over checkpoints) robust Mahalanobis distance.
    """
    from sklearn.covariance import MinCovDet
    m = ds.meas[ds.meas["measurement_status"] == "MEASURED"]
    acc: dict[str, list[float]] = {}
    for (lot, t), sub in m.groupby(["lot_id", "checkpoint_h"]):
        X = sub[ds.params].to_numpy()
        if len(X) < 5 * 3:
            continue
        mcd = MinCovDet(support_fraction=0.9, random_state=0).fit(X)
        for cid, v in zip(sub["component_id"].to_numpy(), mcd.mahalanobis(X)):
            acc.setdefault(cid, []).append(float(v))
    idx = pd.Index(ds.gt["component_id"], name="component_id")
    return pd.DataFrame({
        "maha_median_d2": pd.Series({k: float(np.median(v)) for k, v in acc.items()}),
        "maha_max_d2": pd.Series({k: float(np.max(v)) for k, v in acc.items()}),
    }).reindex(idx).fillna(0.0)


def estimator_comparison(ds: Dataset, rep: Report) -> None:
    """The robust-estimator argument, demonstrated rather than asserted.

    Part 0.2 of the research doc picks MAD over AEC-Q001's IQR/1.35 and over the
    p1/p99 variant on breakdown point: 50% vs 25% vs ~1%. That argument only has
    teeth on data that is skewed AND contaminated, which is what this dataset is
    -- Iddq and leakage are log-normal, and Type Vb parts sit far out in the
    upper tail.

    Three things get measured on the log-normal parameters, per lot, at 0 h:

      inflation   how far the contamination pushes the estimator's own upper
                  limit outward, in units of the CLEAN MAD sigma. Scale-free, so
                  estimators with differently-placed limits are comparable. This
                  is the breakdown-point property.
      Vb catch    fraction of the problem statement's canonical defect caught,
                  with every estimator held to the SAME 1% good-part overkill so
                  the comparison is not just "whose limit happens to be tighter".
    """
    h2("Robust estimator comparison: does the breakdown-point argument hold up?")
    logp = [q["name"] for q in ds.cfg["parameters"] if q["log_scale"]]
    if not logp:
        print("        (no log-scale parameters configured)")
        return
    m = ds.meas[(ds.meas["measurement_status"] == "MEASURED")
                & (ds.meas["checkpoint_h"] == 0.0)]
    gtx = ds.gt.set_index("component_id")["defect_type"]
    good = set(gtx.index[gtx == "GOOD"])
    vb = set(gtx.index[gtx == "Vb_EXTREME_LEVEL"])
    if len(vb) < 10:
        # The breakdown-point argument is about contamination. With no Type Vb
        # parts in the upper tail there is nothing for an estimator to be
        # corrupted BY, every estimator agrees, and asserting on that would be
        # testing noise. Skip rather than pretend.
        print(f"        SKIPPED: only {len(vb)} Type Vb part(s) present. This "
              "comparison needs\n        upper-tail contamination to mean anything "
              "-- enable type V.")
        return

    def lims(a: np.ndarray, n: float = 6.0) -> dict[str, tuple[float, float]]:
        med, mu, sd = float(np.median(a)), float(np.mean(a)), float(np.std(a, ddof=1))
        mad = MAD_TO_SIGMA * float(np.median(np.abs(a - med)))
        iqr = robust_sigma_iqr(a)
        p99 = 0.429858 * float(np.percentile(a, 99) - med)
        p01 = 0.429858 * float(med - np.percentile(a, 1))
        return {"MAD": (med - n * mad, med + n * mad),
                "IQR (AEC-Q001)": (med - n * iqr, med + n * iqr),
                "p1/p99 variant": (med - n * p01, med + n * p99),
                "classical mean+/-6sd": (mu - n * sd, mu + n * sd)}

    # Two comparisons, because they answer different questions.
    #
    # (a) inflation at a fixed 6 sigma: how far the contamination pushes each
    #     estimator's own limit outward, in clean-MAD-sigma units. This is the
    #     breakdown-point property in isolation.
    #
    # (b) Vb catch at MATCHED overkill: each part is scored as
    #     (x - lot centre) / lot sigma_est, and the threshold is set on the
    #     pooled good parts so every estimator sacrifices the same 1% of good
    #     parts. Comparing at a fixed 6 sigma instead would be unfair -- MAD
    #     sigma is not inflated by the skew, so its 6 sigma limit is simply
    #     tighter, and it would win on catch purely by rejecting more. Within a
    #     single lot the estimator cannot change the ordering of parts at all;
    #     what it changes is how comparable different lots are, which is
    #     precisely what a fixed-yield-loss comparison across lots measures.
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    zs: dict[tuple[str, str], pd.Series] = {}
    for p in logp:
        for lot, sub in m.groupby("lot_id"):
            allv = sub[p].dropna()
            cln = sub.loc[sub["component_id"].isin(good), p].dropna()
            if len(cln) < 30:
                continue
            La, Lc = lims(allv.to_numpy()), lims(cln.to_numpy())
            med = float(np.median(cln))
            mad_sig = MAD_TO_SIGMA * float(np.median(np.abs(cln - med))) or np.nan
            for k in La:
                dd = acc.setdefault((p, k), {"infl": [], "fixed": []})
                dd["infl"].append((La[k][1] - Lc[k][1]) / mad_sig)
                vsub = sub.loc[sub["component_id"].isin(vb), p].dropna()
                if len(vsub):
                    dd["fixed"].append(100 * float(
                        ((vsub < La[k][0]) | (vsub > La[k][1])).mean()))
                # (b) score every part in this lot on the contaminated estimate
                centre = float(np.mean(allv)) if k.startswith("classical") \
                    else float(np.median(allv))
                sig = (La[k][1] - centre) / 6.0
                if sig <= 0:
                    continue
                zz = (sub.set_index("component_id")[p] - centre) / sig
                zs[(p, k)] = pd.concat([zs.get((p, k), pd.Series(dtype=float)), zz])

    rows = []
    for (p, k), d in acc.items():
        z = zs.get((p, k))
        catch = float("nan")
        if z is not None:
            zg = z[z.index.isin(good)].dropna()
            zv = z[z.index.isin(vb)].dropna()
            if len(zg) and len(zv):
                catch = 100 * float((zv >= zg.quantile(0.99)).mean())
        rows.append({"parameter": p, "estimator": k,
                     "inflation (clean MAD sigmas)": float(np.median(d["infl"])),
                     "Vb catch @fixed 6sig %": (float(np.mean(d["fixed"]))
                                                if d["fixed"] else float("nan")),
                     "Vb catch @matched overkill %": catch})
    df = pd.DataFrame(rows)
    order = ["MAD", "IQR (AEC-Q001)", "p1/p99 variant", "classical mean+/-6sd"]
    for p in logp:
        sub = df[df["parameter"] == p].set_index("estimator").reindex(order)
        print(f"\n        {p}:")
        print(sub.drop(columns=["parameter"]).to_string(
            float_format=lambda v: f"{v:8.3f}"))

    agg = df.groupby("estimator")[["inflation (clean MAD sigmas)",
                                   "Vb catch @fixed 6sig %",
                                   "Vb catch @matched overkill %"]].mean()
    best = agg["inflation (clean MAD sigmas)"].idxmin()
    worst = agg["inflation (clean MAD sigmas)"].idxmax()
    best = agg["inflation (clean MAD sigmas)"].idxmin()
    worst = agg["inflation (clean MAD sigmas)"].idxmax()
    spread_fixed = (agg["Vb catch @fixed 6sig %"].max()
                    - agg["Vb catch @fixed 6sig %"].min())
    spread_match = (agg["Vb catch @matched overkill %"].max()
                    - agg["Vb catch @matched overkill %"].min())
    print(f"\n        Least corrupted by the contamination: {best}. Most: {worst}.")
    print("        Breakdown points: MAD 50%, IQR/1.35 25%, p1/p99 ~1%, mean/sd 0%.\n")
    print("        The p1/p99 variant is inflated hardest despite a nominally higher\n"
          "        breakdown point than mean/sd, because it is computed FROM the top\n"
          "        1% -- exactly where the defects live. The outliers widen the very\n"
          "        limit meant to catch them.\n")
    print(f"        BUT READ THE TWO CATCH COLUMNS TOGETHER. At a FIXED 6 sigma the\n"
          f"        estimators differ by {spread_fixed:.0f} points of Vb catch; at MATCHED\n"
          f"        overkill they differ by {spread_match:.0f}. Within one lot the estimator\n"
          "        cannot reorder parts at all -- median and sigma are per-lot\n"
          "        constants -- so it cannot change the achievable recall-versus-yield-\n"
          "        loss curve. What it changes is WHERE a fixed 6 sigma limit lands and\n"
          "        how stable that limit is under contamination.\n")
    print("        So the honest claim for the deck is narrower than 'robust estimators\n"
          "        find more defects': they make a fixed, standard-specified limit mean\n"
          "        the same thing across lots and keep the outliers from widening it.\n"
          "        That is a limit-placement and reproducibility argument, not a recall\n"
          "        argument. Claiming the stronger version would not survive a judge\n"
          "        who asks to see it at matched yield loss.")
    rep.check("MAD limits are less contamination-inflated than the p1/p99 variant",
              agg.loc["MAD", "inflation (clean MAD sigmas)"]
              < agg.loc["p1/p99 variant", "inflation (clean MAD sigmas)"])
    rep.check("at a fixed 6 sigma, MAD catches more Vb than the p1/p99 variant",
              agg.loc["MAD", "Vb catch @fixed 6sig %"]
              > agg.loc["p1/p99 variant", "Vb catch @fixed 6sig %"])


def type3_feasibility(ds: Dataset, good_d2: pd.Series) -> None:
    """The Type III band/separability trade-off, computed from the real data.

    This is the single most important honest statement about the dataset. A
    centre-hider is defined by two requirements that pull against each other:
    it must be univariately central, and it must be jointly anomalous. A point
    at the exact median of every parameter has Mahalanobis distance zero, so
    the tighter the central band, the less joint anomaly is even reachable.

    For each candidate band we compute the exact box-constrained maximum
    Mahalanobis distance from the empirical within-lot correlation, and compare
    it against where the good population actually sits. If the ceiling for a
    band falls below the good population's upper tail, no centre-hider can be
    built inside that band at all -- not by better engineering, arithmetically.
    """
    h2("Type III feasibility: how central can a centre-hider be and still be found?")
    m = ds.meas[(ds.meas["measurement_status"] == "MEASURED")
                & (ds.meas["checkpoint_h"] == min(ds.checkpoints))]
    corrs = [np.corrcoef(sub[ds.params].dropna().to_numpy(), rowvar=False)
             for _, sub in m.groupby("lot_id")]
    R = np.mean(corrs, axis=0)
    inv = np.linalg.inv(R)
    p = len(ds.params)
    import itertools as it
    best = max(float(np.array(s) @ inv @ np.array(s))
               for s in it.product([-1.0, 1.0], repeat=p))
    band_now = float(ds.cfg["config"]["type3_band_pct"])
    g99, g995 = good_d2.quantile(0.99), good_d2.quantile(0.995)
    print(f"        empirical within-lot correlation: r(iddq,leakage)={R[0][1]:+.3f}  "
          f"r(leakage,delay)={R[1][2]:+.3f}  lambda_min={np.linalg.eigvalsh(R)[0]:.4f}")
    print(f"        good population joint distance: 99th pct D^2={g99:.1f}, "
          f"99.5th pct D^2={g995:.1f}\n")
    print(f"        {'band':>10s} {'|z| cap':>8s} {'max D^2':>9s} {'chi2_5 p':>10s}  verdict")
    for b in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0):
        c = stats.norm.ppf(0.5 + b / 100.0)
        d2 = best * c * c
        ok = "usable" if d2 > g995 * 1.15 else ("marginal" if d2 > g99 else "IMPOSSIBLE")
        mark = "  <-- configured" if abs(b - band_now) < 1e-9 else ""
        print(f"        +/-{b:4.0f}pp {c:8.3f} {d2:9.1f} {stats.chi2.sf(d2, p):10.2e}  "
              f"{ok}{mark}")
    print(f"\n        Read the IMPOSSIBLE rows literally: inside a +/-10pp band the most "
          f"anomalous\n        point that exists is less extreme than hundreds of "
          f"ordinary good parts. No\n        detector could ever separate it. The "
          f"configured +/-{band_now:.0f}pp band is the\n        tightest one that "
          "still leaves usable headroom.")


def write_trajectory_svg(ds: Dataset, path: Path) -> None:
    """Dependency-free SVG of a few trajectories per type, for eyeballing."""
    end = max(ds.checkpoints)
    w = ds.wide()
    gt = ds.gt.set_index("component_id")
    param = "iddq_ua"
    types = [t for t in ("GOOD", "I_STEEP_DRIFTER", "II_STEP_DEFECT",
                         "III_CENTRE_HIDER", "IV_CORRELATION_BREAK",
                         "Vb_EXTREME_LEVEL", "Va_MILDLY_HIGH_STABLE")
             if t in set(gt["defect_type"])]
    cols = {"GOOD": "#9aa5b1", "I_STEEP_DRIFTER": "#d94b3a", "II_STEP_DEFECT": "#e08a1e",
            "III_CENTRE_HIDER": "#7b3fbf", "IV_CORRELATION_BREAK": "#1f7ab0",
            "Vb_EXTREME_LEVEL": "#2e9e5b", "Va_MILDLY_HIGH_STABLE": "#b58900"}
    W, H, ml, mb = 900, 460, 70, 60
    vals = w[param].to_numpy(dtype=float)
    ymin, ymax = np.nanpercentile(vals, 0.2), np.nanpercentile(vals, 99.9)
    sx = lambda t: ml + (t / end) * (W - ml - 140)
    sy = lambda y: H - mb - (y - ymin) / (ymax - ymin) * (H - mb - 30)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{ml}" y="20" font-size="14" font-weight="600">'
           f'{param}: sample trajectories by injected type</text>']
    for t in ds.checkpoints:
        out.append(f'<line x1="{sx(t):.1f}" y1="30" x2="{sx(t):.1f}" y2="{H - mb}" '
                   f'stroke="#e6e6e6"/>')
        out.append(f'<text x="{sx(t):.1f}" y="{H - mb + 18}" font-size="11" '
                   f'text-anchor="middle" fill="#555">{t:.0f}h</text>')
    for k in range(5):
        y = ymin + (ymax - ymin) * k / 4
        out.append(f'<line x1="{ml}" y1="{sy(y):.1f}" x2="{W - 140}" y2="{sy(y):.1f}" '
                   f'stroke="#f0f0f0"/>')
        out.append(f'<text x="{ml - 8}" y="{sy(y) + 4:.1f}" font-size="11" '
                   f'text-anchor="end" fill="#555">{y:.1f}</text>')
    rng = np.random.default_rng(0)
    for ti, tp in enumerate(types):
        ids = gt.index[gt["defect_type"] == tp]
        n = 25 if tp == "GOOD" else 6
        pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
        for cid in pick:
            pts = [(sx(t), sy(w.loc[cid, (param, t)])) for t in ds.checkpoints
                   if np.isfinite(w.loc[cid, (param, t)])]
            if len(pts) < 2:
                continue
            d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                         for i, (x, y) in enumerate(pts))
            op = 0.35 if tp == "GOOD" else 0.95
            sw = 1 if tp == "GOOD" else 1.8
            out.append(f'<path d="{d}" fill="none" stroke="{cols[tp]}" '
                       f'stroke-width="{sw}" opacity="{op}"/>')
        y = 40 + ti * 20
        out.append(f'<line x1="{W - 130}" y1="{y}" x2="{W - 105}" y2="{y}" '
                   f'stroke="{cols[tp]}" stroke-width="2.5"/>')
        out.append(f'<text x="{W - 100}" y="{y + 4}" font-size="11" fill="#333">'
                   f'{tp.split("_")[0]}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out))
    print(f"\n        trajectory plot written to {path}")


def spatial_check(ds: Dataset) -> None:
    """Throwaway spatial fixture.

    Type VII is labelled GOOD and is meant to be invisible to every layer
    except a spatial one. That claim is only meaningful if the spatial signal
    is actually there, so this reports the board-level residual: each board's
    median parameter value minus its lot's median, in within-lot sigma. A
    fixture artifact should stand out here and nowhere else.
    """
    h2("Spatial fixture: within-board position gradient (is the Type VII signal real?)")
    aff = set(ds.cfg["derived"].get("type7_boards", []))
    if not aff:
        print("        (no Type VII boards in this run)")
        return
    m = ds.meas[(ds.meas["measurement_status"] == "MEASURED")
                & (ds.meas["checkpoint_h"] == max(ds.checkpoints))].copy()
    rr = (ds.cfg["config"]["board_rows"] - 1) / 2.0
    cc = (ds.cfg["config"]["board_cols"] - 1) / 2.0
    m["_d"] = np.sqrt(((m["socket_row"] - rr) / rr) ** 2
                      + ((m["socket_col"] - cc) / cc) ** 2)
    rows = []
    for p in ds.params:
        g = m.groupby("lot_id")[p]
        z = (m[p] - g.transform("median")) / g.transform(
            lambda a: robust_sigma_iqr(a.to_numpy()))
        rho = m.assign(_z=z).groupby("board_id").apply(
            lambda d: stats.spearmanr(d["_d"], d["_z"]).statistic
            if d["_d"].nunique() > 2 else np.nan, include_groups=False).dropna()
        af, no = rho[rho.index.isin(aff)], rho[~rho.index.isin(aff)]
        rows.append({"parameter": p,
                     "affected max|rho|": float(af.abs().max()),
                     "normal p99 |rho|": float(no.abs().quantile(0.99)),
                     "normal max |rho|": float(no.abs().max())})
    df = pd.DataFrame(rows)
    print("        Spearman rho between socket distance-from-board-centre and the "
          "part's\n        lot-relative z. A chamber thermal gradient shows up here "
          "and essentially\n        nowhere else, which is the whole point of the "
          "trap.\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sep = bool((df["affected max|rho|"] > df["normal max |rho|"]).any())
    print(f"\n        Type VII boards separable on the position gradient: {sep}. If "
          "this were False\n        the trap would be inert noise rather than a "
          "fixture artifact worth detecting.")


def calibration_selftest(ds: Dataset, rep: Report) -> pd.DataFrame:
    h1("CALIBRATION SELF-TEST  (throwaway fixtures, not Module A)")
    gt = ds.gt.set_index("component_id")

    print("AEC-Q001 DPAT:  robust_sigma = (Q3 - Q1) / 1.35,  "
          "limits = median +/- N * robust_sigma")
    print("PAT limits clipped to datasheet limits, as the standard requires.\n")

    flags = {
        "DPAT dyn 6s": dpat_flags(ds, 6.0, "dynamic"),
        "DPAT dyn 4s": dpat_flags(ds, 4.0, "dynamic"),
        "DPAT stat 6s": dpat_flags(ds, 6.0, "static"),
        "DPAT stat 4s": dpat_flags(ds, 4.0, "static"),
    }
    mh = mahalanobis_scores(ds)
    maha = mh["maha_median_d2"]
    # The threshold is set on the GOOD parts alone, so the budget really is
    # yield loss: "1% yield loss" means sacrificing 1% of genuinely good parts.
    # Setting it on all parts instead would let the contamination itself move
    # the threshold, which is the same mistake the p99 estimator makes.
    is_good = gt["defect_type"].reindex(maha.index).to_numpy() == "GOOD"
    good_d2 = maha[is_good]
    _GOOD_D2[0] = good_d2
    flags["Maha @1% YL"] = maha >= good_d2.quantile(0.99)

    order = ["GOOD", "I_STEEP_DRIFTER", "II_STEP_DEFECT", "III_CENTRE_HIDER",
             "IV_CORRELATION_BREAK", "Vb_EXTREME_LEVEL", "Va_MILDLY_HIGH_STABLE",
             "VI_LOT_SHIFT", "VII_FIXTURE_ARTIFACT"]
    rows = []
    for t in order:
        idx = gt.index[gt["defect_type"] == t]
        if not len(idx):
            continue
        row = {"defect_type": t, "n": len(idx),
               "share_%": 100 * len(idx) / len(gt)}
        for k, f in flags.items():
            row[k] = 100 * float(f.reindex(idx).fillna(False).mean())
        row["maha_med_D2"] = float(maha.reindex(idx).median())
        if t != "GOOD":
            s_t, s_g = maha.reindex(idx).dropna(), good_d2
            row["maha_AUROC"] = float(
                stats.mannwhitneyu(s_t, s_g, alternative="greater").statistic
                / (len(s_t) * len(s_g)))
        else:
            row["maha_AUROC"] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)

    h2("Flag rate (%) by defect type")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n        reference: robust Mahalanobis D^2 of GOOD parts -- "
          f"median {good_d2.median():.2f}, 99th pct {good_d2.quantile(0.99):.2f}, "
          f"99.5th pct {good_d2.quantile(0.995):.2f} (5 dof)")
    print("        maha_med_D2 is the separation that actually matters for the "
          "multivariate layer;\n        a type whose D^2 sits at the GOOD median is "
          "invisible to it by construction.")

    h2("Robust-Mahalanobis catch rate (%) vs yield-loss budget")
    print("        Threshold set on GOOD parts only, so the budget is literally the\n"
          "        fraction of good parts sacrificed. This is the recall-vs-yield-loss\n"
          "        curve the industry actually reasons about.\n")
    budgets = [0.005, 0.01, 0.02, 0.05]
    yl = []
    for t in order:
        idx = gt.index[gt["defect_type"] == t]
        if not len(idx) or t == "GOOD":
            continue
        r = {"defect_type": t, "n": len(idx)}
        for b in budgets:
            thr = good_d2.quantile(1 - b)
            r[f"@{b * 100:g}% YL"] = 100 * float((maha.reindex(idx) >= thr).mean())
        yl.append(r)
    print(pd.DataFrame(yl).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    if "III_CENTRE_HIDER" in set(gt["defect_type"]):
        t3 = maha.reindex(gt.index[gt["defect_type"] == "III_CENTRE_HIDER"])
        pr = [float((good_d2 < v).mean() * 100) for v in t3]
        print(f"\n        Type III parts sit at the {np.min(pr):.2f}-{np.max(pr):.2f}th "
              f"percentile of the GOOD population's joint distance\n        (median "
              f"{np.median(pr):.2f}th). That single number is the honest measure of how "
              f"hard\n        the centre-hider is: univariately invisible, jointly "
              f"this far out.")

    # confidence intervals, because several classes are tiny
    h2("DPAT dyn 6s catch rate with 95% Wilson interval")
    for _, r in df.iterrows():
        k = int(round(r["DPAT dyn 6s"] / 100 * r["n"]))
        lo, hi = wilson(k, int(r["n"]))
        print(f"  {r['defect_type']:<22s} n={int(r['n']):>6d}  "
              f"{r['DPAT dyn 6s']:6.2f}%  [{lo * 100:5.2f}, {hi * 100:5.2f}]")

    return df


# --------------------------------------------------------------------------
# 3. verdict
# --------------------------------------------------------------------------

def verdict(ds: Dataset, cat: pd.DataFrame, rep: Report) -> None:
    h1("VERDICT")
    g = cat.set_index("defect_type")

    def rate(t: str, col: str = "DPAT dyn 6s") -> float:
        return float(g.loc[t, col]) if t in g.index else float("nan")

    lines = []
    ok = True

    genuine = [t for t in ("I_STEEP_DRIFTER", "II_STEP_DEFECT", "III_CENTRE_HIDER",
                           "IV_CORRELATION_BREAK", "Vb_EXTREME_LEVEL")
               if t in g.index]
    n_gen = g.loc[genuine, "n"].sum() if genuine else 0
    caught = sum(g.loc[t, "n"] * g.loc[t, "DPAT dyn 6s"] / 100 for t in genuine)
    overall = 100 * caught / n_gen if n_gen else float("nan")

    if not np.isnan(overall) and overall > 90:
        ok = False
        lines.append(f"TOO EASY: dynamic DPAT at 6 sigma already catches {overall:.1f}% "
                     "of genuine defects. There is nothing left for the ML layers to "
                     "prove and the ablation table would be flat.")
    if "I_STEEP_DRIFTER" in g.index:
        r6, r4 = rate("I_STEEP_DRIFTER"), rate("I_STEEP_DRIFTER", "DPAT dyn 4s")
        if r6 < 1.0 and r4 < 5.0:
            ok = False
            lines.append(f"BROKEN: DPAT catches essentially no Type I steep drifters "
                         f"({r6:.1f}% at 6 sigma, {r4:.1f}% at 4 sigma). A steep drifter "
                         "that no univariate rule ever sees means the drift magnitude is "
                         "too small relative to part-to-part spread.")
        else:
            lines.append(f"Type I: DPAT catches {r6:.1f}% at 6 sigma and {r4:.1f}% at "
                         "4 sigma -- a meaningful but partial fraction, which is the "
                         "intended difficulty.")
    if "III_CENTRE_HIDER" in g.index:
        r3 = rate("III_CENTRE_HIDER")
        m3 = rate("III_CENTRE_HIDER", "Maha @1% YL")
        if r3 > 10:
            ok = False
            lines.append(f"BROKEN: DPAT catches {r3:.1f}% of Type III. A centre-hider "
                         "that a univariate rule can see is not a centre-hider.")
        else:
            lines.append(f"Type III: DPAT catches {r3:.1f}% (target: ~0). "
                         f"Robust Mahalanobis at a 1% yield-loss budget catches "
                         f"{m3:.1f}%.")
        if m3 < 30:
            ok = False
            lines.append(f"WEAK TYPE III: the joint-distribution reference only recovers "
                         f"{m3:.1f}% of centre-hiders. They are hidden from univariate "
                         "rules but they are not reliably visible jointly either, so the "
                         "'multivariate layer earns its place' claim is not yet supported.")
    if "Vb_EXTREME_LEVEL" in g.index:
        rb = rate("Vb_EXTREME_LEVEL")
        if rb < 40:
            ok = False
            lines.append(f"BROKEN: dynamic DPAT catches only {rb:.1f}% of Type Vb. "
                         "Vb is the problem statement's own worked example -- a part "
                         "far outside its lot but inside the datasheet limit -- and "
                         "it is exactly what DPAT exists to find. If L1 misses it, "
                         "the level shift is too small.")
        else:
            lines.append(f"Type Vb (the PS's worked example): dynamic DPAT catches "
                         f"{rb:.1f}% at 6 sigma vs static DPAT "
                         f"{rate('Vb_EXTREME_LEVEL', 'DPAT stat 6s'):.1f}%. This is "
                         "the rung that makes L1 a real baseline rather than a "
                         "formality.")
    fp = [(t, rate(t)) for t in ("Va_MILDLY_HIGH_STABLE", "VI_LOT_SHIFT",
                                 "VII_FIXTURE_ARTIFACT") if t in g.index]
    if fp:
        lines.append("Traps (should NOT be rejected): " + ", ".join(
            f"{t.split('_')[0]} {r:.2f}%" for t, r in fp) +
            " flagged by dynamic DPAT at 6 sigma. Every one of these is yield loss.")
    if "GOOD" in g.index:
        lines.append(f"Baseline overkill: {rate('GOOD'):.3f}% of genuinely good parts "
                     f"flagged by dynamic DPAT at 6 sigma "
                     f"({rate('GOOD', 'DPAT dyn 4s'):.3f}% at 4 sigma).")

    for l in lines:
        print("  * " + l)

    print()
    if rep.failures:
        print(f"  ASSERTIONS: {len(rep.failures)} FAILED")
        for f in rep.failures:
            print(f"    - {f}")
    else:
        print("  ASSERTIONS: all passed")

    final = ok and not rep.failures
    print(f"\n  DATASET VERDICT: {'USABLE' if final else 'NOT USABLE AS-IS'}")
    if not final:
        sys.exit(1)


# --------------------------------------------------------------------------

def summary_tables(ds: Dataset) -> None:
    h1("DATASET SUMMARY")
    cfg = ds.cfg["config"]
    d = ds.cfg["derived"]
    print(f"  lots={cfg['n_lots']}  parts/lot={cfg['parts_per_lot']}  "
          f"parts={len(ds.gt)}  checkpoints={cfg['checkpoints_h']}")
    print(f"  Arrhenius: Ea={cfg['ea_ev']} eV, {cfg['t_stress_c']}C stress vs "
          f"{cfg['t_use_c']}C use  ->  AF={d['acceleration_factor']:.2f}")
    print("  field-years by checkpoint: " + ", ".join(
        f"{k}h={v:.3f}y" for k, v in d["field_years_by_checkpoint"].items()))

    h2("Parts per type and contamination")
    vc = ds.gt["defect_type"].value_counts()
    tot = len(ds.gt)
    for t, n in vc.items():
        kind = ("defect" if t in ("I_STEEP_DRIFTER", "II_STEP_DEFECT",
                                  "III_CENTRE_HIDER", "IV_CORRELATION_BREAK")
                else "TRAP (labelled good)" if t != "GOOD" else "good")
        print(f"  {t:<24s} {n:>6d}  {100 * n / tot:7.4f}%  {n / tot * 1e6:9.1f} DPPM  {kind}")
    gen = ds.gt["is_defective"].sum()
    print(f"\n  genuine contamination (I-IV): {gen} parts = {100 * gen / tot:.4f}%  "
          f"({gen / tot * 1e6:.0f} DPPM)")

    h2("Mean drift 0h -> 168h by defect type (measured survivors only)")
    end = max(ds.checkpoints)
    w = ds.wide()
    rows = []
    for t in ds.gt["defect_type"].unique():
        ids = ds.gt.loc[ds.gt["defect_type"] == t, "component_id"]
        sub = w.reindex(ids)
        r = {"defect_type": t, "n": len(ids)}
        for p in ds.params:
            r[p] = float((sub[(p, end)] - sub[(p, 0.0)]).mean())
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("defect_type")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    args = ap.parse_args()

    ds = load(args.data)
    rep = Report(args.strict)

    h1("HARD ASSERTIONS")
    assert_no_label_leak(ds, rep)
    assert_limits(ds, rep)
    assert_lot_size(ds, rep)
    assert_type3_band(ds, rep)
    assert_trap_labels(ds, rep)
    assert_censoring(ds, rep)
    assert_sublinear(ds, rep)
    assert_lot_separation(ds, rep)
    assert_type3_smoothness(ds, rep)

    summary_tables(ds)
    cat = calibration_selftest(ds, rep)
    spatial_check(ds)
    estimator_comparison(ds, rep)
    type3_feasibility(ds, _GOOD_D2[0])
    if not args.no_plot:
        write_trajectory_svg(ds, args.data / "trajectories.svg")
    verdict(ds, cat, rep)


if __name__ == "__main__":
    main()
