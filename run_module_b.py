"""Module B: drift prediction ladder, upper bounds, safety slopes.

Reuses the Module A harness and lot-grouped splits. No second harness.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modulea import drift_models as dm
from modulea import evaluation as ev
from modulea import features as ft
from modulea import moduleb as mb
from modulea import plots as pl

OUT = Path("results")
try:
    import lightgbm  # noqa
    USE_LGB = True
except Exception:
    USE_LGB = False
try:
    import quantile_forest  # noqa
    USE_QF = True
except Exception:
    USE_QF = False


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mae(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return float(np.nanmean(d))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = ev.load("data")
    gt = ds.gt.set_index("component_id")
    y_def = ds.labels()
    ty, sev, lot = ds.type_of(), ds.severity_of(), ds.lot_of()
    sp = ev.lot_splits(sorted(set(lot)))
    tr, va, te = (sp.mask(lot, k) for k in ("train", "val", "test"))
    log(f"split: {sp.describe()}")
    log(f"LightGBM={USE_LGB}  quantile-forest={USE_QF}")

    obs, true = mb.targets(ds)
    censored = gt["censor_status_168h"].reindex(obs.index) == "PULLED_FAILED"
    survivor = obs[ds.params[0]].notna() & ~censored

    wide = ft.wide_frame(ds)
    results, preds, bounds = [], {}, {}

    for mode in ("early", "mid"):
        X = mb.build_features(ds, mode)
        mb.assert_no_leak(X, mode)
        for p in ds.params:
            cols = [c for c in X.columns if c.startswith(f"{p}__") or
                    c.startswith(f"z__{p}__")]
            Xp = X[cols]
            yt = true[p]

            # training target: survivors keep their observed value; a pulled
            # part is known to have breached the limit, so the datasheet limit
            # is used as a conservative OBSERVABLE surrogate. The latent value
            # is never used for training -- it is not observable in production.
            y_train_surv = obs[p]
            y_train_cens = obs[p].where(obs[p].notna(), ds.limits[p][1])

            fit_s = tr & y_train_surv.notna()
            fit_c = tr & y_train_cens.notna()
            ev_mask = te & yt.notna()

            models = {}
            lin = dm.LinearSlope(p)
            models["1_linear_slope"] = lin
            pw = dm.PowerLaw(p)
            pw.fit_beta(wide, lot, ds.checkpoints, X.index[tr])
            models["2_power_law"] = pw
            models["3_huber"] = dm.HuberModel()
            models["4_gbm_mae"] = dm.GBM(USE_LGB)

            for name, m in models.items():
                if name in ("3_huber", "4_gbm_mae"):
                    m.fit(Xp[fit_s], y_train_surv[fit_s].to_numpy())
                yhat = pd.Series(m.predict(Xp), index=Xp.index)
                preds[(mode, p, name)] = yhat
                r = {"mode": mode, "parameter": p, "model": name,
                     "MAE": mae(yhat[ev_mask], yt[ev_mask]),
                     "RMSE": float(np.sqrt(np.nanmean(
                         (yhat[ev_mask] - yt[ev_mask]) ** 2))),
                     "medAE": float(np.nanmedian(np.abs(
                         yhat[ev_mask] - yt[ev_mask]))),
                     "bias": float(np.nanmean(yhat[ev_mask] - yt[ev_mask])),
                     "MAE_good": mae(yhat[ev_mask & ~y_def], yt[ev_mask & ~y_def]),
                     "MAE_defective": mae(yhat[ev_mask & y_def], yt[ev_mask & y_def]),
                     "beta": getattr(m, "beta", np.nan)}
                results.append(r)

            # ---- quantile models for the 95% upper bound
            for qn, qm in (("5a_lgbm_q95", dm.GBM(USE_LGB, quantile=0.95)),
                           ("5b_quantile_forest", dm.QuantileForest(USE_QF))):
                qm.fit(Xp[fit_s], y_train_surv[fit_s].to_numpy())
                if qn.startswith("5a"):
                    ub = pd.Series(qm.predict(Xp), index=Xp.index)
                    pt = preds[(mode, p, "4_gbm_mae")]
                else:
                    ub = pd.Series(qm.predict_quantile(Xp, 0.95), index=Xp.index)
                    pt = pd.Series(qm.predict(Xp), index=Xp.index)
                bounds[(mode, p, qn)] = ub
                cov = float((yt[ev_mask] <= ub[ev_mask]).mean())
                width = float(np.nanmean(ub[ev_mask] - pt[ev_mask]))
                results.append({
                    "mode": mode, "parameter": p, "model": qn,
                    "MAE": mae(pt[ev_mask], yt[ev_mask]),
                    "RMSE": float(np.sqrt(np.nanmean((pt[ev_mask] - yt[ev_mask]) ** 2))),
                    "medAE": float(np.nanmedian(np.abs(pt[ev_mask] - yt[ev_mask]))),
                    "bias": float(np.nanmean(pt[ev_mask] - yt[ev_mask])),
                    "MAE_good": mae(pt[ev_mask & ~y_def], yt[ev_mask & ~y_def]),
                    "MAE_defective": mae(pt[ev_mask & y_def], yt[ev_mask & y_def]),
                    "upper95_coverage": cov, "mean_upper_width": width,
                    "beta": np.nan})

            # ---- survivorship bias, in absolute units
            if mode == "early":
                m_s = dm.GBM(USE_LGB).fit(Xp[fit_s], y_train_surv[fit_s].to_numpy())
                m_c = dm.GBM(USE_LGB).fit(Xp[fit_c], y_train_cens[fit_c].to_numpy())
                cen_te = te & censored & yt.notna()
                ps = pd.Series(m_s.predict(Xp), index=Xp.index)
                pc = pd.Series(m_c.predict(Xp), index=Xp.index)
                results.append({
                    "mode": "early", "parameter": p, "model": "SURVIVORSHIP",
                    "MAE": mae(ps[ev_mask], yt[ev_mask]),
                    "MAE_censored_survfit": mae(ps[cen_te], yt[cen_te]),
                    "MAE_censored_censfit": mae(pc[cen_te], yt[cen_te]),
                    "bias_on_censored_survfit": float(
                        np.nanmean(ps[cen_te] - yt[cen_te])),
                    "n_censored_test": int(cen_te.sum()),
                    "limit_hi": ds.limits[p][1]})
        log(f"mode={mode} done")

    res = pd.DataFrame(results)
    res.to_csv(OUT / "module_b.csv", index=False)

    # ---------------- per-type MAE, early mode, best point model -----------
    rows = []
    for p in ds.params:
        yhat = preds[("early", p, "4_gbm_mae")]
        m = te & true[p].notna()
        for t in sorted(pd.unique(ty)):
            for tier in ("(all)", "severe", "moderate", "mild"):
                sel = m & (ty == t)
                if tier != "(all)":
                    if t not in ("III_CENTRE_HIDER", "IV_CORRELATION_BREAK"):
                        continue
                    sel = sel & (sev == tier)
                if sel.sum() == 0:
                    continue
                rows.append({"parameter": p, "defect_type": t, "severity": tier,
                             "n": int(sel.sum()),
                             "MAE": mae(yhat[sel], true[p][sel])})
    pd.DataFrame(rows).to_csv(OUT / "module_b_per_type.csv", index=False)

    # ---------------- MAE by lot ----------------
    lrows = []
    for p in ds.params:
        yhat = preds[("early", p, "4_gbm_mae")]
        m = te & true[p].notna()
        for lt, g in yhat[m].groupby(lot[m]):
            lrows.append({"parameter": p, "lot": lt, "n": len(g),
                          "MAE": mae(g, true[p].reindex(g.index))})
    lots_df = pd.DataFrame(lrows)
    lots_df.to_csv(OUT / "module_b_by_lot.csv", index=False)

    # ---------------- safety slopes ----------------
    Xe = mb.build_features(ds, "early")
    healthy = ~y_def
    srows = []
    for p in ds.params:
        sa = mb.slope_margin_consumption(ds, Xe, p)
        sb = mb.slope_lot_derived(ds, Xe, p, healthy)
        sc = mb.slope_mission_based(ds, p)
        ub = bounds[("early", p, "5a_lgbm_q95")]
        fd = mb.flag_confidence_adjusted(ub, ds, p)
        fb = Xe[f"{p}__s1"] > sb
        fa = Xe[f"{p}__s1"] > sa
        fc = Xe[f"{p}__s1"] > sc
        srows.append({"parameter": p, "mission_slope_c": sc,
                      "flag_a_margin_%": 100 * float(fa[te].mean()),
                      "flag_b_lot_%": 100 * float(fb[te].mean()),
                      "flag_c_mission_%": 100 * float(fc[te].mean()),
                      "flag_d_upper_%": 100 * float(fd[te].mean()),
                      "b_and_d_%": 100 * float((fb & fd)[te].mean()),
                      "b_not_d_%": 100 * float((fb & ~fd)[te].mean()),
                      "d_not_b_%": 100 * float((fd & ~fb)[te].mean()),
                      "b_recall_%": 100 * float(fb[te & y_def].mean()),
                      "d_recall_%": 100 * float(fd[te & y_def].mean()),
                      "b_yield_loss_%": 100 * float(fb[te & ~y_def].mean()),
                      "d_yield_loss_%": 100 * float(fd[te & ~y_def].mean())})
    slopes = pd.DataFrame(srows)
    slopes.to_csv(OUT / "safety_slopes.csv", index=False)

    # union across parameters for the two implemented rules
    fb_any = pd.Series(False, index=Xe.index)
    fd_any = pd.Series(False, index=Xe.index)
    for p in ds.params:
        sb = mb.slope_lot_derived(ds, Xe, p, healthy)
        fb_any |= (Xe[f"{p}__s1"] > sb).fillna(False)
        fd_any |= mb.flag_confidence_adjusted(
            bounds[("early", p, "5a_lgbm_q95")], ds, p).fillna(False)
    union = pd.DataFrame({"b_lot_derived": fb_any, "d_upper_bound": fd_any})
    union.to_csv(OUT / "safety_slope_flags.csv")
    for nm, f in (("b (lot-derived, union)", fb_any), ("d (upper bound, union)", fd_any)):
        m = ev.metrics_from_flags(y_def[te], f[te])
        log(f"{nm}: recall {m['recall_%']:.2f}%  yield loss {m['yield_loss_%']:.2f}%  "
            f"escape {m['escape_rate_%']:.2f}%")

    # save predictions for the conformal stage
    pd.DataFrame({f"{p}": preds[("early", p, "4_gbm_mae")] for p in ds.params}
                 ).to_csv(OUT / "moduleb_point_early.csv.gz", compression="gzip")
    pd.DataFrame({f"{p}": bounds[("early", p, "5a_lgbm_q95")] for p in ds.params}
                 ).to_csv(OUT / "moduleb_upper_early.csv.gz", compression="gzip")
    true.to_csv(OUT / "moduleb_true168.csv.gz", compression="gzip")

    log("done")
    e = res[(res["mode"] == "early") & (res["model"] != "SURVIVORSHIP")]
    print(e.pivot_table(index="model", columns="parameter", values="MAE")
          .to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
