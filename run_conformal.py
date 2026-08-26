"""L5: conformal risk control, applied to Module A and Module B."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from modulea import conformal as cf, evaluation as ev, moduleb as mb, plots as pl

OUT = Path("results")
ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
N_REPEATS = 40


def main():
    ds = ev.load("data"); y = ds.labels(); lot = ds.lot_of()
    sp = ev.lot_splits(sorted(set(lot)))
    # Calibration and evaluation both live in the VAL+TEST lots. Train lots are
    # excluded because the detectors were fitted on them, and a part the model
    # was fitted on is not exchangeable with one it has never seen.
    eligible = y.index[lot.isin(sp.val + sp.test)]
    scores = pd.read_csv(OUT / "scores.csv.gz", index_col=0)

    ub = pd.read_csv(OUT / "moduleb_upper_early.csv.gz", index_col=0)
    lsafe = {p: mb.lot_safe_limit(ds, p) for p in ds.params}
    marg = pd.concat([(ub[p] - lsafe[p].reindex(ub.index))
                      / max(abs(ds.limits[p][1] - ds.limits[p][0]), 1e-9)
                      for p in ds.params], axis=1).max(axis=1)

    targets = {
        "ModuleA_LOF": scores["L4b_LOF"],
        "ModuleA_fused_maxz": scores["CUM_C4"],
        "ModuleB_upper_bound_margin": marg,
    }
    allr = []
    for name, s in targets.items():
        s = s.replace([np.inf, -np.inf], np.nan).fillna(s.min())
        df = cf.sweep(s, y, lot, eligible, ALPHAS, n_repeats=N_REPEATS, seed=0)
        df["target"] = name
        allr.append(df)
    res = pd.concat(allr, ignore_index=True)
    res.to_csv(OUT / "conformal_sweep.csv", index=False)

    summ = res.groupby(["target", "alpha"]).agg(
        emp_FNR_mean=("empirical_FNR", "mean"),
        emp_FNR_p10=("empirical_FNR", lambda a: np.quantile(a, 0.10)),
        emp_FNR_p90=("empirical_FNR", lambda a: np.quantile(a, 0.90)),
        recall_mean=("recall_%", "mean"),
        yield_loss_mean=("yield_loss_%", "mean"),
        yield_loss_p90=("yield_loss_%", lambda a: np.quantile(a, 0.90)),
        n_cal_def=("n_cal_defective", "mean")).reset_index()
    summ["holds"] = summ["emp_FNR_mean"] <= summ["alpha"] + 1e-12
    summ.to_csv(OUT / "conformal_summary.csv", index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- the guarantee plot
    ser = []
    for i, name in enumerate(targets):
        d = summ[summ.target == name]
        ser.append({"x": d["alpha"] * 100, "y": d["emp_FNR_mean"] * 100,
                    "label": name})
        ser.append({"x": d["alpha"] * 100, "y": d["emp_FNR_p90"] * 100,
                    "label": f"{name} p90", "dash": "4,3"})
    ser.append({"x": np.array(ALPHAS) * 100, "y": np.array(ALPHAS) * 100,
                "label": "y = x (the guarantee)", "color": "#333"})
    pl.line_chart(OUT / "fig8_conformal_guarantee.svg", ser,
                  "Conformal risk control: empirical FNR vs the alpha calibrated for",
                  "alpha: target bound on expected false-negative rate (%)",
                  "Empirical FNR on held-out lots (%)",
                  xlim=(0, 21), ylim=(0, 25), legend_title="Detector",
                  )
    ser2 = []
    for name in targets:
        d = summ[summ.target == name]
        ser2.append({"x": d["emp_FNR_mean"] * 100, "y": d["yield_loss_mean"],
                     "label": name})
    pl.line_chart(OUT / "fig9_conformal_tradeoff.svg", ser2,
                  "The honest counterpart: yield loss at the guaranteed escape rate",
                  "Guaranteed escape rate (empirical FNR, %)",
                  "Yield loss: good parts sacrificed (%)",
                  xlim=(0, 21), ylim=(0, 100), legend_title="Detector")
    print("\nwrote conformal_sweep.csv, conformal_summary.csv, fig8, fig9")


if __name__ == "__main__":
    main()
