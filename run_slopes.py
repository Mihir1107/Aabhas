"""Safety-slope comparison, all four definitions, reusing saved Module B output."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from modulea import evaluation as ev, moduleb as mb, features as ft, drift_models as dm
OUT = Path("results")

def main():
    ds = ev.load("data"); y = ds.labels(); lot = ds.lot_of()
    sp = ev.lot_splits(sorted(set(lot))); te = sp.mask(lot, "test"); tr = sp.mask(lot, "train")
    X = mb.build_features(ds, "early")
    ub = pd.read_csv(OUT / "moduleb_upper_early.csv.gz", index_col=0)
    wide = ft.wide_frame(ds)
    healthy = ~y
    rows = []
    flags = {}
    for p in ds.params:
        pw = dm.PowerLaw(p); beta = pw.fit_beta(wide, lot, ds.checkpoints, X.index[tr])
        sa = mb.slope_margin_consumption(ds, X, p)
        sb = mb.slope_lot_derived(ds, X, p, healthy)
        vm = mb.project_to_mission(ds, X, p, beta)
        lsafe = mb.lot_safe_limit(ds, p)
        u = ub[p].reindex(X.index)
        f = {
            "a_margin": (X[f"{p}__s1"] > sa).fillna(False),
            "b_lot_slope": (X[f"{p}__s1"] > sb).fillna(False),
            "c_mission": (vm > ds.limits[p][1]).fillna(False),
            "d_upper_datasheet": mb.flag_confidence_adjusted(u, ds, p).fillna(False),
            "d_upper_lotsafe": mb.flag_confidence_adjusted(u, ds, p, l_safe=lsafe).fillna(False),
        }
        for k, v in f.items():
            flags.setdefault(k, pd.Series(False, index=X.index))
            flags[k] |= v
        rows.append({"parameter": p, "beta": beta,
                     **{f"{k}_flag_%": 100 * float(v[te].mean()) for k, v in f.items()}})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== union across parameters, TEST lots ===")
    out = []
    for k, v in flags.items():
        m = ev.metrics_from_flags(y[te], v[te])
        out.append({"rule": k, "recall_%": m["recall_%"], "yield_loss_%": m["yield_loss_%"],
                    "escape_rate_%": m["escape_rate_%"], "precision_%": m["precision_%"],
                    "cost": m["cost_at_op"]})
    df = pd.DataFrame(out); print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(OUT / "safety_slopes_union.csv", index=False)
    b, d = flags["b_lot_slope"], flags["d_upper_lotsafe"]
    print(f"\n(b) vs (d) disagreement on TEST: both {100*(b&d)[te].mean():.2f}%  "
          f"b only {100*(b&~d)[te].mean():.2f}%  d only {100*(d&b.eq(False))[te].mean():.2f}%")
    pd.DataFrame(flags).to_csv(OUT / "safety_slope_flags.csv")
    print("\nagreement on DEFECTIVE parts (test):")
    for k, v in flags.items():
        print(f"  {k:20s} recall {100*v[te&y].mean():6.2f}%")

if __name__ == "__main__":
    main()
