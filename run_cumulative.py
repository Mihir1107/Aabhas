"""
Cumulative ablation (C0..C4) and ensemble-fusion comparison.

The head-to-head table answers "which single method is best". This one answers
"what does each rung ADD", which is what an ablation is for. Each rung contains
everything below it, fused at score level.

Fusion note: rungs are combined by max robust-z against the good reference, not
by a binary OR. A binary OR collapses continuous scores into a near-binary
signal, which is fine for recall at one operating point and destroys PR-AUC,
because PR-AUC integrates over the whole ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modulea import evaluation as ev
from modulea import multivariate as mv
from modulea import plots as pl

OUT = Path("results")
YL = ev.YIELD_TARGET

# what each cumulative rung contains
RUNG_MEMBERS = {
    "C0": ["L0"],
    "C1": ["L0", "L1_dynamic_MAD"],
    "C2": ["L0", "L1_dynamic_MAD", "L2_both"],
    "C3": ["L0", "L1_dynamic_MAD", "L2_both", "L3a_MCD", "L3b_T2", "L3b_Q", "L3c_kNN"],
    "C4": ["L0", "L1_dynamic_MAD", "L2_both", "L3a_MCD", "L3b_T2", "L3b_Q", "L3c_kNN",
           "L4a_IForest", "L4b_LOF", "L4d_AutoEnc"],
}
RUNG_LABEL = {
    "C0": "L0 alone (static datasheet limits)",
    "C1": "L0 + L1 (DPAT)",
    "C2": "L0 + L1 + L2 (trajectory)",
    "C3": "L0 + L1 + L2 + L3 (robust multivariate)",
    "C4": "L0 + L1 + L2 + L3 + L4 (unsupervised ML)",
}


def main() -> None:
    ds = ev.load("data")
    y, ty, sev, lot = ds.labels(), ds.type_of(), ds.severity_of(), ds.lot_of()
    sp = ev.lot_splits(sorted(set(lot)))
    te, va = sp.mask(lot, "test"), sp.mask(lot, "val")
    good = ~y
    scores = pd.read_csv(OUT / "scores.csv.gz", index_col=0)
    flags = pd.read_csv(OUT / "flags.csv.gz", index_col=0)
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # L0 is binary; give it a score that is 0 for everything in spec so it can
    # participate in a score-level fusion without inventing a ranking.
    scores["L0"] = flags["L0"].astype(float) * 1e6

    rows, curves = [], {}
    for rung, members in RUNG_MEMBERS.items():
        sub = {k: scores[k] for k in members if k in scores.columns}
        if len(sub) == 1:
            s = list(sub.values())[0]
        else:
            s = mv.fuse(sub, "max_z", good_mask=good)
        if rung == "C0":
            m = ev.metrics_from_flags(y[te], flags["L0"][te].astype(bool))
            rows.append({"rung": rung, "contents": RUNG_LABEL[rung],
                         "recall@93%yield": float("nan"), "PR_AUC": float("nan"),
                         "AUROC": float("nan"),
                         "escape_rate_%": m["escape_rate_%"],
                         "yield_loss_%": m["yield_loss_%"], "cost": m["cost_at_op"]})
            continue
        m = ev.metrics_from_score(y[te], s[te], YL)
        rows.append({"rung": rung, "contents": RUNG_LABEL[rung],
                     "recall@93%yield": m["recall@93%yield"], "PR_AUC": m["PR_AUC"],
                     "AUROC": m["AUROC"], "escape_rate_%": m["escape_rate_%"],
                     "yield_loss_%": m["yield_loss_%"], "cost": m["cost_at_op"],
                     "recall@1%YL": m["recall@1%YL"]})
        yl, rec, _ = ev.recall_yield_curve(y[te], s[te])
        curves[rung] = (yl, rec)
        scores[f"CUM_{rung}"] = s
    cum = pd.DataFrame(rows)
    cum.to_csv(OUT / "ablation_cumulative.csv", index=False)

    # ---------------- fusion comparison ----------------
    members = {k: scores[k] for k in
               ("L2_both", "L3a_MCD", "L3b_Q", "L4a_IForest", "L4d_AutoEnc")}
    frows = []
    binf = mv.union_ensemble(members, good & te, YL / len(members))
    mb = ev.metrics_from_flags(y[te], binf[te])
    frows.append({"fusion": "binary OR (decision level)",
                  "recall@93%yield": mb["recall_%"], "PR_AUC": float("nan"),
                  "AUROC": float("nan"), "yield_loss_%": mb["yield_loss_%"],
                  "escape_rate_%": mb["escape_rate_%"], "cost": mb["cost_at_op"]})
    for how in ("max_rank", "mean_rank", "max_z", "weighted_mean_z"):
        s = mv.fuse(members, how, good_mask=good)
        m = ev.metrics_from_score(y[te], s[te], YL)
        frows.append({"fusion": how + " (score level)",
                      "recall@93%yield": m["recall@93%yield"], "PR_AUC": m["PR_AUC"],
                      "AUROC": m["AUROC"], "yield_loss_%": m["yield_loss_%"],
                      "escape_rate_%": m["escape_rate_%"], "cost": m["cost_at_op"]})
        scores[f"FUSE_{how}"] = s
    for k in ("L4d_AutoEnc", "L4b_LOF"):
        m = ev.metrics_from_score(y[te], scores[k][te], YL)
        frows.append({"fusion": f"(best single member: {k})",
                      "recall@93%yield": m["recall@93%yield"], "PR_AUC": m["PR_AUC"],
                      "AUROC": m["AUROC"], "yield_loss_%": m["yield_loss_%"],
                      "escape_rate_%": m["escape_rate_%"], "cost": m["cost_at_op"]})
    fus = pd.DataFrame(frows)
    fus.to_csv(OUT / "fusion_comparison.csv", index=False)

    # ---------------- per type / tier for the cumulative rungs ------------
    prows = []
    for rung in RUNG_MEMBERS:
        key = "L0" if rung == "C0" else f"CUM_{rung}"
        if rung == "C0":
            f = flags["L0"].astype(bool)
        else:
            thr = ev.threshold_at_yield_loss(y[te], scores[key][te], YL)
            f = scores[key] >= thr
        for scope, mask in (("test lots", te),
                            ("all lots", pd.Series(True, index=y.index))):
            for t in sorted(pd.unique(ty)):
                for tier in ("(all)", "severe", "moderate", "mild"):
                    sel = (ty == t) & mask
                    if tier != "(all)":
                        if t not in ("III_CENTRE_HIDER", "IV_CORRELATION_BREAK"):
                            continue
                        sel = sel & (sev == tier)
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    prows.append({"scope": scope, "rung": rung, "defect_type": t,
                                  "severity": tier, "n": n,
                                  "flagged_%": 100 * float(f[sel].mean())})
    pd.DataFrame(prows).to_csv(OUT / "per_type_cumulative.csv", index=False)
    np.savez(OUT / "curves_cumulative.npz",
             **{k: np.vstack(v) for k, v in curves.items()})
    scores.to_csv(OUT / "scores.csv.gz", compression="gzip")

    print(cum.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(fus.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
