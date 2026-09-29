"""Every number the documentation site quotes, read from the committed outputs.

Nothing in the site is typed by hand. The v1.0 -> v1.1 history of this project
is a history of hand-copied numbers drifting away from the code that produced
them, so the docs are generated: re-run the pipeline, re-run make_docs.py, and
every figure on every page moves with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
D = ROOT / "data"

TYPE_ORDER = ["I_STEEP_DRIFTER", "II_STEP_DEFECT", "III_CENTRE_HIDER",
              "IV_CORRELATION_BREAK", "Vb_EXTREME_LEVEL", "Va_MILDLY_HIGH_STABLE",
              "VI_LOT_SHIFT", "VII_FIXTURE_ARTIFACT", "GOOD"]
TYPE_SHORT = {"I_STEEP_DRIFTER": "I", "II_STEP_DEFECT": "II",
              "III_CENTRE_HIDER": "III", "IV_CORRELATION_BREAK": "IV",
              "Vb_EXTREME_LEVEL": "Vb", "Va_MILDLY_HIGH_STABLE": "Va",
              "VI_LOT_SHIFT": "VI", "VII_FIXTURE_ARTIFACT": "VII", "GOOD": "Good"}
TYPE_NAME = {"I_STEEP_DRIFTER": "Steep drifter", "II_STEP_DEFECT": "Step defect",
             "III_CENTRE_HIDER": "Centre-hider", "IV_CORRELATION_BREAK": "Correlation break",
             "Vb_EXTREME_LEVEL": "Extreme level, in spec",
             "Va_MILDLY_HIGH_STABLE": "Mildly high, stable",
             "VI_LOT_SHIFT": "Whole-lot process shift",
             "VII_FIXTURE_ARTIFACT": "Fixture artifact", "GOOD": "Good part"}


def _csv(name, **kw):
    return pd.read_csv(R / name, **kw)


def load() -> dict:
    N: dict = {}
    cfg = json.loads((D / "config.json").read_text(encoding="utf-8"))
    c = cfg["config"]
    N["version"] = cfg.get("version", "dataset")
    N["n_lots"], N["ppl"] = c["n_lots"], c["parts_per_lot"]
    N["n_parts"] = c["n_lots"] * c["parts_per_lot"]
    N["af"] = cfg["derived"]["acceleration_factor"]
    N["fy168"] = cfg["derived"]["field_years_by_checkpoint"]["168.0"]
    N["ea"], N["t_stress"], N["t_use"] = c["ea_ev"], c["t_stress_c"], c["t_use_c"]
    N["corr"] = c["leakage_corr_target"]
    N["band"] = c["type3_band_pct"]
    N["params"] = cfg["parameters"]
    N["type6_lots"] = cfg["derived"]["type6_lots"]

    gt = pd.read_csv(D / "ground_truth.csv")
    vc = gt["defect_type"].value_counts()
    N["type_counts"] = {t: int(vc.get(t, 0)) for t in TYPE_ORDER}
    N["n_def"] = int(gt["is_defective"].sum())
    N["prev"] = 100 * N["n_def"] / N["n_parts"]
    N["n_pulled"] = int((gt["censor_status_168h"] == "PULLED_FAILED").sum())
    N["sev3"] = gt[gt.defect_type == "III_CENTRE_HIDER"]["severity"].value_counts().to_dict()

    N["split"] = _csv("split_counts.csv").set_index("split")

    # ---------------- Module A ----------------
    cum = _csv("ablation_cumulative.csv").set_index("rung")
    N["cum"] = cum
    N["abl"] = _csv("ablation.csv")
    am = _csv("all_methods.csv", index_col=0)
    N["am"] = am
    N["fusion"] = _csv("fusion_comparison.csv")
    N["iso"] = _csv("isoforest_feature_width.csv").set_index("feature_set")
    pt = _csv("per_type.csv")
    N["pt"] = pt
    ptc = _csv("per_type_cumulative.csv")
    N["ptc"] = ptc
    lolo = _csv("leave_one_lot_out.csv")
    g = lolo.groupby("method")["recall"]
    N["lolo"] = {k: (float(g.mean()[k]), float(g.std()[k]), int(g.count()[k]))
                 for k in g.mean().index}
    N["curated"] = json.loads((R / "c4_curated_members.json").read_text())

    flags = pd.read_csv(R / "flags.csv.gz", index_col=0)
    ty = gt.set_index("component_id")["defect_type"].reindex(flags.index)
    vi = ty == "VI_LOT_SHIFT"
    N["vi_static"] = int(flags.loc[vi, "L1_static_MAD"].sum())
    N["vi_dynamic"] = int(flags.loc[vi, "L1_dynamic_MAD"].sum())
    N["vi_n"] = int(vi.sum())
    rows = pt[(pt.scope == "all lots") & (pt.defect_type == "VI_LOT_SHIFT")].set_index("rung")
    N["vi_matched_static"] = float(rows.loc["L1a", "flagged_%"])
    N["vi_matched_dynamic"] = float(rows.loc["L1b", "flagged_%"])

    def ptype(rung, t, sev="(all)", scope="all lots"):
        r = pt[(pt.rung == rung) & (pt.defect_type == t) & (pt.severity == sev)
               & (pt.scope == scope)]
        return r.iloc[0] if len(r) else None
    N["ptype"] = ptype

    # ---------------- Module B ----------------
    mb = _csv("module_b.csv")
    N["mb"] = mb
    e = mb[mb["mode"] == "early"]
    N["mb_lad"] = e[e.model != "SURVIVORSHIP"].pivot_table(
        index="model", columns="parameter", values="MAE")
    N["mb_gd"] = e[e.model == "4_gbm_mae"].set_index("parameter")[
        ["MAE", "MAE_good", "MAE_defective"]]
    mid = mb[mb.model == "4_gbm_mae"].pivot_table(index="parameter", columns="mode",
                                                  values="MAE")
    N["mb_mid_gain"] = (100 * (mid["early"] - mid["mid"]) / mid["early"])
    cov = e[e.model.isin(["5a_lgbm_q95", "5b_quantile_forest"])]
    N["mb_cov"] = cov.pivot_table(index="parameter", columns="model",
                                  values="upper95_coverage")
    N["surv"] = e[e.model == "SURVIVORSHIP"].set_index("parameter")
    N["slopes"] = _csv("safety_slopes_union.csv").set_index("rule")
    N["prov"] = json.loads((R / "module_b_provenance.json").read_text())

    # ---------------- conformal ----------------
    cs = _csv("conformal_summary.csv")
    N["conf"] = cs
    sig = _csv("conformal_significance.csv")
    N["conf_sig"] = sig
    N["conf_min_p"] = float(sig["p_one_sided"].min())
    N["conf_max_se"] = float(sig["excess_in_SE"].max())
    N["conf_n_viol"] = int((sig["verdict"] != "holds").sum())
    sw = _csv("conformal_sweep.csv")
    f = sw[sw.target == "ModuleA_fused_maxz"]
    N["conf_split_over"] = (f.assign(o=f.empirical_FNR > f.alpha)
                            .groupby("alpha")["o"].mean() * 100)

    # ---------------- explainability / decision ----------------
    N["xm"] = json.loads((R / "explainability_metrics.json").read_text())
    N["tiers"] = pd.read_csv(R / "decision_tiers_test.csv", index_col=0)
    N["reports"] = _csv("report_examples.csv")
    txt = (ROOT / "reports" / "01_Vb_extreme_level_PS_worked_example.txt").read_text(encoding="utf-8")
    line = next(l for l in txt.splitlines() if l.startswith("Lot-relative level"))
    N["l1_example"] = line.split(":", 1)[1].strip()

    # ---------------- demo board for the hero ----------------
    dd = ROOT / "demo" / "inspector_data.json"
    N["demo"] = json.loads(dd.read_text(encoding="utf-8")) if dd.exists() else None
    return N
