"""The four quantitative explainability metrics, plus the report examples."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from explain.counterfactual import CounterfactualEngine
from explain.decision import DecisionPolicy
from explain.evidence import Evidence
from explain.rules import PartExplanation

OUT = Path("results")

# What primary reason SHOULD a correct explanation give, per injected archetype?
# Ground truth carries the type, so this is measurable rather than a matter of
# taste. More than one answer can be right: a steep drifter whose forecast also
# breaches is correctly explained either way.
EXPECTED = {
    "I_STEEP_DRIFTER": {"DRIFT_RATE", "FORECAST"},
    "II_STEP_DEFECT": {"STEP"},
    "III_CENTRE_HIDER": {"JOINT"},
    "IV_CORRELATION_BREAK": {"JOINT"},
    "Vb_EXTREME_LEVEL": {"LEVEL"},
    "VII_FIXTURE_ARTIFACT": {"SPATIAL"},
}
TRAPS_SHOULD_PASS = {"Va_MILDLY_HIGH_STABLE", "VI_LOT_SHIFT"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    e = Evidence()
    pol = DecisionPolicy(e)
    pool = (e.scores["CUM_C4"] >= pol.thr_review["CUM_C4"])
    cf = CounterfactualEngine(e, pol)
    rng = np.random.default_rng(0)

    # sample flagged parts per type, from TEST lots where possible
    te = e.lot.isin(e.split.test)
    rows = []
    log("building explanations")
    for t in sorted(set(e.ty)):
        # Traps and GOOD parts are sampled WITHOUT conditioning on the flag
        # pool, because the question for them is "does the system correctly
        # decline to reject this?" -- sampling only the flagged ones would
        # answer a different question and always return 0% PASS.
        cond = pool if t in EXPECTED else pd.Series(True, index=pool.index)
        ids_te = e.gt.index[(e.ty == t) & te & cond]
        ids_all = e.gt.index[(e.ty == t) & cond]
        ids = ids_te if len(ids_te) >= 25 else ids_all
        if len(ids) == 0:
            ids = e.gt.index[(e.ty == t) & te][:25]
        n = min(60, len(ids))
        if n == 0:
            continue
        sel = rng.choice(np.asarray(ids), size=n, replace=False)
        for cid in sel:
            x = PartExplanation(e, pol, cid, pool)
            c = cf.find(cid)
            rows.append({
                "component_id": cid, "defect_type": t,
                "severity": e.sev[cid], "in_test": bool(te[cid]),
                "tier": x.tier, "primary": x.primary,
                "secondary": x.secondary or "",
                "n_evidence_lines": len(x.lines),
                "has_mechanism": len(x.hypotheses) > 0,
                "mech_confidence": x.hypotheses[0]["confidence"] if x.hypotheses else "",
                "cf_found": c is not None,
                "cf_validated": bool(c["validated"]) if c else False,
                "flagged": bool(pool[cid]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "explainability_parts.csv", index=False)

    # ---------------- 1. reason correctness ----------------
    rc = []
    for t, exp in EXPECTED.items():
        s = df[(df.defect_type == t) & df.flagged]
        if not len(s):
            continue
        ok = s.primary.isin(exp)
        ok2 = ok | s.secondary.isin(exp)
        rc.append({"defect_type": t, "n": len(s), "expected": "/".join(sorted(exp)),
                   "primary_correct_%": 100 * float(ok.mean()),
                   "primary_or_secondary_%": 100 * float(ok2.mean()),
                   "top_wrong_reason": (s.loc[~ok, "primary"].mode().iat[0]
                                        if (~ok).any() else "")})
    for t in sorted(TRAPS_SHOULD_PASS | {"GOOD"}):
        s = df[df.defect_type == t]
        if not len(s):
            continue
        not_rejected = s.tier.isin(["PASS", "WATCH", "FIXTURE_SUSPECT"])
        rc.append({"defect_type": t, "n": len(s),
                   "expected": "PASS (trap)" if t in TRAPS_SHOULD_PASS else "PASS",
                   "primary_correct_%": 100 * float((s.tier == "PASS").mean()),
                   "primary_or_secondary_%": 100 * float(not_rejected.mean()),
                   "top_wrong_reason": (s.loc[~not_rejected, "primary"].mode().iat[0]
                                        if (~not_rejected).any() else "")})
    # Type III is 500 DPPM by design, so only a handful are ever flagged and its
    # own row is not quotable alone (n = 1 on the frozen run). III and IV share
    # the same expected reason, so the pooled JOINT class is the defensible
    # aggregate and is emitted explicitly rather than left for a reader to
    # compute.
    _j = [r for r in rc if r["defect_type"] in
          ("III_CENTRE_HIDER", "IV_CORRELATION_BREAK")]
    if len(_j) == 2:
        n_j = sum(r["n"] for r in _j)
        rc.append({
            "defect_type": "JOINT_POOLED_III_IV", "n": n_j, "expected": "JOINT",
            "primary_correct_%": 100 * sum(
                r["n"] * r["primary_correct_%"] / 100 for r in _j) / n_j,
            "primary_or_secondary_%": 100 * sum(
                r["n"] * r["primary_or_secondary_%"] / 100 for r in _j) / n_j,
            "top_wrong_reason": "(pooled -- quote this, not Type III alone)"})
    reason = pd.DataFrame(rc)

    # ---------------- 2. completeness ----------------
    # Completeness means different things for a flag and for a pass. A flagged
    # part needs a primary reason and a mechanism hypothesis; a PASS part needs
    # the evidence lines and an action, and "no rule fired" IS its complete
    # explanation. Scoring them together would penalise correct passes.
    fl_m = df.flagged
    comp = {
        "n_explained": len(df),
        "n_flagged": int(fl_m.sum()),
        "flagged_complete_%": 100 * float(
            ((df.n_evidence_lines >= 9) & (df.primary != "NONE")
             & df.has_mechanism)[fl_m].mean()),
        "passed_complete_%": 100 * float(
            (df.n_evidence_lines >= 9)[~fl_m].mean()),
        "all_have_full_evidence_%": 100 * float((df.n_evidence_lines >= 9).mean()),
        "flagged_with_primary_%": 100 * float((df.loc[fl_m, "primary"] != "NONE").mean()),
        "flagged_with_mechanism_%": 100 * float(df.loc[fl_m, "has_mechanism"].mean()),
    }

    # ---------------- 3. counterfactual validity ----------------
    fl = df[df.flagged]
    uni = fl[fl.primary.isin(["LEVEL", "DRIFT_RATE", "STEP"])]
    multi = fl[fl.primary == "JOINT"]
    cfm = {"n_flagged": len(fl),
           "cf_found_%": 100 * float(fl.cf_found.mean()),
           "cf_validated_%": 100 * float(fl.cf_validated.mean()),
           "cf_validated_given_found_%": (
               100 * float(fl.loc[fl.cf_found, "cf_validated"].mean())
               if fl.cf_found.any() else np.nan),
           "cf_found_univariate_driven_%": (100 * float(uni.cf_found.mean())
                                            if len(uni) else np.nan),
           "cf_found_joint_driven_%": (100 * float(multi.cf_found.mean())
                                       if len(multi) else np.nan),
           "n_univariate_driven": len(uni), "n_joint_driven": len(multi)}

    # ---------------- 4. stability under measurement noise ----------------
    log("stability: perturbing at the generator's own noise magnitude")
    sd_eps = {q["name"]: (q["v0_mean"] * q["v0_sd_part"] if q.get("log_scale")
                          else q["v0_sd_part"]) * e.ds.cfg["config"]["meas_noise_frac_of_part_sd"]
              for q in e.ds.cfg["parameters"]}
    sample = df.sample(min(120, len(df)), random_state=0)["component_id"].tolist()
    base = {cid: PartExplanation(e, pol, cid, pool).primary for cid in sample}
    changes, n_rep = [], 4
    for rep in range(n_rep):
        e2 = Evidence()
        r2 = np.random.default_rng(100 + rep)
        m = e2.ds.meas
        for p in e2.params:
            noise = r2.normal(0, sd_eps[p], len(m))
            m.loc[m["measurement_status"] == "MEASURED", p] += noise[
                (m["measurement_status"] == "MEASURED").to_numpy()]
        from modulea import features as ft
        e2.wide = ft.wide_frame(e2.ds)
        e2.traj, e2.ztraj, e2.lvl, e2.zlvl = ft.build_feature_matrix(e2.ds)
        e2._lot_stats, e2._maha = {}, {}
        pol2 = DecisionPolicy(e2)
        for cid in sample:
            try:
                p2 = PartExplanation(e2, pol2, cid, pool).primary
            except Exception:
                continue
            changes.append({"repeat": rep, "component_id": cid,
                            "base": base[cid], "perturbed": p2,
                            "changed": p2 != base[cid]})
        log(f"  stability repeat {rep + 1}/{n_rep}")
    st = pd.DataFrame(changes)
    st.to_csv(OUT / "explainability_stability.csv", index=False)
    stab = {"n_parts": len(sample), "n_repeats": n_rep,
            "primary_reason_changed_%": 100 * float(st.changed.mean()),
            "stable_%": 100 * float(1 - st.changed.mean())}
    by_type = (st.merge(df[["component_id", "defect_type"]], on="component_id")
               .groupby("defect_type")["changed"].mean().mul(100).round(1))

    # ---------------- the decision layer itself, on every test part ------
    # The ladder reports recall at a 7% yield-loss budget; the six-tier policy
    # an inspector actually sees runs at 2% (REVIEW) and 7% (WATCH). This is
    # the table that shows what the deployed tiers do to each archetype, and
    # it is where v1.1's Type III gap (1 of 60 reached REVIEW) was found.
    log("decision layer: tier of every test-lot part")
    te_ids = e.gt.index[te.to_numpy()]
    tiers = pd.Series({c: pol.decide(c, pool)["tier"] for c in te_ids}, name="tier")
    tab = pd.crosstab(e.ty[tiers.index], tiers)
    tab.to_csv(OUT / "decision_tiers_test.csv")
    yv = e.y[tiers.index]
    held = tiers.isin(["REVIEW", "REJECT"])
    retest = tiers.isin(["MEASUREMENT_INVALID", "FIXTURE_SUSPECT"])
    dec = {"n_test_parts": int(len(tiers)),
           "defects_held_%": 100 * float(held[yv].mean()),
           "defects_retest_%": 100 * float(retest[yv].mean()),
           "defects_released_%": 100 * float((~held & ~retest)[yv].mean()),
           "defects_fixture_suspect": int((tiers[yv] == "FIXTURE_SUSPECT").sum()),
           "good_held_%": 100 * float(held[~yv].mean()),
           "good_not_pass_%": 100 * float((tiers[~yv] != "PASS").mean())}
    s_all = e.scores
    rev = ((s_all[pol.fused_col] >= pol.thr_review[pol.fused_col])
           | (s_all[pol.joint_col] >= pol.thr_review[pol.joint_col]))
    watch = s_all[pol.fused_col] >= pol.thr_watch[pol.fused_col]
    for t in ("III_CENTRE_HIDER", "IV_CORRELATION_BREAK"):
        k = t.split("_")[0]
        dec[f"type{k}_review_all_lots_%"] = 100 * float(rev[e.ty == t].mean())
        dec[f"type{k}_review_or_watch_all_lots_%"] = 100 * float(
            (rev | watch)[e.ty == t].mean())
    print(json.dumps(dec, indent=2))

    # ---------------- Layer 2: do SHAP and Huber agree? ----------------
    # Backs the "two attribution methods disagree" finding. Before this it had
    # no generator in the repository, and its quoted baseline MAE (0.6102) was
    # the sklearn-fallback figure rather than the LightGBM one.
    log("Layer 2: SHAP (LightGBM) versus Huber coefficients, iddq_ua")
    from scipy import stats as _st
    from explain.attribution import DriftAttribution
    from modulea import drift_models as dm
    p_attr = "iddq_ua"
    da = DriftAttribution(e, p_attr).fit()
    mae_full = da.published_mae(0.0)
    cmp_ = da.compare(n_parts=400, seed=0)
    rho, pv = _st.spearmanr(cmp_["rank_shap"], cmp_["rank_huber"])
    tr_m = e.lot.isin(e.split.train).to_numpy()
    te_m = e.lot.isin(e.split.test).to_numpy()
    v0, v24 = e.Xb[f"{p_attr}__v0"], e.Xb[f"{p_attr}__v24"]
    ok = (np.isfinite(v0) & np.isfinite(v24)).to_numpy() & te_m
    corr = float(np.corrcoef(v0[ok], v24[ok])[0, 1])
    drop = [c for c in da.cols if c in (f"{p_attr}__v0", f"z__{p_attr}__v0")]
    keep = [c for c in da.cols if c not in drop]
    yt = e.mb_true[p_attr].to_numpy()
    g2 = dm.GBM(True).fit(e.Xb[keep][da._fit_mask], yt[da._fit_mask])
    pr = g2.predict(e.Xb[keep])
    mm = te_m & np.isfinite(yt)
    mae_drop = float(np.nanmean(np.abs(pr[mm] - yt[mm])))
    attr = {"parameter": p_attr, "n_features": len(cmp_),
            "spearman_rho_shap_vs_huber": float(rho), "p_value": float(pv),
            "corr_v0_v24_test": corr, "mae_full": mae_full,
            "mae_without_v0": mae_drop, "dropped": drop,
            "top3_shap": cmp_.sort_values("rank_shap").index[:3].tolist(),
            "top3_huber": cmp_.sort_values("rank_huber").index[:3].tolist()}
    cmp_.to_csv(OUT / "attribution_comparison.csv")
    print(json.dumps(attr, indent=2))

    stab["scope"] = ("measurements re-drawn at the generator's noise level and "
                     "the RULE layer recomputed; detector scores, the flag pool "
                     "and the tier are held at their original values")
    json.dump({"reason_correctness": reason.to_dict("records"),
               "completeness": comp, "counterfactual": cfm,
               "stability": stab, "attribution": attr, "decision_layer": dec,
               "stability_by_type": by_type.to_dict()},
              open(OUT / "explainability_metrics.json", "w", encoding="utf-8"),
              indent=2, default=str)
    reason.to_csv(OUT / "explainability_reason.csv", index=False)

    print("\n=== 1. REASON CORRECTNESS ===")
    print(reason.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n=== 2. COMPLETENESS ===")
    for k, v in comp.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== 3. COUNTERFACTUAL VALIDITY ===")
    for k, v in cfm.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== 4. STABILITY UNDER NOISE ===")
    for k, v in stab.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  changed % by type:")
    print(by_type.to_string())


if __name__ == "__main__":
    main()
