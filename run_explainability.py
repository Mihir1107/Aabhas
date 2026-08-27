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

    json.dump({"reason_correctness": reason.to_dict("records"),
               "completeness": comp, "counterfactual": cfm,
               "stability": stab,
               "stability_by_type": by_type.to_dict()},
              open(OUT / "explainability_metrics.json", "w"), indent=2, default=str)
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
