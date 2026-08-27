"""Generate the example QA disposition PDFs, covering the hard cases."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from explain.evidence import Evidence
from explain.decision import DecisionPolicy
from explain.rules import PartExplanation
from explain.counterfactual import CounterfactualEngine
from explain.attribution import DriftAttribution, detector_attribution
from explain.report import build_report

OUT = Path("reports"); OUT.mkdir(exist_ok=True)
MODEL_VERSION = "dataset-v1.1 / ModuleA-d13e1ef / ModuleB-early(0h,24h)"

# The two that matter most are the last two: any system can explain a rejection.
CASES = [
    ("Vb_EXTREME_LEVEL", "01_Vb_extreme_level_PS_worked_example", True,
     "the problem statement's own worked example: far outside the lot, inside spec"),
    ("III_CENTRE_HIDER", "02_TypeIII_centre_hider_joint_only", True,
     "invisible to every univariate and trajectory rule; caught only jointly"),
    ("I_STEEP_DRIFTER", "03_TypeI_steep_drifter_moduleB", True,
     "abnormal drift rate; the case Module B is for"),
    ("Va_MILDLY_HIGH_STABLE", "04_TypeVa_trap_correct_NON_rejection", False,
     "elevated but self-consistent: the system explaining why it did NOT reject"),
    ("VII_FIXTURE_ARTIFACT", "05_TypeVII_fixture_not_the_part", None,
     "this is your oven, not your part"),
]


def pick(e, pol, pool, t, want_flagged):
    ids = e.gt.index[(e.ty == t)]
    te = e.lot.isin(e.split.test)
    # A demonstration part must have a clean measurement record: otherwise the
    # data-quality gate fires first and the report shows MEASUREMENT_INVALID,
    # which is correct behaviour but not the case we are trying to illustrate.
    cand = [c for c in ids if te[c] and e.data_quality(c)["valid"]
            and not e.data_quality(c)["pulled_failed"]]
    cand = cand or [c for c in ids if te[c]] or list(ids)
    if want_flagged is True:
        f = [c for c in cand if pool[c]]
        cand = f or cand
    elif want_flagged is False:
        f = [c for c in cand if not pool[c]]
        cand = f or cand
    if t == "VII_FIXTURE_ARTIFACT":                # want one that clusters
        for c in cand:
            if e.spatial_check(c, pool)["clustered"]:
                return c
    if t == "III_CENTRE_HIDER":                    # want one actually caught
        sc = e.scores.loc[cand, "L3a_MCD"]
        return sc.idxmax()
    if want_flagged is True:
        return e.scores.loc[cand, "CUM_C4"].idxmax()
    return cand[0]


def main():
    e = Evidence(); pol = DecisionPolicy(e)
    pool = (e.scores["CUM_C4"] >= pol.thr_review["CUM_C4"])
    cf = CounterfactualEngine(e, pol)
    das = {}
    rows = []
    for t, name, want, why in CASES:
        cid = pick(e, pol, pool, t, want)
        x = PartExplanation(e, pol, cid, pool)
        c = cf.find(cid)
        p = (x.signals.get("drift_param") or x.signals.get("level_param")
             or e.params[0])
        if p not in das:
            das[p] = DriftAttribution(e, p).fit()
        shap = das[p].shap_for(cid)
        det = detector_attribution(e, pol, cid)
        pdf = OUT / f"{name}.pdf"
        build_report(e, pol, x, cf.text(cid, x.tier, c), det, shap, pdf,
                     MODEL_VERSION, param_focus=p)
        (OUT / f"{name}.txt").write_text(x.rule_text())
        rows.append({"case": name, "component_id": cid, "injected_type": t,
                     "decision": x.tier, "primary_reason": x.primary,
                     "counterfactual": c is not None, "why_this_case": why})
        print(f"  {name}: {cid} -> {x.tier} ({x.primary})")
    pd.DataFrame(rows).to_csv("results/report_examples.csv", index=False)


if __name__ == "__main__":
    main()
