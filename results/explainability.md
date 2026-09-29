# Explainability: the third scored axis, made quantitative

The problem statement scores three things: anomaly detection, drift prediction,
and **explainability** — *"can the model justify its classification to a QA
inspector, or is it a complete black box?"* Here are four measured numbers.

Dataset `dataset-v1.2`. Every figure below is written by `run_explainability.py`
into `explainability_metrics.json`; this file is the commentary. The
documentation site (chapter "Explainability") is generated from the same JSON.

## 1. Reason correctness

Does the stated primary reason match the injected archetype? Ground truth holds
the type, so this is measurable.

| defect_type | n | expected | primary correct | primary or secondary | top wrong reason |
|---|---|---|---|---|---|
| I_STEEP_DRIFTER | 60 | DRIFT_RATE/FORECAST | 100.0 | 100.0 | |
| II_STEP_DEFECT | 60 | STEP | 76.7 | 76.7 | DRIFT_RATE |
| III_CENTRE_HIDER | 1 | JOINT | 100.0 | 100.0 | |
| IV_CORRELATION_BREAK | 40 | JOINT | 100.0 | 100.0 | |
| Vb_EXTREME_LEVEL | 60 | LEVEL | 93.3 | 98.3 | JOINT |
| VII_FIXTURE_ARTIFACT | 60 | SPATIAL | 93.3 | 93.3 | STEP |
| GOOD | 60 | PASS | 93.3 | 98.3 | DATA_QUALITY |
| VI_LOT_SHIFT | 60 | PASS (trap) | 95.0 | 98.3 | LEVEL |
| Va_MILDLY_HIGH_STABLE | 60 | PASS (trap) | 51.7 | 66.7 | LEVEL |
| III + IV pooled | 41 | JOINT | 100.0 | 100.0 | |

- **Type II at 76.7%** is the weakest genuine class; the wrong answer is
  `DRIFT_RATE`. A step landing in the first interval looks like a drifter for
  the rest of the trajectory and is genuinely ambiguous.
- **Type VII at 93.3%.** SPATIAL now requires the part's own thermal signature
  as well as board clustering, so a few fixture parts whose shift is weak are
  explained as the component instead. That is the price of no longer excusing
  real defects that sit on busy boards.
- **Type Va at 51.7% fully passed**, 66.7% not held. The joint-detector review
  branch added in the audit catches centre-hiders and also holds some benign
  high parts; this is its measured cost.
- **Type III is n = 1** in this sample; quote the pooled JOINT row.

## 2. Completeness

| Measure | Value |
|---|---|
| parts explained | 461 |
| flagged parts with a complete justification (evidence + reason + mechanism) | **68.8%** |
| passed parts with the full evidence set | **100.0%** |
| flagged parts with a primary reason | 98.6% |
| flagged parts with a mechanism hypothesis | 69.2% |

The gap between "has a reason" and "complete" is the mechanism layer: about 30%
of flagged parts match no mechanism signature, and the system says so rather
than inventing one.

## 3. Counterfactuals

| Measure | Value |
|---|---|
| flagged parts with a counterfactual | **7.9%** |
| of those, valid against the whole measurement record | **100%** |
| univariate-driven flags with a counterfactual (n = 186) | 11.8% |
| joint-driven flags with a counterfactual (n = 46) | 0.0% |

**What changed.** The previous version reported 26.7% offered, 100% valid. Its
validation re-checked only the checkpoint it had just edited, which made 100% a
tautology. A counterfactual is now kept only if, after the change, **every**
checkpoint clears both rules; if another checkpoint still fires, a one-value
change cannot clear the part and nothing is offered. Hence the lower, honest
coverage.

**Scope.** The counterfactual is evaluated against the rule layer (lot-relative
robust z at 6 sigma and robust Mahalanobis D² against the training-lot
reference), not the fitted L2-L4 detectors behind the fused score. It says "the
rules would stop firing", not "the tier would become PASS", and the report text
says exactly that.

## 4. Stability under measurement noise

Every measurement re-drawn at the generator's own noise magnitude, four times,
for 120 parts. The rule layer is recomputed on the perturbed data; detector
scores, the flag pool and the tier are held at their original values.

| Measure | Value |
|---|---|
| primary reason unchanged | **90.0%** |

By type: Types II, III and IV 100% stable, Vb 98.3%, Va 91.7%, Type I 90.4%;
the churn is in `GOOD` (17.5% change), `VI_LOT_SHIFT` (25%) and Type VII
(15.4%, where the thermal-sign test sits near its threshold for weak-gradient
sockets). Explanations of genuine defect signatures are stable; the churn is
among parts where no rule fires strongly.

## What a reliability engineer would dispute

Only two mechanism mappings are marked **well-founded**, and both rest on one
sign argument: heat raises leakage and *slows* the part, while the process
corner makes leaky parts *fast*. That is now also what the decision layer
checks before calling a part FIXTURE_SUSPECT. Everything else is labelled
*plausible* or *speculative* on the page; `JOINT_INCONSISTENCY` is speculative,
because a joint outlier says a part does not resemble its peers, not why. The
system hands the failure-analysis engineer a pre-argued case file; it does not
replace them.

## A finding about Layer 2 that argues for Layer 1

SHAP on LightGBM and an equally accurate Huber model **disagree about which
features matter**: Spearman rho = 0.617 (p = 0.077) over the nine Iddq features.
SHAP ranks `v24` first; Huber ranks `v0` first. `corr(v0, v24) = 0.987` on the
test lots, and dropping `v0` changes test MAE from 0.6103 to 0.6094 uA. Under
collinearity the credit split between near-duplicate inputs is a property of
the model, not the physics. So Layer 1, the deterministic rule text an
inspector can recompute by hand, is the record; SHAP is labelled supporting
evidence on the page. Source: `attribution` in `explainability_metrics.json`
and `attribution_comparison.csv`.

## Example disposition reports

| case | component | injected type | decision | primary reason |
|---|---|---|---|---|
| 01_Vb_extreme_level_PS_worked_example | LOT193-C0488 | Vb | REVIEW | LEVEL |
| 02_TypeIII_centre_hider_joint_only | LOT233-C0222 | III | REVIEW | JOINT |
| 03_TypeI_steep_drifter_moduleB | LOT203-C0246 | I | REVIEW | DRIFT_RATE |
| 04_TypeVa_trap_correct_NON_rejection | LOT192-C0144 | Va | WATCH | NONE |
| 05_TypeVII_fixture_not_the_part | LOT213-C0256 | VII | FIXTURE_SUSPECT | SPATIAL |

Case 02 was WATCH (released) before the audit; the joint-detector review branch
now holds it. Cases 04 and 05 are the ones that show judgement: a correct
non-rejection, and "this is your oven, not your part".
