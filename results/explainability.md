# Explainability: the third scored axis, made quantitative

The problem statement scores three things: anomaly detection, drift prediction,
and **explainability** — *"can the model justify its classification to a QA
inspector, or is it a complete black box?"* This is the axis almost nobody
brings numbers to. Here are four.

Dataset `dataset-v1.1`. Nothing was refitted and no published number changed;
the SHAP model is asserted to reproduce the published Module B rung-4 MAE
exactly (delta 0.00e+00) before it is allowed to emit an explanation.

## 1. Reason correctness

Does the stated primary reason match the injected archetype? Ground truth holds
the type, so this is measurable rather than a matter of taste.

| defect_type | n | expected | primary_correct_% | primary_or_secondary_% | top_wrong_reason |
|---|---|---|---|---|---|
| I_STEEP_DRIFTER | 60 | DRIFT_RATE/FORECAST | 100.0 | 100.0 |  |
| II_STEP_DEFECT | 60 | STEP | 73.3 | 76.7 | DRIFT_RATE |
| III_CENTRE_HIDER | 1 | JOINT | 100.0 | 100.0 |  |
| IV_CORRELATION_BREAK | 40 | JOINT | 95.0 | 100.0 | SPATIAL |
| Vb_EXTREME_LEVEL | 60 | LEVEL | 88.3 | 98.3 | JOINT |
| VII_FIXTURE_ARTIFACT | 60 | SPATIAL | 100.0 | 100.0 |  |
| GOOD | 60 | PASS | 93.3 | 98.3 | DATA_QUALITY |
| VI_LOT_SHIFT | 60 | PASS (trap) | 95.0 | 100.0 |  |
| Va_MILDLY_HIGH_STABLE | 60 | PASS (trap) | 55.0 | 81.7 | LEVEL |

**Read the failures, they are the interesting part.**

- **Type II at 73.3%** is the weakest genuine class. The wrong answer is almost
  always `DRIFT_RATE`. A step defect and a steep drifter both produce a large
  max-interval-jump, so jump magnitude alone cannot separate them; the
  discriminator added here is jump *concentration* (largest interval change over
  the median of the others, threshold 3x). Parts whose step lands in the first
  interval look like a drifter for the rest of the trajectory and are genuinely
  ambiguous from the data. 76.7% when the secondary reason is allowed.
- **Type Va at 55% PASS** is the number to be honest about. Va is the benign
  high-but-stable trap, and the system fully passes only 55% of them, though
  81.7% avoid rejection (PASS or WATCH). The remainder are called `LEVEL`, which
  is *literally true* — they are elevated — but is the wrong disposition. This
  is the false-positive cost of the level rule, measured rather than asserted.
- **Type III shows n=1** because only 8 Type III parts exist in the test lots
  and one clears the flag threshold. That number is not quotable; the per-tier
  Type III detection figures from the Module A session are the ones to cite.
- `GOOD` at 93.3% PASS: the residual is mostly `DATA_QUALITY`, i.e. parts with a
  chamber-trip checkpoint, which the gate correctly routes to
  MEASUREMENT_INVALID rather than to a verdict.

## 2. Explanation completeness

| Measure | Value |
|---|---|
| parts explained | 461 |
| flagged parts with a complete justification (evidence + primary reason + mechanism) | **69.5%** |
| passed parts with a complete justification | **100.0%** |
| all parts carrying the full evidence block | 93.3% |
| flagged parts with a primary reason | 98.6% |
| flagged parts with a mechanism hypothesis | 69.9% |

Completeness is scored separately for flags and passes on purpose: a PASS part
needs its evidence lines and an action, and *"no rule fired"* is its complete
explanation. Scoring them together would penalise correct passes. The gap
between 98.6% (has a primary reason) and 69.5% (complete) is entirely the
mechanism layer: 30% of flagged parts match no mechanism signature, and the
system says so rather than inventing one.

## 3. Counterfactual validity

| Measure | Value |
|---|---|
| flagged parts with a counterfactual found | **26.7%** |
| of those, fraction that actually flip when applied | **100.0%** |
| counterfactual found, univariate-driven flags (n=177) | 29.9% |
| counterfactual found, joint-driven flags (n=44) | 2.3% |

**Validity is 100% because invalid ones are discarded, not reported.** Every
candidate is found by bisection on the live scoring path, then re-applied and
re-scored; if the decision does not actually flip it is dropped and the report
says "no single-parameter change clears the rules".

**Coverage is only 26.7%, and that is a real limitation, not a bug.** The split
explains it: 29.9% for univariate-driven flags against **2.3% for joint-driven
flags**. A part that is anomalous in the *joint* distribution has, by
construction, no single measurement you can move to make it normal — that is
what "only visible jointly" means. Type IV and Type III parts should not have
single-feature counterfactuals, and they do not. For those, the honest
explanation is the mechanism hypothesis plus the Mahalanobis contribution
breakdown, not a counterfactual.

Scope limit, stated on the report page: the counterfactual is evaluated against
the **rule-based** evidence (lot-relative robust z and robust Mahalanobis), not
against the fitted L4 ensemble, because re-scoring those needs model objects
this session was not permitted to refit.

## 4. Explanation stability under measurement noise

Measurements were perturbed at the generator's own noise magnitude
(`meas_noise_frac_of_part_sd` x the part-to-part sd, the same sd_eps the data
was built with), features rebuilt, and the primary reason recomputed.
120 parts x 4 independent noise draws.

| Measure | Value |
|---|---|
| primary reason unchanged under noise | **91.5%** |
| primary reason changed | 8.5% |

Per type: Type III, Type IV, Type II and Type VII are **100% stable**; Vb 98.3%;
Type I 88.5%; and the unstable cases are concentrated in `GOOD` (17.5% change)
and `VI_LOT_SHIFT` (25%). That pattern is the reassuring one: **explanations for
parts that genuinely have a defect signature are stable, and the churn is
among parts where no rule strongly fired**, where the "primary reason" is
picking between near-tied weak signals and flipping between them is expected.
An explanation that flipped on real defects would be disqualifying; one that
flips on quiet good parts is cosmetic.

## The four numbers, together

| Metric | Value |
|---|---|
| Reason correctness (genuine defect classes, primary) | **73–100%**, weakest Type II at 73.3% |
| Explanation completeness (flagged / passed) | **69.5% / 100%** |
| Counterfactual validity (of those offered) | **100%**, offered for 26.7% |
| Explanation stability under noise | **91.5%** overall, **100%** on Types II/III/IV/VII |

## What a reliability engineer would dispute

The mechanism table carries an explicit confidence label per entry, and only
two are marked **well-founded**:

- **Fixture versus process, by the SIGN of the correlation shift.** Heat raises
  leakage and *slows* the part; the process corner makes leaky parts *fast*. So
  a chamber artifact and a process outlier move the leakage/delay correlation in
  opposite directions and are separable before any spatial evidence is
  considered. This is the one to defend in a Q&A.
- **Correlation inversion as evidence of a defect-driven leakage path**, the
  same argument used the other way.

Everything else is labelled **plausible** or **speculative**, and the weakest is
labelled as such in the code and on the page: `JOINT_INCONSISTENCY` is marked
*speculative*, because a joint-distribution outlier says the part does not
resemble its peers — it does **not** say why. Accelerating leakage is consistent
with TDDB but equally with ionic contamination and junction damage; the
trajectory shape alone does not separate them. NBTI and HCI both slow a part
without moving quiescent current, and separating them needs a recovery test.

The framing that follows: **the system does not replace the failure-analysis
engineer, it hands them a prioritised, pre-argued case file.**

## A finding about Layer 2 that argues for Layer 1

SHAP on LightGBM and the coefficients of an equally accurate Huber model
**disagree about which features matter** (Spearman rho = 0.617, p = 0.077).
SHAP ranks `v24` first and `v0` fifth; Huber does the reverse.

Diagnosed, not smoothed over: `corr(v0, v24) = 0.989` and
`corr(lotmed_t0, lotmed_t24) = 0.999`. Dropping `v0` from the model entirely
changes test MAE from 0.6102 to 0.6085 — it is redundant. **Under collinearity
the attribution is not identifiable**: both models predict identically and
distribute credit differently among near-duplicate features.

The consequence is worth stating plainly, because it cuts against the
fashionable answer. **Layer 2 attribution is model-dependent here, so Layer 1 —
the deterministic rule text — is the layer to put in front of an inspector.**
Every Layer 1 number cites the comparison it was made against and can be
recomputed by hand from the lot data. A SHAP value cannot.

## Example disposition reports

| case | component_id | injected_type | decision | primary_reason |
|---|---|---|---|---|
| 01_Vb_extreme_level_PS_worked_example | LOT193-C0488 | Vb_EXTREME_LEVEL | REVIEW | LEVEL |
| 02_TypeIII_centre_hider_joint_only | LOT233-C0222 | III_CENTRE_HIDER | WATCH | JOINT |
| 03_TypeI_steep_drifter_moduleB | LOT203-C0246 | I_STEEP_DRIFTER | REVIEW | DRIFT_RATE |
| 04_TypeVa_trap_correct_NON_rejection | LOT192-C0144 | Va_MILDLY_HIGH_STABLE | WATCH | NONE |
| 05_TypeVII_fixture_not_the_part | LOT213-C0256 | VII_FIXTURE_ARTIFACT | FIXTURE_SUSPECT | SPATIAL |

The last two matter most. Any system can explain a rejection; explaining a
**correct non-rejection** (case 04) and saying **"this is your oven, not your
part"** (case 05) is what demonstrates judgement.

PDFs in `reports/`, plain-text Layer 1 in the matching `.txt` files.
