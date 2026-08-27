# Presentation package: every number, with the condition it was measured under

**Who this is for.** The four of you building the six SIH slides. You have not
read the code and you should not have to. Everything you might put on a slide is
below, with the exact condition attached.

**The one rule.** Copy the number AND its condition. Every number in this file
has a condition line next to it. If you drop the condition, a judge will ask for
it and we will not have it on the slide.

**Vocabulary, once.** *Yield loss* is the percentage of good parts we wrongly
reject. *Escape rate* is the percentage of genuinely defective parts we wrongly
pass. *Recall* is the opposite of escape rate: the percentage of defective parts
we catch. *Test lots* means production batches the model never saw during
training. *Lot* means one manufacturing batch of 500 components.

Dataset is frozen at tag `dataset-v1.1`. Results are at commit `bf71caa`.

---

## READ THIS FIRST: four number-consistency traps

These have bitten us before. Two of them already cost us once each.

**Trap 1. Type VI has three correct answers, and they are very different.**
Type VI is the "whole lot shifted" trap: a batch where the manufacturing process
drifted, so every part in it looks unusual against history but normal against
its own batch. The headline is that static limits scrap them and dynamic limits
do not. The size of that effect depends entirely on where you set the threshold:

| Condition | Static PAT flags | Dynamic PAT flags |
|---|---|---|
| Fixed 6 sigma (the AEC-Q001 standard multiplier) | **11.97%** (359 of 3,000 parts) | **0.00%** (0 parts) |
| Matched to a 7% yield-loss budget | 59.07% (1,772 parts) | 5.97% (179 parts) |

**Use the 6 sigma row.** It is the standard's own operating point, it needs no
arbitrary budget, and it is the cleaner contrast (359 versus 0). If anyone has
"1,770 good parts" in their notes from an earlier conversation, the correct
figure is **1,772**, and it belongs to the yield-loss-matched row, not the
6 sigma row. Do not mix them.

**Trap 2. C1 and L1b are the same detector but report slightly different
scores.** In `ablation_cumulative.csv`, rung C1 shows PR-AUC 0.2636 and AUROC
0.8330. In `ablation.csv`, rung L1b (the same DPAT detector) shows 0.2644 and
0.8407. This is not an error. C1 is defined as "L0 plus L1", so it also includes
the four genuinely good parts that fall outside datasheet limits by chance, and
those sit at the top of the combined ranking. Recall is identical (54.33%).
If you show a cumulative ladder, use the C-numbers throughout. If you show a
head-to-head method table, use the L-numbers throughout. Do not mix.

**Trap 3. The estimator table in the old v1.0 README does not reproduce.**
Those figures (for example MAD catching 29.5%, the p1/p99 variant catching 4.5%)
were computed on a 20-lot trial run, not on the frozen 240-lot dataset. On the
frozen data the same measurements are 38.1% and 16.9%. The direction of the
argument holds. The magnitudes do not. Do not quote the old table.

**Trap 4. One number in Section 2, finding 2 is not in `results/`.** The
Isolation Forest narrow-versus-wide feature test (41.7% versus 67.7%) was an
exploratory diagnostic run during the v1.0 session. It was never saved to a
results file and has not been re-run on v1.1. Either present it with the words
"in an exploratory check on the earlier dataset version", or leave it out and
use the Type III result on its own, which is stronger anyway.

---

# Section 1: The six headline numbers

## 1. Escape rate: 45.7% down to 7.3%

| | Value |
|---|---|
| **The number** | Escape rate falls from **45.7%** to **7.3%**, a 6.3 times reduction |
| **In one sentence** | The industry-standard statistical screen lets about 46 out of every 100 defective parts through. Our full stack lets about 7 through, while rejecting exactly the same number of good parts. |
| **Condition** | Both measured on **test lots only** (48 lots, 24,000 parts, 427 defective), at an **identical 7.00% yield loss**. Start point is rung C1 (AEC-Q001 Dynamic Part Average Testing with the MAD estimator). End point is rung C3 (adds trajectory features and robust multivariate detection). |
| **Source** | `results/ablation_cumulative.csv` |

Say "at identical yield loss" out loud. It is the part that makes the comparison
fair, and it is the first thing a judge will probe.

## 2. Module B prediction error (MAE)

Mean Absolute Error, in physical units. It is the average size of the gap
between what we predicted the 168 hour value would be and what it actually was.

| Parameter | MAE, all parts | MAE, good parts | MAE, defective parts | Ratio |
|---|---|---|---|---|
| Quiescent current (Iddq) | 0.609 uA | 0.479 uA | 7.743 uA | 16.2x |
| Input leakage | 1.296 nA | 1.109 nA | 11.622 nA | 10.5x |
| Propagation delay | 0.068 ns | 0.062 ns | 0.397 ns | 6.4x |
| Supply current | 0.677 mA | 0.621 mA | 3.784 mA | 6.1x |
| Threshold voltage shift | 0.843 mV | 0.781 mV | 4.266 mV | 5.5x |

**Condition:** early-warning mode, which uses the 0 hour and 24 hour readings
ONLY and never sees 96 hours. Test lots. LightGBM, rung 4 of the model ladder.
Source: `results/module_b.csv`.

**Always give both columns.** Error on defective parts is 5.5 to 16 times worse
than on good parts. That is expected, because defective parts are precisely the
ones whose behaviour departs from the population the model learned. But it means
a single headline MAE is a misleading summary for a safety application, and a
judge who asks "and on the parts that matter?" must not catch us without it.

For context, the naive baseline (draw a straight line through the first two
readings and extend it) gives 2.868 uA on Iddq. The full model is 4.7 times
better.

## 3. The conformal guarantee

*Conformal risk control* is a statistical method that puts a proven ceiling on
our escape rate, rather than just a hope. You pick a target, called alpha, and
the method guarantees the expected escape rate stays at or below it.

| | Value |
|---|---|
| **The number** | The guarantee **holds at every alpha tested, from 1% to 20%**, with no statistically significant violation |
| **Condition** | 8 alpha values, **40 independent calibration splits each, split by lot**. Tested with a one-sided t-test of the hypothesis that the true escape rate exceeds alpha. All p-values above 0.28. Largest overshoot was 0.5 standard errors, which is noise. |
| **Source** | `results/conformal_summary.csv`, `results/conformal_significance.csv` |

The cost of the guarantee, which is the honest half of the story:

| Alpha (guaranteed escape rate) | Measured escape rate | Yield loss (good parts sacrificed) |
|---|---|---|
| 1% | 0.80% | 42.7% |
| 2% | 1.77% | 23.8% |
| 3% | 2.90% | 18.2% |
| 5% | 4.98% | 11.6% |
| 7.5% | 7.65% | 7.0% |
| 10% | 9.89% | 4.4% |
| 15% | 14.68% | 1.6% |
| 20% | 19.44% | 0.97% |

**Condition:** the fused Module A detector. Calibration and evaluation both drawn
from validation and test lots only, never training lots. Read Section 4 before
putting the 1% row on a slide.

## 4. The cumulative ladder

Each rung includes everything below it. This is the build story.

| Rung | What it contains | Recall at 93% yield | 95% confidence interval | Escape rate |
|---|---|---|---|---|
| C0 | Static datasheet limits alone | not applicable | | 100.0% |
| C1 | plus AEC-Q001 Dynamic PAT | 54.33% (232 of 427) | [49.6, 59.0] | 45.67% |
| C2 | plus trajectory features | 79.86% (341 of 427) | [75.8, 83.4] | 20.14% |
| C3 | plus robust multivariate | **92.74%** (396 of 427) | **[89.9, 94.8]** | 7.26% |
| C4 | plus unsupervised machine learning | 92.27% (394 of 427) | **[89.3, 94.4]** | 7.73% |

**Condition:** test lots, 7.00% yield loss at every rung, rungs fused by taking
the strongest normalised signal. Source: `results/ablation_cumulative.csv`.

**State the plateau explicitly.** C4 is 2 defective parts below C3, out of 427.
The confidence intervals overlap almost completely, [89.9, 94.8] against
[89.3, 94.4]. The honest reading is **the ladder plateaus at L3**, not that L4
makes things worse. Saying this before a judge finds it is worth more than the
0.47 percentage points it costs us.

C0 has no threshold-free score because it is a fixed pass or fail rule, so
recall at a yield-loss budget is not defined for it. It catches 0.0% of every
injected defect type, which is the point: it is the strawman.

## 5. The explainability metrics

The problem statement scores explainability. Almost nobody brings numbers to it.

| Metric | Value | What it means |
|---|---|---|
| Reason correctness | **73% to 100%** by defect class | Does the stated reason match the actual injected fault |
| Explanation completeness | **69.5%** flagged, **100%** passed | Fraction carrying a full justification |
| Counterfactual validity | **100%** of those offered (offered for 26.7%) | Does the "it would have passed if..." statement actually work when applied |
| Explanation stability | **91.5%** overall, **100%** on Types II, III, IV, VII | Does the reason survive re-measurement noise |

Reason correctness by type:

| Defect type | What the reason should be | Correct |
|---|---|---|
| Type I, steep drifter | abnormal drift rate | 100.0% |
| Type II, step defect | discrete step | 73.3% |
| Type III, centre-hider | joint anomaly | 100.0% (n=1, do not quote) |
| Type IV, correlation break | joint anomaly | 95.0% |
| Type Vb, extreme level | elevated level | 88.3% |
| Type VII, fixture artifact | spatial clustering | 100.0% |
| Good parts | should pass | 93.3% |
| Type VI trap | should pass | 95.0% |
| Type Va trap | should pass | 55.0% (81.7% avoid rejection) |

**Condition:** 461 parts sampled across all types, test lots where sample size
allowed. Stability measured over 120 parts and 4 independent noise draws at the
dataset's own measurement-noise magnitude. Source:
`results/explainability_metrics.json`, `results/explainability.md`.

The Type III row says n=1. Only 8 Type III parts exist in the test lots and one
crossed the flag threshold. **Do not put 100% on a slide.** Use the detection
figures from Section 1 item 4 instead.

## 6. Static versus dynamic limits on a shifted lot

| | Value |
|---|---|
| **The number** | Static limits reject **359 good parts**. Dynamic limits reject **0**. |
| **In one sentence** | When a manufacturing batch drifts as a whole, fixed historical limits scrap hundreds of perfectly good components, while limits recalculated from the batch itself correctly absorb the shift. |
| **Condition** | Type VI trap class, 3,000 parts, **all lots**, at the AEC-Q001 standard fixed multiplier of **6 sigma**. |
| **Source** | `results/flags.csv.gz` via `results/per_type.csv` |

**Why this is an all-lots number and not a test-set number.** There are only six
lot-shift batches in the whole dataset, and by chance the random assignment put
five of them in the first 85 lots and the sixth at lot 146. Our split is
chronological (earliest lots train, latest lots test), so none of them landed in
the test block. Getting even one into the test set would require a 40/20/40
split, which would halve the training data. We chose to keep the split honest
and label the number. One of the six lots does sit in the validation block, which
is a genuine non-training holdout, so the effect is confirmed on unseen data
even though the headline count is across all lots.

If asked: "That is an all-lots figure, because the six shifted batches all fall
early in the production sequence and our split is chronological. We did not move
the split to make it a test number."

---

# Section 2: The five findings that carry slides

## Finding 1: The derivative argument

**Claim.** The same argument works twice: for levels in Module A and for drift
rates in Module B. Limits must be relative to the batch, not to a fixed number.

**Evidence.** In Module A, static limits reject 359 good parts from a shifted
batch while dynamic limits reject 0. In Module B, the equivalent mistake is
using the datasheet limit as the safety threshold. That rule is **structurally
inert on this problem**: every defect we inject is inside the datasheet limits
at every checkpoint by construction, which is what makes them hard, so a
predicted worst case essentially never crosses an engineering limit. Measured,
that rule catches **1.4%** of defects. Swap the datasheet limit for a
batch-derived safe limit and the identical rule catches **17.8%**, at 0.15%
yield loss with 68.5% precision. Source: `results/safety_slopes_union.csv`.

**Slide line.** "Static limits fail on levels and fail again on drift rates. The
same fix works both times: compare each part to the batch it was burned in with."

## Finding 2: Isolation Forest catches 0% of the hardest defect class

**Claim.** A popular anomaly-detection algorithm catches literally none of the
defect class the whole system exists to find, and the reason is structural
rather than a tuning problem.

**Evidence.** Isolation Forest catches **0.0% of 60 Type III parts** (all lots,
7% yield-loss operating point). Type III is the "centre-hider": a part sitting in
the normal range on every individual measurement, abnormal only in the
combination. Isolation Forest works by splitting one measurement at a time, so
it cannot isolate a point that is central on every measurement individually.
This contradicts the ITC 2020 industrial benchmark, where Isolation Forest is
among the strongest conventional methods. We report the contradiction rather
than reordering our table, and the explanation is that our dataset deliberately
over-weights the centre-hider class relative to real production data.

We also checked that this is not simply a "too many features" problem by running
the same method on a narrow feature set as well as the wide one. See Trap 4
above for how to cite that check.

**Slide line.** "Isolation Forest finds none of the centre-hiders. An
axis-by-axis method cannot isolate a part that looks normal on every axis."

## Finding 3: The simple domain method beats the neural network by 54 points

**Claim.** On the class that needs correlation reasoning, a five-parameter
statistical method beat an 83-feature neural network by a wide margin.

**Evidence.** On mild Type IV (correlation break, the hardest tier, 141 parts,
all lots, 7% yield loss): Mahalanobis distance with robust covariance catches
**90.1%**, 95% confidence interval **[84.0, 94.0]**. The best neural approach,
Local Outlier Factor, catches **36.2%**, interval **[28.7, 44.4]**. The
autoencoder catches 31.2%. Isolation Forest catches 2.1%. The intervals do not
come close to overlapping. Source: `results/per_type.csv`.

**Slide line.** "For the defect that hides in the correlations, classical robust
statistics beat the neural network by 54 points. We report what worked, not what
sounds modern."

## Finding 4: Two attribution methods disagree, and that argues against the fancy one

**Claim.** SHAP, the standard machine-learning explanation method, and a simple
robust regression of equal accuracy disagree about which measurements matter.
The disagreement is not resolvable, so the deterministic rule text is what we
put in front of an inspector.

**Evidence.** Rank correlation between the two importance orderings is
**Spearman rho = 0.617, p = 0.077**, which is not significant. SHAP ranks the
24 hour reading first and the 0 hour reading fifth. The robust regression does
the reverse. Diagnosis: the two readings are **0.989 correlated**, and removing
the 0 hour reading from the model entirely changes test error by 0.0017 uA
(0.6102 to 0.6085), which is nothing. When two inputs are near-duplicates, the
credit assigned between them is not identifiable, so it is a property of the
model rather than of the physics. Source: `results/explainability.md`.

**Slide line.** "When two equally accurate models disagree about why, the
explanation you show an inspector should be the one they can recompute by hand."

## Finding 5: Static limits scrap 359 good parts over a process shift

**Claim.** The clearest single demonstration of why limits must be batch-relative.

**Evidence.** 3,000 components from six batches where the manufacturing process
shifted as a whole. Not defective: the parts are fine, the batch simply sits in a
different place. At the AEC-Q001 standard 6 sigma multiplier, static limits
reject **359 of them** and dynamic limits reject **0**. All-lots figure, see
Section 1 item 6 for why.

**Slide line.** "One process excursion, 359 good components scrapped by fixed
limits and zero by adaptive ones."

---

# Section 3: Slide-by-slide asset map

## Slide 1: Title Page
Nothing quantitative. If you want one number, use the escape rate line from
Section 1 item 1.
**Most important asset: none.**

## Slide 2: Proposed Solution
- The problem statement's own worked example, answered: a batch averaging about
  12 uA with a part at 40 to 48 uA that still passes the datasheet limit. Our
  system catches 100% of that class (Type Vb, 300 parts, every rung from C1
  upward). Source: `results/per_type_cumulative.csv`.
- The disposition report as the visible output:
  `reports/01_Vb_extreme_level_PS_worked_example.pdf`
- Six decision tiers, not a pass or fail binary: PASS, WATCH, REVIEW, REJECT,
  MEASUREMENT_INVALID, FIXTURE_SUSPECT.
- Finding 1, the derivative argument, as the framing.

**Most important asset: `reports/01_Vb_extreme_level_PS_worked_example.pdf`.**
It answers the problem statement in its own example, on one page.

## Slide 3: Technical Approach
- The cumulative ladder table, Section 1 item 4.
- **Plot: `results/fig5_cumulative_ladder.svg`** (recall against yield loss, all
  rungs).
- Finding 2 (Isolation Forest) and Finding 3 (Mahalanobis beats the neural
  network) as the two technical arguments.
- **Plot: `results/fig4b_typeIV_by_tier.svg`** shows Finding 3 visually.
- The spatial layer, our original contribution:
  `reports/05_TypeVII_fixture_not_the_part.pdf`

**Most important asset: `results/fig5_cumulative_ladder.svg`.** It is the build
story in one image.

## Slide 4: Feasibility and Viability
- The conformal guarantee, Section 1 item 3.
- **Plot: `results/fig8_conformal_guarantee.svg`** (measured escape rate against
  the target, with spread across 40 splits).
- **Plot: `results/fig9_conformal_tradeoff.svg`** (the cost of the guarantee).
- The validation protocol: lot-grouped chronological splits, never random rows.
  Split sizes in `results/split_counts.csv`.
- Leave-one-lot-out cross-check: autoencoder 94.0% mean recall with 8.3 standard
  deviation across 10 held-out lots, consistent with the main split.
  Source: `results/leave_one_lot_out.csv`.

**Most important asset: `results/fig8_conformal_guarantee.svg`.** It is the only
slide element that shows a proven bound rather than a measured average.

## Slide 5: Impact and Benefits
- Finding 5, the 359 scrapped parts, as the yield-loss argument.
- Escape rate 45.7% to 7.3%, Section 1 item 1.
- Module B early warning: the model predicts the 168 hour outcome from the first
  24 hours, which is where the schedule saving lives. MAE table, Section 1 item 2.
- The explainability metrics, Section 1 item 5, as the "auditable" argument.
- **Plot: `results/fig1_recall_vs_yield_loss.svg`** for the operating-point
  tradeoff.

**Most important asset: the 359-versus-0 number.** It is the one figure a
production manager will react to.

## Slide 6: Research and References
- AEC-Q001 (Part Average Testing), MIL-STD-883 Method 1015 (burn-in), the
  Arrhenius acceleration model at Ea = 0.7 eV giving AF = 77.66 for 125 C stress
  against 55 C use, which makes 168 hours equal 1.488 field years.
- ITC 2020 (Hu, Nguyen, He and Li) for the centre-of-distribution finding that
  Type III implements, and for the benchmark we contradict in Finding 2.
- Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024,
  arXiv:2208.02814).
- Rousseeuw and Van Driessen for the robust covariance estimator in Finding 3.

**Most important asset: the Arrhenius line.** It shows the dataset is grounded in
physics rather than invented.

---

# Section 4: Numbers we must NOT put on a slide

**1. Anything from the v1.0 README estimator table.** It was computed on a 20-lot
trial, not the frozen dataset, and does not reproduce. See Trap 3.

**2. Plain accuracy, in any form.** At 1.75% defect prevalence, a model that
flags nothing scores 98.25% accurate. If accuracy appears anywhere we have
handed a judge a free question. Use recall, escape rate and yield loss.

**3. A single aggregate Module B MAE with no good-versus-defective split.**
0.609 uA sounds excellent until someone asks about the defective parts, where it
is 7.743 uA. Always show both.

**4. Any Type VI figure described as a test-set number.** It is all-lots. See
Section 1 item 6 for the one-sentence explanation.

**5. Any Type III per-tier test-set rate.** There are 8 Type III parts in the
test lots, so a per-tier rate is computed on 2 or 3 parts. Use the all-lots
figures (60 parts) and say "all lots".

**6. The 42.7% yield loss at alpha = 1% presented as our operating point.**

Here is the correct framing, written out so you can use it verbatim:

> "That 42.7% is one end of a tradeoff curve, not our operating point. The curve
> is steep at the low-alpha end because the mild-severity defects sit right at
> the edge of the good-part distribution, so buying the last few percent of
> guaranteed recall costs a great deal of yield. Where you sit on that curve is a
> programme decision driven by the cost ratio between an escape and a scrapped
> part, not a model decision. For reference, our measured performance is 6.3%
> escape at 7% yield loss, and the formally guaranteed version of roughly that
> same point is 5% escape at 11.6% yield loss. That gap is what finite-sample
> statistical rigour costs, and we would rather show it than hide it."

---

# Section 5: Judge question answers

Two sentences each. Where the number lives is in brackets.

**1. Where will the data come from?** A synthetic generator grounded in the
Arrhenius model, with acceleration factor 77.66 at Ea = 0.7 eV, so 168 hours of
burn-in equals 1.488 field years. Production deployment needs representative
data from SAC, which we state as a pilot dependency rather than an assumption.
[`data/config.json`]

**2. How do you handle so few defect labels?** Every unsupervised detector is
fitted on known-good parts from training lots only and never sees a defect label.
We also checked whether that clean reference is doing the work: fitting on
contaminated data instead changes Isolation Forest from 68.2% to 74.9%, so the
assumption is not load-bearing. [`results/all_methods.csv`]

**3. Why not deep learning?** Four timepoints, tabular data, and explainability
is a scored criterion. Our own results support it: on the hardest defect tier a
five-parameter robust statistic beat the neural network 90.1% to 36.2%.
[`results/per_type.csv`]

**4. How do you avoid false alarms?** Batch-relative baselines, six decision
tiers instead of pass or fail, a published yield-loss curve, and a spatial check
that reclassifies chamber problems as fixture faults rather than part rejections.
Measured: at our operating point we reject 6% of good parts, and the trap classes
designed to fool us are correctly passed 95% (Type VI) and 81.7% (Type Va) of the
time. [`results/explainability_metrics.json`]

**5. How do you know your flagged parts are actually defective?** On synthetic
data we know by construction, which is why we built the dataset with traps that
punish over-flagging. On real data this becomes a pilot question, and we would
propose confirming flagged parts by extended HTOL stress as the acceptance
criterion.

**6. How do you validate across new lots?** Lot-grouped chronological splits:
train on lots 0 to 143, validate on 144 to 191, test on 192 to 239, with no
component appearing in two splits. We also ran leave-one-lot-out as a
cross-check, which agrees with the main split (autoencoder 94.0% mean recall,
8.3 standard deviation). [`results/split_counts.csv`,
`results/leave_one_lot_out.csv`]

**7. Can the model explain its decision?** Three layers, and we measured them:
reason correctness 73% to 100% by class, explanation stability under
re-measurement 91.5%, counterfactual validity 100% of those offered. The output
is a one-page disposition report, five examples in `reports/`.
[`results/explainability.md`]

**8. How do you distinguish a sensor fault from a component fault?** A
data-quality gate runs before any model and routes incomplete records to
MEASUREMENT_INVALID rather than to a verdict. For chamber effects we use the
physics: heat makes a part leaky and slow, while process variation makes leaky
parts fast, so the two move the leakage-versus-delay correlation in opposite
directions and are separable by sign before any spatial evidence is used.
[`reports/05_TypeVII_fixture_not_the_part.pdf`]

**9. Does this replace engineering limits?** No. Datasheet limits stay mandatory,
the model never loosens them, and our statistical limits are always clipped to
them, exactly as AEC-Q001 requires.

**10. What happens when the model is uncertain?** It returns WATCH or REVIEW with
the prediction interval rather than forcing a binary call, and if the measurement
record is incomplete it returns MEASUREMENT_INVALID and asks for a re-test.

**11. Does it generalise to another parameter?** The pipeline is
parameter-agnostic: only the limit and the degradation direction are configured
per parameter. We demonstrate on five parameters with different units, scales and
distributions, including two that are log-normal.

**12. What is your false negative rate?** Bounded by conformal risk control, and
we validated the bound rather than asserting it: it holds at every alpha from 1%
to 20% across 40 lot-grouped calibration splits, with no statistically
significant violation (all p above 0.28). [`results/conformal.md`,
`results/fig8_conformal_guarantee.svg`]

## Four more questions our own results have raised

**13. Why does your ablation stop improving at L3?** Adding the machine-learning
rung changes recall by 2 parts out of 427, and the confidence intervals overlap
almost entirely. Our reading is that the robust multivariate layer already
captures the structure in this data, and we report the plateau rather than
claiming a gain we cannot support. [`results/ablation_cumulative.csv`]

**14. Why do only 26.7% of your flagged parts get a counterfactual?** Because a
part that is anomalous only in the joint combination of measurements has no
single measurement you could change to make it normal, which is the definition of
that defect class. The split proves it: 29.9% for parts flagged on a single
measurement against 2.3% for parts flagged on the combination.
[`results/explainability.md`]

**15. Why is the Type Va trap only fully passed 55% of the time?** Type Va is an
elevated but healthy part, and 45% of the time our level rule flags it, which is
literally true but the wrong disposition. 81.7% avoid rejection when WATCH is
counted, and we report the 55% because it is the honest measure of what the level
rule costs us. [`results/explainability_metrics.json`]

**16. How do you know the synthetic defects are realistic?** The defect classes
implement a published finding: the ITC 2020 study of real automotive returns
found escaped defects sit at the centre of the distribution on every individual
parameter, which is exactly our Type III. We also verified that a naive detector
cannot shortcut them, and we fixed a construction artifact we found ourselves
when trajectories turned out 3 to 5 times too smooth. [`results/CHANGED_NUMBERS.md`]

---

# Section 6: Known weaknesses, stated plainly

Rehearse these. Each has an honest answer, not a deflection.

**1. Validation is entirely synthetic.** There is no public dataset with this
shape (multiple parameters, four burn-in checkpoints, lot structure, known
latent defects), so Module B in particular has no external analogue to test
against.
*Answer:* "Correct, and we state it as a pilot dependency rather than a solved
problem. What we can defend is that the generator is grounded in the Arrhenius
model, the defect classes implement a published industrial finding, and we built
in traps that punish over-flagging so the numbers are not self-flattering."

**2. Survivorship bias is measured but not fixed.** A model trained only on parts
that survive to 168 hours under-predicts the true final value of parts that
failed and were pulled by 30.8 uA on Iddq, which is 62% of the entire datasheet
limit. Using the datasheet limit as a stand-in target recovers only 0.6 uA of it.
*Answer:* "We quantified it rather than noting it, which is why we can tell you
it is 62% of the limit. The proper fix is a censored regression loss that treats
those parts as 'known to exceed' rather than imputing a value, and that is
scoped as the next step."

**3. Type VI has zero parts in the test set.** All six shifted batches fall early
in the production sequence and our split is chronological.
*Answer:* "We label every Type VI figure as all-lots rather than moving the split
to flatter ourselves. One of the six does fall in the validation block, so the
effect is confirmed on non-training data."

**4. Type III has only 60 parts in total, 8 in the test lots.** That is a
deliberate choice, since 500 defective parts per million is the realistic rate,
but it means per-tier test-set rates are not informative.
*Answer:* "The rate is realistic and we would rather keep it than inflate the
class for better error bars. We report all-lots figures with confidence
intervals so the uncertainty is visible."

**5. Prediction intervals were miscalibrated before the conformal layer.** The
nominal 95% intervals actually delivered 92.8% to 95.4% coverage depending on
parameter, which is optimistic: the interval is too narrow.
*Answer:* "We measured it and reported it rather than presenting the nominal
number. This is exactly the gap conformal calibration closes, which is why the
guarantee layer is not decoration."

**6. The Type Va trap is only fully passed 55% of the time.** See question 15.
*Answer:* "That is the measured cost of our level rule, and it is why we report a
yield-loss curve rather than a single operating point."

**7. Conformal calibration excludes training lots entirely.** That is more
conservative than strictly necessary and costs calibration sample size.
*Answer:* "A part the detector was fitted on is not exchangeable with one it has
never seen, and exchangeability is the entire assumption the guarantee rests on.
We chose the conservative reading; using validation lots alone would give a
tighter bound and is arguable."

**8. Layer 2 attribution is model-dependent.** See Finding 4.
*Answer:* "Which is why the deterministic rule text is the layer we put in front
of an inspector, and the SHAP output is labelled as supporting evidence on the
report page itself."

---

## Where everything lives

| What | Path |
|---|---|
| Cumulative ladder | `results/ablation_cumulative.csv`, `.md` |
| Head-to-head methods | `results/ablation.csv`, `results/ablation.md` |
| Per defect type and tier | `results/per_type.csv`, `results/per_type_cumulative.csv` |
| Module B | `results/module_b.csv`, `results/module_b.md` |
| Safety slopes | `results/safety_slopes_union.csv` |
| Conformal | `results/conformal.md`, `results/conformal_summary.csv` |
| Explainability | `results/explainability.md`, `results/explainability_metrics.json` |
| What changed from v1.0 | `results/CHANGED_NUMBERS.md` |
| Split sizes | `results/split_counts.csv` |
| Disposition reports | `reports/*.pdf` (and `.txt` for the rule text alone) |
| Plots | `results/fig1` to `fig10`, all SVG |
