# SIH26170 — headline numbers (Module A, Module B, L5 conformal, explainability)

Dataset **`dataset-v1.1`** (commit `1ac9e59`). See `CHANGED_NUMBERS.md` for every figure that moved from v1.0. 120,000 parts in 240
lots; 2,100 defective (1.75%). Lot-grouped split, never row-wise: train
LOT000–143, validate LOT144–191, test LOT192–239. All figures below are on
**test lots only** (24,000 parts, 427 defective) unless stated.

Defective = I, II, III, IV, Vb. Good = normal parts **plus the Va / VI / VII
traps**, so flagging a trap is counted as yield loss. Operating point is 7%
yield loss (93% yield, the ITC 2020 reference). Accuracy is not reported.

**Every DPAT number is a union across 5 parameters × 4 checkpoints.** The
single-parameter equivalent is roughly 30 points lower.

## The six numbers

1. **L0, static datasheet limits: 0.0% recall, 100% escape rate.** Zero on every
   injected type, defect and trap alike. The strawman the ladder is measured
   against, and confirmation that the in-spec guarantee holds.

2. **Module A, AEC-Q001 dynamic PAT with MAD: 54.3% recall at 93% yield**
   (PR-AUC 0.264, escape 45.7%). Union across 5 parameters x 4 checkpoints.
   This is the industry baseline we claim to beat.

3. **Module A cumulative C3 (L0+L1+L2+L3): 92.7% recall at 93% yield**
   (escape 7.3%). Adding L4 gives 92.3% — 2 defective parts fewer out of 427,
   so the ladder plateaus at L3. Escape rate falls 45.7% -> 7.3% at identical
   yield cost.

4. **Module B early-warning MAE on v168, using 0 h and 24 h only:
   0.61 uA (Iddq), 1.30 nA (leakage), 0.068 ns (delay), 0.68 mA (supply),
   0.84 mV (vth shift)** — LightGBM, test lots. That is a 4.7x improvement on
   the linear-extrapolation strawman (2.87 uA on Iddq). Adding the 96 h
   checkpoint improves it a further 21-28%, and the two modes are never blended.

5. **Conformal risk control holds at every alpha from 1% to 20%, with no
   statistically significant violation** (one-sided t-test over 40 lot-grouped
   calibration splits; largest overshoot +0.5 SE). A **1% guaranteed escape
   rate costs 42.7% yield loss** on the fused detector; 5% costs 11.6%; 10%
   costs 4.4%. The escape rate is bounded, not merely low.

6. **Explainability, made quantitative — the third scored axis.** Reason
   correctness **73–100%** by defect class (weakest Type II at 73.3%);
   explanation completeness **69.5%** for flagged parts and **100%** for
   passes; counterfactual validity **100%** of those offered, offered for
   26.7%; explanation stability under measurement noise **91.5%** overall and
   **100%** on Types II, III, IV and VII. Five example QA disposition PDFs in
   `reports/`.

7. **NEGATIVE RESULTS, four of them.** (a) L4 does not beat L3: the cumulative
   ladder plateaus, and on mild Type IV Mahalanobis+MCD still beats every L4
   method. (b) LightGBM does not beat Huber on MAE — they tie, and Huber is the
   better default on interpretability and cost. (c) Module B's 95% prediction
   intervals are **miscalibrated**, delivering 0.928-0.954 coverage against a
   nominal 0.95; the conformal layer is what closes that gap. (d) SHAP on
   LightGBM and an equally accurate Huber model **disagree about which features
   matter** (rho=0.617), because `corr(v0, v24) = 0.989` makes the attribution
   unidentifiable — which is why the deterministic Layer 1 rule text, not SHAP,
   is what goes in front of an inspector.

## The ablation

| Rung | Method | Recall @93% yield | PR-AUC | AUROC | Escape rate | Cost |
|---|---|---|---|---|---|---|
| L0 | Static datasheet limits | n/a | n/a | n/a | 100.00% | 427,000 |
| L1a | DPAT static, MAD | 47.31% | 0.192 | 0.815 | 52.69% | 226,651 |
| L1b | DPAT dynamic, MAD | 54.33% | 0.264 | 0.841 | 45.67% | 196,651 |
| L1c | DPAT dynamic, all 4 estimators | 54.80% | 0.304 | 0.845 | 45.20% | 194,651 |
| L2 | DPAT on trajectory features | 79.16% | 0.708 | 0.891 | 20.84% | 90,651 |
| L3a | Mahalanobis + MCD | 64.87% | 0.381 | 0.861 | 35.13% | 151,651 |
| L3b | PCA T² + Q-residual | 88.99% | 0.781 | 0.941 | 11.01% | 48,651 |
| L3c | kNN distance | 86.42% | 0.818 | 0.937 | 13.58% | 59,651 |
| L4a | Isolation Forest | 68.15% | 0.199 | 0.878 | 31.85% | 137,651 |
| L4b | Autoencoder | 89.46% | 0.856 | 0.965 | 10.54% | 46,651 |
| L4c | Union ensemble | 92.97% | 0.709 | 0.980 | 7.03% | 31,651 |

Cumulative version (each rung contains everything below it, fused by max
robust-z), which is the one that tells the build story:

| Rung | Contents | Recall @93% yield | PR-AUC | Escape rate | Cost |
|---|---|---|---|---|---|
| C0 | L0 alone | n/a | n/a | 100.00% | 427,000 |
| C1 | L0 + L1 | 54.33% | 0.264 | 45.67% | 196,651 |
| C2 | + L2 | 79.86% | 0.711 | 20.14% | 87,651 |
| C3 | + L3 | **92.74%** | 0.800 | 7.26% | 32,651 |
| C4 | + L4 | 92.27% | 0.805 | 7.73% | 34,651 |

**It is not monotone.** C4 sits 0.47 points below C3, which is 2 parts of 427 —
inside sampling noise, so the reading is "plateaus at L3", not "L4 hurts".

## What must be said out loud

**The v1.0 Type III artifact is fixed.** v1.0 trajectories were 3–5× too
smooth and a trivial "too smooth" rule recovered 45.8% of the class. v1.1 gives
Type III the same measurement noise every other part carries and holds the
anomaly direction fixed across checkpoints (it used to flip, which was the
larger of the two causes). Roughness ratio is now 1.040, KS p = 0.896, and the
"too smooth" rule catches 1.7% against a 1.0% chance rate. Assertion A8
enforces it, two-sided. Full before/after in `CHANGED_NUMBERS.md`.

**A label leak, found and removed.** `measurement_status == PULLED_FAILED`
is legitimately observable, but P(defective | pulled) = 1.000 in this dataset
because the generator draws hard failures only from Types I and II. It is worth
17.1% recall at 0% yield loss. It is excluded from every number above; the
same autoencoder with it included scores 93.7% instead of 90.6%, so the leak is
worth 3.1 points.

**A feature bug, found and fixed.** `r1` and `relative_drift` divided by
`|v0| + 1e-9`. For `vth_shift_mv`, which is centred on zero, this divided by
~0 and produced max|z| = 107,888 on ordinary good parts, collapsing L2's PR-AUC
to 0.080. Relative drift is only meaningful for a ratio-scale quantity, so it
is no longer computed for two-sided parameters, and the epsilon is tied to the
parameter's own scale. L2 PR-AUC went 0.080 → 0.719.

**We contradict ITC 2020 on Isolation Forest.** They report Isolation Forest
and the autoencoder leading, with the Gaussian model trailing. Here Isolation
Forest is the *worst* L4 method (67.7%, PR-AUC 0.196) and catches **0.0% of
Type III**. This is not a bug and it is not reordered: an axis-parallel
isolation method cannot isolate a point that is central on every individual
axis, which is the definition of a centre-hider. Our dataset deliberately
over-weights that class relative to real production data, so the ordering
differs for a structural reason. Tested on both a narrow 20-feature and a wide
83-feature matrix; Isolation Forest is worse on the narrow one (41.7%), so it
is not a dimensionality artifact.

**Fusion buys recall and costs ranking quality, and score-level fusion does
not fix it.** Max robust-z fusion reaches 92.3% recall against LOF's 90.2%, but
PR-AUC 0.794 against LOF's 0.882. Moving from rank-space to z-space recovers
+0.085 of PR-AUC, so granularity loss is real — but it is not the main effect.
All 20 false positives in the top 200 fused scores come from one member
(`L3b_Q`): under a max rule the ranking inherits every member's tail false
positives. Greedy member selection on the *validation* lots picks LOF alone.
Fusion and single-detector are different products: fusion for recall at a fixed
operating point, LOF or the autoencoder for a ranked inspector worklist.

**Type VI is absent from the test set, and cannot be moved there.** Five of
the six lot-shift lots fall in the first 85 lots, so getting two into test would
need a 35/20/45 split and getting even one needs 40/20/40, which halves the
training data. The split stays 60/20/20 and **every Type VI figure in this
repository is an all-lots number, labelled as such**. LOT146 does fall in the
validation block, giving 500 Type VI parts in a non-training holdout for the
fitting-free L1 detectors. Counts per split in `split_counts.csv`.

**Small-n warning.** Type III has 60 parts in total and only 8 in the test
lots (2 mild, 4 moderate, 2 severe). Per-tier Type III rates carry Wilson
intervals of roughly ±20 points even pooled over all lots, and per-tier
*test-only* rates are uninformative. All per-type tables therefore report n and
95% Wilson intervals, and the headline per-tier figures are computed on all
lots (legitimate, because no detector was fitted on defective parts).

## Assumptions a domain expert could challenge

1. **Pulled parts are counted as defective in the recall denominator**, per the
   stated definition. They are 17% of all defectives and never ship, so an
   argument exists that they belong in neither numerator nor denominator.
   Excluding them would lower every recall figure.
2. **Unsupervised detectors are fitted on known-good parts.** Production has no
   such oracle, so this is a favour we grant the ML rungs. Measured both ways,
   it turns out not to matter and in one case to hurt: fitting Isolation Forest
   on all train parts (1.75% contaminated) scores **74.9%** against **67.7%**
   for the clean fit, and the PCA Q-residual loses only 2 points (81.7% → 79.4%).
   The clean-reference assumption is therefore not doing the work, which is the
   reassuring answer. Both variants are in `all_methods.csv`.
3. **The 7% yield-loss operating point** comes from ITC 2020's 93% yield goal.
   It is very loose for a space-grade screen. At 1% yield loss the ladder reads
   L1 28.3%, L2 74.9%, L4b (autoencoder) 87.1% — the gap between L1 and the ML
   rungs widens at the tighter, more realistic operating point.
4. **MCD support fraction 0.9, contamination 0.02, OCSVM nu 0.02**, all set
   explicitly rather than by default. OCSVM is fitted on an 8,000-part
   subsample because RBF training is O(n²).
5. **The autoencoder is sklearn's MLPRegressor fitted X→X** (48,12,48), because
   torch is not installed here. It is a bottlenecked reconstruction, but it is
   not a modern deep autoencoder.

## Module B and L5

Full detail in `module_b.md` and `conformal.md`. Three things worth carrying
into the deck:

**The strawman check passed, and it is worth showing.** Linear extrapolation of
the 0->24 h slope over-predicts v168 on every parameter, for 68.8-79.5% of good
parts, over-shooting true drift by 80-138%. That is sub-linear saturating
degradation made visible, and it is why rung 2 (power law) halves the error.

**Survivorship bias, quantified in absolute units.** A model trained only on
parts that survived to 168 h under-predicts the true 168 h value of pulled parts
by **30.8 uA on Iddq, which is 62% of the entire datasheet limit** — and by
41% of limit on leakage, 27% on vth shift. Those are the parts whose drift
matters most.

**The safety slope has to be lot-relative.** Rule (d) tested against the
datasheet limit is inert (1.4% recall), because every injected defect is inside
spec by construction. Against a lot-derived L_safe — the AEC-Q001 dynamic PAT
limit at 168 h — the same rule gives 17.8% recall at 0.15% yield loss. Module
B's value is not predicting limit violations, of which there are none. It is
predicting abnormal drift *relative to peers*: the same static-versus-dynamic
argument Module A makes, one derivative up.

**Framing for the product-assurance question:**

> MIL-STD-883 already uses PDA (Percent Defective Allowable) as a statistical
> bound on lot quality that ISRO accepts as a screening criterion. Conformal
> risk control provides the same class of statistically defensible bound,
> applied to the escape rate of the AI system itself. It makes the model
> auditable inside a product-assurance framework that already exists.

## Explainability

Full detail in `explainability.md`. Three things for the deck:

**The deliverable is a document, not a plot.** One-page QA disposition report
per part: Layer 1 deterministic evidence where every number cites its
comparison basis, the trajectory against the lot's robust envelope with the
forecast band, detector attribution, SHAP, a validated counterfactual, a
mechanism hypothesis explicitly labelled as a hypothesis, and a sign-off block.

**The two examples that matter are the non-rejections.** Any system can explain
a rejection. Case 04 explains why a benign high-but-stable part was *not*
rejected; case 05 says *"this is your oven, not your part"* and raises a fixture
investigation instead of scrapping 
components. Six decision tiers, not a binary:
PASS, WATCH, REVIEW, REJECT, MEASUREMENT_INVALID, FIXTURE_SUSPECT.

**Only two mechanism mappings are marked well-founded**, and both rest on the
same physics: heat raises leakage and *slows* a part, while the process corner
makes leaky parts *fast*, so a fixture artifact and a process outlier move the
leakage/delay correlation in opposite directions. Everything else in the table
is labelled *plausible* or *speculative* on the page, and the joint-anomaly
mapping is labelled *speculative* because a joint outlier says a part does not
resemble its peers, not why.

## Files

`ablation.md` / `ablation.csv` — main table. `per_type.md` / `per_type.csv` —
per type and per severity tier with Wilson intervals. `all_methods.csv` — every
variant including contaminated-fit and leak comparisons.
`estimator_comparison.csv`, `estimator_stability.csv`,
`leave_one_lot_out.csv`. Module B: `module_b.md`/`.csv`,
`module_b_per_type.csv`, `module_b_by_lot.csv`, `safety_slopes_union.csv`.
Conformal: `conformal.md`, `conformal_sweep.csv`,
`conformal_significance.csv`. Explainability: `explainability.md`, `explainability_metrics.json`,
`explainability_reason.csv`, `explainability_stability.csv`,
`report_examples.csv`, and five disposition PDFs in `reports/`.
Figures `fig1`–`fig10` as SVG.
