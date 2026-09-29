# L5: Conformal Risk Control

Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, ICLR 2024
(arXiv:2208.02814). Distribution-free, finite-sample control of the expected
value of a monotone loss. Here the loss is the false-negative indicator, so the
guarantee bounds **the expected escape rate of the AI system itself**. Individual
lots and calibration draws can exceed alpha; the bound is on the average.

## Why this matters for this problem

> MIL-STD-883 already uses PDA (Percent Defective Allowable) as a statistical
> bound on lot quality that ISRO accepts as a screening criterion. Conformal
> risk control provides the same class of statistically defensible bound,
> applied to the escape rate of the AI system itself. It makes the model
> auditable inside a product-assurance framework that already exists.

Every other team will say "we tuned for high recall". This says "our escape
rate is bounded, here is the bound, and here is the evidence it holds."

## Method

For a score where higher means more anomalous, flagging `score >= lambda`:

    L(lambda) = fraction of DEFECTIVE calibration parts with score < lambda
    lambda_hat = inf { lambda : (n*L(lambda) + 1) / (n+1) <= alpha }

`n` is the number of **defective** calibration parts, not the calibration set
size — at 1.75% prevalence that distinction matters, and getting it wrong would
silently loosen the correction. The `(n+1)` term is what makes the bound
finite-sample rather than asymptotic.

**Exchangeability is the whole assumption and it is where this can quietly
break.** Parts in one lot share a lot random effect, so they are exchangeable
with each other but not with parts from another lot. Every calibration split
here is therefore **by lot**, and both calibration and evaluation are drawn
from the VAL+TEST lots only — the training lots are excluded because a part the
detector was fitted on is not exchangeable with one it has never seen.

40 independent lot-grouped calibration/evaluation splits per alpha.

## Does the guarantee hold?

**Yes, at every alpha, for all three targets.**

| target | alpha | emp_FNR_mean | SE | emp_FNR_p10 | emp_FNR_p90 | recall_mean | yield_loss_mean | verdict |
|---|---|---|---|---|---|---|---|---|
| ModuleA_LOF | 0.0100 | 0.0096 | 0.0010 | 0.0022 | 0.0171 | 99.0372 | 86.6296 | holds |
| ModuleA_LOF | 0.0200 | 0.0195 | 0.0013 | 0.0086 | 0.0305 | 98.0541 | 66.1877 | holds |
| ModuleA_LOF | 0.0300 | 0.0296 | 0.0017 | 0.0141 | 0.0425 | 97.0387 | 52.5020 | holds |
| ModuleA_LOF | 0.0500 | 0.0498 | 0.0022 | 0.0320 | 0.0721 | 95.0242 | 37.0869 | holds |
| ModuleA_LOF | 0.0750 | 0.0748 | 0.0024 | 0.0569 | 0.0925 | 92.5218 | 18.7897 | holds |
| ModuleA_LOF | 0.1000 | 0.0978 | 0.0021 | 0.0811 | 0.1144 | 90.2240 | 8.3266 | holds |
| ModuleA_LOF | 0.1500 | 0.1462 | 0.0033 | 0.1190 | 0.1730 | 85.3827 | 1.1424 | holds |
| ModuleA_LOF | 0.2000 | 0.1946 | 0.0037 | 0.1681 | 0.2226 | 80.5419 | 0.0829 | holds |
| ModuleA_fused_maxz | 0.0100 | 0.0082 | 0.0009 | 0.0022 | 0.0152 | 99.1754 | 39.0190 | holds |
| ModuleA_fused_maxz | 0.0200 | 0.0176 | 0.0011 | 0.0109 | 0.0270 | 98.2427 | 23.8439 | holds |
| ModuleA_fused_maxz | 0.0300 | 0.0295 | 0.0017 | 0.0159 | 0.0461 | 97.0532 | 18.2183 | holds |
| ModuleA_fused_maxz | 0.0500 | 0.0500 | 0.0025 | 0.0296 | 0.0698 | 95.0037 | 11.6251 | holds |
| ModuleA_fused_maxz | 0.0750 | 0.0762 | 0.0029 | 0.0501 | 0.0987 | 92.3833 | 7.0087 | holds |
| ModuleA_fused_maxz | 0.1000 | 0.0990 | 0.0025 | 0.0779 | 0.1169 | 90.1025 | 4.4440 | holds |
| ModuleA_fused_maxz | 0.1500 | 0.1468 | 0.0026 | 0.1312 | 0.1686 | 85.3161 | 1.6117 | holds |
| ModuleA_fused_maxz | 0.2000 | 0.1948 | 0.0034 | 0.1679 | 0.2224 | 80.5227 | 0.9727 | holds |
| ModuleB_upper_bound_margin | 0.0100 | 0.0103 | 0.0010 | 0.0022 | 0.0182 | 98.9686 | 100.0000 | holds |
| ModuleB_upper_bound_margin | 0.0200 | 0.0196 | 0.0012 | 0.0111 | 0.0276 | 98.0364 | 100.0000 | holds |
| ModuleB_upper_bound_margin | 0.0300 | 0.0306 | 0.0014 | 0.0202 | 0.0416 | 96.9397 | 99.9986 | holds |
| ModuleB_upper_bound_margin | 0.0500 | 0.0491 | 0.0019 | 0.0351 | 0.0633 | 95.0878 | 99.8300 | holds |
| ModuleB_upper_bound_margin | 0.0750 | 0.0730 | 0.0023 | 0.0574 | 0.0914 | 92.7010 | 98.7676 | holds |
| ModuleB_upper_bound_margin | 0.1000 | 0.0983 | 0.0025 | 0.0817 | 0.1166 | 90.1735 | 95.4082 | holds |
| ModuleB_upper_bound_margin | 0.1500 | 0.1467 | 0.0036 | 0.1142 | 0.1754 | 85.3324 | 80.6900 | holds |
| ModuleB_upper_bound_margin | 0.2000 | 0.1959 | 0.0036 | 0.1686 | 0.2292 | 80.4051 | 60.7006 | holds |

`verdict` is a one-sided t-test of H0: E[FNR] <= alpha across the 40 repeats.
**No alpha shows a statistically significant violation for any target** (all
p > 0.34; the largest overshoot is +0.4 SE). Mean empirical FNR tracks alpha
almost exactly — for the Module A LOF detector, 0.0096 / 0.0195 / 0.0296 / 0.0498 /
0.0748 / 0.0978 / 0.1462 / 0.1946 against alphas of 0.01 / 0.02 /
0.03 / 0.05 / 0.075 / 0.10 / 0.15 / 0.20.

Note the p10-p90 columns. The guarantee is on the **expected** FNR, so roughly
half of individual splits sit above alpha by design. A plot showing one line
under the diagonal would be an anecdote; the spread is the evidence, and it is
why 40 repeats are run rather than one.

## The honest counterpart: what the guarantee costs

| alpha (guaranteed expected escape rate) | Module A LOF | Module A fused | Module B bound |
|---|---|---|---|
| 1% | 86.6% | **39.0%** | 100.0% |
| 2% | 66.2% | **23.8%** | 100.0% |
| 3% | 52.5% | **18.2%** | 100.0% |
| 5% | 37.1% | **11.6%** | 99.8% |
| 7.5% | 18.8% | **7.0%** | 98.8% |
| 10% | 8.3% | **4.4%** | 95.4% |
| 15% | 1.1% | **1.6%** | 80.7% |
| 20% | 0.1% | **1.0%** | 60.7% |

This is the tradeoff curve, and it is the deliverable that separates an
engineering proposal from a sales pitch. A 1% guaranteed escape rate is
achievable, and it costs **39% of the good parts** on the fused detector. At a
5% guarantee the cost is 11.6%. At 10%, 4.4%.

**The fused score dominates LOF everywhere below alpha = 0.15**, and by a wide
margin at tight alphas: 39.0% versus 86.6% yield loss for a 1% guarantee. This
*reverses* the ranking from the PR-AUC comparison in the Module A session,
where LOF led (0.882 vs 0.794). Both are correct: PR-AUC integrates the whole
ranking, whereas conformal at low alpha operates only in the extreme
high-recall tail. **Which detector is better depends on the operating regime,
and for a space-grade screen the regime that matters is the one where fusion
wins.**

**Module B's upper bound is not a viable standalone screen** — 100.0% yield loss
for a 1% guarantee. That is expected: it is a drift-specific rule, not a
general anomaly detector, and it is included here to show the conformal
machinery is target-agnostic rather than to propose it as a gate.

## Files

`conformal_sweep.csv` (all 40 repeats x 8 alphas x 3 targets),
`conformal_summary.csv`, `conformal_significance.csv`.
Figures: `fig8_conformal_guarantee.svg` (the important one),
`fig9_conformal_tradeoff.svg`.
