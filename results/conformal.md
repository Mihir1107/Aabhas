# L5: Conformal Risk Control

Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, ICLR 2024
(arXiv:2208.02814). Distribution-free, finite-sample control of the expected
value of a monotone loss. Here the loss is the false-negative indicator, so the
guarantee bounds **the escape rate of the AI system itself**.

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
| ModuleA_LOF | 0.0100 | 0.0096 | 0.0010 | 0.0022 | 0.0171 | 99.0372 | 86.6112 | holds |
| ModuleA_LOF | 0.0200 | 0.0195 | 0.0013 | 0.0086 | 0.0305 | 98.0541 | 66.0033 | holds |
| ModuleA_LOF | 0.0300 | 0.0294 | 0.0017 | 0.0141 | 0.0425 | 97.0558 | 52.4801 | holds |
| ModuleA_LOF | 0.0500 | 0.0496 | 0.0022 | 0.0320 | 0.0685 | 95.0392 | 37.3615 | holds |
| ModuleA_LOF | 0.0750 | 0.0750 | 0.0025 | 0.0568 | 0.0925 | 92.5041 | 18.9164 | holds |
| ModuleA_LOF | 0.1000 | 0.0975 | 0.0022 | 0.0790 | 0.1144 | 90.2460 | 8.3089 | holds |
| ModuleA_LOF | 0.1500 | 0.1466 | 0.0034 | 0.1190 | 0.1734 | 85.3426 | 1.1318 | holds |
| ModuleA_LOF | 0.2000 | 0.1943 | 0.0036 | 0.1676 | 0.2225 | 80.5737 | 0.0795 | holds |
| ModuleA_fused_maxz | 0.0100 | 0.0080 | 0.0009 | 0.0022 | 0.0152 | 99.1975 | 42.6647 | holds |
| ModuleA_fused_maxz | 0.0200 | 0.0177 | 0.0012 | 0.0109 | 0.0281 | 98.2263 | 23.8234 | holds |
| ModuleA_fused_maxz | 0.0300 | 0.0290 | 0.0018 | 0.0164 | 0.0461 | 97.0996 | 18.2379 | holds |
| ModuleA_fused_maxz | 0.0500 | 0.0498 | 0.0025 | 0.0296 | 0.0698 | 95.0207 | 11.6171 | holds |
| ModuleA_fused_maxz | 0.0750 | 0.0765 | 0.0028 | 0.0512 | 0.0987 | 92.3542 | 7.0078 | holds |
| ModuleA_fused_maxz | 0.1000 | 0.0989 | 0.0025 | 0.0788 | 0.1169 | 90.1065 | 4.4417 | holds |
| ModuleA_fused_maxz | 0.1500 | 0.1468 | 0.0026 | 0.1312 | 0.1686 | 85.3161 | 1.6102 | holds |
| ModuleA_fused_maxz | 0.2000 | 0.1944 | 0.0034 | 0.1678 | 0.2224 | 80.5567 | 0.9691 | holds |
| ModuleB_upper_bound_margin | 0.0100 | 0.0105 | 0.0010 | 0.0040 | 0.0187 | 98.9542 | 99.9500 | holds |
| ModuleB_upper_bound_margin | 0.0200 | 0.0202 | 0.0015 | 0.0097 | 0.0285 | 97.9774 | 99.7876 | holds |
| ModuleB_upper_bound_margin | 0.0300 | 0.0310 | 0.0017 | 0.0194 | 0.0427 | 96.9015 | 99.4293 | holds |
| ModuleB_upper_bound_margin | 0.0500 | 0.0498 | 0.0026 | 0.0326 | 0.0633 | 95.0229 | 98.3837 | holds |
| ModuleB_upper_bound_margin | 0.0750 | 0.0737 | 0.0030 | 0.0534 | 0.0913 | 92.6291 | 95.6940 | holds |
| ModuleB_upper_bound_margin | 0.1000 | 0.0992 | 0.0040 | 0.0712 | 0.1256 | 90.0769 | 91.4863 | holds |
| ModuleB_upper_bound_margin | 0.1500 | 0.1499 | 0.0041 | 0.1176 | 0.1750 | 85.0099 | 77.8326 | holds |
| ModuleB_upper_bound_margin | 0.2000 | 0.1982 | 0.0044 | 0.1649 | 0.2353 | 80.1803 | 62.9466 | holds |

`verdict` is a one-sided t-test of H0: E[FNR] <= alpha across the 40 repeats.
**No alpha shows a statistically significant violation for any target** (all
p > 0.28; the largest overshoot is +0.5 SE). Mean empirical FNR tracks alpha
almost exactly — for the Module A LOF detector, 0.0096 / 0.0195 / 0.0294 /
0.0496 / 0.0750 / 0.0975 / 0.1466 / 0.1943 against alphas of 0.01 / 0.02 /
0.03 / 0.05 / 0.075 / 0.10 / 0.15 / 0.20.

Note the p10-p90 columns. The guarantee is on the **expected** FNR, so roughly
half of individual splits sit above alpha by design. A plot showing one line
under the diagonal would be an anecdote; the spread is the evidence, and it is
why 40 repeats are run rather than one.

## The honest counterpart: what the guarantee costs

| alpha (guaranteed escape rate) | Module A LOF | Module A fused | Module B bound |
|---|---|---|---|
| 1% | 86.6% yield loss | **42.7%** | 99.9% |
| 2% | 66.0% | **23.8%** | 99.8% |
| 3% | 52.5% | **18.2%** | 99.4% |
| 5% | 37.4% | **11.6%** | 98.4% |
| 7.5% | 18.9% | **7.0%** | 95.7% |
| 10% | 8.3% | **4.4%** | 91.5% |
| 15% | 1.1% | **1.6%** | 77.8% |
| 20% | 0.08% | 0.97% | 62.9% |

This is the tradeoff curve, and it is the deliverable that separates an
engineering proposal from a sales pitch. A 1% guaranteed escape rate is
achievable, and it costs **43% of the good parts** on the fused detector. At a
5% guarantee the cost is 11.6%. At 10%, 4.4%.

**The fused score dominates LOF everywhere below alpha = 0.15**, and by a wide
margin at tight alphas: 42.7% versus 86.6% yield loss for a 1% guarantee. This
*reverses* the ranking from the PR-AUC comparison in the Module A session,
where LOF led (0.882 vs 0.794). Both are correct: PR-AUC integrates the whole
ranking, whereas conformal at low alpha operates only in the extreme
high-recall tail. **Which detector is better depends on the operating regime,
and for a space-grade screen the regime that matters is the one where fusion
wins.**

**Module B's upper bound is not a viable standalone screen** — 99.9% yield loss
for a 1% guarantee. That is expected: it is a drift-specific rule, not a
general anomaly detector, and it is included here to show the conformal
machinery is target-agnostic rather than to propose it as a gate.

## Files

`conformal_sweep.csv` (all 40 repeats x 8 alphas x 3 targets),
`conformal_summary.csv`, `conformal_significance.csv`.
Figures: `fig8_conformal_guarantee.svg` (the important one),
`fig9_conformal_tradeoff.svg`.
