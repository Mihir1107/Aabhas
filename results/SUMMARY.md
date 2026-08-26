# Module A, L0 to L4 — headline numbers

Dataset `dataset-v1.0` (commit `8ca44bc`), unmodified. 120,000 parts in 240
lots; 2,100 defective (1.75%). Lot-grouped split, never row-wise: train
LOT000–143, validate LOT144–191, test LOT192–239. All figures below are on
**test lots only** (24,000 parts, 427 defective) unless stated.

Defective = I, II, III, IV, Vb. Good = normal parts **plus the Va / VI / VII
traps**, so flagging a trap is counted as yield loss. Operating point is 7%
yield loss (93% yield, the ITC 2020 reference). Accuracy is not reported.

**Every DPAT number is a union across 5 parameters × 4 checkpoints.** The
single-parameter equivalent is roughly 30 points lower.

## The six numbers

1. **L0, static datasheet limits: 0.0% recall, 100% escape rate.** Zero on
   every injected type, defect and trap alike. This confirms the generator's
   in-spec guarantee has no hole, and it is the strawman the whole ladder is
   measured against.

2. **L1, AEC-Q001 dynamic PAT with MAD: 54.3% recall at 93% yield**
   (PR-AUC 0.264, escape rate 45.7%). Static PAT on the same estimator gets
   47.1%. This is the industry baseline we claim to beat.

3. **L4c, union ensemble: 93.7% recall at 93% yield** (escape rate 6.3%,
   AUROC 0.982). Best single detector is LOF at 91.1% (PR-AUC 0.893); the
   autoencoder is 90.6% (PR-AUC 0.882).

4. **Escape rate falls from 45.7% (L1) to 6.3% (L4c) at the same 7% yield
   loss** — a 7.2× reduction in test escapes at identical yield cost. Cost at
   1000:1 falls from 196,651 to 28,651.

5. **Dynamic PAT vs static PAT on a lot-wide process shift (Type VI, n=3,000):
   static flags 59%, dynamic flags 6%.** Static PAT scraps 1,770 good parts
   over a process excursion that is not a part defect. This is the single
   cleanest justification for dynamic limits in the whole study.

6. **NEGATIVE RESULT — L4 does not beat L3 on the mild tier.** On mild Type IV
   (n=141), Mahalanobis+MCD catches **90.1%** [84.0–94.0] while the best L4
   method reaches 36.2% and Isolation Forest reaches 1.4%. On mild Type III
   (n=21) the PCA Q-residual leads at 61.9% against 28.6–33.3% for L4. The ML
   rungs do not close the gap they were supposed to close.

## The ablation

| Rung | Method | Recall @93% yield | PR-AUC | AUROC | Escape rate | Cost |
|---|---|---|---|---|---|---|
| L0 | Static datasheet limits | n/a | n/a | n/a | 100.00% | 427,000 |
| L1a | DPAT static, MAD | 47.07% | 0.192 | 0.815 | 52.93% | 227,651 |
| L1b | DPAT dynamic, MAD | 54.33% | 0.264 | 0.840 | 45.67% | 196,651 |
| L1c | DPAT dynamic, all 4 estimators | 54.80% | 0.304 | 0.843 | 45.20% | 194,651 |
| L2 | DPAT on trajectory features | 80.33% | 0.719 | 0.897 | 19.67% | 85,651 |
| L3a | Mahalanobis + MCD | 65.34% | 0.383 | 0.862 | 34.66% | 149,651 |
| L3b | PCA T² + Q-residual | 90.16% | 0.791 | 0.946 | 9.84% | 43,651 |
| L3c | kNN distance | 87.59% | 0.830 | 0.941 | 12.41% | 54,651 |
| L4a | Isolation Forest | 67.68% | 0.196 | 0.877 | 32.32% | 139,651 |
| L4b | Autoencoder | 90.63% | 0.882 | 0.964 | 9.37% | 41,651 |
| L4c | Union ensemble | 93.68% | 0.712 | 0.982 | 6.32% | 28,651 |

The ladder climbs monotonically in recall, **but not by rung as designed**:
L3b/L3c already reach 88–90%, and L4 adds 1–3 points on the aggregate while
losing badly on the mild tiers. See the caveats.

## What must be said out loud

**A dataset defect, reported and not fixed.** Type III parts have a
construction fingerprint: their trajectories are 3–5× smoother than a real
part's, because the generator places them at deterministic lot quantiles per
checkpoint and they inherit no independent measurement noise. A trivial
"too smooth" rule catches **45.8% of Type III at 1% yield loss** using no
multivariate reasoning. Every Type III figure above is therefore an upper
bound, and L2's Type III performance in particular is suspect. The generator
was not modified. Fix for a future version: add per-checkpoint noise after
quantile placement and re-verify the band.

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

**The union ensemble mostly buys recall, and it is worth it here.** It adds
2.6 points of recall over the best single detector (93.7% vs 91.1%) at the same
7% budget, and cuts cost from 39,651 to 28,651. But its PR-AUC (0.712) is
*lower* than LOF's (0.893) — the union is good at the operating point and worse
as a ranking. If a downstream stage needs a ranked worklist rather than a
binary call, use LOF or the autoencoder, not the union.

**Type VI is absent from the test set.** All six lot-shift lots fall in
LOT001–LOT146, so the chronological split leaves none in test. The Type VI
figures quoted above are computed on all lots. With only six trap lots placed
at random, a chronological split can miss them entirely; a future split should
be stratified on trap-lot presence.

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

## Files

`ablation.md` / `ablation.csv` — main table. `per_type.md` / `per_type.csv` —
per type and per severity tier with Wilson intervals. `all_methods.csv` — every
variant including contaminated-fit and leak comparisons.
`estimator_comparison.csv`, `estimator_stability.csv`,
`leave_one_lot_out.csv`. Figures `fig1`–`fig4b` as SVG.
