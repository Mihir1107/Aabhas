# Module A ablation, L0 to L4

Dataset `dataset-v1.0` (commit 8ca44bc), unmodified. 120000 parts, 2100 defective (1.75%).

**Protocol.** Lot-grouped split, never row-wise: train 144 lots (LOT000..LOT143), val 48 lots (LOT144..LOT191), test 48 lots (LOT192..LOT239). Detectors are fitted on GOOD parts of TRAIN lots; all numbers below are on TEST lots only (n=24000, 427 defective).

**Definitions.** Defective = I, II, III, IV, Vb. Good = normal parts plus the Va / VI / VII traps, so flagging a trap counts as yield loss. Yield loss = fraction of GOOD rejected. Escape rate = fraction of DEFECTIVE passed. Accuracy is never reported.

**Every DPAT number below is a UNION across the 5 parameters and 4 checkpoints** unless stated otherwise. Single-parameter numbers are roughly 30 points lower; see the reconciliation section.

## Main ablation (operating point: 7% yield loss = 93% yield)

| rung | method | features | recall@93%yield | PR_AUC | AUROC | escape_rate_% | cost |
|---|---|---|---|---|---|---|---|
| L0 | Static datasheet limits | 5 params, observed checkpoints | n/a | n/a | n/a | 100.00 | 427000.00 |
| L1a | DPAT static, MAD | 5 params x 4 cp, union, k=6 | 47.07 | 0.19 | 0.81 | 52.93 | 227651.00 |
| L1b | DPAT dynamic, MAD | 5 params x 4 cp, union, k=6 | 54.33 | 0.26 | 0.84 | 45.67 | 196651.00 |
| L1c | DPAT dynamic, all 4 estimators | union of MAD/IQR/p1p99/classical | 54.80 | 0.30 | 0.84 | 45.20 | 194651.00 |
| L2 | DPAT on trajectory features | level z + 63 trajectory z | 80.33 | 0.72 | 0.90 | 19.67 | 85651.00 |
| L3a | Mahalanobis + MCD | 5 raw params, per lot x checkpoint | 65.34 | 0.38 | 0.86 | 34.66 | 149651.00 |
| L3b | PCA T2 + Q-residual | 83-feature design matrix | 90.16 | 0.79 | 0.95 | 9.84 | 43651.00 |
| L3b-Q | PCA Q-residual alone | 83-feature design matrix | 81.73 | 0.64 | 0.91 | 18.27 | 79651.00 |
| L3c | kNN distance | 83-feature design matrix | 87.59 | 0.83 | 0.94 | 12.41 | 54651.00 |
| L4a | Isolation Forest | 83-feature design matrix | 67.68 | 0.20 | 0.88 | 32.32 | 139651.00 |
| L4b | Autoencoder | 83-feature design matrix | 90.63 | 0.88 | 0.96 | 9.37 | 41651.00 |
| L4b' | LOF | 83-feature design matrix | 91.10 | 0.89 | 0.96 | 8.90 | 39651.00 |
| L4c | Union ensemble | 5 members, max percentile rank | 93.68 | 0.71 | 0.98 | 6.32 | 28651.00 |

`cost` = 1000 x n_FN + 1 x n_FP at the 7% operating point. L0 has no continuous score, so its threshold-free metrics are n/a rather than faked.

## Cost-optimal operating point (1000:1)

| rung | method | cost_min | cost_min_YL_% | cost_min_recall_% |
|---|---|---|---|---|
| L1a | DPAT static, MAD | 22175.00 | 94.07 | 100.00 |
| L1b | DPAT dynamic, MAD | 22134.00 | 93.90 | 100.00 |
| L1c | DPAT dynamic, all 4 estimators | 22407.00 | 95.05 | 100.00 |
| L2 | DPAT on trajectory features | 23565.00 | 99.97 | 100.00 |
| L3a | Mahalanobis + MCD | 22963.00 | 97.41 | 100.00 |
| L3b | PCA T2 + Q-residual | 23569.00 | 99.98 | 100.00 |
| L3b-Q | PCA Q-residual alone | 23425.00 | 99.37 | 100.00 |
| L3c | kNN distance | 23573.00 | 100.00 | 100.00 |
| L4a | Isolation Forest | 23573.00 | 100.00 | 100.00 |
| L4b | Autoencoder | 21880.00 | 67.37 | 98.59 |
| L4b' | LOF | 23573.00 | 100.00 | 100.00 |
| L4c | Union ensemble | 10028.00 | 25.57 | 99.06 |

## DPAT estimator comparison

| estimator | recall_fixed_k6_% | yield_loss_fixed_k6_% | recall_matched_overkill_% |
|---|---|---|---|
| MAD | 12.88 | 0.17 | 12.88 |
| IQR | 12.18 | 0.16 | 13.11 |
| p1p99 | 10.07 | 0.01 | 22.25 |
| classical | 9.60 | 0.05 | 17.33 |

## Cross-lot stability of the estimators

| parameter | estimator | sigma_CV_% | limit_CV_% | corr(contamination, sigma) |
|---|---|---|---|---|
| iddq_ua | MAD | 22.36 | 21.83 | -0.12 |
| iddq_ua | IQR | 22.45 | 21.88 | -0.13 |
| iddq_ua | p1p99 | 23.71 | 21.91 | -0.01 |
| iddq_ua | classical | 20.27 | 20.44 | -0.06 |
| leakage_na | MAD | 24.80 | 24.49 | -0.14 |
| leakage_na | IQR | 24.54 | 24.32 | -0.14 |
| leakage_na | p1p99 | 25.61 | 23.85 | 0.03 |
| leakage_na | classical | 21.66 | 22.11 | -0.01 |

## Leave-one-lot-out (10 held-out lots, seeded)

| method | recall mean | recall std | PR_AUC mean | PR_AUC std |
|---|---|---|---|---|
| L1_dynamic_MAD | 61.25 | 14.57 | 0.30 | 0.15 |
| L3a_MCD | 66.72 | 7.14 | 0.42 | 0.14 |
| L4d_AutoEnc | 91.88 | 10.72 | 0.85 | 0.12 |
