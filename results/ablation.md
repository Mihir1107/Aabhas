# Module A ablation, L0 to L4

Dataset `dataset-v1.2`. 120000 parts, 2100 defective (1.75%).

**Protocol.** Lot-grouped split, never row-wise: train 144 lots (LOT000..LOT143), val 48 lots (LOT144..LOT191), test 48 lots (LOT192..LOT239). Detectors are fitted on GOOD parts of TRAIN lots; all numbers below are on TEST lots only (n=24000, 427 defective).

**Definitions.** Defective = I, II, III, IV, Vb. Good = normal parts plus the Va / VI / VII traps, so flagging a trap counts as yield loss. Yield loss = fraction of GOOD rejected. Escape rate = fraction of DEFECTIVE passed. Accuracy is never reported.

**Every DPAT number below is a UNION across the 5 parameters and 4 checkpoints** unless stated otherwise. Single-parameter numbers are roughly 30 points lower; see the reconciliation section.

## Main ablation (operating point: 7% yield loss = 93% yield)

| rung | method | features | recall@93%yield | PR_AUC | AUROC | escape_rate_% | cost |
|---|---|---|---|---|---|---|---|
| L0 | Static datasheet limits | 5 params, observed checkpoints | n/a | n/a | n/a | 100.00 | 427000.00 |
| L1a | DPAT static, MAD | 5 params x 4 cp, union, k=6 | 47.31 | 0.19 | 0.81 | 52.69 | 226651.00 |
| L1b | DPAT dynamic, MAD | 5 params x 4 cp, union, k=6 | 54.33 | 0.26 | 0.84 | 45.67 | 196651.00 |
| L1c | DPAT dynamic, all 4 estimators | union of MAD/IQR/p1p99/classical | 54.80 | 0.30 | 0.84 | 45.20 | 194651.00 |
| L2 | DPAT on trajectory features | level z + 63 trajectory z | 79.16 | 0.70 | 0.89 | 20.84 | 90651.00 |
| L3a | Mahalanobis + MCD | 5 raw params, per lot x checkpoint | 64.64 | 0.38 | 0.86 | 35.36 | 152651.00 |
| L3b | PCA T2 + Q-residual | 83-feature design matrix | 88.99 | 0.78 | 0.94 | 11.01 | 48651.00 |
| L3b-Q | PCA Q-residual alone | 83-feature design matrix | 81.26 | 0.63 | 0.91 | 18.74 | 81651.00 |
| L3c | kNN distance | 83-feature design matrix | 86.42 | 0.82 | 0.94 | 13.58 | 59651.00 |
| L4a | Isolation Forest | 83-feature design matrix | 67.68 | 0.19 | 0.88 | 32.32 | 139651.00 |
| L4b | Autoencoder | 83-feature design matrix | 89.46 | 0.84 | 0.96 | 10.54 | 46651.00 |
| L4b' | LOF | 83-feature design matrix | 90.16 | 0.88 | 0.96 | 9.84 | 43651.00 |
| L4c | Union ensemble | 5 members, max percentile rank | 92.74 | 0.71 | 0.98 | 7.26 | 32651.00 |

`cost` = 1000 x n_FN + 1 x n_FP at the 7% operating point. L0 has no continuous score, so its threshold-free metrics are n/a rather than faked.

## Cost-optimal operating point (1000:1)

| rung | method | cost_min | cost_min_YL_% | cost_min_recall_% |
|---|---|---|---|---|
| L1a | DPAT static, MAD | 22174.00 | 94.07 | 100.00 |
| L1b | DPAT dynamic, MAD | 21185.00 | 89.87 | 100.00 |
| L1c | DPAT dynamic, all 4 estimators | 20398.00 | 86.53 | 100.00 |
| L2 | DPAT on trajectory features | 23526.00 | 99.80 | 100.00 |
| L3a | Mahalanobis + MCD | 22960.00 | 97.40 | 100.00 |
| L3b | PCA T2 + Q-residual | 23526.00 | 99.80 | 100.00 |
| L3b-Q | PCA Q-residual alone | 23429.00 | 99.39 | 100.00 |
| L3c | kNN distance | 22881.00 | 97.06 | 100.00 |
| L4a | Isolation Forest | 23229.00 | 98.54 | 100.00 |
| L4b | Autoencoder | 20421.00 | 69.66 | 99.06 |
| L4b' | LOF | 23001.00 | 67.88 | 98.36 |
| L4c | Union ensemble | 11306.00 | 35.24 | 99.30 |

## DPAT estimator comparison

| estimator | recall_fixed_k6_% | yield_loss_fixed_k6_% | recall_matched_overkill_% |
|---|---|---|---|
| MAD | 11.94 | 0.17 | 11.94 |
| IQR | 11.48 | 0.16 | 11.71 |
| p1p99 | 9.13 | 0.01 | 22.25 |
| classical | 9.37 | 0.05 | 16.39 |

## Cross-lot stability of the estimators

| parameter | estimator | sigma_CV_% | limit_CV_% | corr(contamination, sigma) |
|---|---|---|---|---|
| iddq_ua | MAD | 22.37 | 21.83 | -0.13 |
| iddq_ua | IQR | 22.44 | 21.87 | -0.13 |
| iddq_ua | p1p99 | 23.37 | 21.59 | 0.01 |
| iddq_ua | classical | 20.08 | 20.33 | -0.06 |
| leakage_na | MAD | 24.82 | 24.49 | -0.14 |
| leakage_na | IQR | 24.51 | 24.30 | -0.14 |
| leakage_na | p1p99 | 25.43 | 23.64 | 0.04 |
| leakage_na | classical | 21.41 | 21.96 | -0.01 |

## Leave-one-lot-out (10 held-out lots, seeded)

| method | recall mean | recall std | PR_AUC mean | PR_AUC std |
|---|---|---|---|---|
| L1_dynamic_MAD | 61.25 | 14.57 | 0.30 | 0.15 |
| L3a_MCD | 66.72 | 7.14 | 0.42 | 0.14 |
| L4d_AutoEnc | 92.32 | 8.60 | 0.87 | 0.11 |
