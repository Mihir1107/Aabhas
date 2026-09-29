# Cumulative ablation, C0 to C4

Dataset `dataset-v1.2`. Each rung contains everything below it, fused at score level by max robust-z against training-lot parts (no labels). Test lots only (n=24000, 427 defective). Operating point 7% yield loss.

| rung | contents | recall@93%yield | PR_AUC | AUROC | escape_rate_% | cost |
|---|---|---|---|---|---|---|
| C0 | L0 alone (static datasheet limits) | n/a | n/a | n/a | 100.00 | 427000.00 |
| C1 | L0 + L1 (DPAT) | 54.33 | 0.26 | 0.83 | 45.67 | 196651.00 |
| C2 | L0 + L1 + L2 (trajectory) | 79.86 | 0.71 | 0.90 | 20.14 | 87651.00 |
| C3 | L0 + L1 + L2 + L3 (robust multivariate) | 92.74 | 0.80 | 0.98 | 7.26 | 32651.00 |
| C4 | L0 + L1 + L2 + L3 + L4 (unsupervised ML) | 92.27 | 0.80 | 0.98 | 7.73 | 34651.00 |

**The cumulative table is NOT monotone.** See the diagnosis below.

C0 is a binary rule with no ranking, so its threshold-free metrics are n/a rather than faked.

## Ensemble fusion: decision level vs score level

| fusion | recall@93%yield | PR_AUC | AUROC | yield_loss_% | escape_rate_% | cost |
|---|---|---|---|---|---|---|
| binary OR (decision level, own yield loss) | 88.76 | n/a | n/a | 2.44 | 11.24 | 48575.00 |
| max_rank (score level) | 92.74 | 0.71 | 0.98 | 7.00 | 7.26 | 32651.00 |
| mean_rank (score level) | 82.67 | 0.57 | 0.94 | 7.00 | 17.33 | 75651.00 |
| max_z (score level) | 92.27 | 0.79 | 0.98 | 7.00 | 7.73 | 34651.00 |
| weighted_mean_z (score level) | 90.16 | 0.79 | 0.97 | 7.00 | 9.84 | 43651.00 |
| (best single member: L4d_AutoEnc) | 89.46 | 0.84 | 0.96 | 7.00 | 10.54 | 46651.00 |
| (best single member: L4b_LOF) | 90.16 | 0.88 | 0.96 | 7.00 | 9.84 | 43651.00 |

## Per-type and per-tier, cumulative rungs (all lots)

| defect_type | severity | n | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|
| GOOD | (all) | 112460.00 | 0.00 | 5.80 | 5.80 | 6.00 | 5.80 |
| III_CENTRE_HIDER | (all) | 60.00 | 0.00 | 0.00 | 0.00 | 60.00 | 53.30 |
| III_CENTRE_HIDER | mild | 21.00 | 0.00 | 0.00 | 0.00 | 52.40 | 42.90 |
| III_CENTRE_HIDER | moderate | 22.00 | 0.00 | 0.00 | 0.00 | 50.00 | 45.50 |
| III_CENTRE_HIDER | severe | 17.00 | 0.00 | 0.00 | 0.00 | 82.40 | 76.50 |
| II_STEP_DEFECT | (all) | 600.00 | 0.00 | 46.70 | 91.30 | 95.00 | 94.80 |
| IV_CORRELATION_BREAK | (all) | 420.00 | 0.00 | 21.00 | 14.30 | 82.90 | 81.70 |
| IV_CORRELATION_BREAK | mild | 141.00 | 0.00 | 0.00 | 2.80 | 48.90 | 45.40 |
| IV_CORRELATION_BREAK | moderate | 130.00 | 0.00 | 3.10 | 5.40 | 100.00 | 100.00 |
| IV_CORRELATION_BREAK | severe | 149.00 | 0.00 | 56.40 | 32.90 | 100.00 | 100.00 |
| I_STEEP_DRIFTER | (all) | 720.00 | 0.00 | 56.70 | 99.00 | 99.90 | 100.00 |
| VII_FIXTURE_ARTIFACT | (all) | 640.00 | 0.00 | 21.20 | 84.20 | 84.70 | 85.80 |
| VI_LOT_SHIFT | (all) | 3000.00 | 0.00 | 5.60 | 5.70 | 5.10 | 5.10 |
| Va_MILDLY_HIGH_STABLE | (all) | 1800.00 | 0.00 | 71.40 | 54.60 | 51.50 | 52.10 |
| Vb_EXTREME_LEVEL | (all) | 300.00 | 0.00 | 100.00 | 100.00 | 100.00 | 100.00 |

Type VI has **zero parts in the test set**; every Type VI figure in this repository is an ALL-LOTS number and is labelled as such. See `split_counts.csv`.