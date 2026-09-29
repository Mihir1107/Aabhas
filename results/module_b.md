# Module B: drift prediction

Dataset `dataset-v1.2`. Lot-grouped chronological split, reusing the Module A harness: train 144 lots (LOT000..LOT143), val 48 lots (LOT144..LOT191), test 48 lots (LOT192..LOT239). All figures on TEST lots.

**Early-warning mode is the headline** and uses 0 h and 24 h ONLY. `assert_no_leak` refuses to build an early feature matrix containing any 96 h or 168 h derived column, so the separation is mechanical rather than a matter of discipline.

## Rung 1 gate: does linear extrapolation over-predict?

Yes, on every parameter. Between **68.8% and 79.5%** of good parts are over-predicted (chance would be 50%), and the over-shoot is **80% to 138% of the true drift** — the strawman roughly doubles it. That is exactly what a sub-linear saturating law (beta 0.55-0.69, recovered from the training lots) predicts, and it confirms the generator and the feature computation agree before anything else was built.

## The ladder: MAE on v168, early-warning mode

| model | iddq_ua | leakage_na | prop_delay_ns | supply_current_ma | vth_shift_mv |
|---|---|---|---|---|---|
| 1_linear_slope | 2.8682 | 6.9294 | 0.3129 | 3.3285 | 5.1347 |
| 2_power_law | 1.1674 | 2.3783 | 0.1469 | 1.3587 | 1.5133 |
| 3_huber | 0.6044 | 1.3101 | 0.0684 | 0.6837 | 0.8393 |
| 4_gbm_mae | 0.6103 | 1.2975 | 0.0674 | 0.6771 | 0.8415 |
| 5a_lgbm_q95 | 0.6103 | 1.2975 | 0.0674 | 0.6771 | 0.8415 |
| 5b_quantile_forest | 0.6061 | 1.3122 | 0.0681 | 0.6848 | 0.8478 |

Recovered beta per parameter, fitted within training lots only: iddq_ua 0.609, leakage_na 0.574, prop_delay_ns 0.693, supply_current_ma 0.658, vth_shift_mv 0.557.

**LightGBM does not beat Huber.** On `iddq_ua` Huber is marginally better (0.6044 vs 0.6103); LightGBM wins by a similar hair on the other four. The two are a tie for practical purposes, and the honest reading is that once the power-law structure is in the features, the remaining signal is close to linear in them. Huber is the better default: it is interpretable, has no hyperparameters to defend, and trains in a fraction of the time.

The real jump is rung 1 to rung 2 to rung 3: linear extrapolation to power law halves the error, and power law to Huber halves it again.

## MAE, good vs defective parts

| parameter | MAE | MAE_good | MAE_defective | ratio |
|---|---|---|---|---|
| iddq_ua | 0.6103 | 0.4809 | 7.7534 | 16.1230 |
| leakage_na | 1.2975 | 1.1101 | 11.6419 | 10.4873 |
| prop_delay_ns | 0.0674 | 0.0616 | 0.3874 | 6.2857 |
| supply_current_ma | 0.6771 | 0.6208 | 3.7862 | 6.0988 |
| vth_shift_mv | 0.8415 | 0.7801 | 4.2314 | 5.4240 |

**Errors on defective parts are 5.4x to 16.1x larger than on good parts**, and the aggregate MAE hides this completely. That is not a failure of the model: defective parts are the ones whose drift departs from the population the model learned. It does mean a single headline MAE is a misleading summary for a safety application.

## Prediction intervals: coverage is BELOW nominal

| parameter | model | upper95_coverage | mean_upper_width |
|---|---|---|---|
| iddq_ua | 5a_lgbm_q95 | 0.9350 | 1.0330 |
| iddq_ua | 5b_quantile_forest | 0.9406 | 1.1387 |
| leakage_na | 5a_lgbm_q95 | 0.9264 | 2.3891 |
| leakage_na | 5b_quantile_forest | 0.9358 | 2.6226 |
| prop_delay_ns | 5a_lgbm_q95 | 0.9475 | 0.1374 |
| prop_delay_ns | 5b_quantile_forest | 0.9540 | 0.1537 |
| supply_current_ma | 5a_lgbm_q95 | 0.9411 | 1.4026 |
| supply_current_ma | 5b_quantile_forest | 0.9479 | 1.5418 |
| vth_shift_mv | 5a_lgbm_q95 | 0.9332 | 1.6627 |
| vth_shift_mv | 5b_quantile_forest | 0.9440 | 1.8208 |

Nominal is 0.95. LightGBM's quantile objective delivers **0.926 to 0.948** (mean 0.937) and the quantile forest **0.936 to 0.954** (mean 0.944). Both are miscalibrated, both in the optimistic direction — the interval is too narrow, so the true value exceeds the 'worst case' more often than advertised. This is reported rather than presented as calibrated, and it is precisely the gap the conformal layer closes: conformal calibration gives a finite-sample guarantee where the quantile objective gives only an asymptotic hope.

## Early vs mid-test mode

| parameter | early | mid | improvement_% |
|---|---|---|---|
| iddq_ua | 0.6103 | 0.4777 | 21.7290 |
| leakage_na | 1.2975 | 0.9910 | 23.6198 |
| prop_delay_ns | 0.0674 | 0.0493 | 26.8265 |
| supply_current_ma | 0.6771 | 0.4890 | 27.7848 |
| vth_shift_mv | 0.8415 | 0.6291 | 25.2472 |

Adding the 96 h checkpoint improves MAE by **22% to 28%**. The two modes are never blended.

## Survivorship bias, in absolute units

| parameter | n_censored_test | bias_on_censored_survfit | bias_%_of_limit | MAE_censored_survfit | MAE_censored_censfit |
|---|---|---|---|---|---|
| iddq_ua | 88.0000 | -30.9139 | 61.8278 | 31.0233 | 30.4059 |
| leakage_na | 88.0000 | -40.5068 | 40.5068 | 41.0015 | 38.9860 |
| prop_delay_ns | 88.0000 | -1.1427 | 11.4265 | 1.1723 | 1.1090 |
| supply_current_ma | 88.0000 | -11.0765 | 17.0408 | 11.3505 | 10.7572 |
| vth_shift_mv | 88.0000 | -13.5066 | 27.0131 | 14.0777 | 13.7980 |

A model trained only on parts that survived to 168 h **under-predicts the true 168 h value of pulled parts by 30.9 uA on Iddq — 62% of the entire datasheet limit** — and by 40.5 nA on leakage (41% of limit), 13.5 mV on vth_shift (27%), 11.1 mA on supply current (17%) and 1.14 ns on propagation delay (11%). Those are the parts whose drift matters most, and a survivors-only pipeline is blind to all of them.

Training with the datasheet limit as an observable lower-bound surrogate for pulled parts recovers only **0.06 to 2.0 units** of that gap. The honest conclusion is that the surrogate barely helps, because the latent values sit far beyond the limit; the fix is to treat these as censored observations in the loss, not to impute a point value. That is flagged as future work rather than claimed.

## Safety slopes, four definitions

| rule | recall_% | yield_loss_% | escape_rate_% | precision_% | cost |
|---|---|---|---|---|---|
| a_margin | 29.7424 | 0.5388 | 70.2576 | 50.0000 | 300127.0000 |
| b_lot_slope | 27.6347 | 0.2630 | 72.3653 | 65.5556 | 309062.0000 |
| b_lot_slope_ORACLE | 27.8689 | 0.2630 | 72.1311 | 65.7459 | 308062.0000 |
| c_mission | 62.9977 | 31.4640 | 37.0023 | 3.4999 | 165417.0000 |
| d_upper_datasheet | 0.9368 | 0.0000 | 99.0632 | 100.0000 | 423000.0000 |
| d_upper_lotsafe | 16.1593 | 0.2460 | 83.8407 | 54.3307 | 358058.0000 |

**(a) margin consumption** and **(b) lot-derived** behave sensibly: 29.7% and 27.6% recall at 0.54% and 0.26% yield loss.

**(c) mission-based** flags 63.0% of defects but at **31.5% yield loss**. This is not a broken rule, it is a different question: projecting to 15 field-years at Ea=0.7 eV asks *will this part survive the mission*, not *is this part abnormal*. It flags good parts that simply started high. It belongs in qualification, not in anomaly screening. (An earlier version compared the burn-in slope directly against a mission-average rate and flagged 46-74% of everything; that was apples to oranges, because sub-linear drift makes the early slope over-state the long-run rate. It now projects with the power law instead.)

**(d) confidence-adjusted against the DATASHEET limit is inert: 0.9% recall.** This is the dataset's central premise showing through rather than a modelling failure — every injected defect is inside spec at every checkpoint by construction, so a predicted bound essentially never crosses an engineering limit. Against a **lot-derived L_safe** (the AEC-Q001 dynamic PAT limit at 168 h, clipped to the datasheet limit) the same rule gives 16.2% recall at 0.25% yield loss with 54% precision. The safe limit is built from PRIOR lots only: a lot's own 168 h readings do not exist at the 24 h decision.

The practical consequence for the deck: **Module B's value is not in predicting limit violations, because there are none to predict. It is in predicting abnormal drift rate relative to peers.** The safe limit has to be lot-relative, which is the same argument Module A makes about static versus dynamic limits, one derivative up.

## MAE by lot

Across the 48 test lots, per-lot MAE on `iddq_ua` ranges 0.406 to 0.864 (median 0.617). No lot is a systematic outlier, so the model is not failing on a particular production window.

Full per-type and per-tier MAE in `module_b_per_type.csv`.
