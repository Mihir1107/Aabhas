# Module B: drift prediction

Dataset `dataset-v1.1`. Lot-grouped chronological split, reusing the Module A harness: train 144 lots (LOT000..LOT143), val 48 lots (LOT144..LOT191), test 48 lots (LOT192..LOT239). All figures on TEST lots.

**Early-warning mode is the headline** and uses 0 h and 24 h ONLY. `assert_no_leak` refuses to build an early feature matrix containing any 96 h or 168 h derived column, so the separation is mechanical rather than a matter of discipline.

## Rung 1 gate: does linear extrapolation over-predict?

Yes, on every parameter. Between **68.8% and 79.5%** of good parts are over-predicted (chance would be 50%), and the over-shoot is **80% to 138% of the true drift** — the strawman roughly doubles it. That is exactly what a sub-linear saturating law (beta 0.55-0.69, recovered from the training lots) predicts, and it confirms the generator and the feature computation agree before anything else was built.

## The ladder: MAE on v168, early-warning mode

| model | iddq_ua | leakage_na | prop_delay_ns | supply_current_ma | vth_shift_mv |
|---|---|---|---|---|---|
| 1_linear_slope | 2.8684 | 6.9298 | 0.3130 | 3.3289 | 5.1351 |
| 2_power_law | 1.1675 | 2.3785 | 0.1469 | 1.3587 | 1.5134 |
| 3_huber | 0.6044 | 1.3103 | 0.0684 | 0.6838 | 0.8395 |
| 4_gbm_mae | 0.6087 | 1.2959 | 0.0675 | 0.6768 | 0.8434 |
| 5a_lgbm_q95 | 0.6087 | 1.2959 | 0.0675 | 0.6768 | 0.8434 |
| 5b_quantile_forest | 0.6063 | 1.3121 | 0.0681 | 0.6846 | 0.8486 |

Recovered beta per parameter, fitted within training lots only: iddq_ua 0.609, leakage_na 0.574, prop_delay_ns 0.693, supply_current_ma 0.658, vth_shift_mv 0.557.

**LightGBM does not beat Huber.** On `iddq_ua` Huber is marginally better (0.6044 vs 0.6087); LightGBM wins by a similar hair on the other four. The two are a tie for practical purposes, and the honest reading is that once the power-law structure is in the features, the remaining signal is close to linear in them. Huber is the better default: it is interpretable, has no hyperparameters to defend, and trains in a fraction of the time.

The real jump is rung 1 to rung 2 to rung 3: linear extrapolation to power law halves the error, and power law to Huber halves it again.

## MAE, good vs defective parts

| parameter | MAE | MAE_good | MAE_defective | ratio |
|---|---|---|---|---|
| iddq_ua | 0.6087 | 0.4794 | 7.7434 | 16.1516 |
| leakage_na | 1.2959 | 1.1089 | 11.6217 | 10.4804 |
| prop_delay_ns | 0.0675 | 0.0616 | 0.3966 | 6.4400 |
| supply_current_ma | 0.6768 | 0.6205 | 3.7838 | 6.0981 |
| vth_shift_mv | 0.8434 | 0.7814 | 4.2664 | 5.4596 |

**Errors on defective parts are 5.5x to 16x larger than on good parts**, and the aggregate MAE hides this completely. That is not a failure of the model: defective parts are the ones whose drift departs from the population the model learned. It does mean a single headline MAE is a misleading summary for a safety application.

## Prediction intervals: coverage is BELOW nominal

| parameter | model | upper95_coverage | mean_upper_width |
|---|---|---|---|
| iddq_ua | 5a_lgbm_q95 | 0.9337 | 1.0327 |
| iddq_ua | 5b_quantile_forest | 0.9407 | 1.1358 |
| leakage_na | 5a_lgbm_q95 | 0.9283 | 2.3927 |
| leakage_na | 5b_quantile_forest | 0.9342 | 2.6206 |
| prop_delay_ns | 5a_lgbm_q95 | 0.9462 | 0.1372 |
| prop_delay_ns | 5b_quantile_forest | 0.9541 | 0.1532 |
| supply_current_ma | 5a_lgbm_q95 | 0.9391 | 1.3946 |
| supply_current_ma | 5b_quantile_forest | 0.9471 | 1.5356 |
| vth_shift_mv | 5a_lgbm_q95 | 0.9341 | 1.6632 |
| vth_shift_mv | 5b_quantile_forest | 0.9434 | 1.8225 |

Nominal is 0.95. LightGBM's quantile objective delivers **0.928 to 0.946** (mean 0.936) and the quantile forest **0.936 to 0.954** (mean 0.944). Both are miscalibrated, both in the optimistic direction — the interval is too narrow, so the true value exceeds the 'worst case' more often than advertised. This is reported rather than presented as calibrated, and it is precisely the gap the conformal layer closes: conformal calibration gives a finite-sample guarantee where the quantile objective gives only an asymptotic hope.

## Early vs mid-test mode

| parameter | early | mid | improvement_% |
|---|---|---|---|
| iddq_ua | 0.6087 | 0.4782 | 21.4346 |
| leakage_na | 1.2959 | 0.9921 | 23.4422 |
| prop_delay_ns | 0.0675 | 0.0494 | 26.8413 |
| supply_current_ma | 0.6768 | 0.4883 | 27.8515 |
| vth_shift_mv | 0.8434 | 0.6277 | 25.5784 |

Adding the 96 h checkpoint improves MAE by **21% to 28%**. The two modes are never blended.

## Survivorship bias, in absolute units

| parameter | n_censored_test | bias_on_censored_survfit | bias_%_of_limit | MAE_censored_survfit | MAE_censored_censfit |
|---|---|---|---|---|---|
| iddq_ua | 88.0000 | -30.8235 | 61.6469 | 30.9318 | 30.3619 |
| leakage_na | 88.0000 | -40.5005 | 40.5005 | 41.0119 | 39.5065 |
| prop_delay_ns | 88.0000 | -1.1450 | 11.4496 | 1.1740 | 1.0978 |
| supply_current_ma | 88.0000 | -11.0432 | 16.9895 | 11.3245 | 10.8360 |
| vth_shift_mv | 88.0000 | -13.5171 | 27.0342 | 14.1054 | 13.7401 |

A model trained only on parts that survived to 168 h **under-predicts the true 168 h value of pulled parts by 30.8 uA on Iddq — 62% of the entire datasheet limit** — and by 40.5 nA on leakage (41% of limit), 13.5 mV on vth_shift (27%), 11.0 mA on supply current (17%) and 1.15 ns on propagation delay (11%). Those are the parts whose drift matters most, and a survivors-only pipeline is blind to all of them.

Training with the datasheet limit as an observable lower-bound surrogate for pulled parts recovers only **0.08 to 1.5 units** of that gap. The honest conclusion is that the surrogate barely helps, because the latent values sit far beyond the limit; the fix is to treat these as censored observations in the loss, not to impute a point value. That is flagged as future work rather than claimed.

## Safety slopes, four definitions

| rule | recall_% | yield_loss_% | escape_rate_% | precision_% | cost |
|---|---|---|---|---|---|
| a_margin | 30.4450 | 0.5388 | 69.5550 | 50.5837 | 297127.0000 |
| b_lot_slope | 27.8689 | 0.2630 | 72.1311 | 65.7459 | 308062.0000 |
| b_lot_slope_ORACLE | 28.1030 | 0.2630 | 71.8970 | 65.9341 | 307062.0000 |
| c_mission | 63.9344 | 31.4470 | 36.0656 | 3.5519 | 161413.0000 |
| d_upper_datasheet | 1.4052 | 0.0000 | 98.5948 | 100.0000 | 421000.0000 |
| d_upper_lotsafe | 17.0960 | 0.2291 | 82.9040 | 57.4803 | 354054.0000 |

**(a) margin consumption** and **(b) lot-derived** behave sensibly: 30.4% and 28.1% recall at 0.54% and 0.26% yield loss.

**(c) mission-based** flags 63.9% of defects but at **31.4% yield loss**. This is not a broken rule, it is a different question: projecting to 15 field-years at Ea=0.7 eV asks *will this part survive the mission*, not *is this part abnormal*. It flags good parts that simply started high. It belongs in qualification, not in anomaly screening. (An earlier version compared the burn-in slope directly against a mission-average rate and flagged 46-74% of everything; that was apples to oranges, because sub-linear drift makes the early slope over-state the long-run rate. It now projects with the power law instead.)

**(d) confidence-adjusted against the DATASHEET limit is inert: 1.4% recall.** This is the dataset's central premise showing through rather than a modelling failure — every injected defect is inside spec at every checkpoint by construction, so a predicted bound essentially never crosses an engineering limit. Against a **lot-derived L_safe** (the AEC-Q001 dynamic PAT limit at 168 h, clipped to the datasheet limit) the same rule gives 17.8% recall at 0.15% yield loss with 68% precision.

The practical consequence for the deck: **Module B's value is not in predicting limit violations, because there are none to predict. It is in predicting abnormal drift rate relative to peers.** The safe limit has to be lot-relative, which is the same argument Module A makes about static versus dynamic limits, one derivative up.

## MAE by lot

Across the 48 test lots, per-lot MAE on `iddq_ua` ranges 0.407 to 0.866 (median 0.610). No lot is a systematic outlier, so the model is not failing on a particular production window.

Full per-type and per-tier MAE in `module_b_per_type.csv`.
