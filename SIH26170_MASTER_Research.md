# SIH26170 MASTER RESEARCH DOCUMENT
## AI-Driven Anomaly Detection in Component Burn-In and Screening
**ISRO | Software | Smart Automation | Idea submission deadline: 20 Sep 2026**

Synthesis of four independent research passes, with conflicts resolved and claims verified.

---

# PART 0: RECONCILIATION AND VERDICT

Read this section first. It resolves the contradictions between the four research sources and flags one error that would have cost us the round.

## 0.1 CRITICAL ERROR IN THREE OF THE FOUR SOURCES

**Every other research pass proposed a deck of 10, 10, and 17 slides. The official SIH template caps the submission at SIX slides including the title slide.**

The template's instruction page states it explicitly: maximum six slides, avoid paragraphs, use points and diagrams and infographics, use only the provided template without changing the idea-detail pointers, and submit as PDF only.

That means five content slides, and their titles are fixed by the template:

1. Title Page
2. Proposed Solution
3. Technical Approach
4. Feasibility and Viability
5. Impact and Benefits
6. Research and References

Any slide plan that does not map onto exactly these six is unusable. Part 11 of this document contains the corrected plan. Do not use the slide outlines from the other research passes.

## 0.2 THE FORMULA CONFLICT, RESOLVED

Two sources gave different formulas for the dynamic limit. Both are mathematically valid. They are different members of the same estimator family, and understanding the difference is itself a differentiator.

The goal in every case is to estimate the spread of the *good* population without letting the outliers we are hunting corrupt that estimate. Three standard estimators exist:

| Method | Formula for robust sigma | Gaussian constant (verified) | Breakdown point |
|---|---|---|---|
| **MAD** | `1.4826 x median(abs(x - median(x)))` | 1/0.674490 = **1.482602** | **50%** |
| **IQR (AEC-Q001 Rev-D)** | `(Q3 - Q1) / 1.35` | IQR of N(0,1) = **1.348980** | **25%** |
| **Percentile (p1/p99)** | `(p99 - median) x 0.43` | 1/2.326348 = **0.429858** | **~1%** |

All three constants are confirmed correct. The percentile variant with the 0.43 factor that appeared in one research pass is legitimate arithmetic.

**But the breakdown point is the deciding argument.** The breakdown point is the fraction of contaminated data an estimator tolerates before it becomes meaningless. The p99 is *itself computed from* the top 1% of the data, which is precisely where our latent defects live. Using p99 to set the limit that catches outliers lets the outliers widen their own limit.

**Our position, which is stronger than any single source:**

- **Primary estimator: MAD-based** (50% breakdown, maximum robustness). This also matches the fourth research pass's recommendation.
- **Report AEC-Q001's IQR/1.35 alongside it** because it is what the published standard specifies and naming the standard is our credibility anchor.
- **Use the p1/p99 variant only for the asymmetric case**, where the parameter distribution is skewed and a symmetric limit would over-reject on one side. AEC-Q001 itself permits derived methods for non-normal distributions when statistically justified.

Putting all three on one slide with the breakdown-point column is a genuinely sophisticated technical argument that costs three lines.

**Sigma multiplier note:** AEC-Q001 conventionally uses 6. Industry practice varies, and proteanTecs describes PAT deployments at plus or minus 4 sigma. Treat the multiplier as a tunable that trades escape rate against yield loss, and show the tradeoff curve rather than asserting a number.

## 0.3 WHAT TO KEEP FROM EACH SOURCE

**Keep from Source A (the Arrhenius/AEC/spatial pass):**
- Arrhenius physics-of-failure framing (strong, and it grounds the synthetic generator)
- Asymmetric loss function with the tau hyperparameter (excellent, see 0.4)
- Quantile Random Forest (Meinshausen) as a named method
- NNR and GDBN spatial concepts, **but repurposed**, see 0.5

**Keep from Source B (the concise landscape pass):**
- F-beta with beta greater than 1 as an explicit metric
- The clean multi-stage pipeline framing
- Counterfactual explanations ("this part would have passed if 24h leakage had been below X")

**Keep from Source C (the ESS/dashboard pass):**
- The proteanTecs industrial validation, **corrected**, see 0.6
- SDCL for label bootstrapping, **with a caveat**, see 0.7

**Keep from Source D (the reliability-assurance pass), which was the strongest of the three:**
- **Censoring and test-stop bias.** If a part fails at 96h it is pulled, so it has no 168h value. Training only on parts that survived to 168h is survivorship bias. None of the other passes caught this. It is a real methodological trap.
- **Grouped-by-lot validation.** Random row splits leak lot-level information and inflate scores. Split by lot, or leave-one-lot-out.
- **Data-quality gate before the AI gate.** An anomaly may be the tester, not the part.
- **PASS / WATCH / REVIEW / REJECT** multi-tier output instead of binary.
- **Never let a model override a hard engineering limit.** This also matches AEC-Q001's own rule that PAT limits shall never exceed device specification limits.
- The honesty rule about not presenting synthetic results as industrial validation.
- The judge-question preparation list.

## 0.4 UNIFYING THE THREE RISK-CALIBRATION IDEAS

The four passes independently proposed asymmetric loss, quantile regression with prediction intervals, and conformal risk control. These are not competing. They are three rungs of increasing rigor on the same ladder, and presenting them as a ladder is more impressive than presenting any one.

| Rung | Mechanism | What it gives you | Effort |
|---|---|---|---|
| 1 | **Asymmetric loss** (quantile loss with tau = 0.9) | The model *tends* to over-predict drift | Low |
| 2 | **Quantile regression / QRF** | A calibrated prediction *interval*, decide on the upper bound | Medium |
| 3 | **Conformal risk control** | A distribution-free, finite-sample *guarantee* that expected FNR stays below alpha | Medium |

Note that rungs 1 and 2 are the same mathematical object: quantile loss with a skewed tau *is* asymmetric loss. Saying this explicitly shows we understand the mechanism rather than listing buzzwords.

Rung 3 is the differentiator. Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024) extends conformal prediction to bound the expected value of any monotone loss, and the paper's own worked examples explicitly include bounding the false negative rate. It requires only exchangeability, no distributional assumption.

**The framing that wins:** MIL-STD-883 already uses PDA (Percent Defective Allowable) as a statistical bound on lot quality that ISRO accepts as a screening criterion. Conformal risk control provides the same class of statistically defensible bound, but on the escape rate of the AI system itself. It makes the model auditable inside a product-assurance framework that already exists.

## 0.5 FIXING THE SPATIAL ANALYSIS IDEA

Source A proposed Nearest Neighbour Residual (NNR) and Good Die in a Bad Neighbourhood (GDBN). Both are real, established industry techniques, and NNR appears in the ITC literature alongside DPAT.

**But as written they do not apply to this problem statement.** NNR and GDBN operate on **wafer map XY coordinates**. Our PS describes packaged components measured at 0h, 24h, 96h and 168h during burn-in. There is no wafer map. Presenting wafer-level spatial analysis for a burn-in dataset would be a visible domain error.

**The fix, which turns the weakness into one of our best ideas:** the spatial concept transfers cleanly to a different coordinate system that burn-in *does* have. Components sit in numbered sockets on burn-in boards inside an oven. That gives every part a physical position.

Applying neighbour-residual logic over **socket and board position** lets the system answer a question no other team will even ask:

> Is this anomaly a property of the *component*, or of the *chamber*?

If flagged parts cluster on one board edge or one oven shelf, the likely cause is a thermal gradient or a socket contact problem, not a latent defect. Rejecting those parts is pure yield loss. Detecting the pattern and telling the inspector "this is a fixture issue, not a part issue" is exactly the kind of thing a real reliability engineer worries about, and it directly serves Source D's fault-versus-sensor discrimination requirement.

This is our most defensible original contribution. Put it on the Technical Approach slide.

## 0.6 THE PROTEANTECS EVIDENCE, CORRECTED

Source C stated "10 parts flagged, 7 failed." That specific claim could not be verified and appears garbled. **The real published numbers are stronger and are verifiable:**

- A manufacturer reduced DPPM by **396** using ML-driven IDDQ-based outlier detection, saving over **$250,000** in test and packaging costs.
- **70% of flagged outliers subsequently failed HTOL stress testing**, confirming they were genuine latent defects rather than false alarms.
- Reported up to **10x DPPM reduction without compromising yield**.

Use the corrected numbers. The 70% HTOL confirmation figure is the single most useful external datapoint we have, because it is direct industrial evidence that statistically-flagged in-spec parts really are defective. It answers the obvious judge question "how do you know your flagged parts are actually bad?" with someone else's validated data.

**Bonus finding from the same source, and it is important:** proteanTecs documents that PAT's effectiveness *degrades at advanced process nodes* because the within-wafer distribution widens until it approaches the full process distribution, at which point lot-relative limits stop separating anything. Their fix is family-based grouping, meaning comparing each part against a more tightly matched peer group.

This is independent industrial confirmation of Source D's hierarchical-baseline argument. Both say the same thing: **choosing the right comparison population matters more than choosing the algorithm.** That belongs in our design.

## 0.7 SDCL, USEFUL BUT CIRCULAR

Sigma Deviation Count Labeling is a real published method. It labels initially-passed samples by their sigma deviation, then trains supervised models on those pseudo-labels. It is a legitimate answer to our label-scarcity problem and worth citing.

**The caveat we must state if we use it:** a supervised model trained on labels generated by a statistical rule can, at best, learn to reproduce that statistical rule. It cannot discover defects the rule missed. So SDCL is useful for **bootstrapping** a supervised model, and useless for **validating** that we beat the statistical baseline.

Stating this limitation out loud is worth more than the technique itself. It demonstrates we can spot circular evaluation, which is a genuinely rare skill in a hackathon deck.

## 0.8 CLAIMS TO DELETE

**"85% reduction in unnecessary testing hours."** Fabricated. It appeared in one research pass with no source. Do not put an unsourced quantitative claim in front of ISRO judges. If we want an efficiency claim, derive it from our own experiment (for example, "in our simulation, N% of parts flagged at 24h were confirmed defective at 168h") and label it as simulated.

**"125 degrees C is the universal burn-in condition."** Overstated. Source D is right: 125 C is a common condition and it is what our PS specifies, but the correct condition depends on device technology, package, maximum junction temperature and the applicable qualification standard. Phrase it as "the condition specified in this problem statement" rather than a universal law.

**Any deep sequence model as the headline.** With four timepoints and two of them as input, LSTMs, Transformers, TranAD and Anomaly Transformer are the wrong tools. All four research passes independently reached this conclusion. State the choice deliberately in the deck: it reads as engineering maturity.

---

# PART 1: DOMAIN FOUNDATION

## 1.1 What burn-in is

Environmental Stress Screening deliberately applies stress to precipitate latent manufacturing, material and workmanship defects into observable failures before deployment. Applied at component level it is called burn-in: parts are operated under electrical bias at elevated temperature for an extended period, with parameters measured at interval checkpoints.

This targets the **infant mortality** region of the bathtub curve: a high but rapidly falling early-life failure rate driven by inherent manufacturing defects, followed by a long flat useful-life region and finally wear-out. Burn-in exists to consume that first region on the ground rather than in orbit.

Burn-in is a **screening** process, not a lifetime guarantee. Distinguish it clearly from:
- **Qualification testing:** demonstrates design and technology capability
- **Reliability testing:** estimates failure behaviour statistically
- **ESS:** removes defective hardware from a production population
- **Final electrical test:** confirms compliance after stress

## 1.2 The physics: Arrhenius and acceleration

Thermal stress accelerates failure mechanisms including oxide breakdown, electromigration, corrosion and ionic diffusion. The Arrhenius model expresses failure rate as a function of absolute temperature:

```
lambda = A * exp(-Ea / (k*T))
```

where `Ea` is activation energy in eV, `k` is Boltzmann's constant (8.617e-5 eV/K), and `T` is absolute temperature in Kelvin.

The acceleration factor between stress and use temperature:

```
AF = exp[ (Ea/k) * (1/T_use - 1/T_stress) ]
```

**Computed values (verified):**

| Ea (eV) | AF at 55 C use | AF at 40 C use | AF at 25 C use |
|---|---|---|---|
| 0.3 | 6.5 | 10.7 | 18.8 |
| 0.5 | 22.4 | 52.2 | 132.7 |
| **0.7** | **77.7** | 254.2 | 937.5 |
| 1.0 | 501.5 | 2,728.7 | 17,606.6 |

At the standard **Ea = 0.7 eV** convention, 125 C stress against 55 C field use gives **AF = 77.7**. Our checkpoints therefore map to:

| Checkpoint | Equivalent field exposure |
|---|---|
| 24 h | ~0.21 years |
| 96 h | ~0.85 years |
| **168 h** | **~1.49 years** |

**Why this table earns a slide:** it makes the 0/24/96/168 h schedule mean something physical, it justifies the sub-linear saturating shape of good-part degradation in our synthetic generator, and it lets us state the mission relevance of the 168 h endpoint quantitatively. It also connects Module B directly to physics: forecasting the 168 h value is forecasting the trajectory of thermally-accelerated physical breakdown, not curve-fitting.

## 1.3 The governing standards

| Standard | Scope | Relevance |
|---|---|---|
| **MIL-STD-883 Method 1015** | Burn-In Test. Screens marginal devices with inherent or manufacturing-induced defects causing time- and stress-dependent failures. Minimum 125 C under bias, commonly 160 to 168 h. Interim electrical measurements permitted. | This is the PS's test setup. The 168 h endpoint originates here. |
| **PDA (Percent Defective Allowable)** | Lot accept/reject criterion, commonly 5% | Direct precedent for a statistical bound on escape rate |
| **MIL-STD-883 TM 1005 / 1010 / 1011** | Steady-state life, temperature cycling, thermal shock | Adjacent screens in the same flow |
| **AEC-Q001** | Part Average Testing. Statistically removes parts with abnormal characteristics that pass absolute limits. | The direct answer to Module A |
| **AEC-Q100 / Q101** | Stress-test qualification for ICs and discretes | Referenced by Q001 |
| **NASA EEE-INST-002** | EEE parts selection, screening, qualification, derating for space | The space-grade framing. Also notes burn-in must not exceed specified maximum junction temperature. |
| **NASA EEE Parts Screening (Preferred Practice 1401)** | Screening guidance | Emphasises specifying both permitted defective fraction **and allowable parameter drift** |
| **JAXA JERG-2-027** | Commercial EEE parts in space | Describes dynamic burn-in with parameter recording for drift calculation |
| **ISRO-PAS / ISRO-PAX series** | ISRO product assurance specifications | Shows we know the customer's own framework |

**Two points worth putting in the deck:**

1. NASA screening guidance already calls for specifying **allowable parameter drift**, not just final absolute limits. Our system is not inventing a new requirement, it is automating one the standards already ask for. That reframes the whole proposal from "novel AI idea" to "closing a known gap in an existing process," which is a much easier sell to a conservative reviewer.

2. NASA's own parts-assurance literature states plainly that system-level "test as you fly" programmes provide **no guarantee** against a latent part-level defect escaping. That is the gap this PS exists to close.

**Local relevance:** ISRO's Space Applications Centre in Ahmedabad explicitly performs qualification, screening and characterisation of all electronic components, plus construction and failure analysis of electronic components. This PS is not hypothetical. It is SAC's daily workflow, in our own city.

## 1.4 Vocabulary

Using this language correctly signals domain competence faster than any architecture diagram.

- **DPPM:** Defective Parts Per Million shipped. The industry quality unit.
- **Test escape:** a defective part that passes screening. Our false negative.
- **Yield loss / overkill:** good parts wrongly rejected. Our false positive.
- **ELFR:** Early Life Failure Rate.
- **Iddq:** quiescent supply current. **Delta-Iddq:** differences across conditions or time, used specifically because absolute thresholds fail when background leakage is high.
- **Test insert:** one measurement stage in the flow. Our checkpoints.
- **PAT limits:** statistically derived limits, always tighter than datasheet limits.
- **RMA:** Returned Material Authorisation, a customer return.
- **HTOL:** High Temperature Operating Life, the longer-duration stress test used to confirm suspected defects.

## 1.5 Failure mechanisms behind abnormal drift

Useful for the physical-explanation layer (Part 6). Abnormal electrical drift may be associated with oxide defects, metallisation degradation, contamination or ionic migration, defective wire bonds, solder or interconnect weakness, junction leakage, package-induced stress, thermal expansion mismatch, die damage, contact resistance change, or moisture and hermeticity problems.

---

# PART 2: DECODING THE PROBLEM STATEMENT

## 2.1 Module A is Dynamic Part Average Testing

The PS's worked example (lot average 10 uA, part at 45 uA, datasheet max 50 uA) is a textbook DPAT illustration. Naming it is our single highest-leverage move.

**Static PAT** derives limits from historical lots and holds them fixed. **Dynamic PAT** recomputes limits from the current lot. Dynamic is preferred because the reference population is the parts actually being tested, which avoids lot-to-lot variation and yields tighter limits without rejecting good parts.

**The AEC-Q001 method:**

```
Robust Mean  = Q2 (median)
Robust Sigma = (Q3 - Q1) / 1.35
Upper Limit  = Robust Mean + 6 * Robust Sigma
Lower Limit  = Robust Mean - 6 * Robust Sigma
```

Generalised to sigma multiplier N:
```
Lower = Q1 - ((N - 0.675) / 1.35) * IQR
Upper = Q3 + ((N - 0.675) / 1.35) * IQR
```

The standard's own reasoning: ordinary mean and standard deviation are poor statistics here precisely because they are sensitive to the outliers being sought.

**Rules the implementation must respect:**
- PAT limits **shall not exceed** device specification limits. The AI never loosens an engineering limit.
- Minimum **30 parts per lot** for limit estimation.
- The 1.35 divisor is unreliable below n = 20. Fall back to a broader peer group when a lot is small.
- New limits must be re-established after design changes, die shrinks or process changes.

**The subtle point worth one line:** under a perfect Gaussian, plus or minus 6 sigma is a two-sided tail of about 2e-9, which is **0.002 DPPM**. Real PAT rejection rates are orders of magnitude higher. So PAT limits are not a Gaussian tail argument, they are a **mixture-model** argument: parts beyond the limit are presumed to belong to a different, defective sub-population. Framing it this way is correct and almost never stated.

## 2.2 The insight that justifies going beyond DPAT

This is the most valuable single finding across all four research passes.

Hu, Nguyen, He and Li (**ITC 2020**, UCSB with NXP Semiconductors) analysed real automotive microcontroller test data containing genuine customer returns, parts that escaped a full production screen out of millions shipped. Their finding:

> The escaped latent-defect part sits **almost at the centre of the data distribution in every single dimension**. It cannot be separated from good parts by examining any individual parameter.

They classify three defect types: benign defects (centre of distribution, no impact), catastrophic defects (far from centre, trivially screened), and **latent reliability defects** which look normal at test and only evolve into failures later under field ageing.

proteanTecs report the same phenomenon from a different angle: an outlier chip whose leakage was not only inside PAT limits but *lower* than the highest-leakage good chip on the wafer. Tightening the univariate limit to catch it would have rejected functional chips.

**Three things this buys us:**

1. A rigorous, citable reason why univariate DPAT is structurally insufficient, which is exactly the failure mode the PS complains about.
2. An evidence-based justification for the multivariate and temporal layers, rather than an aesthetic one.
3. **A hard design constraint on our synthetic data.** Our hardest injected defects must sit at the univariate median and be visible only jointly. If our synthetic bad parts are univariate outliers, we have built a trivial problem, our metrics are meaningless, and any judge with domain knowledge will see it immediately.

**Their benchmark results on real industrial data (AUROC):** Gaussian model worst at ~0.837 average, Isolation Forest and Autoencoder strongest among conventional methods at ~0.88, their self-labelling approach ~0.90. **No single method dominated across chips**, which directly supports an ensemble design.

They also use **estimated yield**, defined as one minus the fraction of good parts rejected alongside the caught failure, with a stated **93% yield goal**. That is the industry's real operating metric and we should adopt it.

## 2.3 Module B is degradation extrapolation

Predict `Value_168h` from `Value_0h` and `Value_24h`, then flag if the implied drift rate exceeds a safety slope.

The relevant literature is degradation path modelling: Wiener process models with drift coefficient lambda and diffusion sigma, often with **random effects** to capture unit-to-unit heterogeneity, and first-hitting-time formulations for life prediction. Accelerated variants couple the drift coefficient to temperature via Arrhenius.

**Scoping decision, agreed by all four research passes:** four timepoints with two as input is a small longitudinal panel regression, not sequence modelling. Robust statistics and gradient-boosted trees are correct here. Deep sequence models are not.

**Leakage warning (from Source D, important):** if we use `Value_96h` to predict `Value_168h`, that is a different problem from the one specified. Define two explicit modes:
- **Early-warning mode:** uses 0h and 24h only. This is the headline model, matching the PS.
- **Mid-test update mode:** uses 0h, 24h and 96h. A secondary capability.

Never blur them.

---

# PART 3: MODULE A, TECHNICAL LADDER

Build and evaluate every rung. The ablation table across these rungs is our strongest single slide.

| Rung | Method | Role |
|---|---|---|
| **L0** | Static datasheet limits | The strawman the PS asks us to beat |
| **L1** | **DPAT per AEC-Q001**, robust median plus k times robust sigma, per lot, per parameter | Our reproduced *industry* baseline. Beating this is the actual claim. |
| **L2** | **Trajectory features** | Turns 4 raw points into drift descriptors |
| **L3** | **Robust multivariate** | Catches centre-of-distribution defects |
| **L4** | **Unsupervised ML** | Non-linear structure |
| **L5** | **Ensemble plus conformal risk control** | Guaranteed escape rate |
| **L6** | **Spatial socket/board residual** | Separates component faults from chamber faults |

## L1 detail: the peer group matters more than the algorithm

Both the ITC and proteanTecs evidence converge on this. Define a hierarchical baseline:

```
Baseline = f(part family, lot, test condition, socket/board, date code)
```

Fall back up the hierarchy when a level has too few samples (remember the n < 20 and n < 30 rules). proteanTecs call the tighter grouping "family based" and report it as the source of their 10x DPPM improvement.

## L2 detail: trajectory features

Per part, per parameter:

**Absolute deltas**
```
d1 = v24 - v0
d2 = v96 - v24
d3 = v168 - v96
```

**Slopes**
```
s1 = (v24 - v0) / 24
s2 = (v96 - v24) / 72
s3 = (v168 - v96) / 72
```

**Relative drift** (epsilon prevents blow-up when v0 is near zero)
```
r1 = (v24 - v0) / (abs(v0) + eps)
```

**Curvature / acceleration**
```
a = s2 - s1
```
A good part decelerates as burn-in saturates. Many defects accelerate. This is one of the most diagnostic single features.

**Residual from lot trend**
```
e_it = x_it - x_lot_hat(t)
```
Captures parts that drift differently from their peers, independent of their absolute level.

**Lot-relative z-scores of every feature above**, using median and MAD. This is DPAT applied to *drift* rather than *level*, and is the cleanest novel idea we can claim at the feature level.

Also include: **max single-interval jump** (catches step defects), **monotonicity flag**, and **sign-change count**.

## L3 detail: robust multivariate

- **Mahalanobis distance** with **Minimum Covariance Determinant** covariance estimation (Rousseeuw and Van Driessen fast-MCD). Plain sample covariance is outlier-corrupted, and MCD is the robust analogue of the same argument AEC-Q001 makes for median over mean. Use `sklearn.covariance.MinCovDet` or `EllipticEnvelope`.
- **PCA-based**: Hotelling's T-squared for in-model variation plus **Q-residual / SPE** for out-of-model variation. Multivariate PCA outlier detection for production test has published precedent (O'Neill, ITC 2008).
- **k-NN distance / Nearest Neighbour Residual**, an established industry technique alongside DPAT.

Honest caveat to state: Hotelling's T-squared assumes Gaussianity, which parametric test data often violates. Non-parametric k-NN rules avoid the assumption. Naming this tradeoff is a credibility win.

**Published validation for this exact combination:** an IEEE paper reports using **XGBoost plus Mahalanobis distance** at final test to capture weak ICs that passed final test but failed system-level test or quality-engineering ageing, recognising 2x to 3x the weak-IC ratio in SLT and more than 10x in QEA. That is direct precedent for L3 plus L4.

## L4 detail: unsupervised ML

`IsolationForest`, `LocalOutlierFactor`, `OneClassSVM` with RBF kernel, and a small dense autoencoder on the feature vector. Per ITC 2020, expect Isolation Forest and the autoencoder to lead, and the plain Gaussian model to trail.

Isolation Forest caveat: its score is less intuitive to inspectors, small datasets produce unstable boundaries, and a genuinely new but valid lot can look anomalous. Use it as a challenger, never as the sole decision-maker.

## L5 detail: ensemble with a union rule

Because false negatives are catastrophic and false positives merely cost money, use a **union rule**: flag if *any* detector flags. This maximises recall by construction and is the correct asymmetric-cost design. Then use conformal calibration to set thresholds so the *combined* FNR is bounded.

## L6 detail: the spatial layer (our original contribution)

Compute, for each part, the residual against its physical neighbours on the burn-in board:

```
spatial_residual_i = x_i - median(x_j for j in neighbours(i))
```

Then test whether flagged parts cluster spatially. If they do, the likely cause is a chamber thermal gradient, a socket contact issue or a channel offset, not a latent defect. Output a **fixture-fault warning** instead of rejecting the parts.

This is what separates a screening tool from a rejection tool, and it directly addresses a failure mode that would cost real yield.

---

# PART 4: MODULE B, TECHNICAL APPROACH

## 4.1 Inputs

`v0`, `v24`, `d1`, `s1`, `r1`, lot-relative z-score of `v0`, lot-relative z-score of `d1`, plus contextual lot statistics (rolling lot median at 0h and 24h) so the model can see whether the part is already diverging from its peers.

Include temperature and voltage as features if the dataset has stress-condition variation, with Arrhenius-transformed terms where appropriate.

## 4.2 Model ladder

1. **Linear slope extrapolation:** `v168_hat = v0 + s1 * 168`. The physical strawman. It will over-predict because real degradation saturates.
2. **Power-law fit:** `v(t) = v0 + lambda * t^beta` with beta between 0.5 and 1. Physically motivated, two free parameters.
3. **Huber regression.** Source D's recommended first model. Resistant to outliers, still interpretable. Good default.
4. **LightGBM / XGBoost.** Expected best on MAE.
5. **Quantile Random Forest** (Meinshausen). Retains the full target distribution in the leaves rather than averaging, so it emits conditional quantiles natively. Strong choice when we want intervals without a separate model.
6. **Conformalised quantile regression.** Calibrated intervals with coverage guarantees.

## 4.3 Predict a bound, not a point

For a safety decision the mean prediction is the wrong quantity. What matters is the plausible worst case.

- Optimise a point model for **MAE**, because MAE is the stated scoring metric and we must report it.
- Separately emit a **95% upper bound** on `v168`.
- Flag if the **upper bound** crosses the safety slope, not if the point estimate does.

Then report **interval coverage**: does the nominal 95% interval actually contain the truth 95% of the time? A calibration plot here is a genuinely distinctive slide element.

**The tuning tension to acknowledge (Source A caught this well):** if the asymmetry parameter tau is pushed too far, MAE inflates from systematic over-prediction. Since MAE is explicitly scored, we cannot maximise conservatism blindly. Show the tau sweep with FNR on one axis and MAE on the other, and justify the operating point. Presenting that tradeoff honestly is stronger than claiming to have optimised both.

## 4.4 Defining the safety slope

The PS says "a calculated safety slope" without defining it. That is an opportunity. Offer four definitions, use at least two, and show where they disagree.

**(a) Margin-consumption slope.** Given limit `L`:
```
s_safe = (L - v24) / (168 - 24)
```
Simple and explainable. Possibly too naive alone.

**(b) Lot-derived slope.** From healthy parts in the same peer group:
```
s_safe = Q_0.99(healthy slopes)
```
or, consistent with our robust framing:
```
s_safe = median(slopes) + 6 * robust_sigma(slopes)
```
This is DPAT applied to the drift-rate distribution. Needs no external assumptions.

**(c) Mission-based slope.** Using the Arrhenius AF, back-calculate the maximum drift rate that still leaves margin at end-of-mission. The physically meaningful one, and the one that connects to Part 1.2.

**(d) Confidence-adjusted rule.** Flag if:
```
UpperBound_168 > L_safe
```
This is the one that actually drives the decision.

---

# PART 5: DECISION AND RISK LAYER

## 5.1 Data-quality gate comes first

Before any model runs, validate: range and unit checks, timestamp ordering, duplicate detection, missing-value classification, calibration status, chamber temperature and voltage validity, socket and channel health.

**Do not silently impute a missing measurement that should be treated as a failed test record.** A missing 96h value because the chamber tripped is a different object from a missing 96h value because the part was pulled after failing.

## 5.2 Multi-tier output, not binary

```
PASS
WATCH               (weak indicators, continue and monitor)
REVIEW              (engineering review required)
REJECT              (hard limit violated, or high-confidence forecast breach)
MEASUREMENT_INVALID (data quality gate failed)
FIXTURE_SUSPECT     (spatial clustering detected)
```

The last two are ours and neither appeared in any single research pass. They matter because a system that only says PASS or FAIL forces every ambiguous part into a wrong bucket.

## 5.3 Decision logic, explicitly rule-based

```
IF absolute datasheet limit violated        -> REJECT
IF data quality gate failed                 -> MEASUREMENT_INVALID
IF spatial cluster detected among flags     -> FIXTURE_SUSPECT
IF upper bound of predicted v168 > L_safe   -> REJECT
IF lot-relative anomaly severe              -> REVIEW
IF drift slope > safety slope               -> REVIEW
IF multiple weak indicators present         -> WATCH
ELSE                                        -> PASS
```

A rule-based fusion layer is more explainable than a learned weighting, and explainability is a scored criterion. **The model never overrides a hard engineering limit in either direction.**

## 5.4 Positioning: decision support, not autonomous rejection

Final rejection stays under approved engineering rules and QA control. The system detects, ranks, explains, forecasts and recommends. This framing pre-empts the "will AI replace QA" objection and matches how safety-critical processes actually get certified.

---

# PART 6: EXPLAINABILITY

Three layers. The PS scores this explicitly.

## Layer 1: rule-level, deterministic, always available

```
Component: C-1048          Decision: REVIEW

Absolute limit:      PASS (45.2 uA vs 50.0 uA max)
Lot-relative:        19.5 robust sigma above lot median
                     (lot median 10.1 uA, robust sigma 1.8 uA, n=142)
Drift 0h to 24h:     +28%  (healthy-lot median +4%)
Drift rate:          0.21 uA/h  (lot median 0.03 uA/h, 7.0x)
Predicted 168h:      41.8 uA
95% upper bound:     47.9 uA   -> exceeds safety slope threshold of 28.0 uA
Spatial check:       no neighbour clustering, component-specific
Data quality:        all checkpoints valid, calibration current

Primary reason:      accelerating leakage trend, lot-relative magnitude
Recommended action:  repeat measurement, then engineering review
```

This costs nothing and is more convincing to an inspector than any saliency map.

## Layer 2: model-level

- **SHAP** on the Module B gradient-boosted model, per-part force or waterfall plot.
- **Per-channel reconstruction error** for the autoencoder: which parameter could the model not explain.
- **Per-detector attribution** in the ensemble: which detectors fired and how strongly.
- **Counterfactual:** "this part would have moved from REVIEW to PASS if the 24h value had been below 18 uA." Directly actionable, and a strong slide element.

Precedent that this is a live research direction: Ni et al., ACM TODAES 2025, on Shapley values for reducing defect escapes; and Senoner, Netland and Feuerriegel, Management Science 68(8) 2022, on explainable AI improving process quality in semiconductor manufacturing.

## Layer 3: physical failure-mechanism hypothesis

Map the signature to plausible physics, clearly labelled as a hypothesis for the failure-analysis engineer, not a diagnosis:

| Observed signature | Mechanism to investigate |
|---|---|
| Steadily accelerating leakage | Gate-oxide or dielectric degradation, TDDB precursor |
| Discrete step between intervals | Intermittent resistive short, marginal bond, ESD latent damage |
| Delay drift with stable current | NBTI or hot-carrier injection |
| Elevated leakage, normal delay, jointly anomalous | Localised defect not yet affecting timing paths |
| Multiple parts, same board region | Chamber gradient or socket contact, not a part defect |

**Framing:** the system does not replace the failure-analysis engineer, it hands them a prioritised, pre-argued case file.

## Deliverable artifact

A one-page **QA Disposition Report** per flagged part, exportable as PDF, containing all three layers plus the trajectory plot with the lot envelope, the datasheet limit line, the forecast point and the forecast uncertainty band. A mock-up of this on the Technical Approach slide is worth more than an architecture diagram.

---

# PART 7: DATA STRATEGY

## 7.1 The honest problem

There is no public ISRO burn-in dataset. Every team faces this. How we handle it is itself a differentiator. Address it head-on on the Feasibility slide.

**Never claim synthetic results are industrial validation.** Phrase every number as "prototype demonstration on simulated burn-in trajectories; production performance requires validated representative data."

## 7.2 Three sources

**(a) Physics-informed synthetic generator, primary.** Spec below.

**(b) SECOM (UCI ML Repository), for external validity.** Real semiconductor fab process data: 1,567 samples, 590 sensor features, 104 fails, roughly 1:14 imbalance, with missing values and many low-variance columns. The closest public match for rare in-spec failures under heavy imbalance with unlabelled feature semantics. Running our pipeline on SECOM proves the method is not tuned to our own generator.

**(c) ODDS library benchmarks, for method comparison.** thyroid, glass, satimage-2, shuttle, smtp, speech. Used by ITC 2020, so our numbers become directly comparable to a published ITC paper. That is an unusually strong claim for an SIH deck.

## 7.3 Synthetic generator specification

**Structure:** N lots x M parts (M >= 30 per AEC-Q001), P parameters, 4 timepoints, plus a **socket/board position** field for the spatial layer.

**Parameters:** Iddq / standby current (uA), input leakage (nA), propagation delay (ns), supply current, threshold-voltage shift. Include realistic cross-correlations, because leakage and delay are not independent.

**Good-part model with hierarchical variance:**
```
x_it = mu_lot(t) + u_lot + u_i + eps_it

mu_lot(t) = v0 + lambda * t^beta      (beta 0.5 to 1.0, saturating)
u_lot   ~ lot-level random effect
u_i     ~ component-level random effect
eps_it  ~ measurement noise
```
The lot-level and component-level random effects are what make DPAT genuinely need to be *dynamic*. Without lot-to-lot shift, static limits would suffice and our whole argument collapses.

**Arrhenius grounding:** use the AF table from Part 1.2 to set the degradation magnitude so that the 168h endpoint corresponds to roughly 1.5 field-years at Ea = 0.7 eV.

**Defect archetypes, with the hard constraint:**

> **Every injected defective part must remain inside datasheet limits at all four checkpoints.**

Say this out loud in the deck. It demonstrates we understood the problem rather than the keywords.

| Type | Signature | Caught by |
|---|---|---|
| **I. Steep drifter** | Elevated lambda (5 to 10x lot median), low enough v0 to stay in spec | Module B, lot-relative slope |
| **II. Step defect** | Nominal drift plus a discrete jump between checkpoints | Max-interval-jump, monotonicity |
| **III. Centre-hider** | Near lot median on every univariate parameter, anomalous only in the joint distribution | **Multivariate layer only** |
| **IV. Correlation break** | Individually normal, but the usual leakage-delay relationship is inverted | Mahalanobis, PCA residual |
| **V. High-but-stable** | Elevated v0, minimal drift. **Should NOT be rejected.** | Tests our false-positive control |
| **VI. Lot-wide shift** | Entire lot shifted. **Process issue, not part defects.** | Tests hierarchical baseline |
| **VII. Fixture artifact** | Anomalies clustered by socket position | **Spatial layer only** |

Types V, VI and VII are the ones that prove we built a *discriminating* system rather than a trigger-happy one. Most teams will only build Types I and II.

**Contamination:** 0.5 to 2% overall, with Type III at roughly 500 DPPM. Keep the hard class rare, because that is the realistic regime.

**Censoring, and this is the trap Source D caught:** simulate parts that fail hard at 96h and get pulled, leaving no 168h value. Then verify our training pipeline does not silently drop them, because dropping them means training only on survivors and systematically under-estimating drift. Distinguish in the data schema between:
- true 168h measurement
- predicted 168h measurement
- missing because the part failed and was pulled
- missing because of an equipment or process issue

**Ground truth:** held out in a separate labels file, never touched by the pipeline, mirroring the PS's "actual hidden ground-truth values."

## 7.4 Label bootstrapping with SDCL

If we want a supervised classifier despite label scarcity, SDCL (Sigma Deviation Count Labeling) generates pseudo-labels from sigma deviations on initially-passed samples. Useful for bootstrapping.

**State the circularity caveat:** a model trained on rule-generated labels can at best reproduce the rule. Use SDCL to bootstrap, never to validate that we beat the statistical baseline. Say this in the deck. Spotting circular evaluation is a rare skill and it costs one sentence.

---

# PART 8: EVALUATION

## 8.1 Validation methodology, non-negotiable

**Never randomly split rows.** Components from the same lot in both train and test leaks lot-level information and inflates every score.

```
Train:      earlier lots
Validation: later lots
Test:       completely unseen lot
```

Also use leave-one-lot-out and device-family holdout. This demonstrates generalisation to future production rather than memorisation of one batch. If a judge asks one methodological question, it will be this one.

## 8.2 Anomaly detection, FN-penalised

**Lead metric: recall at fixed yield loss.** Report the full recall-versus-yield-loss curve, not a single point. This is how the industry actually reasons, and it encodes the asymmetry the PS demands. Use the industry's **93% yield goal** as a reference operating point.

**Escape rate**, the operational headline:
```
Escape Rate = (defective components incorrectly passed) / (all defective components)
```

**Also report:**
- **PR-AUC** as the primary threshold-free metric. State why: under roughly 1% prevalence, ROC-AUC is dominated by the majority class and reads optimistically. Note for balance that ITC 2020 uses AUROC because with a single known failure it degenerates cleanly into a yield-loss reading. Report both, lead with PR-AUC, explain the choice.
- **F-beta with beta = 2 or higher**, which weights recall above precision by design.
- **Cost-weighted objective:** `Cost = C_FN * n_FN + C_FP * n_FP` with `C_FN : C_FP` around 1000:1. Show the cost-minimising threshold sits deep in the high-recall region. That is the quantitative justification for our conservatism.
- **Guaranteed-FNR validation:** empirical FNR on held-out data versus the alpha we calibrated for, across repeated splits.
- **Detection lead time:** how early can we flag a part that eventually fails.

**Never report plain accuracy.** At 1% contamination a model that flags nothing scores 99%. Say this in one line on the slide. It shows we know the metric traps.

## 8.3 Drift prediction

- **MAE** on `v168`, the stated metric, optimised directly
- RMSE, median absolute error, MAPE only where values are not near zero
- MAE broken out for good versus defective parts separately, because errors on defective parts matter more
- **Prediction-interval coverage** and interval width
- Error broken down by lot and device family

## 8.4 Explainability, made quantitative

Explainability is usually scored qualitatively. Making it quantitative is a differentiator:

- **Percentage of decisions carrying a valid, complete reason**
- **Reason-correctness:** for flagged parts, does the stated primary evidence match the injected defect archetype? This gives a hard percentage.
- **Explanation stability:** does the reason change under small measurement perturbations?
- **Counterfactual validity:** does the stated counterfactual actually flip the decision when applied?

## 8.5 The ablation table, our best slide

| Rung | Method | Recall @ 93% yield | PR-AUC | Escape rate | Guaranteed FNR |
|---|---|---|---|---|---|
| L0 | Static datasheet limits | | | | n/a |
| L1 | AEC-Q001 DPAT | | | | n/a |
| L2 | plus trajectory features | | | | n/a |
| L3 | plus robust multivariate | | | | n/a |
| L4 | plus unsupervised ensemble | | | | n/a |
| L5 | plus conformal risk control | | | | <= alpha |

A monotone improvement across rungs, with the industry standard sitting at L1 as a named reference point, tells the entire story without narration.

---

# PART 9: TECH STACK

| Layer | Choice | Justification |
|---|---|---|
| Core | Python 3.11, NumPy, pandas, SciPy | |
| Robust stats | `sklearn.covariance.MinCovDet`, `EllipticEnvelope`, `RobustScaler` | MCD for robust covariance |
| Detectors | `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM` | Per ITC 2020 benchmark |
| Regression | LightGBM / XGBoost, `HuberRegressor`, `QuantileRegressor`, `RandomForestQuantileRegressor` | Best-in-class on small tabular |
| Conformal | `mapie` or `crepes`, or ~40 lines direct | FNR guarantee |
| Explainability | `shap`, plus custom rule-text generator | Standard and model-agnostic |
| Dashboard | Streamlit for speed, FastAPI plus React if time allows | Demo surface |
| Plots | Plotly / Matplotlib | Trajectory with lot envelope |
| Reports | ReportLab or WeasyPrint | QA Disposition Report PDF |
| Packaging | Docker | Deployability story |

**On deliberately not using deep learning:** four timepoints, tabular parametric data, 30 to 500 parts per lot, and explainability as a scored criterion. Robust statistics and gradient-boosted trees are correct; deep sequence models are not. **State this as an explicit engineering decision.** Judges have seen a hundred gratuitous transformers. A team that justifies not using one reads as more senior, not less ambitious.

**Deployment note for the Feasibility slide:** the whole pipeline runs on a laptop. For a screening tool deployed at a test floor, with data-security constraints typical of ISRO, local or edge deployment with no cloud dependency is a genuine feasibility strength, not a limitation.

---

# PART 10: DIFFERENTIATION

## 10.1 What the median competing submission will look like

LSTM or autoencoder over the four timepoints. Isolation Forest. A Streamlit dashboard. Accuracy and F1. Synthetic data where defective parts violate datasheet limits and are therefore trivially separable. No standards citation. "Explainability" as a SHAP bar chart with no domain interpretation. Random train-test split.

## 10.2 Our seven wedges, as one-liners

1. *"AEC-Q001 Dynamic Part Average Testing is the industry standard for exactly this. We implement it as our baseline, then beat it."*
2. *"Published industrial data shows escaped latent defects sit at the median of every individual parameter. Univariate limits structurally cannot catch them."*
3. *"Our false-negative rate is not tuned, it is bounded, with a distribution-free finite-sample guarantee. It is the ML analogue of MIL-STD-883's PDA."*
4. *"We use socket position to tell a bad component apart from a bad oven. Rejecting parts for a chamber gradient is pure yield loss."*
5. *"Our defective parts pass every datasheet limit at every checkpoint, by construction. We built the hard problem on purpose."*
6. *"We validate by holding out entire lots, not random rows, because random splits leak lot information and inflate every score."*
7. *"The output is not a score. It is a one-page QA disposition report a Group Head can sign."*

## 10.3 One-line pitch

> **A physics-grounded, lot-adaptive, uncertainty-aware screening system that detects latent component degradation before any datasheet limit is violated, with a statistically guaranteed escape rate and an explanation an inspector can act on.**

## 10.4 Naming

Optional, but a name makes the deck memorable. Candidates from the research: **LADS** (Latent Anomaly and Drift Screening), **DriftGuard**, **AegisBurn**, **EEE-Sentinel**. Pick one and use it consistently, or skip it entirely. Do not spend more than ten minutes on this.

---

# PART 11: PPT PLAN, SIX SLIDES

Hard constraints: six slides maximum including title, points and diagrams over paragraphs, template pointers unchanged, PDF submission only, delete the instructions slide before uploading.

### Slide 1: Title Page
PS ID SIH26170 | Title | Theme: Smart Automation | Category: Software | Team ID | Team Name. Nothing else.

### Slide 2: Proposed Solution
Template pointers: detailed explanation of the solution, how it addresses the problem, innovation and uniqueness.

**Hero visual, the most important asset in the entire deck:** a trajectory plot. Flat datasheet limit line at the top. A tight band of good-part trajectories at the bottom. One part climbing steeply toward the limit but never crossing it. Annotation: *"Passes every limit. Fails in orbit."*

Beside it, the numeric anchor:
```
Datasheet limit   50 uA
Lot median        10 uA
This component    45 uA
Traditional       PASS
Our system        HIGH RISK
```

**Content blocks:**
- The gap in one line: static limits catch violations, not abnormal behaviour
- Module A: dynamic lot-relative outlier detection, AEC-Q001 DPAT baseline extended to multivariate
- Module B: drift forecast from 0h and 24h with a calibrated upper bound
- **Innovation box:** guaranteed escape rate via conformal risk control, plus fixture-versus-component discrimination, plus an inspector-readable disposition report

### Slide 3: Technical Approach
Template pointers: technologies used, methodology and process for implementation.

**Visual:** left-to-right pipeline.
```
Burn-in data (0/24/96/168h)
  -> Data quality gate
  -> Absolute limit check
  -> [Module A detector stack | Module B drift predictor]
  -> Spatial fixture check
  -> Conformal calibration
  -> PASS / WATCH / REVIEW / REJECT + Disposition Report
```

Beside it, the **ablation ladder L0 to L5** as a small stepped bar chart with real numbers.

**Content:** tech stack as icons. The DPAT formula shown explicitly, one line, because it proves domain fluency. The robust-estimator comparison with the breakdown-point column if space allows. The one-line justification for classical over deep.

### Slide 4: Feasibility and Viability
Template pointers: feasibility analysis, potential challenges and risks, strategies for overcoming them.

**Address the data problem head-on. Do not hide it.**

Data strategy: physics-grounded synthetic generator (include the Arrhenius AF table, it is compact and impressive), plus SECOM, plus ODDS benchmarks for external validity.

Risk table:

| Risk | Mitigation |
|---|---|
| No public ISRO burn-in data | Three-source strategy, synthetic plus SECOM plus ODDS |
| Few defective labels | One-class and semi-supervised methods, SDCL bootstrap |
| Lot-to-lot variation | Hierarchical peer-group baselines with fallback |
| Sensor or chamber drift | Spatial residual check, calibration gate |
| Data leakage in validation | Grouped lot-based splits, leave-one-lot-out |
| Over-rejection | Explicit yield-loss curve, WATCH/REVIEW tiers |
| Defective escape | Conformal FNR bound, upper-bound decision rule |
| Synthetic-only validation | Results labelled as simulated, pilot plan stated |

Compute: runs on a laptop, deployable locally with no cloud dependency.

### Slide 5: Impact and Benefits
Template pointers: impact on target audience, benefits (social, economic, environmental).

- **Mission:** a single escaped latent defect can compromise a payload and a multi-year mission. Non-recoverable in orbit.
- **Economic:** reduced over-rejection of expensive long-lead space-grade parts. Early rejection at 24h frees oven capacity and reduces energy consumption. External precedent: an ML outlier-detection deployment reduced DPPM by 396 and saved over $250,000 in test and packaging costs, with 70% of flagged outliers confirmed by HTOL stress testing.
- **Process:** ISRO's own component screening at SAC Ahmedabad gains an audit-ready statistical layer over existing MIL-STD-883 flows. Augments, does not replace.
- **Transferable:** automotive, medical implants, defence electronics, any zero-defect domain.

Icons and short lines only. This is the least technical slide, do not waste it on prose.

### Slide 6: Research and References
Two columns. Standards block, papers block, datasets block. Most teams waste this slide, so a dense credible one is cheap differentiation.

### Deck-wide rules
- Every slide carries one numeric or diagrammatic anchor. No slide is text-only.
- One accent colour for "defective", one for "good", one for "limit". Reuse across all charts.
- No em dashes anywhere in the copy.
- Export to PDF and inspect the render. Fonts and charts shift.
- Delete the instructions slide before uploading.

---

# PART 12: ACTION PLAN

Today is 26 Aug. Team target: submission-ready before 10 Sep. Portal deadline: 20 Sep.

## Phase 0: Freeze definitions before any code (26 to 28 Aug)

Write down and lock:
- Target component type and primary parameter
- Time points and absolute limit
- Direction of degradation
- Definition of "defective"
- Prediction horizon and mode (early-warning vs mid-test)
- Acceptable false-positive burden
- Safety slope definition

Do not begin coding before these are fixed. Every research pass that skipped this step produced a scope that drifted.

## Week 1: 26 Aug to 1 Sep, lock the story

| Task | Owner | Output |
|---|---|---|
| Circulate this document, team reads Part 0 and Part 2.2 | Mihir | Shared understanding |
| Pull AEC-Q001 Rev-D and MIL-STD-883 TM1015, verify exact wording | Mihir | Verified citations |
| Skim ITC 2020 and the Conformal Risk Control abstract | Mihir | Confidence on the two key claims |
| **Lock the six-slide outline, one sentence per slide** | Whole team | The spine |
| Build the data dictionary and schema | Model track | Field definitions |
| Synthetic generator v1: good parts plus Type I | Model track | Working data |

## Week 2: 2 to 8 Sep, build in parallel

| Task | Owner | Output |
|---|---|---|
| Add defect Types II through VII, verify all stay in spec | Model track | The hard dataset |
| Implement L0, L1 (DPAT), L2 features | Model track | First ablation rows |
| Implement L3, L4, run ablation with grouped lot splits | Model track | Ablation table |
| Module B: linear, Huber, LightGBM, quantile. Report MAE | Model track | MAE number |
| **Build all six slides with placeholder numbers from day one** | PPT track | Draft deck |
| **Draw the Slide 2 hero visual** | PPT track | The single most important asset |

## Week 3: 9 to 14 Sep, polish and buffer

| Task | Output |
|---|---|
| Conformal layer plus guarantee-validation plot | Final ablation row |
| Spatial fixture-check layer | Slide 3 differentiator |
| QA Disposition Report generator plus screenshot | Slide 3 asset |
| Validate on SECOM | External validity claim |
| Internal review, then review with Shivangi Ma'am | Feedback |
| Export PDF, verify render, submit | Submitted |

## Sequencing rule

The deck never blocks on the model. Build slides with placeholder numbers on day one and swap real numbers in as they land. A finished deck with three real numbers beats a half-finished deck waiting on a fourth.

## The one thing to protect

**Slide 2's hero visual.** If a judge understands the escape diagram in three seconds, everything after it lands. Assign it to whoever is best at visual design and give them the full two weeks.

---

# PART 13: JUDGE QUESTION PREPARATION

Expect these. Have a one-sentence answer for each.

1. **Where will the data come from?** Synthetic generator grounded in Arrhenius physics, validated on SECOM and ODDS. Production deployment requires representative data from SAC, which we state as a pilot dependency.
2. **How do you handle so few defect labels?** One-class and unsupervised methods trained on known-good parts, confirmed rejects held out for validation, SDCL only for bootstrapping.
3. **Why not deep learning?** Four timepoints, tabular, small n per lot, and explainability is scored. Robust statistics and GBMs dominate this regime.
4. **How do you avoid false alarms?** Hierarchical peer-group baselines, WATCH and REVIEW tiers instead of binary reject, explicit yield-loss curve, spatial fixture check.
5. **How do you know your flagged parts are actually defective?** Our synthetic ground truth, plus published industrial precedent where 70% of ML-flagged outliers subsequently failed HTOL stress testing.
6. **How do you validate across new lots?** Grouped lot-based splits and leave-one-lot-out. Never random row splits.
7. **Can the model explain its decision?** Three layers: deterministic rule text, SHAP attribution, physical mechanism hypothesis. One-page disposition report per part.
8. **How do you distinguish a sensor fault from a component fault?** Spatial residual analysis over socket and board position, plus a calibration and data-quality gate before the model runs.
9. **Does this replace engineering limits?** No. Absolute datasheet limits remain mandatory and the model never loosens them, per AEC-Q001's own rule.
10. **What happens when the model is uncertain?** It returns WATCH or REVIEW with the prediction interval, rather than forcing a binary decision.
11. **Does it generalise to another parameter?** The pipeline is parameter-agnostic. Only the limit and degradation direction are configured per parameter.
12. **What is your false negative rate?** Bounded by alpha via conformal risk control, with the empirical validation plot to prove the bound holds.

---

# PART 14: WHAT NOT TO DO

- Do not exceed six slides.
- Do not claim deep learning is automatically better.
- Do not remove or loosen absolute datasheet limits.
- Do not use a single global baseline across all lots.
- Do not randomly split rows from the same lot across train and test.
- Do not use 96h data in the early-warning model.
- Do not report plain accuracy.
- Do not hide uncertainty.
- Do not present synthetic results as industrial validation.
- Do not put an unsourced quantitative claim in the deck.
- Do not build synthetic defects that violate datasheet limits.
- Do not present wafer-map spatial methods for packaged burn-in data.
- Do not call every outlier defective.
- Do not let the model make an unexplained irreversible rejection with no engineering review path.

---

# PART 15: REFERENCES

## Standards
- **AEC-Q001 Rev-D**, Guidelines for Part Average Testing, Automotive Electronics Council, Dec 2011: http://www.aecouncil.com/Documents/AEC_Q001_Rev_D.pdf
- **AEC-Q001 Rev-C** (additional worked detail): http://www.aecouncil.com/Documents/AEC_Q001_Rev_C.pdf
- **MIL-STD-883 Method 1015.10**, Burn-In Test: https://ai-hmi.com/wp-content/uploads/2015/03/std883_1015.pdf
- **MIL-STD-883** full standard, NASA S3VI mirror: https://s3vi.ndc.nasa.gov/ssri-kb/static/resources/std883.pdf
- **MIL-STD-883K Chg-3**: https://s3vi.ndc.nasa.gov/ssri-kb/static/resources/MIL-STD-883K_CHG-3.pdf
- **NASA EEE-INST-002**: https://nepp.nasa.gov/pages/EEE-INST-002.cfm
- **NASA EEE Parts Screening**, Preferred Practice 1401: https://extapps.ksc.nasa.gov/reliability/Documents/Preferred_Practices/1401.pdf
- **NASA parts assurance overview**: https://ntrs.nasa.gov/api/citations/20210025929/downloads/parts_assurance_ASQ-final.docx.pdf
- **JAXA JERG-2-027**, Commercial EEE Parts in Space: https://sma.jaxa.jp/TechDoc/Docs/E_JAXA-JERG-2-027.pdf
- **Environmental Stress Screening Guidelines**, DTIC: https://apps.dtic.mil/sti/tr/pdf/ADA347723.pdf
- **ISRO Standards** (ISRO-PAS / ISRO-PAX): https://www.sac.gov.in/SAC_Industry_Portal/isro_standards.html
- **ISRO SAC Electronics QA**: https://www.sac.gov.in/SAC_Industry_Portal/rnqa_electronics.html

## Core papers
- Hu, Nguyen, He, Li, *Advanced Outlier Detection Using Unsupervised Learning for Screening Potential Customer Returns*, **ITC 2020**, UCSB and NXP: https://par.nsf.gov/servlets/purl/10253082 (the centre-of-distribution finding and our benchmark comparison)
- Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, **ICLR 2024**, arXiv:2208.02814: https://arxiv.org/abs/2208.02814 Code: https://github.com/aangelopoulos/conformal-risk
- *Performing Machine Learning Based Outlier Detection for Automotive Grade Products* (XGBoost plus Mahalanobis, weak ICs failing SLT and QEA): https://ieeexplore.ieee.org/document/10118207/
- *Statistical outlier screening for latent defects*, IEEE: https://ieeexplore.ieee.org/abstract/document/6531962/
- Powell, Pair et al., *Delta Iddq for testing reliability*: https://ieeexplore.ieee.org/abstract/document/843876/
- Engelke, Polian et al., *Delta-IDDQ Testing of Resistive Short Defects*: https://ira.informatik.uni-freiburg.de/nanoscale/2006-Engelke-Delta-IDDQ_testing.pdf
- O'Neill, *Production Multivariate Outlier Detection Using Principal Components*, ITC 2008
- Sumikawa et al., *Screening Customer Returns with Multivariate Test Analysis*, ITC 2012
- Ni, Rui, Zhuo, Li, Wen, Nie, *Reducing Testing Costs and Minimizing Defect Escapes Using Dynamic Neighborhood Range and Shapley Values*, **ACM TODAES 2025**
- Senoner, Netland, Feuerriegel, *Using explainable artificial intelligence to improve process quality: evidence from semiconductor manufacturing*, **Management Science** 68(8), 2022
- *An AI-Based Framework for Burn-in Reduction in the Semiconductor Manufacturing Industry*, Springer: https://link.springer.com/chapter/10.1007/978-3-031-59361-1_5
- *Analysis of Latent Defect Detection Using Sigma Deviation Count Labeling (SDCL)*
- Meeker and Escobar, *A Review of Accelerated Test Models*, arXiv:0708.0369: https://arxiv.org/pdf/0708.0369 (Arrhenius formalism)
- *Degradation modeling and RUL prediction for electronic device under multiple stress influences*, Scientific Reports 2025: https://www.nature.com/articles/s41598-025-03786-y
- Rousseeuw and Van Driessen, *A Fast Algorithm for the Minimum Covariance Determinant Estimator*, Technometrics 41(3), 1999
- Liu, Ting, Zhou, *Isolation Forest*, ICDM 2008
- Meinshausen, *Quantile Regression Forests*, JMLR 2006
- *A Survey on Explainable Anomaly Detection*, ACM: https://dl.acm.org/doi/10.1145/3609333

## Industry and applied
- proteanTecs, *How To Improve DPPM By 10X Without Affecting Yield*: https://semiengineering.com/how-to-improve-dppm-by-10x-without-affecting-yield/
- proteanTecs, *Cut Defects, Not Yield: Outlier Detection with ML Precision* (DPPM reduced 396, $250k saved, 70% of flagged outliers failed HTOL): https://semiwiki.com/analytics/proteantecs/354024-cut-defects-not-yield-outlier-detection-with-ml-precision/
- *Part Average Testing finds and rejects outlier ICs*, EDN: https://www.edn.com/part-average-testing-finds-and-rejects-outlier-ics/
- yieldHUB, *Part Average Test (PAT)*, static vs dynamic rationale: https://www.yieldhub.com/solutions/services-and-modules/part-average-test-pat/
- Keysight, *Moving from Static Limits to Dynamic PAT Limits*: https://www.keysight.com/us/en/assets/3121-1257/case-studies/Moving-from-Static-Limits-to-Dynamic-Part-Average-Test-PAT-Limits.pdf
- *Real Time Dynamic Application of Part Average Testing*, CS MANTECH: https://csmantech.org/wp-content/acfrcwduploads/field_5e8cddf5ddd10/post_2557/048.pdf
- TI, *Understanding of Long-Term Stability and Acceleration Factor*: https://www.ti.com/lit/ml/slap177/slap177.pdf
- Analog Devices, *Details of Reliability Calculations* (Ea = 0.7 eV convention): https://www.analog.com/en/support/quality-and-reliability/reliability/wafer-fabrication-data/details-of-reliability-calculations.html
- NASA S3VI, *Burn-In*: https://s3vi.ndc.nasa.gov/ssri-kb/topics/47/

## Datasets
- **SECOM**, UCI ML Repository, 1,567 x 590, 104 fails: https://archive.ics.uci.edu/ml/datasets/SECOM
- **ODDS**, Outlier Detection DataSets, Stony Brook: http://odds.cs.stonybrook.edu
- Awesome Industrial Datasets, SECOM entry: https://github.com/jonathanwvd/awesome-industrial-datasets/blob/master/markdown/secom.md
