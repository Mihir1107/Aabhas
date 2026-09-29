# Burn-in screening — SIH26170

AI-driven anomaly detection and drift prediction for component burn-in
(ISRO, Smart Automation). A part can pass every datasheet limit at every
checkpoint and still carry a latent defect. This repository screens each
component against the lot it was burned in with (**Module A**), forecasts its
168 h value from the first 24 hours (**Module B**), puts a distribution-free
bound on the escape rate, and explains every decision on a one-page report a QA
inspector can sign.

**Documentation: <https://mihir1107.github.io/Aabhas/>** — the full guide,
chapter by chapter, generated from `results/` so its numbers cannot drift from
the code. Building slides? Use
[`results/PRESENTATION_PACKAGE.md`](results/PRESENTATION_PACKAGE.md): every
number there carries the condition it was measured under.

## Headline results

All on the 48 held-out **test lots** (24,000 parts, 427 defective) unless
marked, dataset `dataset-v1.2`.

| | Result | Condition |
|---|---|---|
| Escape rate | **45.7% → 7.3%** | AEC-Q001 dynamic PAT alone vs the full stack (C1 → C3), both at identical 7.00% yield loss |
| Shifted lot | **358 vs 0** good parts scrapped | static vs dynamic PAT, Type VI trap, 3,000 parts, all lots, 6σ |
| Module B | Iddq MAE **0.61 µA** on the 168 h value | from 0 h + 24 h only, LightGBM; several times larger on defective parts (see docs) |
| Escape-rate bound | **no significant violation** at any target from 1% to 20% | conformal risk control, 40 lot-grouped splits per target |

Exact figures and every caveat: the docs site, or `results/`.

## Quick start

```bash
pip install -r requirements.txt           # lightgbm and svglib are REQUIRED
python -m modulea.test_harness            # self-test the evaluation harness
python validate_dataset.py                # 24 dataset assertions + calibration
python -m streamlit run demo/app.py       # the QA inspector, live
```

No Python? Open `demo/inspector.html` in a browser: the same inspector over
three exported test lots, fully offline.

## Reproducing everything

Stages run in order; each reads the previous stage's output from `results/`.
Running them out of order silently reuses stale artifacts.

```bash
python generate_data.py          # writes data/  (optional: data/ is committed)
python validate_dataset.py       # 24 hard assertions + calibration tables
python run_ablation.py           # Module A, L0-L4 -> scores, flags, per-type
python run_cumulative.py         # cumulative ladder C0-C4, score-level fusion
python run_module_b.py           # Module B + upper bounds        (needs lightgbm)
python run_slopes.py             # safety-slope rules             (needs Module B)
python run_conformal.py          # L5 conformal risk control      (needs A and B)
python make_report.py            # Module A tables + figures 1-5
python make_mb_report.py         # Module B tables + figures 6, 7, 10
python run_explainability.py     # explainability metrics, decision-layer table
python make_reports.py           # the QA disposition PDFs         (needs svglib)
python make_demo_data.py && python build_demo.py   # offline inspector page
python make_docs.py              # the documentation site in docs/
```

About 20 minutes end to end. `results/module_b_provenance.json` records which
gradient-boosting backend actually ran: without lightgbm the code falls back to
scikit-learn and the Module B figures shift slightly.

## Repository layout

| Path | Contents |
|---|---|
| `generate_data.py`, `validate_dataset.py` | Synthetic dataset generator and its validator |
| `data/` | The frozen dataset: measurements (no labels), ground truth, config |
| `modulea/` | Evaluation harness, detectors L0–L4, features, Module B, conformal |
| `explain/` | Decision policy, evidence, rule text, counterfactuals, PDF report |
| `run_*.py`, `make_*.py` | Pipeline stages, in the order above |
| `results/` | Every table, figure and metric the pipeline produces |
| `reports/` | Five example QA disposition reports (PDF + rule text) |
| `demo/` | Streamlit inspector and its offline HTML fallback |
| `docs/`, `docs_src/` | The generated documentation site and its generator |

## The dataset

`dataset-v1.2`: 240 lots × 500 components, five parameters (`iddq_ua`,
`leakage_na`, `prop_delay_ns`, `supply_current_ma`, `vth_shift_mv`) at 0, 24,
96 and 168 h of burn-in at 125 °C. Arrhenius at Ea = 0.7 eV against 55 °C use
gives AF = 77.66, so 168 h stands in for 1.488 field-years.

Measurements (`data/burnin_measurements.csv.gz`) carry no labels. Ground truth
(`data/ground_truth.csv`) holds, per part: the injected type, severity,
failure mechanism, censoring status and the latent 168 h values of censored
parts.

| Type | Signature | Label |
|---|---|---|
| I | Steep drifter, drift rate 5–10× the lot median | defect |
| II | Discrete step between two checkpoints | defect |
| III | Centre-hider: 17th–83rd percentile on every parameter, anomalous only jointly | defect |
| IV | Correlation break: leaky *and* slow | defect |
| Vb | 5.5–7.5σ outside the lot, inside the datasheet limit (the PS's worked example) | defect |
| Va | Mildly high along the process corner, stable | **good (trap)** |
| VI | Whole-lot process shift | **good (trap)** |
| VII | Socket thermal gradient on a burn-in board | **good (trap)** |

Genuine defects are 1.75% of parts. **Every injected part stays inside the
datasheet limits at every measured checkpoint, by construction**, so a static
limit catches none of them. `measurement_status` separates `PULLED_FAILED`
(hard failure between 96 h and 168 h) from `MISSING_EQUIPMENT` (a chamber trip
or tester fault, which also hits good parts), so a missing reading is not a
label. The generator's model, the Type III feasibility ceiling and the
correlation-sensitivity study are covered in the
[dataset chapter](https://mihir1107.github.io/Aabhas/dataset.html).

**What changed in v1.2.** v1.1 shrank every over-limit injection to *exactly*
`limit − 3% margin`, so 158 readings sat at 97.000 nA and 139 at 9.880 ns, and
the rule "value equals the cap" caught about 12% of defects at near-zero yield
loss. v1.2 pulls each active shrink back by a random factor from its own RNG
stream; parts that never needed shrinking are unchanged, and assertion A1b now
guards against it. The Module A ladder figures are unchanged to two decimals.
See [`results/CHANGED_NUMBERS.md`](results/CHANGED_NUMBERS.md).

## Validation methodology

Never split rows randomly: components from one lot share a lot effect, and a
row-wise split leaks it. Splits are chronological by lot — train LOT000–143,
validate LOT144–191, test LOT192–239 — and `modulea/evaluation.py` returns lot
ids, so a row-wise split cannot be expressed. Accuracy is never reported: at
1.75% prevalence a detector that flags nothing scores 98.25%.

## Known limitations

Validation is entirely synthetic; survivorship bias in Module B is measured but
not yet corrected; Type III has 60 parts (8 in the test lots) and Type VI has
none in the test lots, so those figures are all-lots with intervals. Full list
in the [limitations chapter](https://mihir1107.github.io/Aabhas/limitations.html).
