# QA inspector demo

The problem statement scores explainability by whether the system can *justify
its classification to a QA inspector*. This is that inspector's screen.

## Run it

```bash
pip install -r requirements.txt
python -m streamlit run demo/app.py     # from the repository root, not from demo/
```

Opens on <http://localhost:8501>. First load takes about 10 seconds (it builds
the feature matrices once and caches them); the first visit to a lot takes a few
seconds more while every component in it is dispositioned, then it is instant.

`streamlit run demo/app.py` is the usual invocation and is equivalent, but the
`streamlit` console script exits silently on at least one Windows setup we
tested while `python -m streamlit` works there. Use the `python -m` form if the
short one does nothing.

## What it shows

Pick any of the 48 held-out test lots — 24,000 components no detector was
fitted on — and work the triage list.

- **Six-tier disposition**, not a pass/fail bit, each with the action a QA
  inspector would actually take. `FIXTURE_SUSPECT` and `MEASUREMENT_INVALID`
  are the two that make the system defensible: one says the chamber is at
  fault rather than the component, the other says the data is not good enough
  to judge on.
- **The evidence the decision rests on**, generated live by the same
  `explain/` code that writes the PDF disposition reports in `reports/`.
- **Trajectories** for all five parameters against the lot median and the
  AEC-Q001 6σ dynamic PAT band, so "abnormal relative to its own lot" is
  visible rather than asserted.
- **Module B's 168 h forecast** from 0 h and 24 h only, against a safe limit
  derived from prior lots.
- **Board map**, which is the one to demo. Try **LOT238**: 35 components come
  back `FIXTURE_SUSPECT` and on the socket grid they visibly cluster by
  position. That is the system declining to scrap 35 good parts because the
  anomaly tracks the oven, not the components. A flagged part is only called
  `FIXTURE_SUSPECT` when its board is clustered **and** its own shift since 0 h
  has the thermal sign (leakage up, delay up, vth down), so a real defect on a
  busy board is still dispositioned as a component. **LOT192** has a real chamber
  trip, and shows the system refusing to disposition on invalid readings.
- **Reveal ground truth**, hidden until pressed. Read the evidence, decide,
  then check whether the system was right.

## On authenticity

Every tier, score, evidence line and forecast is computed live by the committed
pipeline. Nothing is mocked, staged, or hand-picked, and the app reflects
whatever `results/` currently holds — if you re-run the pipeline, the demo
changes with it.

Ground truth is present in the same `Dataset` object the app loads, but it is
never an input to any displayed decision. It is read in exactly one place: the
reveal control. The header says so on screen.

## Offline fallback

`demo/inspector.html` is a single self-contained page with the same views over
three pre-exported lots. It needs no Python, no server and no network — open
the file. Use it only if Streamlit will not run on the demo machine; the
Streamlit app is the real deliverable because it runs the pipeline rather than
a snapshot of it.

Rebuild the fallback after re-running the pipeline:

```bash
python make_demo_data.py && python build_demo.py
```
