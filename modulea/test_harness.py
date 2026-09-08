"""Self-test for the evaluation harness on cases with known answers.

Run: python3 -m modulea.test_harness
"""
import numpy as np
import pandas as pd

from modulea import evaluation as ev


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def main() -> None:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
        ok = ok and cond

    print("=== 1. perfect separator ===")
    y = np.array([True] * 10 + [False] * 990)
    s = np.where(y, 100.0, 0.0)
    m = ev.metrics_from_score(y, s)
    check("PR-AUC = 1", approx(m["PR_AUC"], 1.0))
    check("AUROC = 1", approx(m["AUROC"], 1.0))
    check("recall = 100%", approx(m["recall@93%yield"], 100.0))
    check("escape rate = 0", approx(m["escape_rate_%"], 0.0))

    print("\n=== 2. pure noise, score independent of label ===")
    rng = np.random.default_rng(0)
    y = np.zeros(20000, dtype=bool); y[:200] = True
    s = rng.normal(size=20000)
    m = ev.metrics_from_score(y, s)
    check("AUROC ~ 0.5", abs(m["AUROC"] - 0.5) < 0.05, f"{m['AUROC']:.3f}")
    check("PR-AUC ~ prevalence (0.01)", abs(m["PR_AUC"] - 0.01) < 0.006,
          f"{m['PR_AUC']:.4f}")
    check("recall@93%yield ~ 7% (chance)", abs(m["recall@93%yield"] - 7.0) < 4.0,
          f"{m['recall@93%yield']:.1f}%")

    print("\n=== 3. yield-loss threshold is calibrated on GOOD parts only ===")
    y = np.zeros(10000, dtype=bool); y[:1000] = True
    s = np.concatenate([np.full(1000, 50.0), np.arange(9000, dtype=float)])
    thr = ev.threshold_at_yield_loss(y, s, 0.10)
    realised = float((s[~y] >= thr).mean())
    check("exactly 10% of GOOD rejected", abs(realised - 0.10) < 0.002,
          f"{realised * 100:.2f}%")
    contaminated = float(np.quantile(s, 0.90))
    check("differs from the all-parts quantile (contamination would move it)",
          not approx(thr, contaminated), f"good-only {thr:.0f} vs all-parts {contaminated:.0f}")

    print("\n=== 4. recall/yield-loss curve is monotone and anchored ===")
    y = np.zeros(5000, dtype=bool); y[:100] = True
    s = rng.normal(size=5000) + y * 2.0
    yl, rec, _ = ev.recall_yield_curve(y, s)
    check("recall non-decreasing in yield loss", bool(np.all(np.diff(rec) >= -1e-9)))
    check("yield loss non-decreasing", bool(np.all(np.diff(yl) >= -1e-9)))
    check("curve starts at (0, 0)", approx(yl[0], 0.0) and approx(rec[0], 0.0))
    check("curve ends at (1, 1)", approx(yl[-1], 1.0) and approx(rec[-1], 1.0))

    print("\n=== 5. cost optimum at 1000:1 sits deep in the high-recall region ===")
    y = np.zeros(20000, dtype=bool); y[:200] = True
    s = rng.normal(size=20000) + y * 1.5
    co = ev.cost_optimal(y, s)
    m = ev.metrics_from_score(y, s)
    check("cost-optimal recall > recall at 7% yield loss",
          co["recall"] * 100 > m["recall@93%yield"],
          f"{co['recall'] * 100:.1f}% vs {m['recall@93%yield']:.1f}%")
    check("cost-optimal beats flag-nothing", co["cost"] < ev.COST_FN * 200)
    # hand-computed reference
    thr = co["threshold"]
    fn = int((y & (s < thr)).sum()); fp = int((~y & (s >= thr)).sum())
    check("cost matches hand recomputation", approx(co["cost"], 1000 * fn + fp),
          f"{co['cost']:.0f} vs {1000 * fn + fp}")

    print("\n=== 6. flag-based metrics, hand-checked ===")
    y = np.array([True, True, True, False, False, False, False, False])
    f = np.array([True, False, True, True, False, False, False, False])
    m = ev.metrics_from_flags(y, f)
    check("recall = 2/3", approx(m["recall_%"], 200 / 3))
    check("yield loss = 1/5", approx(m["yield_loss_%"], 20.0))
    check("escape rate = 1/3", approx(m["escape_rate_%"], 100 / 3))
    check("precision = 2/3", approx(m["precision_%"], 200 / 3))
    check("cost = 1000*1 + 1*1", approx(m["cost_at_op"], 1001.0))
    check("threshold-free metrics are NaN, not faked", np.isnan(m["PR_AUC"]))

    print("\n=== 7. accuracy is never reported ===")
    m = ev.metrics_from_score(np.array([True] * 10 + [False] * 990),
                              np.arange(1000, dtype=float))
    check("no key contains 'accuracy'",
          not any("acc" in k.lower() for k in m))

    print("\n=== 8. lot splits are disjoint and chronological ===")
    lots = [f"LOT{i:03d}" for i in range(240)]
    sp = ev.lot_splits(lots)
    check("no lot in two splits",
          not (set(sp.train) & set(sp.val)) and not (set(sp.val) & set(sp.test))
          and not (set(sp.train) & set(sp.test)))
    check("all lots used", len(sp.train) + len(sp.val) + len(sp.test) == 240)
    check("chronological", max(sp.train) < min(sp.val) and max(sp.val) < min(sp.test),
          sp.describe())

    print("\n=== 9. a row-wise split would leak; verify grouping is by lot ===")
    lot_of = pd.Series([f"LOT{i // 100:03d}" for i in range(24000)],
                       index=[f"C{i:05d}" for i in range(24000)])
    sp2 = ev.lot_splits(lot_of)
    tr = lot_of[sp2.mask(lot_of, "train")].index
    te = lot_of[sp2.mask(lot_of, "test")].index
    check("no component in both train and test", len(set(tr) & set(te)) == 0)
    check("no LOT appears in both",
          len(set(lot_of[tr]) & set(lot_of[te])) == 0)

    print("\n=== 10. leave-one-lot-out folds are clean ===")
    folds = list(ev.leave_one_lot_out(lots, n_folds=5))
    check("5 folds requested, 5 returned", len(folds) == 5)
    check("held-out lot never in its own training set",
          all(h not in tr for h, tr in folds))
    check("training sets have 239 lots", all(len(tr) == 239 for _, tr in folds))

    print("\n=== 11. every runner calls functions that actually exist ===")
    # run_module_b.py called mb.slope_mission_based() for several commits after
    # that function was renamed to project_to_mission(). Python only notices at
    # the call site, 180 lines into a 20-minute script, so the crash landed
    # AFTER module_b.csv was written and BEFORE the safety slopes and the
    # prediction files the conformal stage consumes. The published artifacts and
    # the code that claims to produce them silently diverged for four commits.
    # This resolves every modulea/explain attribute a runner references,
    # statically, in under a second.
    import ast as _ast, importlib as _il, pathlib as _pl
    root = _pl.Path(__file__).resolve().parent.parent
    scripts = sorted(root.glob("run_*.py")) + sorted(root.glob("make_*.py")) + \
        [root / "sweep_correlation.py", root / "validate_dataset.py"]
    missing = []
    for f in scripts:
        if not f.exists():
            continue
        tree = _ast.parse(f.read_text(encoding="utf-8"))
        alias = {}
        for n in _ast.walk(tree):
            if isinstance(n, _ast.ImportFrom) and n.module and \
                    n.module.split(".")[0] in ("modulea", "explain"):
                for a in n.names:
                    alias[a.asname or a.name] = f"{n.module}.{a.name}"
            elif isinstance(n, _ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] in ("modulea", "explain"):
                        alias[a.asname or a.name] = a.name
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Attribute) and isinstance(n.value, _ast.Name) \
                    and n.value.id in alias:
                try:
                    mod = _il.import_module(alias[n.value.id])
                except Exception:
                    continue
                if not hasattr(mod, n.attr):
                    missing.append(f"{f.name}:{n.lineno} {n.value.id}.{n.attr}")
    check(f"no runner references a missing function ({len(scripts)} scripts)",
          not missing, "; ".join(missing))

    print("\n" + ("ALL HARNESS SELF-TESTS PASSED" if ok else "HARNESS SELF-TESTS FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
