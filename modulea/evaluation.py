"""
Evaluation harness for Module A.

Built and self-tested BEFORE any detector, because if the harness is wrong then
every number downstream is wrong and the error is invisible.

Three things this file is strict about.

1. LOT GROUPING. Splits are always by lot, never by row. Components from one
   lot appearing in both train and test leaks lot-level information -- the lot
   random effect is exactly what dynamic limits are supposed to absorb -- and
   inflates every score. `lot_splits` is the only sanctioned way to divide the
   data and it returns lot ids, not row indices, so a row-wise split is not
   expressible.

2. LABELS. Defective = I, II, III, IV, Vb. Everything else is GOOD, and that
   deliberately includes the Va / VI / VII traps. Flagging a trap is yield
   loss and is counted as such; that is the entire reason the traps exist.

3. NO ACCURACY. At 1.75% prevalence a detector that flags nothing scores
   98.25%. `metrics_from_score` will refuse to compute it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFECT_TYPES = ("I_STEEP_DRIFTER", "II_STEP_DEFECT", "III_CENTRE_HIDER",
                "IV_CORRELATION_BREAK", "Vb_EXTREME_LEVEL")
TRAP_TYPES = ("Va_MILDLY_HIGH_STABLE", "VI_LOT_SHIFT", "VII_FIXTURE_ARTIFACT")

COST_FN = 1000.0     # a test escape
COST_FP = 1.0        # a good part scrapped
YIELD_TARGET = 0.07  # 93% yield goal, ITC 2020 reference operating point


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class Dataset:
    meas: pd.DataFrame
    gt: pd.DataFrame
    cfg: dict

    @property
    def params(self) -> list[str]:
        return [q["name"] for q in self.cfg["parameters"]]

    @property
    def checkpoints(self) -> list[float]:
        return list(self.cfg["config"]["checkpoints_h"])

    @property
    def limits(self) -> dict[str, tuple[float, float]]:
        return {q["name"]: (q["limit_lo"], q["limit_hi"]) for q in self.cfg["parameters"]}

    @property
    def log_params(self) -> list[str]:
        return [q["name"] for q in self.cfg["parameters"] if q.get("log_scale")]

    def labels(self) -> pd.Series:
        """is_defective, indexed by component_id, in a fixed canonical order."""
        y = self.gt.set_index("component_id")["defect_type"].isin(DEFECT_TYPES)
        return y.rename("is_defective")

    def type_of(self) -> pd.Series:
        return self.gt.set_index("component_id")["defect_type"]

    def severity_of(self) -> pd.Series:
        s = self.gt.set_index("component_id")["severity"]
        return s.fillna("").astype(str)

    def lot_of(self) -> pd.Series:
        return self.gt.set_index("component_id")["lot_id"]

    def completed_burnin(self) -> pd.Series:
        """False for parts pulled from the oven after a hard failure.

        These parts never reach the end of burn-in, so they cannot ship and
        cannot become a test escape. Whether they belong in the recall
        denominator is a definitional choice, not a fact; see ASSUMPTIONS in
        the report. Both variants are reported.
        """
        return (self.gt.set_index("component_id")["censor_status_168h"]
                != "PULLED_FAILED")


def load(datadir: str | Path = "data") -> Dataset:
    d = Path(datadir)
    src = d / "burnin_measurements.csv.gz"
    if not src.exists():
        src = d / "burnin_measurements.csv"
    meas = pd.read_csv(src)
    gt = pd.read_csv(d / "ground_truth.csv")
    cfg = json.loads((d / "config.json").read_text())
    return Dataset(meas, gt, cfg)


# --------------------------------------------------------------------------
# lot-grouped splitting
# --------------------------------------------------------------------------

@dataclass
class LotSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def describe(self) -> str:
        return (f"train {len(self.train)} lots ({self.train[0]}..{self.train[-1]}), "
                f"val {len(self.val)} lots ({self.val[0]}..{self.val[-1]}), "
                f"test {len(self.test)} lots ({self.test[0]}..{self.test[-1]})")

    def mask(self, lot_of: pd.Series, which: str) -> pd.Series:
        return lot_of.isin(getattr(self, which))


def lot_splits(lot_ids, train=0.60, val=0.20) -> LotSplit:
    """Chronological lot split: earlier lots train, later lots validate, the
    last block is a completely unseen test set.

    Lot ids sort lexicographically in production order (LOT000..LOT239), so a
    plain sort is a time ordering. This mirrors the real question -- does the
    detector generalise to future production -- rather than memorisation of one
    batch.
    """
    lots = sorted(pd.unique(pd.Series(lot_ids)))
    n = len(lots)
    a, b = int(round(n * train)), int(round(n * (train + val)))
    return LotSplit(lots[:a], lots[a:b], lots[b:])


def leave_one_lot_out(lot_ids, n_folds: int | None = None, seed: int = 0):
    """Yield (held_out_lot, training_lots). With 240 lots a full LOLO means 240
    refits, which is not affordable for L3/L4, so `n_folds` subsamples the held
    out lots. The subsample is seeded and reported."""
    lots = sorted(pd.unique(pd.Series(lot_ids)))
    if n_folds is not None and n_folds < len(lots):
        rng = np.random.default_rng(seed)
        held = sorted(rng.choice(lots, size=n_folds, replace=False).tolist())
    else:
        held = lots
    for h in held:
        yield h, [l for l in lots if l != h]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _as_arrays(y_true, score):
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError("non-finite scores; impute or drop before scoring")
    return y, s


def threshold_at_yield_loss(y_true, score, yield_loss: float) -> float:
    """Score threshold that rejects `yield_loss` of the GOOD parts.

    Note the threshold is set on the good parts alone. Setting it on all parts
    would let the contamination move it, which is the same error the p99
    estimator makes.
    """
    y, s = _as_arrays(y_true, score)
    good = s[~y]
    if len(good) == 0:
        raise ValueError("no good parts to calibrate a yield-loss threshold on")
    return float(np.quantile(good, 1.0 - yield_loss))


def recall_yield_curve(y_true, score, n_points: int = 400):
    """Full recall-versus-yield-loss curve. Thresholds are quantiles of the
    GOOD score distribution, so the x axis is literally yield loss."""
    y, s = _as_arrays(y_true, score)
    good, bad = s[~y], s[y]
    yls = np.unique(np.concatenate([[0.0], np.geomspace(1e-4, 1.0, n_points)]))
    rec, real_yl = [], []
    for q in yls:
        thr = np.quantile(good, 1.0 - q) if q > 0 else np.inf
        rec.append(float((bad >= thr).mean()) if len(bad) else np.nan)
        real_yl.append(float((good >= thr).mean()))
    return np.array(real_yl), np.array(rec), yls


def cost_optimal(y_true, score, c_fn: float = COST_FN, c_fp: float = COST_FP):
    """Minimise Cost = c_fn * n_FN + c_fp * n_FP over the threshold."""
    y, s = _as_arrays(y_true, score)
    order = np.argsort(-s)
    ys = y[order]
    n_bad = int(y.sum())
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    fn = n_bad - tp
    cost = c_fn * fn + c_fp * fp
    cost0 = c_fn * n_bad                      # flag nothing
    i = int(np.argmin(cost))
    if cost0 <= cost[i]:
        return {"cost": float(cost0), "threshold": float(np.inf),
                "yield_loss": 0.0, "recall": 0.0}
    thr = float(s[order][i])
    n_good = int((~y).sum())
    return {"cost": float(cost[i]), "threshold": thr,
            "yield_loss": float(fp[i] / n_good),
            "recall": float(tp[i] / n_bad) if n_bad else float("nan")}


def metrics_from_score(y_true, score, yield_target: float = YIELD_TARGET,
                       extra_yls=(0.005, 0.01, 0.02, 0.05)) -> dict:
    """Threshold-free metrics plus the operating point at a fixed yield loss.

    PR-AUC leads. Under ~1.75% prevalence ROC-AUC is dominated by the majority
    class and reads optimistically; it is reported too, because ITC 2020 uses
    AUROC and comparability matters, but it is not the headline.
    """
    y, s = _as_arrays(y_true, score)
    if y.sum() == 0 or (~y).sum() == 0:
        raise ValueError("need both classes present")
    thr = threshold_at_yield_loss(y, s, yield_target)
    flag = s >= thr
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    recall = tp / max(int(y.sum()), 1)
    prec = tp / max(tp + fp, 1)
    beta2 = 4.0
    fbeta = ((1 + beta2) * prec * recall / (beta2 * prec + recall)
             if (prec + recall) > 0 else 0.0)
    out = {
        "n": int(len(y)), "n_defective": int(y.sum()),
        "prevalence_%": 100 * float(y.mean()),
        f"recall@{100 * (1 - yield_target):.0f}%yield": 100 * recall,
        "yield_loss_%": 100 * fp / max(int((~y).sum()), 1),
        "escape_rate_%": 100 * fn / max(int(y.sum()), 1),
        "precision_%": 100 * prec,
        "F2": fbeta,
        "PR_AUC": float(average_precision_score(y, s)),
        "AUROC": float(roc_auc_score(y, s)),
    }
    for q in extra_yls:
        t = threshold_at_yield_loss(y, s, q)
        out[f"recall@{q * 100:g}%YL"] = 100 * float((s[y] >= t).mean())
    co = cost_optimal(y, s)
    out["cost_min"] = co["cost"]
    out["cost_min_yield_loss_%"] = 100 * co["yield_loss"]
    out["cost_min_recall_%"] = 100 * co["recall"]
    out["cost_at_op"] = COST_FN * fn + COST_FP * fp
    return out


def metrics_from_flags(y_true, flags) -> dict:
    """Single operating point, for detectors that are inherently binary (L0, and
    DPAT at one fixed multiplier). Threshold-free metrics are not defined for a
    binary rule and are returned as NaN rather than faked."""
    y = np.asarray(y_true, dtype=bool)
    f = np.asarray(flags, dtype=bool)
    tp, fp = int((f & y).sum()), int((f & ~y).sum())
    fn = int((~f & y).sum())
    recall = tp / max(int(y.sum()), 1)
    prec = tp / max(tp + fp, 1)
    beta2 = 4.0
    return {
        "n": int(len(y)), "n_defective": int(y.sum()),
        "recall_%": 100 * recall,
        "yield_loss_%": 100 * fp / max(int((~y).sum()), 1),
        "escape_rate_%": 100 * fn / max(int(y.sum()), 1),
        "precision_%": 100 * prec,
        "F2": ((1 + beta2) * prec * recall / (beta2 * prec + recall)
               if (prec + recall) > 0 else 0.0),
        "PR_AUC": float("nan"), "AUROC": float("nan"),
        "cost_at_op": COST_FN * fn + COST_FP * fp,
    }


def per_type_breakdown(type_of: pd.Series, severity: pd.Series,
                       flag: pd.Series) -> pd.DataFrame:
    """Catch rate by defect type, and for III/IV by severity tier."""
    rows = []
    for t in sorted(pd.unique(type_of)):
        sel = type_of == t
        rows.append({"defect_type": t, "severity": "(all)", "n": int(sel.sum()),
                     "flagged_%": 100 * float(flag[sel].mean())})
        if t in ("III_CENTRE_HIDER", "IV_CORRELATION_BREAK"):
            for tier in ("severe", "moderate", "mild"):
                s2 = sel & (severity == tier)
                if s2.sum():
                    rows.append({"defect_type": t, "severity": tier,
                                 "n": int(s2.sum()),
                                 "flagged_%": 100 * float(flag[s2].mean())})
    return pd.DataFrame(rows)
