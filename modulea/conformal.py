"""
L5: Conformal Risk Control.

Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control", ICLR 2024
(arXiv:2208.02814). Distribution-free, finite-sample control of the EXPECTED
value of a monotone loss. Here the loss is the false-negative indicator, so the
guarantee is a bound on the escape rate of the AI system itself.

The mechanism, for a score where HIGHER means more anomalous and a threshold
lambda where we flag `score >= lambda`:

    L(lambda) = fraction of DEFECTIVE calibration parts with score < lambda
                (i.e. the false negative rate at that threshold)

L is non-decreasing in lambda. Given n calibration points and a target alpha,
choose

    lambda_hat = inf { lambda : (n/(n+1)) * L(lambda) + B/(n+1) <= alpha }

with B the loss upper bound (1 for a 0/1 loss). The (n+1) correction is what
makes the guarantee finite-sample rather than asymptotic: it pays for the fact
that the calibration set is finite and the test point is exchangeable with it.

Then E[L(lambda_hat)] <= alpha on future exchangeable data.

EXCHANGEABILITY IS THE WHOLE ASSUMPTION, and it is where this can quietly
break. Parts from one lot share a lot random effect, so they are exchangeable
with each other but NOT with parts from a different lot. Calibrating on rows
drawn at random from lots that also appear in test would leak lot-level
information and make the guarantee look tighter than it is. Every calibration
split here is therefore BY LOT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate(scores_cal: np.ndarray, y_cal: np.ndarray, alpha: float,
              n_grid: int = 2000) -> float:
    """Return lambda_hat controlling the false-negative rate at level alpha.

    Only the DEFECTIVE calibration points enter the loss, because the loss is
    the false-negative indicator. n is therefore the number of defective
    calibration parts, not the calibration set size -- that distinction matters
    a lot at 1.75% prevalence and getting it wrong would silently loosen the
    correction term.
    """
    s = np.asarray(scores_cal, float)
    y = np.asarray(y_cal, bool)
    pos = s[y]
    n = len(pos)
    if n == 0:
        return float("inf")
    grid = np.unique(np.concatenate([
        np.quantile(s, np.linspace(0, 1, n_grid)), pos, [np.inf]]))
    grid = np.sort(grid)
    # L is non-decreasing in lambda; find the largest lambda meeting the bound
    best = -np.inf
    for lam in grid:
        loss = float((pos < lam).mean())
        rhs = (n * loss + 1.0) / (n + 1.0)
        if rhs <= alpha:
            best = lam
        else:
            break
    return float(best)


def evaluate(scores_test: np.ndarray, y_test: np.ndarray, lam: float) -> dict:
    s = np.asarray(scores_test, float)
    y = np.asarray(y_test, bool)
    flag = s >= lam
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return {
        "lambda": lam,
        "empirical_FNR": float((~flag[y]).mean()) if n_pos else np.nan,
        "recall_%": 100 * float(flag[y].mean()) if n_pos else np.nan,
        "yield_loss_%": 100 * float(flag[~y].mean()) if n_neg else np.nan,
        "n_defective": n_pos,
    }


def lot_grouped_splits(lot_of: pd.Series, eligible: pd.Index, n_repeats: int,
                       frac_cal: float = 0.5, seed: int = 0):
    """Split the eligible LOTS into calibration and evaluation halves.

    Splitting by lot rather than by row is what keeps exchangeability honest:
    within a lot, parts share a lot random effect, so a row-wise split would put
    near-duplicates of the calibration parts into the evaluation set.
    """
    rng = np.random.default_rng(seed)
    lots = np.array(sorted(pd.unique(lot_of.reindex(eligible))))
    for r in range(n_repeats):
        perm = rng.permutation(lots)
        k = int(round(len(perm) * frac_cal))
        yield set(perm[:k].tolist()), set(perm[k:].tolist())


def sweep(scores: pd.Series, y: pd.Series, lot_of: pd.Series,
          eligible: pd.Index, alphas, n_repeats: int = 40,
          frac_cal: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Repeat the whole calibrate-then-evaluate cycle over many lot-grouped
    splits. A guarantee demonstrated once is an anecdote; the spread is the
    evidence."""
    rows = []
    for rep, (cal_lots, ev_lots) in enumerate(
            lot_grouped_splits(lot_of, eligible, n_repeats, frac_cal, seed)):
        m_cal = lot_of.reindex(eligible).isin(cal_lots)
        m_ev = lot_of.reindex(eligible).isin(ev_lots)
        idx_cal = eligible[m_cal.to_numpy()]
        idx_ev = eligible[m_ev.to_numpy()]
        sc, yc = scores.reindex(idx_cal).to_numpy(), y.reindex(idx_cal).to_numpy()
        se, ye = scores.reindex(idx_ev).to_numpy(), y.reindex(idx_ev).to_numpy()
        for a in alphas:
            lam = calibrate(sc, yc, a)
            r = evaluate(se, ye, lam)
            r.update({"alpha": a, "repeat": rep,
                      "n_cal_defective": int(np.asarray(yc, bool).sum())})
            rows.append(r)
    return pd.DataFrame(rows)
