"""
Assemble, for every part, the evidence a decision rests on.

Everything here is READ-ONLY with respect to the frozen artifacts. Nothing is
refitted and no existing number is recomputed: detector scores come from
`results/scores.csv.gz`, Module B predictions from the saved CSVs, and the
feature blocks are a deterministic pure function of the frozen dataset.

ONE DELIBERATE SUBSTITUTION, stated rather than buried. Part 6 asks for
per-channel AUTOENCODER reconstruction error to localise an anomaly to a
measurement. The Module A run did not persist it, and recovering it means
refitting, which this session is not allowed to do. Instead the channel
localiser is the per-parameter decomposition of the robust Mahalanobis
distance:

    D^2 = z^T S^-1 z = sum_j z_j * (S^-1 z)_j

which is exact, additive, deterministic, and needs no model. It is arguably the
better instrument anyway: an inspector can be told "Iddq contributes 61% of the
joint anomaly" and check it by hand, which is not true of a reconstruction
error from a neural network.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

from modulea import evaluation as ev
from modulea import features as ft
from modulea import moduleb as mb

# Anchored to the repository root rather than the working directory. As a bare
# relative path this resolved against wherever the process happened to start, so
# anything launched from a subdirectory -- the Streamlit app in demo/ being the
# obvious case -- died with FileNotFoundError on results/scores.csv.gz. Set
# AABHAS_RESULTS to point somewhere else.
_ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("AABHAS_RESULTS", _ROOT / "results"))
DATA = Path(os.environ.get("AABHAS_DATA", _ROOT / "data"))
IQR_TO_SIGMA = 1.35
MAD_TO_SIGMA = 1.4826


class Evidence:
    """Everything needed to explain any part, loaded once."""

    def __init__(self, datadir: str | Path | None = None):
        datadir = DATA if datadir is None else datadir
        self.ds = ds = ev.load(datadir)
        self.params = ds.params
        self.checkpoints = ds.checkpoints
        self.limits = ds.limits
        self.gt = ds.gt.set_index("component_id")
        self.y = ds.labels()
        self.ty = ds.type_of()
        self.sev = ds.severity_of()
        self.lot = ds.lot_of()
        self.split = ev.lot_splits(sorted(set(self.lot)))

        self.scores = pd.read_csv(RESULTS / "scores.csv.gz", index_col=0)
        self.flags = pd.read_csv(RESULTS / "flags.csv.gz", index_col=0)
        self.mb_point = pd.read_csv(RESULTS / "moduleb_point_early.csv.gz", index_col=0)
        self.mb_upper = pd.read_csv(RESULTS / "moduleb_upper_early.csv.gz", index_col=0)
        self.mb_true = pd.read_csv(RESULTS / "moduleb_true168.csv.gz", index_col=0)

        self.wide = ft.wide_frame(ds)
        self.traj, self.ztraj, self.lvl, self.zlvl = ft.build_feature_matrix(ds)
        self.Xb = mb.build_features(ds, "early")
        self.l_safe = {p: mb.lot_safe_limit(ds, p) for p in self.params}

        self._lot_stats: dict = {}
        self._maha: dict = {}

    # ---------------- lot reference statistics ----------------
    def lot_stats(self, lot: str, t: float) -> pd.DataFrame:
        """Median, robust sigma (MAD and IQR) and n for each parameter, for one
        lot at one checkpoint, computed from MEASURED values only.

        Both estimators are reported because AEC-Q001 specifies IQR/1.35 while
        the MAD variant has the higher breakdown point; naming which one a
        number came from is what makes it auditable.
        """
        key = (lot, t)
        if key in self._lot_stats:
            return self._lot_stats[key]
        m = self.ds.meas
        s = m[(m["lot_id"] == lot) & (m["checkpoint_h"] == t)
              & (m["measurement_status"] == "MEASURED")]
        rows = []
        for p in self.params:
            a = s[p].dropna().to_numpy()
            med = float(np.median(a)) if len(a) else np.nan
            mad = MAD_TO_SIGMA * float(np.median(np.abs(a - med))) if len(a) else np.nan
            q1, q3 = (np.percentile(a, [25, 75]) if len(a) else (np.nan, np.nan))
            rows.append({"parameter": p, "n": len(a), "median": med,
                         "sigma_mad": mad, "sigma_iqr": (q3 - q1) / IQR_TO_SIGMA})
        out = pd.DataFrame(rows).set_index("parameter")
        self._lot_stats[key] = out
        return out

    def healthy_lot_drift(self, lot: str) -> pd.Series:
        """Median 0->24 h drift among the lot's parts, the peer baseline the
        rule text compares a part's drift against."""
        d = {}
        for p in self.params:
            v0 = self.wide[(p, 0.0)]
            v24 = self.wide[(p, 24.0)]
            sel = self.lot == lot
            d[p] = float((v24 - v0)[sel].median())
        return pd.Series(d)

    # ---------------- joint anomaly, decomposed by parameter ----------------
    def maha_contributions(self, cid: str, t: float | None = None):
        """Robust Mahalanobis D^2 for one part, and each parameter's additive
        share of it.

        Fitted per lot per checkpoint with MinCovDet on the five parameters,
        exactly as the Module A L3a detector does, so the D^2 quoted in a report
        is the same quantity the detector used.
        """
        lot = self.lot[cid]
        t = t if t is not None else max(self.checkpoints)
        key = (lot, t)
        if key not in self._maha:
            m = self.ds.meas
            s = m[(m["lot_id"] == lot) & (m["checkpoint_h"] == t)
                  & (m["measurement_status"] == "MEASURED")]
            X = s[self.params].to_numpy(float)
            if len(X) < 30:
                self._maha[key] = None
            else:
                mcd = MinCovDet(support_fraction=0.9, random_state=0).fit(X)
                inv = np.linalg.inv(mcd.covariance_)
                d2_all = mcd.mahalanobis(X)
                self._maha[key] = (mcd.location_, inv,
                                   pd.Series(d2_all, index=s["component_id"].to_numpy()))
        got = self._maha[key]
        if got is None:
            return np.nan, pd.Series(dtype=float), np.nan
        loc, inv, d2_all = got
        row = self.wide.loc[cid, [(p, t) for p in self.params]].to_numpy(float)
        if not np.isfinite(row).all():
            return np.nan, pd.Series(dtype=float), np.nan
        z = row - loc
        contrib = z * (inv @ z)                      # exact additive split
        d2 = float(z @ inv @ z)
        good = d2_all[d2_all.index.isin(
            self.gt.index[self.gt["defect_type"] == "GOOD"])]
        p99 = float(np.quantile(good, 0.99)) if len(good) else np.nan
        return d2, pd.Series(contrib, index=self.params), p99

    # ---------------- spatial ----------------
    def spatial_check(self, cid: str, flagged: pd.Series) -> dict:
        """Is this part's anomaly shared by its physical neighbours?

        A chamber gradient or a bad socket row produces flags that cluster on a
        board. A latent part defect does not. This is what turns 'reject' into
        'this is your oven, not your part'.
        """
        g = self.gt
        board = g.loc[cid, "board_id"]
        meta = self.ds.meas.drop_duplicates("component_id").set_index("component_id")
        same = meta.index[meta["board_id"] == board]
        n = len(same)
        k = int(flagged.reindex(same).fillna(False).sum())
        base = float(flagged.mean())
        expected = base * n
        # binomial tail: how surprising is this many flags on one board
        from scipy import stats
        p = float(stats.binom.sf(k - 1, n, max(base, 1e-9))) if n else np.nan
        return {"board_id": board, "n_on_board": n, "n_flagged_on_board": k,
                "expected_flagged": expected, "p_value": p,
                "clustered": bool(p < 0.01 and k >= 3)}

    # ---------------- data quality gate (Part 5.1) ----------------
    def data_quality(self, cid: str) -> dict:
        """Runs BEFORE any model, per Part 5.1. A missing 96 h value because the
        chamber tripped is a different object from one because the part failed
        and was pulled, and the two must never be collapsed."""
        m = self.ds.meas
        s = m[m["component_id"] == cid].sort_values("checkpoint_h")
        st = s["measurement_status"].tolist()
        n_missing_equip = int((s["measurement_status"] == "MISSING_EQUIPMENT").sum())
        pulled = bool((s["measurement_status"] == "PULLED_FAILED").any())
        return {"statuses": dict(zip(s["checkpoint_h"], st)),
                "n_missing_equipment": n_missing_equip,
                "pulled_failed": pulled,
                "valid": n_missing_equip == 0,
                "n_checkpoints_measured": int((s["measurement_status"] == "MEASURED").sum())}
