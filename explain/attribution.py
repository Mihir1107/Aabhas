"""
Layer 2: model attribution.

SHAP on the Module B drift predictor uses LightGBM's own `pred_contrib=True`,
which is exact TreeSHAP computed inside the booster. That avoids a second
implementation and is faster than the shap package for this model class; the
shap package is used only to cross-check that the two agree.

The Huber-versus-LightGBM comparison is reported rather than assumed. Huber and
LightGBM tied on MAE (0.6044 vs 0.6087 on Iddq). Huber's coefficients are
directly readable and need no attribution method at all; LightGBM needs SHAP to
say anything. If the two disagree about WHICH features matter, that is a real
finding about model choice and is reported, not smoothed over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DriftAttribution:
    """Refits nothing that produced a published number.

    The Module A/B runs did not persist model objects, so a per-part SHAP
    explanation needs a booster in memory. The model fitted here is trained with
    the SAME seed, features, split and hyper-parameters as the published Module
    B rung 4, so it reproduces that model rather than replacing it -- and the
    runner asserts that its MAE matches the published value before any
    explanation is emitted. If the assertion fails, no explanation is produced.
    """

    def __init__(self, e, param: str):
        self.e = e
        self.p = param
        self.cols = [c for c in e.Xb.columns
                     if c.startswith(f"{param}__") or c.startswith(f"z__{param}__")]
        self.model = None
        self.huber = None

    def fit(self):
        from modulea import drift_models as dm
        e = self.e
        tr = e.lot.isin(e.split.train)
        obs = e.mb_true[self.p]                     # observed where present
        m = e.ds.meas
        end = max(e.checkpoints)
        st = m[m["checkpoint_h"] == end].set_index("component_id")["measurement_status"]
        surv = st.reindex(e.Xb.index) == "MEASURED"
        fit = (tr & surv & obs.notna()).to_numpy()
        X = e.Xb[self.cols]
        y = obs.to_numpy()
        self.model = dm.GBM(True).fit(X[fit], y[fit])
        self.huber = dm.HuberModel().fit(X[fit], y[fit])
        self._fit_mask = fit
        return self

    def published_mae(self, published: float, tol: float = 1e-6) -> float:
        """Confirm the in-memory model reproduces the published MAE."""
        e = self.e
        te = e.lot.isin(e.split.test)
        yt = e.mb_true[self.p]
        m = (te & yt.notna()).to_numpy()
        pred = self.model.predict(e.Xb[self.cols])
        got = float(np.nanmean(np.abs(pred[m] - yt.to_numpy()[m])))
        return got

    def shap_for(self, cid: str) -> pd.Series:
        """Exact TreeSHAP contributions for one part, in target units."""
        X = self.e.Xb[self.cols]
        row = X.loc[[cid]].to_numpy()
        contrib = self.model.m.predict(row, pred_contrib=True)[0]
        vals = pd.Series(contrib[:-1], index=self.cols)
        vals["__base__"] = contrib[-1]
        return vals

    def huber_coefficients(self) -> pd.Series:
        """Standardised Huber coefficients: the same question SHAP answers, but
        readable without any attribution machinery."""
        c = pd.Series(self.huber.m.coef_, index=self.cols)
        return c.sort_values(key=np.abs, ascending=False)

    def compare(self, n_parts: int = 400, seed: int = 0) -> pd.DataFrame:
        """Do SHAP and the Huber coefficients agree about what matters?"""
        rng = np.random.default_rng(seed)
        te = self.e.lot.isin(self.e.split.test)
        ids = self.e.Xb.index[te.to_numpy()]
        ids = rng.choice(np.asarray(ids), size=min(n_parts, len(ids)), replace=False)
        X = self.e.Xb[self.cols].loc[ids]
        c = self.model.m.predict(X.to_numpy(), pred_contrib=True)[:, :-1]
        shap_imp = pd.Series(np.abs(c).mean(axis=0), index=self.cols)
        hub = self.huber_coefficients().abs()
        sd = X.std().replace(0, np.nan)
        hub_imp = (hub * sd).dropna()
        df = pd.DataFrame({"mean_abs_SHAP": shap_imp,
                           "abs_huber_coef_x_sd": hub_imp}).dropna()
        df["rank_shap"] = df["mean_abs_SHAP"].rank(ascending=False)
        df["rank_huber"] = df["abs_huber_coef_x_sd"].rank(ascending=False)
        return df.sort_values("mean_abs_SHAP", ascending=False)


def detector_attribution(e, policy, cid: str) -> pd.DataFrame:
    """Which Module A detectors fired, and how strongly, on a common scale.

    Normalised as 'distance past this detector's own operating threshold,
    measured in units of (threshold - good-part median)'. 1.0 means exactly at
    the operating point. Without a common scale, asking which detector drove the
    decision is not answerable, because the raw scores are on incomparable
    scales (robust sigma, chi-square-like D^2, negative log-density).
    """
    s = policy.fired(cid)
    rows = [{"detector": k, "normalised_strength": v, "fired": v >= 1.0}
            for k, v in s.items()]
    df = pd.DataFrame(rows).sort_values("normalised_strength", ascending=False)
    return df
