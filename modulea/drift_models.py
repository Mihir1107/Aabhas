"""Module B model ladder.

Every model exposes fit(X, y) / predict(X). Quantile models additionally expose
predict_upper(X) for the 95% bound that actually drives the decision.

LightGBM is used where available; if the import fails the runner falls back to
sklearn's HistGradientBoostingRegressor, which is the same histogram-based GBDT
algorithm, and the substitution is reported rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

TARGET_H = 168.0


class LinearSlope:
    """Rung 1. v168 = v0 + s1 * 168. The physical strawman.

    Degradation in the generator is v0 + lambda * t^beta with beta in [0.5, 1.0],
    so it decelerates. Extrapolating the 0->24 h slope linearly to 168 h must
    therefore OVER-predict. If it does not, something upstream is wrong and the
    runner stops.
    """
    def __init__(self, param: str):
        self.p = param

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X[f"{self.p}__v0"] + X[f"{self.p}__s1"] * TARGET_H).to_numpy()


class PowerLaw:
    """Rung 2. v(t) = v0 + lambda * t^beta, two free parameters.

    With only 0 h and 24 h visible, lambda and beta are not jointly identifiable
    from a single part: two points fix one degree of freedom. beta is therefore
    estimated ONCE per parameter from the training lots' aggregate drift curve
    and held fixed, and lambda is then solved per part from d1. That is the
    honest version of a two-parameter fit under a two-point constraint, and it
    is stated rather than implied.

    beta is estimated with the three corrections established earlier: difference
    per part, aggregate within lot, and use the mean rather than the median
    (measurement noise biases a median drift low at 24 h, which flattens the
    log-log slope).
    """
    def __init__(self, param: str, beta: float | None = None):
        self.p = param
        self.beta = beta

    def fit_beta(self, wide: pd.DataFrame, lot_of: pd.Series, checkpoints, idx):
        ts = [t for t in checkpoints if t > 0]
        t0 = min(checkpoints)
        x = np.log(np.array(ts))
        d = pd.DataFrame({t: wide[(self.p, t)] - wide[(self.p, t0)] for t in ts})
        d = d.reindex(idx)
        per_lot = []
        for _lot, g in d.groupby(lot_of.reindex(idx)):
            mu = g.mean(axis=0).to_numpy()
            if np.isfinite(mu).all() and (mu > 0).all():
                per_lot.append(float(np.polyfit(x, np.log(mu), 1)[0]))
        self.beta = float(np.mean(per_lot)) if per_lot else 0.7
        return self.beta

    def fit(self, X, y):
        return self

    def predict(self, X):
        b = self.beta
        lam = X[f"{self.p}__d1"] / (24.0 ** b)
        return (X[f"{self.p}__v0"] + lam * (TARGET_H ** b)).to_numpy()


class HuberModel:
    """Rung 3. Robust linear regression, resistant to outliers, interpretable."""
    def __init__(self, epsilon: float = 1.35, alpha: float = 1e-4):
        self.sc = StandardScaler()
        self.m = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)

    def fit(self, X, y):
        Z = self.sc.fit_transform(np.nan_to_num(X.to_numpy(), nan=0.0))
        self.m.fit(Z, y)
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(np.nan_to_num(X.to_numpy(), nan=0.0)))


class GBM:
    """Rung 4. Gradient-boosted trees, optimised for MAE directly.

    loss is absolute_error because MAE is the metric the problem statement
    names; optimising squared error and reporting MAE would be a mismatch.
    """
    def __init__(self, use_lgb: bool, quantile: float | None = None, seed: int = 0):
        self.q = quantile
        self.use_lgb = use_lgb
        if use_lgb:
            import lightgbm as lgb
            obj = "quantile" if quantile else "regression_l1"
            kw = dict(objective=obj, n_estimators=400, learning_rate=0.05,
                      num_leaves=31, min_child_samples=40, verbose=-1,
                      random_state=seed, n_jobs=-1)
            if quantile:
                kw["alpha"] = quantile
            self.m = lgb.LGBMRegressor(**kw)
        else:
            kw = dict(max_iter=400, learning_rate=0.05, min_samples_leaf=40,
                      random_state=seed)
            if quantile:
                self.m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=quantile, **kw)
            else:
                self.m = HistGradientBoostingRegressor(
                    loss="absolute_error", **kw)

    def fit(self, X, y):
        self.m.fit(X.to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X.to_numpy())


class QuantileForest:
    """Rung 5b. Meinshausen's quantile regression forest.

    Keeps the full set of training targets in each leaf instead of averaging,
    so conditional quantiles come out natively from one fitted model rather than
    needing a separate model per quantile. Falls back to a compact in-house
    implementation over sklearn's RandomForest when quantile-forest is absent.
    """
    def __init__(self, use_qf: bool, seed: int = 0, n_estimators: int = 200):
        self.use_qf = use_qf
        self.seed = seed
        if use_qf:
            from quantile_forest import RandomForestQuantileRegressor
            self.m = RandomForestQuantileRegressor(
                n_estimators=n_estimators, min_samples_leaf=20,
                random_state=seed, n_jobs=-1)
        else:
            self.m = RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=20,
                random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        Xa = np.nan_to_num(X.to_numpy(), nan=0.0)
        self.m.fit(Xa, y)
        if not self.use_qf:
            leaves = self.m.apply(Xa)
            self._store = []
            for t in range(leaves.shape[1]):
                df = pd.DataFrame({"leaf": leaves[:, t], "y": y})
                self._store.append(df.groupby("leaf")["y"].apply(np.array).to_dict())
        return self

    def predict(self, X):
        return self.m.predict(np.nan_to_num(X.to_numpy(), nan=0.0))

    def predict_quantile(self, X, q: float):
        Xa = np.nan_to_num(X.to_numpy(), nan=0.0)
        if self.use_qf:
            return self.m.predict(Xa, quantiles=[q]).ravel()
        leaves = self.m.apply(Xa)
        out = np.empty(len(Xa))
        for i in range(len(Xa)):
            vals = []
            for t in range(leaves.shape[1]):
                v = self._store[t].get(leaves[i, t])
                if v is not None:
                    vals.append(v)
            out[i] = np.quantile(np.concatenate(vals), q) if vals else np.nan
        return out
