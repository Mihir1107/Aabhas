"""L3 robust multivariate and L4 unsupervised ML.

Fitting policy. Where a method allows it, covariance/density is fitted on GOOD
parts from TRAIN lots only, then everything is scored. That is what the research
doc asks for, but it is an assumption worth naming: in production nobody knows
which parts are good, so a clean reference set is a favour we are granting the
unsupervised methods. Both variants are therefore run -- clean-fit and
contaminated-fit (all train parts, ~1.75% contaminated) -- and the gap is
reported. If the gap is large, the clean-fit numbers are optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# --------------------------------------------------------------------------
# L3a: Mahalanobis, MCD vs plain sample covariance
# --------------------------------------------------------------------------

def mahalanobis_per_lot(ds, robust: bool = True, log_transform: bool = False,
                        agg: str = "median", support_fraction: float = 0.9
                        ) -> pd.Series:
    """Per lot and checkpoint, fit a covariance on the five parameters and score
    every part by Mahalanobis distance. Aggregate over checkpoints.

    robust=True uses Minimum Covariance Determinant, which is the multivariate
    analogue of preferring the median to the mean: a plain sample covariance is
    itself inflated by the outliers being hunted.

    log_transform applies log to the log-normal parameters first. Mahalanobis
    assumes elliptical contours, which log-normal marginals violate, so this is
    worth measuring rather than assuming.
    """
    m = ds.meas[ds.meas["measurement_status"] == "MEASURED"]
    cols = ds.params
    logset = set(ds.log_params) if log_transform else set()
    acc: dict[str, list[float]] = {}
    for (_lot, _t), sub in m.groupby(["lot_id", "checkpoint_h"]):
        X = sub[cols].to_numpy(dtype=float)
        if log_transform:
            for j, c in enumerate(cols):
                if c in logset:
                    X[:, j] = np.log(np.maximum(X[:, j], 1e-9))
        if len(X) < 5 * len(cols):
            continue
        est = (MinCovDet(support_fraction=support_fraction, random_state=0)
               if robust else EmpiricalCovariance())
        d2 = est.fit(X).mahalanobis(X)
        for cid, v in zip(sub["component_id"].to_numpy(), d2):
            acc.setdefault(cid, []).append(float(v))
    f = np.median if agg == "median" else np.max
    s = pd.Series({k: float(f(v)) for k, v in acc.items()})
    return s.reindex(ds.gt["component_id"]).fillna(0.0)


# --------------------------------------------------------------------------
# shared design matrix for L3b / L4
# --------------------------------------------------------------------------

def design_matrix(zlvl: pd.DataFrame, ztraj: pd.DataFrame,
                  traj: pd.DataFrame, include_censor_flags: bool = False
                  ) -> pd.DataFrame:
    """Lot-relative z-scored level and trajectory features, plus explicit
    censoring indicators.

    NaN policy: the FEATURE TABLES are never imputed -- a censored part keeps
    NaN for anything undefined. But sklearn estimators cannot consume NaN, so
    inside this design matrix only, missing entries are filled with 0, which in
    lot-relative z space means "lot-typical". The censoring indicators are
    carried alongside so a model can tell an imputed zero from a measured one.
    """
    # censored_pulled is EXCLUDED by default and this is deliberate. It is
    # legitimately observable -- measurement_status is in the measurements file
    # -- but in this dataset P(defective | PULLED_FAILED) = 1.000, because the
    # generator draws hard failures exclusively from Types I and II. A detector
    # keying on it books 17.1% recall at 0% yield loss without detecting
    # anything. That is the generator's construction showing through, not
    # capability, so it is kept out of the design matrix and reported
    # separately.
    blocks = [zlvl, ztraj]
    if include_censor_flags:
        blocks.append(traj[["censored_pulled", "n_missing_checkpoints"]])
    X = pd.concat(blocks, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(0.0)


# --------------------------------------------------------------------------
# L3b: PCA T-squared and Q residual
# --------------------------------------------------------------------------

def pca_scores(X: pd.DataFrame, fit_idx: pd.Index, n_components: float = 0.95):
    """Hotelling's T-squared (in-model) and Q residual / SPE (out-of-model).

    They answer different questions: T2 is an unusual position within the
    subspace the good parts occupy, Q is variation that does not lie in that
    subspace at all. Reported separately because a correlation break should
    show up in Q rather than T2.

    Caveat worth stating: T2 assumes Gaussianity, which parametric test data
    routinely violates. That is why the non-parametric kNN rule is also run.
    """
    sc = StandardScaler().fit(X.loc[fit_idx])
    Z = sc.transform(X)
    Zf = sc.transform(X.loc[fit_idx])
    p = PCA(n_components=n_components, random_state=0).fit(Zf)
    T = p.transform(Z)
    var = p.explained_variance_
    t2 = (T ** 2 / np.maximum(var, 1e-12)).sum(axis=1)
    recon = p.inverse_transform(T)
    q = ((Z - recon) ** 2).sum(axis=1)
    return (pd.Series(t2, index=X.index), pd.Series(q, index=X.index),
            int(p.n_components_))


def knn_distance(X: pd.DataFrame, fit_idx: pd.Index, k: int = 20,
                 max_fit: int = 40000, seed: int = 0) -> pd.Series:
    """Mean distance to the k nearest good-part neighbours. Non-parametric, so
    it avoids the ellipticity assumption T-squared needs."""
    rng = np.random.default_rng(seed)
    fi = np.asarray(fit_idx)
    if len(fi) > max_fit:
        fi = rng.choice(fi, size=max_fit, replace=False)
    sc = StandardScaler().fit(X.loc[fi])
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(sc.transform(X.loc[fi]))
    d, _ = nn.kneighbors(sc.transform(X))
    return pd.Series(d.mean(axis=1), index=X.index)


# --------------------------------------------------------------------------
# L4: unsupervised ML
# --------------------------------------------------------------------------

def isolation_forest(X, fit_idx, contamination=0.02, seed=0) -> pd.Series:
    sc = StandardScaler().fit(X.loc[fit_idx])
    m = IsolationForest(n_estimators=300, contamination=contamination,
                        random_state=seed, n_jobs=-1).fit(sc.transform(X.loc[fit_idx]))
    return pd.Series(-m.score_samples(sc.transform(X)), index=X.index)


def lof(X, fit_idx, k=20, max_fit=40000, contamination=0.02, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    fi = np.asarray(fit_idx)
    if len(fi) > max_fit:
        fi = rng.choice(fi, size=max_fit, replace=False)
    sc = StandardScaler().fit(X.loc[fi])
    m = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=contamination,
                           n_jobs=-1).fit(sc.transform(X.loc[fi]))
    return pd.Series(-m.score_samples(sc.transform(X)), index=X.index)


def ocsvm(X, fit_idx, nu=0.02, max_fit=8000, seed=0) -> pd.Series:
    """RBF one-class SVM. Training is O(n^2) so the good-part reference is
    subsampled to `max_fit`; the subsample size is reported, not hidden."""
    rng = np.random.default_rng(seed)
    fi = np.asarray(fit_idx)
    if len(fi) > max_fit:
        fi = rng.choice(fi, size=max_fit, replace=False)
    sc = StandardScaler().fit(X.loc[fi])
    m = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(sc.transform(X.loc[fi]))
    return pd.Series(-m.decision_function(sc.transform(X)), index=X.index)


def autoencoder(X, fit_idx, hidden=(48, 12, 48), max_fit=40000, seed=0):
    """Small dense autoencoder. torch is not installed in this environment, so
    this is sklearn's MLPRegressor fitted X -> X, which is the same object: a
    bottlenecked feed-forward reconstruction. Returns the total reconstruction
    error and the per-feature error, the latter because the explainability
    layer will need it.
    """
    rng = np.random.default_rng(seed)
    fi = np.asarray(fit_idx)
    if len(fi) > max_fit:
        fi = rng.choice(fi, size=max_fit, replace=False)
    sc = StandardScaler().fit(X.loc[fi])
    Zf = sc.transform(X.loc[fi])
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                     solver="adam", max_iter=200, random_state=seed,
                     early_stopping=True, n_iter_no_change=8).fit(Zf, Zf)
    Z = sc.transform(X)
    err = (Z - m.predict(Z)) ** 2
    return (pd.Series(err.sum(axis=1), index=X.index),
            pd.DataFrame(err, index=X.index, columns=X.columns))


def union_ensemble(scores: dict[str, pd.Series], y_good_mask: pd.Series,
                   yield_loss: float) -> pd.Series:
    """Union rule: flag if ANY detector flags, each detector thresholded at the
    SAME per-detector yield-loss budget measured on good parts.

    False negatives are catastrophic and false positives merely cost money, so
    a union maximises recall by construction. Whether it is worth the yield it
    costs is an empirical question and is reported, not assumed -- note that a
    union of d detectors each at yield loss q costs somewhere between q and d*q
    depending on how correlated their errors are.
    """
    flag = None
    for _, s in scores.items():
        thr = np.quantile(s[y_good_mask], 1 - yield_loss)
        f = s >= thr
        flag = f if flag is None else (flag | f)
    return flag
