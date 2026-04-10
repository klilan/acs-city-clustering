"""
gmm.py
Gaussian Mixture Model clustering via EM.

Mixture log-likelihood:
    l(theta) = sum_i log( sum_k pi_k * N(x_i | mu_k, Sigma_k) )

Parameters theta = {pi_k, mu_k, Sigma_k} estimated by sklearn's EM implementation.
Covariance type defaults to 'full'; 'tied' and 'diag' are tested in sensitivity analysis.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


RANDOM_SEED = 42
DEFAULT_COV_TYPE = "full"


def fit_gmm(
    X: np.ndarray,
    k: int,
    covariance_type: str = DEFAULT_COV_TYPE,
    n_init: int = 10,
) -> GaussianMixture:
    """Fit a k-component GMM."""
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=RANDOM_SEED,
    )
    gmm.fit(X)
    return gmm


def bic(gmm: GaussianMixture, X: np.ndarray) -> float:
    return gmm.bic(X)


def aic(gmm: GaussianMixture, X: np.ndarray) -> float:
    return gmm.aic(X)


def soft_assignments(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """Return n x K posterior responsibility matrix."""
    return gmm.predict_proba(X)


def sweep_k(
    X: np.ndarray,
    k_range: range,
    covariance_type: str = DEFAULT_COV_TYPE,
) -> pd.DataFrame:
    """Fit GMM for each k; return BIC, AIC, and log-likelihood."""
    records = []
    for k in k_range:
        gmm = fit_gmm(X, k, covariance_type=covariance_type)
        records.append({
            "k": k,
            "bic": bic(gmm, X),
            "aic": aic(gmm, X),
            "log_likelihood": gmm.lower_bound_,
            "converged": gmm.converged_,
        })
    return pd.DataFrame(records)
