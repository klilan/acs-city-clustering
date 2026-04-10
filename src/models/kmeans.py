"""
kmeans.py
K-means clustering wrapper.

Loss function (WCSS):
    J(C, mu) = sum_k sum_{x_i in C_k} ||x_i - mu_k||^2
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


RANDOM_SEED = 42


def fit_kmeans(X: np.ndarray, k: int, n_init: int = 20) -> KMeans:
    """Fit K-means with k clusters. n_init restarts guard against local minima."""
    km = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_SEED)
    km.fit(X)
    return km


def wcss(km: KMeans) -> float:
    """Within-cluster sum of squares (inertia)."""
    return km.inertia_


def sweep_k(X: np.ndarray, k_range: range) -> pd.DataFrame:
    """Fit K-means for each k in k_range; return WCSS for elbow plot."""
    records = []
    for k in k_range:
        km = fit_kmeans(X, k)
        records.append({"k": k, "wcss": wcss(km)})
    return pd.DataFrame(records)
