"""
metrics.py
Cluster evaluation metrics: silhouette score, elbow/WCSS, BIC sweep.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Global silhouette score. Range [-1, 1]; higher is better cohesion/separation."""
    return silhouette_score(X, labels)


def silhouette_per_city(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return silhouette_samples(X, labels)


def elbow_table(wcss_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate WCSS sweep with second differences for elbow detection."""
    df = wcss_df.copy().sort_values("k").reset_index(drop=True)
    df["delta_wcss"] = df["wcss"].diff()
    df["delta2_wcss"] = df["delta_wcss"].diff()
    return df


def summarize_clusters(
    X: pd.DataFrame,
    labels: np.ndarray,
    city_index: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-cluster centroid profiles in original feature units."""
    df = X.copy()
    df["cluster"] = labels
    df["city"] = city_index["NAME"].values
    summary = df.groupby("cluster")[X.columns.tolist()].mean()
    summary["n_cities"] = df.groupby("cluster").size()
    return summary
