"""
stability.py
MOE-bootstrap stability analysis.

For each city i and each resample b in 1..B:
    x_i^(b) ~ Uniform( x_i - moe_i, x_i + moe_i )   (element-wise, independent)

A city is "stable" if its cluster assignment matches the nominal assignment in >= 95% of resamples.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


RANDOM_SEED = 42


def moe_bootstrap_gmm(
    X: np.ndarray,
    X_moe: np.ndarray,
    gmm: GaussianMixture,
    B: int = 1000,
) -> pd.DataFrame:
    """
    Bootstrap stability for a fitted GMM.

    For each resample, perturb X within MOE bounds and reassign cities using
    the fitted gmm parameters (predict on perturbed X).

    Returns DataFrame with columns: city_idx, nominal_label, stability_rate
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = X.shape[0]
    nominal_labels = gmm.predict(X)
    assignment_counts = np.zeros((n, gmm.n_components), dtype=int)

    for _ in tqdm(range(B), desc="MOE bootstrap (GMM)"):
        noise = rng.uniform(-1, 1, size=X.shape)
        X_perturbed = X + noise * X_moe
        labels_b = gmm.predict(X_perturbed)
        for i, lab in enumerate(labels_b):
            assignment_counts[i, lab] += 1

    stability_rate = assignment_counts[np.arange(n), nominal_labels] / B

    return pd.DataFrame({
        "city_idx": np.arange(n),
        "nominal_label": nominal_labels,
        "stability_rate": stability_rate,
    })


def moe_bootstrap_kmeans(
    X: np.ndarray,
    X_moe: np.ndarray,
    km: KMeans,
    B: int = 1000,
) -> pd.DataFrame:
    """Bootstrap stability for a fitted KMeans model."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = X.shape[0]
    nominal_labels = km.predict(X)
    assignment_counts = np.zeros((n, km.n_clusters), dtype=int)

    for _ in tqdm(range(B), desc="MOE bootstrap (KMeans)"):
        noise = rng.uniform(-1, 1, size=X.shape)
        X_perturbed = X + noise * X_moe
        labels_b = km.predict(X_perturbed)
        for i, lab in enumerate(labels_b):
            assignment_counts[i, lab] += 1

    stability_rate = assignment_counts[np.arange(n), nominal_labels] / B

    return pd.DataFrame({
        "city_idx": np.arange(n),
        "nominal_label": nominal_labels,
        "stability_rate": stability_rate,
    })


def stability_summary(stability_df: pd.DataFrame, threshold: float = 0.95) -> dict:
    """Aggregate stability statistics."""
    stable = stability_df["stability_rate"] >= threshold
    return {
        "mean_stability": stability_df["stability_rate"].mean(),
        "median_stability": stability_df["stability_rate"].median(),
        "pct_stable": stable.mean(),
        "n_unstable": (~stable).sum(),
        "threshold": threshold,
    }
