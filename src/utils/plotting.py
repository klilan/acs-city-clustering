"""
plotting.py
Shared plotting utilities. All figures saved to results/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("results/figures")

STYLE = "seaborn-v0_8-whitegrid"
PALETTE = "tab10"


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")


def elbow_plot(
    wcss_df: pd.DataFrame, save: bool = True, out_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(wcss_df["k"], wcss_df["wcss"], marker="o")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("WCSS (Inertia)")
        ax.set_title("K-Means Elbow Plot")
    if save:
        _save(fig, out_dir, "elbow_plot.png")
    return fig


def bic_plot(
    bic_df: pd.DataFrame, save: bool = True, out_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bic_df["k"], bic_df["bic"], marker="s", label="BIC")
        ax.plot(bic_df["k"], bic_df["aic"], marker="^", linestyle="--", label="AIC")
        ax.set_xlabel("Number of Components (K)")
        ax.set_ylabel("Information Criterion")
        ax.set_title("GMM Model Selection")
        ax.legend()
    if save:
        _save(fig, out_dir, "bic_aic_plot.png")
    return fig


def silhouette_plot(
    sil_samples: np.ndarray,
    labels: np.ndarray,
    k: int,
    save: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    from matplotlib import cm
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        y_lower = 10
        for cluster_id in range(k):
            vals = np.sort(sil_samples[labels == cluster_id])
            size = vals.shape[0]
            y_upper = y_lower + size
            color = cm.tab10(cluster_id / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, str(cluster_id))
            y_lower = y_upper + 10
        ax.axvline(x=sil_samples.mean(), color="red", linestyle="--", label="Mean silhouette")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("City (by cluster)")
        ax.set_title(f"Silhouette Plot (K={k})")
        ax.legend()
    if save:
        _save(fig, out_dir, f"silhouette_k{k}.png")
    return fig


def cluster_profile_heatmap(
    summary: pd.DataFrame,
    feature_cols: list,
    save: bool = True,
    out_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    data = summary[feature_cols]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, max(4, len(summary) * 0.6)))
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("Cluster Centroid Profiles (standardized)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Cluster")
    if save:
        _save(fig, out_dir, "cluster_profiles.png")
    return fig


def stability_histogram(
    stability_df: pd.DataFrame, save: bool = True, out_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(stability_df["stability_rate"], bins=20, edgecolor="white")
        ax.axvline(x=0.95, color="red", linestyle="--", label="95% threshold")
        ax.set_xlabel("Stability Rate (fraction of 1,000 resamples)")
        ax.set_ylabel("Number of Cities")
        ax.set_title("MOE-Bootstrap Assignment Stability")
        ax.legend()
    if save:
        _save(fig, out_dir, "stability_histogram.png")
    return fig


def correlation_heatmap(
    corr: pd.DataFrame, save: bool = True, out_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.4,
            annot_kws={"size": 7},
        )
        ax.set_title("Feature Correlation Matrix (Pearson)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        fig.tight_layout()
    if save:
        _save(fig, out_dir, "correlation_matrix.png")
    return fig


def pca_scree_plot(
    explained_variance_ratio: np.ndarray, save: bool = True, out_dir: Path = FIGURES_DIR,
) -> plt.Figure:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        ax.plot(
            range(1, len(explained_variance_ratio) + 1),
            np.cumsum(explained_variance_ratio),
            marker="o",
            color="red",
            label="Cumulative variance",
        )
        ax.axhline(y=0.90, color="gray", linestyle="--", label="90% threshold")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Scree Plot")
        ax.legend()
    if save:
        _save(fig, out_dir, "pca_scree.png")
    return fig
