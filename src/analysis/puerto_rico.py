"""
puerto_rico.py
Separate clustering analysis for Puerto Rico (FIPS state=72) vs. continental US.

Rationale:
  - PR uses a different ACS sampling design (smaller populations, distinct collection
    patterns), and its 1-year estimates carry larger MOEs relative to city size.
  - Running a joint K-selection on 546 cities conflates two structurally different
    populations; this module re-runs the full pipeline on each subset independently.

NOTE ON SAMPLE SIZE:
  Only 6 PR cities meet the 65k population threshold, so PR clustering is exploratory.
  K is capped at PR_K_MAX (default 3). Stability bootstrap is skipped for PR because
  n=6 makes resampling statistics unreliable.

Outputs (written relative to project root):
  results/tables/pr_analysis/          -- PR tables
  results/tables/states_dc_analysis/   -- 50 states + DC tables
  results/figures/pr_analysis/         -- PR figures
  results/figures/states_dc_analysis/  -- 50 states + DC figures

Usage:
  python -m src.analysis.puerto_rico
  # or from reproduce_results.py: from src.analysis.puerto_rico import main as run_pr_analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from src.models.kmeans import fit_kmeans, sweep_k as kmeans_sweep
from src.models.gmm import fit_gmm, sweep_k as gmm_sweep, soft_assignments
from src.evaluation.metrics import (
    silhouette,
    silhouette_per_city,
    elbow_table,
    summarize_clusters,
)
from src.evaluation.stability import moe_bootstrap_gmm, moe_bootstrap_kmeans, stability_summary

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PR_FIPS = 72
STATES_DC_K_RANGE = range(2, 11)
PR_K_MAX = 3  # hard cap: n=6 cannot support more than 3 meaningful clusters
PR_K_RANGE = range(2, PR_K_MAX + 1)
BOOTSTRAP_B = 1000

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

STYLE = "seaborn-v0_8-whitegrid"


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_processed(processed_dir: Path = PROCESSED_DIR) -> tuple:
    X_scaled = pd.read_csv(processed_dir / "X_scaled.csv")
    X_moe = pd.read_csv(processed_dir / "X_moe_scaled.csv")
    X_raw = pd.read_csv(processed_dir / "X_raw_rates.csv")
    city_index = pd.read_csv(processed_dir / "city_index.csv")
    return X_scaled, X_moe, X_raw, city_index


def split_subsets(
    city_index: pd.DataFrame,
    X_scaled: pd.DataFrame,
    X_moe: pd.DataFrame,
    X_raw: pd.DataFrame,
) -> dict:
    """
    Return a dict with keys 'pr' and 'continental', each holding
    (X_scaled, X_moe, X_raw, city_index) sliced to that subset.
    """
    pr_mask = city_index["state"] == PR_FIPS
    cont_mask = ~pr_mask

    subsets = {}
    for label, mask in [("pr", pr_mask), ("states_dc", cont_mask)]:
        idx = mask[mask].index
        subsets[label] = {
            "X_scaled": X_scaled.loc[idx].reset_index(drop=True),
            "X_moe": X_moe.loc[idx].reset_index(drop=True),
            "X_raw": X_raw.loc[idx].reset_index(drop=True),
            "city_index": city_index.loc[idx].reset_index(drop=True),
        }
    return subsets


# ---------------------------------------------------------------------------
# MOE comparison
# ---------------------------------------------------------------------------

def moe_comparison(
    X_moe_pr: pd.DataFrame,
    X_moe_cont: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare median (scaled) MOE per feature between PR and continental cities.
    Returns a DataFrame indexed by feature with columns: pr_median_moe, cont_median_moe, ratio.
    Ratio > 1 means PR has larger uncertainty for that feature.
    """
    pr_med = X_moe_pr.median()
    cont_med = X_moe_cont.median()
    df = pd.DataFrame({"pr_median_moe": pr_med, "cont_median_moe": cont_med})
    df["ratio"] = df["pr_median_moe"] / df["cont_median_moe"]
    df.index = df.index.str.replace("_moe", "")
    return df.sort_values("ratio", ascending=False)


# ---------------------------------------------------------------------------
# Figures (saved to subset-specific dirs to avoid overwriting main outputs)
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _elbow_plot(wcss_df: pd.DataFrame, out_dir: Path, label: str) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(wcss_df["k"], wcss_df["wcss"], marker="o")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("WCSS (Inertia)")
        ax.set_title(f"K-Means Elbow Plot — {label}")
    _save(fig, out_dir / "elbow_plot.png")


def _bic_plot(bic_df: pd.DataFrame, out_dir: Path, label: str) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bic_df["k"], bic_df["bic"], marker="s", label="BIC")
        ax.plot(bic_df["k"], bic_df["aic"], marker="^", linestyle="--", label="AIC")
        ax.set_xlabel("Number of Components (K)")
        ax.set_ylabel("Information Criterion")
        ax.set_title(f"GMM Model Selection — {label}")
        ax.legend()
    _save(fig, out_dir / "bic_aic_plot.png")


def _silhouette_plot(
    sil_samples: np.ndarray,
    labels: np.ndarray,
    k: int,
    out_dir: Path,
    label: str,
) -> None:
    from matplotlib import cm
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        y_lower = 10
        for cluster_id in range(k):
            vals = np.sort(sil_samples[labels == cluster_id])
            size = vals.shape[0]
            if size == 0:
                continue
            y_upper = y_lower + size
            color = cm.tab10(cluster_id / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, str(cluster_id))
            y_lower = y_upper + 10
        ax.axvline(x=sil_samples.mean(), color="red", linestyle="--", label="Mean silhouette")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("City (by cluster)")
        ax.set_title(f"Silhouette Plot (K={k}) — {label}")
        ax.legend()
    _save(fig, out_dir / f"silhouette_k{k}.png")


def _stability_histogram(stability_df: pd.DataFrame, out_dir: Path, label: str) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(stability_df["stability_rate"], bins=20, edgecolor="white")
        ax.axvline(x=0.95, color="red", linestyle="--", label="95% threshold")
        ax.set_xlabel("Stability Rate (fraction of resamples)")
        ax.set_ylabel("Number of Cities")
        ax.set_title(f"MOE-Bootstrap Stability — {label}")
        ax.legend()
    _save(fig, out_dir / "stability_histogram.png")


def _moe_comparison_plot(moe_df: pd.DataFrame, out_dir: Path) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(moe_df))
        width = 0.35
        ax.bar(x - width / 2, moe_df["pr_median_moe"], width, label="Puerto Rico")
        ax.bar(x + width / 2, moe_df["cont_median_moe"], width, label="Continental US")
        ax.set_xticks(x)
        ax.set_xticklabels(moe_df.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Median scaled MOE")
        ax.set_title("MOE Magnitude: Puerto Rico vs. Continental US")
        ax.legend()
        fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "moe_comparison.png")


# ---------------------------------------------------------------------------
# Core analysis runner
# ---------------------------------------------------------------------------

def run_subset_analysis(
    X_scaled: pd.DataFrame,
    X_moe: pd.DataFrame,
    X_raw: pd.DataFrame,
    city_index: pd.DataFrame,
    label: str,
    k_range: range,
    B: int = BOOTSTRAP_B,
    run_stability: bool = True,
) -> dict:
    """
    Run the full clustering pipeline on one subset (PR or continental).

    Returns a summary dict with selected K, silhouette scores, and stability stats.
    Saves all tables and figures under results/{tables,figures}/{label}_analysis/.
    """
    tables_dir = RESULTS_DIR / "tables" / f"{label}_analysis"
    figures_dir = RESULTS_DIR / "figures" / f"{label}_analysis"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X = X_scaled.to_numpy()
    X_moe_arr = X_moe.fillna(0).to_numpy()
    feature_cols = X_scaled.columns.tolist()
    n = len(city_index)

    print(f"\n{'=' * 60}")
    print(f"  Subset: {label.upper()}  |  n={n}  |  K range: {list(k_range)}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # K selection
    # ------------------------------------------------------------------
    print("  K-means sweep...")
    km_sweep_df = kmeans_sweep(X, k_range)
    elbow_df = elbow_table(km_sweep_df)
    elbow_df.to_csv(tables_dir / "kmeans_elbow.csv", index=False)
    _elbow_plot(km_sweep_df, figures_dir, label)

    print("  GMM sweep...")
    gmm_sweep_df = gmm_sweep(X, k_range)
    gmm_sweep_df.to_csv(tables_dir / "gmm_bic_aic.csv", index=False)
    _bic_plot(gmm_sweep_df, figures_dir, label)

    k_best_gmm = int(gmm_sweep_df.loc[gmm_sweep_df["bic"].idxmin(), "k"])
    k_best_km = int(elbow_df.loc[elbow_df["delta2_wcss"].idxmax(), "k"])
    print(f"  Selected K (GMM/BIC): {k_best_gmm}")
    print(f"  Selected K (KMeans/elbow): {k_best_km}")

    # ------------------------------------------------------------------
    # Fit final models at GMM-selected K
    # ------------------------------------------------------------------
    k = k_best_gmm

    print(f"  Fitting GMM (K={k})...")
    gmm = fit_gmm(X, k)
    gmm_labels = gmm.predict(X)
    gmm_proba = soft_assignments(gmm, X)

    print(f"  Fitting K-means (K={k})...")
    km = fit_kmeans(X, k)
    km_labels = km.predict(X)

    sil_gmm = silhouette(X, gmm_labels) if len(np.unique(gmm_labels)) >= 2 else float("nan")
    sil_km = silhouette(X, km_labels) if len(np.unique(km_labels)) >= 2 else float("nan")
    print(f"  Silhouette (GMM):    {sil_gmm:.4f}")
    print(f"  Silhouette (KMeans): {sil_km:.4f}")

    if len(np.unique(gmm_labels)) >= 2:
        sil_samples = silhouette_per_city(X, gmm_labels)
        _silhouette_plot(sil_samples, gmm_labels, k, figures_dir, label)

    # ------------------------------------------------------------------
    # Cluster profiles
    # ------------------------------------------------------------------
    profile = summarize_clusters(X_raw, gmm_labels, city_index)
    profile.to_csv(tables_dir / "cluster_profiles.csv")

    assignments = city_index.copy()
    assignments["cluster_gmm"] = gmm_labels
    assignments["cluster_kmeans"] = km_labels
    proba_cols = [f"GMM_p_k{i}" for i in range(k)]
    assignments[proba_cols] = gmm_proba
    assignments.to_csv(tables_dir / "city_assignments.csv", index=False)

    # ------------------------------------------------------------------
    # Stability bootstrap (skipped for very small subsets)
    # ------------------------------------------------------------------
    gmm_stab_summary = {}
    km_stab_summary = {}

    if not run_stability:
        print(f"  Stability bootstrap SKIPPED (n={n} too small for reliable resampling).")
    else:
        print(f"  MOE bootstrap stability (B={B})...")
        stab_gmm = moe_bootstrap_gmm(X, X_moe_arr, gmm, B=B)
        stab_gmm["city"] = city_index["NAME"].values
        stab_gmm.to_csv(tables_dir / "stability_gmm.csv", index=False)
        _stability_histogram(stab_gmm, figures_dir, label)
        gmm_stab_summary = stability_summary(stab_gmm)
        print(f"  GMM stability: {gmm_stab_summary}")

        stab_km = moe_bootstrap_kmeans(X, X_moe_arr, km, B=B)
        stab_km["city"] = city_index["NAME"].values
        stab_km.to_csv(tables_dir / "stability_kmeans.csv", index=False)
        km_stab_summary = stability_summary(stab_km)
        print(f"  KMeans stability: {km_stab_summary}")

    return {
        "label": label,
        "n": n,
        "k_gmm": k_best_gmm,
        "k_km": k_best_km,
        "silhouette_gmm": sil_gmm,
        "silhouette_km": sil_km,
        "gmm_stability": gmm_stab_summary,
        "km_stability": km_stab_summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading processed data...")
    X_scaled, X_moe, X_raw, city_index = load_processed()
    print(f"  Total cities: {len(city_index)}")

    subsets = split_subsets(city_index, X_scaled, X_moe, X_raw)

    pr_n = len(subsets["pr"]["city_index"])
    cont_n = len(subsets["continental"]["city_index"])
    print(f"  Puerto Rico: {pr_n} cities")
    print(f"  Continental: {cont_n} cities")

    # ------------------------------------------------------------------
    # MOE comparison
    # ------------------------------------------------------------------
    moe_df = moe_comparison(subsets["pr"]["X_moe"], subsets["continental"]["X_moe"])
    out_tables = RESULTS_DIR / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    moe_df.to_csv(out_tables / "pr_moe_comparison.csv")
    _moe_comparison_plot(moe_df, RESULTS_DIR / "figures")
    print("\nMOE comparison (sorted by PR/continental ratio):")
    print(moe_df.to_string())

    # ------------------------------------------------------------------
    # 50 states + DC analysis (full K range, with stability)
    # ------------------------------------------------------------------
    states_dc = subsets["states_dc"]
    cont_summary = run_subset_analysis(
        states_dc["X_scaled"], states_dc["X_moe"], states_dc["X_raw"], states_dc["city_index"],
        label="states_dc",
        k_range=STATES_DC_K_RANGE,
        B=BOOTSTRAP_B,
        run_stability=True,
    )

    # ------------------------------------------------------------------
    # Puerto Rico analysis (capped K range, no stability bootstrap)
    # ------------------------------------------------------------------
    pr = subsets["pr"]
    pr_summary = run_subset_analysis(
        pr["X_scaled"], pr["X_moe"], pr["X_raw"], pr["city_index"],
        label="pr",
        k_range=PR_K_RANGE,
        B=BOOTSTRAP_B,
        run_stability=False,
    )

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rows = []
    for s in [cont_summary, pr_summary]:
        row = {
            "subset": s["label"],
            "n": s["n"],
            "k_selected_gmm": s["k_gmm"],
            "k_selected_kmeans": s["k_km"],
            "silhouette_gmm": round(s["silhouette_gmm"], 4),
            "silhouette_kmeans": round(s["silhouette_km"], 4),
        }
        if s["gmm_stability"]:
            row["gmm_mean_stability"] = round(s["gmm_stability"]["mean_stability"], 3)
            row["gmm_pct_stable_95"] = round(s["gmm_stability"]["pct_stable"], 3)
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("subset")
    print(summary_df.to_string())
    summary_df.to_csv(out_tables / "pr_continental_comparison.csv")

    print("\nAll outputs written to results/tables/ and results/figures/")


if __name__ == "__main__":
    main()
