"""
reproduce_results.py
End-to-end pipeline: preprocess -> cluster -> evaluate -> stability -> figures/tables.

Usage:
    python reproduce_results.py

Requires:
    - data/raw/acs_places_raw.csv  (run src/data/fetch_acs.py first)
    - All packages in requirements.txt

Random seed: 42 (fixed globally below)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Fix global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from src.data.preprocess import main as run_preprocessing
from src.models.kmeans import fit_kmeans, sweep_k as kmeans_sweep
from src.models.gmm import fit_gmm, sweep_k as gmm_sweep, soft_assignments
from src.evaluation.metrics import (
    silhouette,
    silhouette_per_city,
    elbow_table,
    summarize_clusters,
)
from src.evaluation.stability import (
    moe_bootstrap_gmm,
    moe_bootstrap_kmeans,
    stability_summary,
)
from src.utils.plotting import (
    elbow_plot,
    bic_plot,
    silhouette_plot,
    cluster_profile_heatmap,
    stability_histogram,
    pca_scree_plot,
)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE = range(2, 11)
K_BOOTSTRAP = 1000


def load_processed():
    X_scaled = pd.read_csv(PROCESSED_DIR / "X_scaled.csv")
    X_moe = pd.read_csv(PROCESSED_DIR / "X_moe_scaled.csv")
    X_raw = pd.read_csv(PROCESSED_DIR / "X_raw_rates.csv")
    city_index = pd.read_csv(PROCESSED_DIR / "city_index.csv")
    return X_scaled, X_moe, X_raw, city_index


def main():
    # -----------------------------------------------------------------------
    # 1. Preprocessing
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    if not (PROCESSED_DIR / "X_scaled.csv").exists():
        run_preprocessing()
    else:
        print("  Processed data found — skipping preprocessing.")

    X_scaled, X_moe_scaled, X_raw, city_index = load_processed()
    X = X_scaled.to_numpy()
    X_moe = X_moe_scaled.fillna(0).to_numpy()   # NaN MOEs treated as 0 (no perturbation)
    feature_cols = X_scaled.columns.tolist()

    print(f"  Cities: {len(city_index)}  |  Features: {X.shape[1]}")

    # PCA scree (diagnostics only — clustering uses full feature matrix)
    pca_loadings = pd.read_csv(PROCESSED_DIR / "pca_loadings.csv", index_col=0)
    # Reconstruct explained variance from loadings norms (approximate)
    print("  Saving PCA scree plot...")
    # Use sklearn PCA directly for explained variance
    from sklearn.decomposition import PCA
    pca = PCA(random_state=RANDOM_SEED).fit(X)
    pca_scree_plot(pca.explained_variance_ratio_)

    # -----------------------------------------------------------------------
    # 2. K selection — sweep K for both models
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: K Selection")
    print("=" * 60)

    print("  Fitting K-means over K =", list(K_RANGE), "...")
    km_sweep = kmeans_sweep(X, K_RANGE)
    elbow_df = elbow_table(km_sweep)
    elbow_df.to_csv(TABLES_DIR / "kmeans_elbow.csv", index=False)
    elbow_plot(km_sweep)
    print(elbow_df.to_string(index=False))

    print("\n  Fitting GMM over K =", list(K_RANGE), "...")
    gmm_sweep_df = gmm_sweep(X, K_RANGE)
    gmm_sweep_df.to_csv(TABLES_DIR / "gmm_bic_aic.csv", index=False)
    bic_plot(gmm_sweep_df)
    print(gmm_sweep_df.to_string(index=False))

    # Select K: GMM BIC minimum
    k_best_gmm = int(gmm_sweep_df.loc[gmm_sweep_df["bic"].idxmin(), "k"])
    k_best_km = int(km_sweep.loc[km_sweep["wcss"].diff().diff().idxmin(), "k"])  # elbow heuristic
    print(f"\n  Selected K (GMM/BIC): {k_best_gmm}")
    print(f"  Selected K (KMeans/elbow): {k_best_km}")

    # -----------------------------------------------------------------------
    # 3. Fit final models
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Fit Final Models")
    print("=" * 60)

    k = k_best_gmm   # use GMM-selected K as primary

    print(f"  Fitting GMM (K={k}, full covariance)...")
    gmm = fit_gmm(X, k)
    gmm_labels = gmm.predict(X)
    gmm_proba = soft_assignments(gmm, X)

    print(f"  Fitting K-means (K={k})...")
    km = fit_kmeans(X, k)
    km_labels = km.predict(X)

    # Silhouette scores
    sil_gmm = silhouette(X, gmm_labels)
    sil_km = silhouette(X, km_labels)
    print(f"  Silhouette (GMM): {sil_gmm:.4f}")
    print(f"  Silhouette (KMeans): {sil_km:.4f}")

    sil_samples_gmm = silhouette_per_city(X, gmm_labels)
    silhouette_plot(sil_samples_gmm, gmm_labels, k)

    # -----------------------------------------------------------------------
    # 4. Cluster profiles
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Cluster Profiles")
    print("=" * 60)

    profile = summarize_clusters(X_raw, gmm_labels, city_index)
    profile.to_csv(TABLES_DIR / "cluster_profiles.csv")
    cluster_profile_heatmap(
        summarize_clusters(X_scaled, gmm_labels, city_index),
        feature_cols,
    )

    print(profile.to_string())

    # Save city assignments
    assignments = city_index.copy()
    assignments["cluster_gmm"] = gmm_labels
    assignments["cluster_kmeans"] = km_labels
    assignments[["GMM_" + f"p_k{i}" for i in range(k)]] = gmm_proba
    assignments.to_csv(TABLES_DIR / "city_assignments.csv", index=False)

    # -----------------------------------------------------------------------
    # 5. Stability analysis (MOE bootstrap)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 5: MOE Bootstrap Stability (B={K_BOOTSTRAP})")
    print("=" * 60)

    print("  Running GMM stability bootstrap...")
    stab_gmm = moe_bootstrap_gmm(X, X_moe, gmm, B=K_BOOTSTRAP)
    stab_gmm["city"] = city_index["NAME"].values
    stab_gmm.to_csv(TABLES_DIR / "stability_gmm.csv", index=False)
    stability_histogram(stab_gmm)
    gmm_summary = stability_summary(stab_gmm)
    print("  GMM stability summary:", gmm_summary)

    print("  Running K-means stability bootstrap...")
    stab_km = moe_bootstrap_kmeans(X, X_moe, km, B=K_BOOTSTRAP)
    stab_km["city"] = city_index["NAME"].values
    stab_km.to_csv(TABLES_DIR / "stability_kmeans.csv", index=False)
    km_summary = stability_summary(stab_km)
    print("  KMeans stability summary:", km_summary)

    # -----------------------------------------------------------------------
    # 6. Sensitivity analysis — covariance type and feature subsets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Sensitivity Analysis")
    print("=" * 60)

    sensitivity_records = []
    for cov_type in ["full", "tied", "diag"]:
        for k_test in [k - 1, k, k + 1]:
            g = fit_gmm(X, k_test, covariance_type=cov_type)
            labs = g.predict(X)
            if len(np.unique(labs)) < 2:
                continue
            sil = silhouette(X, labs)
            sensitivity_records.append({
                "cov_type": cov_type,
                "k": k_test,
                "bic": g.bic(X),
                "silhouette": sil,
            })

    sens_df = pd.DataFrame(sensitivity_records)
    sens_df.to_csv(TABLES_DIR / "sensitivity_cov_k.csv", index=False)
    print(sens_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE — All figures saved to results/figures/")
    print("       All tables saved to results/tables/")
    print("=" * 60)
    print(f"  Final model: GMM, K={k}, full covariance")
    print(f"  Silhouette:  {sil_gmm:.4f}")
    print(f"  Mean stability rate (GMM): {gmm_summary['mean_stability']:.3f}")
    print(f"  % cities stable at 95%: {gmm_summary['pct_stable']:.1%}")


if __name__ == "__main__":
    main()
