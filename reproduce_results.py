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
    correlation_heatmap,
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

    # Multicollinearity diagnostics
    print("  Saving correlation matrix...")
    corr = pd.read_csv(PROCESSED_DIR / "correlation_matrix.csv", index_col=0)
    correlation_heatmap(corr)
    high_corr = [
        (c1, c2, corr.loc[c1, c2])
        for i, c1 in enumerate(corr.columns)
        for c2 in corr.columns[i + 1:]
        if abs(corr.loc[c1, c2]) >= 0.8
    ]
    if high_corr:
        print(f"  High correlations (|r| >= 0.8):")
        for c1, c2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"    {c1} × {c2}: {r:.3f}")
    else:
        print("  No feature pairs with |r| >= 0.8")

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
    k_best_km = int(km_sweep.loc[km_sweep["wcss"].diff().diff().idxmax(), "k"])  # elbow heuristic
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

    # Feature subset sensitivity — drop one policy domain at a time
    print("\n  Feature subset sensitivity (leave-one-domain-out):")
    DOMAIN_FEATURES = {
        "housing":           ["homeownership_rate", "median_gross_rent", "rent_burden_rate", "vacancy_rate"],
        "household_finance": ["poverty_rate", "gini_index", "median_household_income", "childhood_poverty_rate"],
        "economic_health":   ["labor_force_participation_rate", "youth_unemployment_rate"],
        "education":         ["hs_or_higher_rate", "bachelors_or_higher_rate"],
    }

    subset_records = []
    for dropped_domain, dropped_cols in DOMAIN_FEATURES.items():
        keep_idx = [i for i, c in enumerate(feature_cols) if c not in dropped_cols]
        X_sub = X[:, keep_idx]
        g_sub = fit_gmm(X_sub, k)
        labs_sub = g_sub.predict(X_sub)
        if len(np.unique(labs_sub)) < 2:
            continue
        sil_sub = silhouette(X_sub, labs_sub)
        # Agreement with full-feature GMM labels
        agreement = np.mean(labs_sub == gmm_labels)
        subset_records.append({
            "dropped_domain": dropped_domain,
            "n_features": len(keep_idx),
            "silhouette": round(sil_sub, 4),
            "label_agreement_with_full": round(agreement, 4),
        })
        print(f"    drop {dropped_domain}: sil={sil_sub:.4f}, agreement={agreement:.3f}")

    subset_df = pd.DataFrame(subset_records)
    subset_df.to_csv(TABLES_DIR / "sensitivity_feature_subsets.csv", index=False)

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
