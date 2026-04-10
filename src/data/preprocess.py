"""
preprocess.py
Clean raw ACS data, engineer rate features, z-score standardize, and optionally apply PCA.

Input:  data/raw/acs_places_raw.csv
Output: data/processed/X_scaled.csv        (standardized feature matrix)
        data/processed/X_moe_scaled.csv     (MOE matrix, same scale as X)
        data/processed/city_index.csv       (NAME, state, place identifiers)
        data/processed/pca_loadings.csv     (if PCA applied)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RANDOM_SEED = 42
POP_THRESHOLD = 65_000
PCA_VARIANCE_THRESHOLD = 0.90   # apply PCA only if top components explain < threshold of variance

RAW_PATH = Path("data/raw/acs_places_raw.csv")
PROCESSED_DIR = Path("data/processed")

FEATURE_COLS = [
    "homeownership_rate",
    "median_gross_rent",
    "rent_burden_rate",
    "vacancy_rate",
    "poverty_rate",
    "gini_index",
    "median_household_income",
    "childhood_poverty_rate",
    "labor_force_participation_rate",
    "youth_unemployment_rate",
    "hs_or_higher_rate",
    "bachelors_or_higher_rate",
]

MOE_COLS = [f"{c}_moe" for c in FEATURE_COLS]


def engineer_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to rates/proportions and propagate MOEs."""
    out = df.copy()

    # Homeownership rate
    out["homeownership_rate"] = out["owner_occupied"] / out["total_occupied"]
    out["homeownership_rate_moe"] = _ratio_moe(
        out["owner_occupied"], out["total_occupied"],
        out["owner_occupied_moe"], out["total_occupied_moe"],
    )

    # Rent burden rate (>= 30% income)
    out["rent_burden_rate"] = out["rent_burden_30plus"] / out["renter_total"]
    out["rent_burden_rate_moe"] = _ratio_moe(
        out["rent_burden_30plus"], out["renter_total"],
        out["rent_burden_30plus_moe"], out["renter_total_moe"],
    )

    # Vacancy rate
    out["vacancy_rate"] = out["vacant_units"] / out["total_units"]
    out["vacancy_rate_moe"] = _ratio_moe(
        out["vacant_units"], out["total_units"],
        out["vacant_units_moe"], out["total_units_moe"],
    )

    # Poverty rate
    out["poverty_rate"] = out["poverty_count"] / out["poverty_universe"]
    out["poverty_rate_moe"] = _ratio_moe(
        out["poverty_count"], out["poverty_universe"],
        out["poverty_count_moe"], out["poverty_universe_moe"],
    )

    # Gini index (already a rate)
    out["gini_index_moe"] = out.get("gini_index_moe", np.nan)

    # Childhood poverty rate
    out["childhood_poverty_rate"] = out["childhood_poverty_count_all"] / out["poverty_universe"]
    out["childhood_poverty_rate_moe"] = _ratio_moe(
        out["childhood_poverty_count_all"], out["poverty_universe"],
        out["childhood_poverty_count_all_moe"], out["poverty_universe_moe"],
    )

    # Labor force participation rate
    out["labor_force_participation_rate"] = out["labor_force"] / out["pop_16plus"]
    out["labor_force_participation_rate_moe"] = _ratio_moe(
        out["labor_force"], out["pop_16plus"],
        out["labor_force_moe"], out["pop_16plus_moe"],
    )

    # Youth unemployment rate (proxy: youth unemployed / labor force)
    out["youth_unemployed"] = out["youth_unemployed_m"].fillna(0) + out["youth_unemployed_f"].fillna(0)
    out["youth_unemployment_rate"] = out["youth_unemployed"] / out["labor_force"]
    out["youth_unemployment_rate_moe"] = np.nan  # simplified; composite MOE

    # HS or higher rate
    hs_cols = [
        "hs_diploma", "ged_or_alt", "some_college_1yr", "assoc_degree",
        "bachelors_degree", "masters_degree", "professional_degree", "doctoral_degree",
    ]
    out["hs_or_higher_count"] = out[hs_cols].sum(axis=1)
    out["hs_or_higher_rate"] = out["hs_or_higher_count"] / out["edu_universe"]
    out["hs_or_higher_rate_moe"] = np.nan  # composite

    # Bachelor's or higher rate
    ba_cols = ["bachelors_degree", "masters_degree", "professional_degree", "doctoral_degree"]
    out["ba_or_higher_count"] = out[ba_cols].sum(axis=1)
    out["bachelors_or_higher_rate"] = out["ba_or_higher_count"] / out["edu_universe"]
    out["bachelors_or_higher_rate_moe"] = np.nan  # composite

    return out


def _ratio_moe(num: pd.Series, den: pd.Series, num_moe: pd.Series, den_moe: pd.Series) -> pd.Series:
    """Approximate MOE for a ratio p = num/den using Census recommended formula."""
    p = num / den
    # Census formula: MOE(p) = (1/den) * sqrt(num_moe^2 - p^2 * den_moe^2)
    # Use sum formula when radicand is negative (rare edge case)
    radicand = num_moe**2 - p**2 * den_moe**2
    radicand = radicand.clip(lower=0)   # fallback to sum formula avoids complex numbers
    return (1 / den) * np.sqrt(radicand)


def drop_missing(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=feature_cols)
    print(f"  Dropped {before - len(df)} rows with missing features (kept {len(df)})")
    return df


def standardize(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


def maybe_apply_pca(X_scaled: pd.DataFrame, threshold: float = PCA_VARIANCE_THRESHOLD):
    """Apply PCA if top 2 components explain < threshold of variance, for potential dim reduction."""
    pca_full = PCA(random_state=RANDOM_SEED)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = int(np.searchsorted(cumvar, threshold)) + 1
    print(f"  PCA: {n_components_90} components explain >= {threshold*100:.0f}% of variance")

    loadings = pd.DataFrame(
        pca_full.components_.T,
        index=X_scaled.columns,
        columns=[f"PC{i+1}" for i in range(pca_full.n_components_)],
    )
    return pca_full, loadings, n_components_90


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw ACS data...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Loaded {len(df)} rows")

    # NOTE: Population filter — requires joining B01003_001E total population
    # If you fetched it in fetch_acs.py, filter here:
    if "total_population" in df.columns:
        df = df[df["total_population"] >= POP_THRESHOLD]
        print(f"  After population filter (>={POP_THRESHOLD}): {len(df)} rows")

    print("Engineering rate features...")
    df = engineer_rates(df)

    print("Dropping rows with missing features...")
    df = drop_missing(df, FEATURE_COLS)

    # Save city identifiers
    city_index = df[["NAME", "state", "place"]].reset_index(drop=True)
    city_index.to_csv(PROCESSED_DIR / "city_index.csv", index=False)

    X = df[FEATURE_COLS].reset_index(drop=True)
    X_moe = df[MOE_COLS].reset_index(drop=True)

    # Save unscaled for reference
    X.to_csv(PROCESSED_DIR / "X_raw_rates.csv", index=False)

    print("Standardizing features...")
    X_scaled, scaler = standardize(X)
    X_scaled.to_csv(PROCESSED_DIR / "X_scaled.csv", index=False)

    # Scale MOEs by same scaler (divide by std, no mean shift — MOEs are already differences)
    X_moe_scaled = X_moe / scaler.scale_
    X_moe_scaled.columns = MOE_COLS
    X_moe_scaled.to_csv(PROCESSED_DIR / "X_moe_scaled.csv", index=False)

    print("Running PCA for diagnostics...")
    pca, loadings, n_90 = maybe_apply_pca(X_scaled)
    loadings.to_csv(PROCESSED_DIR / "pca_loadings.csv")

    print(f"\nPreprocessing complete. Files written to {PROCESSED_DIR}/")
    print(f"  Feature matrix shape: {X_scaled.shape}")


if __name__ == "__main__":
    main()
