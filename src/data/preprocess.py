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
from propagate_se import moe_to_se, wrap_ufloat, unwrap_ufloat, safe_divide_elements

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


def _ratio_moe_se(
    num: pd.Series, den: pd.Series,
    num_moe: pd.Series, den_moe: pd.Series,
) -> pd.Series:
    """
    Propagate MOE for a ratio p = num/den using propagate_se.

    Converts ACS 90% MOEs to SEs, wraps as ufloats, divides element-wise
    (using safe_divide_elements to avoid spurious correlation when num is a
    subset of den), then unwraps and converts the propagated SE back to a
    90% MOE.
    """
    num_se = moe_to_se(num_moe)   # z=1.645 (ACS default)
    den_se = moe_to_se(den_moe)

    import uncertainties as uc
    result_moes = []
    for n_val, d_val, n_se, d_se in zip(num, den, num_se, den_se):
        if pd.isna(n_val) or pd.isna(d_val) or d_val == 0:
            result_moes.append(np.nan)
            continue
        n_uf = uc.ufloat(n_val, max(n_se, 0) if not pd.isna(n_se) else 0)
        d_uf = uc.ufloat(d_val, max(d_se, 0) if not pd.isna(d_se) else 0)
        ratio = safe_divide_elements(n_uf, d_uf)
        if isinstance(ratio, float):
            result_moes.append(np.nan)
        else:
            result_moes.append(ratio.std_dev * 1.645)  # SE → 90% MOE

    return pd.Series(result_moes, index=num.index)


def engineer_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to rates/proportions and propagate MOEs via propagate_se."""
    out = df.copy()

    # Homeownership rate
    out["homeownership_rate"] = out["owner_occupied"] / out["total_occupied"]
    out["homeownership_rate_moe"] = _ratio_moe_se(
        out["owner_occupied"], out["total_occupied"],
        out["owner_occupied_moe"], out["total_occupied_moe"],
    )

    # Rent burden rate (>= 30% income)
    out["rent_burden_rate"] = out["rent_burden_30plus"] / out["renter_total"]
    out["rent_burden_rate_moe"] = _ratio_moe_se(
        out["rent_burden_30plus"], out["renter_total"],
        out["rent_burden_30plus_moe"], out["renter_total_moe"],
    )

    # Vacancy rate
    out["vacancy_rate"] = out["vacant_units"] / out["total_units"]
    out["vacancy_rate_moe"] = _ratio_moe_se(
        out["vacant_units"], out["total_units"],
        out["vacant_units_moe"], out["total_units_moe"],
    )

    # Poverty rate
    out["poverty_rate"] = out["poverty_count"] / out["poverty_universe"]
    out["poverty_rate_moe"] = _ratio_moe_se(
        out["poverty_count"], out["poverty_universe"],
        out["poverty_count_moe"], out["poverty_universe_moe"],
    )

    # Gini index (already a rate; MOE fetched directly)
    out["gini_index_moe"] = out.get("gini_index_moe", np.nan)

    # Childhood poverty rate
    out["childhood_poverty_rate"] = out["childhood_poverty_count_all"] / out["poverty_universe"]
    out["childhood_poverty_rate_moe"] = _ratio_moe_se(
        out["childhood_poverty_count_all"], out["poverty_universe"],
        out["childhood_poverty_count_all_moe"], out["poverty_universe_moe"],
    )

    # Labor force participation rate
    out["labor_force_participation_rate"] = out["labor_force"] / out["pop_16plus"]
    out["labor_force_participation_rate_moe"] = _ratio_moe_se(
        out["labor_force"], out["pop_16plus"],
        out["labor_force_moe"], out["pop_16plus_moe"],
    )

    # Youth unemployment rate (proxy: youth unemployed / labor force)
    # No MOE available for youth unemployed counts — treated as zero uncertainty (limitation)
    out["youth_unemployed"] = out["youth_unemployed_m"].fillna(0) + out["youth_unemployed_f"].fillna(0)
    out["youth_unemployment_rate"] = out["youth_unemployed"] / out["labor_force"]
    out["youth_unemployment_rate_moe"] = np.nan

    # HS or higher rate — composite count, no direct MOE available (limitation)
    hs_cols = [
        "hs_diploma", "ged_or_alt", "some_college_lt1yr", "some_college_1yr",
        "assoc_degree", "bachelors_degree", "masters_degree", "professional_degree", "doctoral_degree",
    ]
    out["hs_or_higher_count"] = out[hs_cols].sum(axis=1)
    out["hs_or_higher_rate"] = out["hs_or_higher_count"] / out["edu_universe"]
    out["hs_or_higher_rate_moe"] = np.nan

    # Bachelor's or higher rate — composite count, no direct MOE available (limitation)
    ba_cols = ["bachelors_degree", "masters_degree", "professional_degree", "doctoral_degree"]
    out["ba_or_higher_count"] = out[ba_cols].sum(axis=1)
    out["bachelors_or_higher_rate"] = out["ba_or_higher_count"] / out["edu_universe"]
    out["bachelors_or_higher_rate_moe"] = np.nan

    return out


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

    # Population filter applied in fetch_acs.py (places >= 65,000); log count here for confirmation.
    print(f"  Places after population filter: {len(df)}")

    print("Engineering rate features...")
    df = engineer_rates(df)

    print("Dropping rows with missing features...")
    df = drop_missing(df, FEATURE_COLS)

    # Save city identifiers
    city_index = df[["NAME", "state", "place"]].reset_index(drop=True)
    city_index.to_csv(PROCESSED_DIR / "city_index.csv", index=False)

    X = df[FEATURE_COLS].reset_index(drop=True)
    X_moe = df[MOE_COLS].reset_index(drop=True)

    # Report MOE coverage — features with no MOE are treated as zero uncertainty in stability analysis
    missing_moe = [col for col in MOE_COLS if X_moe[col].isna().all()]
    if missing_moe:
        print(f"  WARNING: {len(missing_moe)} feature(s) have no MOE data and will be treated as "
              f"zero uncertainty in stability analysis (understates instability for these features):")
        for col in missing_moe:
            print(f"    - {col.replace('_moe', '')}")

    # Save unscaled for reference
    X.to_csv(PROCESSED_DIR / "X_raw_rates.csv", index=False)

    print("Standardizing features...")
    X_scaled, scaler = standardize(X)
    X_scaled.to_csv(PROCESSED_DIR / "X_scaled.csv", index=False)

    # Scale MOEs by same scaler (divide by std, no mean shift — MOEs are already differences)
    X_moe_scaled = X_moe / scaler.scale_
    X_moe_scaled.columns = MOE_COLS
    X_moe_scaled.to_csv(PROCESSED_DIR / "X_moe_scaled.csv", index=False)

    print("Computing feature correlation matrix...")
    corr = X.corr()
    corr.to_csv(PROCESSED_DIR / "correlation_matrix.csv")

    print("Running PCA for diagnostics...")
    pca, loadings, n_90 = maybe_apply_pca(X_scaled)
    loadings.to_csv(PROCESSED_DIR / "pca_loadings.csv")

    print(f"\nPreprocessing complete. Files written to {PROCESSED_DIR}/")
    print(f"  Feature matrix shape: {X_scaled.shape}")


if __name__ == "__main__":
    main()
