"""
fetch_acs.py
Pull ACS 1-Year Estimates for U.S. census places with population >= 65,000.

Requires a free Census API key: https://api.census.gov/data/key_signup.html
Set environment variable: CENSUS_API_KEY=your_key_here

Output: data/raw/acs_places_raw.csv
"""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ACS 1-year survey year
ACS_YEAR = 2023

# Census variable codes -> human-readable names
# Each feature variable is paired with its MOE (_M suffix)
VARIABLES = {
    # Population (used for size filter; controlled total — no MOE published)
    "B01003_001E": "total_population",
    # Housing
    "B25064_001E": "median_gross_rent",
    "B25064_001M": "median_gross_rent_moe",
    "B25003_002E": "owner_occupied",          # numerator for homeownership rate
    "B25003_001E": "total_occupied",           # denominator
    "B25003_002M": "owner_occupied_moe",
    "B25003_001M": "total_occupied_moe",
    "B25070_010E": "rent_burden_30plus",       # renter cost burden >= 30%
    "B25070_001E": "renter_total",
    "B25070_010M": "rent_burden_30plus_moe",
    "B25070_001M": "renter_total_moe",
    "B25002_003E": "vacant_units",
    "B25002_001E": "total_units",
    "B25002_003M": "vacant_units_moe",
    "B25002_001M": "total_units_moe",
    # Household Finance
    "B17001_002E": "poverty_count",
    "B17001_001E": "poverty_universe",
    "B17001_002M": "poverty_count_moe",
    "B17001_001M": "poverty_universe_moe",
    "B19083_001E": "gini_index",
    "B19083_001M": "gini_index_moe",
    "B19013_001E": "median_household_income",
    "B19013_001M": "median_household_income_moe",
    "B17001B_002E": "childhood_poverty_count",  # children under 18 in poverty (Black alone, use B17001_004E for all)
    "B17001_004E": "childhood_poverty_count_all",
    "B17001_004M": "childhood_poverty_count_all_moe",
    # Economic Health
    "B23025_002E": "labor_force",
    "B23025_001E": "pop_16plus",
    "B23025_002M": "labor_force_moe",
    "B23025_001M": "pop_16plus_moe",
    "B23001_008E": "youth_unemployed_m",       # male 16-19 unemployed
    "B23001_094E": "youth_unemployed_f",       # female 16-19 unemployed
    # Education
    "B15003_017E": "hs_diploma",
    "B15003_018E": "ged_or_alt",
    "B15003_019E": "some_college_lt1yr",
    "B15003_020E": "some_college_1yr",
    "B15003_021E": "assoc_degree",
    "B15003_022E": "bachelors_degree",
    "B15003_023E": "masters_degree",
    "B15003_024E": "professional_degree",
    "B15003_025E": "doctoral_degree",
    "B15003_001E": "edu_universe",
}

RAW_OUTPUT = Path("data/raw/acs_places_raw.csv")


def fetch_acs(api_key: str, year: int = ACS_YEAR) -> pd.DataFrame:
    base_url = f"https://api.census.gov/data/{year}/acs/acs1"
    var_list = ",".join(["NAME"] + list(VARIABLES.keys()))

    params = {
        "get": var_list,
        "for": "place:*",
        "in": "state:*",
        "key": api_key,
    }

    print(f"Fetching ACS {year} 1-year estimates for all places...")
    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as e:
        raise ValueError(
            f"Census API returned non-JSON response (status {resp.status_code}).\n"
            f"Response text: {resp.text[:500]}"
        ) from e
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns=VARIABLES)

    # Convert numeric columns
    skip_cols = {"NAME", "state", "place"}
    for col in df.columns:
        if col not in skip_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Raw rows fetched: {len(df)}")
    return df


def filter_population(df: pd.DataFrame, pop_threshold: int = 65_000) -> pd.DataFrame:
    """Keep only places with population >= pop_threshold using ACS total population."""
    before = len(df)
    df = df[df["total_population"] >= pop_threshold].copy()
    print(f"  Population filter (>= {pop_threshold:,}): {before} → {len(df)} places")
    return df


def main():
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "CENSUS_API_KEY not set. Export it before running:\n"
            "  export CENSUS_API_KEY=your_key_here"
        )

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_acs(api_key)
    df = filter_population(df)
    df.to_csv(RAW_OUTPUT, index=False)
    print(f"Saved raw data to {RAW_OUTPUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
