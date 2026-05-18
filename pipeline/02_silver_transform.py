"""
Silver layer: normalize, standardize, and join Bronze tables.

Produces data/silver/silver_master.parquet — one row per (country_iso3, year)
with all signals joined and missing-data flags added.

Run after 01_bronze_ingest.py:  python 02_silver_transform.py
"""
import re
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False
    warnings.warn("pycountry not installed; ISO-3 lookups will be partial.")

BRONZE_DIR = Path(__file__).resolve().parent.parent / "data/bronze"
SILVER_DIR = Path(__file__).resolve().parent.parent / "data/silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)

TODAY = pd.Timestamp.today().normalize()


# ---------------------------------------------------------------------------
# ISO-3 normalization helpers
# ---------------------------------------------------------------------------

# Manual overrides for codes that pycountry gets wrong or that HDX uses differently
ISO3_OVERRIDES: dict[str, str] = {
    "XKX": "XKX",  # Kosovo (not in ISO 3166)
    "PSE": "PSE",  # Palestine
    "SSD": "SSD",  # South Sudan
    "COD": "COD",  # DRC
    "TMP": "TLS",  # Timor-Leste old code
    "ROM": "ROU",  # Romania old code
}

_name_to_iso3_cache: dict[str, str] = {}


def name_to_iso3(name: str) -> str | None:
    """Best-effort country name → ISO-3 code lookup."""
    if not HAS_PYCOUNTRY or not isinstance(name, str):
        return None
    name = name.strip()
    if name in _name_to_iso3_cache:
        return _name_to_iso3_cache[name]
    try:
        c = pycountry.countries.lookup(name)
        code = c.alpha_3
        _name_to_iso3_cache[name] = code
        return code
    except LookupError:
        # Try fuzzy search on common aliases
        results = pycountry.countries.search_fuzzy(name)
        if results:
            code = results[0].alpha_3
            _name_to_iso3_cache[name] = code
            return code
    _name_to_iso3_cache[name] = None
    return None


def ensure_iso3(df: pd.DataFrame, iso_col: str = "country_iso3", name_col: str | None = None) -> pd.DataFrame:
    """Ensure a clean iso3 column exists; fill from name_col if missing."""
    df = df.copy()
    if iso_col not in df.columns:
        df[iso_col] = None

    # Apply manual overrides
    df[iso_col] = df[iso_col].map(lambda x: ISO3_OVERRIDES.get(x, x) if isinstance(x, str) else x)

    # Fill from name where iso3 is missing
    if name_col and name_col in df.columns:
        mask = df[iso_col].isna()
        df.loc[mask, iso_col] = df.loc[mask, name_col].map(name_to_iso3)

    return df


# ---------------------------------------------------------------------------
# Bronze → Silver transforms (one per source)
# ---------------------------------------------------------------------------

def transform_hno(path: Path) -> pd.DataFrame:
    """Clean HNO data. Output: country_iso3, year, sector, people_in_need, severity_level."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"  HNO raw: {df.shape}, columns: {list(df.columns[:8])}")

    # HDX HAPI HNO columns (may vary by API version)
    # HAPI HNO schema: location_code, location_name, population, population_status,
    #   sector_name, reference_period_start, admin1_name
    # population_status: 'INN'=In Need, 'TGT'=Targeted, 'AFF'=Affected, etc.
    df = df.rename(columns={
        "location_code": "country_iso3",
        "location_name": "country_name",
        "population": "people_in_need",
        "sector_name": "sector",
        "admin1_name": "admin1",
        "reference_period_start": "date_start",
    })

    # Keep only "In Need" rows (INN); drop targeted/affected counts
    if "population_status" in df.columns:
        df = df[df["population_status"] == "INN"].copy()

    # Derive year
    if "date_start" in df.columns:
        df["year"] = pd.to_datetime(df["date_start"], errors="coerce").dt.year
    if "year" not in df.columns:
        df["year"] = datetime.today().year

    df = ensure_iso3(df, name_col="country_name")
    df["people_in_need"] = pd.to_numeric(df.get("people_in_need", 0), errors="coerce").fillna(0)

    keep = ["country_iso3", "country_name", "year", "sector", "people_in_need", "admin1"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["country_iso3"])
    df = df[df["people_in_need"] > 0]

    return df


def transform_funding(path: Path) -> pd.DataFrame:
    """Clean FTS funding data. Output: country_iso3, year, funding_requested, funding_received, hrp_status."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"  Funding raw: {df.shape}, columns: {list(df.columns[:8])}")

    # Possible column names across different FTS CSV formats
    rename = {
        # HDX HAPI /coordination-context/funding columns
        "location_code": "country_iso3",
        "location_name": "country_name",
        "requirements_usd": "funding_requested",
        "funding_usd": "funding_received",
        "funding_pct": "coverage_pct_raw",
        "appeal_type": "hrp_status",
        "appeal_name": "plan_name",
        "appeal_code": "plan_id",
        "reference_period_start": "date_start",
        # Legacy CSV column names
        "Country": "country_name",
        "iso3": "country_iso3",
        "Year": "year",
        "Revisited requirements": "funding_requested",
        "Requirements": "funding_requested",
        "Total Funding": "funding_received",
        "Funded %": "coverage_pct_raw",
        "Plan Type": "hrp_status",
        "plan_name": "plan_name",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df = ensure_iso3(df, name_col="country_name")
    for col in ["funding_requested", "funding_received"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[,$]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0)
            )
        else:
            df[col] = 0.0

    # Derive year from date_start if present (HAPI returns ISO dates)
    if "year" not in df.columns:
        if "date_start" in df.columns:
            df["year"] = pd.to_datetime(df["date_start"], errors="coerce").dt.year
        else:
            df["year"] = datetime.today().year
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(datetime.today().year).astype(int)
    df["has_hrp"] = df.get("hrp_status", pd.Series([""] * len(df))).str.contains(
        r"HRP|Humanitarian Response Plan", case=False, na=False
    )

    keep = ["country_iso3", "country_name", "year", "funding_requested", "funding_received",
            "has_hrp", "hrp_status", "plan_name"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].dropna(subset=["country_iso3"])


def transform_cbpf(path: Path) -> pd.DataFrame:
    """Clean CBPF pooled fund data. Output: country_iso3, year, cbpf_allocated."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"  CBPF raw: {df.shape}, columns: {list(df.columns[:8])}")

    rename = {
        "CountryCode": "country_iso3",
        "country_code": "country_iso3",
        "CountryName": "country_name",
        "country_name": "country_name",
        "AllocationYear": "year",
        "allocation_year": "year",
        "AmountUSD": "cbpf_allocated",
        "amount_usd": "cbpf_allocated",
        "ApprovalYear": "year",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = ensure_iso3(df, name_col="country_name")

    if "year" not in df.columns:
        df["year"] = datetime.today().year
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(datetime.today().year).astype(int)

    amount_col = "cbpf_allocated" if "cbpf_allocated" in df.columns else None
    if amount_col is None:
        # Try to find any USD column
        usd_cols = [c for c in df.columns if "usd" in c.lower() or "amount" in c.lower()]
        if usd_cols:
            df["cbpf_allocated"] = pd.to_numeric(df[usd_cols[0]], errors="coerce").fillna(0)
        else:
            df["cbpf_allocated"] = 0.0
    else:
        df["cbpf_allocated"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    agg = df.groupby(["country_iso3", "year"], as_index=False)["cbpf_allocated"].sum()
    return agg


def transform_inform(path: Path) -> pd.DataFrame:
    """Clean INFORM severity index. Output: country_iso3, year, inform_severity."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    print(f"  INFORM raw: {df.shape}, columns: {list(df.columns[:8])}")

    # INFORM XLSX columns: 'ISO3', 'INFORM Severity Index', 'COUNTRY'
    iso_col = next((c for c in df.columns if c.upper() == "ISO3" or c.lower() == "iso3"), None)
    sev_col = next((c for c in df.columns if "inform severity index" in c.lower()), None)
    if sev_col is None:
        sev_col = next((c for c in df.columns if "inform" in c.lower() and "sever" in c.lower()), None)
    if sev_col is None:
        sev_col = next((c for c in df.columns if "sever" in c.lower()), None)
    name_col = next((c for c in df.columns if c.upper() == "COUNTRY" or "country" in c.lower()), None)

    if iso_col:
        df = df.rename(columns={iso_col: "country_iso3"})
    if sev_col:
        df = df.rename(columns={sev_col: "inform_severity"})
    if name_col:
        df = df.rename(columns={name_col: "country_name"})

    df = ensure_iso3(df, name_col="country_name")
    df["inform_severity"] = pd.to_numeric(df.get("inform_severity", pd.Series([np.nan] * len(df))), errors="coerce")

    # INFORM index often doesn't have a year column — infer from filename or use current
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if year_col:
        df["year"] = pd.to_numeric(df[year_col], errors="coerce").fillna(datetime.today().year).astype(int)
    else:
        df["year"] = datetime.today().year

    keep = ["country_iso3", "year", "inform_severity"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["country_iso3", "inform_severity"])

    # INFORM tracks multiple crises per country — take max severity to avoid join explosion
    df = df.groupby(["country_iso3"], as_index=False)["inform_severity"].max()
    return df


# ---------------------------------------------------------------------------
# Join all silver sources into master table
# ---------------------------------------------------------------------------

def compute_consecutive_underfunded(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each country, compute how many consecutive years (ending at max year present)
    the crisis has been underfunded (<50% coverage).
    """
    df = df.copy()
    df = df.sort_values(["country_iso3", "year"])

    result_rows = []
    for iso3, group in df.groupby("country_iso3"):
        group = group.sort_values("year").reset_index(drop=True)
        consec = 0
        for _, row in group.iterrows():
            underfunded = row.get("coverage_ratio", 0) < 0.5
            if underfunded:
                consec += 1
            else:
                consec = 0
            result_rows.append({**row.to_dict(), "consecutive_years_underfunded": consec})

    return pd.DataFrame(result_rows)


def build_silver_master() -> pd.DataFrame:
    """Join all transformed sources into a single Silver master table."""
    print("\nTransforming Bronze → Silver …")

    hno = transform_hno(BRONZE_DIR / "bronze_hno.parquet")
    funding = transform_funding(BRONZE_DIR / "bronze_funding.parquet")
    cbpf = transform_cbpf(BRONZE_DIR / "bronze_cbpf.parquet")
    inform = transform_inform(BRONZE_DIR / "bronze_inform.parquet")

    # Aggregate HNO to country level (sum across sectors, keep max severity)
    if not hno.empty:
        agg_dict = {
            "people_in_need": ("people_in_need", "sum"),
            "country_name": ("country_name", "first"),
        }
        if "severity_level" in hno.columns:
            agg_dict["severity_level"] = ("severity_level", "max")
        hno_country = hno.groupby(["country_iso3", "year"], as_index=False).agg(**agg_dict)
    else:
        print("  WARNING: HNO empty — creating stub from funding country list")
        if not funding.empty:
            hno_country = funding[["country_iso3", "country_name", "year"]].copy()
            hno_country["people_in_need"] = np.nan
            hno_country["severity_level"] = np.nan
        else:
            hno_country = pd.DataFrame(columns=["country_iso3", "year", "people_in_need", "country_name"])

    # Join funding — use most-recent funding year per country to maximise join coverage
    if not funding.empty:
        funding_agg = funding.groupby(["country_iso3", "year"], as_index=False).agg(
            funding_requested=("funding_requested", "sum"),
            funding_received=("funding_received", "sum"),
            has_hrp=("has_hrp", "any"),
            country_name=("country_name", "first"),
        )
        # Best funding year per country (keep most recent ≤ current year)
        current_year = datetime.today().year
        funding_latest = (
            funding_agg[funding_agg["year"] <= current_year]
            .sort_values("year", ascending=False)
            .drop_duplicates("country_iso3")
            .rename(columns={"year": "funding_year"})
        )
        # Outer join on country only, then reconcile year
        master = pd.merge(hno_country, funding_latest, on="country_iso3", how="outer",
                          suffixes=("", "_fund"))
        # Use HNO year if available, otherwise funding year
        master["year"] = master["year"].combine_first(master.get("funding_year", pd.Series(dtype=float)))
        master["country_name"] = master["country_name"].fillna(master.get("country_name_fund", None))
    else:
        master = hno_country.copy()
        master["funding_requested"] = 0.0
        master["funding_received"] = 0.0
        master["has_hrp"] = False

    # Join CBPF
    if not cbpf.empty:
        master = pd.merge(master, cbpf, on=["country_iso3", "year"], how="left")
    else:
        master["cbpf_allocated"] = 0.0

    # Join INFORM — join on country only (INFORM has one severity per country after aggregation)
    if not inform.empty:
        master = pd.merge(master, inform[["country_iso3", "inform_severity"]], on="country_iso3", how="left")
    else:
        master["inform_severity"] = np.nan

    # Clean up types
    for col in ["funding_requested", "funding_received", "cbpf_allocated"]:
        master[col] = pd.to_numeric(master.get(col, 0), errors="coerce").fillna(0)
    master["people_in_need"] = pd.to_numeric(master.get("people_in_need", np.nan), errors="coerce")
    master["has_hrp"] = master.get("has_hrp", False).fillna(False).astype(bool)

    # Coverage ratio
    master["coverage_ratio"] = master.apply(
        lambda r: min(r["funding_received"] / r["funding_requested"], 1.0)
        if r["funding_requested"] > 0 else 0.0,
        axis=1,
    )

    # Staleness: assume data is from Jan 1 of the reported year
    master["data_date"] = pd.to_datetime(master["year"].astype(str) + "-01-01", errors="coerce")
    master["data_staleness_days"] = (TODAY - master["data_date"]).dt.days.clip(lower=0)

    # Consecutive years underfunded
    master = compute_consecutive_underfunded(master)

    # Drop duplicate country_name columns
    dup_cols = [c for c in master.columns if c.endswith("_fund") or c.endswith("_y")]
    master = master.drop(columns=dup_cols, errors="ignore")

    master = master.dropna(subset=["country_iso3"])
    master = master.sort_values(["country_iso3", "year"]).reset_index(drop=True)

    print(f"\nSilver master: {master.shape[0]:,} rows, {master.shape[1]} columns")
    print(f"  Countries: {master['country_iso3'].nunique()}")
    print(f"  Years: {sorted(master['year'].dropna().unique().tolist())}")
    print(f"  Missing HNO PIN: {master['people_in_need'].isna().sum()} rows")

    return master


def main():
    print("=" * 60)
    print("UNOCHA Geo-Insight — Silver Transform")
    print("=" * 60)
    master = build_silver_master()
    out = SILVER_DIR / "silver_master.parquet"
    master.to_parquet(out, index=False)
    print(f"\nSaved → {out}")
    print("Next: python 03_gold_scoring.py")


if __name__ == "__main__":
    main()
