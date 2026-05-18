"""
Bronze layer: fetch raw humanitarian datasets and save as parquet files.

Run this first:  python 01_bronze_ingest.py

On Databricks, the parquet files can be read directly as Delta tables or via
spark.read.parquet(). Locally they sit in data/bronze/.

Data sources fetched here:
  - HDX HAPI: humanitarian needs (HNO), population
  - HDX CSV:  global requirements & funding, humanitarian response plans
  - INFORM:   global crisis severity index (CSV from HDX)
  - CBPF:     country-based pooled fund allocations (JSON API)
"""
import os
import sys
import time
import base64
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

BRONZE_DIR = Path("data/bronze")
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

# HDX HAPI requires app_identifier as base64("app_name:contact_email")
_raw_email = os.getenv("HDX_HAPI_APP_ID", "unocha-hackathon@student.cmu.edu")
HDX_HAPI_APP_ID = base64.b64encode(f"unocha-hackathon:{_raw_email}".encode()).decode()
HDX_HAPI_BASE = "https://hapi.humdata.org/api/v1"

HEADERS = {"User-Agent": f"unocha-hackathon/{_raw_email}"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def hapi_get_all(endpoint: str, params: dict, page_size: int = 1000, max_pages: int = 20) -> list[dict]:
    """Page through a HDX HAPI endpoint and return all records (up to max_pages)."""
    params = {**params, "app_identifier": HDX_HAPI_APP_ID, "limit": page_size, "offset": 0}
    records = []
    page = 0
    while page < max_pages:
        resp = requests.get(f"{HDX_HAPI_BASE}{endpoint}", params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("data", [])
        records.extend(batch)
        print(f"    page {page+1}: {len(batch)} records (total so far: {len(records)})")
        if len(batch) < page_size:
            break
        params["offset"] += page_size
        page += 1
        time.sleep(0.2)
    if page == max_pages:
        print(f"  Hit max_pages={max_pages} limit — increase if you need more data")
    return records


def save_bronze(df: pd.DataFrame, name: str) -> Path:
    out = BRONZE_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    print(f"  Saved {len(df):,} rows → {out}")
    return out


# ---------------------------------------------------------------------------
# 1. HNO — Humanitarian Needs Overview (people in need + severity via HAPI)
# ---------------------------------------------------------------------------

def fetch_hno() -> pd.DataFrame:
    print("Fetching HNO (humanitarian needs) via HDX HAPI …")
    # admin_level=0 gives country-level aggregates only (not subnational)
    # population_status=INN filters to "In Need" only (not Targeted/Affected)
    records = hapi_get_all(
        "/affected-people/humanitarian-needs",
        params={"output_format": "json", "admin_level": 0, "population_status": "INN"},
    )
    if not records:
        print("  WARNING: No HNO records returned. Check your app_identifier or network.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# 2. Global Requirements & Funding (FTS) — CSV from HDX
# ---------------------------------------------------------------------------

def fetch_funding_csv() -> pd.DataFrame:
    """Fetch funding via HDX HAPI /coordination-context/funding endpoint."""
    print("Fetching funding data via HDX HAPI /coordination-context/funding …")
    try:
        records = hapi_get_all(
            "/coordination-context/funding",
            params={"output_format": "json"},
            max_pages=20,
        )
        if records:
            df = pd.DataFrame(records)
            print(f"  Funding via HAPI: {len(df):,} rows, cols: {list(df.columns)}")
            return df
    except Exception as e:
        print(f"  HAPI funding failed ({e}). No funding data.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. CBPF — Country-Based Pooled Funds allocations
# ---------------------------------------------------------------------------

def fetch_cbpf() -> pd.DataFrame:
    """Fetch CBPF pooled fund data via HDX HAPI (most reliable endpoint)."""
    print("Fetching CBPF pooled fund allocations …")

    # Primary: HDX HAPI funding endpoint
    try:
        records = hapi_get_all(
            "/coordination-context/funding",
            params={"output_format": "json"},
            max_pages=10,
        )
        if records:
            df = pd.DataFrame(records)
            print(f"  CBPF via HAPI /funding: {len(df):,} rows, cols: {list(df.columns[:6])}")
            return df
    except Exception as e:
        print(f"  HAPI /funding failed ({e}), trying CBPF data hub …")

    # Fallback: CBPF data hub with correct 2024 API paths
    for url in [
        "https://cbpf.data.unocha.org/api/v2/PooledFunds",
        "https://cbpf.data.unocha.org/api/v1/PooledFunds/Allocations",
        "https://cbpf.data.unocha.org/api/Data/Allocations",
    ]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                records = data if isinstance(data, list) else data.get("data", data.get("value", []))
                df = pd.DataFrame(records)
                print(f"  CBPF from {url}: {len(df):,} rows")
                return df
        except Exception:
            continue

    print("  CBPF unavailable — will be absent from scoring (funding signal from HRP only).")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. INFORM Severity Index
# ---------------------------------------------------------------------------

INFORM_HDXAPI = "https://data.humdata.org/api/action/package_show?id=inform-global-crisis-severity-index"


def fetch_inform() -> pd.DataFrame:
    """Fetch INFORM Severity Index — tries CSV, then XLSX, then HAPI severity."""
    print("Fetching INFORM Severity Index from HDX …")
    try:
        meta_resp = requests.get(INFORM_HDXAPI, headers=HEADERS, timeout=20)
        meta_resp.raise_for_status()
        resources = meta_resp.json()["result"]["resources"]
        print(f"  INFORM dataset has {len(resources)} resources: {[(r.get('name','?'), r.get('format','?')) for r in resources[:5]]}")

        # INFORM only publishes XLSX — use the most recent one
        xlsx_res = next((r for r in resources if r.get("format", "").upper() in ("XLSX", "XLS")), None)
        if xlsx_res:
            try:
                from io import BytesIO
                url = xlsx_res["download_url"]
                print(f"  Downloading INFORM XLSX: {xlsx_res['name']}")
                resp = requests.get(url, headers=HEADERS, timeout=120)
                resp.raise_for_status()
                xl = pd.ExcelFile(BytesIO(resp.content), engine="openpyxl")
                # The key sheet is 'INFORM Severity - all crises', header row = 1
                target_sheet = next(
                    (s for s in xl.sheet_names if "all crises" in s.lower()),
                    xl.sheet_names[2] if len(xl.sheet_names) > 2 else xl.sheet_names[0],
                )
                df = xl.parse(target_sheet, header=1)
                # Drop metadata rows (CRISIS == 'Weights' or 'a-z')
                if "CRISIS" in df.columns:
                    df = df[~df["CRISIS"].isin(["Weights", "(a-z)"])].reset_index(drop=True)
                print(f"  INFORM (XLSX sheet='{target_sheet}'): {len(df):,} rows, cols: {list(df.columns[:6])}")
                return df
            except Exception as e2:
                print(f"  INFORM XLSX failed: {e2}")

    except Exception as e:
        print(f"  INFORM metadata fetch failed ({e})")

    # Last resort: HDX HAPI severity endpoint
    try:
        print("  Trying HDX HAPI /coordination-context/operational-presence for severity …")
        records = hapi_get_all(
            "/coordination-context/operational-presence",
            params={"output_format": "json"},
            max_pages=5,
        )
        if records:
            df = pd.DataFrame(records)
            print(f"  Operational presence (severity proxy): {len(df):,} rows")
            return df
    except Exception as e:
        print(f"  HAPI severity fallback also failed ({e})")

    print("  INFORM unavailable — severity_mult will default to 0.5 for all crises.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 5. Population data (COD-PS) via HDX HAPI
# ---------------------------------------------------------------------------

def fetch_population() -> pd.DataFrame:
    print("Fetching population data via HDX HAPI …")
    try:
        records = hapi_get_all(
            "/population-social/population",
            params={"output_format": "json"},
        )
        df = pd.DataFrame(records)
        print(f"  Population: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  Population fetch failed ({e}). Returning empty DataFrame.")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("UNOCHA Geo-Insight — Bronze Ingest")
    print("=" * 60)

    # HNO
    hno = fetch_hno()
    if not hno.empty:
        save_bronze(hno, "bronze_hno")
    else:
        print("  HNO is empty — pipeline will fall back to INFORM severity proxy.")

    # Funding
    funding = fetch_funding_csv()
    if not funding.empty:
        save_bronze(funding, "bronze_funding")

    # CBPF
    cbpf = fetch_cbpf()
    if not cbpf.empty:
        save_bronze(cbpf, "bronze_cbpf")

    # INFORM
    inform = fetch_inform()
    if not inform.empty:
        save_bronze(inform, "bronze_inform")

    # Population
    pop = fetch_population()
    if not pop.empty:
        save_bronze(pop, "bronze_population")

    print("\nBronze ingest complete. Files in data/bronze/")
    print("Next: python 02_silver_transform.py")


if __name__ == "__main__":
    main()
