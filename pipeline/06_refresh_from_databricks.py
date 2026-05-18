"""
06_refresh_from_databricks.py
=================================
Rebuilds Bronze → Silver → Gold using official Databricks volume files.

Input files (place in data/raw/ or pass paths below):
  - hpc_hno_2024.csv, hpc_hno_2025.csv, hpc_hno_2026.csv
  - fts_requirements_funding_global.csv
  - fts_requirements_funding_globalcluster_global.csv
  - humanitarian-response-plans.csv
  - 202604-inform-severity-april-2026.xlsx
  - Allocations__20260518_145817_UTC.csv  (CBPF)

Run:
  python 06_refresh_from_databricks.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent   # project root
RAW   = Path.home() / "Downloads"          # where Databricks files were saved
DATA  = _HERE / "data"
GOLD_OUT    = DATA / "gold" / "gold_ranked_crises.parquet"
SECTOR_OUT  = DATA / "gold" / "sector_funding_gaps.csv"
SILVER_OUT  = DATA / "silver" / "silver_master.parquet"

for p in [DATA / "bronze", DATA / "silver", DATA / "gold"]:
    p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_HERE))
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from scoring_logic import score_dataframe

# ── Country name → ISO3 map (for CBPF allocations) ──────────────────────────
_CBPF_NAME_MAP = {
    "Afghanistan": "AFG", "CAR": "CAF", "DRC": "COD", "Ethiopia": "ETH",
    "Iraq": "IRQ", "Jordan": "JOR", "Lebanon": "LBN", "Myanmar": "MMR",
    "Nigeria": "NGA", "Pakistan": "PAK", "oPt": "PSE", "Somalia": "SOM",
    "South Sudan": "SSD", "Sudan": "SDN", "Syria": "SYR",
    "Syria Cross border": "SYR", "Yemen": "YEM", "Haiti": "HTI",
    "Mali": "MLI", "Niger": "NER", "Burkina Faso": "BFA", "Chad": "TCD",
    "Cameroon": "CMR", "Colombia": "COL", "Venezuela": "VEN",
    "Mozambique": "MOZ", "Zimbabwe": "ZWE", "Uganda": "UGA",
    "Kenya": "KEN", "Libya": "LBY", "Ukraine": "UKR", "Bangladesh": "BGD",
    "Philippines": "PHL", "Indonesia": "IDN", "Central African Republic": "CAF",
}

# ── HNO cluster → FTS cluster name keywords ──────────────────────────────────
_CLUSTER_LABEL = {
    "FSC": "Food Security", "HEA": "Health", "WSH": "Water",
    "SHL": "Shelter", "NUT": "Nutrition", "EDU": "Education",
    "PRO": "Protection", "PRO-GBV": "GBV", "PRO-CPN": "Child Protection",
    "AGR": "Agriculture", "CCM": "Coordination",
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — HNO: national-level PIN per country per year (2024–2026)
# ─────────────────────────────────────────────────────────────────────────────
print("📥 Step 1: Loading HNO data …")

def _load_hno(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["Country ISO3"] != "#country+code"]          # drop HXL header row
    national = df[(df["Cluster"] == "ALL") & (df["Category"].isna())].copy()
    national = national[["Country ISO3", "In Need", "Targeted"]].rename(columns={
        "Country ISO3": "country_iso3",
        "In Need":      "people_in_need",
        "Targeted":     "people_targeted",
    })
    for col in ["people_in_need", "people_targeted"]:
        national[col] = pd.to_numeric(national[col], errors="coerce")
    national["year"] = year
    return national.dropna(subset=["people_in_need"]).drop_duplicates("country_iso3")

hno_frames = []
for yr, fname in [(2024, "hpc_hno_2024.csv"), (2025, "hpc_hno_2025.csv"), (2026, "hpc_hno_2026.csv")]:
    p = RAW / fname
    if p.exists():
        f = _load_hno(p, yr)
        hno_frames.append(f)
        print(f"  HNO {yr}: {len(f)} countries")
    else:
        print(f"  ⚠️  {fname} not found, skipping")

hno_all = pd.concat(hno_frames, ignore_index=True)
print(f"  Total HNO rows: {len(hno_all)} across {hno_all['country_iso3'].nunique()} countries")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — FTS: requirements + funding per country per year (2024–2026)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📥 Step 2: Loading FTS funding data …")

fts_path = RAW / "fts_requirements_funding_global.csv"
fts = pd.read_csv(fts_path)
fts = fts[fts["year"].isin([2024, 2025, 2026])].copy()
fts = fts.dropna(subset=["requirements", "funding"])
fts = fts[fts["requirements"] > 0]

# Keep one row per country+year (some have multiple plan types — aggregate)
fts_agg = (
    fts.groupby(["countryCode", "year"], as_index=False)
    .agg(funding_requested=("requirements", "sum"), funding_received=("funding", "sum"))
)
fts_agg.rename(columns={"countryCode": "country_iso3"}, inplace=True)
fts_agg["coverage_ratio"] = (fts_agg["funding_received"] / fts_agg["funding_requested"]).clip(0, 1)
print(f"  FTS rows: {len(fts_agg)} across {fts_agg['country_iso3'].nunique()} countries, years 2024-2026")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — HRP flag per country per year
# ─────────────────────────────────────────────────────────────────────────────
print("\n📥 Step 3: Loading HRP flag data …")

hrp_path = RAW / "humanitarian-response-plans.csv"
hrp = pd.read_csv(hrp_path)
hrp = hrp[hrp["code"] != "#response+code"]
hrp["year_val"] = pd.to_numeric(hrp["years"], errors="coerce")

hrp_set = set()
for _, row in hrp.iterrows():
    yr = row["year_val"]
    if pd.isna(yr):
        continue
    locs = str(row.get("locations", "") or "")
    for iso in locs.split(" | "):
        iso = iso.strip()
        if len(iso) == 3:
            hrp_set.add((iso, int(yr)))

print(f"  HRP (country, year) pairs: {len(hrp_set)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — INFORM Severity (April 2026) — take max severity per ISO3
# ─────────────────────────────────────────────────────────────────────────────
print("\n📥 Step 4: Loading INFORM Severity …")

inform_path = RAW / "202604-inform-severity-april-2026.xlsx"
if not inform_path.exists():
    inform_path = RAW / "202604-inform-severity-april-2026-1.xlsx"

try:
    inform_raw = pd.read_excel(
        inform_path, sheet_name="INFORM Severity - country", header=None
    )
    # Row 0 has column labels, real data starts row 3
    # Col 3 = ISO3, Col 5 = INFORM Severity Index
    data_rows = inform_raw.iloc[3:].copy()
    data_rows.columns = range(len(data_rows.columns))
    inform = data_rows[[3, 5]].copy()
    inform.columns = ["country_iso3", "inform_severity"]
    inform["inform_severity"] = pd.to_numeric(inform["inform_severity"], errors="coerce")
    inform = inform.dropna()
    # If multiple crises per country, take max severity
    inform = inform.groupby("country_iso3", as_index=False)["inform_severity"].max()
    print(f"  INFORM countries: {len(inform)}")
except Exception as e:
    print(f"  ⚠️  Could not load INFORM: {e} — will use default severity 5.0")
    inform = pd.DataFrame(columns=["country_iso3", "inform_severity"])

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — CBPF Allocations (use existing bronze if available, else from file)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📥 Step 5: Loading CBPF Allocations …")

cbpf_existing = DATA / "bronze" / "bronze_cbpf.parquet"
if cbpf_existing.exists():
    cbpf_df = pd.read_parquet(cbpf_existing)
    print(f"  Using existing bronze_cbpf: {cbpf_df.shape}")
    # Expect columns: country_iso3, year, cbpf_allocated
    cbpf_agg = cbpf_df.rename(columns={
        c: "country_iso3" for c in cbpf_df.columns if "iso" in c.lower() or "country_code" in c.lower()
    })
    if "cbpf_allocated" not in cbpf_agg.columns:
        # Try to compute from whatever columns exist
        amt_col = [c for c in cbpf_agg.columns if "amount" in c.lower() or "budget" in c.lower()]
        if amt_col:
            cbpf_agg["cbpf_allocated"] = pd.to_numeric(cbpf_agg[amt_col[0]], errors="coerce")
    cbpf_agg = cbpf_agg[cbpf_agg.get("year", pd.Series([2024])).isin([2024, 2025, 2026])] \
        if "year" in cbpf_agg.columns else cbpf_agg
else:
    alloc_path = RAW / "Allocations__20260518_145817_UTC.csv"
    alloc = pd.read_csv(alloc_path)
    alloc["country_iso3"] = alloc["PooledFund"].map(_CBPF_NAME_MAP)
    alloc["cbpf_allocated"] = pd.to_numeric(alloc["Budget"], errors="coerce")
    cbpf_agg = alloc.groupby(["country_iso3", "Year"], as_index=False)["cbpf_allocated"].sum()
    cbpf_agg.rename(columns={"Year": "year"}, inplace=True)
    cbpf_agg = cbpf_agg.dropna(subset=["country_iso3"])
    print(f"  CBPF from Allocations file: {cbpf_agg.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Join → Silver master
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔗 Step 6: Joining to Silver master …")

# Start with HNO (anchor — every country with PIN)
silver = hno_all.copy()

# Add FTS funding
silver = silver.merge(fts_agg, on=["country_iso3", "year"], how="left")

# HRP flag
silver["has_hrp"] = silver.apply(
    lambda r: (r["country_iso3"], int(r["year"])) in hrp_set, axis=1
)
# Countries with no HRP: set funding to 0, requested to 0 (unknown)
silver["funding_requested"] = silver["funding_requested"].fillna(0)
silver["funding_received"]  = silver["funding_received"].fillna(0)
silver["coverage_ratio"]    = np.where(
    silver["funding_requested"] > 0,
    (silver["funding_received"] / silver["funding_requested"]).clip(0, 1),
    0.0,
)

# INFORM severity
silver = silver.merge(inform, on="country_iso3", how="left")
silver["inform_severity"] = silver["inform_severity"].fillna(5.0)  # default mid

# CBPF
if "country_iso3" in cbpf_agg.columns and "year" in cbpf_agg.columns:
    silver = silver.merge(cbpf_agg[["country_iso3", "year", "cbpf_allocated"]],
                          on=["country_iso3", "year"], how="left")
else:
    silver["cbpf_allocated"] = 0.0
silver["cbpf_allocated"] = silver["cbpf_allocated"].fillna(0)

# Country names (from pycountry)
try:
    import pycountry
    def _country_name(iso3: str) -> str:
        try:
            return pycountry.countries.get(alpha_3=iso3).name
        except Exception:
            return iso3
    silver["country_name"] = silver["country_iso3"].apply(_country_name)
except ImportError:
    silver["country_name"] = silver["country_iso3"]

# Data staleness (HNO files are from 2024/2025/2026 — estimate staleness)
from datetime import date
today = date.today()
silver["data_date"] = pd.to_datetime(silver["year"].astype(str) + "-06-01")
silver["data_staleness_days"] = (pd.Timestamp(today) - silver["data_date"]).dt.days.clip(0)

# Consecutive years underfunded (across 2024–2026)
silver_sorted = silver.sort_values(["country_iso3", "year"])
silver_sorted["underfunded"] = silver_sorted["coverage_ratio"] < 0.5

def _consec(group):
    """Count how many recent consecutive years underfunded (from latest year back)."""
    group = group.sort_values("year", ascending=False)
    count = 0
    for val in group["underfunded"]:
        if val:
            count += 1
        else:
            break
    return count

consec = silver_sorted.groupby("country_iso3").apply(_consec).reset_index()
consec.columns = ["country_iso3", "consecutive_years_underfunded"]
silver = silver.merge(consec, on="country_iso3", how="left")
silver["consecutive_years_underfunded"] = silver["consecutive_years_underfunded"].fillna(0).astype(int)

print(f"  Silver rows: {len(silver)} | Countries: {silver['country_iso3'].nunique()} | Years: {sorted(silver['year'].unique())}")
silver.to_parquet(SILVER_OUT, index=False)
print(f"  Saved → {SILVER_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Score → Gold ranked crises
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏅 Step 7: Scoring → Gold table …")

score_input = silver[[
    "country_iso3", "country_name", "year",
    "people_in_need", "funding_requested", "funding_received",
    "has_hrp", "cbpf_allocated", "inform_severity",
    "consecutive_years_underfunded", "data_staleness_days",
]].copy()

gold = score_dataframe(score_input, apply_pareto=True)
gold["coverage_pct"] = gold["coverage_ratio"] * 100

# Rank
gold = gold.sort_values("gap_score", ascending=False).reset_index(drop=True)
gold["rank"] = range(1, len(gold) + 1)

gold.to_parquet(GOLD_OUT, index=False)
print(f"  Gold rows: {len(gold)} | Saved → {GOLD_OUT}")
print("\n  Top 10 overlooked crises:")
cols = ["rank", "country_name", "year", "gap_score", "coverage_pct", "people_in_need"]
print(gold[cols].head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Sector gaps: FTS cluster global × HNO cluster PIN (2024+2025+2026)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 Step 8: Rebuilding sector funding gaps …")

# Load FTS cluster global
ftsc_path = RAW / "fts_requirements_funding_globalcluster_global.csv"
ftsc = pd.read_csv(ftsc_path)
ftsc = ftsc[ftsc["year"].isin([2024, 2025, 2026])].copy()
ftsc_agg = (
    ftsc.groupby(["countryCode", "year", "cluster"], as_index=False)
    .agg(requirements=("requirements", "sum"), funding=("funding", "sum"))
)
ftsc_agg.rename(columns={"countryCode": "country_iso3"}, inplace=True)

# Map FTS cluster names → HNO cluster codes
_FTS_TO_HNO = {
    "Food Security":                    "FSC",
    "Food security":                    "FSC",
    "Agriculture":                      "AGR",
    "Health":                           "HEA",
    "Nutrition":                        "NUT",
    "Education":                        "EDU",
    "Emergency Shelter and NFI":        "SHL",
    "Shelter and Non-Food Items":       "SHL",
    "Shelter":                          "SHL",
    "Water Sanitation Hygiene":         "WSH",
    "Water, Sanitation and Hygiene":    "WSH",
    "WASH":                             "WSH",
    "Protection":                       "PRO",
    "Gender-Based Violence":            "PRO-GBV",
    "GBV":                              "PRO-GBV",
    "Child Protection":                 "PRO-CPN",
    "Mine Action":                      "PRO-MIN",
    "Camp Coordination":                "CCM",
    "Coordination and support services": "CCM",
    "Multipurpose Cash":                "MPC",
    "Multi-Purpose Cash":               "MPC",
}

def _map_cluster(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    for key, code in _FTS_TO_HNO.items():
        if key.lower() in name.lower():
            return code
    return None

ftsc_agg["hno_cluster"] = ftsc_agg["cluster"].apply(_map_cluster)
ftsc_mapped = ftsc_agg.dropna(subset=["hno_cluster"]).copy()

# Aggregate per country+year+hno_cluster
sector = (
    ftsc_mapped.groupby(["country_iso3", "year", "hno_cluster"], as_index=False)
    .agg(requirements=("requirements", "sum"), funding=("funding", "sum"))
)
sector["coverage_pct"] = np.where(
    sector["requirements"] > 0,
    (sector["funding"] / sector["requirements"] * 100).clip(0, 100),
    0.0,
)
sector["funding_gap_pct"] = 100 - sector["coverage_pct"]
sector = sector.sort_values(["year", "funding_gap_pct"], ascending=[False, False])

# Use 2024 as primary year for backward compat (keep most recent per country+cluster)
sector_latest = sector.sort_values("year", ascending=False).drop_duplicates(
    subset=["country_iso3", "hno_cluster"], keep="first"
)

sector_latest.to_csv(SECTOR_OUT, index=False)
print(f"  Sector gaps: {len(sector_latest)} country×cluster pairs | {sector_latest['country_iso3'].nunique()} countries")
print(f"  Saved → {SECTOR_OUT}")
print("\n  Top 10 worst-funded sectors:")
print(sector_latest[["country_iso3","year","hno_cluster","coverage_pct","funding_gap_pct"]]
      .sort_values("funding_gap_pct", ascending=False).head(10).to_string(index=False))

print("\n✅ Pipeline complete!")
print(f"  Gold: {len(gold)} rows, {gold['country_iso3'].nunique()} countries")
print(f"  Sector gaps: {len(sector_latest)} rows, {sector_latest['country_iso3'].nunique()} countries")
