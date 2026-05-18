"""
Gold layer: apply gap scoring and produce the ranked crisis table.

Logs parameters and outputs to MLflow.

Run after 02_silver_transform.py:  python 03_gold_scoring.py
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("MLflow not installed — skipping experiment tracking.")

from scoring_logic import score_dataframe, classify_neglect_type

SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
GOLD_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds (tune these during the hackathon)
MIN_PEOPLE_IN_NEED = 100_000       # filter out tiny crises (<100k PIN)
MAX_DATA_STALENESS_DAYS = 730      # ignore data older than 2 years
COVERAGE_GAP_FLOOR = 0.0           # include all coverage levels (no floor filter)


# ---------------------------------------------------------------------------
# Score pipeline
# ---------------------------------------------------------------------------

def load_silver(year: int | None = None) -> pd.DataFrame:
    path = SILVER_DIR / "silver_master.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Silver master not found. Run 02_silver_transform.py first."
        )
    df = pd.read_parquet(path)
    if year is not None:
        df = df[df["year"] == year]
    return df


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply inclusion filters and return (filtered_df, filter_stats)."""
    n_start = len(df)

    # Keep only rows with meaningful need
    df_f = df[df["people_in_need"].fillna(0) >= MIN_PEOPLE_IN_NEED].copy()
    n_after_pin = len(df_f)

    # Drop very stale data
    df_f = df_f[df_f["data_staleness_days"].fillna(0) <= MAX_DATA_STALENESS_DAYS].copy()
    n_after_staleness = len(df_f)

    stats = {
        "n_start": n_start,
        "n_after_pin_filter": n_after_pin,
        "n_after_staleness_filter": n_after_staleness,
        "min_people_in_need": MIN_PEOPLE_IN_NEED,
        "max_staleness_days": MAX_DATA_STALENESS_DAYS,
    }
    return df_f, stats


def run_scoring_pipeline(year: int | None = None) -> pd.DataFrame:
    """Load Silver, filter, score, and return ranked Gold table."""
    print(f"\nLoading Silver data (year={year or 'all'}) …")
    df = load_silver(year)
    print(f"  Loaded {len(df):,} rows")

    df, filter_stats = apply_filters(df)
    print(f"  After filters: {len(df):,} rows")
    for k, v in filter_stats.items():
        print(f"    {k}: {v}")

    # Score
    print("\nScoring …")
    scored = score_dataframe(df)

    # Add neglect classification for UI
    scored["neglect_type"] = scored.apply(classify_neglect_type, axis=1)

    # Add display columns
    scored["coverage_pct"] = (scored["coverage_ratio"] * 100).round(1)
    scored["rank"] = range(1, len(scored) + 1)

    return scored, filter_stats


# ---------------------------------------------------------------------------
# Sanity check: verify known overlooked crises appear in top 20
# ---------------------------------------------------------------------------

KNOWN_OVERLOOKED = ["SSD", "YEM", "SOM", "MMR", "HTI", "COD", "AFG", "SDN"]

def sanity_check(scored: pd.DataFrame) -> dict:
    top20_iso3 = set(scored.head(20)["country_iso3"].tolist())
    found = [iso for iso in KNOWN_OVERLOOKED if iso in top20_iso3]
    missing = [iso for iso in KNOWN_OVERLOOKED if iso not in top20_iso3]
    print(f"\nSanity check — known overlooked crises in top 20:")
    print(f"  Found: {found}")
    print(f"  Missing from top 20 (may need data): {missing}")
    return {"found_in_top20": found, "missing_from_top20": missing}


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_to_mlflow(scored: pd.DataFrame, filter_stats: dict, sanity: dict):
    if not HAS_MLFLOW:
        return
    mlflow.set_experiment("unocha-geo-insight-gap-scoring")
    with mlflow.start_run(run_name=f"gold_scoring_{datetime.today().strftime('%Y%m%d_%H%M')}"):
        # Log parameters
        mlflow.log_params({
            "min_people_in_need": MIN_PEOPLE_IN_NEED,
            "max_staleness_days": MAX_DATA_STALENESS_DAYS,
            "n_scored": len(scored),
            "scoring_version": "v1.0",
        })
        # Log metrics
        mlflow.log_metrics({
            "n_countries_scored": len(scored),
            "n_structural_neglect": int((scored["neglect_type"] == "structural").sum()),
            "n_acute": int((scored["neglect_type"] == "acute").sum()),
            "n_no_hrp": int((~scored["has_hrp"]).sum()),
            "n_known_overlooked_in_top20": len(sanity["found_in_top20"]),
            "mean_gap_score": float(scored["gap_score"].mean()),
            "median_coverage_pct": float(scored["coverage_pct"].median()),
        })
        # Log top 20 as JSON artifact
        top20 = scored.head(20)[
            ["rank", "country_iso3", "country_name", "year", "gap_score",
             "coverage_pct", "people_in_need", "neglect_type", "low_confidence"]
        ].to_dict(orient="records")
        mlflow.log_text(json.dumps(top20, indent=2, default=str), "top20_ranked_crises.json")
        print(f"  MLflow run logged.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("UNOCHA Geo-Insight — Gold Scoring")
    print("=" * 60)

    # Score most recent year by default
    most_recent_year = None  # None = all years; set to e.g. 2024 for single year

    scored, filter_stats = run_scoring_pipeline(year=most_recent_year)

    # Sanity check
    sanity = sanity_check(scored)

    # Save Gold
    out_path = GOLD_DIR / "gold_ranked_crises.parquet"
    scored.to_parquet(out_path, index=False)
    print(f"\nSaved Gold table → {out_path}")

    # Also save a human-readable CSV of top 30
    top30 = scored.head(30)[[
        "rank", "country_iso3", "country_name", "year",
        "gap_score", "coverage_pct", "people_in_need",
        "funding_requested", "funding_received",
        "inform_severity", "neglect_type",
        "consecutive_years_underfunded", "has_hrp", "low_confidence"
    ]]
    top30_path = GOLD_DIR / "top30_overlooked_crises.csv"
    top30.to_csv(top30_path, index=False)
    print(f"Top 30 preview → {top30_path}")
    print("\n", top30[["rank", "country_name", "gap_score", "coverage_pct",
                        "people_in_need", "neglect_type"]].to_string(index=False))

    # Log to MLflow
    log_to_mlflow(scored, filter_stats, sanity)

    print("\nNext: python 04_agent.py  (or  streamlit run 05_dashboard.py)")


if __name__ == "__main__":
    main()
