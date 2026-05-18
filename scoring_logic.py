"""
Gap scoring engine for UNOCHA Geo-Insight hackathon.

All functions are pure Python / pandas — importable in local notebooks,
Databricks notebooks, and the Streamlit dashboard without modification.

MCDM design (CMU Decision Modeling):
  - Pareto filtering removes dominated crises before scoring
  - Need scale uses 95th-percentile reference (not global max) to prevent
    Ringer Bid effect where one extreme crisis warps everyone else's scores
  - Weights are configurable via SCORING_WEIGHTS (Swing Weighting principle:
    weight = value of improving that dimension from worst to best)
  - Confidence intervals via ±20% sensitivity on key inputs
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Scoring weights — tunable via Swing Weighting (CMU dm-choice-mcdm.md)
#
# Swing weighting rationale:
#   "If you could move only ONE dimension from worst to best, which matters most?"
#   funding_gap   → 100 pts  (most impactful: directly determines resource reach)
#   need_scale    → 80  pts  (scale of human suffering)
#   severity_mult → 50  pts  (independent urgency signal)
#   Sum = 230 → normalized weights below
# ---------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "funding_gap": 100 / 230,    # 0.435
    "need_scale":  80  / 230,    # 0.348
    "severity":    50  / 230,    # 0.217
}

NEGLECT_BONUS_PER_YEAR = 0.15   # +15% per consecutive underfunded year


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def compute_coverage_ratio(funding_received: float, funding_requested: float) -> float:
    """Fraction of requested funding actually received (capped at 1.0)."""
    if funding_requested <= 0:
        return 0.0
    return min(funding_received / funding_requested, 1.0)


def compute_gap_score(
    people_in_need: float,
    funding_received: float,
    funding_requested: float,
    inform_severity: float | None,
    consecutive_years_underfunded: int = 0,
    reference_pin: float = None,       # 95th-pct PIN across all crises (Ringer Bid fix)
    weights: dict = None,
) -> dict:
    """
    Composite gap score for a single crisis row.

    Returns a dict with all intermediate components for full transparency.

    MCDM formula (weighted sum, CMU dm-choice-mcdm.md §2):
        funding_gap   = 1 - coverage_ratio                 [primary signal]
        need_scale    = log1p(PIN) / log1p(reference_PIN)  [Ringer Bid–safe normalization]
        severity_mult = INFORM_severity / 10               [independent urgency]
        base_score    = w1*funding_gap + w2*need_scale*funding_gap + w3*severity*funding_gap
                      = funding_gap * (w1 + w2*need_scale + w3*severity_mult)
        neglect_factor = 1 + years_underfunded * 0.15
        gap_score      = base_score * neglect_factor
    """
    w = weights or SCORING_WEIGHTS

    # 1. Funding gap
    coverage = compute_coverage_ratio(funding_received, funding_requested)
    funding_gap = 1.0 - coverage

    # 2. Need scale — 95th-pct reference prevents Ringer Bid distortion
    ref = reference_pin if (reference_pin and reference_pin > 0) else 1e7
    need_scale = float(np.clip(np.log1p(people_in_need) / np.log1p(ref), 0.0, 1.0))

    # 3. INFORM severity (0–10 → 0–1); default to midpoint when missing
    sev = float(np.clip(float(inform_severity) if inform_severity is not None else 5.0, 0.0, 10.0))
    severity_mult = sev / 10.0

    # 4. Weighted sum (funding_gap gates both need and severity — a crisis with
    #    full funding can't be "overlooked" regardless of need or severity)
    base_score = funding_gap * (
        w["funding_gap"]
        + w["need_scale"] * need_scale
        + w["severity"] * severity_mult
    )

    # 5. Structural neglect bonus (CMU dm-choice-mcdm.md: temporal signals)
    neglect_factor = 1.0 + (max(0, consecutive_years_underfunded) * NEGLECT_BONUS_PER_YEAR)
    final_score = base_score * neglect_factor

    return {
        "coverage_ratio":  round(coverage, 4),
        "funding_gap":     round(funding_gap, 4),
        "need_scale":      round(need_scale, 4),
        "severity_mult":   round(severity_mult, 4),
        "base_score":      round(base_score, 4),
        "neglect_factor":  round(neglect_factor, 4),
        "gap_score":       round(final_score, 4),
    }


def compute_confidence_interval(
    people_in_need: float,
    funding_received: float,
    funding_requested: float,
    inform_severity: float | None,
    consecutive_years_underfunded: int,
    reference_pin: float,
    uncertainty_pct: float = 0.20,
) -> dict:
    """
    Sensitivity analysis: vary key inputs ±uncertainty_pct and return
    (gap_score_low, gap_score_mid, gap_score_high).

    CMU dm-uncertainty.md: surface uncertainty rather than false precision.
    Humanitarian data is routinely 20-30% off in needs assessments.
    """
    base = compute_gap_score(
        people_in_need, funding_received, funding_requested,
        inform_severity, consecutive_years_underfunded, reference_pin,
    )["gap_score"]

    # Pessimistic: more people in need, less funding received
    low_score = compute_gap_score(
        people_in_need * (1 + uncertainty_pct),
        funding_received * (1 - uncertainty_pct),
        funding_requested,
        inform_severity, consecutive_years_underfunded, reference_pin,
    )["gap_score"]

    # Optimistic: fewer people in need, more funding received
    high_score = compute_gap_score(
        people_in_need * (1 - uncertainty_pct),
        funding_received * (1 + uncertainty_pct),
        funding_requested,
        inform_severity, consecutive_years_underfunded, reference_pin,
    )["gap_score"]

    return {
        "gap_score_low":  round(min(base, high_score), 4),
        "gap_score_mid":  round(base, 4),
        "gap_score_high": round(max(base, low_score), 4),
        "gap_score_range": round(abs(low_score - high_score), 4),
    }


# ---------------------------------------------------------------------------
# Pareto filtering (CMU dm-choice-mcdm.md §3)
# ---------------------------------------------------------------------------

def pareto_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove Pareto-dominated crises before scoring.

    A crisis is dominated if another crisis is strictly better on ALL three
    primary dimensions: funding_gap, people_in_need, inform_severity.
    Dominated crises are set aside (not deleted) — kept for transparency.

    Returns: (non_dominated_df, dominated_df)
    """
    df = df.copy().reset_index(drop=True)
    dims = ["funding_gap_raw", "people_in_need", "inform_severity"]

    # Compute raw funding gap for comparison
    df["funding_gap_raw"] = df.apply(
        lambda r: 1.0 - compute_coverage_ratio(r["funding_received"], r["funding_requested"]),
        axis=1,
    )
    # Fill missing for comparison
    for col in dims:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    is_dominated = pd.Series(False, index=df.index)

    for i in df.index:
        for j in df.index:
            if i == j:
                continue
            # j dominates i if j >= i on all dims and > i on at least one
            all_ge = all(
                df.at[j, d] >= df.at[i, d]
                for d in dims if d in df.columns
            )
            any_gt = any(
                df.at[j, d] > df.at[i, d]
                for d in dims if d in df.columns
            )
            if all_ge and any_gt:
                is_dominated[i] = True
                break

    dominated = df[is_dominated].copy()
    non_dominated = df[~is_dominated].copy()

    print(f"  Pareto filter: {len(non_dominated)} non-dominated, {len(dominated)} dominated (set aside)")
    return non_dominated, dominated


# ---------------------------------------------------------------------------
# DataFrame-level scoring pipeline
# ---------------------------------------------------------------------------

def score_dataframe(df: pd.DataFrame, apply_pareto: bool = True) -> pd.DataFrame:
    """
    Apply gap scoring to a Silver-layer DataFrame.

    Expected columns:
        country_iso3, country_name, year,
        people_in_need, funding_requested, funding_received,
        inform_severity (nullable),
        consecutive_years_underfunded (nullable int, defaults to 0),
        has_hrp (bool), data_staleness_days (nullable int)

    Returns DataFrame with scored columns, confidence intervals, and
    a pareto_dominated flag.
    """
    df = df.copy()

    # Defaults
    df["inform_severity"] = df["inform_severity"].fillna(5.0)
    df["consecutive_years_underfunded"] = (
        df.get("consecutive_years_underfunded", pd.Series([0] * len(df)))
        .fillna(0).astype(int)
    )
    df["people_in_need"] = df["people_in_need"].fillna(0)
    df["funding_requested"] = df["funding_requested"].fillna(0)
    df["funding_received"] = df["funding_received"].fillna(0)

    # 95th-percentile reference PIN (Ringer Bid prevention, CMU MCDM §2)
    reference_pin = float(df["people_in_need"].quantile(0.95))
    if reference_pin <= 0:
        reference_pin = float(df["people_in_need"].max()) or 1e7

    # Pareto filter — compare only across different countries, not same country different years
    df["pareto_dominated"] = False
    if apply_pareto and len(df) > 1:
        df["funding_gap_raw"] = df.apply(
            lambda r: 1.0 - compute_coverage_ratio(r["funding_received"], r["funding_requested"]),
            axis=1,
        )
        dims = ["funding_gap_raw", "people_in_need", "inform_severity"]
        for i in df.index:
            for j in df.index:
                if i == j:
                    continue
                # Skip same-country comparisons (different years of the same crisis)
                if "country_iso3" in df.columns and df.at[i, "country_iso3"] == df.at[j, "country_iso3"]:
                    continue
                all_ge = all(df.at[j, d] >= df.at[i, d] for d in dims if d in df.columns)
                any_gt = any(df.at[j, d] > df.at[i, d] for d in dims if d in df.columns)
                if all_ge and any_gt:
                    df.at[i, "pareto_dominated"] = True
                    break

    # Score all rows (dominated rows still scored, but flagged)
    scores = df.apply(
        lambda row: compute_gap_score(
            people_in_need=row["people_in_need"],
            funding_received=row["funding_received"],
            funding_requested=row["funding_requested"],
            inform_severity=row["inform_severity"],
            consecutive_years_underfunded=int(row["consecutive_years_underfunded"]),
            reference_pin=reference_pin,
        ),
        axis=1,
        result_type="expand",
    )
    # Drop columns that score_dataframe will recompute to avoid duplicates
    overlap = [c for c in scores.columns if c in df.columns]
    df = df.drop(columns=overlap, errors="ignore")
    df = pd.concat([df, scores], axis=1)

    # Confidence intervals
    ci = df.apply(
        lambda row: compute_confidence_interval(
            people_in_need=row["people_in_need"],
            funding_received=row["funding_received"],
            funding_requested=row["funding_requested"],
            inform_severity=row["inform_severity"],
            consecutive_years_underfunded=int(row["consecutive_years_underfunded"]),
            reference_pin=reference_pin,
        ),
        axis=1,
        result_type="expand",
    )
    overlap_ci = [c for c in ci.columns if c in df.columns]
    df = df.drop(columns=overlap_ci, errors="ignore")
    df = pd.concat([df, ci], axis=1)

    # Confidence flags
    df["low_confidence"] = (
        (df["funding_requested"] == 0)
        | (df.get("data_staleness_days", pd.Series([0] * len(df))) > 548)
    )

    # Clean up temp column
    df = df.drop(columns=["funding_gap_raw"], errors="ignore")

    return df.sort_values("gap_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Classification and display helpers
# ---------------------------------------------------------------------------

def classify_neglect_type(row: pd.Series) -> str:
    """Classify a crisis as structural, acute, improving, or ongoing."""
    years = int(row.get("consecutive_years_underfunded", 0))
    coverage = float(row.get("coverage_ratio", 0.0))
    if years >= 3:
        return "structural"
    if years == 0 and coverage < 0.3:
        return "acute"
    if coverage > 0.6:
        return "improving"
    return "ongoing"


def build_explanation_prompt(rows: list[dict], query: str) -> str:
    """
    Build the user message for Claude to generate per-crisis briefing notes.

    Uses neutral framing (CMU odi-decisions.md §5 — Framing Trap):
    - Express gaps as coverage shortfalls (neutral delta), not as "people dying"
    - Include confidence interval range so analysts calibrate uncertainty
    - Do NOT anchor to historical funding levels
    """
    crisis_lines = []
    for r in rows:
        ci_range = r.get("gap_score_range", None)
        ci_str = f" [score uncertainty range: ±{ci_range:.3f}]" if ci_range else ""
        dominated = " [Pareto-dominated — weaker on all dimensions than a higher-ranked crisis]" if r.get("pareto_dominated") else ""
        crisis_lines.append(
            f"- {r.get('country_name', r.get('country_iso3'))} ({r.get('country_iso3')}):\n"
            f"  gap_score={r.get('gap_score', 'N/A')}{ci_str}{dominated}\n"
            f"  funding_coverage={r.get('coverage_ratio', 0)*100:.1f}%"
            f" (gap={r.get('funding_gap', 'N/A')})\n"
            f"  people_in_need={r.get('people_in_need', 'N/A'):,.0f}\n"
            f"  INFORM_severity={r.get('inform_severity', 'N/A')}\n"
            f"  neglect_type={r.get('neglect_type', 'N/A')}"
            f", years_underfunded={r.get('consecutive_years_underfunded', 0)}\n"
            f"  has_hrp={r.get('has_hrp', False)}"
            f", low_confidence={r.get('low_confidence', False)}"
        )

    return (
        f"User query: {query}\n\n"
        f"Top ranked crises by gap score:\n" + "\n".join(crisis_lines) + "\n\n"
        "BRIEFING NOTE RULES (follow strictly):\n"
        "1. NEUTRAL FRAMING: Express each situation as a coverage gap (e.g. '38% of requested funding "
        "has been received, leaving a 62% shortfall') — never use loss/death framing or gain framing.\n"
        "2. NO ANCHORING: Do not compare to historical funding or reference prior years as a baseline.\n"
        "3. COUNTER-ARGUMENT: For each crisis, state one reason its gap score might be overstated "
        "(e.g. unreported bilateral funding, data staleness, no HRP so 0% is an assumption).\n"
        "4. UNCERTAINTY: If low_confidence=True, say so explicitly and quantify the score range.\n"
        "5. GROUND ALL NUMBERS in the data above. Do not invent figures."
    )
