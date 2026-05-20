"""
Streamlit Command Center — UNOCHA Geo-Insight Hackathon

Launch:  streamlit run 05_dashboard.py

Layout:
  - Left sidebar:  filters (year, region, coverage ceiling, neglect type)
  - Top:           ranked crisis table with color-coded gap scores
  - Middle:        choropleth world map (gap score)
  - Bottom:        conversational agent interface (Claude API)
"""
import os
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent.parent   # project root (one level above app/)
load_dotenv(_HERE / ".env", override=True)
GOLD_PATH = _HERE / "data/gold/gold_ranked_crises.parquet"
SECTOR_PATH = _HERE / "data/gold/sector_funding_gaps.csv"

st.set_page_config(
    page_title="Geo-Insight: Overlooked Crises",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — visual polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Page background ─────────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── App title ───────────────────────────────────────────────────────────── */
h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #1a202c !important;
    letter-spacing: -0.02em;
}
h2 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #2d3748 !important;
    margin-top: 0.5rem;
}
h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #2d3748 !important;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #718096 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1a202c !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #1e2a3a !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #a0aec0 !important;
    font-size: 0.8rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f7fafc !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px;
}
[data-testid="stSidebar"] label {
    color: #cbd5e0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Sidebar widget inputs: force dark text so readable on white bg ───────── */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] *,
[data-testid="stSidebar"] .stNumberInput input {
    color: #1a202c !important;
}

/* Selectbox placeholder and selected value */
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [class*="placeholder"],
[data-testid="stSidebar"] [class*="singleValue"],
[data-testid="stSidebar"] [class*="ValueContainer"] * {
    color: #1a202c !important;
}

/* Slider value label */
[data-testid="stSidebar"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
    color: #cbd5e0 !important;
}

/* Number input +/- buttons */
[data-testid="stSidebar"] [data-baseweb="base-input"] button {
    color: #4a5568 !important;
}

/* Expander title in sidebar — ensure readable text */
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    color: #2d3748 !important;
    font-weight: 500 !important;
}

/* ── Tabs — pill style, no underline indicator ───────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.35rem;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 0.25rem;
    border-bottom: none;
}
[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1.1rem !important;
    border-radius: 7px !important;
    color: #64748b !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.15s ease;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #1a202c !important;
    font-weight: 600 !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.10) !important;
    border: none !important;
}
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {
    color: #334155 !important;
    background: rgba(255,255,255,0.5) !important;
}

/* ── Dataframe / table ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] thead tr th {
    background: #f7fafc !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #4a5568 !important;
    border-bottom: 1px solid #e2e8f0 !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: #ebf4ff !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.15s ease !important;
    border: 1px solid #e2e8f0 !important;
}
[data-testid="stButton"] > button:hover {
    border-color: #1a73e8 !important;
    color: #1a73e8 !important;
    background: #ebf4ff !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
    background: #f7fafc !important;
}

/* ── Alert / info / warning banners ─────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-size: 0.88rem !important;
}

/* ── Chat messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem;
}

/* ── Caption text ────────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 0.78rem !important;
    color: #718096 !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #1a73e8 !important;
}

/* ── Hide Streamlit branding ─────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_gold() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(GOLD_PATH)


@st.cache_data(ttl=300)
def load_sector_gaps() -> pd.DataFrame:
    if not SECTOR_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SECTOR_PATH)
    numeric_cols = ["people_in_need", "people_targeted", "requirements", "funding",
                    "coverage_pct", "funding_gap_pct"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_most_recent(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent year per country."""
    if df.empty or "year" not in df.columns:
        return df
    return df.loc[df.groupby("country_iso3")["year"].idxmax()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filters")

    if df.empty:
        st.sidebar.warning("No data loaded. Run the pipeline first.")
        return df

    years = sorted(df["year"].dropna().astype(int).unique(), reverse=True)
    year_choice = st.sidebar.selectbox("Year", options=["Most recent"] + [str(y) for y in years])

    if year_choice == "Most recent":
        filtered = get_most_recent(df)
    else:
        filtered = df[df["year"].astype(int) == int(year_choice)]

    regions = {
        "All": None,
        "Sub-Saharan Africa": ["SSD","SOM","COD","CAF","NER","MLI","TCD","BFA","ETH","MOZ","ZWE","ZMB","UGA","RWA","BDI","NGA","CMR","AGO","MDG","MWI","TZA","KEN","GIN","SLE","LBR"],
        "Middle East": ["YEM","SYR","IRQ","LBN","PSE","LBY"],
        "South Asia": ["AFG","PAK","BGD","NPL","MMR"],
        "East Africa": ["ETH","SOM","SSD","KEN","UGA","TZA","DJI","ERI"],
        "Latin America": ["HTI","VEN","COL","GTM","HND","SLV","NIC"],
        "North Africa / Sahel": ["LBY","SDN","SDN","TCD","NER","MLI","BFA"],
    }
    region_choice = st.sidebar.selectbox("Region", options=list(regions.keys()))
    if regions[region_choice]:
        filtered = filtered[filtered["country_iso3"].isin(regions[region_choice])]

    max_cov = st.sidebar.slider(
        "Max funding coverage (%)",
        min_value=0, max_value=100, value=100, step=5,
        help="Show only crises funded below this percentage",
    )
    filtered = filtered[filtered["coverage_pct"].fillna(0) <= max_cov]

    neglect_options = ["All", "structural", "acute", "ongoing", "improving"]
    neglect_choice = st.sidebar.selectbox(
        "Neglect type",
        options=neglect_options,
        help="structural = underfunded 3+ consecutive years; acute = newly underfunded",
    )
    if neglect_choice != "All":
        filtered = filtered[filtered["neglect_type"] == neglect_choice]

    min_pin = st.sidebar.number_input(
        "Min people in need",
        min_value=0, value=100_000, step=100_000,
        help="Filter out small crises below this threshold",
    )
    filtered = filtered[filtered["people_in_need"].fillna(0) >= min_pin]

    st.sidebar.markdown("---")

    # --- MCDM Weight Sensitivity (dm-choice-mcdm.md Swing Weighting) ---
    with st.sidebar.expander("⚖️ Weight Sensitivity (What-If)", expanded=False):
        st.caption("Swing Weighting: drag to see how gap scores change with different priorities.")
        w_gap = st.slider("Funding Gap weight", 0.1, 1.0, 0.435, 0.05, key="w_gap")
        w_need = st.slider("Need Scale weight", 0.1, 1.0, 0.348, 0.05, key="w_need")
        w_sev = st.slider("Severity weight", 0.1, 1.0, 0.217, 0.05, key="w_sev")
        total = w_gap + w_need + w_sev
        st.caption(f"Normalized: gap={w_gap/total:.2f} need={w_need/total:.2f} sev={w_sev/total:.2f}")
        if st.button("Apply custom weights", key="apply_weights"):
            import sys as _sys2
            from scipy.stats import spearmanr
            _sys2.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
            from scoring_logic import score_dataframe
            custom_weights = {"funding_gap": w_gap/total, "need_scale": w_need/total, "severity": w_sev/total}
            df_full_reloaded = load_gold()
            if not df_full_reloaded.empty:
                rescored = score_dataframe(df_full_reloaded[
                    ["country_iso3","country_name","year","people_in_need",
                     "funding_requested","funding_received","inform_severity",
                     "consecutive_years_underfunded","has_hrp","data_staleness_days"]
                ].copy(), apply_pareto=False)
                rescored["coverage_pct"] = rescored["coverage_ratio"] * 100

                # ── Spearman ρ ranking stability (CMU dm-choice-mcdm.md §3) ──
                # Compare rank order between default and custom weights
                default_ranked = df_full_reloaded.sort_values("gap_score", ascending=False).reset_index(drop=True)
                custom_ranked  = rescored.sort_values("gap_score", ascending=False).reset_index(drop=True)
                # Align on country_iso3 for fair comparison
                merged_ranks = default_ranked[["country_iso3"]].assign(default_rank=range(len(default_ranked)))
                custom_rank_map = {row["country_iso3"]: i for i, row in custom_ranked.iterrows()}
                merged_ranks["custom_rank"] = merged_ranks["country_iso3"].map(custom_rank_map)
                merged_ranks = merged_ranks.dropna()
                rho, p_val = spearmanr(merged_ranks["default_rank"], merged_ranks["custom_rank"])

                # Flag which top-10 crises changed rank position
                default_top10 = set(default_ranked.head(10)["country_iso3"])
                custom_top10  = set(custom_ranked.head(10)["country_iso3"])
                rank_changers = default_top10.symmetric_difference(custom_top10)

                rescored["rank_changed"] = rescored["country_iso3"].isin(rank_changers)
                st.session_state["custom_scored"] = rescored
                st.session_state["rank_stability"] = {
                    "rho": round(float(rho), 3),
                    "rank_changers": sorted(rank_changers),
                }
                st.success(f"Rescored {len(rescored)} crises with custom weights.")
        if "custom_scored" in st.session_state:
            cs = st.session_state["custom_scored"]

            # ── Ranking stability badge ───────────────────────────────────────
            stab = st.session_state.get("rank_stability", {})
            if stab:
                rho = stab["rho"]
                changers = stab["rank_changers"]
                if rho >= 0.90:
                    stability_label = f"✅ High (ρ = {rho:.2f}) — top-10 order is robust"
                elif rho >= 0.70:
                    stability_label = f"⚠️ Moderate (ρ = {rho:.2f}) — some rank shifts"
                else:
                    stability_label = f"🔴 Low (ρ = {rho:.2f}) — ranking changes significantly"
                st.info(f"**Ranking stability:** {stability_label}")
                if changers:
                    st.caption(f"Top-10 entries that changed: {', '.join(changers)}")

            display_cs = cs[["country_name","gap_score","coverage_ratio","rank_changed"]].head(10).copy()
            display_cs = display_cs.rename(columns={
                "country_name":"Country","gap_score":"Gap Score",
                "coverage_ratio":"Coverage","rank_changed":"Rank ↕"
            })
            display_cs["Coverage"] = display_cs["Coverage"].map("{:.1%}".format)
            display_cs["Rank ↕"]  = display_cs["Rank ↕"].map(lambda x: "↕" if x else "")
            st.dataframe(display_cs, use_container_width=True, height=280)
            st.caption("↕ = this crisis moved in/out of top 10 vs default weights")
            if st.button("Clear custom weights", key="clear_weights"):
                del st.session_state["custom_scored"]
                st.session_state.pop("rank_stability", None)

    # --- Utility Curve / Risk Preference (CMU dm-uncertainty.md — Expected Utility) ---
    with st.sidebar.expander("🎯 Risk Preference (Utility Curve)", expanded=False):
        st.caption(
            "**How should funding be concentrated?**\n\n"
            "Risk-averse managers prefer spreading funds across many crises (diminishing returns). "
            "Risk-seeking managers concentrate on the highest-gap crises."
        )
        alpha = st.slider(
            "Risk preference (α)",
            min_value=0.3, max_value=3.0, value=1.0, step=0.1,
            key="utility_alpha",
            help=(
                "α < 1 → risk-averse: compress high scores, rank more equitably\n"
                "α = 1 → risk-neutral: standard gap score ranking\n"
                "α > 1 → risk-seeking: amplify top scores, concentrate on #1 crisis"
            ),
        )
        col_a, col_b, col_c = st.sidebar.columns(3)
        col_a.caption("🔵 Averse\nα < 1")
        col_b.caption("⚪ Neutral\nα = 1")
        col_c.caption("🔴 Seeking\nα > 1")

        if alpha != 1.0 and not filtered.empty:
            # U(gap_score) = gap_score^α  — power utility transform
            util_df = filtered.copy()
            util_df["utility_score"] = util_df["gap_score"].clip(lower=0) ** alpha
            util_df = util_df.sort_values("utility_score", ascending=False).reset_index(drop=True)

            # Rank shift vs standard gap_score ranking
            std_order  = filtered["country_iso3"].tolist()
            util_order = util_df["country_iso3"].tolist()
            rank_shifts = {
                iso: std_order.index(iso) - util_order.index(iso)
                for iso in util_order if iso in std_order
            }

            display_u = util_df[["country_name", "gap_score", "utility_score"]].head(10).copy()
            display_u["Rank Shift"] = display_u["country_iso3"].map(rank_shifts) if "country_iso3" in display_u.columns else 0
            display_u = util_df.head(10)[["country_name", "gap_score", "utility_score"]].copy()
            display_u["Rank Δ"] = [
                rank_shifts.get(iso, 0) for iso in util_df.head(10)["country_iso3"]
            ]
            display_u["Rank Δ"] = display_u["Rank Δ"].map(
                lambda d: f"▲{d}" if d > 0 else (f"▼{abs(d)}" if d < 0 else "—")
            )
            display_u = display_u.rename(columns={
                "country_name": "Country",
                "gap_score": "Raw Gap",
                "utility_score": f"Utility (α={alpha})",
            })
            display_u["Raw Gap"] = display_u["Raw Gap"].map("{:.3f}".format)
            display_u[f"Utility (α={alpha})"] = display_u[f"Utility (α={alpha})"].map("{:.3f}".format)
            st.dataframe(display_u, use_container_width=True, height=280, hide_index=True)

            if alpha < 1.0:
                st.caption(
                    f"**Risk-averse (α={alpha}):** Top scores compressed — ranking favours "
                    "spreading allocation across more crises rather than concentrating on #1."
                )
            else:
                st.caption(
                    f"**Risk-seeking (α={alpha}):** Top scores amplified — ranking reinforces "
                    "concentrating funds on the highest-gap crises."
                )

            # Store utility-adjusted df so render_table can optionally use it
            st.session_state["utility_df"] = util_df
            st.session_state["utility_alpha"] = alpha
        else:
            st.session_state.pop("utility_df", None)
            if alpha == 1.0:
                st.caption("Move the slider to apply a utility transform to the ranking.")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data sources: HDX HNO, FTS (OCHA), CBPF, INFORM Severity Index. "
        "⚠️ Low-confidence rows indicate missing or stale (>18mo) data."
    )

    return filtered.sort_values("gap_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def render_metrics(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Crises shown", len(df))
    total_pin = df["people_in_need"].fillna(0).sum()
    col2.metric("Total people in need", f"{total_pin/1e6:.1f}M")
    structural = (df["neglect_type"] == "structural").sum()
    col3.metric("Structural neglect", structural)
    no_hrp = (~df["has_hrp"].fillna(False)).sum()
    col4.metric("No HRP in place", no_hrp)


# ---------------------------------------------------------------------------
# Ranked table
# ---------------------------------------------------------------------------

NEGLECT_COLORS = {
    "structural": "#c0392b",
    "acute": "#e67e22",
    "ongoing": "#f39c12",
    "improving": "#27ae60",
}


def render_table(df: pd.DataFrame):
    # ── Utility curve override (CMU dm-uncertainty.md — Expected Utility Theory) ──
    alpha = st.session_state.get("utility_alpha", 1.0)
    utility_df = st.session_state.get("utility_df")
    if utility_df is not None and alpha != 1.0:
        df = utility_df   # use utility-adjusted ranking
        pref = "risk-averse (spread)" if alpha < 1.0 else "risk-seeking (concentrate)"
        st.info(
            f"📐 **Utility mode active (α = {alpha}, {pref})** — "
            "ranking reflects your risk preference, not raw gap score. "
            "Reset α = 1.0 in the sidebar to restore default ranking.",
            icon="🎯",
        )
        st.subheader(f"Ranked Crises — Utility Score (α = {alpha})")
    else:
        st.subheader("Ranked Crises by Gap Score")

    # Build Gap Score display with confidence interval if available
    def fmt_gap_score(row):
        mid = row.get("gap_score_mid") or row.get("gap_score")
        lo = row.get("gap_score_low")
        hi = row.get("gap_score_high")
        if pd.notna(lo) and pd.notna(hi) and lo != hi:
            return f"{mid:.3f} [{lo:.3f}–{hi:.3f}]"
        return f"{mid:.3f}" if pd.notna(mid) else "N/A"

    display_df = df.copy()
    display_df["Gap Score (CI)"] = display_df.apply(fmt_gap_score, axis=1)

    # Pareto flag — only highlight non-dominated (frontier) crises; leave dominated blank
    if "pareto_dominated" in display_df.columns:
        display_df["Pareto"] = display_df["pareto_dominated"].map(
            lambda x: "" if x else "★ frontier"
        )

    display_cols = {
        "country_name": "Country",
        "year": "Year",
        "Gap Score (CI)": "Gap Score [CI]",
        "coverage_pct": "Coverage %",
        "people_in_need": "People in Need",
        "neglect_type": "Neglect Type",
        "consecutive_years_underfunded": "Yrs Underfunded",
        "has_hrp": "Has HRP",
        "Pareto": "Pareto",
        "low_confidence": "⚠️ Low Conf.",
    }
    table = display_df[[c for c in display_cols if c in display_df.columns]].rename(columns=display_cols)
    table.insert(0, "Rank", range(1, len(table) + 1))

    if "People in Need" in table.columns:
        table["People in Need"] = table["People in Need"].apply(
            lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else (f"{x:,.0f}" if pd.notna(x) else "N/A")
        )
    if "Coverage %" in table.columns:
        def fmt_coverage(row):
            cov = row.get("Coverage %")
            hrp = row.get("Has HRP", True)
            if pd.isna(cov):
                return "N/A"
            if cov == 0 and not hrp:
                return "0%*"   # assumption, not measured
            return f"{cov:.1f}%"
        # Apply row-wise to access both columns
        if "Has HRP" in table.columns:
            table["Coverage %"] = table.apply(fmt_coverage, axis=1)
        else:
            table["Coverage %"] = table["Coverage %"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )

    st.dataframe(table, use_container_width=True, height=380)

    # Note on 0%* assumption
    if display_df["coverage_pct"].eq(0).any() and (~display_df.get("has_hrp", pd.Series([True]*len(display_df))).fillna(True)).any():
        st.caption("\\* **0%** = no Humanitarian Response Plan (HRP) on record — funding coverage is assumed 0%, not measured. Treat these rows as lower-bound estimates only.")

    # CI explanation
    with st.expander("What is the confidence interval [CI]?", expanded=False):
        st.caption(
            "The gap score CI is computed via **Monte Carlo simulation** (1 000 runs per crisis). "
            "Inputs follow triangular distributions: PIN ±30%, funding received ±20% — "
            "reflecting typical humanitarian data uncertainty (HNO assessments are routinely 20–30% off). "
            "The CI shows the **P10–P90 range** across simulated scenarios. "
            "A wide CI means ranking position is highly sensitive to data quality. "
            "Source: CMU Decision Modeling — dm-uncertainty.md §2 Monte Carlo."
        )


# ---------------------------------------------------------------------------
# Choropleth map
# ---------------------------------------------------------------------------

def render_map(df: pd.DataFrame):
    st.subheader("Gap Score World Map")

    if df.empty or "country_iso3" not in df.columns:
        st.info("No data to display on map.")
        return

    map_df = df.copy()
    map_df["gap_score_display"] = map_df["gap_score"].fillna(0).round(3)
    map_df["hover_text"] = map_df.apply(
        lambda r: (
            f"{r.get('country_name', r.get('country_iso3', ''))}<br>"
            f"Gap Score: {r.get('gap_score', 0):.3f}<br>"
            f"Coverage: {r.get('coverage_pct', 0):.1f}%<br>"
            f"PIN: {r.get('people_in_need', 0)/1e6:.1f}M<br>"
            f"Neglect: {r.get('neglect_type', 'N/A')}"
            + (" ⚠️" if r.get("low_confidence") else "")
        ),
        axis=1,
    )

    fig = px.choropleth(
        map_df,
        locations="country_iso3",
        color="gap_score_display",
        hover_name="country_name",
        hover_data={"country_iso3": False, "gap_score_display": False, "hover_text": True},
        color_continuous_scale=[
            [0.0, "#2ecc71"],   # well-funded: green
            [0.3, "#f1c40f"],   # moderate gap: yellow
            [0.6, "#e67e22"],   # high gap: orange
            [1.0, "#c0392b"],   # severe gap: red
        ],
        range_color=[0, map_df["gap_score_display"].quantile(0.95)],
        labels={"gap_score_display": "Gap Score"},
        title="Humanitarian Funding Gap Score (higher = more overlooked)",
    )
    fig.update_layout(
        height=450,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Gap Score", tickformat=".2f"),
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
    )
    # Override hover to use our custom text
    fig.update_traces(
        hovertemplate="%{customdata[2]}<extra></extra>",
        customdata=map_df[["country_iso3", "gap_score_display", "hover_text"]].values,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sector drilldown bar chart (bonus — if sector data available)
# ---------------------------------------------------------------------------

def render_sector_drilldown(df_full: pd.DataFrame, selected_country: str):
    if "sector" not in df_full.columns:
        return
    sector_df = df_full[df_full["country_iso3"] == selected_country].copy()
    if sector_df.empty:
        return
    sector_df = sector_df.groupby("sector", as_index=False)["people_in_need"].sum()
    sector_df = sector_df.sort_values("people_in_need", ascending=False).head(10)
    fig = px.bar(
        sector_df, x="sector", y="people_in_need",
        title=f"People in Need by Sector — {selected_country}",
        labels={"people_in_need": "People in Need", "sector": "Sector"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sector Analysis — country × cluster funding gaps (from Databricks FTS join)
# ---------------------------------------------------------------------------

# Human-readable labels for HNO cluster codes
_CLUSTER_LABELS = {
    "FSC":     "Food Security",
    "HEA":     "Health",
    "WSH":     "Water, Sanitation & Hygiene",
    "SHL":     "Shelter",
    "NUT":     "Nutrition",
    "EDU":     "Education",
    "PRO":     "Protection",
    "PRO-GBV": "Protection / GBV",
    "PRO-CPN": "Protection / Child",
    "AGR":     "Agriculture / Livelihoods",
    "CCM":     "Coordination",
}


def render_sector_analysis():
    """
    Bonus task: expose sector-level gaps so analysts can see cases where a
    high aggregate funding figure masks severe cluster-level shortfalls.

    Data: data/gold/sector_funding_gaps.csv — 51 rows, country × HNO cluster.
    Produced by joining official HNO 2024 data with FTS cluster funding data
    from the Databricks Unity Catalog volume (cmu_hackathon.common.unocha).
    """
    df = load_sector_gaps()

    if df.empty:
        st.info(
            "Sector gap data not found. "
            "Expected: `data/gold/sector_funding_gaps.csv` "
            "(produced by the Databricks sector-analysis notebook)."
        )
        return

    st.subheader("Sector-Level Funding Gaps")
    st.caption(
        "Each row is a **country × sector** combination. "
        "A country may appear well-funded in aggregate while individual clusters remain severely underfunded. "
        "Data: official HNO 2024 joined with FTS cluster allocations from OCHA Databricks workspace."
    )

    # ---- Filters ----
    col_f1, col_f2, col_f3 = st.columns(3)

    all_clusters = sorted(df["hno_cluster"].dropna().unique())
    cluster_labels = {c: _CLUSTER_LABELS.get(c, c) for c in all_clusters}
    with col_f1:
        selected_clusters = st.multiselect(
            "Filter by sector",
            options=all_clusters,
            default=[],
            format_func=lambda c: cluster_labels.get(c, c),
            placeholder="All sectors",
        )

    all_countries = sorted(df["country_iso3"].dropna().unique())
    with col_f2:
        selected_countries = st.multiselect(
            "Filter by country",
            options=all_countries,
            default=[],
            placeholder="All countries",
        )

    with col_f3:
        max_cov_sector = st.slider(
            "Max coverage % shown",
            min_value=0, max_value=100, value=100, step=5,
            key="sector_cov_filter",
            help="Show only sector-country pairs funded below this percentage",
        )

    df_view = df.copy()
    if selected_clusters:
        df_view = df_view[df_view["hno_cluster"].isin(selected_clusters)]
    if selected_countries:
        df_view = df_view[df_view["country_iso3"].isin(selected_countries)]
    df_view = df_view[df_view["coverage_pct"].fillna(0) <= max_cov_sector]
    df_view = df_view.sort_values("funding_gap_pct", ascending=False).reset_index(drop=True)

    if df_view.empty:
        st.info("No sector-country combinations match the current filters.")
        return

    # ---- Top sector gaps bar chart ----
    top_n = min(20, len(df_view))
    top_df = df_view.head(top_n).copy()
    top_df["label"] = top_df["country_iso3"] + " / " + top_df["hno_cluster"].map(
        lambda c: _CLUSTER_LABELS.get(c, c)
    )
    top_df["pin_m"] = top_df["people_in_need"].div(1e6) if "people_in_need" in top_df.columns else pd.Series(0.0, index=top_df.index)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Funding Gap %",
        x=top_df["label"],
        y=top_df["funding_gap_pct"],
        marker=dict(
            color=top_df["funding_gap_pct"],
            colorscale=[[0, "#f1c40f"], [0.5, "#e67e22"], [1.0, "#c0392b"]],
            cmin=50, cmax=100,
            showscale=True,
            colorbar=dict(title="Gap %", ticksuffix="%"),
        ),
        text=top_df["funding_gap_pct"].map(lambda v: f"{v:.0f}%"),
        textposition="outside",
        customdata=top_df[["country_iso3", "hno_cluster", "pin_m", "coverage_pct",
                            "requirements", "funding"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Country: %{customdata[0]}<br>"
            "Cluster: %{customdata[1]}<br>"
            "People in Need: %{customdata[2]:.1f}M<br>"
            "Coverage: %{customdata[3]:.1f}%<br>"
            "Required: $%{customdata[4]:,.0f}<br>"
            "Funded: $%{customdata[5]:,.0f}<br>"
            "<extra></extra>"
        ),
    ))
    fig_bar.update_layout(
        title=f"Top {top_n} Worst-Funded Country × Sector Combinations",
        xaxis=dict(tickangle=-40, title=""),
        yaxis=dict(title="Funding Gap (%)", range=[0, 115]),
        height=420,
        margin=dict(t=50, b=120, l=40, r=40),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Heatmap: coverage % by country × cluster ----
    st.markdown("#### Coverage Heatmap — Country × Sector")
    st.caption("Green = funded; Red = severe gap. Grey = no data for that combination.")

    pivot = df_view.pivot_table(
        index="country_iso3", columns="hno_cluster",
        values="coverage_pct", aggfunc="mean",
    )
    pivot.columns = [_CLUSTER_LABELS.get(c, c) for c in pivot.columns]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[
            [0.0, "#c0392b"],   # 0%  → red
            [0.3, "#e67e22"],   # 30% → orange
            [0.6, "#f1c40f"],   # 60% → yellow
            [1.0, "#2ecc71"],   # 100% → green
        ],
        zmin=0, zmax=100,
        colorbar=dict(title="Coverage %", ticksuffix="%"),
        hoverongaps=False,
        hovertemplate=(
            "Country: %{y}<br>"
            "Sector: %{x}<br>"
            "Coverage: %{z:.1f}%<br>"
            "<extra></extra>"
        ),
    ))
    fig_heat.update_layout(
        height=max(300, len(pivot) * 28 + 80),
        margin=dict(t=20, b=40, l=60, r=40),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ---- Summary metrics ----
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Sector-country pairs", len(df_view))
    if "people_in_need" in df_view.columns:
        total_pin_sector = df_view["people_in_need"].fillna(0).sum()
        col_m2.metric("Total PIN (sectors)", f"{total_pin_sector/1e6:.1f}M")
    else:
        col_m2.metric("Total requirements", f"${df_view['requirements'].fillna(0).sum()/1e6:.0f}M")
    severe = (df_view["funding_gap_pct"] >= 80).sum()
    col_m3.metric("≥80% unfunded pairs", int(severe))
    zero_funded = (df_view["funding"].fillna(0) == 0).sum()
    col_m4.metric("Zero funding received", int(zero_funded))

    # ---- Raw table ----
    with st.expander("Raw sector gap table", expanded=False):
        raw_cols = [c for c in ["country_iso3", "hno_cluster", "people_in_need", "people_targeted",
                                "requirements", "funding", "coverage_pct", "funding_gap_pct"]
                    if c in df_view.columns]
        display_sector = df_view[raw_cols].copy()
        display_sector["hno_cluster"] = display_sector["hno_cluster"].map(
            lambda c: f"{c} — {_CLUSTER_LABELS.get(c, c)}"
        )
        if "people_in_need" in display_sector.columns:
            display_sector["people_in_need"] = display_sector["people_in_need"].apply(
                lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else f"{x:,.0f}" if pd.notna(x) else "N/A"
            )
        display_sector["requirements"] = display_sector["requirements"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        display_sector["funding"] = display_sector["funding"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
        )
        display_sector["coverage_pct"] = display_sector["coverage_pct"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        display_sector["funding_gap_pct"] = display_sector["funding_gap_pct"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        col_rename = {
            "country_iso3": "Country", "hno_cluster": "Sector",
            "people_in_need": "People in Need", "people_targeted": "People Targeted",
            "requirements": "Requirements", "funding": "Funding Received",
            "coverage_pct": "Coverage %", "funding_gap_pct": "Funding Gap %",
        }
        display_sector.rename(columns={k: v for k, v in col_rename.items() if k in display_sector.columns}, inplace=True)
        display_sector.insert(0, "#", range(1, len(display_sector) + 1))
        st.dataframe(display_sector, use_container_width=True, hide_index=True)

    st.caption(
        "Source: official HNO 2024 data + FTS cluster allocations 2024 "
        "(joined in Databricks Unity Catalog, cmu_hackathon.common.unocha). "
        "FTS cluster names mapped to HNO codes via keyword matching. "
        "Aggregated by groupby(countryCode, hno_cluster)."
    )


# ---------------------------------------------------------------------------
# EVPI Panel — Expected Value of Perfect Information (dm-uncertainty.md §1)
# ---------------------------------------------------------------------------

def _render_evpi_panel(df: pd.DataFrame):
    """
    Show how much ranking uncertainty would decrease if PIN data improved
    from ±30% to ±5% precision (CMU dm-uncertainty.md — EVPI concept).

    Helps OCHA analysts decide whether commissioning a more detailed HNO is worth it.
    """
    with st.expander("📊 Value of Perfect Information (EVPI) — Is better data worth it?", expanded=False):
        st.caption(
            "EVPI answers: *'If PIN uncertainty improved from ±30% to ±5%, how much would "
            "the gap-score ranking ambiguity decrease?'* "
            "High EVPI → a more detailed Humanitarian Needs Overview (HNO) would meaningfully "
            "clarify this crisis's ranking. Low EVPI → current data is already decisive."
        )

        if df.empty:
            st.info("No data to analyse.")
            return

        required_cols = {"people_in_need", "funding_received", "funding_requested",
                         "inform_severity", "consecutive_years_underfunded"}
        if not required_cols.issubset(df.columns):
            st.info("EVPI requires full crisis data columns.")
            return

        sys_path_added = False
        try:
            import sys as _sys
            _pkg = str(Path(__file__).resolve().parent.parent / "core")
            if _pkg not in _sys.path:
                _sys.path.insert(0, _pkg)
                sys_path_added = True
            from scoring_logic import compute_evpi

            top5 = df.head(5).copy()
            ref_pin = float(df["people_in_need"].quantile(0.95)) or 1e7

            rows = []
            for _, row in top5.iterrows():
                evpi = compute_evpi(
                    people_in_need=float(row.get("people_in_need", 0)),
                    funding_received=float(row.get("funding_received", 0)),
                    funding_requested=float(row.get("funding_requested", 0)),
                    inform_severity=row.get("inform_severity"),
                    consecutive_years_underfunded=int(row.get("consecutive_years_underfunded", 0)),
                    reference_pin=ref_pin,
                    n_simulations=300,
                )
                rows.append({
                    "country": row.get("country_name", row.get("country_iso3", "?")),
                    "current_range": evpi["current_range"],
                    "improved_range": evpi["improved_range"],
                    "evpi_pct": evpi["evpi_pct_reduction"],
                    "interpretation": evpi["interpretation"],
                })

            evpi_df = pd.DataFrame(rows)

            # Plotly grouped bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Current (±30% PIN, ±20% funding)",
                x=evpi_df["country"],
                y=evpi_df["current_range"],
                marker_color="#e67e22",
                text=evpi_df["current_range"].map(lambda v: f"±{v:.3f}"),
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="If PIN improved to ±5%",
                x=evpi_df["country"],
                y=evpi_df["improved_range"],
                marker_color="#27ae60",
                text=evpi_df["improved_range"].map(lambda v: f"±{v:.3f}"),
                textposition="outside",
            ))
            fig.update_layout(
                barmode="group",
                title="Gap Score P10–P90 Range: Current vs. Improved Data Quality",
                yaxis_title="Score Uncertainty Range (P90 − P10)",
                xaxis_title="",
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            summary = evpi_df[["country", "evpi_pct", "interpretation"]].rename(columns={
                "country": "Country",
                "evpi_pct": "Uncertainty Reduction %",
                "interpretation": "Verdict",
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.caption(
                "Monte Carlo: 300 simulations per crisis. "
                "Triangular distribution: PIN (±30% current / ±5% improved), funding (±20% current / ±5% improved). "
                "Source: CMU Decision Modeling — dm-uncertainty.md §1 EVPI."
            )

        except ImportError as e:
            st.warning(f"scoring_logic not available: {e}")


# ---------------------------------------------------------------------------
# Conversational agent interface
# ---------------------------------------------------------------------------

@st.fragment
def render_chat(df_filtered: pd.DataFrame):
    st.subheader("Ask the Command Center")

    if "ANTHROPIC_API_KEY" not in os.environ or not os.environ["ANTHROPIC_API_KEY"]:
        st.warning("Set ANTHROPIC_API_KEY in your .env file to enable the conversational agent.")
        return

    # Lazy import — ensure the hackathon directory is on sys.path
    import sys
    _project_root = Path(__file__).resolve().parent.parent
    _agent_dir = str(_project_root / "agent")
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    try:
        from agent_runner import run_query
    except ImportError:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("agent04", _project_root / "agent" / "04_agent.py")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        run_query = _mod.run_query

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    EXAMPLE_QUERIES = [
        "Which crises have the highest PIN but lowest funding?",
        "Show structurally neglected crises in Sub-Saharan Africa.",
        "Are there countries with active HRPs but under 10% funded?",
        "Which crises have been underfunded for 3+ consecutive years?",
        "Find crises with a similar profile to Yemen — high severity, low coverage.",
        "Which overlooked crises should a pooled fund manager prioritise this year?",
    ]

    with st.expander("Example queries", expanded=False):
        for q in EXAMPLE_QUERIES:
            if st.button(q, key=f"eq_{q[:20]}"):
                st.session_state.pending_query = q

    user_input = st.chat_input("Ask about overlooked crises …")
    if "pending_query" in st.session_state:
        user_input = st.session_state.pop("pending_query")

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing humanitarian data …"):
                try:
                    result        = run_query(user_input)
                    response      = result["response_text"]
                    confidence    = result.get("confidence_level", "medium")
                    if result.get("query_rationale"):
                        response += f"\n\n---\n*Query interpretation: {result['query_rationale']}*"
                    agent_error   = None
                except Exception as e:
                    response      = f"Error running agent: {e}\n\nMake sure the Gold table exists (run 03_gold_scoring.py first)."
                    confidence    = "low"
                    agent_error   = str(e)

            # ── Low-confidence escalation (CMU agentic-tech.md — human checkpoint) ──
            _response_contains_warning = (
                "LOW CONFIDENCE" in response or "data quality warning" in response.lower()
            )
            if confidence == "low" or _response_contains_warning:
                st.warning(
                    "⚠️ **Low-confidence response** — the agent flagged data quality issues "
                    "with this result. Figures may be based on stale or incomplete HNO/FTS data. "
                    "Verify against the latest HRP report before using in allocation decisions.",
                    icon="⚠️",
                )
                _conf_badge = "🔴 LOW"
            elif confidence == "medium":
                _conf_badge = "🟡 MEDIUM"
            else:
                _conf_badge = "🟢 HIGH"

            st.markdown(response)
            st.caption(f"Confidence: {_conf_badge}")

            # ── Feedback button → eval_feedback.jsonl (CMU aimd-scaling-evaluation.md) ──
            # Negative traces automatically become new eval golden cases.
            _msg_idx = len(st.session_state.chat_history)
            _fb_key  = f"thumbsdown_{_msg_idx}"
            if st.button("👎 Flag this response", key=_fb_key, help="Mark as incorrect / unhelpful — adds to eval dataset"):
                _feedback_path = _HERE / "data" / "eval_feedback.jsonl"
                _feedback_path.parent.mkdir(parents=True, exist_ok=True)
                _entry = {
                    "ts":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "query":    user_input,
                    "response": response[:2000],
                    "confidence": confidence,
                    "reason":   "user_flagged",
                }
                with open(_feedback_path, "a") as _f:
                    _f.write(json.dumps(_entry) + "\n")
                st.success("Flagged — this response will be included in the next evaluation run.")

            st.session_state.chat_history.append({
                "role":       "assistant",
                "content":    response,
                "confidence": confidence,
            })


# ---------------------------------------------------------------------------
# Data quality drift alert (oai-monitoring-governance.md)
# ---------------------------------------------------------------------------

def _render_data_quality_alert(df: pd.DataFrame):
    """Show banner if significant share of data is stale or low-confidence."""
    if df.empty or "data_staleness_days" not in df.columns:
        return
    stale_pct = (df["data_staleness_days"].fillna(0) > 365).mean() * 100
    low_conf_pct = df.get("low_confidence", pd.Series([False]*len(df))).fillna(False).mean() * 100
    if stale_pct > 30 or low_conf_pct > 40:
        st.warning(
            f"⚠️ **Data Quality Alert** — {stale_pct:.0f}% of loaded crises have data older than 12 months "
            f"and {low_conf_pct:.0f}% are flagged low-confidence. "
            "Gap scores may reflect outdated conditions. Interpret rankings with caution and verify against latest HRP reports.",
            icon="🕐",
        )
    elif stale_pct > 10:
        st.info(
            f"ℹ️ {stale_pct:.0f}% of crises have data older than 12 months. "
            "Low-confidence rows are marked ⚠️ in the ranked table.",
            icon="📅",
        )


# ---------------------------------------------------------------------------
# Evidently Data Drift Analysis (oai-monitoring-governance.md — Evidently)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _compute_drift(df: pd.DataFrame) -> dict:
    """
    Run Evidently DataDriftPreset comparing 2024 (reference) vs 2025 (current).
    Returns a dict with per-column drift results and summary stats.
    Cached so it doesn't re-run on every widget interaction.
    """
    DRIFT_COLS_NUM = ["gap_score", "coverage_pct", "people_in_need", "inform_severity"]
    DRIFT_COLS_CAT = ["neglect_type"]

    ref = df[df["year"] == 2024][DRIFT_COLS_NUM + DRIFT_COLS_CAT].reset_index(drop=True)
    cur = df[df["year"] == 2025][DRIFT_COLS_NUM + DRIFT_COLS_CAT].reset_index(drop=True)

    if ref.empty or cur.empty:
        return {}

    try:
        from evidently import Dataset, DataDefinition
        from evidently.presets import DataDriftPreset
        from evidently import Report

        definition = DataDefinition(
            numerical_columns=DRIFT_COLS_NUM,
            categorical_columns=DRIFT_COLS_CAT,
        )
        ref_ds = Dataset.from_pandas(ref, data_definition=definition)
        cur_ds = Dataset.from_pandas(cur, data_definition=definition)

        report = Report([DataDriftPreset()])
        snap   = report.run(reference_data=ref_ds, current_data=cur_ds)
        d      = snap.dict()

        # Parse metric results
        col_results = {}
        drifted_count = 0
        for m in d["metrics"]:
            name = m.get("metric_name", "")
            val  = m.get("value")
            if "ValueDrift" in name and "column=" in name:
                # e.g. "ValueDrift(column=gap_score,method=K-S p_value,threshold=0.05)"
                col = name.split("column=")[1].split(",")[0]
                p   = float(val) if val is not None else None
                drifted = p is not None and p < 0.05
                if drifted:
                    drifted_count += 1
                col_results[col] = {"p_value": p, "drifted": drifted}
            elif "DriftedColumnsCount" in name and isinstance(val, dict):
                drifted_count = int(val.get("count", 0))

        # Raw distributions for plotting
        distributions = {
            col: {
                "ref_mean":  float(ref[col].mean()),
                "cur_mean":  float(cur[col].mean()),
                "ref_median":float(ref[col].median()),
                "cur_median":float(cur[col].median()),
                "ref_values":ref[col].dropna().tolist(),
                "cur_values":cur[col].dropna().tolist(),
            }
            for col in DRIFT_COLS_NUM
        }

        return {
            "col_results":    col_results,
            "drifted_count":  drifted_count,
            "total_cols":     len(DRIFT_COLS_NUM) + len(DRIFT_COLS_CAT),
            "distributions":  distributions,
            "ref_year":       2024,
            "cur_year":       2025,
            "ref_n":          len(ref),
            "cur_n":          len(cur),
        }
    except Exception as e:
        return {"error": str(e)}


def render_drift_analysis(df: pd.DataFrame):
    """
    Evidently DataDrift tab: compare 2024 vs 2025 gold data.
    Shows which humanitarian metrics have shifted significantly year-over-year.
    (CMU oai-monitoring-governance.md — Evidently, data drift monitoring)
    """
    st.subheader("📊 Data Drift Analysis — 2024 → 2025")
    st.caption(
        "Evidently DataDriftPreset compares the **2024 reference cohort** vs **2025 current cohort** "
        "of humanitarian crisis data. Significant drift in `gap_score` or `coverage_pct` signals "
        "that the funding landscape has shifted — scores from last year may no longer reflect current reality."
    )

    drift = _compute_drift(df)

    if not drift:
        st.info("Drift analysis requires data for both 2024 and 2025. Check that the Gold table has multi-year data.")
        return
    if "error" in drift:
        st.error(f"Evidently error: {drift['error']}")
        return

    # ── Summary header ──────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reference year", drift["ref_year"], f"n={drift['ref_n']} crises")
    col_b.metric("Current year",   drift["cur_year"], f"n={drift['cur_n']} crises")
    drifted = drift["drifted_count"]
    total   = drift["total_cols"]
    col_c.metric(
        "Columns drifted",
        f"{drifted}/{total}",
        delta="⚠️ significant" if drifted > 0 else "✅ stable",
        delta_color="inverse" if drifted > 0 else "normal",
    )

    if drifted > 0:
        st.warning(
            f"**{drifted} of {total} tracked columns show statistically significant drift** "
            "(Kolmogorov-Smirnov p < 0.05). This means the humanitarian data distribution "
            "changed meaningfully between 2024 and 2025 — rankings from prior-year analysis "
            "may no longer be valid.",
            icon="📉",
        )
    else:
        st.success("No significant drift detected — data distributions are stable year-over-year.", icon="✅")

    # ── Per-column drift table ───────────────────────────────────────────────
    st.markdown("#### Drift by Column")
    col_labels = {
        "gap_score":       "Gap Score",
        "coverage_pct":    "Funding Coverage %",
        "people_in_need":  "People in Need",
        "inform_severity": "INFORM Severity",
        "neglect_type":    "Neglect Type (categorical)",
    }
    rows = []
    for col, res in drift["col_results"].items():
        p   = res["p_value"]
        rows.append({
            "Metric":   col_labels.get(col, col),
            "Method":   "chi-square" if col == "neglect_type" else "K-S test",
            "p-value":  f"{p:.4f}" if p is not None else "N/A",
            "Drifted":  "⚠️ YES" if res["drifted"] else "✅ No",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Distribution comparison plots for numeric columns ───────────────────
    st.markdown("#### Distribution Shift — Reference (2024) vs Current (2025)")

    dists = drift.get("distributions", {})
    plot_cols = [c for c in ["gap_score", "coverage_pct", "people_in_need", "inform_severity"] if c in dists]

    for col in plot_cols:
        d_info   = dists[col]
        is_drift = drift["col_results"].get(col, {}).get("drifted", False)
        label    = col_labels.get(col, col)
        drift_tag = " ⚠️ DRIFTED" if is_drift else " ✅ stable"

        with st.expander(f"{label}{drift_tag}", expanded=is_drift):
            ref_vals = d_info["ref_values"]
            cur_vals = d_info["cur_values"]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=ref_vals, name=f"2024 (ref, n={len(ref_vals)})",
                opacity=0.65, marker_color="#3498db",
                histnorm="probability density",
            ))
            fig.add_trace(go.Histogram(
                x=cur_vals, name=f"2025 (cur, n={len(cur_vals)})",
                opacity=0.65, marker_color="#e74c3c",
                histnorm="probability density",
            ))
            fig.update_layout(
                barmode="overlay",
                height=260,
                margin=dict(t=20, b=20, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title=label,
                yaxis_title="Density",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mean shift callout
            ref_m = d_info["ref_mean"]
            cur_m = d_info["cur_mean"]
            delta = cur_m - ref_m
            pct   = (delta / ref_m * 100) if ref_m != 0 else 0
            st.caption(
                f"Mean: {ref_m:.3f} (2024) → {cur_m:.3f} (2025) "
                f"({'▲' if delta >= 0 else '▼'} {abs(pct):.1f}%)"
            )

    # ── Interpretation for analysts ─────────────────────────────────────────
    with st.expander("What does this mean for fund managers?", expanded=False):
        st.markdown("""
**Drifted columns signal that the humanitarian landscape changed between 2024 and 2025:**

- **`gap_score` drift** (p < 0.05): The overall funding-gap distribution shifted. Countries that were
  most underfunded in 2024 may not be the same ones in 2025. Rankings should be based on 2025 data only.

- **`coverage_pct` drift** (p < 0.05): Funding coverage ratios changed significantly across the cohort.
  This likely reflects new HRP pledges or FTS reporting updates — not necessarily improved on-the-ground conditions.

- **`people_in_need` stable** (p > 0.05): The scale of documented need has not changed significantly.
  New funding (if any) went to reducing the gap, not because need decreased.

- **`inform_severity` stable** (p ≈ 1.0): INFORM crisis severity index is essentially unchanged —
  confirming that the underlying crisis conditions are persistent, not improving.

**Decision implication:** Drift in `gap_score` and `coverage_pct` with stable `people_in_need`
and `inform_severity` is the signature of **funding volatility without condition improvement** —
the most dangerous scenario for structural neglect analysis.
        """)

    st.caption(
        "Method: Evidently DataDriftPreset v0.7+ · "
        "Numerical columns: Kolmogorov-Smirnov test (threshold p=0.05) · "
        "Categorical: chi-square test · "
        "Reference: 2024 gold cohort (n=24) · Current: 2025 gold cohort (n=22)"
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("🌍 Geo-Insight: Overlooked Humanitarian Crises")
    st.caption(
        "Decision support for **CBPF pooled fund managers** — identifies where humanitarian need "
        "significantly outpaces funding coverage, enabling evidence-based allocation prioritisation. "
        "Built for UN OCHA · Databricks Hackathon 2026 · Data: HDX HNO, FTS, CBPF, INFORM Severity Index"
    )

    df_full = load_gold()

    if df_full.empty:
        st.error(
            "No Gold data found. Please run the pipeline:\n"
            "```\n"
            "python 01_bronze_ingest.py\n"
            "python 02_silver_transform.py\n"
            "python 03_gold_scoring.py\n"
            "```"
        )
        st.stop()

    df_filtered = render_sidebar(df_full)

    # --- Data quality drift alert (oai-monitoring-governance.md concept drift) ---
    _render_data_quality_alert(df_full)

    render_metrics(df_filtered)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ranked Table", "World Map", "Sector Gaps", "Data Drift", "Ask the Agent"])

    with tab1:
        render_table(df_filtered)
        if not df_filtered.empty:
            top_iso3 = df_filtered.iloc[0]["country_iso3"]
            if "sector" in df_full.columns:
                st.markdown("---")
                render_sector_drilldown(df_full, top_iso3)
        st.markdown("---")
        _render_evpi_panel(df_filtered)

    with tab2:
        render_map(df_filtered)

    with tab3:
        render_sector_analysis()

    with tab4:
        render_drift_analysis(df_full)

    with tab5:
        render_chat(df_filtered)

    # Scoring formula transparency
    with st.expander("How is the Gap Score calculated?"):
        st.markdown("""
**Composite Gap Score — MCDM Weighted Sum (CMU Decision Modeling)**

```
coverage_ratio  = min(funding_received / funding_requested, 1.0)
funding_gap     = 1 - coverage_ratio
need_scale      = log1p(PIN) / log1p(95th-pct PIN)   ← Ringer Bid–safe reference
severity_mult   = INFORM_severity / 10
base_score      = funding_gap × (0.435 + 0.348×need_scale + 0.217×severity_mult)
neglect_factor  = 1 + (consecutive_years_underfunded × 0.15)
gap_score       = base_score × neglect_factor
```

**Weight derivation (Swing Weighting):**
Weights answer "if you could move ONE dimension from worst to best, which matters most?"
- `funding_gap` → 100 pts (most impactful: directly determines resource reach)
- `need_scale`  → 80 pts  (scale of human suffering)
- `severity`    → 50 pts  (independent urgency signal from INFORM)

**Ringer Bid prevention:** Need scale uses the 95th-percentile PIN as reference, not the global max.
This prevents one extreme crisis from warping the normalization for all others.

**Confidence interval:** ±20% sensitivity on PIN and funding received, reflecting typical
humanitarian data uncertainty. A wide CI means ranking position is data-quality sensitive.

**Pareto filter:** Crises dominated on all three primary dimensions are flagged but not removed,
ensuring transparency while keeping focus on non-dominated crises.

**Uncertainty flags ⚠️:** No HRP → 0% coverage is an assumption, not a measurement (shown as **0%**).
Data >18 months old → figures may not reflect current conditions.

**Data window note:** `consecutive_years_underfunded` is computed from available data (2024–2026 only).
A value of 3 means underfunded across all 3 years in our dataset — not necessarily across all historical years.

**⚠️ MCDM design note — preferential independence:** In a strict MCDM weighted sum, the preference
weight for criterion A must not depend on the value of criterion B (preferential independence axiom).
This formula deliberately violates that: `need_scale` and `severity` only affect the score when a
funding gap exists — a fully-funded crisis scores 0 regardless of PIN or INFORM severity. This is
an intentional design choice: *"overlooked"* requires both humanitarian need **and** a funding shortfall.
A crisis receiving 100% of requested funds is, by definition, not overlooked — even if it is large or severe.
This interaction term is appropriate for this application domain, but users should be aware that the
three dimensions are **not evaluated independently**.
        """)


if __name__ == "__main__":
    main()
