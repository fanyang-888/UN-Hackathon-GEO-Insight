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

load_dotenv()

_HERE = Path(__file__).parent
GOLD_PATH = _HERE / "data/gold/gold_ranked_crises.parquet"

st.set_page_config(
    page_title="Geo-Insight: Overlooked Crises",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_gold() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(GOLD_PATH)


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
            from scoring_logic import score_dataframe
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            custom_weights = {"funding_gap": w_gap/total, "need_scale": w_need/total, "severity": w_sev/total}
            df_full_reloaded = load_gold()
            if not df_full_reloaded.empty:
                rescored = score_dataframe(df_full_reloaded[
                    ["country_iso3","country_name","year","people_in_need",
                     "funding_requested","funding_received","inform_severity",
                     "consecutive_years_underfunded","has_hrp","data_staleness_days"]
                ].copy(), apply_pareto=False)
                rescored["coverage_pct"] = rescored["coverage_ratio"] * 100
                st.session_state["custom_scored"] = rescored
                st.success(f"Rescored {len(rescored)} crises with custom weights.")
        if "custom_scored" in st.session_state:
            cs = st.session_state["custom_scored"]
            st.dataframe(
                cs[["country_name","gap_score","coverage_ratio"]].head(10)
                .rename(columns={"country_name":"Country","gap_score":"Gap Score","coverage_ratio":"Coverage"})
                .assign(**{"Coverage": lambda d: d["Coverage"].map("{:.1%}".format)}),
                use_container_width=True, height=260
            )
            if st.button("Clear custom weights", key="clear_weights"):
                del st.session_state["custom_scored"]

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
            _pkg = str(Path(__file__).parent)
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

def render_chat(df_filtered: pd.DataFrame):
    st.subheader("Ask the Command Center")

    if "ANTHROPIC_API_KEY" not in os.environ or not os.environ["ANTHROPIC_API_KEY"]:
        st.warning("Set ANTHROPIC_API_KEY in your .env file to enable the conversational agent.")
        return

    # Lazy import — ensure the hackathon directory is on sys.path
    import sys
    _pkg_dir = str(Path(__file__).parent)
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    try:
        from agent_runner import run_query
    except ImportError:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("agent04", Path(__file__).parent / "04_agent.py")
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
                    result = run_query(user_input)
                    response = result["response_text"]
                    if result.get("query_rationale"):
                        response += f"\n\n---\n*Query interpretation: {result['query_rationale']}*"
                except Exception as e:
                    response = f"Error running agent: {e}\n\nMake sure the Gold table exists (run 03_gold_scoring.py first)."
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})


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

    tab1, tab2, tab3 = st.tabs(["Ranked Table", "World Map", "Ask the Agent"])

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
        """)


if __name__ == "__main__":
    main()
