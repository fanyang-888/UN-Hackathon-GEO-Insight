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
# Theme injection — reads st.session_state["theme"] to pick light vs dark CSS
# ---------------------------------------------------------------------------
def _inject_theme():
    dark = st.session_state.get("theme", "light") == "dark"

    if dark:
        css_vars = """
        --bg-page:       #161310;
        --bg-sidebar:    #0f0e0c;
        --bg-main:       #161310;
        --bg-table-even: #0f0e0c;
        --bg-table-head: #1c1917;
        --bg-table-hover:rgba(212,56,13,.07);
        --border:        #ea580c;
        --border-soft:   #1c1917;
        --border-sidebar:#2c2420;
        --txt-primary:   #faf7f2;
        --txt-secondary: #a8a29e;
        --txt-muted:     #57534e;
        --txt-sidebar:   #faf7f2;
        --txt-sidebar-dim:#57534e;
        --txt-table-head:#57534e;
        --txt-table-body:#a8a29e;
        --accent:        #ea580c;
        --accent-2:      #dc2626;
        --score-hi:      #f87171;
        --score-mid:     #fb923c;
        --score-lo:      #fbbf24;
        --stat1: #ea580c; --stat2: #faf7f2; --stat3: #dc2626; --stat4: #d97706;
        --pill-str-bg:rgba(220,38,38,.14); --pill-str-fg:#f87171;
        --pill-acu-bg:rgba(234,88,12,.14); --pill-acu-fg:#fb923c;
        --pill-imp-bg:rgba(22,163,74,.14); --pill-imp-fg:#4ade80;
        --sb-select-bg:#1c1917; --sb-select-fg:#78716c; --sb-select-bd:#292524;
        """
    else:
        css_vars = """
        --bg-page:       #f7f3ed;
        --bg-sidebar:    #18120e;
        --bg-main:       #f7f3ed;
        --bg-table-even: #f0ebe4;
        --bg-table-head: #1a1612;
        --bg-table-hover:#fde8d8;
        --border:        #1a1612;
        --border-soft:   #e8e0d5;
        --border-sidebar:#2c1e15;
        --txt-primary:   #1a1612;
        --txt-secondary: #5c4a3a;
        --txt-muted:     #b0a090;
        --txt-sidebar:   #f7f3ed;
        --txt-sidebar-dim:#5c4a3a;
        --txt-table-head:#b0a090;
        --txt-table-body:#5c4a3a;
        --accent:        #d4380d;
        --accent-2:      #b91c1c;
        --score-hi:      #b91c1c;
        --score-mid:     #d4380d;
        --score-lo:      #92400e;
        --stat1: #d4380d; --stat2: #1a1612; --stat3: #b91c1c; --stat4: #c2870a;
        --pill-str-bg:#fee2e2; --pill-str-fg:#b91c1c;
        --pill-acu-bg:#ffedd5; --pill-acu-fg:#d4380d;
        --pill-imp-bg:#dcfce7; --pill-imp-fg:#15803d;
        --sb-select-bg:#241810; --sb-select-fg:#a89080; --sb-select-bd:#3d2d22;
        """

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Playfair+Display:ital,wght@0,700;0,900;1,700;1,900&display=swap');

:root {{ {css_vars} }}

html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif !important; }}

/* ── PAGE BACKGROUND ── */
.main, .main .block-container {{
    background: var(--bg-main) !important;
    color: var(--txt-primary) !important;
}}
.main .block-container {{
    padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;
}}
/* stApp root */
[data-testid="stAppViewContainer"] > section:first-child,
[data-testid="stAppViewContainer"] > div {{
    background: var(--bg-main) !important;
}}

/* ── TYPOGRAPHY ── */
h1 {{
    font-family: 'Playfair Display', serif !important;
    font-size: 4.5rem !important; font-weight: 900 !important;
    line-height: 0.9 !important; letter-spacing: -0.03em !important;
    color: var(--txt-primary) !important;
}}
h2 {{
    font-size: 0.72rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.16em !important;
    color: var(--txt-muted) !important;
    border-bottom: 1px solid var(--border-soft) !important;
    padding-bottom: 0.4rem !important; margin-bottom: 0.9rem !important;
    border-top: none !important; margin-top: 0 !important;
}}
h3 {{
    font-size: 0.78rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important;
    color: var(--txt-muted) !important;
}}
.main p, .main li, .main label {{ color: var(--txt-primary) !important; }}
.main span {{ color: inherit; }}

/* ── SIDEBAR — hardcoded literals; sidebar is ALWAYS dark regardless of day/night ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebar"] > div > div > div,
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] {{
    background: {'#0f0e0c' if dark else '#18120e'} !important;
    border-right: none !important;
}}
/* All text white by default */
[data-testid="stSidebar"] * {{ color: #faf7f2 !important; }}
/* Muted items — must stay LIGHT (sidebar always dark) */
[data-testid="stSidebar"] .stMarkdown p {{
    color: #a89080 !important; font-size: 0.78rem;
}}
[data-testid="stSidebar"] label {{
    color: #a89080 !important;
    font-size: 0.7rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.1em;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #faf7f2 !important;
    border-color: rgba(255,255,255,0.08) !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}}
/* Form inputs: dark text on light input background */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input,
[data-testid="stSidebar"] .stNumberInput input {{ color: #000 !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [class*="singleValue"],
[data-testid="stSidebar"] [class*="ValueContainer"] * {{ color: #000 !important; }}
[data-testid="stSidebar"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stTickBarMin"] {{
    color: #a89080 !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(255,255,255,0.04) !important;
    border: 2px solid {'#2c2420' if dark else '#2c1e15'} !important;
    border-radius: 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
    color: #faf7f2 !important; font-weight: 700 !important;
    background: transparent !important;
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem !important;
}}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
    background: rgba(0,0,0,0.2) !important;
}}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] p,
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] span,
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] label {{
    color: #78716c !important;
}}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 0 !important; background: var(--bg-table-head) !important;
    border-radius: 0 !important; padding: 0 !important;
    border: 3px solid var(--border) !important; border-bottom: none !important;
}}
[data-testid="stTabs"] button[role="tab"]::after,
[data-testid="stTabs"] button[role="tab"]::before {{
    display: none !important; border: none !important; background: none !important;
}}
[data-testid="stTabs"] button[role="tab"] {{
    font-size: 0.68rem !important; font-weight: 800 !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important;
    padding: 0.6rem 1.2rem !important; border-radius: 0 !important;
    color: var(--txt-table-head) !important;
    border-right: 1px solid var(--border-sidebar) !important;
    background: var(--bg-table-head) !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: var(--txt-sidebar) !important; font-weight: 900 !important;
    background: var(--accent) !important; border-right-color: var(--accent) !important;
}}
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {{
    color: var(--txt-secondary) !important; background: rgba(255,255,255,0.06) !important;
}}
[data-testid="stTabs"] [role="tabpanel"] {{
    border: 3px solid var(--border) !important; border-top: none !important;
    padding: 1.5rem !important;
    background: var(--bg-main) !important; color: var(--txt-primary) !important;
}}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {{
    border: 3px solid var(--border) !important; border-radius: 0 !important; overflow: hidden;
}}
[data-testid="stDataFrame"] thead tr th {{
    background: var(--bg-table-head) !important; color: var(--txt-table-head) !important;
    font-weight: 900 !important; font-size: 0.62rem !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important;
}}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) td  {{ background: var(--bg-main) !important; color: var(--txt-table-body) !important; }}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {{ background: var(--bg-table-even) !important; color: var(--txt-table-body) !important; }}
[data-testid="stDataFrame"] tbody tr:hover td {{ background: var(--bg-table-hover) !important; }}

/* ── BUTTONS ── */
[data-testid="stButton"] > button {{
    border-radius: 0 !important; font-weight: 800 !important;
    font-size: 0.72rem !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important; padding: 0.5rem 1.1rem !important;
    border: 2px solid var(--border) !important;
    background: transparent !important; color: var(--txt-primary) !important;
    box-shadow: 3px 3px 0 var(--accent) !important;
    transition: all 0.1s !important;
}}
[data-testid="stButton"] > button:hover {{
    background: var(--accent) !important; color: #fff !important;
    box-shadow: none !important; transform: translate(3px,3px) !important;
}}
/* theme toggle — compact, right-aligned, accent fill */
[data-testid="stButton"]:has(button[kind="secondary"]) > button,
div[data-testid="column"]:last-child [data-testid="stButton"] > button {{
    font-size: 0.68rem !important; padding: 0.35rem 0.7rem !important;
    width: 100% !important;
    background: var(--accent) !important; color: #fff !important;
    border-color: var(--accent) !important; box-shadow: none !important;
}}
div[data-testid="column"]:last-child [data-testid="stButton"] > button:hover {{
    background: var(--accent-2) !important; border-color: var(--accent-2) !important;
    transform: none !important;
}}

/* ── EXPANDERS (main) ── */
.main [data-testid="stExpander"] {{
    border: 2px solid var(--border) !important; border-radius: 0 !important; overflow: hidden;
}}
.main [data-testid="stExpander"] summary {{
    font-weight: 800 !important; font-size: 0.7rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    padding: 0.65rem 1rem !important;
    background: var(--bg-table-head) !important; color: var(--txt-sidebar) !important;
}}
.main [data-testid="stExpander"] summary span {{ color: var(--txt-sidebar) !important; }}
[data-testid="stExpanderDetails"] {{
    background: var(--bg-main) !important; color: var(--txt-primary) !important; padding: 1rem !important;
}}
[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] span,
[data-testid="stExpanderDetails"] li {{ color: var(--txt-primary) !important; }}

/* ── ALERTS ── */
[data-testid="stAlert"] {{
    border-radius: 0 !important; border: 2px solid var(--border) !important;
    border-left-width: 5px !important; font-size: 0.82rem !important;
    padding: 0.65rem 1rem !important; background: var(--bg-table-even) !important;
}}
[data-testid="stAlert"] p, [data-testid="stAlert"] span {{ color: var(--txt-primary) !important; }}

/* ── CHAT ── */
[data-testid="stChatMessage"] {{
    border-radius: 0 !important; border: 2px solid var(--border-soft) !important;
    padding: 0.8rem 1rem !important; margin-bottom: 0.5rem;
    background: var(--bg-table-even) !important;
}}
[data-testid="stChatMessage"] p {{ color: var(--txt-primary) !important; }}

/* ── CAPTION ── */
.stCaption, [data-testid="stCaptionContainer"] p, small {{
    font-size: 0.72rem !important; color: var(--txt-muted) !important;
    font-weight: 500 !important; letter-spacing: 0.04em;
}}

/* ── st.metric FALLBACK ── */
[data-testid="stMetric"] {{
    background: var(--bg-table-even) !important;
    border: 2px solid var(--border) !important; border-radius: 0 !important;
    padding: 1rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.6rem !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.12em !important;
    color: var(--txt-muted) !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 2rem !important; font-weight: 900 !important;
    color: var(--txt-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.04em !important;
}}

/* ── CODE ── */
.main code, .main pre {{
    background: var(--bg-table-head) !important; color: var(--txt-sidebar) !important;
    border-radius: 0 !important; font-size: 0.8rem;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-main); }}
::-webkit-scrollbar-thumb {{ background: var(--border); }}

/* ── CUSTOM STAT CARDS ── */
.stat-grid {{
    display: grid; grid-template-columns: repeat(4,1fr);
    border: 3px solid var(--border); margin-bottom: 1.75rem;
}}
.stat-card {{
    padding: 1rem 1.1rem;
    border-right: 3px solid var(--border);
    background: var(--bg-main);
}}
.stat-card:last-child {{ border-right: none; }}
.stat-card::before {{
    content: ''; display: block; height: 4px; margin-bottom: 0.65rem;
}}
.stat-card:nth-child(1)::before {{ background: var(--stat1); }}
.stat-card:nth-child(2)::before {{ background: var(--stat2); }}
.stat-card:nth-child(3)::before {{ background: var(--stat3); }}
.stat-card:nth-child(4)::before {{ background: var(--stat4); }}
.stat-value {{
    font-size: 2.4rem; font-weight: 900;
    color: var(--txt-primary); line-height: 1; letter-spacing: -0.05em;
    font-family: 'JetBrains Mono', monospace;
}}
.stat-label {{
    font-size: 0.59rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--txt-muted); margin-top: 0.2rem;
}}
.stat-sub {{ font-size: 0.68rem; color: var(--txt-secondary); margin-top: 0.1rem; }}

/* ── HERO ── */
.hero-eyebrow {{
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: var(--txt-muted); margin-bottom: 0.5rem;
}}
.hero-headline {{
    font-family: 'Playfair Display', serif;
    font-size: 5rem; font-weight: 900;
    color: var(--txt-primary); line-height: 0.88; letter-spacing: -0.03em;
}}
.hero-headline em {{ font-style: italic; color: var(--accent); }}
.hero-rule-thick {{ border: none; border-top: 4px solid var(--border); margin: 1rem 0 0.2rem; }}
.hero-rule-thin  {{ border: none; border-top: 1px solid var(--border-soft); margin: 0 0 0.75rem; }}
.hero-dateline {{
    font-size: 0.64rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.12em; color: var(--txt-muted); margin-bottom: 1.5rem;
}}

/* ── HIDE BRANDING ── */
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}
/* Force sidebar toggle button always visible */
[data-testid="stSidebarCollapsedControl"] {{
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}}
[data-testid="stSidebarCollapsedControl"] * {{
    visibility: visible !important;
}}
</style>
<script>
(function() {{
    function fixSidebarToggle() {{
        var btn = document.querySelector('[data-testid="stSidebarCollapsedControl"]');
        if (btn) {{
            btn.style.cssText += '; visibility: visible !important; display: flex !important; opacity: 1 !important; pointer-events: auto !important; z-index: 999999 !important;';
        }}
    }}
    // Run immediately and on DOM changes
    fixSidebarToggle();
    var observer = new MutationObserver(fixSidebarToggle);
    observer.observe(document.body, {{ childList: true, subtree: true }});
    // Also wire up [ key shortcut manually
    document.addEventListener('keydown', function(e) {{
        if (e.key === '[') {{
            var btn = document.querySelector('[data-testid="stSidebarCollapsedControl"]');
            if (btn) btn.click();
            var closeBtn = document.querySelector('[data-testid="stSidebarCollapseButton"]');
            if (closeBtn) closeBtn.click();
        }}
    }});
}})();
</script>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------

def render_hero(df_full: pd.DataFrame):
    """Editorial Brutalism hero — Playfair Display + warm accent."""
    if not df_full.empty:
        # Use most-recent-year per country to avoid double-counting multi-year rows
        recent = df_full.loc[df_full.groupby("country_iso3")["year"].idxmax()]
        n_crises  = recent["country_iso3"].nunique()
        total_pin = recent["people_in_need"].fillna(0).sum() / 1e6
        n_struct  = int((recent["neglect_type"] == "structural").sum()) if "neglect_type" in recent.columns else 0
    else:
        n_crises, total_pin, n_struct = "—", 0, 0

    # Day / Night toggle — top-right of main area
    theme = st.session_state.get("theme", "light")
    icon  = "🌙 Night" if theme == "light" else "☀️ Day"
    col_spacer, col_btn = st.columns([10, 1])
    with col_btn:
        if st.button(icon, key="theme_toggle"):
            st.session_state["theme"] = "dark" if theme == "light" else "light"
            st.rerun()

    st.markdown(f"""
<div class="hero-eyebrow">Humanitarian Funding Intelligence · 2025</div>
<div class="hero-headline">Where Need<br>Outpaces <em>Funding</em></div>
<div class="hero-rule-thick"></div>
<div class="hero-rule-thin"></div>
<div class="hero-dateline">
  {n_crises} countries tracked &nbsp;·&nbsp; {total_pin:.0f}M people in need &nbsp;·&nbsp;
  {n_struct} structural neglect cases &nbsp;·&nbsp;
  HDX HNO · FTS · INFORM · Claude Sonnet 4.6
</div>
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
        min_value=0, value=0, step=100_000,
        help="Filter out crises below this PIN threshold. Proxy countries (FTS only, no HNO data) have PIN=0 and pass when threshold is 0.",
    )
    filtered = filtered[(filtered["people_in_need"].fillna(0) >= min_pin)]

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
    """Bordered stat grid — Editorial Brutalism style."""
    if df.empty:
        return
    n_crises   = len(df)
    total_pin  = df["people_in_need"].fillna(0).sum()
    structural = int((df["neglect_type"] == "structural").sum()) if "neglect_type" in df.columns else 0
    no_hrp     = int((~df["has_hrp"].fillna(False)).sum()) if "has_hrp" in df.columns else 0
    pin_str    = f"{total_pin/1e6:.1f}M" if total_pin >= 1e6 else f"{total_pin:,.0f}"

    st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-value">{n_crises}</div>
    <div class="stat-label">Crises shown</div>
    <div class="stat-sub">after sidebar filters</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{pin_str}</div>
    <div class="stat-label">People in need</div>
    <div class="stat-sub">across filtered crises</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{structural}</div>
    <div class="stat-label">Structural neglect</div>
    <div class="stat-sub">underfunded 3+ years</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{no_hrp}</div>
    <div class="stat-label">No HRP in place</div>
    <div class="stat-sub">coverage assumed 0%</div>
  </div>
</div>
""", unsafe_allow_html=True)


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

    # Format donor concentration columns if available
    if "top1_donor_share" in display_df.columns:
        display_df["Top Donor %"] = display_df["top1_donor_share"].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "—"
        )
    if "donor_hhi" in display_df.columns:
        display_df["HHI"] = display_df["donor_hhi"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        )
    if "data_tier" in display_df.columns:
        display_df["Data Tier"] = display_df["data_tier"].map(
            {"hno": "HNO ✓", "fts_proxy": "FTS proxy"}
        ).fillna("—")

    display_cols = {
        "country_name": "Country",
        "year": "Year",
        "Data Tier": "Data Tier",
        "Gap Score (CI)": "Gap Score [CI]",
        "coverage_pct": "Coverage %",
        "people_in_need": "People in Need",
        "neglect_type": "Neglect Type",
        "consecutive_years_underfunded": "Yrs Underfunded",
        "Top Donor %": "Top Donor %",
        "HHI": "Donor HHI",
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

    st.dataframe(table, use_container_width=True, height=380, hide_index=True)

    # Note on 0%* assumption
    if display_df["coverage_pct"].eq(0).any() and (~display_df.get("has_hrp", pd.Series([True]*len(display_df))).fillna(True)).any():
        st.caption("\\* **0%** = no Humanitarian Response Plan (HRP) on record — funding coverage is assumed 0%, not measured. Treat these rows as lower-bound estimates only.")

    # Data tier legend
    with st.expander("📋 Column guide", expanded=False):
        st.caption(
            "**Data Tier** — HNO ✓: full humanitarian needs assessment (people in need, INFORM severity, funding). "
            "FTS proxy: only FTS financial data available — no official PIN figure, gap score based on funding gap + severity only. "
            "Proxy rankings are conservative estimates.\n\n"
            "**Top Donor %** — share of total funding from the single largest donor. High % = donor concentration risk. "
            "If one donor withdraws, coverage collapses.\n\n"
            "**Donor HHI** — Herfindahl-Hirschman Index for donor concentration (0 = perfectly distributed, 1 = monopoly). "
            "HHI > 0.25 indicates high concentration; < 0.15 = diverse donor base."
        )

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
            + (f"PIN: {r.get('people_in_need', 0)/1e6:.1f}M<br>" if pd.notna(r.get('people_in_need')) else "PIN: N/A (proxy)<br>")
            + f"Neglect: {r.get('neglect_type', 'N/A')}<br>"
            + (f"Donors: {int(r['n_donors'])} (HHI {r['donor_hhi']:.3f})<br>" if pd.notna(r.get('n_donors')) else "")
            + (f"Data: {r.get('data_tier','')}")
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

    # ── NEW message at the TOP (newest first layout) ────────────────────────
    if user_input:
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

            # ── Feedback button ──────────────────────────────────────────────
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

        # Append both to history AFTER displaying
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role":       "assistant",
            "content":    response,
            "confidence": confidence,
        })
        # Show older history below the new exchange
        history_to_show = st.session_state.chat_history[:-2]
    else:
        history_to_show = st.session_state.chat_history

    # ── HISTORY: reverse order (newest → oldest) below the current exchange ──
    if history_to_show:
        st.markdown("---")
        for msg in reversed(history_to_show):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


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
# Responsible AI Scorecard (explicitly requested by hackathon judges)
# ---------------------------------------------------------------------------

def render_rai_scorecard():
    """
    Responsible AI Scorecard — covers transparency, uncertainty, fairness,
    human oversight, and data governance.

    Referenced in hackathon judge guidance:
    'Show a quick view of your MLflow tracing, tool-calling logic, and
    Responsible AI Scorecard to prove your agents are reliable and governed.'
    """
    st.subheader("Responsible AI Scorecard")
    st.caption(
        "This scorecard documents the design choices made to ensure the Geo-Insight "
        "Command Center is transparent, auditable, and safe for use in humanitarian "
        "resource allocation decisions. Each dimension is grounded in a specific "
        "architectural or data decision — not a checkbox exercise."
    )

    # ── Dimension 1: Transparency ────────────────────────────────────────────
    with st.expander("✅ 1 — Transparency & Explainability", expanded=True):
        col1, col2 = st.columns([1, 2])
        col1.metric("Status", "✅ Implemented")
        col2.markdown(
            "**Every gap score is fully decomposable** into 5 intermediate components "
            "(`funding_gap`, `need_scale`, `severity_mult`, `base_score`, `neglect_factor`). "
            "The formula is displayed in-app, in README, and stored in every Gold table row. "
            "No black-box model — a junior analyst can replicate any score in a spreadsheet."
        )
        st.markdown("""
| Component | What it means | Source |
|-----------|--------------|--------|
| `funding_gap` | `1 − (received / requested)` — primary underfunding signal | FTS |
| `need_scale` | `log1p(PIN) / log1p(P95_PIN)` — scale of humanitarian need, Ringer Bid-safe | HNO |
| `severity_mult` | `INFORM_severity / 10` — independent crisis urgency signal | INFORM |
| `neglect_factor` | `1 + (years_underfunded × 0.15)` — structural vs acute penalty | Derived |
| `gap_score` | `base_score × neglect_factor` — final composite score | Computed |

**Swing Weighting:** Weights (0.435 / 0.348 / 0.217) derived by answering *"if you could improve ONE dimension from worst to best, which matters most?"* — a principled MCDM technique, not arbitrary tuning.
        """)

    # ── Dimension 2: Uncertainty & Confidence ───────────────────────────────
    with st.expander("✅ 2 — Uncertainty Quantification", expanded=True):
        col1, col2 = st.columns([1, 2])
        col1.metric("Status", "✅ Implemented")
        col2.markdown(
            "**Three-layer uncertainty system.** Rankings are never shown as point estimates alone."
        )
        st.markdown("""
| Layer | Method | Shown as |
|-------|--------|---------|
| **Monte Carlo CI** | 1 000 simulations, PIN ±30% / funding ±20% triangular distributions | Gap Score [P10–P90] in table |
| **Low-confidence flag** | No HRP (0% = assumption) or data >18 months old | ⚠️ column in table |
| **EVPI** | Expected Value of Perfect Information — score uncertainty under improved data | EVPI expander in Tab 1 |

**EVPI implication:** If P10–P90 range > 0.15, rank position is data-quality sensitive — a better HNO assessment is worth commissioning before allocation decisions.
        """)

    # ── Dimension 3: Fairness & Bias Awareness ───────────────────────────────
    with st.expander("⚠️ 3 — Fairness & Bias Awareness", expanded=False):
        col1, col2 = st.columns([1, 2])
        col1.metric("Status", "⚠️ Partially mitigated")
        col2.markdown(
            "Known biases are documented and partially mitigated — not eliminated."
        )
        st.markdown("""
| Bias type | How it arises | Mitigation |
|-----------|--------------|------------|
| **Reporting bias** | Countries with no HRP have 0% coverage *by assumption*, not measurement | `0%*` footnote in table; `low_confidence = True` |
| **Ringer Bid distortion** | One very large crisis (Sudan 34M PIN) would compress all others' `need_scale` | 95th-percentile reference instead of global max |
| **FTS proxy inflation** | Countries without HNO PIN score on funding gap + severity only, may rank unexpectedly high | `data_tier = fts_proxy` column; explicit warning in column guide |
| **Neglect factor stacking** | Structural neglect bonus (+15%/year) is uncapped — could theoretically inflate to any multiple | Capped at 3 years data window (max ×1.45 in practice) |
| **Media attention bias** | FTS funding is partly driven by media salience — underfunded ≠ overlooked if needs are simply smaller | INFORM severity as independent signal partially corrects |
        """)

    # ── Dimension 4: Neutral Framing ─────────────────────────────────────────
    with st.expander("✅ 4 — Neutral Framing (Framing Trap Avoidance)", expanded=False):
        col1, col2 = st.columns([1, 2])
        col1.metric("Status", "✅ Implemented")
        col2.markdown(
            "Agent briefing notes follow strict neutral framing rules, "
            "inspired by CMU ODI decision design guidance (Framing Trap)."
        )
        st.markdown("""
**Agent system prompt rules (enforced per query):**
1. Express gaps as **coverage shortfalls** (e.g. "38% funded, 62% shortfall") — never as loss/death framing
2. **No anchoring** to prior-year funding — each assessment is independent
3. **Mandatory counter-argument** per crisis (e.g. "unreported bilateral flows may overstate gap")
4. **Explicit uncertainty** disclosure if `low_confidence = True`
5. **Ground all numbers** in the data table — no hallucinated figures

These rules are injected into the system prompt and verified by the red-team tool in the ReAct loop.
        """)

    # ── Dimension 5: Human Oversight ─────────────────────────────────────────
    with st.expander("✅ 5 — Human Oversight (Human-in-the-Loop)", expanded=False):
        col1, col2 = st.columns([1, 2])
        col1.metric("Status", "✅ Implemented")
        col2.markdown(
            "The system is explicitly designed as **decision support**, not automated action."
        )
        st.markdown("""
| Safeguard | Implementation |
|-----------|---------------|
| **No auto-allocation** | Rankings are displayed for analyst review; no system can trigger fund transfers |
| **Low-confidence escalation** | If agent confidence = 'low', UI shows a red warning banner before displaying results |
| **👎 Flag button** | Every agent response has a flag button; flagged responses are logged to `eval_feedback.jsonl` |
| **MLflow tracing** | Every query is traced (root span + per-tool child spans) — full audit trail of agent reasoning |
| **Red-team challenge** | A second Claude call acts as adversarial auditor, surfacing weaknesses in every ranking |
| **LLM-as-Judge eval** | `05_evaluate_agent.py` runs 4 golden test cases and scores agent responses for grounding, neutrality, defensibility |
        """)

    # ── Dimension 6: Data Governance ─────────────────────────────────────────
    with st.expander("✅ 6 — Data Governance & Source Transparency", expanded=False):
        col1, col2 = st.columns([1, 2])
        col1.metric("Status", "✅ Implemented")
        col2.markdown(
            "All data sources are publicly available, declared, and versioned. "
            "No proprietary or personally-identifiable data used."
        )
        st.markdown("""
| Source | What it provides | Why FTS (not CBPF) for primary funding |
|--------|-----------------|---------------------------------------|
| **HNO 2024/2025/2026** | People in need, targeted — 24 countries | Official OCHA humanitarian needs data |
| **FTS Global** | Requirements + funding received — 77 countries | Most comprehensive country-level funding tracker |
| **INFORM Severity** | Independent crisis severity 0–10 — 67 countries | Non-self-reported, ACAPS-validated |
| **HRP list** | Humanitarian Response Plan presence — 78 countries | Distinguishes measured vs assumed 0% coverage |
| **FTS Flows API** | Donor breakdown per plan — 72 countries | Donor concentration analysis (HHI, top donor %) |

**Why FTS instead of CBPF for primary coverage metric?**
The CBPF allocation file (`Allocations__20260518_145817_UTC.csv`) available in the Databricks volume only contains **2018 data** — not current. The FTS requirements/funding file covers 2024–2026 across 77 countries and is the OCHA-official tracker used in HRP reporting. CBPF allocations are preserved in `bronze_cbpf.parquet` for supplementary use.

All sources declared in [`data_sources.md`](../data_sources.md).
        """)

    # ── Dimension 7: Model Governance ────────────────────────────────────────
    with st.expander("✅ 7 — Model Governance (MLflow)", expanded=False):
        col1, col2 = st.columns([1, 2])
        col1.metric("Status", "✅ Implemented")
        col2.markdown(
            "Every agent interaction is logged and traceable via MLflow. "
            "Scoring parameters are versioned."
        )
        st.markdown("""
| Governance artefact | Location | Contents |
|--------------------|---------|---------|
| **MLflow traces** | `mlflow ui --port 5001` → Geo-Insight experiment | Root span per query + child span per tool call; full I/O logged |
| **MLflow eval runs** | `geo-insight-eval` experiment | 4 golden cases, LLM-as-Judge scores for grounding / neutrality / defensibility |
| **Scoring parameters** | `core/scoring_logic.py` · `SCORING_WEIGHTS` dict | Logged to MLflow on each scoring run |
| **Gold table version** | `data/gold/gold_ranked_crises.parquet` | Timestamped rebuild via `06_refresh_from_databricks.py` |
| **Feedback log** | `data/eval_feedback.jsonl` | User-flagged responses → next eval batch |
        """)

    # ── Summary status table ──────────────────────────────────────────────────
    st.markdown("### Summary")
    st.dataframe(pd.DataFrame([
        {"Dimension": "Transparency & Explainability",   "Status": "✅", "Key evidence": "5-component formula, in-app + README"},
        {"Dimension": "Uncertainty Quantification",      "Status": "✅", "Key evidence": "Monte Carlo CI + EVPI + low_confidence flag"},
        {"Dimension": "Fairness & Bias Awareness",       "Status": "⚠️", "Key evidence": "Biases documented; Ringer Bid + proxy tier mitigated"},
        {"Dimension": "Neutral Framing",                 "Status": "✅", "Key evidence": "Agent system prompt rules + red-team enforcer"},
        {"Dimension": "Human Oversight",                 "Status": "✅", "Key evidence": "No auto-action; MLflow traces; flag button; escalation banner"},
        {"Dimension": "Data Governance",                 "Status": "✅", "Key evidence": "All public sources; FTS vs CBPF rationale documented"},
        {"Dimension": "Model Governance",                "Status": "✅", "Key evidence": "MLflow tracing + eval + feedback loop"},
    ]), use_container_width=True, hide_index=True)

    st.caption(
        "Legend: ✅ = implemented · ⚠️ = partially mitigated (known limitation) · ❌ = not addressed. "
        "This scorecard is a living document — limitations are disclosed, not hidden."
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.session_state.setdefault("theme", "light")
    _inject_theme()

    df_full = load_gold()
    render_hero(df_full)

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Ranked Table", "World Map", "Sector Gaps", "Data Drift", "Ask the Agent", "🛡️ RAI Scorecard"
    ])

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

    with tab6:
        render_rai_scorecard()

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
