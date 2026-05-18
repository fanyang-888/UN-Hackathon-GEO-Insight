# Geo-Insight: Overlooked Humanitarian Crises

**CMU × Databricks × UN OCHA Hackathon 2026 · Solo Build · May 18–21, 2026**

An end-to-end Agentic Command Center that identifies humanitarian crises where documented need significantly outpaces funding coverage. Accepts natural-language queries and returns ranked crises with defensible, auditable gap scores.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Gap Score Formula](#gap-score-formula)
3. [Data Sources](#data-sources)
4. [File Structure](#file-structure)
5. [Setup & Run](#setup--run)
6. [Dashboard Pages](#dashboard-pages)
7. [Agent Design](#agent-design)
8. [CMU Knowledge Base Integration](#cmu-knowledge-base-integration)
9. [Known Limitations](#known-limitations)
10. [Changelog](#changelog)

---

## Architecture

```
Public APIs (HDX HAPI, INFORM, CBPF)
        ↓
  [01 Bronze Layer]  — raw parquet files in data/bronze/
        ↓
  [02 Silver Layer]  — normalized, ISO-3 matched, INFORM joined → data/silver/
        ↓
  [03 Gold Layer]    — gap scores, CI, Pareto flags, MLflow logged → data/gold/
        ↓
  [04 Claude Agent]  — ReAct tool-calling loop (decompose → query → validate → brief)
        ↓
  [05 Streamlit UI]  — Ranked Table · World Map · Ask the Agent
```

**Tech stack:** Python 3.12 · Pandas · Claude API (claude-sonnet-4-6) · MLflow · Streamlit · Plotly · HDX HAPI

---

## Gap Score Formula

Composite score based on CMU Decision Modeling (MCDM Weighted Sum + Swing Weighting):

```
coverage_ratio  = min(funding_received / funding_requested, 1.0)
funding_gap     = 1 − coverage_ratio

need_scale      = log1p(PIN) / log1p(95th-pct PIN)    ← Ringer Bid–safe
severity_mult   = INFORM_severity / 10

base_score      = funding_gap × (0.435 + 0.348×need_scale + 0.217×severity_mult)

neglect_factor  = 1 + (consecutive_years_underfunded × 0.15)

gap_score       = base_score × neglect_factor
```

**Weight derivation (Swing Weighting):**
> "If you could move ONE dimension from worst to best, which matters most?"
- `funding_gap` → 100 pts → normalized 0.435
- `need_scale`  → 80 pts  → normalized 0.348
- `severity`    → 50 pts  → normalized 0.217

**Confidence interval:** ±20% sensitivity on PIN and funding_received (humanitarian data typically 20–30% off).

**Pareto filter:** A crisis is "dominated" if another *different* crisis is strictly better on all three primary dimensions (funding_gap, PIN, INFORM severity). Dominated crises are scored and shown but flagged. Same-country cross-year comparisons are excluded.

---

## Data Sources

| Source | What it provides | Endpoint |
|--------|-----------------|----------|
| HDX HAPI `/humanitarian-needs` | People in Need (PIN) by country | `admin_level=0, population_status=INN` |
| HDX HAPI `/coordination-context/funding` | Funding requested & received (FTS) | paginated JSON |
| INFORM Severity Index | Independent crisis severity score 0–10 | XLSX via HDX dataset API |
| CBPF Pooled Funds | Country-Based Pooled Fund allocations | HDX HAPI `/funding` |

All sources declared in [`data_sources.md`](data_sources.md).

---

## File Structure

```
unocha-hackathon/
├── 01_bronze_ingest.py      # Fetch raw data → data/bronze/*.parquet
├── 02_silver_transform.py   # Normalize, ISO-3 match, join → data/silver/silver_master.parquet
├── 03_gold_scoring.py       # Gap score + MLflow logging → data/gold/gold_ranked_crises.parquet
├── 04_agent.py              # Claude API agent (4 tools + ReAct loop)
├── 05_dashboard.py          # Streamlit dashboard
├── agent_runner.py          # Import shim for digit-prefixed 04_agent.py
├── scoring_logic.py         # Pure Python gap score functions (importable anywhere)
├── data_sources.md          # Declared external data sources (hackathon requirement)
├── requirements.txt         # Python dependencies
├── .env.example             # Template: ANTHROPIC_API_KEY=...
└── data/
    ├── bronze/              # Raw parquet files per source
    ├── silver/              # Joined master table (151 rows, 3 years, 109 countries)
    └── gold/                # Scored & ranked (66 rows, 24 countries with PIN data)
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the pipeline (in order)

```bash
python 01_bronze_ingest.py      # ~2 min, fetches live data from HDX HAPI
python 02_silver_transform.py   # ~5 sec
python 03_gold_scoring.py       # ~10 sec, logs to MLflow
```

### 4. Launch dashboard

```bash
streamlit run 05_dashboard.py
# Opens at http://localhost:8501
```

### 5. (Optional) Test agent via CLI

```bash
ANTHROPIC_API_KEY=your-key python 04_agent.py
```

---

## Dashboard Pages

### Ranked Table

Default landing page. Shows all crises sorted by gap score descending.

| Column | Description |
|--------|-------------|
| Rank | Gap score rank (1 = most overlooked) |
| Country | Country name |
| Year | Data year (2024–2026) |
| Gap Score [CI] | Composite score with ±20% confidence interval, e.g. `1.201 [1.167–1.233]` |
| Coverage % | `funding_received / funding_requested × 100` |
| People in Need | PIN from HNO (millions) |
| Neglect Type | `structural` (≥3 yrs underfunded) · `acute` (newly underfunded) · `ongoing` · `improving` |
| Yrs Underfunded | Consecutive years below 50% funding coverage |
| Has HRP | Whether a Humanitarian Response Plan exists (no HRP → coverage = 0% assumption) |
| Pareto | `★ frontier` = not dominated by any other crisis on all dimensions |
| ⚠️ Low Conf. | True if no HRP or data >18 months old |

**Sidebar filters:**
- **Year** — `Most recent` (one row per country, latest year) or specific year (2024/2025/2026)
- **Region** — Sub-Saharan Africa, Middle East, South Asia, etc.
- **Max funding coverage %** — slider to show only under-X%-funded crises
- **Neglect type** — filter by structural / acute / ongoing / improving
- **Min people in need** — exclude small crises below threshold

**⚖️ Weight Sensitivity (What-If) panel** *(CMU dm-choice-mcdm.md)*:
Expandable section in sidebar with three sliders for `funding_gap`, `need_scale`, `severity` weights. Click "Apply custom weights" to rescore all crises with your weights and see how the ranking changes. Demonstrates Swing Weighting robustness — top crises remain stable across reasonable weight perturbations.

**Expanders:**
- *What is the confidence interval [CI]?* — explains ±20% sensitivity methodology
- *How is the Gap Score calculated?* — full formula with weight rationale

---

### World Map

Choropleth map (Plotly, natural earth projection) colored by gap score:

| Color | Gap Score Range | Meaning |
|-------|----------------|---------|
| Green | 0.00–0.30 | Well-funded |
| Yellow | 0.30–0.60 | Moderate gap |
| Orange | 0.60–0.90 | High gap |
| Red | 0.90+ | Severe gap — most overlooked |

Color scale capped at 95th percentile to prevent one extreme crisis from washing out all others (Ringer Bid prevention).

Hover shows: country name, gap score, coverage %, PIN, neglect type, ⚠️ low-confidence flag.

---

### Ask the Agent

Conversational interface powered by Claude API (claude-sonnet-4-6).

**Architecture:** ReAct 5-step loop (CMU agents.md)
1. `decompose_query` — NL → structured filters `{region, coverage_ceiling, min_pin, neglect_type, year_range, top_n}`
2. `query_gold_table` — parameterized query against Gold parquet
3. `validate_ranking` — self-audit: flags low-confidence rows, anomalies, data quality issues
4. `generate_briefing_notes` — per-crisis 2–3 sentence briefings (neutral framing, counter-argument required)
5. Synthesize final response

**Bias guardrails (CMU odi-decisions.md):**
- Neutral framing: "38% of requested funding received, leaving 62% shortfall" — never loss/death framing
- No anchoring: never compare to historical baseline
- Forced counter-argument: every crisis briefing includes one reason the score might be overstated
- De novo scoring: agent ignores prior query results when ranking

**Session memory (CMU agents.md Reflexion):**
Previous query results are stored in `_session_memory`. Follow-up questions containing "tell me more", "#1", "top crisis", "elaborate" automatically inject prior crisis list as context.

**Example queries:**
- "Which crises have the highest PIN but lowest funding?"
- "Show structurally neglected crises in Sub-Saharan Africa"
- "Are there countries with active HRPs but under 10% funded?"
- "Which crises have been underfunded for 3+ consecutive years?"

**Data quality alert banner** *(CMU oai-monitoring-governance.md)*:
If >30% of loaded crises have data older than 12 months, a warning banner appears at the top. If >10%, an info notice. Prevents analysts from acting on stale data.

---

## Agent Design

### Tools

```python
decompose_query(query, top_n, year)
    → {region, coverage_ceiling_pct, min_people_in_need, neglect_type, year, top_n}

query_gold_table(filters)
    → list[crisis_dict]   # from data/gold/gold_ranked_crises.parquet

generate_briefing_notes(crises, query)
    → str   # markdown briefing, neutral framing, counter-argument per crisis

validate_ranking(top_crises, concerns, confidence_level, corrections)
    → {n_low_confidence, n_without_hrp, concerns, corrections, confidence_level}
```

### System prompt structure

1. **Identity** — OCHA Geo-Insight Command Center
2. **ReAct loop** — explicit THOUGHT 1→5 scaffold
3. **Bias guardrails** — framing rules, anchoring prohibition, counter-argument mandate
4. **Hard rules** — never fabricate, never anchor, always cite data, validate before briefing

Internal `THOUGHT N:` reasoning is stripped from user-visible output via `_strip_thoughts()`.

---

## CMU Knowledge Base Integration

| CMU Module | Concept Applied | Where |
|---|---|---|
| `dm-choice-mcdm.md` | Swing Weighting for defensible weights | `scoring_logic.py` SCORING_WEIGHTS |
| `dm-choice-mcdm.md` | Pareto filtering (cross-country only) | `scoring_logic.py` `score_dataframe()` |
| `dm-choice-mcdm.md` | Ringer Bid prevention via 95th-pct reference | `scoring_logic.py` `compute_gap_score()` |
| `dm-choice-mcdm.md` | Weight sensitivity "what-if" panel | `05_dashboard.py` sidebar expander |
| `dm-uncertainty.md` | ±20% CI on PIN and funding | `scoring_logic.py` `compute_confidence_interval()` |
| `agents.md` | ReAct 5-step Thought→Action→Observation | `04_agent.py` SYSTEM_PROMPT |
| `agents.md` | Self-evaluation tool (`validate_ranking`) | `04_agent.py` tool dispatch |
| `agents.md` | Reflexion session memory for follow-up | `04_agent.py` `_session_memory` |
| `hallucinations.md` | Forced grounding: all numbers from tool observations | `04_agent.py` Hard Rules |
| `odi-decisions.md` | Neutral framing (no loss/gain language) | `scoring_logic.py` `build_explanation_prompt()` |
| `odi-decisions.md` | No anchoring to historical baselines | `04_agent.py` SYSTEM_PROMPT |
| `odi-decisions.md` | Counter-argument mandate per crisis | `04_agent.py` SYSTEM_PROMPT |
| `oai-monitoring-governance.md` | Data quality drift alert banner | `05_dashboard.py` `_render_data_quality_alert()` |

---

## Known Limitations

| Issue | Impact | Notes |
|-------|--------|-------|
| No HRP for many countries | Coverage = 0% assumption, not measured | Flagged `low_confidence=True` |
| HNO data only 109 countries | Missing COD, MMR and others | Increase `max_pages` in `01_bronze_ingest.py` |
| INFORM max-severity aggregation | Multi-crisis countries (Nigeria, Chad) get highest crisis severity, may overstate | Logged in briefing counter-arguments |
| Funding data spans multi-year appeals | `funding_year` may not match `HNO year` | Silver join uses most-recent funding ≤ current year |
| Gold only scores countries with PIN data | 24 of 109 countries have PIN; rest dropped at filter step | Increase HNO coverage via more API pages |
| MLflow runs locally | Not synced to Databricks in local mode | Run on Databricks for Unity Catalog integration |

---

## Changelog

### 2026-05-16 (Session 2)

**Bug fixes:**
- `year` dropdown `ValueError`: `"2026.0"` → `int()` fails; fixed by `astype(int)` before display and comparison
- Year filter showed no data for 2024/2025: Silver `drop_duplicates()` was removing multi-year rows; removed deduplication, Silver now has 151 rows across 3 years
- Pareto filter over-flagged: cross-year same-country domination caused 80% of rows marked; fixed by skipping `country_iso3` matches in Pareto loop
- INFORM join created duplicate rows per country: INFORM has multiple crises per country (e.g. Yemen ×3); fixed by `groupby(country_iso3).inform_severity.max()` before join
- Agent `THOUGHT N:` leaked to user output: fixed `_strip_thoughts()` to handle `**THOUGHT N:**` markdown bold format
- Agent / Dashboard relative paths failed when launched from parent directory: fixed all paths to `Path(__file__).parent / "data/..."` 

**CMU-driven enhancements:**
- `⚖️ Weight Sensitivity` panel in sidebar: sliders for MCDM weights, live rescore (`dm-choice-mcdm.md`)
- Pareto display changed from `↓ dominated` (noisy) to `★ frontier` (only non-dominated rows marked) (`dm-choice-mcdm.md`)
- Data quality drift alert banner: warns when >30% of data is >12 months old (`oai-monitoring-governance.md`)
- Agent Reflexion session memory: stores last query results, injects context for follow-up questions (`agents.md`)

### 2026-05-15 (Session 1)

**Initial build:**
- Bronze ingest: HDX HAPI HNO, FTS funding, INFORM XLSX, CBPF
- Silver transform: ISO-3 normalization, INFORM max-severity aggregation, coverage ratio
- Gold scoring: MCDM weighted sum, ±20% CI, Pareto filter, MLflow logging
- Agent: 4-tool ReAct loop with bias guardrails (neutral framing, no anchoring, counter-argument)
- Dashboard: Ranked Table with CI display, World Map choropleth, Ask the Agent chat interface
- Sanity check: SSD, YEM, SOM, HTI, AFG, SDN all in top 20 ✓
