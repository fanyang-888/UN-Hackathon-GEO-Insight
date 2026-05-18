# Geo-Insight: Overlooked Humanitarian Crises

**CMU × Databricks × UN OCHA Hackathon 2026 · Solo Build · May 18–21, 2026**

An end-to-end Agentic Command Center that identifies humanitarian crises where documented need significantly outpaces funding coverage. Accepts natural-language queries and returns ranked crises with defensible, auditable gap scores — designed for **CBPF pooled fund managers** making allocation decisions.

---

## Table of Contents

1. [Architecture](#architecture)
2. [File Structure](#file-structure)
3. [Setup & Run](#setup--run)
4. [Gap Score Formula](#gap-score-formula)
5. [Data Sources](#data-sources)
6. [Dashboard Pages](#dashboard-pages)
7. [Agent Design](#agent-design)
8. [Workflow Integration](#workflow-integration)
9. [CMU Knowledge Base Integration](#cmu-knowledge-base-integration)
10. [Known Limitations](#known-limitations)
11. [Changelog](#changelog)

---

## Architecture

```
Official Databricks Volume (cmu_hackathon.common.unocha)
  hpc_hno_2024/2025/2026.csv · fts_requirements_funding_global.csv
  fts_requirements_funding_globalcluster_global.csv
  humanitarian-response-plans.csv · 202604-inform-severity-april-2026.xlsx
        ↓
  [06 Refresh Pipeline]  — reads official files → rebuilds Bronze/Silver/Gold
        ↓
  [01–03 Original Pipeline]  — HDX HAPI API fallback (24 countries)
        ↓
  [Bronze Layer]   data/bronze/*.parquet  — raw per-source tables
        ↓
  [Silver Layer]   data/silver/silver_master.parquet  — joined, ISO-3 matched
        ↓
  [Gold Layer]     data/gold/gold_ranked_crises.parquet  — gap scores, CI, Pareto
                   data/gold/sector_funding_gaps.csv     — 44 countries × 9 clusters
        ↓
  [RAG Index]      rag_search.py  — TF-IDF vector index over Gold table
        ↓
  [04 Claude Agent]  — ReAct loop: decompose → query/RAG → red-team → brief
        ↓
  [05 Streamlit UI]  — Ranked Table · World Map · Sector Gaps · Ask the Agent
```

**Tech stack:** Python 3.12 · Pandas · Claude API (claude-sonnet-4-6) · MLflow · Streamlit · Plotly · scikit-learn (TF-IDF RAG)

---

## File Structure

```
unocha-hackathon/
├── README.md
├── requirements.txt
├── .gitignore
├── pipeline/                 # Data pipeline (Bronze → Silver → Gold)
│   ├── 01_bronze_ingest.py       # Fetch from HDX HAPI → data/bronze/
│   ├── 02_silver_transform.py    # Normalize, ISO-3 match → data/silver/
│   ├── 03_gold_scoring.py        # Gap score + CI + MLflow → data/gold/
│   └── 06_refresh_from_databricks.py  # Full rebuild from Databricks files
├── agent/                    # Claude agent (query → RAG → red-team → brief)
│   ├── 04_agent.py               # 6-tool ReAct agent loop
│   ├── agent_runner.py           # Import shim for dashboard
│   └── rag_search.py             # TF-IDF RAG index over Gold table
├── app/                      # Streamlit dashboard (4 tabs)
│   └── 05_dashboard.py
├── core/                     # Shared scoring logic (importable anywhere)
│   └── scoring_logic.py          # Gap score, Monte Carlo CI, EVPI
├── data_sources.md
└── data/
    ├── bronze/               # Raw parquet per source
    │   ├── bronze_hno.parquet
    │   ├── bronze_funding.parquet
    │   ├── bronze_cbpf.parquet
    │   └── bronze_inform.parquet
    ├── silver/
    │   └── silver_master.parquet   # 66 rows · 24 countries · 3 years
    └── gold/
        ├── gold_ranked_crises.parquet  # 66 rows · scored & ranked
        ├── sector_funding_gaps.csv     # 390 rows · 44 countries × 9 clusters
        └── top30_overlooked_crises.csv
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
# Edit .env:  ANTHROPIC_API_KEY=sk-ant-...
```

### 3a. Run with official Databricks data (recommended)

Download the following files from the Databricks Unity Catalog volume
(`/Volumes/cmu_hackathon/common/unocha/`) to `~/Downloads/`:

| File | Size | Purpose |
|------|------|---------|
| `hpc_hno_2024.csv` | 31 MB | Official HNO 2024 needs data |
| `hpc_hno_2025.csv` | 26 MB | Official HNO 2025 needs data |
| `hpc_hno_2026.csv` | 6 KB | Official HNO 2026 needs data |
| `fts_requirements_funding_global.csv` | 272 KB | Country-level FTS requirements & funding |
| `fts_requirements_funding_globalcluster_global.csv` | 1.25 MB | Cluster-level FTS data |
| `humanitarian-response-plans.csv` | 125 KB | HRP presence by country & year |
| `202604-inform-severity-april-2026.xlsx` | 2 MB | INFORM Severity April 2026 |

Then run:

```bash
python pipeline/06_refresh_from_databricks.py   # ~30 sec, rebuilds Silver + Gold
```

### 3b. Run with HDX HAPI fallback

```bash
python pipeline/01_bronze_ingest.py      # ~2 min, live API calls
python pipeline/02_silver_transform.py
python pipeline/03_gold_scoring.py       # logs to MLflow
```

### 4. Launch dashboard

```bash
streamlit run app/05_dashboard.py
# Opens at http://localhost:8501
```

### 5. (Optional) Test agent via CLI

```bash
python agent/04_agent.py
```

---

## Gap Score Formula

Composite score — CMU Decision Modeling (MCDM Weighted Sum + Swing Weighting):

```python
coverage_ratio  = min(funding_received / funding_requested, 1.0)
funding_gap     = 1 − coverage_ratio

need_scale      = log1p(PIN) / log1p(95th-pct PIN)   # Ringer Bid-safe
severity_mult   = INFORM_severity / 10

base_score      = funding_gap × (0.435 + 0.348×need_scale + 0.217×severity_mult)

neglect_factor  = 1 + (consecutive_years_underfunded × 0.15)

gap_score       = base_score × neglect_factor
```

**Weight derivation (Swing Weighting):**
> "If you could move ONE dimension from worst to best, which matters most?"

| Dimension | Points | Normalized | Rationale |
|-----------|--------|-----------|-----------|
| `funding_gap` | 100 | 0.435 | Directly determines resource reach |
| `need_scale` | 80 | 0.348 | Scale of human suffering |
| `severity` | 50 | 0.217 | Independent urgency signal (INFORM) |

**Confidence interval:** Monte Carlo simulation (1 000 runs). PIN follows triangular distribution ±30%, funding ±20% — reflecting typical HNO assessment uncertainty. Returns P10/P50/P90 range.

**EVPI (Expected Value of Perfect Information):** Compares current ±30%/±20% uncertainty vs. ±5% improved data quality. Shown in dashboard to help OCHA analysts decide whether commissioning a more detailed HNO assessment is worth it.

**Pareto filter:** A crisis is "dominated" if another *different* crisis is strictly better on all three primary dimensions. Same-country cross-year comparisons excluded. Dominated crises shown but flagged — not removed.

**Ringer Bid prevention:** 95th-percentile PIN as reference prevents one extreme crisis (Sudan, 34M PIN) from compressing all others toward zero on the need scale.

---

## Data Sources

### Official (Databricks Unity Catalog volume)

| File | Source | Coverage |
|------|--------|---------|
| `hpc_hno_2024/2025/2026.csv` | OCHA HPC (official HNO) | 24 countries, 3 years |
| `fts_requirements_funding_global.csv` | FTS (OCHA) | 77 countries, 2024–2026 |
| `fts_requirements_funding_globalcluster_global.csv` | FTS (OCHA) cluster-level | 44 countries, 2024–2026 |
| `humanitarian-response-plans.csv` | OCHA HPC | 78 countries, HRP presence |
| `202604-inform-severity-april-2026.xlsx` | ACAPS INFORM | 67 countries, April 2026 |

### API fallback (01_bronze_ingest.py)

| Source | Endpoint | What it provides |
|--------|---------|-----------------|
| HDX HAPI | `/humanitarian-needs` | PIN by country (`admin_level=0, population_status=INN`) |
| HDX HAPI | `/coordination-context/funding` | FTS requirements & received |
| HDX HAPI | INFORM dataset | Crisis severity 0–10 |
| HDX HAPI | CBPF `/funding` | Pooled fund allocations |

All sources declared in [`data_sources.md`](data_sources.md).

---

## Dashboard Pages

### Tab 1 — Ranked Table

Default landing. Crises sorted by gap score descending. Key columns:

| Column | Description |
|--------|-------------|
| Gap Score [CI] | Composite score with Monte Carlo P10–P90, e.g. `1.162 [1.128–1.195]` |
| Coverage % | `funding_received / funding_requested × 100`; `0%*` = no HRP (assumed, not measured) |
| Neglect Type | `structural` (≥3 yrs) · `acute` · `ongoing` · `improving` |
| Pareto | `★ frontier` = not dominated on all three primary dimensions |
| ⚠️ Low Conf. | No HRP or data >18 months old |

**Sidebar filters:** Year · Region · Max coverage % · Neglect type · Min PIN

**⚖️ Weight Sensitivity panel:** Drag sliders to rescore all crises with custom MCDM weights — shows ranking stability across weight perturbations.

**📊 EVPI panel:** Shows top-5 crises' gap score uncertainty range under current vs. improved data quality. Helps analysts decide whether more detailed HNO assessments are worth commissioning.

### Tab 2 — World Map

Plotly choropleth (natural earth projection) colored by gap score: green (funded) → red (severe gap). Color scale capped at 95th percentile. Hover shows gap score, coverage %, PIN, neglect type.

### Tab 3 — Sector Gaps

**Bonus task:** Surfaces cases where a high aggregate funding figure masks severe cluster-level shortfalls. Data: FTS cluster-level allocations (official Databricks files).

Components:
- **Top-20 bar chart** — worst-funded country × sector pairs, colored by gap severity
- **Coverage heatmap** — 44 countries × 9 clusters matrix, green → red
- **Filters** — sector multiselect, country multiselect, max coverage % slider
- **4 summary metrics** — pairs shown, total PIN, ≥80% unfunded count, zero-funded count

Coverage: **359 country × cluster pairs across 44 countries.**

Key findings:
- 🇧🇫 BFA Agriculture: 0% funded, 4.4M PIN
- 🇾🇪 YEM Agriculture: 0% funded, 23.1M total PIN
- 🇸🇩 SDN multiple clusters: <10% funded, 33.7M PIN

### Tab 4 — Ask the Agent

Conversational interface powered by Claude API (claude-sonnet-4-6). See [Agent Design](#agent-design).

---

## Agent Design

### Tools (6 total)

```python
decompose_query(query, top_n, year)
    # Self-consistency: 3-way majority vote across parallel decompositions
    → {region, coverage_ceiling_pct, min_people_in_need, neglect_type, year, top_n}

query_gold_table(filters)
    → list[crisis_dict]   # from gold_ranked_crises.parquet

semantic_search(query, top_k)
    # TF-IDF RAG — for "similar to X" / "find crises like Y" queries
    → list[crisis_dict with similarity_score]

generate_briefing_notes(crises, query)
    # Neutral framing, forced counter-argument per crisis
    → str (markdown)

validate_ranking(top_crises, concerns, confidence_level, corrections)
    → {n_low_confidence, n_without_hrp, concerns, corrections, confidence_level}

red_team_challenge(top_crises, main_response_summary)
    # Second Claude call with adversarial auditor system prompt
    → str (critique markdown)
```

### ReAct Loop (6 THOUGHTs)

```
THOUGHT 1: Route — structured filters (query_gold_table) OR semantic similarity (semantic_search)?
THOUGHT 2: Decompose & query with self-consistency (3-way majority vote)
THOUGHT 3: Validate results — flag low-confidence, anomalies
THOUGHT 4: Generate briefing notes — neutral framing + counter-arguments
THOUGHT 5: Red-team challenge — adversarial second opinion on the ranking
THOUGHT 6: Synthesize final answer, strip internal reasoning
```

### Bias Guardrails (CMU odi-decisions.md)

- **Neutral framing:** "38% of requested funding received" — never loss/gain language
- **No anchoring:** never compare to historical baseline
- **Forced counter-argument:** every crisis briefing includes one reason the score may be overstated
- **Grounding mandate:** all numbers must come from tool observations, never hallucinated

### Session Memory (Reflexion)

Follow-up queries containing "tell me more", "#1", "top crisis", "elaborate" automatically inject the prior query's crisis list. Persisted to `data/session_memory.json`.

### RAG Semantic Search

TF-IDF vector index (sklearn, ngram 1–2, sublinear TF) built over the Gold crisis table. Each crisis becomes a rich text document with region, severity, neglect type, and funding descriptors. Used when query contains comparative language: "similar to Yemen", "like DRC", "find crises matching a certain profile".

---

## Workflow Integration

This system fits into **existing OCHA and CBPF analyst workflows** — it supports decisions, not automates them.

### Typical analyst session

```
1. Open dashboard → Ranked Table (most recent year, all regions)
2. Filter by region of interest + neglect type
3. Read gap scores + CI to understand ranking confidence
4. Switch to Sector Gaps tab → identify which specific clusters are underfunded
   beyond what the country-level aggregate shows
5. Ask the Agent: "Which of these should we prioritize for CBPF allocation?"
6. Agent returns ranked shortlist + briefing notes + red-team critique
7. Check EVPI panel: high uncertainty → recommend more detailed HNO first
8. Export ranked table for allocation committee memo
```

### Scaling to Databricks

- `06_refresh_from_databricks.py` reads from `~/Downloads/` locally — replace with `/Volumes/cmu_hackathon/common/unocha/` to run on-cluster with no other changes
- `03_gold_scoring.py` logs to MLflow — visible in Databricks MLflow UI when run on-cluster
- Gold parquet → Delta table: one additional `spark.createDataFrame().write.format("delta").saveAsTable()` call
- RAG TF-IDF → Databricks Vector Search Index: identical `search(query, top_k)` API surface

### What this does NOT do

- Does not replace HNO/HRP assessment processes — uses published figures only
- Does not recommend specific NGO recipients — surfaces country/sector priorities
- Does not automate allocation decisions — provides ranked shortlist for human review
- Does not access real-time conflict data (ACLED not integrated)

---

## CMU Knowledge Base Integration

| CMU Module | Concept Applied | Where |
|---|---|---|
| `dm-choice-mcdm.md` | Swing Weighting for defensible weights | `scoring_logic.py` SCORING_WEIGHTS |
| `dm-choice-mcdm.md` | Pareto filtering (cross-country only) | `scoring_logic.py` `score_dataframe()` |
| `dm-choice-mcdm.md` | Ringer Bid prevention (95th-pct reference) | `scoring_logic.py` `compute_gap_score()` |
| `dm-choice-mcdm.md` | Weight sensitivity "what-if" panel | `05_dashboard.py` sidebar |
| `dm-uncertainty.md` | Monte Carlo CI (P10/P50/P90, 1 000 runs) | `scoring_logic.py` `compute_confidence_interval()` |
| `dm-uncertainty.md` | EVPI: current vs. improved data quality | `scoring_logic.py` `compute_evpi()` |
| `agents.md` | ReAct 6-step Thought→Action→Observation | `04_agent.py` SYSTEM_PROMPT |
| `agents.md` | Self-evaluation (`validate_ranking` tool) | `04_agent.py` tool dispatch |
| `agents.md` | Reflexion session memory + persistence | `04_agent.py` / `session_memory.json` |
| `rag.md` | TF-IDF vector index, cosine similarity | `rag_search.py` `CrisisRAG` |
| `in-context-learning.md` | Self-consistency: 3-way majority vote | `04_agent.py` `_majority_vote_filters()` |
| `aimd-tools-agents-code.md` | Multi-agent debate / red-team pattern | `04_agent.py` `tool_red_team_challenge()` |
| `hallucinations.md` | Grounding mandate (numbers from tools only) | `04_agent.py` Hard Rules |
| `odi-decisions.md` | Neutral framing, no anchoring, counter-argument | `04_agent.py` SYSTEM_PROMPT |
| `oai-monitoring-governance.md` | Data quality drift alert banner | `05_dashboard.py` `_render_data_quality_alert()` |
| `de-foundations.md` | Medallion architecture (Bronze/Silver/Gold) | Pipeline structure |

---

## Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| HNO national PIN: 24 countries | Other countries absent from Gold ranking | Sector Gaps tab uses FTS data (44 countries) |
| INFORM max-severity aggregation | Multi-crisis countries (NGA, TCD) may overstate severity | Flagged in briefing counter-arguments |
| Funding data spans multi-year appeals | `funding_year` may lag `HNO year` | Silver join uses most-recent funding ≤ current year |
| AGR cluster 0% funded many rows | FTS rarely tracks Agriculture cluster separately | Valid signal — AGR seldom appears in OCHA cluster appeals |
| MLflow runs locally | Not synced to Databricks MLflow UI | Replace local artifact path with Unity Catalog path |
| CBPF Allocations file covers 2018 only | Historical, not 2024–2026 | `bronze_cbpf.parquet` (HDX HAPI) used as primary |

---

## Changelog

### 2026-05-18 (Session 3)

**Official Databricks data integration:**
- New `06_refresh_from_databricks.py`: full Bronze→Silver→Gold pipeline from official files
  - HNO 2024+2025+2026 (multi-year, `consecutive_years_underfunded` from real 3-year window)
  - FTS global (77 countries, 2024–2026)
  - HRP flags from `humanitarian-response-plans.csv` (78 countries)
  - INFORM severity from April 2026 XLSX (67 countries)
- Sector gaps: 18 countries / 50 rows → **44 countries / 359 rows**
- Top crises: Burkina Faso, Mali, Venezuela — 3-year structural neglect, <16% coverage

**New Sector Gaps tab (Tab 3):**
- FTS cluster × HNO code join, 44-country heatmap, top-20 bar chart
- Surfaces aggregate-masking-sector-gap cases (PDF bonus task)

**Agent enhancements:**
- `semantic_search` tool: TF-IDF RAG (`rag_search.py`) for similarity queries
- `red_team_challenge` tool: adversarial second Claude call to audit rankings
- Self-consistency on `decompose_query`: 3-way majority vote
- Session memory persisted to `data/session_memory.json`
- Fixed: memory save was after `return` (unreachable) — moved before

**Scoring enhancements:**
- Monte Carlo CI replaces ±20% linear sensitivity (P10/P50/P90, 1 000 runs)
- `compute_evpi()`: quantifies value of improving data quality
- EVPI panel added to Ranked Table tab

### 2026-05-16 (Session 2)

**Bug fixes:** Year dropdown cast error · Multi-year Silver deduplication · Pareto cross-year over-flagging · INFORM join row explosion · `THOUGHT N:` leak to user output · All paths fixed to `Path(__file__).parent`

**CMU enhancements:** Weight Sensitivity panel · Data quality drift alert banner · `★ frontier` Pareto display · Reflexion session memory (initial)

### 2026-05-15 (Session 1)

**Initial build:** Bronze ingest · Silver normalize · Gold MCDM scoring + MLflow · 4-tool ReAct agent · Ranked Table + World Map + Agent chat · Sanity check: SSD/YEM/SOM/HTI/AFG/SDN in top 20 ✓
