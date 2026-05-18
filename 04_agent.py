"""
Claude API agentic layer for the UNOCHA Geo-Insight Command Center.

The agent:
  1. Receives a natural-language query from the user
  2. Uses tool_use to decompose the query into structured filters
  3. Queries the Gold Delta/parquet table with those filters
  4. Generates a ranked response with per-crisis briefing notes

Run standalone:  python 04_agent.py
Or import run_query() into 05_dashboard.py.
"""
import os
import json
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

GOLD_PATH = Path(__file__).parent / "data/gold/gold_ranked_crises.parquet"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ---------------------------------------------------------------------------
# Tool definitions (Claude will call these)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "decompose_query",
        "description": (
            "Parse a natural-language humanitarian crisis query into structured filter criteria. "
            "Extract geographic scope, severity floor, funding coverage ceiling, sector focus, and year range. "
            "If a field is not specified in the query, return null for it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": ["string", "null"],
                    "description": "Geographic region e.g. 'Sub-Saharan Africa', 'Middle East', 'South Asia'. Null if global.",
                },
                "country_iso3": {
                    "type": ["string", "null"],
                    "description": "Specific ISO-3 country code if a single country is requested.",
                },
                "max_coverage_pct": {
                    "type": ["number", "null"],
                    "description": "Only return crises where funding coverage is below this percentage (0-100). E.g. 10 means <10% funded.",
                },
                "min_people_in_need": {
                    "type": ["number", "null"],
                    "description": "Minimum people in need threshold (absolute number). Null means use system default.",
                },
                "sector": {
                    "type": ["string", "null"],
                    "description": "Sector focus e.g. 'food security', 'health', 'shelter'. Null if not sector-specific.",
                },
                "year": {
                    "type": ["integer", "null"],
                    "description": "Specific year to filter on. Null means most recent available.",
                },
                "neglect_type": {
                    "type": ["string", "null"],
                    "enum": ["structural", "acute", "improving", "ongoing", None],
                    "description": "Filter by neglect classification. 'structural' = underfunded 3+ consecutive years.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top results to return (default 10, max 30).",
                    "default": 10,
                },
                "query_rationale": {
                    "type": "string",
                    "description": "Brief explanation of how you interpreted the query and which filters you applied.",
                },
            },
            "required": ["top_n", "query_rationale"],
        },
    },
    {
        "name": "query_gold_table",
        "description": (
            "Query the Gold ranked crisis table with structured filters and return matching crises "
            "sorted by gap score descending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": ["string", "null"]},
                "country_iso3": {"type": ["string", "null"]},
                "max_coverage_pct": {"type": ["number", "null"]},
                "min_people_in_need": {"type": ["number", "null"]},
                "neglect_type": {"type": ["string", "null"]},
                "year": {"type": ["integer", "null"]},
                "top_n": {"type": "integer", "default": 10},
            },
            "required": ["top_n"],
        },
    },
    {
        "name": "generate_briefing_notes",
        "description": (
            "Generate concise humanitarian briefing notes for the top-ranked crises. "
            "Each note should explain WHY the crisis is overlooked, cite the gap score components, "
            "and surface any data uncertainty."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "crises": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of crisis dicts from query_gold_table.",
                },
                "original_query": {
                    "type": "string",
                    "description": "The original user query for context.",
                },
            },
            "required": ["crises", "original_query"],
        },
    },
    {
        "name": "validate_ranking",
        "description": (
            "Self-evaluation step (CMU hallucinations.md — Self-Evaluation strategy). "
            "Before returning final results, critically audit the ranking for: "
            "(1) possible data quality issues that inflate scores, "
            "(2) crises whose 0% coverage reflects no HRP rather than true neglect, "
            "(3) any anchoring to historical funding levels in the interpretation, "
            "(4) whether the top-ranked crises pass a sanity check against known major crises. "
            "Return a structured audit with confidence_level and any corrections needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_crises": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "The ranked crisis list to audit.",
                },
                "concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific concerns about data quality or ranking validity.",
                },
                "confidence_level": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Overall confidence in the ranking given data quality.",
                },
                "corrections": {
                    "type": "string",
                    "description": "Any corrections or caveats to surface in the final response.",
                },
            },
            "required": ["top_crises", "concerns", "confidence_level", "corrections"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Region → ISO-3 lookup (partial, extend as needed)
REGION_TO_ISO3: dict[str, list[str]] = {
    "sub-saharan africa": [
        "SSD", "SOM", "COD", "CAF", "NER", "MLI", "TCD", "BFA", "ETH",
        "MOZ", "ZWE", "ZMB", "UGA", "RWA", "BDI", "TGO", "GNB", "SLE",
        "LBR", "GIN", "CMR", "NGA", "AGO", "MDG", "MWI", "TZA", "KEN",
    ],
    "middle east": ["YEM", "SYR", "IRQ", "LBN", "PSE", "LBY"],
    "north africa": ["LBY", "SDN", "EGY", "DZA", "TUN", "MAR"],
    "south asia": ["AFG", "PAK", "BGD", "NPL", "MMR", "LKA"],
    "latin america": ["HTI", "VEN", "COL", "GTM", "HND", "SLV", "NIC"],
    "central asia": ["TJK", "KGZ", "UZB", "TKM"],
    "east africa": ["ETH", "SOM", "SSD", "KEN", "UGA", "TZA", "DJI", "ERI"],
    "west africa": ["NER", "MLI", "BFA", "NGA", "CMR", "GIN", "SLE", "LBR"],
    "horn of africa": ["ETH", "SOM", "ERI", "DJI"],
}


def _load_gold() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold table not found at {GOLD_PATH}. Run 03_gold_scoring.py first."
        )
    return pd.read_parquet(GOLD_PATH)


def tool_query_gold_table(params: dict) -> list[dict]:
    df = _load_gold()

    # Year filter
    if params.get("year"):
        df = df[df["year"] == int(params["year"])]
    else:
        # Default: most recent year per country
        if "year" in df.columns:
            df = df.loc[df.groupby("country_iso3")["year"].idxmax()]

    # Region filter
    region = params.get("region")
    if region:
        region_key = region.lower()
        match_key = next((k for k in REGION_TO_ISO3 if region_key in k or k in region_key), None)
        if match_key:
            allowed = REGION_TO_ISO3[match_key]
            df = df[df["country_iso3"].isin(allowed)]

    # Country filter
    if params.get("country_iso3"):
        df = df[df["country_iso3"] == params["country_iso3"].upper()]

    # Coverage ceiling filter
    if params.get("max_coverage_pct") is not None:
        df = df[df["coverage_pct"] <= float(params["max_coverage_pct"])]

    # PIN floor
    if params.get("min_people_in_need") is not None:
        df = df[df["people_in_need"].fillna(0) >= float(params["min_people_in_need"])]

    # Neglect type filter
    if params.get("neglect_type"):
        df = df[df["neglect_type"] == params["neglect_type"]]

    top_n = min(int(params.get("top_n", 10)), 30)
    result = df.sort_values("gap_score", ascending=False).head(top_n)

    # Serialize — replace NaN with None for JSON
    records = result[[
        "country_iso3", "country_name", "year", "gap_score", "coverage_pct",
        "people_in_need", "funding_requested", "funding_received",
        "inform_severity", "neglect_type", "consecutive_years_underfunded",
        "has_hrp", "low_confidence",
    ]].copy()
    records = records.where(pd.notna(records), None)
    return records.to_dict(orient="records")


def tool_validate_ranking(top_crises: list[dict], concerns: list[str],
                          confidence_level: str, corrections: str) -> str:
    """
    Self-evaluation audit (CMU hallucinations.md — Self-Evaluation strategy).
    The agent calls this to critique its own ranking before finalizing.
    We echo back a structured summary that becomes part of the observation chain.
    """
    n_low_conf = sum(1 for c in top_crises if c.get("low_confidence"))
    n_no_hrp = sum(1 for c in top_crises if not c.get("has_hrp"))
    n_structural = sum(1 for c in top_crises if c.get("neglect_type") == "structural")

    audit = {
        "confidence_level": confidence_level,
        "n_low_confidence_in_top": n_low_conf,
        "n_without_hrp_in_top": n_no_hrp,
        "n_structural_neglect_in_top": n_structural,
        "concerns_raised": concerns,
        "corrections": corrections,
        "reminder": (
            "Surface confidence_level and all concerns in your final response. "
            "If confidence_level is 'low', lead with a data quality warning. "
            "Do not suppress concerns to appear more authoritative."
        ),
    }
    return json.dumps(audit, indent=2)


def tool_generate_briefing_notes(crises: list[dict], original_query: str) -> str:
    """Generate per-crisis briefing notes inline (called by the agent loop)."""
    if not crises:
        return "No crises matched the query filters."
    notes = []
    for c in crises:
        pin = c.get("people_in_need")
        pin_str = f"{int(pin):,}" if pin else "unknown"
        cov = c.get("coverage_pct")
        cov_str = f"{cov:.1f}%" if cov is not None else "unknown"
        gap = c.get("gap_score")
        gap_str = f"{gap:.3f}" if gap is not None else "N/A"
        hrp = "HRP in place" if c.get("has_hrp") else "no formal HRP"
        neglect = c.get("neglect_type", "unknown")
        years_under = c.get("consecutive_years_underfunded", 0)
        confidence = " [LOW CONFIDENCE — missing or stale data]" if c.get("low_confidence") else ""

        note = (
            f"**{c.get('country_name', c.get('country_iso3'))} ({c.get('year', 'N/A')})**{confidence}\n"
            f"Gap score: {gap_str} | Funding coverage: {cov_str} | People in need: {pin_str} | {hrp}\n"
            f"Neglect classification: {neglect}"
            + (f" ({years_under} consecutive years underfunded)" if years_under > 1 else "")
        )
        notes.append(note)
    return "\n\n".join(notes)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a humanitarian data analyst for the UN OCHA Geo-Insight Command Center.

## ReAct Workflow (follow this exactly — CMU agents.md)

For every query, execute this Thought→Action→Observation loop:

THOUGHT 1: Interpret the query. What geographic scope, coverage ceiling, need floor, and sector does it imply?
  → ACTION: call decompose_query
  → OBSERVATION: note the structured filters extracted

THOUGHT 2: Are the filters reasonable? Could this query be interpreted differently? Note any ambiguity.
  → ACTION: call query_gold_table with the filters
  → OBSERVATION: note how many crises returned, which top-ranked, any surprising results

THOUGHT 3: Do any top results look suspicious? (0% coverage with no HRP, very stale data, unusual PIN figures?)
  → ACTION: call validate_ranking — audit the list for data quality issues and anchoring bias
  → OBSERVATION: note confidence level and any corrections

THOUGHT 4: Now generate the briefing notes grounded ONLY in tool observations.
  → ACTION: call generate_briefing_notes
  → OBSERVATION: read the notes

THOUGHT 5: Synthesize final response.

## Critical Bias Guardrails (CMU odi-decisions.md)

NEUTRAL FRAMING — Never use loss/death language or gain language. Always express as coverage shortfall:
  ✓ "Funding coverage stands at 23%, leaving a 77% gap against the requested amount."
  ✗ "Millions face death without immediate funding."

NO ANCHORING — Do not reference historical funding levels as a baseline or anchor for the current gap.
  ✓ "The current coverage ratio is 18%."
  ✗ "Funding has dropped from the 2022 level of 45%."

COUNTER-ARGUMENT REQUIRED — For every top-ranked crisis, state the strongest reason its score might be overstated (e.g. bilateral funding not captured in FTS, no HRP so 0% is an assumption not a measurement, data >18 months old).

DE NOVO SCORING — Evaluate each crisis on its current data only. Do not let the name or reputation of a well-known crisis (e.g. Yemen, Syria) anchor your interpretation before seeing the numbers.

## Hard Rules

- NEVER fabricate or interpolate funding or population figures. Only cite numbers returned by tools.
- ALWAYS surface low_confidence=True flags with explicit explanation.
- Distinguish structural neglect (≥3 consecutive underfunded years) from acute emergencies.
- This system is decision SUPPORT — never say a coordinator "should" fund a specific crisis. Say "the data suggests X warrants further assessment."
- If validate_ranking returns confidence_level="low", lead the final response with a prominent data quality warning."""


import re as _re

def _strip_thoughts(text: str) -> str:
    """Remove internal THOUGHT N: reasoning paragraphs from the final response."""
    paragraphs = text.split("\n\n")
    filtered = []
    for p in paragraphs:
        # Strip markdown bold/italic markers before checking
        clean = _re.sub(r"[\*_]{1,2}", "", p.lstrip())
        if _re.match(r"THOUGHT\s+\d+", clean):
            continue
        # Also drop lone horizontal rules that directly follow a stripped THOUGHT
        if p.strip() in ("---", "***", "___") and not filtered:
            continue
        filtered.append(p)
    return "\n\n".join(filtered).strip()


# Reflexion-style session memory (agents.md): persist last query context for follow-up questions
_session_memory: dict = {}


def run_query(user_query: str, verbose: bool = False) -> dict:
    """
    Run a user query through the agent and return a structured result.

    Implements Reflexion-style session memory (agents.md): if the query references
    a prior result ("tell me more about #1", "what about the top crisis?"), the
    previous crisis list is injected as context.

    Returns:
        {
            "response_text": str,         # Final assistant message for display
            "crises": list[dict],         # Raw crisis records
            "filters_applied": dict,      # Decomposed query filters
            "query_rationale": str,       # Agent's interpretation
        }
    """
    # Detect follow-up references and inject prior context
    followup_triggers = ["#1", "#2", "#3", "top crisis", "first crisis", "that crisis",
                         "tell me more", "more about", "elaborate", "explain further"]
    is_followup = any(t in user_query.lower() for t in followup_triggers)

    prior_context = ""
    if is_followup and _session_memory.get("last_crises"):
        top = _session_memory["last_crises"][:3]
        prior_context = (
            "\n\n[Session Memory — prior query results, use as context for this follow-up]:\n"
            + "\n".join(
                f"#{i+1}. {c.get('country_name','?')} — gap_score={c.get('gap_score','?')}, "
                f"coverage={c.get('coverage_ratio',0)*100:.1f}%, PIN={c.get('people_in_need',0):,.0f}"
                for i, c in enumerate(top)
            )
        )

    messages = [{"role": "user", "content": user_query + prior_context}]
    crises = []
    filters_applied = {}
    query_rationale = ""
    tool_results_for_next = []

    max_iterations = 6
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n[Agent iteration {iteration + 1}]")

        kwargs = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "tools": TOOLS,
            "messages": messages,
        }

        response = client.messages.create(**kwargs)

        if verbose:
            print(f"  stop_reason: {response.stop_reason}")

        # Collect assistant message
        assistant_content = []
        tool_calls = []

        for block in response.content:
            assistant_content.append(block)
            if block.type == "tool_use":
                tool_calls.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            # Extract final text response; strip internal THOUGHT reasoning lines
            text_blocks = [b for b in response.content if hasattr(b, "text")]
            raw = "\n".join(b.text for b in text_blocks)
            response_text = _strip_thoughts(raw)
            return {
                "response_text": response_text,
                "crises": crises,
                "filters_applied": filters_applied,
                "query_rationale": query_rationale,
            }
            # Reflexion: save results to session memory for follow-up queries
            if crises:
                _session_memory["last_crises"] = crises
                _session_memory["last_query"] = user_query

        if response.stop_reason != "tool_use" or not tool_calls:
            # Unexpected stop — return what we have
            text_blocks = [b for b in response.content if hasattr(b, "text")]
            response_text = _strip_thoughts("\n".join(b.text for b in text_blocks))
            return {
                "response_text": response_text or "No results returned.",
                "crises": crises,
                "filters_applied": filters_applied,
                "query_rationale": query_rationale,
            }

        # Execute tool calls
        tool_results = []
        for tc in tool_calls:
            tool_name = tc.name
            tool_input = tc.input
            if verbose:
                print(f"  Tool call: {tool_name}({json.dumps(tool_input, default=str)[:200]})")

            try:
                if tool_name == "decompose_query":
                    query_rationale = tool_input.get("query_rationale", "")
                    filters_applied = {k: v for k, v in tool_input.items() if k != "query_rationale"}
                    result_content = json.dumps(tool_input)

                elif tool_name == "query_gold_table":
                    crises = tool_query_gold_table(tool_input)
                    result_content = json.dumps(crises, default=str)
                    if verbose:
                        print(f"    → {len(crises)} crises returned")

                elif tool_name == "generate_briefing_notes":
                    briefings = tool_generate_briefing_notes(
                        tool_input.get("crises", crises),
                        tool_input.get("original_query", user_query),
                    )
                    result_content = briefings

                elif tool_name == "validate_ranking":
                    result_content = tool_validate_ranking(
                        top_crises=tool_input.get("top_crises", crises),
                        concerns=tool_input.get("concerns", []),
                        confidence_level=tool_input.get("confidence_level", "medium"),
                        corrections=tool_input.get("corrections", ""),
                    )

                else:
                    result_content = json.dumps({"error": f"Unknown tool: {tool_name}"})

            except Exception as e:
                result_content = json.dumps({"error": str(e), "trace": traceback.format_exc()[:500]})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result_content,
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "response_text": "Agent reached max iterations without completing.",
        "crises": crises,
        "filters_applied": filters_applied,
        "query_rationale": query_rationale,
    }


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "Which crises have the highest proportion of people in need but the lowest fund allocations?",
    "Show me countries with active HRPs where funding received is under 10%.",
    "Which regions are consistently underfunded across 3 or more consecutive years?",
    "Show me acute food insecurity hotspots in Sub-Saharan Africa.",
]


def main():
    print("=" * 60)
    print("UNOCHA Geo-Insight — Agent CLI")
    print("=" * 60)

    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\nDemo queries:")
        for i, q in enumerate(DEMO_QUERIES, 1):
            print(f"  {i}. {q}")
        choice = input("\nEnter query number or type your own: ").strip()
        try:
            query = DEMO_QUERIES[int(choice) - 1]
        except (ValueError, IndexError):
            query = choice

    print(f"\nQuery: {query}\n")
    result = run_query(query, verbose=True)
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(result["response_text"])
    if result["query_rationale"]:
        print(f"\n[Query interpretation: {result['query_rationale']}]")


if __name__ == "__main__":
    main()
