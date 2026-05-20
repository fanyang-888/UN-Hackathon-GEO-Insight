"""
05_evaluate_agent.py — LLM-as-Judge evaluation for the Geo-Insight agent.

Day 2 pattern (Z's MLflow session):
  1. Define golden test cases (eval dataset)
  2. Run agent on each case
  3. Score with Claude as LRM judge on 4 axes (0-4 each):
       correctness  — right countries / neglect types returned?
       groundedness — no hallucinated numbers?
       neutrality   — no advocacy / loss-framing language?
       completeness — query fully answered?
  4. Log everything to MLflow experiment "geo-insight-eval"

Run:
    python pipeline/05_evaluate_agent.py
"""

import os, sys, json, time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE / "agent"))

from dotenv import load_dotenv
load_dotenv(_HERE / ".env", override=True)

import mlflow
import anthropic

mlflow.set_experiment("geo-insight-eval")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL  = "claude-sonnet-4-6"

# ── Golden evaluation dataset ─────────────────────────────────────────────────
# Four cases from the hackathon PDF sample queries
GOLDEN_CASES = [
    {
        "id":    "global_top_underfunded",
        "query": "Which crises have the highest proportion of people in need but the lowest fund allocations?",
        "expect_countries": ["BFA", "MLI", "VEN", "HTI", "MMR"],
        "expect_route":     "structured",
        "description":      "Global top underfunded — should surface BFA/MLI/VEN structural neglect",
    },
    {
        "id":    "hrp_under_20pct",
        "query": "Show me countries with active HRPs where funding received is under 20%.",
        "expect_countries": ["BFA", "MLI", "VEN"],
        "expect_route":     "structured",
        "description":      "has_hrp=True + coverage<20% — BFA (13.8%), MLI (16%), VEN (15.4%) all qualify",
    },
    {
        "id":    "structural_neglect_3yr",
        "query": "Which crises have been underfunded for 3 or more consecutive years?",
        "expect_countries": ["BFA", "MLI", "VEN", "HTI"],
        "expect_route":     "structured",
        "description":      "Should classify as neglect_type=structural and surface Sahel + Lat Am",
    },
    {
        "id":    "semantic_similar_to_yemen",
        "query": "Find crises with a similar profile to Yemen — high severity, low coverage.",
        "expect_countries": ["HTI", "MMR", "BFA", "SSD"],
        "expect_route":     "semantic",
        "description":      "Semantic/RAG path — should call semantic_search, not just query_gold_table",
    },
]

# ── LRM Judge prompt ─────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert evaluator auditing responses from a humanitarian data analysis agent.
Score the agent's response on FOUR dimensions, each 0–4 (integers only):

correctness (0-4):
  4 = All expected countries/topics appear prominently in the top results
  3 = Most expected items present; minor omissions
  2 = Some expected items present but key ones missing
  1 = Mostly wrong countries or topics
  0 = Completely off-target

groundedness (0-4):
  4 = Every number is explicitly attributed to a data source / tool observation; no invented figures
  3 = Nearly all numbers grounded; one minor unclear attribution
  2 = Some numbers appear without clear sourcing
  1 = Several figures lack attribution
  0 = Response contains hallucinated numbers

neutrality (0-4):
  4 = Pure coverage-shortfall framing; no advocacy, no loss/gain language
  3 = Mostly neutral; one mild framing slip
  2 = Some advocacy or emotionally charged language
  1 = Clear advocacy framing throughout
  0 = Strong advocacy / alarmist / misleading language

completeness (0-4):
  4 = Query fully answered; all sub-parts addressed
  3 = Main question answered; minor sub-parts missed
  2 = Core question partially answered
  1 = Only superficial answer to part of the query
  0 = Query not answered

Return ONLY valid JSON, no explanation, no markdown fences:
{"correctness": N, "groundedness": N, "neutrality": N, "completeness": N, "judge_notes": "one sentence"}
"""


def _judge_response(query: str, response: str, case: dict) -> dict:
    """Call Claude as LRM judge. Returns scores dict."""
    user_msg = (
        f"Query: {query}\n\n"
        f"Expected countries (if applicable): {case.get('expect_countries', 'any')}\n"
        f"Expected neglect type (if applicable): {case.get('expect_neglect', 'any')}\n\n"
        f"Agent response:\n{response[:3000]}"   # truncate to stay within context
    )
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        m = __import__("re").search(r"\{.*\}", text, __import__("re").DOTALL)
        return json.loads(m.group()) if m else {"error": "parse_failed", "raw": text}
    except Exception as e:
        return {"error": str(e)}


def run_evaluation():
    print("=" * 60)
    print("Geo-Insight Agent — LLM-as-Judge Evaluation")
    print("=" * 60)

    # Lazy import agent (avoids loading at module level)
    from agent_runner import run_query

    results = []
    for case in GOLDEN_CASES:
        print(f"\n▶ [{case['id']}] {case['query'][:60]}…")

        with mlflow.start_run(run_name=case["id"]):
            # ── Log case metadata ────────────────────────────────────────────
            mlflow.log_param("query",              case["query"])
            mlflow.log_param("case_id",            case["id"])
            mlflow.log_param("expect_route",       case.get("expect_route", "any"))
            mlflow.log_param("expect_countries",   str(case.get("expect_countries", [])))

            # ── Run agent ───────────────────────────────────────────────────
            t0 = time.time()
            try:
                result  = run_query(case["query"])
                response_text   = result["response_text"]
                route_used      = "semantic" if any(
                    "semantic" in str(c) for c in result.get("filters_applied", {}).values()
                ) else result.get("filters_applied", {}).get("route", "structured")
                num_crises      = len(result.get("crises", []))
                actual_isos     = [c.get("country_iso3", "") for c in result.get("crises", [])]
                latency         = round(time.time() - t0, 2)
                agent_error     = None
            except Exception as e:
                response_text   = f"AGENT ERROR: {e}"
                route_used      = "error"
                num_crises      = 0
                actual_isos     = []
                latency         = round(time.time() - t0, 2)
                agent_error     = str(e)

            # ── Coverage check (how many expected countries appeared) ────────
            expected = case.get("expect_countries", [])
            hits     = [iso for iso in expected if iso in actual_isos]
            coverage = len(hits) / len(expected) if expected else 1.0

            # ── Route check ─────────────────────────────────────────────────
            route_correct = (
                case.get("expect_route", route_used) == route_used
                or case.get("expect_route") == "any"
            )

            # ── LRM judge ───────────────────────────────────────────────────
            print(f"   Running LRM judge…")
            scores = _judge_response(case["query"], response_text, case)

            # ── Log metrics ─────────────────────────────────────────────────
            mlflow.log_metric("latency_seconds",    latency)
            mlflow.log_metric("num_crises",         num_crises)
            mlflow.log_metric("country_coverage",   round(coverage, 3))
            mlflow.log_metric("route_correct",      int(route_correct))

            if "error" not in scores:
                for dim in ["correctness", "groundedness", "neutrality", "completeness"]:
                    mlflow.log_metric(f"judge_{dim}", scores.get(dim, 0))
                total = sum(scores.get(d, 0) for d in ["correctness", "groundedness", "neutrality", "completeness"])
                mlflow.log_metric("judge_total_16", total)
                mlflow.log_metric("judge_pct",      round(total / 16 * 100, 1))
                print(f"   Scores: {scores}")
            else:
                print(f"   Judge error: {scores}")

            if agent_error:
                mlflow.set_tag("agent_error", agent_error[:200])

            # ── Log artifacts ────────────────────────────────────────────────
            mlflow.log_text(response_text,          "agent_response.md")
            mlflow.log_text(json.dumps(scores, indent=2), "judge_scores.json")
            mlflow.log_text(
                json.dumps({"expected": expected, "actual": actual_isos, "hits": hits}, indent=2),
                "country_coverage.json",
            )

            # ── Prompt registry: log SYSTEM_PROMPT version ──────────────────
            try:
                from agent_runner import run_query as _rq
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("agent04", _HERE / "agent" / "04_agent.py")
                _mod  = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                mlflow.log_text(_mod.SYSTEM_PROMPT, "system_prompt_v1.txt")
            except Exception:
                pass

            results.append({
                "case_id":         case["id"],
                "latency":         latency,
                "num_crises":      num_crises,
                "country_coverage": coverage,
                "route_correct":   route_correct,
                "scores":          scores,
                "agent_error":     agent_error,
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Case':<30} {'Latency':>8} {'Crises':>7} {'Coverage':>9} {'Judge/16':>9}")
    print("-" * 60)
    for r in results:
        sc = r["scores"]
        total = sum(sc.get(d, 0) for d in ["correctness", "groundedness", "neutrality", "completeness"]) \
                if "error" not in sc else "ERR"
        print(f"{r['case_id']:<30} {r['latency']:>7.1f}s {r['num_crises']:>7} "
              f"{r['country_coverage']:>8.0%} {str(total):>9}")
    print("=" * 60)
    print("\n✅ Results logged to MLflow experiment 'geo-insight-eval'")
    print("   View: mlflow ui  →  http://localhost:5000")


if __name__ == "__main__":
    run_evaluation()
