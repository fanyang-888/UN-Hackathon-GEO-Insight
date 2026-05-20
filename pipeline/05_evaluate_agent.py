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

import os, sys, json, time, hashlib
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
Score the agent's response on FOUR dimensions, each 0–4 (integers only).

CRITICAL: For each score, you MUST quote the specific phrase from the agent response that informed
your rating. Your scores are invalid without direct textual evidence.

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
{
  "correctness": N,
  "correctness_evidence": "quoted phrase from response",
  "groundedness": N,
  "groundedness_evidence": "quoted phrase from response",
  "neutrality": N,
  "neutrality_evidence": "quoted phrase from response",
  "completeness": N,
  "completeness_evidence": "quoted phrase from response",
  "judge_notes": "one sentence overall assessment"
}
"""


def _call_judge_once(user_msg: str) -> dict:
    """Single LRM judge call. Returns parsed scores dict or error."""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        m = __import__("re").search(r"\{.*\}", text, __import__("re").DOTALL)
        return json.loads(m.group()) if m else {"error": "parse_failed", "raw": text}
    except Exception as e:
        return {"error": str(e)}


def _judge_response(query: str, response: str, case: dict) -> dict:
    """Call Claude as LRM judge TWICE and report scores + per-dimension variance.

    Debiasing strategy (CMU aimd-scaling-evaluation.md):
    - Run judge twice independently (reduces single-sample verbosity/position bias)
    - Average scores across both runs
    - Flag dimensions where the two calls differ by >1 point (unstable eval signal)
    """
    user_msg = (
        f"Query: {query}\n\n"
        f"Expected countries (if applicable): {case.get('expect_countries', 'any')}\n"
        f"Expected neglect type (if applicable): {case.get('expect_neglect', 'any')}\n\n"
        f"Agent response:\n{response[:3000]}"
    )

    run1 = _call_judge_once(user_msg)
    run2 = _call_judge_once(user_msg)

    dims = ["correctness", "groundedness", "neutrality", "completeness"]

    if "error" in run1 and "error" in run2:
        return run1  # both failed — return error

    # Use whichever run(s) succeeded
    valid_runs = [r for r in [run1, run2] if "error" not in r]
    merged: dict = {}

    for dim in dims:
        vals = [r[dim] for r in valid_runs if dim in r]
        if vals:
            merged[dim] = round(sum(vals) / len(vals), 2)  # averaged score
            merged[f"{dim}_variance"] = max(vals) - min(vals)  # 0 if single run
        # carry evidence from first valid run
        ev_key = f"{dim}_evidence"
        for r in valid_runs:
            if ev_key in r:
                merged[ev_key] = r[ev_key]
                break

    # Stability flag — any dimension where two judges disagree by >1
    unstable_dims = [d for d in dims if merged.get(f"{d}_variance", 0) > 1]
    merged["judge_notes"] = valid_runs[0].get("judge_notes", "")
    merged["judge_score_variance_max"] = max(
        (merged.get(f"{d}_variance", 0) for d in dims), default=0
    )
    merged["unstable_dimensions"] = unstable_dims

    return merged


def _check_route(result: dict, expect_route: str) -> str:
    """Extract route label from agent result dict."""
    filters = result.get("filters_applied", {})
    if isinstance(filters, dict):
        route = filters.get("route", "")
    else:
        route = ""
    # Fallback: check if any filter value contains 'semantic'
    if not route:
        route = "semantic" if any("semantic" in str(v) for v in (filters.values() if isinstance(filters, dict) else [])) else "structured"
    return route


def run_pass_k_consistency(run_query_fn, n_runs: int = 3):
    """
    Pass_k stability test (CMU agentic-tech.md — Pass_k reliability metric).

    Runs the semantic routing case n_runs times and measures:
    - route_consistency: fraction of runs that used semantic_search correctly
    - country_set_consistency: fraction of runs returning the same top-3 countries

    Production standard: Pass_8 ≥ 80% for mission-critical applications.
    """
    SEMANTIC_CASE = next(c for c in GOLDEN_CASES if c["id"] == "semantic_similar_to_yemen")
    query         = SEMANTIC_CASE["query"]
    expect_route  = SEMANTIC_CASE["expect_route"]
    expect_isos   = set(SEMANTIC_CASE["expect_countries"])

    print(f"\n▶ [pass_k_consistency] Running '{query[:50]}…' × {n_runs}")

    routes        = []
    top3_sets     = []

    for i in range(n_runs):
        try:
            result      = run_query_fn(query)
            route_used  = _check_route(result, expect_route)
            routes.append(route_used)
            actual_isos = [c.get("country_iso3", "") for c in result.get("crises", [])]
            top3_sets.append(set(actual_isos[:3]))
            print(f"   run {i+1}: route={route_used}, top3={actual_isos[:3]}")
        except Exception as e:
            print(f"   run {i+1}: ERROR — {e}")
            routes.append("error")
            top3_sets.append(set())

    route_consistency   = sum(1 for r in routes if r == expect_route) / n_runs
    # Country consistency: pairwise Jaccard similarity of top-3 sets across runs
    if len(top3_sets) >= 2:
        pairwise_jaccards = []
        for a_idx in range(len(top3_sets)):
            for b_idx in range(a_idx + 1, len(top3_sets)):
                a, b = top3_sets[a_idx], top3_sets[b_idx]
                union = a | b
                pairwise_jaccards.append(len(a & b) / len(union) if union else 1.0)
        country_set_consistency = sum(pairwise_jaccards) / len(pairwise_jaccards)
    else:
        country_set_consistency = 1.0

    print(f"   Route consistency ({n_runs} runs): {route_consistency:.0%}")
    print(f"   Country-set Jaccard consistency:   {country_set_consistency:.0%}")

    with mlflow.start_run(run_name="pass_k_consistency"):
        mlflow.log_param("n_runs",        n_runs)
        mlflow.log_param("case_id",       SEMANTIC_CASE["id"])
        mlflow.log_param("expect_route",  expect_route)
        mlflow.log_metric(f"route_consistency_{n_runs}",        round(route_consistency, 3))
        mlflow.log_metric(f"country_set_consistency_{n_runs}",  round(country_set_consistency, 3))
        mlflow.log_text(json.dumps({
            "routes": routes,
            "top3_sets": [list(s) for s in top3_sets],
        }, indent=2), "pass_k_detail.json")

    return {
        "route_consistency":        route_consistency,
        "country_set_consistency":  country_set_consistency,
    }


def run_rag_evaluation(run_query_fn):
    """
    RAG retrieval quality metrics (CMU aimd-scaling-evaluation.md — RAGAS framework).

    Measures:
    - context_precision_at_5: of the top-5 retrieved crises, what fraction appear in
      the expected countries list? (equivalent to RAGAS Context Precision)
    - route_correctness: did the agent correctly route to semantic_search?
    - rag_vs_structured_overlap: do the same countries appear if we force structured path?
      Tests whether the RAG path actually adds value over exact filtering.
    """
    SEMANTIC_CASE    = next(c for c in GOLDEN_CASES if c["id"] == "semantic_similar_to_yemen")
    query            = SEMANTIC_CASE["query"]
    expect_isos      = set(SEMANTIC_CASE["expect_countries"])
    expect_route     = SEMANTIC_CASE["expect_route"]

    print(f"\n▶ [rag_evaluation] Context Precision + RAG vs. Structured baseline")

    # ── Run 1: normal agent (RAG path) ──────────────────────────────────────
    result       = run_query_fn(query)
    actual_isos  = [c.get("country_iso3", "") for c in result.get("crises", [])]
    top5_isos    = actual_isos[:5]
    route_used   = _check_route(result, expect_route)

    hits_in_top5         = [iso for iso in top5_isos if iso in expect_isos]
    context_precision_5  = len(hits_in_top5) / 5 if top5_isos else 0.0
    route_correct        = int(route_used == expect_route)

    print(f"   RAG top-5:          {top5_isos}")
    print(f"   Context Precision@5: {context_precision_5:.0%} ({len(hits_in_top5)}/5 expected)")
    print(f"   Route correct:       {route_correct} (used={route_used}, expect={expect_route})")

    # ── Run 2: structured baseline (use hrp_under_20pct as proxy for similar countries) ──
    struct_case   = next(c for c in GOLDEN_CASES if c["id"] == "hrp_under_20pct")
    struct_result = run_query_fn(struct_case["query"])
    struct_isos   = set(c.get("country_iso3", "") for c in struct_result.get("crises", []))

    rag_set    = set(actual_isos[:5])
    union_size = len(rag_set | struct_isos)
    rag_vs_structured_overlap = (
        len(rag_set & struct_isos) / union_size if union_size else 1.0
    )
    print(f"   Structured baseline: {sorted(struct_isos)}")
    print(f"   RAG ∩ Structured / RAG ∪ Structured: {rag_vs_structured_overlap:.0%}")

    with mlflow.start_run(run_name="rag_evaluation"):
        mlflow.log_param("case_id",       SEMANTIC_CASE["id"])
        mlflow.log_param("expect_isos",   str(sorted(expect_isos)))
        mlflow.log_metric("context_precision_at_5",    round(context_precision_5, 3))
        mlflow.log_metric("route_correct",             route_correct)
        mlflow.log_metric("rag_vs_structured_overlap", round(rag_vs_structured_overlap, 3))
        mlflow.log_text(json.dumps({
            "rag_top5":          top5_isos,
            "structured_top":    sorted(struct_isos),
            "hits_in_top5":      hits_in_top5,
        }, indent=2), "rag_eval_detail.json")

    return {
        "context_precision_at_5":    context_precision_5,
        "route_correct":             route_correct,
        "rag_vs_structured_overlap": rag_vs_structured_overlap,
    }


def load_negative_feedback() -> list[dict]:
    """
    Load human-flagged negative feedback from the dashboard's thumbs-down button.
    (CMU aimd-scaling-evaluation.md — eval dataset from real negative traces,
     not hand-crafted cases.)

    Returns a list of golden-case-compatible dicts ready to append to GOLDEN_CASES.
    """
    feedback_path = _HERE / "data" / "eval_feedback.jsonl"
    if not feedback_path.exists():
        return []
    cases = []
    with open(feedback_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cases.append({
                    "id":               f"feedback_{entry.get('ts', '')[:10]}_{len(cases)}",
                    "query":            entry["query"],
                    "expect_countries": [],   # no ground truth — judge evaluates blind
                    "expect_route":     "any",
                    "description":      f"Negative feedback: {entry.get('reason', 'unspecified')}",
                    "_response":        entry.get("response", ""),   # pre-recorded
                })
            except Exception:
                continue
    return cases


def run_evaluation():
    print("=" * 60)
    print("Geo-Insight Agent — LLM-as-Judge Evaluation")
    print("=" * 60)

    # Lazy import agent (avoids loading at module level)
    from agent_runner import run_query

    # ── Augment golden cases with real negative feedback from dashboard ──────
    feedback_cases = load_negative_feedback()
    if feedback_cases:
        print(f"\nℹ️  Loaded {len(feedback_cases)} negative-feedback case(s) from eval_feedback.jsonl")
    all_cases = GOLDEN_CASES + feedback_cases

    results = []
    for case in all_cases:
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
                    if f"{dim}_variance" in scores:
                        mlflow.log_metric(f"judge_{dim}_variance", scores[f"{dim}_variance"])
                total = sum(scores.get(d, 0) for d in ["correctness", "groundedness", "neutrality", "completeness"])
                mlflow.log_metric("judge_total_16",          total)
                mlflow.log_metric("judge_pct",               round(total / 16 * 100, 1))
                mlflow.log_metric("judge_score_variance_max", scores.get("judge_score_variance_max", 0))
                if scores.get("unstable_dimensions"):
                    mlflow.set_tag("unstable_judge_dims", ",".join(scores["unstable_dimensions"]))
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

            # ── Prompt registry: auto-version via MD5 hash ──────────────────
            # (oai-monitoring-governance.md — prompt version registry)
            # Hash prevents silent overwrite when prompt changes between eval runs.
            try:
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("agent04", _HERE / "agent" / "04_agent.py")
                _mod  = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                prompt_hash = hashlib.md5(_mod.SYSTEM_PROMPT.encode()).hexdigest()[:8]
                mlflow.set_tag("system_prompt_hash", prompt_hash)
                mlflow.log_text(_mod.SYSTEM_PROMPT, f"system_prompt_{prompt_hash}.txt")
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

    # ── Pass_k consistency (reliability over 3 runs) ─────────────────────────
    print("\n" + "=" * 60)
    print("Pass_k Consistency Test (semantic routing, n=3)")
    print("=" * 60)
    pk_results = run_pass_k_consistency(run_query, n_runs=3)

    # ── RAG retrieval quality metrics ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RAG Retrieval Quality Evaluation")
    print("=" * 60)
    rag_results = run_rag_evaluation(run_query)

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
    print("-" * 60)
    print(f"\n{'Pass_k route consistency (n=3)':<40} {pk_results['route_consistency']:.0%}")
    print(f"{'Pass_k country-set consistency (n=3)':<40} {pk_results['country_set_consistency']:.0%}")
    print(f"{'RAG Context Precision@5':<40} {rag_results['context_precision_at_5']:.0%}")
    print(f"{'RAG vs Structured overlap':<40} {rag_results['rag_vs_structured_overlap']:.0%}")
    if feedback_cases:
        print(f"\nℹ️  {len(feedback_cases)} negative-feedback case(s) evaluated (from dashboard thumbs-down)")
    print("=" * 60)
    print("\n✅ Results logged to MLflow experiment 'geo-insight-eval'")
    print("   View: mlflow ui  →  http://localhost:5000")


if __name__ == "__main__":
    run_evaluation()
