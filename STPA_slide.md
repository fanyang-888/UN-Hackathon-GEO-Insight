# STPA Risk Analysis — Geo-Insight Command Center

## Control Structure

```
┌─────────────────────────────────────────────┐
│         CBPF Fund Manager (Controller)      │
│  Goal: Allocate pooled funds to overlooked  │
│         humanitarian crises                 │
└──────────────┬──────────────────────────────┘
               │ Control Actions
               │  • Accepts/rejects crisis ranking
               │  • Sets allocation priority
               │  • Requests deeper analysis
               ▼
┌─────────────────────────────────────────────┐
│      Geo-Insight Agent (Controlled Process) │
│  • Decomposes query (Claude API)            │
│  • Queries Gold table (gap scores)          │
│  • Validates ranking (self-eval)            │
│  • Red-team challenges (critic agent)       │
└──────────────┬──────────────────────────────┘
               │ Feedback (Process Model)
               │  • Gap score + CI
               │  • Confidence level (high/med/low)
               │  • Data staleness badge
               │  • Red-team objections
               ▼
┌─────────────────────────────────────────────┐
│         Data Sources (Controlled Process)   │
│  HDX HAPI · FTS · CBPF · INFORM Severity   │
└─────────────────────────────────────────────┘
```

## Unsafe Control Actions (UCAs)

| UCA # | Controller | Action | Context | Hazard |
|-------|-----------|--------|---------|--------|
| UCA-1 | Fund Manager | Accepts ranking | Low-confidence response shown without warning | Allocates funds based on stale/missing data |
| UCA-2 | Agent | Provides ranking | No HRP country scored 0% (assumption, not fact) | Overstates neglect; legitimate crises penalised |
| UCA-3 | Agent | Cites figure | Number from LLM parametric memory, not tool | Hallucinated statistic enters allocation decision |
| UCA-4 | Fund Manager | Dismisses crisis | Red-team objection too prominent | Underfunds genuinely overlooked crisis |
| UCA-5 | Agent | Routes to semantic search | Structured filter would be more precise | Wrong crises returned; manager unaware of path |

## Safety Constraints → System Responses

| UCA | Constraint | Implementation |
|-----|-----------|----------------|
| UCA-1 | Agent MUST surface confidence level with every response | `validate_ranking` tool → `st.warning()` + 🔴/🟡/🟢 badge |
| UCA-2 | 0%-coverage no-HRP crises MUST be flagged as assumptions | `has_hrp=False` → "0%*" label + footnote in ranked table |
| UCA-3 | Agent MUST only cite numbers returned by tool observations | System prompt hard rule: "NEVER fabricate figures" |
| UCA-4 | Red-team objections MUST be balanced, not dominant | Final synthesis step explicitly balances main + critic view |
| UCA-5 | Route used (semantic/structured) MUST be logged and auditable | MLflow Tracing: `route` attribute on root span |
