# My Geo-Insight Story
### UNOCHA × CMU × Databricks Hackathon · May 2026 · Solo Build

> *This document is written for interview preparation. Each section maps to a common behavioral question. The stories are real — use them as-is or adapt the framing to the role.*

---

## The 30-Second Version

In May 2026, I built an end-to-end AI system in four days — solo — that identifies the world's most overlooked humanitarian crises by combining data from five public sources, a custom gap-scoring formula, and a Claude-powered conversational agent. The system now covers 77 countries, surfaces $500M+ in funding gaps, and includes a Responsible AI Scorecard that proactively audits its own outputs. More importantly, I learned how to turn ambiguous, messy real-world data into something a non-technical UN coordinator could actually act on — and how to keep building when nothing works.

---

## The Full Story: Four Days, One System, Many Lessons

### Day 0 — Before I Started: Why I Said Yes to a Hackathon I Wasn't Sure I Could Win

I almost didn't enter. The challenge brief said *"analytical quality and ranking defensibility"* — not frontend polish, not the most features. That scared me a little. Defensibility means you have to stand behind every number. In a world where humanitarian data is politically sensitive and people's lives depend on resource allocation decisions, you can't hand-wave uncertainty.

But that's exactly why I said yes. I wanted to build something that could be wrong — and *know* when it was wrong.

I gave myself one rule going in: **if I can't explain why a crisis ranked #1, I shouldn't show it.**

---

### Day 1 — The Data Is Never What You Think It Is

I started by pulling data from the official Databricks volume. The plan was simple: ingest HNO (people in need) + FTS (funding) → compute a gap → rank → done.

Within two hours, the plan was broken.

The HNO data only covered **24 countries**. The challenge asked for global coverage. The CBPF pooled funds file — which I'd counted on for funding data — turned out to be from **2018**. Eight years old. And the INFORM severity index had a different country naming convention than every other source.

I had a choice: shrink the scope and do 24 countries cleanly, or figure out a way to cover more of the world honestly.

I chose a third option I hadn't planned for: a **two-tier data model**.

Countries with full HNO data would get a complete gap score — funding gap, need scale, severity, and neglect factor. Countries without HNO would still appear, scored on what data existed (funding gap + severity), with a visible badge saying "FTS proxy — PIN not available." The system would never hide what it didn't know. It would just label it.

By end of Day 1, I had 77 countries in the pipeline and a working Bronze → Silver → Gold medallion architecture. The CBPF file was flagged as stale and documented in the Responsible AI Scorecard. The 2018 data didn't disappear — it became a data governance finding.

**What I learned:** Real-world data problems aren't solved by finding cleaner data. They're solved by deciding what your system owes the user: clarity about what it knows and what it doesn't.

---

### Day 2 — Building the Formula Nobody Could Argue With

The hardest design problem wasn't technical. It was this: **how do you rank human suffering in a way that a CBPF fund manager could defend in a meeting?**

I spent half a day on the gap score formula — not coding it, but arguing with myself about it.

First attempt: simple `funding_received / funding_requested`. Clean, fast, defensible. But a country with 100,000 people and 0% coverage would outrank Sudan with 34 million people and 10% coverage. That's wrong.

Second attempt: weight by people in need. Better. But a small, severe crisis could still be buried by a larger, less urgent one.

Third attempt: the formula that stuck.

```
base_score = funding_gap × (0.435 + 0.348×need_scale + 0.217×severity_mult)
gap_score  = base_score × neglect_factor
```

Every weight has a justification. I derived them using **swing weighting** — a decision modeling technique from my coursework: *"If you could move ONE dimension from worst to best, which matters most?"* Funding gap wins (0.435). Need scale is second (0.348). INFORM severity is third (0.217). If a judge asks why, I can walk through that derivation in two minutes.

The neglect factor was for the bonus question: *how do you distinguish a crisis that's always been overlooked from one that's newly underfunded?* My answer: `1 + (consecutive_years_underfunded × 0.15)`. Three straight underfunded years gives a 45% multiplier. That's not arbitrary — it encodes the idea that structural neglect compounds deferred need year over year.

Then I added Monte Carlo confidence intervals (1,000 simulations, PIN ±30%, funding ±20%). Because a gap score of 1.165 presented without uncertainty is false precision. The CI shows up in the UI as `1.165 [1.144–1.185]` — visible proof that the system knows what it doesn't know.

**What I learned:** A formula nobody can argue with isn't one that's mathematically perfect. It's one where every coefficient has a human rationale you can say out loud.

---

### Day 3 — The Agent That Questions Itself

By Day 3, the pipeline worked. The dashboard showed rankings. But I kept thinking: what happens when the agent is wrong?

I built six tools into the ReAct agent. The first four were expected: decompose the query, query the data, validate results, generate briefing notes. The fifth changed everything.

**`red_team_challenge`** — a second Claude call with an adversarial system prompt: *"You are a skeptical auditor. Find the strongest argument against this ranking."*

The agent now argues with itself before giving you an answer. When Mali ranked #1 for CBPF allocation, the adversarial auditor flagged that 30% of Mali's population-in-need lives in areas under AES access restrictions — meaning the gap score might overstate deliverable impact. That caveat made it into the briefing note. A fund manager reading it would know to verify accessible zones before treating the gap as fully actionable.

That's not a system that's less useful because it admits uncertainty. It's a system that's more trustworthy because it does.

I also added **self-consistency** to the query decomposition step: three parallel decompositions of the same natural-language query, majority vote on the filters. It slows the agent by about 30%, but it catches cases where one parse goes wrong.

**What I learned:** The most important question when building an AI system isn't "how do I make it right?" It's "what does it say when it's wrong?" The answer to that question determines whether anyone can trust it.

---

### Day 3, 11PM — The Bug That Almost Broke Me

I opened the dashboard for a final check before sleep. The hero stats read: **24 structural neglect cases**.

That's wrong. I knew there were 8.

I traced it through the dashboard code. The `render_hero()` function was summing all rows in the dataframe — multi-year rows, one per country per year. Sudan appearing in 2024 and 2025 counted twice. A crisis that was structural in 2024 and structural again in 2025 inflated the number.

The fix was three lines:

```python
recent = df_full.loc[df_full.groupby("country_iso3")["year"].idxmax()]
n_struct = int((recent["neglect_type"] == "structural").sum())
```

Take the most recent year per country. Then count. Eight. Correct.

I wrote the fix. It worked. Then I sat for a minute.

That bug had been in production. Every time I'd looked at the dashboard and felt confident about the hero stats, the number was wrong. A user could have cited "24 structural neglect cases" in a briefing. That number would have been false.

I didn't spiral. I wrote it up in the changelog, documented it in Known Limitations as a resolved issue, and went to sleep. But I also rebuilt my instinct for what to verify — not just "does it run" but "does the number make sense."

**What I learned:** The most dangerous bugs are the ones that look right. Build verification habits, not just tests.

---

### Day 4 — MLflow Traces Disappearing (And Why Environment Variables Are a Lie)

On the final morning, I opened MLflow to show the agent traces. Nothing. The UI showed zero runs for the `geo-insight-agent` experiment.

I knew the traces were being written — I'd seen them in CLI mode. But when the agent ran from Streamlit, they vanished.

I dug into it. Streamlit runs in a subprocess. Environment variables set in the terminal don't reliably propagate. MLflow was silently defaulting to a different tracking URI — a temp directory, not my `mlflow.db`.

The fix:

```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MLFLOW_DB = _PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{_MLFLOW_DB}")
```

Hard-coded absolute path derived from the file's own location. No environment variable required. No silent failure possible.

One hour before submission, the traces appeared.

**What I learned:** In production systems, silence is not success. If a logging call fails without raising an error, you will not know — until the moment you need the logs most.

---

## Interview Question Map

*Use these sections to answer behavioral questions directly. Each is a complete STAR story.*

---

### "Tell me about a time you dealt with ambiguous requirements."

**Situation:** The hackathon brief said "build a system to identify overlooked crises" but didn't define what "overlooked" means quantitatively, or what data quality was acceptable.

**Task:** I had to design a scoring system that was both defensible to domain experts and honest about its own limitations — without a product manager or defined spec.

**Action:** I derived the gap score formula using swing weighting, a structured MCDM technique that forces explicit trade-offs. For data quality, I invented a two-tier model: countries with full HNO data get a complete score; countries without PIN data get a partial score with a visible badge. The system never presents false confidence.

**Result:** 77 countries covered across two tiers, with every limitation documented in a Responsible AI Scorecard that's part of the product. Judges received not just a ranking but an auditable system that explains its own uncertainty.

---

### "Tell me about a time you failed and what you learned."

**Situation:** Three days into the build, I discovered that the hero stats on my dashboard had been showing inflated numbers the entire time — 24 structural neglect cases instead of the correct 8.

**Task:** Fix the bug and understand why I missed it.

**Action:** The root cause was aggregating multi-year rows without deduplicating by country first. The fix was three lines of code. But the lesson required more work: I rebuilt my verification checklist to include "sanity check headline numbers against expected ranges" — not just "does the code run."

**Result:** Correct stats in production. More importantly, a new habit of verifying outputs against domain knowledge, not just code behavior.

---

### "Describe a time you had to learn something new quickly under pressure."

**Situation:** On Day 1, the data I'd planned to use was either too narrow (24 countries) or too stale (2018 CBPF data). I needed to expand coverage to be competitive, but I hadn't planned for a two-tier architecture.

**Task:** Design and implement a new data model in under a day, without breaking what already worked.

**Action:** I read the FTS API documentation, designed the two-tier model (HNO full-score vs. FTS proxy), and implemented Step 6b in the pipeline — 53 additional countries fetched and merged — in about four hours. I also added a `data_tier` column that flows through Silver → Gold → UI, so the transparency is automatic, not bolted on later.

**Result:** Coverage went from 24 to 77 countries by end of Day 1. The FTS proxy tier has been a differentiator in the submission — most participants stayed with the 24-country HNO data.

---

### "Tell me about a time you made a system more trustworthy."

**Situation:** Initial versions of the agent gave confident-sounding answers with no acknowledgment of uncertainty or potential bias in the underlying data.

**Task:** Make the agent honest without making it useless.

**Action:** I added three layers: (1) Monte Carlo confidence intervals on every gap score, (2) a `validate_ranking` tool that flags low-confidence results, and (3) a `red_team_challenge` tool — a second adversarial Claude call that finds the strongest counter-argument before the main answer is finalized. I also added neutral framing rules to the system prompt: coverage percentage language, not loss/gain language; forced counter-argument per briefing note; grounding mandate (no hallucinated figures).

**Result:** When Mali ranked #1, the agent proactively flagged that 30% of the PIN may be in inaccessible zones. That caveat came from the adversarial auditor, not from me. A fund manager reading the briefing note would make a better decision because of it.

---

### "How do you approach a problem you've never seen before?"

My approach in this project:

1. **Define what success looks like before writing code.** For the gap score, "success" meant a humanitarian analyst could defend the ranking in a meeting. That definition shaped every design decision.

2. **Make the constraints visible.** I don't hide what the system can't do. Bad data becomes a data tier badge. Uncertainty becomes a confidence interval. Access constraints become counter-arguments in the briefing.

3. **Build for the failure case first.** What does the system say when it's wrong? If I can't answer that, the system isn't ready.

4. **Document as I go.** Every design choice that seemed obvious in the moment would be opaque two days later. The README, the changelog, and the Responsible AI Scorecard aren't post-hoc documentation — they're part of the build.

---

### "What's a technical decision you're proud of?"

The `red_team_challenge` tool.

Most AI systems in this space generate confident briefings. Mine generates a briefing and then challenges it. The adversarial auditor is a second Claude call with a system prompt that says: *find the strongest argument against this ranking.*

It's not just a safety feature. It's a product feature. A fund manager who reads "here's why Mali is #1, and here's the strongest reason that might be wrong" is better equipped than one who just reads "here's why Mali is #1."

The technical implementation was simple — one extra API call with a different system prompt. The design insight was harder: **the most useful thing an AI system can do is tell you what it doesn't know.**

---

## Numbers That Tell the Story

| What I built | Scale |
|---|---|
| Countries covered | 77 (started at 24) |
| People in need tracked | 229 million |
| Structural neglect cases identified | 8 |
| Data sources integrated | 5 |
| Agent tools built | 6 |
| Gap score simulations per country | 1,000 (Monte Carlo) |
| Lines of Python written | ~4,000 |
| Days to build | 4 (solo) |
| Demo video | 15 minutes |
| Slide deck | 8 pages |

---

## What I'd Do Differently

Honest reflection matters in interviews. Here's mine:

**I'd integrate ACLED conflict data.** The gap score uses INFORM severity as the independent urgency signal. ACLED would provide a real-time conflict event count as a second independent signal — less susceptible to assessment lag. I documented this as a limitation; I'd fix it if I had a Day 5.

**I'd build a better FTS proxy.** The 53 FTS proxy countries have PIN = 0 and therefore systematically lower gap scores than they probably deserve. A population-based need estimator (using COD population data) would give a first-order PIN estimate for countries without a formal HNO. Better than nothing, and more defensible than zero.

**I'd spend more time on the demo.** The system is strong. The demo was good. But the best product in the world loses to a mediocre product with a better story. I'd script the demo more tightly — 3 concrete questions, 3 clear answers, less time on technical architecture.

---

## The One Thing

If someone asked me what I actually learned in four days — not the technical things, but the real thing — it's this:

**A system that says "I don't know" is more useful than one that always has an answer.**

The rankings, the confidence intervals, the adversarial auditor, the data tier badges, the Responsible AI Scorecard — all of it comes back to one design principle: the users of this system are making decisions that affect people's lives. They deserve to know not just what the data says, but how much to trust it.

That's not a humanitarian principle. That's an engineering principle. And it's the one I'll carry into every system I build from here.

---

*Built by Fan Yang · CMU AIM · May 2026*
*GitHub: https://github.com/fanyang-888/UN-Hackathon-GEO-Insight*
