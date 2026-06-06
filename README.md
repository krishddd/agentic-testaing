# agentic-testaing

> End-to-end evaluation, tracing, and static-analysis platform for RAG and
> agentic-AI systems. Captures API traces and security logs, runs
> comprehensive knowledge-coverage suites, computes epistemic metrics, and
> emits mindmap-style coverage reports. Designed to slot in as a CI-time
> regression gate.

`agentic-testaing` treats every agentic system as a probable-failure
machine that needs to be exercised across knowledge, reasoning, safety, and
performance dimensions on every release. It supplies the harness (probes,
traces, judges) and the receipts (verdict JSON, evidence, mindmaps) so a
team can answer "did the agent regress?" without running the agent by hand.

---

## What the platform does

1. **Loads a target** — any HTTP-exposed RAG / agent endpoint, defined by a
   profile in `config/`.
2. **Replays a knowledge-coverage suite** — datasets in `dataset/` +
   `data/` walk the target through curated topics, edge cases, and
   adversarial prompts.
3. **Captures every interaction** — request, response, latency, tool
   calls, and tokens land in `api_responses/` and `traces/`.
4. **Scores the response** — heuristic and judge-LLM verdicts, plus
   epistemic-metric calculations (calibration, uncertainty, citation
   integrity).
5. **Logs security events** — refusal patterns, jailbreak attempts,
   schema violations recorded to `security_logs/`.
6. **Runs static analysis** — per-target reports under
   `static_analysis_reports/` (deps, configuration, prompt-template
   smells).
7. **Renders a mindmap** — coverage and verdict summary as a hierarchical
   mindmap exported to `mindmaps/` for review.
8. **Persists results** — pass/fail/score JSON under `evaluation_results/`
   so a CI pipeline can diff a run vs a baseline.

---

## End-to-end pipeline

```
target profile (config/)
        │
        ▼
api/    FastAPI control plane
        │
        ▼
src/    test runner + judge orchestration
        │
        ├──► dataset/, data/             knowledge-coverage prompts
        │
        ▼
target (HTTP RAG / agent)
        │
        ├──► trace recorder        →  traces/<run_id>/*.jsonl
        ├──► raw responses         →  api_responses/<run_id>/*.json
        ├──► security observer     →  security_logs/<run_id>/*.json
        ▼
judge / metrics
        │
        ├──► heuristic verdict       (exact / contains / regex / schema)
        ├──► judge-LLM verdict       (groundedness, relevance, helpful)
        └──► epistemic metrics       (calibration, uncertainty, citations)
        │
        ▼
evaluation_results/<run_id>/
        ├─ verdicts.json
        ├─ scores.json
        └─ epistemic_metrics.json
        │
        ▼
mindmap renderer  →  mindmaps/<run_id>.html / .png
static analyser   →  static_analysis_reports/<target>/
```

Re-runs against the same target are diffed by `run_id` so CI can flag
regressions on a per-topic, per-metric basis.

---

## Epistemic metrics

The platform computes the metrics documented in `epistemic_metrics_guide.md`
and `Agent_Output_Metrics_Explained.md`:

- **Groundedness** — share of answer tokens supported by retrieved
  citations.
- **Citation integrity** — every citation resolves, no fabricated sources.
- **Calibration** — does the agent's expressed confidence track its actual
  accuracy?
- **Uncertainty surfaced** — does the agent flag low-confidence answers
  instead of guessing?
- **Refusal correctness** — refuses what it should refuse, answers what it
  can.
- **Hallucination rate** — unsupported claims as a fraction of all claims.
- **Tool-call accuracy** — correct tool / args under tool-use scenarios.

Each metric ships with a baseline + an alert threshold; CI runs fail if
the threshold is crossed.

---

## Datasets

Knowledge-coverage datasets live in `dataset/` (curated topics, multi-hop
chains, adversarial variants) and `data/` (per-topic factsheets used by
the judge to score correctness). Test suites under `tests/` mount them:

- `test_comprehensive_knowledge.py` — broad coverage across topics.
- `test_expanded_knowledge.py` — long-tail and edge-case variants.
- Demo scripts in `test_demo/` for showcasing capabilities.

---

## Licensed deployment

The platform ships with a licensing artefact in `License/` (`certificate.pem`,
`key.pem`, `license.key`). These are kept out of source control by
`.gitignore`. The runtime checks the license at startup; without a valid
key the runner refuses to start.

---

## Quickstart

```bash
git clone https://github.com/krishddd/agentic-testaing.git
cd agentic-testaing
pip install -r requirement.txt
cp .env.example .env  # target endpoint + API keys

# Start the API
uvicorn app:app --reload --port 8000

# Run the eval suite
pytest tests/ -v

# Or trigger a run via API
curl -X POST http://localhost:8000/runs \
     -H 'Content-Type: application/json' \
     -d '{"target": "my-agent", "suite": "comprehensive_knowledge"}'
```

Inspect outputs:

```bash
ls evaluation_results/<run_id>/
open mindmaps/<run_id>.html
```

---

## Project structure

```
app.py                            FastAPI entry point
api/                              REST endpoints (runs, results, judge)
src/                              Test runner, judge, metrics
config/                           Target profiles + judge config
database/                         Result storage layer
dataset/                          Knowledge-coverage prompt sets
data/                             Factsheets / expected answers
api_responses/                    Raw captured responses (gitignored)
traces/                           Per-run trace JSONL (gitignored)
security_logs/                    Refusal / jailbreak observations (gitignored)
evaluation_results/               Verdict JSON, scores, epistemic metrics
static_analysis_reports/          Per-target dependency + config audits
mindmaps/                         Mindmap exports per run
License/                          Signed license artefacts (gitignored)
tests/                            Pytest suites (knowledge, regression)
test_demo/                        Demo / showcase scripts
docs/                             Long-form notes
```

---

## Companion docs

- `Agent_Output_Metrics_Explained.md` — every metric, its formula, and
  why it matters.
- `epistemic_metrics_guide.md` — guide to interpreting calibration,
  uncertainty, and citation metrics.
- `demo_showcase_scenarios.md` — scripted walkthroughs for demos.
- `Core Performance Metrics.docx` — performance benchmarking notes.

---

## CI usage

Wire `pytest tests/ -v --junitxml=results.xml` plus a diff against the
prior run's `evaluation_results/` into your CI to gate releases. The
mindmap output is publishable as a build artefact for human review.

---

## Status

Personal portfolio. Designed to plug in as a CI-time eval / regression gate
for any RAG or agentic system.

## License

MIT (License/ artefacts ship under the project's own licensing model)
