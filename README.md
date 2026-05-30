# agentic-testaing

> RAG and agentic-AI evaluation suite — behavioural tests, tracing, mindmaps,
> and static analysis in one workflow.

`agentic-testaing` is an end-to-end test harness for agentic AI and RAG
systems. It captures API traces, security logs, evaluation results, and
mindmap-style coverage reports across a curated knowledge dataset.

## Features

- **Comprehensive and expanded knowledge-coverage test suites** under
  `tests/`.
- **API trace + security-log capture** — every probe replayable.
- **Evaluation result store** with mindmap export for coverage visualisation.
- **Dataset management** — versioned `dataset/` and `data/` folders.
- **Static analysis** — per-target reports under
  `static_analysis_reports/`.
- **Licensed deployment** — signed `License/` artefacts (kept out of source
  control).

## Tech stack

Python · pytest · FastAPI · static-analysis tooling

## Quickstart

```bash
git clone https://github.com/krishddd/agentic-testaing.git
cd agentic-testaing
pip install -r requirements.txt
cp .env.example .env  # add target endpoint + API keys

# Run the eval suite
pytest tests/ -v

# Or start the API
uvicorn app:app --reload --port 8000
```

## Project structure

```
api/                       FastAPI endpoints
src/                       Pipeline + agent code
tests/                     Knowledge-coverage test suites
dataset/, data/            Evaluation datasets
api_responses/             Captured API traces
security_logs/             Probe / verdict log
evaluation_results/        Pass-fail + score JSON
mindmaps/                  Coverage mindmap exports
static_analysis_reports/   Per-target static-analysis output
```

## Status

Personal portfolio — designed to slot in as a CI-time eval / regression gate
for any RAG or agentic system.

## License

MIT
