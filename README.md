# llm-eval-harness

[![CI](https://github.com/byte-ish/llm-eval-harness/actions/workflows/evals.yml/badge.svg)](https://github.com/byte-ish/llm-eval-harness/actions/workflows/evals.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Type-checked: mypy --strict](https://img.shields.io/badge/types-mypy%20--strict-blue.svg)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-black.svg)](https://docs.astral.sh/ruff/)

A **versioned, CI-gated test suite for LLM features.** Treats LLM output quality the way an SDET treats code quality: regression-tested, fixture-based, reviewable.

> *"I want to upgrade the model, change a prompt, or swap providers. Run my eval suite against old and new — tell me, with numbers, whether it's safe to ship."*

---

## What it does

- **Defines suites as YAML** with `dataset_version` baked in — every run records which dataset version it scored against, so the regression detector refuses to compare a model change against a dataset change.
- **Async runner** with concurrency bound, per-request timeouts, exponential-backoff retry on `429` / `5xx` / connection errors, and per-case error isolation (one failure doesn't abort the run).
- **Three scorers ship today**, all behind the same `async` `Scorer` Protocol:
  - `exact_match` — case-insensitive substring presence
  - `regex_match` — single pattern via `re.search`
  - `llm_judge` — separate judge model scores subjective quality against a per-case rubric; tolerates malformed JSON and out-of-range scores without crashing the run
- **Pre-flight `--max-cost` guardrail** refuses to run when the estimated spend exceeds the budget (`--force` to override).
- **Regression gate**: compares a run to a stored baseline, flags per-category pass-rate drops > 5%, exits non-zero so CI fails the build. Refuses to compare across `dataset_version` boundaries unless you opt in with `--allow-dataset-mismatch`.
- **HTML report** rendered every run — per-category scores, per-case input/response/scorer-reason, cost, latency, and a regression diff when comparing.
- **Docker + docker-compose** ship in the box; `compose up` runs the harness plus an nginx report-viewer on `:8080`.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/byte-ish/llm-eval-harness.git
cd llm-eval-harness
uv sync

# 2. Run the offline test suite (no API key needed)
uv run pytest

# 3. Run a real eval against Anthropic
export ANTHROPIC_API_KEY=...
uv run python run_evals.py \
  --suite summarisation \
  --model claude-opus-4-7 \
  --max-cost 0.50

# 4. Run with the regression gate (after at least one prior run)
uv run python run_evals.py \
  --suite summarisation \
  --model claude-opus-4-7 \
  --baseline results/baseline.json \
  --max-cost 0.50

# 5. Compare two providers side-by-side on the same suite (Phase 7)
export OPENAI_API_KEY=...
uv run python run_evals.py \
  --suite summarisation \
  --compare anthropic:claude-haiku-4-5-20251001,openai:gpt-4o-mini \
  --max-cost 0.50
```

### Model spec syntax

Models are named `<provider>:<model_id>`:

| Spec | Provider | Resolves to |
| --- | --- | --- |
| `anthropic:claude-opus-4-7` | Anthropic API | `AnthropicAdapter` |
| `openai:gpt-4o-mini` | OpenAI API | `OpenAIAdapter` |
| `bedrock:anthropic.claude-haiku-4-5-v1:0` | AWS Bedrock | `BedrockAdapter` |
| `claude-opus-4-7` (bare) | Anthropic (default) | `AnthropicAdapter` |

`--compare` accepts a comma-separated list of specs and emits a side-by-side HTML report at `reports/<run_id>_compare.html` showing per-category pass-rate deltas, cost delta, and latency delta against the first model (the reference).

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success — no regression |
| `1` | Regression detected vs `--baseline` |
| `2` | `dataset_version` mismatch between baseline and current |
| `3` | Pre-flight `--max-cost` budget refused (use `--force` to bypass) |

---

## Sample output

A run produces two artifacts in addition to the console summary:

- **`results/<run_id>.json`** — full structured `RunReport` for reproducibility
- **`reports/<run_id>.html`** — self-contained HTML report a reviewer can open

A pre-generated example lives in [`reports/sample_report.html`](reports/sample_report.html) (committed so the project demonstrates output without setup).

The rich console summary looks like this:

```text
                     Results: summarisation (dataset 1.2.0)
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Category            ┃ Pass rate ┃ Cases ┃ Δ baseline┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ earnings_summary    │    100.0% │     3 │     +0.0% │
│ formatting          │     66.7% │     3 │     +0.0% │
│ guidance_change     │    100.0% │     3 │     +0.0% │
│ ma_activity         │     66.7% │     3 │     +0.0% │
│ regulatory_notice   │    100.0% │     3 │     +0.0% │
│ subjective_quality  │    100.0% │     1 │     +0.0% │
├─────────────────────┼───────────┼───────┼───────────┤
│ Overall             │     87.5% │    16 │     +0.0% │
└─────────────────────┴───────────┴───────┴───────────┘
 Total cost USD  $0.0738
 p50 latency ms  680
 p95 latency ms  1850
 Resolved model  claude-opus-4-7-20260101
 Run ID          20260609-090000000000
OK: no regression (overall +0.0% vs baseline)
```

---

## Architecture

Four layers, dependency arrow always points down. The runner knows nothing about specific scorers, scorers know nothing about specific adapters, output knows nothing about how results were produced.

```
  Test suite layer   YAML datasets  →  EvalCase fixtures  →  config loader
        │
  Runner layer       async dispatch  →  ModelAdapter Protocol  →  metadata capture
        │
  Scorer layer       exact_match  →  regex_match  →  llm_judge
        │
  Output layer       Store  →  RegressionDiff  →  HTML report  →  CI exit code
```

Each layer talks to the one below only through the frozen dataclasses in [`harness/models.py`](harness/models.py).

For the full system design — including capacity math, failure modes, Mermaid diagrams, sequence flow, and the security/cost notes — see [`design-llm-eval-harness.md`](design-llm-eval-harness.md). Mermaid diagrams live under [`diagrams/`](diagrams/) and render inline in VS Code with the `bierner.markdown-mermaid` extension.

---

## Defining a suite

Eval suites are YAML files under `evals/`. The full schema, including worked examples for each scorer, lives in **[`evals/README.md`](evals/README.md)**.

Minimum shape:

```yaml
name: summarisation
dataset_version: "1.2.0"
default_system: You are a financial summarisation assistant…
cases:
  - id: earnings_001
    category: earnings_summary
    scorer: exact_match
    user: |
      Acme Corp announced Q3 revenue of $4.2 billion…
    expected_contains: ["Acme", "Q3", "$4.2 billion"]
```

**Dataset versioning is load-bearing.** Bump `dataset_version` (SemVer-style) whenever you add, remove, or modify a case, and refresh `results/baseline.json` in the same PR. The regression detector refuses to compare across mismatched versions unless you pass `--allow-dataset-mismatch`.

---

## Adding a scorer

```python
# harness/scorers/my_scorer.py
from harness.models import EvalCase, ScorerResult


class MyScorer:
    name = "my_scorer"

    async def score(self, response: str, case: EvalCase) -> ScorerResult:
        return ScorerResult(passed=True, score=1.0, reason="ok")
```

Then register it in [`harness/scorers/registry.py`](harness/scorers/registry.py):

```python
def build_default_registry(...) -> dict[str, Scorer]:
    return {
        ...,
        MyScorer.name: MyScorer(),
    }
```

The runner imports `Scorer` (the Protocol) — never your concrete class — so no other file needs to change. If your scorer needs an external model call (like `llm_judge`), accept the adapter + cost_fn in `__init__` and pass them in from the CLI registry builder.

---

## Adding a model adapter

```python
# harness/adapters/my_provider.py
from harness.adapters.base import AdapterResponse


class MyAdapter:
    name = "my_provider"

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> AdapterResponse:
        # … call provider, measure latency, capture resolved model ID …
        return AdapterResponse(
            text=...,
            input_tokens=...,
            output_tokens=...,
            latency_ms=...,
            resolved_model_id=...,
        )
```

Add an entry to [`harness/prices.yaml`](harness/prices.yaml) for cost computation, then point the CLI's `_build_adapter` at your class (or extend it to dispatch on a `--provider` flag in Phase 7).

---

## Case study — a worked example

You've been asked to evaluate swapping the production summarisation prompt from `claude-opus-4-7` to the cheaper `claude-haiku-4-5-20251001`. Here's the workflow:

1. **Baseline the current model.** Run the suite against the prod model and snapshot the result:

   ```bash
   uv run python run_evals.py \
     --suite summarisation --model claude-opus-4-7 \
     --max-cost 0.50 --update-baseline
   ```

   The committed [`results/baseline.json`](results/baseline.json) plays this role for the demo.

2. **Run the candidate.** Same suite, new model, compare:

   ```bash
   uv run python run_evals.py \
     --suite summarisation --model claude-haiku-4-5-20251001 \
     --baseline results/baseline.json \
     --max-cost 0.10
   ```

3. **Read the verdict.**
   - Console prints a per-category pass-rate table with a `Δ baseline` column.
   - HTML report at `reports/<run_id>.html` shows per-case input / response / scorer reason, with a banner at the top: **"Regression detected"** or **"No regression"**.
   - Exit code is `1` if any category dropped more than 5%, else `0`.

4. **Decide.** Either:
   - Ship the new model and run `--update-baseline` to lock in the new baseline.
   - Reject the swap because a flagged category lost too many cases — and the HTML report shows *exactly which case IDs* newly fail, with the response text.

The point: instead of "the cheaper model seemed fine in spot-checks", you have a reviewable artifact, a number, and a CI gate.

---

## CI integration

The shipped workflow ([`.github/workflows/evals.yml`](.github/workflows/evals.yml)) has two jobs:

- **`lint-type-test`** runs on every push and PR with no secrets: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest`.
- **`eval-with-regression-gate`** runs the live eval suite against the real Anthropic API and exits non-zero on regression — but only if the `ANTHROPIC_API_KEY` repository secret is set. Without the secret, the job runs and skips the live step with a notice, so forks and unauthenticated PRs aren't gated on credentials they can't have.

The HTML report is uploaded as a build artifact so reviewers can download it from the workflow run.

---

## Running with Docker

A multi-stage `Dockerfile` is included. The runtime image runs as a non-root user (`harness`) and has no build tooling baked in.

```bash
# Build
docker build -t llm-eval-harness:local .

# Run
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/reports:/app/reports \
  llm-eval-harness:local \
  --suite summarisation --model claude-opus-4-7 --max-cost 0.50
```

Or via `docker compose` — also brings up an nginx report-viewer on `:8080`:

```bash
ANTHROPIC_API_KEY=... docker compose run --rm harness \
  --suite summarisation --model claude-opus-4-7 --max-cost 0.50
docker compose up -d report-viewer
open http://localhost:8080/sample_report.html
```

---

## Development

```bash
uv sync                                  # install deps
uv run pytest -v                         # offline test suite
uv run ruff check .                      # lint
uv run ruff format .                     # format
uv run mypy --strict harness/ run_evals.py
uv run pre-commit run --all-files        # all hooks
```

Pre-commit installs ruff (lint+format) and mypy on staged files; CI runs the same checks against the full tree.

A VS Code dev container config lives at [`.devcontainer/`](.devcontainer/devcontainer.json) — Reopen in Container and the environment is ready, including the Mermaid markdown extension for previewing the design diagrams.

### Project structure

```
harness/
├── adapters/   # ModelAdapter Protocol + AnthropicAdapter
├── scorers/    # Scorer Protocol + exact_match, regex_match, llm_judge, registry
├── templates/  # Jinja2 HTML report template
├── budget.py   # Pre-flight cost estimate + cost_fn factory
├── config.py   # YAML loader + price map
├── models.py   # Frozen dataclasses — the contracts the rest of the system speaks
├── prices.yaml # Model price map (per million tokens)
├── regression.py # Baseline comparison + dataset-version guard
├── report.py   # HTML report rendering
├── runner.py   # Async dispatch + per-case error isolation
└── store.py    # RunReport JSON persistence
```

For the full plan with phase-by-phase scope and acceptance criteria, see [`project_plan.md`](project_plan.md).

---

## License

MIT — see [`LICENSE`](LICENSE).
