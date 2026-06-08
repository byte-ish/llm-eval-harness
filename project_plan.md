# llm-eval-harness — Project Plan

> Master planning document. This is the single source of truth for what we are
> building, why, and in what order. Read this at the start of every session.
> Each phase has a clear goal, a scope boundary, and a definition of done.
> Do not build ahead of the current phase.

---

## 1. The problem this solves

When teams ship LLM features, they have no equivalent of a test suite. Model
changes, prompt edits, provider switches, and temperature tweaks are all leaps
of faith — there is no repeatable way to answer "did this change make the model
worse at my specific task?"

This harness treats LLM output quality the way a software engineer treats code
quality: versioned test cases, fast feedback, regression gates, CI integration,
and fixture-based testing. It turns "the summaries seem fine" into a measured,
reviewable decision.

**The core user story:** "I want to upgrade my model / change my prompt. Run my
eval suite against old and new, and tell me — with numbers — whether it's safe."

---

## 2. Design principles (these govern every phase)

1. **Contracts before features.** The data models (EvalCase, EvalResult,
   RunReport) and the scorer/adapter Protocols are the backbone. Get them right
   before building anything on top. Everything else plugs into these.
2. **Pluggable everything.** Scorers and model adapters are swappable via a
   Protocol interface. Adding a new scorer or provider must never require
   touching the runner.
3. **Deterministic core, probabilistic edges.** The runner, scoring dispatch,
   storage, and regression logic are fully deterministic and unit-tested. LLM
   calls default to `temperature=0` with per-case overrides allowed. Only the
   actual model output is non-deterministic, and those calls are mocked in tests.
4. **No real API calls in the test suite.** Every test runs offline against
   recorded fixtures. The suite must pass with no API key set.
5. **Lean dependencies.** Each dependency must earn its place. Prefer the stdlib.
   Ask before adding anything not already approved.
6. **Reproducible runs.** Every run records the model (including the resolved
   provider model ID after alias resolution), prompt, parameters, dataset
   version, and harness version so any result can be explained later.
7. **The output is a deliverable.** A run produces a human-readable report, not
   just a log. The report is what a reviewer reads to make a decision.
8. **Static types, enforced.** `mypy --strict` runs in CI on `harness/` and
   `run_evals.py`. No `Any` in public signatures; no implicit `Optional`.
9. **Dataset versioning is load-bearing.** Every suite has a `dataset_version`;
   baseline comparisons refuse to run across mismatched versions unless
   explicitly overridden. Prevents "did the dataset change or did the model
   regress?" ambiguity.

---

## 3. Approved dependencies

Do not add anything outside this list without asking first.

| Dependency | Purpose | Added in |
|---|---|---|
| `anthropic` | Anthropic SDK (async client) | Phase 2 |
| `pyyaml` | Load eval datasets | Phase 1 |
| `pytest` | Test runner | Phase 0 |
| `pytest-asyncio` | Async test support | Phase 2 |
| `rich` | Console summary tables + progress bar | Phase 1 |
| `jinja2` | HTML report templating | Phase 5 |
| `mypy` | Static type checking (dev dep) | Phase 0 |
| `pre-commit` | Format/lint hooks on commit (dev dep) | Phase 0 |

Tooling: `uv` for env + dependency management, Python 3.12, `ruff` for lint/format,
`mypy --strict` for type checking, `pre-commit` for commit-time hooks.

License: MIT (see `LICENSE`).

---

## 4. Target architecture (end state)

Four layers, built bottom-up across the phases:

```
  Test suite layer    YAML datasets  ->  EvalCase fixtures  ->  dataset registry
        |
  Runner layer        async dispatch  ->  model adapter  ->  metadata capture
        |
  Scorer layer        exact/regex  ->  LLM-as-judge  ->  embedding sim  ->  RAGAS
        |
  Output layer        results store  ->  regression detector  ->  HTML report  ->  CI gate
```

The dependency arrow always points down: the runner knows nothing about specific
scorers, scorers know nothing about specific adapters, output knows nothing about
how results were produced. Each layer talks to the one below only through the
data contracts in `harness/models.py`.

---

## 5. Repository layout (end state)

Build this incrementally — do not create empty files for later phases until that
phase begins.

```
llm-eval-harness/
├── CLAUDE.md                  # session context (Phase 0)
├── PROJECT_PLAN.md            # this file
├── README.md                  # filled in Phase 6
├── LICENSE                    # MIT — Phase 0
├── pyproject.toml             # incl. mypy --strict config
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml    # ruff + mypy hooks — Phase 0
├── evals/
│   ├── README.md              # dataset YAML format docs — Phase 1
│   └── summarisation.yaml     # Phase 1
├── harness/
│   ├── __init__.py
│   ├── models.py              # core contracts — Phase 1
│   ├── config.py              # loads suite + price map — Phase 1
│   ├── runner.py              # async dispatch — Phase 2
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py            # ModelAdapter Protocol — Phase 1
│   │   ├── anthropic.py       # Phase 2
│   │   ├── openai.py          # Phase 7
│   │   └── bedrock.py         # Phase 7
│   ├── scorers/
│   │   ├── __init__.py
│   │   ├── base.py            # Scorer Protocol — Phase 1
│   │   ├── exact_match.py     # Phase 1
│   │   ├── regex_match.py     # Phase 3
│   │   ├── llm_judge.py       # Phase 4
│   │   └── embedding_sim.py   # Phase 7 (optional)
│   ├── store.py               # results persistence — Phase 2
│   ├── regression.py          # baseline comparison — Phase 5
│   └── report.py              # HTML rendering — Phase 5
├── results/                   # gitignored except baseline.json
│   └── baseline.json
├── reports/                   # gitignored
├── templates/
│   └── report.html.j2         # Phase 5
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_exact_match.py
│   ├── test_regex_match.py
│   ├── test_llm_judge.py
│   ├── test_runner.py
│   ├── test_regression.py
│   └── fixtures/
│       └── mock_responses.py  # recorded API shapes
├── .github/
│   └── workflows/
│       └── evals.yml          # Phase 6
└── run_evals.py               # CLI entrypoint
```

---

## 6. Core data contracts (build first, in Phase 1)

These are the most important objects in the system. Use dataclasses.

```python
# EvalSuite — a loaded YAML suite with suite-level prompt defaults
# Returned by config.load(); the runner iterates suite.cases.
@dataclass(frozen=True)
class EvalSuite:
    name: str
    dataset_version: str         # required; bumped on dataset edits
    default_system: str | None   # falls into any case without its own system
    default_user: str | None     # rare; usually each case provides its own user
    cases: tuple[EvalCase, ...]  # frozen — tuple, not list

# EvalCase — one test case loaded from YAML
# system/user are optional at the case level — config.py resolves them from
# the suite-level defaults at load time. The runner only ever sees fully-
# resolved prompts. If neither case nor suite provides one, load fails loudly.
@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    scorer: str                  # name of scorer to apply (registry key)
    user: str | None = None      # falls back to suite.default_user
    system: str | None = None    # falls back to suite.default_system
    expected_contains: list[str] = field(default_factory=list)
    expected_regex: str | None = None
    judge_rubric: str | None = None
    temperature: float | None = None  # per-case override; default is 0
    tags: list[str] = field(default_factory=list)

# ScorerResult — output of any scorer
@dataclass(frozen=True)
class ScorerResult:
    passed: bool
    score: float                 # 0.0–1.0, even for binary scorers
    reason: str

# EvalResult — one case after running + scoring
@dataclass(frozen=True)
class EvalResult:
    case_id: str
    category: str
    response: str
    scorer_result: ScorerResult
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str                   # resolved provider model ID (post-alias)
    temperature: float
    error: str | None = None

# RunReport — the full output of one run
@dataclass(frozen=True)
class RunReport:
    run_id: str                  # timestamp-based
    model: str                   # the resolved provider model ID
    suite: str                   # suite name
    dataset_version: str         # mirrors EvalSuite.dataset_version
    results: list[EvalResult]
    started_at: str
    finished_at: str
    harness_version: str         # importlib.metadata.version("llm-eval-harness")
    # computed properties: pass_rate, pass_rate_by_category,
    # total_cost_usd, p50_latency_ms, p95_latency_ms
```

```python
# Scorer Protocol — every scorer implements this, nothing more
class Scorer(Protocol):
    name: str
    def score(self, response: str, case: EvalCase) -> ScorerResult: ...

# ModelAdapter Protocol — every provider implements this
class ModelAdapter(Protocol):
    name: str
    async def complete(self, system: str, user: str,
                       **params) -> AdapterResponse: ...
    # AdapterResponse carries: text, input_tokens, output_tokens, latency_ms
```

---

## 7. Phased build plan

Each phase is a self-contained unit of work with its own definition of done.
Run the tests at the end of every phase; do not start the next phase until the
current one is green.

### Phase 0 — Scaffold and contracts skeleton  (Jira: AI-424)
**Goal:** A clonable repo with structure, tooling, license, hooks, and CLAUDE.md in place.
- Create directory structure for Phase 0–1 files only
- `pyproject.toml` — uv, Python 3.12, ruff config, **mypy `--strict` config**
- `.gitignore` (`.env`, `results/*` except `baseline.json`, `reports/*`,
  `__pycache__`, `.venv`)
- `.env.example` with `ANTHROPIC_API_KEY=`
- `LICENSE` — MIT
- `.pre-commit-config.yaml` — `ruff format`, `ruff check`, `mypy --strict` on commit
- `CLAUDE.md` (see section 8 for required content)
- Empty `harness/` package with `__init__.py` files. `harness/__init__.py`
  exposes `__version__` via `importlib.metadata.version("llm-eval-harness")`.
- Branch: `feat/AI-424-phase-0-scaffold`
**Done when:** `uv sync` succeeds; `ruff check .` and `mypy --strict harness/`
both pass on the empty package; `pre-commit run --all-files` passes.

### Phase 1 — Data contracts + first scorer (no API)  (Jira: AI-425)
**Goal:** The deterministic core, fully tested, with zero network dependency.
- Implement `harness/models.py` fully (section 6) — including `EvalSuite` and
  `dataset_version` on `RunReport`
- Implement `harness/scorers/base.py` (Scorer Protocol)
- Implement `harness/scorers/exact_match.py` (all `expected_contains` strings
  must be present, case-insensitive; score is fraction matched, passed = all)
- Implement `harness/config.py` to load a YAML suite into an `EvalSuite`:
  - Reads top-level `dataset_version`, `default_system`, `default_user`
  - Resolves each case's `system`/`user` from case → suite default → loud error
  - Loads the model price map
- Implement `harness/adapters/base.py` (ModelAdapter Protocol + AdapterResponse)
- Write `evals/summarisation.yaml` with 12 synthetic financial-text test cases.
  Top-level `dataset_version: "1.0.0"` and a shared `default_system`; cases
  carry their own `user`.
- Write `evals/README.md` — full YAML schema reference with examples for
  exact_match, regex_match, and llm_judge cases (the latter two register
  the schema even if the scorers come later)
- Tests:
  - `test_models.py` — computed properties, edge cases, `EvalSuite.cases`
    immutability
  - `test_exact_match.py` — pass, partial, fail, empty, case-insensitivity
  - `test_config.py` — dataset_version round-trips; suite default applied;
    case override wins; missing system+default raises a clear error at load
    time (not at runtime)
- Branch: `feat/AI-425-phase-1-contracts`
**Done when:** `pytest` passes with no API key set; `mypy --strict` clean;
coverage of models, config resolution, and exact_match is complete.

### Phase 2 — Async runner + Anthropic adapter + storage  (Jira: AI-426)
**Goal:** End-to-end run against the real API, results saved to disk.
- Implement `harness/adapters/anthropic.py`:
  - Async client; captures text, tokens, wall-clock latency; cost from price map
  - **Defaults `temperature=0`**; honours per-case override
  - **Per-request timeout** (default 60s, configurable via CLI)
  - **Retry policy:** exponential backoff with jitter on 429 / 5xx /
    connection error; max 3 attempts; only the final terminal error is
    recorded as `EvalResult.error`
  - Records the **resolved provider model ID** (after alias resolution) on
    `AdapterResponse`; this flows through to `EvalResult.model`
- Implement `harness/runner.py`:
  - Async dispatch with a concurrency semaphore
  - Per-case error isolation so one failure doesn't abort the run
  - `rich.progress` progress bar showing cases completed / total + ETA
- Implement `harness/store.py` (serialise RunReport to
  `results/<run_id>.json`, and load a RunReport back)
- Implement `run_evals.py` CLI:
  - `--suite`, `--model`, `--concurrency`
  - `--timeout-seconds` (default 60), `--temperature` (override default 0)
  - `--max-cost USD` — pre-flight estimate (price map × cases × token budget);
    refuses to run if the estimate exceeds the budget unless `--force` is set
  - Prints a `rich` summary table (per-category pass rate, cost, p50/p95)
- Tests: `test_runner.py` using a mock adapter from `fixtures/mock_responses.py`:
  success, partial failure, all-fail, retry success after one 429, retry
  exhaustion, timeout, pre-flight budget refusal. No real calls.
- Branch: `feat/AI-426-phase-2-runner`
**Done when:** `python run_evals.py --suite summarisation` runs against the real
API end-to-end and writes a valid result JSON; runner tests pass offline;
`mypy --strict` clean.

### Phase 3 — Regex scorer + scorer registry  (Jira: AI-427)
**Goal:** Multiple scorers, selected per case, via a registry.
- Implement `harness/scorers/regex_match.py`
- Implement a scorer registry (name -> Scorer instance) so a case's `scorer`
  field resolves to the right implementation
- Extend the runner to dispatch each case to its declared scorer
- Add regex cases to the dataset (bump `dataset_version` to `1.1.0`)
- Tests: `test_regex_match.py`; registry resolution tests
- Branch: `feat/AI-427-phase-3-regex-registry`
**Done when:** A single suite can mix exact_match and regex_match cases and each
is scored by the correct scorer.

### Phase 4 — LLM-as-judge scorer  (Jira: AI-428)
**Goal:** Subjective quality scoring via a judge model.
- Implement `harness/scorers/llm_judge.py` — takes a per-case `judge_rubric`,
  calls the judge model, parses a structured `{score, reason}` JSON response,
  normalises score to 0.0–1.0
- Make the judge model configurable and separate from the model under test
- Handle judge parse failures gracefully (record as error, do not crash run)
- Tests: `test_llm_judge.py` with a mocked judge response (valid JSON,
  malformed JSON, out-of-range score)
- Branch: `feat/AI-428-phase-4-llm-judge`
**Done when:** A case with a rubric is scored by the judge offline in tests, and
end-to-end against the real judge model.

### Phase 5 — Regression detection + HTML report  (Jira: AI-429)
**Goal:** The decision-making layer and the human-readable output.
- Implement `harness/regression.py`:
  - Compare a RunReport to `results/baseline.json`
  - **Refuse to compare** if `baseline.dataset_version != current.dataset_version`
    — emit a clear "dataset changed; refresh baseline or pass
    `--allow-dataset-mismatch`" error
  - Flag any category whose `pass_rate` drops > 0.05
  - Return a structured diff (per-category delta, newly failing case IDs)
- Add `--baseline`, `--update-baseline`, and `--allow-dataset-mismatch` flags
- `run_evals.py` exit code is **non-zero on detected regression** (sets up the
  Phase 6 CI gate)
- Implement `harness/report.py` + `templates/report.html.j2`: render a run to
  `reports/<run_id>.html` showing per-category scores, pass/fail per case
  (with input, response excerpt, scorer reason), cost, latency, and (if
  comparing) the regression diff
- Tests: `test_regression.py` (improvement, regression, new category,
  threshold boundary, dataset-version mismatch)
- Branch: `feat/AI-429-phase-5-regression`
**Done when:** Running with `--baseline` prints a clear pass/fail verdict, exits
non-zero on regression, and the HTML report renders a readable summary a
reviewer could act on.

### Phase 6 — CI integration + README + sample artifacts  (Jira: AI-430)
**Goal:** Make it a credible, runnable open-source portfolio project.
- `.github/workflows/evals.yml`:
  - Always (no secret needed): `ruff check`, `ruff format --check`,
    `mypy --strict`, `pytest`
  - Conditional on `ANTHROPIC_API_KEY` secret: run the eval suite + regression
    gate; fail the build on regression
- Write the README:
  - Problem statement + SDET framing
  - **Status badges**: CI, license (MIT), Python 3.12
  - Architecture diagram
  - **Embedded screenshot** of the `rich` summary table
  - **Link / screenshot thumbnail** to the committed sample HTML report
  - Quickstart (`uv sync` → `uv run pytest` → real run)
  - Dataset YAML format pointer to `evals/README.md`
  - How to add a scorer; how to add a model adapter
  - **"Case Study" section**: a worked example showing a baselined model vs.
    a candidate model, the resulting diff, and the verdict. Synthetic numbers
    are fine — the goal is to show the harness's output in narrative form so
    a reviewer immediately gets the value prop
- Commit a sample `results/baseline.json` and a pre-rendered
  `reports/sample_report.html` so the repo demonstrates output without setup
- Branch: `feat/AI-430-phase-6-ci-readme`
**Done when:** A fresh clone can run `uv run pytest` green with no key;
`mypy --strict` clean in CI; the README tells the story end-to-end; the
sample report is viewable.

### Phase 7 — Multi-model comparison (stretch)  (Jira: AI-431)
**Goal:** Run one suite across providers and compare.
- Implement `openai.py` and `bedrock.py` adapters behind the same Protocol
- Add `run_evals.py --compare model_a,model_b` producing a side-by-side report
- Optional: `embedding_sim.py` scorer
- Branch: `feat/AI-431-phase-7-multi-model`
**Done when:** One command runs the same suite across two models and emits a
comparison report a model-selection decision could be based on.

---

## 8. CLAUDE.md — required content (create in Phase 0)

The session-context file must contain at least:

- **Project one-liner** and a pointer to this PROJECT_PLAN.md as the source of truth
- **Commands:** `uv sync`, `uv run pytest`, `uv run ruff check .`,
  `uv run python run_evals.py --suite summarisation`
- **Current phase marker** — a single line stating which phase is active so a new
  session knows where we are. Update it as phases complete.
- **Code style:** type hints on all functions; docstrings on public functions;
  dataclasses not Pydantic; `ruff` formatting; no bare excepts
- **Hard rules:**
  - Never commit `.env` or `results/*.json` except `baseline.json`
  - No real API calls in tests — always mock via `tests/fixtures/`
  - Any new scorer implements the Scorer Protocol; any new adapter implements
    the ModelAdapter Protocol; never special-case them in the runner
  - Do not add dependencies outside the approved list without asking
  - Do not build beyond the current phase
  - Every phase ships on its own feature branch `feat/<JIRA-KEY>-<slug>`;
    never commit phase work directly to `main`
  - Bump `dataset_version` in `evals/summarisation.yaml` whenever cases are
    added, removed, or modified — and refresh `results/baseline.json` in the
    same PR
- **Compound-engineering note:** when I correct a mistake, add a rule here so it
  doesn't recur.

---

## 9. How we will work together

- I want to understand each piece before we move on. Explain design choices and
  tradeoffs as we go; don't just generate code.
- At the start of each phase, restate the goal and the definition of done, then
  propose the order of files to build.
- Build the data contracts and tests for a layer before its implementation where
  practical (the SDET habit — tests describe the contract).
- At the end of each phase: run the suite, update the current-phase marker in
  CLAUDE.md, and summarise what changed.
- If you find yourself wanting to build something from a later phase, stop and
  flag it instead.

### Branch + PR workflow (every phase)

- **One feature branch per phase**, named `feat/<JIRA-KEY>-<short-slug>` — e.g.
  `feat/AI-424-phase-0-scaffold`, `feat/AI-425-phase-1-contracts`. The Jira key
  is mandatory; it links the branch to the ticket.
- Cut the branch off `main` at the start of the phase. Never commit phase work
  directly to `main`.
- Commits inside the branch reference the Jira key in the message
  (e.g. `AI-425: add EvalSuite container`).
- At the end of the phase: open a PR with the Jira key in the title, the
  phase's acceptance criteria as a checklist, and a link back to the ticket.
  Squash-merge into `main` only after CI is green and acceptance criteria
  are checked off.
- After merge: update the Jira ticket to Done and bump the CLAUDE.md
  current-phase marker to the next phase.

---

## 10. First action for this session

We are starting **Phase 0** under Jira ticket **AI-424**. Before writing any code:

1. Cut a branch `feat/AI-424-phase-0-scaffold` off `main`.
2. Confirm the directory structure for Phase 0–1 files.
3. Propose `pyproject.toml` (with `mypy --strict` + `ruff` config), `.gitignore`,
   `.env.example`, `LICENSE` (MIT), `.pre-commit-config.yaml`, and the initial
   `CLAUDE.md`.
4. Wait for my go-ahead, then scaffold.

Do not begin Phase 1 implementation until Phase 0 is complete, the PR is
merged, and I confirm.
