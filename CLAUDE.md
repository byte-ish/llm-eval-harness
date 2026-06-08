# CLAUDE.md — llm-eval-harness

A versioned, CI-gated test suite for LLM features. Treats LLM output quality
the way an SDET treats code quality: regression-tested, fixture-based,
reviewable.

**Source of truth:** [`project_plan.md`](./project_plan.md). Read it at the
start of every session before making any changes. The plan governs scope,
sequencing, and acceptance criteria for each phase.

## Current phase

**Phase 1 — AI-425 — data contracts + first scorer** (active branch: `feat/AI-425-phase-1-contracts`)

Update this line when a phase merges and the next phase begins.

## Commands

| Task | Command |
|---|---|
| Install deps | `uv sync` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Type-check | `uv run mypy --strict harness/` |
| Pre-commit (all files) | `uv run pre-commit run --all-files` |
| Run an eval suite (Phase 2+) | `uv run python run_evals.py --suite summarisation` |

## Code style

- Type hints on every function. `mypy --strict` runs in CI.
- No `Any` in public signatures; no implicit `Optional`.
- Docstrings on public functions and classes; not required on private helpers.
- Use `@dataclass(frozen=True)` for contracts. Do NOT use Pydantic.
- `ruff format` is the formatter — do not hand-format.
- No bare `except:` — always catch a specific exception.

## Hard rules

- Never commit `.env` or `results/*.json` (the `.gitignore` allow-list keeps
  `baseline.json` only).
- No real API calls in tests. Always mock via `tests/fixtures/`.
- Every new scorer implements the `Scorer` Protocol; every new adapter
  implements the `ModelAdapter` Protocol. Never special-case them in the
  runner.
- Do not add dependencies outside the table in `project_plan.md` §3 without
  asking.
- Do not build beyond the current phase. If a later-phase need appears, flag
  it and stop.
- Every phase ships on its own feature branch `feat/<JIRA-KEY>-<slug>`. Never
  commit phase work directly to `main`. Commit messages reference the Jira
  key; PR title format is `<JIRA-KEY>: <phase title>`.
- Bump `dataset_version` in `evals/summarisation.yaml` whenever a case is
  added, removed, or modified — and refresh `results/baseline.json` in the
  same PR.

## Compound-engineering note

When a mistake is corrected during a session, add the rule that would have
prevented it to this file under "Hard rules" so it doesn't recur.
