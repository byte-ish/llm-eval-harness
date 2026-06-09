# Eval suites — YAML schema reference

A suite is a YAML file inside `evals/` that the harness loads via
`harness.config.load_suite()`. This document describes the schema and the
worked examples for each scorer.

## File-level fields

| Field | Required? | Notes |
| --- | --- | --- |
| `name` | optional | Suite name. Defaults to the file stem if omitted. |
| `dataset_version` | **required** | Semver-style string. Bump on every dataset change. The Phase 5 regression detector refuses to compare runs across mismatched versions. |
| `default_system` | optional | Suite-wide system prompt. Each case can override. |
| `default_user` | optional | Suite-wide user template. Rare — usually each case provides its own. |
| `cases` | **required** | List of case objects. |

## Case-level fields

| Field | Required? | Notes |
| --- | --- | --- |
| `id` | **required** | Unique within the suite. |
| `category` | **required** | Used for per-category pass-rate regression checks. |
| `scorer` | **required** | One of: `exact_match`, `regex_match` (Phase 3), `llm_judge` (Phase 4). |
| `user` | optional | Falls back to `default_user`. Loading fails if both are missing. |
| `system` | optional | Falls back to `default_system`. Loading fails if both are missing. |
| `expected_contains` | conditional | Required (non-empty) for `exact_match`. |
| `expected_regex` | conditional | Required for `regex_match`. |
| `judge_rubric` | conditional | Required for `llm_judge`. |
| `temperature` | optional | Per-case override of the default `0.0`. |
| `tags` | optional | Free-form list of strings for filtering/annotation. |

## Worked examples

### `exact_match`

Every entry in `expected_contains` must appear in the model response
(case-insensitive). `score = matched / total`, `passed = all matched`.

```yaml
- id: earnings_001
  category: earnings_summary
  scorer: exact_match
  user: |
    Acme Corp announced Q3 2026 revenue of $4.2 billion...
  expected_contains: ["Acme", "Q3", "$4.2 billion"]
```

### `regex_match` (Phase 3 — schema only here)

Single regex over the response. `passed` if the regex finds a match.

```yaml
- id: date_format_001
  category: formatting
  scorer: regex_match
  user: "Return today's date as YYYY-MM-DD."
  expected_regex: "^\\d{4}-\\d{2}-\\d{2}$"
```

### `llm_judge` (Phase 4 — schema only here)

A separate judge model scores the response against a rubric and returns JSON
`{score: 0..1, reason: str}`.

```yaml
- id: tone_001
  category: tone
  scorer: llm_judge
  user: "Reply to a frustrated customer."
  judge_rubric: |
    Score the response from 0 to 1 on professional tone, empathy, and
    actionability. Reject responses that are dismissive or vague.
```

## Versioning rule

Whenever you add, remove, or modify a case, **bump `dataset_version`** in the
suite file and refresh `results/baseline.json` in the same PR. The harness
refuses to compare runs across mismatched dataset versions unless explicitly
overridden with `--allow-dataset-mismatch`.
