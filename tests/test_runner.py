"""Tests for `harness.runner` — async dispatch + per-case error isolation."""

from harness import __version__
from harness.models import EvalCase, EvalSuite
from harness.runner import run_suite
from harness.scorers import ExactMatchScorer
from tests.fixtures.mock_responses import MockAdapter, MockResponse


def _suite_with(cases: list[EvalCase], dataset_version: str = "1.0.0") -> EvalSuite:
    return EvalSuite(
        name="test",
        dataset_version=dataset_version,
        default_system="s",
        default_user="u",
        cases=tuple(cases),
    )


def _case(case_id: str, expected: list[str]) -> EvalCase:
    return EvalCase(
        id=case_id,
        category="cat",
        scorer="exact_match",
        user="u",
        system="s",
        expected_contains=expected,
    )


def _scorers() -> dict[str, ExactMatchScorer]:
    return {"exact_match": ExactMatchScorer()}


def _zero_cost(_inp: int, _out: int) -> float:
    return 0.0


async def test_all_pass() -> None:
    cases = [_case("c1", ["hello"]), _case("c2", ["world"])]
    adapter = MockAdapter(
        responses=[
            MockResponse(text="hello there"),
            MockResponse(text="world wide"),
        ]
    )
    report = await run_suite(
        _suite_with(cases),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        show_progress=False,
    )
    assert report.pass_rate == 1.0
    assert len(report.results) == 2


async def test_partial_fail_via_adapter_error_does_not_abort_run() -> None:
    cases = [_case("c1", ["hello"]), _case("c2", ["world"])]
    adapter = MockAdapter(
        responses=[MockResponse(text="hello there"), MockResponse(text="world wide")],
        raise_on_call=[None, RuntimeError("simulated 500")],
    )
    report = await run_suite(
        _suite_with(cases),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        show_progress=False,
    )
    assert len(report.results) == 2
    c2 = next(r for r in report.results if r.case_id == "c2")
    assert c2.error == "simulated 500"
    assert c2.scorer_result.passed is False
    c1 = next(r for r in report.results if r.case_id == "c1")
    assert c1.scorer_result.passed is True


async def test_dataset_version_propagated() -> None:
    cases = [_case("c1", ["hello"])]
    adapter = MockAdapter(responses=[MockResponse(text="hello")])
    report = await run_suite(
        _suite_with(cases, dataset_version="9.9.9"),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        show_progress=False,
    )
    assert report.dataset_version == "9.9.9"


async def test_resolved_model_id_on_report() -> None:
    cases = [_case("c1", ["hello"])]
    adapter = MockAdapter(
        responses=[MockResponse(text="hello", resolved_model_id="claude-opus-4-7-20260101")]
    )
    report = await run_suite(
        _suite_with(cases),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        show_progress=False,
    )
    assert report.model == "claude-opus-4-7-20260101"
    assert report.results[0].model == "claude-opus-4-7-20260101"


async def test_temperature_override_flows_through() -> None:
    cases = [_case("c1", ["x"])]
    adapter = MockAdapter(responses=[MockResponse(text="x")])
    report = await run_suite(
        _suite_with(cases),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        temperature_override=0.4,
        show_progress=False,
    )
    assert report.results[0].temperature == 0.4


async def test_per_case_temperature_used_when_no_override() -> None:
    case = EvalCase(
        id="c1",
        category="cat",
        scorer="exact_match",
        user="u",
        system="s",
        expected_contains=["x"],
        temperature=0.7,
    )
    adapter = MockAdapter(responses=[MockResponse(text="x")])
    report = await run_suite(
        _suite_with([case]),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        show_progress=False,
    )
    assert report.results[0].temperature == 0.7


async def test_unknown_scorer_recorded_as_error_not_crash() -> None:
    case = EvalCase(
        id="c1",
        category="cat",
        scorer="not_a_real_scorer",
        user="u",
        system="s",
        expected_contains=["x"],
    )
    adapter = MockAdapter(responses=[MockResponse(text="x")])
    report = await run_suite(
        _suite_with([case]),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        show_progress=False,
    )
    assert report.results[0].error is not None
    assert "scorer" in report.results[0].error.lower()
    assert report.results[0].scorer_result.passed is False


async def test_harness_version_recorded() -> None:
    cases = [_case("c1", ["hello"])]
    adapter = MockAdapter(responses=[MockResponse(text="hello")])
    report = await run_suite(
        _suite_with(cases),
        adapter,
        _scorers(),
        cost_fn=_zero_cost,
        show_progress=False,
    )
    assert report.harness_version == __version__


async def test_cost_fn_invoked_with_tokens() -> None:
    captured: list[tuple[int, int]] = []

    def fake_cost(inp: int, out: int) -> float:
        captured.append((inp, out))
        return 0.01

    cases = [_case("c1", ["hello"])]
    adapter = MockAdapter(responses=[MockResponse(text="hello", input_tokens=42, output_tokens=17)])
    report = await run_suite(
        _suite_with(cases),
        adapter,
        _scorers(),
        cost_fn=fake_cost,
        show_progress=False,
    )
    assert captured == [(42, 17)]
    assert report.results[0].cost_usd == 0.01
