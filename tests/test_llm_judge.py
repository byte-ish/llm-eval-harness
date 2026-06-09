"""Tests for `harness.scorers.llm_judge` — judge-driven subjective scoring."""

import pytest

from harness.adapters.base import AdapterResponse
from harness.models import EvalCase
from harness.scorers.llm_judge import LLMJudgeScorer
from tests.fixtures.mock_responses import MockAdapter, MockResponse


def _case_with_rubric(rubric: str | None) -> EvalCase:
    return EvalCase(
        id="c1",
        category="cat",
        scorer="llm_judge",
        user="u",
        system="s",
        judge_rubric=rubric,
    )


def _flat_cost(_inp: int, _out: int) -> float:
    return 0.0001


class TestLLMJudgeValidJson:
    async def test_high_score_passes(self) -> None:
        judge = MockAdapter(
            responses=[MockResponse(text='{"score": 0.9, "reason": "Excellent coverage"}')]
        )
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score(
            "Some candidate response.", _case_with_rubric("Score for coverage.")
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.reason == "Excellent coverage"
        assert result.cost_usd == pytest.approx(0.0001)

    async def test_low_score_does_not_pass(self) -> None:
        judge = MockAdapter(
            responses=[MockResponse(text='{"score": 0.3, "reason": "Misses key points"}')]
        )
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is False
        assert result.score == 0.3

    async def test_threshold_boundary(self) -> None:
        # score == threshold passes
        judge = MockAdapter(responses=[MockResponse(text='{"score": 0.5, "reason": "borderline"}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost, pass_threshold=0.5)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is True

    async def test_custom_threshold(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text='{"score": 0.6, "reason": "good"}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost, pass_threshold=0.8)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is False


class TestLLMJudgeMalformedJson:
    async def test_invalid_json_recorded_as_parse_error(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text="not json at all")])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is False
        assert result.score == 0.0
        assert "judge parse error" in result.reason

    async def test_missing_score_field(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text='{"reason": "no score field"}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is False
        assert "score" in result.reason

    async def test_missing_reason_field(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text='{"score": 0.9}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is False
        assert "reason" in result.reason

    async def test_score_is_not_a_number(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text='{"score": "high", "reason": "ok"}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is False
        assert "not a number" in result.reason


class TestLLMJudgeOutOfRangeScore:
    async def test_score_above_one_clamped(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text='{"score": 1.5, "reason": "overshoot"}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.score == 1.0
        assert result.passed is True

    async def test_score_below_zero_clamped(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text='{"score": -0.4, "reason": "negative"}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.score == 0.0
        assert result.passed is False


class TestLLMJudgeMarkdownFences:
    async def test_strips_json_code_fence(self) -> None:
        judge = MockAdapter(
            responses=[MockResponse(text='```json\n{"score": 0.8, "reason": "good"}\n```')]
        )
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is True
        assert result.score == 0.8


class TestLLMJudgeAdapterError:
    async def test_adapter_error_recorded_as_scorer_failure(self) -> None:
        judge = MockAdapter(
            responses=[MockResponse(text="unused")],
            raise_on_call=[RuntimeError("judge model down")],
        )
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert result.passed is False
        assert "judge adapter error" in result.reason
        assert result.cost_usd == 0.0  # no completed call → no cost


class TestLLMJudgeConfigGuard:
    async def test_no_rubric_raises(self) -> None:
        judge = MockAdapter(responses=[MockResponse(text='{"score": 1, "reason": "ok"}')])
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=_flat_cost)
        with pytest.raises(ValueError, match="no judge_rubric"):
            await scorer.score("response", _case_with_rubric(None))


class TestLLMJudgeCostCapture:
    async def test_cost_fn_invoked_with_judge_tokens(self) -> None:
        captured: list[tuple[int, int]] = []

        def cost_fn(inp: int, out: int) -> float:
            captured.append((inp, out))
            return 0.002

        judge = MockAdapter(
            responses=[
                MockResponse(
                    text='{"score": 0.7, "reason": "ok"}',
                    input_tokens=120,
                    output_tokens=30,
                )
            ]
        )
        scorer = LLMJudgeScorer(judge_adapter=judge, judge_cost_fn=cost_fn)
        result = await scorer.score("response", _case_with_rubric("rubric"))
        assert captured == [(120, 30)]
        assert result.cost_usd == 0.002


class TestLLMJudgeAdapterShape:
    async def test_adapter_called_with_rubric_and_response(self) -> None:
        # Make sure the rubric AND response are passed into the judge prompt.
        captured_prompts: list[str] = []

        class CapturingAdapter:
            name = "capturing"

            async def complete(
                self,
                system: str,
                user: str,
                *,
                temperature: float = 0.0,
                max_tokens: int | None = None,
                timeout: float = 60.0,
            ) -> AdapterResponse:
                captured_prompts.append(user)
                return AdapterResponse(
                    text='{"score": 0.9, "reason": "ok"}',
                    input_tokens=10,
                    output_tokens=10,
                    latency_ms=1.0,
                    resolved_model_id="judge-1",
                )

        scorer = LLMJudgeScorer(
            judge_adapter=CapturingAdapter(),
            judge_cost_fn=_flat_cost,
        )
        await scorer.score("the candidate said FOO", _case_with_rubric("score how relevant it is"))
        assert "score how relevant it is" in captured_prompts[0]
        assert "the candidate said FOO" in captured_prompts[0]
