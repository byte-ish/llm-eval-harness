"""Built-in scorer registry.

`build_default_registry()` returns the ship-set. The runner never imports
concrete scorer types — it looks them up by `case.scorer` against whatever
registry the caller hands it.

`LLMJudgeScorer` is conditional: it needs a judge adapter + cost_fn, so the
caller passes them in. When omitted, the registry skips `llm_judge`. The CLI
inspects the suite to decide whether to build the judge adapter at all.

`DEFAULT_SCORER_NAMES` is the set of *registered* scorer names independent of
whether each is wired in — `load_suite(known_scorers=DEFAULT_SCORER_NAMES)`
catches typos like `llm-judge` at load time even when no judge adapter is
configured.
"""

from harness.adapters.base import ModelAdapter
from harness.budget import CostFn
from harness.scorers.base import Scorer
from harness.scorers.exact_match import ExactMatchScorer
from harness.scorers.llm_judge import LLMJudgeScorer
from harness.scorers.regex_match import RegexMatchScorer

DEFAULT_SCORER_NAMES: frozenset[str] = frozenset(
    {ExactMatchScorer.name, RegexMatchScorer.name, LLMJudgeScorer.name}
)


def build_default_registry(
    judge_adapter: ModelAdapter | None = None,
    judge_cost_fn: CostFn | None = None,
) -> dict[str, Scorer]:
    """Return the default scorer registry.

    Includes `llm_judge` only when both `judge_adapter` and `judge_cost_fn`
    are provided. Suites that contain no `llm_judge` cases need not pass them.
    """
    registry: dict[str, Scorer] = {
        ExactMatchScorer.name: ExactMatchScorer(),
        RegexMatchScorer.name: RegexMatchScorer(),
    }
    if judge_adapter is not None and judge_cost_fn is not None:
        registry[LLMJudgeScorer.name] = LLMJudgeScorer(judge_adapter, judge_cost_fn)
    return registry
