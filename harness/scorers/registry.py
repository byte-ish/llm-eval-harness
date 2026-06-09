"""Built-in scorer registry — `name → Scorer instance`.

`build_default_registry()` returns the ship-set. The runner never imports
concrete scorer types; it looks them up by `case.scorer` against whatever
registry the caller hands it. New scorers (Phase 4 `llm_judge`, Phase 7
`embedding_sim`) drop in here without touching the runner.
"""

from harness.scorers.base import Scorer
from harness.scorers.exact_match import ExactMatchScorer
from harness.scorers.regex_match import RegexMatchScorer


def build_default_registry() -> dict[str, Scorer]:
    """Return the default scorer registry (one instance of each)."""
    return {
        ExactMatchScorer.name: ExactMatchScorer(),
        RegexMatchScorer.name: RegexMatchScorer(),
    }
