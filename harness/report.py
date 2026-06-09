"""HTML report rendering — Jinja2 over `RunReport` and `ComparisonReport`.

`render(report, diff, output_path)` writes the single-run report.
`render_comparison(comparison, output_path)` writes the multi-model
side-by-side report. Both templates live inside the package at
`harness/templates/` so they ship with the wheel.
"""

from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from harness.compare import ComparisonReport
from harness.models import RunReport
from harness.regression import RegressionDiff


def render(report: RunReport, diff: RegressionDiff | None, output_path: Path) -> Path:
    """Render `report` (+ optional `diff`) to an HTML file. Returns the path."""
    env = _env()
    template = env.get_template("report.html.j2")
    html = template.render(report=report, diff=diff)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_comparison(comparison: ComparisonReport, output_path: Path) -> Path:
    """Render a multi-model `ComparisonReport` to an HTML file. Returns the path."""
    env = _env()
    template = env.get_template("report_compare.html.j2")
    html = template.render(comparison=comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_templates_dir())),
        autoescape=select_autoescape(["html", "j2"]),
    )


def _templates_dir() -> Path:
    """Return the on-disk directory containing the Jinja2 templates.

    Uses `importlib.resources` so this works both in editable installs (uv sync)
    and shipped wheels.
    """
    ref = resources.files("harness").joinpath("templates")
    return Path(str(ref))
