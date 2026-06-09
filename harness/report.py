"""HTML report rendering — Jinja2 over the `RunReport` (+ optional `RegressionDiff`).

`render(report, diff, output_path)` writes a self-contained HTML page to
`output_path` and returns the path. The template lives inside the package at
`harness/templates/report.html.j2` so it ships with the wheel.
"""

from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from harness.models import RunReport
from harness.regression import RegressionDiff


def render(report: RunReport, diff: RegressionDiff | None, output_path: Path) -> Path:
    """Render `report` (+ optional `diff`) to an HTML file. Returns the path."""
    templates_dir = _templates_dir()
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("report.html.j2")
    html = template.render(report=report, diff=diff)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _templates_dir() -> Path:
    """Return the on-disk directory containing the Jinja2 templates.

    Uses `importlib.resources` so this works both in editable installs (uv sync)
    and shipped wheels.
    """
    ref = resources.files("harness").joinpath("templates")
    return Path(str(ref))
