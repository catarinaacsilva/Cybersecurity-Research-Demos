"""PsDC Categorizer server (default port 8002).

REST: ``GET /api/meta`` (fields with their automatic and expert categories, structural
options), plus a REST mirror of the live operation. WebSocket ``/ws`` live operation:

- ``model {fields, frequency, retention, granularity}``: categorize the app's data model,
  measure its dynamic tags and score it.
"""

from __future__ import annotations

from pathlib import Path

from catdemos.common.adult import NAMES
from catdemos.common.server import Args, Live, LiveError, make_app, names
from catdemos.psdc.categorize import FREQUENCY, RETENTION, Categorizer

app = make_app("PsDC Categorizer", Path(__file__).parent / "static")
categorizer = Categorizer()
# Warm-up: the first profiler fit in a process is ~1 s (solver and thread-pool start-up), so run
# the page's starting model now rather than on the first visitor's click.
categorizer.model(["age", "sex", "occupation"], "daily", "1 year", "exact")


def _choice(a: Args, name: str, options: list[str], default: str) -> str:
    """One of ``options`` (the default when absent)."""
    v = str(a.get(name, default))
    if v not in options:
        raise LiveError(f"{name} must be one of {options}")
    return v


def _model(a: Args) -> dict:
    """Live op ``model``: validate the data model and evaluate it."""
    return categorizer.model(
        names(a, "fields", NAMES),
        _choice(a, "frequency", list(FREQUENCY), "daily"),
        _choice(a, "retention", list(RETENTION), "1 year"),
        _choice(a, "granularity", ["exact", "coarse"], "exact"),
    )


live = Live(app, {"model": _model})


@app.get("/api/meta")
def meta() -> dict:
    """Setup data: fields, their automatic and expert categories, structural options."""
    return categorizer.meta()


@app.get("/api/model")
def model(
    fields: str = "", frequency: str = "daily", retention: str = "1 year", granularity: str = "exact"
) -> dict:
    """REST mirror of the ``model`` live operation (fields as a comma-separated list)."""
    return live.call(
        "model",
        {"fields": fields, "frequency": frequency, "retention": retention, "granularity": granularity},
    )
