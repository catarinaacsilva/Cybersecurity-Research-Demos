"""DP vs Profiling server (default port 8003).

REST: ``GET /api/meta`` (targets, released fields, epsilon grid), plus a REST mirror of the
live operation. WebSocket ``/ws`` live operation:

- ``point {target, hidden, i}``: profiling and re-identification with the ``hidden`` fields
  released under DP at ``EPS_GRID[i]``, plus the part of the curve measured so far.
"""

from __future__ import annotations

from pathlib import Path

from catdemos.common.server import Args, Live, LiveError, arg, make_app, names
from catdemos.dp.study import EPS_GRID, STUDY_TARGETS, Study

app = make_app("DP vs Profiling", Path(__file__).parent / "static")
study = Study()
# Warm-up: pay the first-fit start-up cost now, with the page's starting setting.
study.point("married", list(study.released("married")), 15)


def _point(a: Args) -> dict:
    """Live op ``point``: validate target, hidden fields and epsilon index, then measure."""
    target = str(a.get("target", "married"))
    if target not in STUDY_TARGETS:
        raise LiveError(f"target must be one of {list(STUDY_TARGETS)}")
    hidden = names(a, "hidden", study.released(target))
    return study.point(target, hidden, arg(a, "i", int, 0, len(EPS_GRID) - 1))


live = Live(app, {"point": _point})


@app.get("/api/meta")
def meta() -> dict:
    """Setup data: targets, released fields per target, epsilon grid."""
    return study.meta()


@app.get("/api/point")
def point(i: int, target: str = "married", hidden: str = "") -> dict:
    """REST mirror of the ``point`` live operation (hidden as a comma-separated list)."""
    return live.call("point", {"target": target, "hidden": hidden, "i": i})
