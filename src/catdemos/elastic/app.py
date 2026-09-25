"""Elastic Privacy Map server (default port 8004).

REST: ``GET /api/meta`` (map anchors, radius bounds, held-out benchmark vs paper) and
``GET /api/sample`` (a simulated walk with an injected crisis), plus a REST mirror of the
live operation. WebSocket ``/ws`` live operation:

- ``walk {points: [[lat, lng, t_ms], ...]}``: the whole pipeline on the path drawn so far.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from catdemos.common.server import Args, Live, LiveError, make_app
from catdemos.elastic.osm import BBOX, CAMPUS, FABRICA
from catdemos.elastic.pipeline import FEATURES, R_MAX, R_MIN, load_or_train

app = make_app("Elastic Privacy Map", Path(__file__).parent / "static")
engine = load_or_train()
MAX_POINTS = 20_000


def _walk(a: Args) -> dict:
    """Live op ``walk``: validate the drawn points and run the whole pipeline."""
    try:
        pts = np.asarray(a.get("points"), dtype=float)
    except (TypeError, ValueError):
        raise LiveError("points must be numeric [lat, lng, t_ms] triples") from None
    if pts.ndim != 2 or pts.shape[1] != 3 or not 2 <= len(pts) <= MAX_POINTS or not np.isfinite(pts).all():
        raise LiveError(f"points must be 2..{MAX_POINTS} finite [lat, lng, t_ms] triples")
    return engine.walk_latlng(pts)


live = Live(app, {"walk": _walk})


class Walk(BaseModel):
    """REST body for ``POST /api/walk``."""

    points: list[list[float]] = Field(max_length=MAX_POINTS)


@app.get("/api/meta")
def meta() -> dict:
    """Setup data: map anchors, radius bounds and the held-out benchmark against the paper."""
    return {
        "fabrica": FABRICA,
        "campus": CAMPUS,
        "bbox": BBOX,
        "r_min": R_MIN,
        "r_max": R_MAX,
        "features": FEATURES,
        "graph": engine.graph.source,
        "bench": engine.bench,
    }


@app.post("/api/walk")
def walk(w: Walk) -> dict:
    """REST mirror of the ``walk`` live operation."""
    return live.call("walk", {"points": w.points})


@app.get("/api/sample")
def sample(seed: int = 1, crisis: bool = True) -> dict:
    """A simulated walk on real Aveiro streets, with one injected crisis or routine only."""
    return {"points": engine.sample(seed, crisis)}
