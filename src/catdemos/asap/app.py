"""ASAP Latent Map server (default port 8001).

REST: ``GET /api/overview`` (map, histogram, default threshold), plus REST mirrors of the
live operations. WebSocket ``/ws`` live operations:

- ``probe {x, y}``: decode a latent point, run the detectors, label it.
- ``threshold {t}``: confusion matrix and MCC on the test split at threshold ``t``.
"""

from __future__ import annotations

from pathlib import Path

from catdemos.asap.model import LATENT_LIM, load_or_train
from catdemos.common.server import Args, Live, arg, make_app

app = make_app("ASAP Latent Map", Path(__file__).parent / "static")
engine = load_or_train()


def _probe(a: Args) -> dict:
    """Live op ``probe``: validate the latent point and score its decoded app."""
    return engine.probe(
        arg(a, "x", float, -LATENT_LIM, LATENT_LIM), arg(a, "y", float, -LATENT_LIM, LATENT_LIM)
    )


def _threshold(a: Args) -> dict:
    """Live op ``threshold``: confusion matrix at the requested threshold."""
    return engine.at_threshold(arg(a, "t", float, -1e3, 1e3))


live = Live(app, {"probe": _probe, "threshold": _threshold})


@app.get("/api/overview")
def overview() -> dict:
    """Setup data for the page: map background, sample dots, histogram, default threshold."""
    return engine.overview()


@app.get("/api/probe")
def probe(x: float, y: float) -> dict:
    """REST mirror of the ``probe`` live operation."""
    return live.call("probe", {"x": x, "y": y})


@app.get("/api/threshold")
def threshold(t: float) -> dict:
    """REST mirror of the ``threshold`` live operation."""
    return live.call("threshold", {"t": t})
