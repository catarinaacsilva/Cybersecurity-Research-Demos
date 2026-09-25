"""Hub server (default port 8000): the starter page linking to every demo server.

REST: ``GET /api/demos`` lists the demos with their ports; the page builds each link from
the host it was loaded from and polls every demo's ``/health`` for a status dot.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from catdemos.common.server import make_app
from catdemos.demos import DEMOS

app = make_app("Catarina Silva @ IEETA: demos", Path(__file__).parent / "static")


@app.get("/api/demos")
def demos() -> list[dict]:
    """The demo servers, their ports and what each card shows."""
    return [asdict(d) for d in DEMOS]
