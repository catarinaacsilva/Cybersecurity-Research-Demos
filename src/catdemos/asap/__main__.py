"""Run this demo server: ``python -m catdemos.<demo> [--host H] [--port P]``."""

from catdemos.common.server import serve
from catdemos.demos import DEMOS

if __name__ == "__main__":
    serve("catdemos.asap.app:app", next(d.port for d in DEMOS if d.key == "asap"))
