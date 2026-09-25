"""Run this demo server: ``python -m catdemos.<demo> [--host H] [--port P]``."""

from catdemos.common.server import serve
from catdemos.demos import HUB_PORT

if __name__ == "__main__":
    serve("catdemos.hub.app:app", HUB_PORT)
