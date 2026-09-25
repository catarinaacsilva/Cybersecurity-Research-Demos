"""Shared FastAPI plumbing for every demo server.

REST carries setup and structured data (``/api/meta``, overviews, samples) and mirrors every
live operation for scripting and tests. WebSocket ``/ws`` carries the live, pointer-driven
updates: the client sends ``{"op", "seq", "args"}`` and receives ``{"op", "seq", "ms",
"data" | "error"}``. Both paths call the same handler through :class:`Live`, which
serialises access to the (non thread-safe) engines with a lock.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

COMMON_STATIC = Path(__file__).parent / "static"
log = logging.getLogger("catdemos")


def make_app(title: str, static_dir: Path) -> FastAPI:
    """App serving ``static_dir/index.html`` at ``/`` plus the shared theme at ``/common``."""
    app = FastAPI(title=title, docs_url="/docs", redoc_url=None)
    # The hub (another port) polls /health, so cross-origin GETs must be allowed.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

    @app.middleware("http")
    async def compute_time(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        """Add the server compute time and cache revalidation headers."""
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Compute-ms"] = f"{(time.perf_counter() - t0) * 1e3:.2f}"
        response.headers["Access-Control-Expose-Headers"] = "X-Compute-ms"
        # Revalidate pages and assets (ETag), so a restarted server never runs stale viewer code.
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe used by the hub and by run.sh."""
        return {"status": "ok", "demo": title}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """The demo's page."""
        return FileResponse(static_dir / "index.html")

    app.mount("/common", StaticFiles(directory=COMMON_STATIC), name="common")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


Args = dict[str, Any]
Handler = Callable[[Args], Any]


class LiveError(ValueError):
    """Invalid client arguments: sent back as an error, never raised to the server."""


def arg(args: Args, name: str, kind: type = float, lo: float | None = None, hi: float | None = None) -> Any:
    """Fetch and validate one argument of a live operation (finite numbers; ints must be integral)."""
    if name not in args:
        raise LiveError(f"missing argument {name!r}")
    raw = args[name]
    if isinstance(raw, bool):  # bool is an int subclass: True would pass as 1
        raise LiveError(f"argument {name!r} must be a number")
    try:
        v = float(raw)
    except (TypeError, ValueError) as e:
        raise LiveError(f"argument {name!r}: {e}") from None
    if not math.isfinite(v):
        raise LiveError(f"argument {name!r} must be finite")
    if kind is int:
        if not v.is_integer():
            raise LiveError(f"argument {name!r} must be an integer")
        v = int(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise LiveError(f"argument {name!r}={v} outside [{lo}, {hi}]")
    return v


def names(args: Args, name: str, allowed: list[str] | tuple[str, ...]) -> list[str]:
    """A list of names drawn from ``allowed`` (a JSON list, or a comma-separated string from REST)."""
    v = args.get(name, [])
    if isinstance(v, str):
        v = [x for x in v.split(",") if x]
    if not isinstance(v, list) or not all(isinstance(x, str) and x in allowed for x in v):
        raise LiveError(f"{name} must be a list drawn from {list(allowed)}")
    return v


class Live:
    """Named live operations, served on ``/ws`` and callable from REST endpoints."""

    def __init__(self, app: FastAPI, ops: dict[str, Handler]) -> None:
        """Register ``ops`` and serve them on ``/ws``."""
        self.ops = ops
        self._lock = threading.Lock()

        @app.websocket("/ws")
        async def ws(sock: WebSocket) -> None:
            """One live session: answer each request in order, off the event loop."""
            await sock.accept()
            try:
                while True:
                    text = await sock.receive_text()
                    t0 = time.perf_counter()  # after the wait: "ms" is server work only
                    try:
                        msg = json.loads(text)
                    except ValueError:  # not JSON: answer, keep the session alive
                        await sock.send_json({"op": None, "seq": None, "error": "message is not JSON"})
                        continue
                    if not isinstance(msg, dict):
                        await sock.send_json({"op": None, "seq": None, "error": "message must be an object"})
                        continue
                    reply: dict[str, Any] = {"op": msg.get("op"), "seq": msg.get("seq")}
                    try:
                        reply["data"] = await run_in_threadpool(
                            self._run, msg.get("op"), msg.get("args") or {}
                        )
                    except LiveError as e:
                        reply["error"] = str(e)
                    except Exception:  # an engine bug must not end the session (the client would resend)
                        log.exception("live op %r failed", reply["op"])
                        reply["error"] = "internal error"
                    reply["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
                    await sock.send_json(reply)
            except WebSocketDisconnect:
                return

    def _run(self, op: Any, args: Args) -> Any:
        """Validate the operation name and arguments, then run the handler under the engine lock."""
        if op not in self.ops:
            raise LiveError(f"unknown operation {op!r}")
        if not isinstance(args, dict):
            raise LiveError("args must be an object")
        with self._lock:
            return self.ops[op](args)

    def call(self, op: str, args: Args) -> Any:
        """REST entry point: same handler, errors mapped to HTTP 422."""
        try:
            return self._run(op, args)
        except LiveError as e:
            raise HTTPException(422, str(e)) from None


def serve(app_path: str, default_port: int) -> None:
    """``python -m catdemos.<demo> [--host H] [--port P]``."""
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=default_port)
    a = ap.parse_args()
    uvicorn.run(app_path, host=a.host, port=a.port, log_level="warning")
