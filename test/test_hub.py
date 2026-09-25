"""Hub server, demo registry and command-line entry points."""

import io
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from fastapi.testclient import TestClient

from catdemos import __main__ as cli
from catdemos.common import server
from catdemos.demos import DEMOS, HUB_PORT
from catdemos.hub.app import app


class HubTest(unittest.TestCase):
    def test_lists_every_demo(self):
        c = TestClient(app)
        demos = c.get("/api/demos").json()
        self.assertEqual([d["key"] for d in demos], ["asap", "psdc", "dp", "elastic"])
        self.assertEqual(len({d["port"] for d in demos} | {HUB_PORT}), 5)  # no port clashes
        self.assertIn("Catarina Silva", c.get("/").text)

    def test_every_page_has_both_logos_and_the_favicon(self):
        from pathlib import Path

        import catdemos

        for key in ["hub", *(d.key for d in DEMOS)]:
            html = (Path(catdemos.__file__).parent / key / "static" / "index.html").read_text()
            for asset in ("/common/favicon.svg", "/common/ieeta-white.svg", "/common/ua-white.svg"):
                self.assertIn(asset, html, key)


class CliTest(unittest.TestCase):
    def test_usage(self):
        with mock.patch.object(sys, "argv", ["catdemos"]), self.assertRaises(SystemExit) as e:
            cli.main()
        self.assertIn("usage", str(e.exception.code))

    def test_warmup_reports_each_model(self):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["catdemos", "warmup"]), redirect_stdout(out):
            cli.main()
        self.assertEqual(out.getvalue().count("ready"), 3)

    def test_serve_parses_host_and_port(self):
        with mock.patch.object(sys, "argv", ["x", "--port", "9123"]), mock.patch("uvicorn.run") as run:
            server.serve("catdemos.hub.app:app", 8000)
        run.assert_called_once_with("catdemos.hub.app:app", host="127.0.0.1", port=9123, log_level="warning")

    def test_module_entry_points_use_registry_ports(self):
        ports = {d.key: d.port for d in DEMOS} | {"hub": HUB_PORT}
        for key, port in ports.items():
            with mock.patch.object(server, "serve") as serve:
                runpy.run_module(f"catdemos.{key}", run_name="__main__")
            serve.assert_called_once_with(f"catdemos.{key}.app:app", port)


if __name__ == "__main__":
    unittest.main()
