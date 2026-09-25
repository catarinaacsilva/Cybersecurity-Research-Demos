"""Shared server plumbing and dataset loaders."""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from catdemos.common.datasets import load_adult, load_naticus
from catdemos.common.server import make_app


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        static = Path(self.tmp.name)
        (static / "index.html").write_text("<h1>demo</h1>")
        self.client = TestClient(make_app("Demo", static))

    def tearDown(self):
        self.tmp.cleanup()

    def test_health_and_timing_header(self):
        r = self.client.get("/health")
        self.assertEqual(r.json(), {"status": "ok", "demo": "Demo"})
        self.assertGreaterEqual(float(r.headers["X-Compute-ms"]), 0.0)

    def test_index_and_shared_theme(self):
        self.assertIn("demo", self.client.get("/").text)
        self.assertIn("--orange: #FF8145", self.client.get("/common/theme.css").text)
        self.assertEqual(self.client.get("/common/client.js").status_code, 200)

    def test_logos_and_favicon_are_served_as_svg(self):
        for name in ("favicon", "ieeta", "ieeta-white", "ua", "ua-white"):
            r = self.client.get(f"/common/{name}.svg")
            self.assertEqual(r.status_code, 200, name)
            self.assertEqual(r.headers["content-type"], "image/svg+xml", name)
            self.assertIn("<title>", r.text, name)

    def test_live_socket_survives_bad_messages(self):
        from catdemos.common.server import Live, arg

        app = make_app("Demo", Path(self.tmp.name))
        Live(app, {"double": lambda a: 2 * arg(a, "x", float, 0, 10)})
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_text("{not json")
            self.assertEqual(ws.receive_json()["error"], "message is not JSON")
            ws.send_json([1, 2])
            self.assertEqual(ws.receive_json()["error"], "message must be an object")
            ws.send_json({"op": "double", "seq": 7, "args": {"x": "abc"}})
            self.assertIn("argument 'x'", ws.receive_json()["error"])
            ws.send_json({"op": "double", "seq": 8, "args": {}})
            self.assertIn("missing", ws.receive_json()["error"])
            ws.send_json({"op": "double", "seq": 9, "args": {"x": 4}})
            r = ws.receive_json()
            self.assertEqual((r["seq"], r["data"]), (9, 8.0))

    def test_live_socket_survives_handler_errors(self):
        from catdemos.common.server import Live

        def boom(_a):
            raise KeyError("engine bug")

        app = make_app("Demo", Path(self.tmp.name))
        Live(app, {"boom": boom, "echo": lambda a: a})
        with TestClient(app).websocket_connect("/ws") as ws, self.assertLogs("catdemos", "ERROR"):
            ws.send_json({"op": "boom", "seq": 1, "args": {}})
            self.assertEqual(ws.receive_json()["error"], "internal error")
            ws.send_json({"op": "echo", "seq": 2, "args": {"x": 1}})
            self.assertEqual(ws.receive_json()["data"], {"x": 1})

    def test_arg_rejects_non_finite_bool_and_fractional_int(self):
        from catdemos.common.server import LiveError, arg

        for bad in ("nan", float("inf"), "-inf", True, None, [1]):
            with self.assertRaises(LiveError, msg=repr(bad)):
                arg({"x": bad}, "x", float, -1, 1)
        with self.assertRaises(LiveError):
            arg({"i": 2.9}, "i", int, 0, 10)
        self.assertEqual(arg({"i": "3"}, "i", int, 0, 10), 3)
        self.assertEqual(arg({"i": 3.0}, "i", int, 0, 10), 3)
        self.assertEqual(arg({"x": "0.5"}, "x", float, -1, 1), 0.5)

    def test_names_accepts_lists_and_csv_only_of_allowed_strings(self):
        from catdemos.common.server import LiveError, names

        self.assertEqual(names({"f": ["a", "b"]}, "f", ["a", "b", "c"]), ["a", "b"])
        self.assertEqual(names({"f": "a,,c"}, "f", ["a", "b", "c"]), ["a", "c"])
        self.assertEqual(names({}, "f", ["a"]), [])
        for bad in (["z"], [{"a": 1}], [["a"]], 3, {"a": 1}):
            with self.assertRaises(LiveError, msg=repr(bad)):
                names({"f": bad}, "f", ["a", "b"])

    def test_reported_time_excludes_idle_wait(self):
        import time

        from catdemos.common.server import Live

        app = make_app("Demo", Path(self.tmp.name))
        Live(app, {"echo": lambda a: a})
        with TestClient(app).websocket_connect("/ws") as ws:
            time.sleep(0.3)  # the client is idle before sending
            ws.send_json({"op": "echo", "seq": 1, "args": {"x": 1}})
            self.assertLess(ws.receive_json()["ms"], 100)

    def test_assets_are_revalidated(self):
        self.assertEqual(self.client.get("/common/theme.css").headers["cache-control"], "no-cache")

    def test_cors_for_hub_health_polling(self):
        r = self.client.get("/health", headers={"Origin": "http://localhost:8000"})
        self.assertEqual(r.headers["access-control-allow-origin"], "*")


class DatasetTest(unittest.TestCase):
    def test_cached_archives_match_pinned_hashes(self):
        import hashlib

        from catdemos.common.datasets import SHA256, fetch

        for name, digest in SHA256.items():
            self.assertEqual(hashlib.sha256(fetch(name).read_bytes()).hexdigest(), digest)

    def test_naticus_shape_and_labels(self):
        d = load_naticus()
        self.assertEqual(d.X.shape, (29332, 86))
        self.assertEqual(len(d.perms), 86)
        self.assertEqual(set(d.y.tolist()), {0, 1})

    def test_adult_rows_are_complete(self):
        rows = load_adult()
        self.assertGreater(len(rows), 45000)
        self.assertEqual({r["income"] for r in rows}, {"<=50K", ">50K"})
        self.assertFalse(any("?" in r.values() for r in rows[:1000]))


if __name__ == "__main__":
    unittest.main()
