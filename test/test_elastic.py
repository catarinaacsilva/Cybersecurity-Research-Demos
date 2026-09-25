"""Elastic Privacy engine and API."""

import unittest

import numpy as np
from fastapi.testclient import TestClient

from catdemos.elastic import app as elastic_app
from catdemos.elastic import pipeline as P
from catdemos.elastic.osm import _grid, to_latlng, to_xy


class SignalTest(unittest.TestCase):
    def test_projection_round_trip(self):
        ll = np.array([[40.6381, -8.6577], [40.6320, -8.6598]])
        np.testing.assert_allclose(to_latlng(to_xy(ll)), ll, atol=1e-9)
        # Fábrica to the campus is ~700 m.
        self.assertAlmostEqual(np.linalg.norm(np.diff(to_xy(ll), axis=0)), 700, delta=60)

    def test_resample_normalises_drawing_speed(self):
        xy = np.column_stack([np.linspace(0, 300, 301), np.zeros(301)])
        t = np.linspace(0, 3.0, 301)  # drawn in 3 s -> should become a ~220 s walk
        out = P.resample(xy, t)
        speed = np.linalg.norm(np.diff(out, axis=0), axis=1)
        self.assertAlmostEqual(np.median(speed), P.WALK_SPEED, delta=0.05)

    def test_gps_error_is_stable_and_correlated(self):
        xy = np.zeros((2000, 2))
        a, b = P.gps(xy, 3), P.gps(xy, 3)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(a.std(), P.GPS_SIGMA, delta=0.6)
        self.assertLess(np.diff(a, axis=0).std(), 1.0)  # small per-second jitter

    def test_features_shape(self):
        rng = np.random.default_rng(0)
        route = _grid().random_routes(1, rng)[0]
        xy = P.walk(route, rng)
        F = P.features(xy)
        self.assertEqual(F.shape, (len(xy), len(P.FEATURES)))
        self.assertTrue(np.isfinite(F).all())
        self.assertEqual(P.features(xy[:5]).shape, (5, len(P.FEATURES)))

    def test_crisis_injection(self):
        rng = np.random.default_rng(1)
        xy = P.walk(_grid().random_routes(1, rng)[0], rng)
        out, lab = P.inject_crisis(xy, rng)
        self.assertEqual(out.shape, xy.shape)
        self.assertTrue(60 <= lab.sum() <= 150 or lab.sum() == len(xy) // 2)

    def test_policy_bounds(self):
        s = np.random.default_rng(0).normal(0, 3, (500, P.N_IN))
        r = P.policy(np.random.default_rng(1).normal(0, 1, P.N_THETA), s)
        self.assertTrue(((r >= P.R_MIN) & (r <= P.R_MAX)).all())


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = elastic_app.engine

    def test_benchmark_shows_elasticity(self):
        b = self.eng.bench
        self.assertGreater(b["shield_m"], 3 * b["crisis_error_m"])
        self.assertGreater(b["crisis_detected"], 0.8)

    def test_sample_walk(self):
        pts = np.asarray(self.eng.sample(2), dtype=float)
        d = self.eng.walk_latlng(pts)
        self.assertEqual(len(d["raw"]), d["seconds"])
        self.assertGreater(d["summary"]["alert_seconds"], 0)
        self.assertTrue(all(P.R_MIN <= r <= P.R_MAX for r in d["radius"]))

    def test_grid_fallback_engine_is_not_cached(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            fake = SimpleNamespace(graph=SimpleNamespace(source="grid"))
            with (
                mock.patch.object(P, "cache_dir", return_value=Path(tmp)),
                mock.patch.object(P, "train", return_value=fake),
            ):
                self.assertIs(P.load_or_train(), fake)
            self.assertEqual(list(Path(tmp).iterdir()), [])


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(elastic_app.app)

    def test_endpoints(self):
        self.assertIn("bench", self.c.get("/api/meta").json())
        pts = self.c.get("/api/sample", params={"seed": 4}).json()["points"]
        r = self.c.post("/api/walk", json={"points": pts})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["obf"]), r.json()["seconds"])

    def test_routine_sample_raises_few_alerts(self):
        pts = self.c.get("/api/sample", params={"seed": 7, "crisis": False}).json()["points"]
        d = self.c.post("/api/walk", json={"points": pts}).json()
        self.assertLess(d["summary"]["alert_seconds"] / d["seconds"], 0.3)

    def test_bad_walk(self):
        self.assertEqual(self.c.post("/api/walk", json={"points": [[1, 2]]}).status_code, 422)
        self.assertEqual(self.c.post("/api/walk", json={"points": [[1, 2, 3]]}).status_code, 422)

    def test_websocket_walk(self):
        pts = self.c.get("/api/sample", params={"seed": 5}).json()["points"][:120]
        with self.c.websocket_connect("/ws") as ws:
            ws.send_json({"op": "walk", "seq": 1, "args": {"points": pts}})
            r = ws.receive_json()
            self.assertEqual(r["data"]["seconds"], len(r["data"]["radius"]))
            ws.send_json({"op": "walk", "seq": 2, "args": {"points": "x"}})
            self.assertIn("error", ws.receive_json())

    def test_frontend_served(self):
        self.assertIn("Elastic Privacy Map", self.c.get("/").text)
        self.assertEqual(self.c.get("/static/elastic.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
