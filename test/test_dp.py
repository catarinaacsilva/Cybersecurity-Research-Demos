"""DP vs Profiling engine and API."""

import unittest

from fastapi.testclient import TestClient

from catdemos.dp import app as dp_app
from catdemos.dp.study import EPS_GRID


class StudyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = dp_app.study

    def test_nothing_hidden_means_no_effect(self):
        d = self.s.point("married", [], 0)
        self.assertEqual((d["profiling"], d["reid"]), (d["no_dp"]["profiling"], d["no_dp"]["reid"]))

    def test_dp_defeats_linkage_not_profiling(self):
        # The paper's claim: with every field hidden, a moderate epsilon kills re-identification
        # while profiling keeps much of its advantage over guessing.
        hidden = list(self.s.released("married"))
        d = self.s.point("married", hidden, len(EPS_GRID) - 4)
        self.assertLess(d["kept"]["reid"], 0.2)
        self.assertGreater(d["kept"]["profiling"], 0.5)

    def test_exact_relationship_keeps_profiling(self):
        hidden = [f for f in self.s.released("married") if f != "relationship"]
        d = self.s.point("married", hidden, 5)
        self.assertLess(d["reid"], 0.05)
        self.assertGreater(d["profiling"], 0.95)

    def test_curve_accumulates(self):
        for i in (2, 3):
            d = self.s.point("income", ["age"], i)
        self.assertGreaterEqual(len(d["curve"]), 2)


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(dp_app.app)

    def test_endpoints(self):
        meta = self.c.get("/api/meta").json()
        self.assertNotIn("marital-status", meta["released"]["married"])
        self.assertEqual(self.c.get("/api/point", params={"i": 3, "hidden": "age,sex"}).status_code, 200)

    def test_bad_input(self):
        self.assertEqual(self.c.get("/api/point", params={"i": 99}).status_code, 422)
        self.assertEqual(
            self.c.get("/api/point", params={"i": 1, "hidden": "marital-status"}).status_code, 422
        )
        self.assertEqual(self.c.get("/api/point", params={"i": 1, "target": "female"}).status_code, 422)

    def test_malformed_lists_are_rejected(self):
        with self.c.websocket_connect("/ws") as ws:
            ws.send_json({"op": "point", "seq": 1, "args": {"i": 1, "hidden": [{"a": 1}]}})
            self.assertIn("hidden must be", ws.receive_json()["error"])
            ws.send_json({"op": "point", "seq": 2, "args": {"i": 2.5}})
            self.assertIn("integer", ws.receive_json()["error"])

    def test_websocket(self):
        with self.c.websocket_connect("/ws") as ws:
            ws.send_json(
                {"op": "point", "seq": 1, "args": {"target": "income", "hidden": ["age"], "i": 4, "k": "x"}}
            )
            self.assertEqual(ws.receive_json()["data"]["hidden"], ["age"])

    def test_frontend_served(self):
        self.assertIn("DP vs Profiling", self.c.get("/").text)
        self.assertEqual(self.c.get("/static/dp.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
