"""ASAP Latent Map engine and API."""

import unittest

import numpy as np
from fastapi.testclient import TestClient
from sklearn.metrics import matthews_corrcoef

from catdemos.asap import app as asap_app
from catdemos.asap.model import LABELS, tier_of


class TierTest(unittest.TestCase):
    def test_psdc_tiers(self):
        self.assertEqual(tier_of("android.permission.READ_SMS"), 1)
        self.assertEqual(tier_of("android.permission.ACCESS_WIFI_STATE"), 2)
        self.assertEqual(tier_of("android.permission.VIBRATE"), 3)
        # Vendor namespaces never count as direct personal data.
        self.assertEqual(tier_of("com.vendor.permission.READ_SMS"), 3)


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = asap_app.engine

    def test_detector_quality(self):
        at = self.eng.at_threshold(self.eng.threshold)
        self.assertGreater(at["mcc"], 0.8)  # supervised ceiling on this dataset is ~0.93

    def test_threshold_matches_sklearn(self):
        t = float(np.median(self.eng.test_score))
        at = self.eng.at_threshold(t)
        self.assertEqual(at["tp"] + at["fp"] + at["fn"] + at["tn"], self.eng.test_y.size)
        ref = matthews_corrcoef(self.eng.test_y, self.eng.test_score >= t)
        self.assertAlmostEqual(at["mcc"], ref, places=3)

    def test_probe_is_consistent(self):
        d = self.eng.probe(0.2, -0.4)
        self.assertIn(d["label"], LABELS)
        self.assertTrue(0.0 <= d["p_malicious"] <= 1.0)
        self.assertEqual(sum(d["tiers"].values()), d["n_perms"])
        self.assertEqual(d["flagged"], d["score"] >= self.eng.threshold)

    def test_label_grows_with_risk(self):
        self.assertEqual(self.eng.label(0.0, 0.0), "A")
        self.assertEqual(self.eng.label(1.0, 1.0), "E")

    def test_overview_grid(self):
        ov = self.eng.overview()
        self.assertEqual(len(ov["grid"]), ov["grid_n"])
        self.assertEqual(sum(ov["hist"]["benign"]) + sum(ov["hist"]["malicious"]) > 0, True)


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(asap_app.app)

    def test_endpoints(self):
        self.assertEqual(self.c.get("/api/overview").status_code, 200)
        self.assertIn("mcc", self.c.get("/api/threshold", params={"t": 0.0}).json())
        self.assertIn("perms", self.c.get("/api/probe", params={"x": 0.1, "y": 0.1}).json())

    def test_probe_outside_plane_is_rejected(self):
        self.assertEqual(self.c.get("/api/probe", params={"x": 3, "y": 0}).status_code, 422)

    def test_websocket_live_ops(self):
        with self.c.websocket_connect("/ws") as ws:
            ws.send_json({"op": "probe", "seq": 1, "args": {"x": 0.3, "y": 0.1}})
            r = ws.receive_json()
            self.assertEqual((r["op"], r["seq"]), ("probe", 1))
            self.assertEqual(r["data"], asap_app.engine.probe(0.3, 0.1))
            ws.send_json({"op": "threshold", "seq": 2, "args": {"t": 0.0}})
            self.assertIn("mcc", ws.receive_json()["data"])
            ws.send_json({"op": "probe", "seq": 3, "args": {"x": 9, "y": 0}})
            self.assertIn("outside", ws.receive_json()["error"])
            ws.send_json({"op": "nope", "seq": 4})
            self.assertIn("unknown operation", ws.receive_json()["error"])

    def test_non_finite_input_is_rejected_not_fatal(self):
        self.assertEqual(self.c.get("/api/probe", params={"x": "nan", "y": 0}).status_code, 422)
        self.assertEqual(self.c.get("/api/threshold", params={"t": "inf"}).status_code, 422)
        with self.c.websocket_connect("/ws") as ws:
            ws.send_json({"op": "probe", "seq": 1, "args": {"x": "nan", "y": 0}})
            self.assertIn("finite", ws.receive_json()["error"])
            ws.send_json({"op": "threshold", "seq": 2, "args": {"t": "nan"}})
            self.assertIn("finite", ws.receive_json()["error"])
            ws.send_json({"op": "threshold", "seq": 3, "args": {"t": 0.0}})
            self.assertIn("mcc", ws.receive_json()["data"])  # the session is still open

    def test_frontend_served(self):
        self.assertIn("ASAP Latent Map", self.c.get("/").text)
        self.assertEqual(self.c.get("/static/asap.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
