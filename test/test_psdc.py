"""PsDC categorizer engine and API."""

import unittest

import numpy as np
from fastapi.testclient import TestClient

from catdemos.common.adult import NAMES, randomised_response
from catdemos.psdc import app as psdc_app
from catdemos.psdc.categorize import CATEGORIES, EXPERT


class CensusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = psdc_app.categorizer.census

    def test_randomised_response_keep_rate(self):
        out = randomised_response(np.zeros(20000, dtype=int), 4, 1.0, np.random.default_rng(0))
        self.assertAlmostEqual((out == 0).mean(), np.exp(1.0) / (np.exp(1.0) + 3), delta=0.01)

    def test_target_source_is_never_used_to_infer_it(self):
        self.assertNotIn("marital-status", self.c.usable("married", ["age", "marital-status"]))

    def test_relationship_reveals_marriage(self):
        self.assertGreater(self.c.accuracy("married", ("relationship",)), 0.95)

    def test_coarse_granularity_lowers_uniqueness(self):
        fields = ("age", "hours-per-week", "occupation", "education")
        self.assertLess(self.c.uniqueness(fields, coarse=True), self.c.uniqueness(fields))

    def test_top_values(self):
        self.assertEqual(self.c.top_values("sex", 2), ["Male", "Female"])


class CategorizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = psdc_app.categorizer

    def test_automatic_categories_match_the_expert(self):
        self.assertEqual({n: self.k.auto[n]["category"] for n in NAMES}, EXPERT)
        self.assertTrue(all(0.34 <= self.k.auto[n]["confidence"] <= 1 for n in NAMES))

    def test_model(self):
        m = self.k.model(["age", "sex", "relationship"], "daily", "1 year", "exact")
        self.assertEqual([f["name"] for f in m["fields"]], ["age", "sex", "relationship"])
        tags = {t["target"]: t for t in m["tags"]}
        self.assertTrue(tags["female"]["collected"])
        self.assertGreater(tags["married"]["accuracy"], tags["married"]["baseline"])
        self.assertTrue(0 < m["score"] <= 1)

    def test_structure_scales_the_score(self):
        low = self.k.model(["age", "income"], "once", "1 day", "exact")["score"]
        high = self.k.model(["age", "income"], "continuous", "forever", "exact")["score"]
        self.assertLess(low, high)

    def test_empty_model(self):
        m = self.k.model([], "daily", "1 year", "exact")
        self.assertEqual((m["score"], m["unique"]), (0.0, 0.0))


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(psdc_app.app)

    def test_meta_and_rest(self):
        meta = self.c.get("/api/meta").json()
        self.assertEqual(meta["categories"], CATEGORIES)
        r = self.c.get("/api/model", params={"fields": "age,sex", "granularity": "coarse"})
        self.assertEqual(r.status_code, 200)

    def test_bad_input(self):
        self.assertEqual(self.c.get("/api/model", params={"fields": "ssn"}).status_code, 422)
        self.assertEqual(self.c.get("/api/model", params={"frequency": "hourly"}).status_code, 422)

    def test_websocket(self):
        with self.c.websocket_connect("/ws") as ws:
            ws.send_json(
                {"op": "model", "seq": 1, "args": {"fields": ["age", "occupation"], "retention": "forever"}}
            )
            self.assertEqual(len(ws.receive_json()["data"]["fields"]), 2)
            ws.send_json({"op": "model", "seq": 2, "args": {"fields": "not-a-list-item"}})
            self.assertIn("error", ws.receive_json())

    def test_frontend_served(self):
        self.assertIn("PsDC Categorizer", self.c.get("/").text)
        self.assertEqual(self.c.get("/static/psdc.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
