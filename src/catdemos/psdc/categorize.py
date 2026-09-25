"""PsDC categorization engine (after "Proactive Data Categorization for Privacy in DevPrivOps",
Information 2025, and "Semantic and Numerical Feature Clustering for Automated Privacy
Quantification", FiCloud 2025).

PsDC describes the data a system collects on three levels:

1. Direct data categorization: each field gets a category, found automatically from the data
   (paper 22). Semantic features: character n-gram TF-IDF of the field's name and its most
   frequent real values, compared with a short definition of each category; the closest
   definition is the category and the similarities give the confidence. Numerical features
   computed on the values (mutual information with sensitive facts) set how much each field
   weighs in the score. The expert categories are only used to report agreement.
2. Dynamic tags: what the *combination* of collected fields reveals. Each tag is measured, not
   declared: a profiler is retrained on the collected fields to infer every sensitive fact
   that was not collected, and the share of people made unique by the collected fields is
   counted.
3. Structural attributes: how the data is handled (collection frequency, retention,
   granularity). Granularity changes the data itself (coarse = binned values), so it changes
   the measured tags; frequency and retention scale the final score.

The privacy score aggregation (weights below) is this demo's simplification of PsDC's
"privacy value" of data.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import normalize

from catdemos.common.adult import FIELDS, NAMES, TARGETS, Census

CATEGORIES = ["identity", "socio-economic", "sensitive"]
# Short definitions of the categories: the only "knowledge" the categorizer is given.
DEFINITIONS = {
    "identity": "demographic identity of a person: age, birth, sex, gender, male, female, race, ethnicity, "
    "white, black, asian, nationality, country of origin",
    "socio-economic": "work and education: job, occupation, employment, employer, private, government, "
    "self employed, working hours per week, school, degree, college, bachelors, masters",
    "sensitive": "private life: family, marital status, married, divorced, spouse, husband, wife, child, "
    "household relationship, money, income, capital, gain, loss, finances, investments, health, beliefs",
}
# Expert categories: used only to report how well the automatic categorization agrees.
EXPERT = {
    "age": "identity",
    "sex": "identity",
    "race": "identity",
    "native-country": "identity",
    "education": "socio-economic",
    "workclass": "socio-economic",
    "occupation": "socio-economic",
    "hours-per-week": "socio-economic",
    "marital-status": "sensitive",
    "relationship": "sensitive",
    "capital-gain": "sensitive",
    "capital-loss": "sensitive",
    "income": "sensitive",
}
CATEGORY_WEIGHT = {"identity": 0.6, "socio-economic": 0.35, "sensitive": 1.0}
FREQUENCY = {"once": 0.55, "monthly": 0.7, "daily": 0.85, "continuous": 1.0}
RETENTION = {"1 day": 0.55, "1 month": 0.7, "1 year": 0.85, "forever": 1.0}


class Categorizer:
    """Automatic PsDC categorization of the census fields, plus live tags and score."""

    def __init__(self, census: Census | None = None) -> None:
        """Categorize every field once; tags and scores are measured on demand (and cached)."""
        self.census = census or Census()
        sim = self._semantic()
        conf = np.exp(sim / 0.05)
        conf /= conf.sum(1, keepdims=True)
        self.auto = {
            n: {"category": CATEGORIES[int(sim[i].argmax())], "confidence": round(float(conf[i].max()), 2)}
            for i, n in enumerate(NAMES)
        }
        self.agreement = float(np.mean([self.auto[n]["category"] == EXPERT[n] for n in NAMES]))
        self.weight = self._numerical_weight()

    def _semantic(self) -> np.ndarray:
        """(fields, categories) cosine similarity of name + frequent values to category definitions."""
        docs = []
        for name, kind, _ in FIELDS:
            words = [name.replace("-", " ")]
            if kind == "cat":
                words += [v.replace("-", " ") for v in self.census.top_values(name, 8)]
            docs.append(" ".join(words))
        defs = [DEFINITIONS[c] for c in CATEGORIES]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True).fit(docs + defs)
        F = np.asarray(normalize(csr_matrix(vec.transform(docs)).toarray()))
        D = np.asarray(normalize(csr_matrix(vec.transform(defs)).toarray()))
        return F @ D.T

    def _numerical_weight(self) -> dict[str, float]:
        """Per-field weight in [0.5, 1]: how much sensitive information its values carry (max MI)."""
        c = self.census
        mi = np.array([max(mutual_info_score(c.fields[n].coarse, c.y[t]) for t in TARGETS) for n in NAMES])
        mi = mi / mi.max()
        return {n: round(0.5 + 0.5 * float(m), 3) for n, m in zip(NAMES, mi, strict=True)}

    def model(self, fields: list[str], frequency: str, retention: str, granularity: str) -> dict:
        """Categorize the collected fields, measure their dynamic tags and score the data model."""
        fields = [f for f in NAMES if f in set(fields)]  # canonical order, no duplicates
        coarse = granularity == "coarse"
        c = self.census
        tags = []
        for t, (label, _, src) in TARGETS.items():
            if src in fields:
                tags.append({"target": t, "label": label, "collected": True})
                continue
            acc, base = c.accuracy(t, tuple(fields), coarse=coarse), c.baseline(t)
            tags.append(
                {
                    "target": t,
                    "label": label,
                    "collected": False,
                    "accuracy": round(acc, 3),
                    "baseline": round(base, 3),
                    "strength": round(max(0.0, (acc - base) / (1 - base)), 3),
                }
            )
        unique = c.uniqueness(tuple(fields), coarse)

        w = {n: CATEGORY_WEIGHT[self.auto[n]["category"]] * self.weight[n] for n in NAMES}
        direct = sum(w[f] for f in fields) / sum(w.values())
        inferred = [t["strength"] for t in tags if not t["collected"]]
        dynamic = max([unique, *inferred]) if fields else 0.0
        structural = (FREQUENCY[frequency] + RETENTION[retention]) / 2
        score = structural * (0.5 * direct + 0.5 * dynamic)
        return {
            "fields": [
                {"name": f, **self.auto[f], "expert": EXPERT[f], "weight": self.weight[f]} for f in fields
            ],
            "tags": tags,
            "unique": round(unique, 3),
            "score": round(score, 3),
            "parts": {
                "direct": round(direct, 3),
                "dynamic": round(dynamic, 3),
                "structural": round(structural, 3),
            },
        }

    def meta(self) -> dict:
        """Fields with their automatic and expert categories, and the structural options."""
        return {
            "fields": [
                {
                    "name": n,
                    "kind": k,
                    "desc": d,
                    **self.auto[n],
                    "expert": EXPERT[n],
                    "weight": self.weight[n],
                    "values": self.census.top_values(n, 4),
                }
                for n, k, d in FIELDS
            ],
            "categories": CATEGORIES,
            "agreement": round(self.agreement, 3),
            "frequency": list(FREQUENCY),
            "retention": list(RETENTION),
            "granularity": ["exact", "coarse"],
        }
