"""UCI Adult census core shared by the PsDC and the DP-vs-Profiling demos.

A fixed-seed sample of 9,000 complete records (6,000 to train, 3,000 to test). Every field is
encoded once; the measurements below retrain a profiler (logistic regression) on any subset
of fields, optionally coarsened (lower granularity) or released under local differential
privacy, and a nearest-neighbour linkage attack measures re-identification.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from catdemos.common.datasets import load_adult

SEED = 0
N_TRAIN, N_TEST, N_LINK = 6000, 3000, 800

# (name, kind, description). The description feeds PsDC's semantic features.
FIELDS: list[tuple[str, str, str]] = [
    ("age", "num", "age of the person in years, date of birth, demographic identity"),
    ("sex", "cat", "sex gender of the person, demographic identity"),
    ("race", "cat", "race ethnicity of the person, demographic identity"),
    ("native-country", "cat", "country of birth nationality origin of the person, demographic identity"),
    ("education", "cat", "highest education degree level attained, school"),
    ("workclass", "cat", "employment sector type of employer, work"),
    ("occupation", "cat", "job occupation role at work"),
    ("hours-per-week", "num", "hours worked per week at work"),
    ("marital-status", "cat", "marital status married divorced widowed single, family life"),
    ("relationship", "cat", "role in the household husband wife child, family life"),
    ("capital-gain", "num", "capital gains from investments, money finances"),
    ("capital-loss", "num", "capital losses from investments, money finances"),
    ("income", "cat", "yearly income bracket above or below 50K dollars, money finances"),
]
NAMES = [f[0] for f in FIELDS]

# Sensitive facts an adversary may try to infer: label, test on a raw record, source field.
TARGETS: dict[str, tuple[str, Callable[[dict[str, str]], bool], str]] = {
    "income": ("income > 50K", lambda r: r["income"] == ">50K", "income"),
    "married": ("married", lambda r: r["marital-status"].startswith("Married"), "marital-status"),
    "female": ("female", lambda r: r["sex"] == "Female", "sex"),
}


@dataclass
class Field:
    """One field encoded for all rows of the sample."""

    kind: str
    codes: np.ndarray  # categorical: int codes; numeric: float values (clipped)
    k: int  # number of categories (categorical)
    lo: float
    hi: float
    mu: float
    sd: float
    coarse: np.ndarray  # int codes at low granularity (5 quantile bins, or top-4 categories + other)
    k_coarse: int

    def encode(self, v: np.ndarray) -> np.ndarray:
        """One-hot (categorical) or standardised (numeric) columns for values ``v``."""
        if self.kind == "num":
            return ((v - self.mu) / self.sd)[:, None].astype(np.float32)
        return one_hot(v, self.k)

    def privatise(self, v: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
        """Local-DP release of ``v``: Laplace (numeric, clipped) or k-ary randomised response."""
        if self.kind == "num":
            noisy = v + rng.laplace(0.0, (self.hi - self.lo) / eps, v.size)
            return np.clip(noisy, self.lo, self.hi)
        return randomised_response(v, self.k, eps, rng)


def one_hot(v: np.ndarray, k: int) -> np.ndarray:
    """(n, k) float32 one-hot matrix of integer codes ``v``."""
    out = np.zeros((v.size, k), dtype=np.float32)
    out[np.arange(v.size), v.astype(int)] = 1.0
    return out


def randomised_response(v: np.ndarray, k: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """k-ary randomised response: keep with prob e^eps / (e^eps + k - 1), else any other value."""
    keep = np.exp(eps) / (np.exp(eps) + k - 1)
    flip = rng.random(v.size) >= keep
    other = (v + rng.integers(1, k, v.size)) % k
    return np.where(flip, other, v)


def _coarsen(kind: str, codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Low-granularity codes: numeric -> 5 quantile bins; categorical -> top 4 values + other."""
    if kind == "num":
        edges = np.unique(np.quantile(codes, [0.2, 0.4, 0.6, 0.8]))
        c = np.digitize(codes, edges)
        return c, int(c.max()) + 1
    top = np.argsort(-np.bincount(codes))[:4]
    remap = np.full(int(codes.max()) + 1, 4)
    remap[top] = np.arange(len(top))
    c = remap[codes]
    return c, int(c.max()) + 1


class Census:
    """Encoded census sample plus cached profiler and linkage measurements."""

    def __init__(self) -> None:
        """Load and encode a fixed-seed sample of 9,000 complete Adult records."""
        rows = load_adult()
        rng = np.random.default_rng(SEED)
        rows = [rows[i] for i in rng.choice(len(rows), N_TRAIN + N_TEST, replace=False)]
        self.fields: dict[str, Field] = {}
        self._values: dict[str, list[str]] = {}
        for name, kind, _ in FIELDS:
            col = [r[name] for r in rows]
            vals, cnt = np.unique(np.asarray(col), return_counts=True)
            self._values[name] = [str(v) for v in vals[np.argsort(-cnt)]]
            if kind == "num":
                v = np.asarray(col, dtype=np.float64)
                lo, hi = np.quantile(v, [0.01, 0.99])
                if hi <= lo:  # capital-gain/loss are mostly 0: use the max instead of p99
                    hi = float(v.max())
                v = np.clip(v, lo, hi)
                c, kc = _coarsen("num", v)
                self.fields[name] = Field(
                    "num", v, 0, float(lo), float(hi), float(v.mean()), float(v.std() + 1e-9), c, kc
                )
            else:
                cats, codes = np.unique(np.asarray(col), return_inverse=True)
                c, kc = _coarsen("cat", codes)
                self.fields[name] = Field("cat", codes, len(cats), 0, 0, 0, 1, c, kc)
        self.y = {t: np.array([f(r) for r in rows], dtype=np.int8) for t, (_, f, _) in TARGETS.items()}
        self.tr, self.te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
        self._cache: OrderedDict[tuple, float] = OrderedDict()

    # ---------------------------------------------------------------- helpers
    def top_values(self, name: str, n: int) -> list[str]:
        """The ``n`` most frequent raw values of a field (numbers as text)."""
        return self._values[name][:n]

    def baseline(self, target: str) -> float:
        """Majority-class accuracy on the test people: the no-information guess."""
        y = self.y[target][self.te]
        return float(max(y.mean(), 1 - y.mean()))

    def usable(self, target: str, fields: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Sorted fields without the target's own source field (that would not be inference)."""
        src = TARGETS[target][2]
        return tuple(sorted(f for f in set(fields) if f != src))

    def matrix(
        self,
        fields: tuple[str, ...],
        coarse: bool = False,
        eps: float | None = None,
        noisy: frozenset[str] = frozenset(),
        seed: int = SEED,
    ) -> np.ndarray:
        """Design matrix; fields in ``noisy`` are released under DP with ``eps`` each."""
        rng = np.random.default_rng(seed)
        cols = []
        for name in fields:
            f = self.fields[name]
            if coarse:
                cols.append(one_hot(f.coarse, f.k_coarse))
            elif eps is not None and name in noisy:
                cols.append(f.encode(f.privatise(f.codes, eps, rng)))
            else:
                cols.append(f.encode(f.codes))
        return np.hstack(cols)

    def _cached(self, key: tuple, compute) -> float:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        val = float(compute())
        self._cache[key] = val
        if len(self._cache) > 20000:
            self._cache.popitem(last=False)
        return val

    def cached(self, key: tuple) -> float | None:
        """A measurement if it was already computed, without computing it."""
        return self._cache.get(key)

    # ----------------------------------------------------------- measurements
    def accuracy(
        self,
        target: str,
        fields: tuple[str, ...],
        coarse: bool = False,
        eps: float | None = None,
        noisy: frozenset[str] = frozenset(),
    ) -> float:
        """Test accuracy of a profiler trained and tested on the (possibly released) fields."""
        fields = self.usable(target, fields)
        noisy = frozenset(noisy) & frozenset(fields) if eps is not None else frozenset()
        key = ("acc", target, fields, coarse, eps if noisy else None, noisy)

        def fit() -> float:
            if not fields:
                return self.baseline(target)
            X = self.matrix(fields, coarse, eps, noisy, seed=SEED + 1)
            y = self.y[target]
            # newton-cholesky: ~20 ms per fit here, ~60x faster than lbfgs at equal accuracy.
            clf = LogisticRegression(solver="newton-cholesky").fit(X[self.tr], y[self.tr])
            return clf.score(X[self.te], y[self.te])

        return self._cached(key, fit)

    def reidentification(
        self, fields: tuple[str, ...], eps: float | None = None, noisy: frozenset[str] = frozenset()
    ) -> float:
        """Share of people whose nearest released record (ties shared) is their own.

        The adversary knows the exact values of ``fields`` for 800 people and links them to the
        released records, where fields in ``noisy`` carry DP noise.
        """
        fields = tuple(sorted(fields))
        noisy = frozenset(noisy) & frozenset(fields) if eps is not None else frozenset()
        key = ("reid", fields, eps if noisy else None, noisy)

        def link() -> float:
            if not fields:
                return 1.0 / N_LINK
            rows = slice(N_TRAIN, N_TRAIN + N_LINK)
            # float64: in float32 the distance of a record to itself is off by up to ~1e-5, above the
            # tie tolerance below.
            aux = self.matrix(fields)[rows].astype(np.float64)
            rel = self.matrix(fields, eps=eps, noisy=noisy, seed=SEED + 2)[rows] if noisy else aux
            rel = rel.astype(np.float64, copy=False)
            d = (aux**2).sum(1)[:, None] - 2 * aux @ rel.T + (rel**2).sum(1)[None, :]
            dmin = d.min(1, keepdims=True)
            ties = (d <= dmin + 1e-6).sum(1)
            self_hit = np.diag(d) <= dmin[:, 0] + 1e-6
            return (self_hit / ties).mean()

        return self._cached(key, link)

    def uniqueness(self, fields: tuple[str, ...], coarse: bool = False) -> float:
        """Share of the 9,000 people who are unique on the combination of ``fields``."""
        fields = tuple(sorted(fields))

        def count() -> float:
            if not fields:
                return 0.0
            cols = [self.fields[f].coarse if coarse else self.fields[f].codes for f in fields]
            _, inv, cnt = np.unique(np.column_stack(cols), axis=0, return_inverse=True, return_counts=True)
            return (cnt[inv.ravel()] == 1).mean()

        return self._cached(("uniq", fields, coarse), count)
