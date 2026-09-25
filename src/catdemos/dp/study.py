"""DP vs Profiling engine (after "Evaluating the Effectiveness of Differential Privacy Against
Profiling", ICCT-Europe 2025).

A census release publishes every field except the sensitive one being studied. The viewer
chooses which fields to *hide with differential privacy* (local DP per field: Laplace noise for
numbers, randomised response for categories, epsilon per field); the others are released
exactly. Two attacks are measured on the same release:

- re-identification: an adversary who knows the true values of 800 people links each of them
  to the nearest released record;
- profiling: a model trained on the released data infers the sensitive fact of unseen people.

The paper's finding, reproduced live: DP makes re-identification collapse while profiling
keeps most of its accuracy.
"""

from __future__ import annotations

import numpy as np

from catdemos.common.adult import N_LINK, NAMES, TARGETS, Census

EPS_GRID = np.round(np.logspace(-2, 1, 21), 4)  # 0.01 .. 10, per hidden field
STUDY_TARGETS = ("married", "income")


class Study:
    """Profiling vs re-identification of a census release with some fields under DP."""

    def __init__(self, census: Census | None = None) -> None:
        """Share (or build) the encoded census sample."""
        self.census = census or Census()

    def released(self, target: str) -> tuple[str, ...]:
        """Every field except the sensitive one studied."""
        return self.census.usable(target, NAMES)

    def point(self, target: str, hidden: list[str], i: int) -> dict:
        """Both attacks at EPS_GRID[i], every point of this setting measured so far, and references."""
        c = self.census
        fields = self.released(target)
        noisy = frozenset(hidden) & frozenset(fields)
        eps = float(EPS_GRID[i])
        prof = c.accuracy(target, fields, eps=eps, noisy=noisy)
        reid = c.reidentification(fields, eps=eps, noisy=noisy)
        curve = []
        for e in EPS_GRID:
            e = float(e)
            k = frozenset(noisy)
            pa = c.cached(("acc", target, fields, False, e if k else None, k))
            pr = c.cached(("reid", fields, e if k else None, k))
            if pa is not None and pr is not None:
                curve.append({"eps": e, "profiling": pa, "reid": pr})
        base = c.baseline(target)
        exact = {"profiling": c.accuracy(target, fields), "reid": c.reidentification(fields)}
        return {
            "eps": eps,
            "hidden": sorted(noisy),
            "profiling": round(prof, 4),
            "reid": round(reid, 4),
            "curve": curve,
            "no_dp": {k: round(v, 4) for k, v in exact.items()},
            "baseline": round(base, 4),
            "chance_reid": 1.0 / N_LINK,
            "kept": {
                "profiling": round(_kept(prof, exact["profiling"], base), 3),
                "reid": round(_kept(reid, exact["reid"], 1.0 / N_LINK), 3),
            },
        }

    def meta(self) -> dict:
        """Released fields per target, epsilon grid and sample sizes."""
        return {
            "targets": {t: TARGETS[t][0] for t in STUDY_TARGETS},
            "released": {t: list(self.released(t)) for t in STUDY_TARGETS},
            "eps_grid": EPS_GRID.tolist(),
            "n_link": N_LINK,
        }


def _kept(v: float, v0: float, floor: float) -> float:
    """Share of the no-DP advantage over ``floor`` that survives (0 when there was none)."""
    return min(1.0, max(0.0, (v - floor) / (v0 - floor))) if v0 - floor > 1e-3 else 0.0
