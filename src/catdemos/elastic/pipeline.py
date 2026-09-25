"""Elastic Privacy engine (after "Self-Adaptive Governance for Elastic Privacy in Mental
Health Digital Phenotyping", 2026).

1. Behaviour: 1 Hz location traces. Routine walks follow real Aveiro streets (OSM); crises
   are injected as pacing, freezing or erratic wandering, as in the paper's synthetic setup.
2. Crisis detection: an autoencoder (sklearn MLP) fitted on routine 20 s windows of
   movement features; the anomaly score is the reconstruction error relative to routine.
3. Governance agent: a tiny neural policy (log score and its ~20 s and ~60 s moving averages ->
   membrane radius) optimised by neuro-evolution (cross-entropy method, numpy).
4. Privacy membrane: planar-Laplace geo-indistinguishability with epsilon = 2 / radius, so
   the expected displacement of each released point equals the radius.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from catdemos.common.datasets import cache_dir
from catdemos.elastic.osm import Graph, load_graph, to_latlng, to_xy

VERSION = "elastic-v4"
SEED = 0
WIN = 20  # seconds per feature window
WALK_SPEED = 1.35  # m/s, used to turn drawing time into walking time
R_MIN, R_MAX = 5.0, 200.0  # membrane radius bounds (m)
MAX_SECONDS = 3600  # one hour of walking; longer drawings keep their most recent hour
PAPER = {"shield_m": 117.4, "crisis_error_m": 9.9}
FEATURES = [
    "mean speed",
    "speed std",
    "mean |accel|",
    "turning entropy",
    "gyration radius",
    "straightness",
    "dwell share",
]


# ------------------------------------------------------------------ simulation
GPS_SIGMA = 2.0  # metres, per axis


def walk(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a clean 1 Hz walk along a polyline (metres), with speed noise and short stops."""
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    v0 = rng.normal(WALK_SPEED, 0.12)
    speeds, s = [], 0.0
    while s < cum[-1]:
        v = max(0.3, v0 + rng.normal(0, 0.15))
        if rng.random() < 0.004:  # short routine stop (crossing, shop window)
            speeds += [0.0] * int(rng.integers(5, 25))
        speeds.append(v)
        s += v
    d = np.minimum(np.r_[0.0, np.cumsum(speeds)], cum[-1])
    return np.column_stack([np.interp(d, cum, route[:, 0]), np.interp(d, cum, route[:, 1])])


def gps(xy: np.ndarray, seed: int = SEED, rho: float = 0.95) -> np.ndarray:
    """Phone-like GPS error: AR(1) in time (fused location drifts, it does not jump), so the
    per-second jitter is ~0.6 m while the absolute error is GPS_SIGMA. A fixed seed keeps a
    redrawn path's noise stable."""
    n = np.random.default_rng(seed).normal(0, GPS_SIGMA * np.sqrt(1 - rho**2), xy.shape)
    n[0] /= np.sqrt(1 - rho**2)  # start in the stationary distribution
    return xy + lfilter([1.0], [1.0, -rho], n, axis=0)  # e[t] = rho * e[t-1] + n[t]


def inject_crisis(xy: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Replace a 60-150 s stretch by a crisis pattern; returns (trace, labels)."""
    n = len(xy)
    length = int(min(rng.integers(60, 150), n // 2))
    a = int(rng.integers(WIN, max(WIN + 1, n - length)))
    c = xy[a].copy()
    kind = rng.integers(3)
    t = np.arange(length)
    if kind == 0:  # pacing back and forth
        ang = rng.uniform(0, np.pi)
        amp = rng.uniform(6, 15)
        u = amp * np.sin(2 * np.pi * t / rng.uniform(8, 16))
        pat = c + np.column_stack([u * np.cos(ang), u * np.sin(ang)])
    elif kind == 1:  # freezing
        pat = c + rng.normal(0, 0.8, (length, 2))
    else:  # erratic wandering
        head = np.cumsum(rng.normal(0, 1.4, length))
        step = rng.uniform(0.4, 1.8, length)
        pat = c + np.cumsum(np.column_stack([step * np.cos(head), step * np.sin(head)]), axis=0)
    out = xy.copy()
    out[a : a + length] = pat
    out[a + length :] += pat[-1] - xy[a + length - 1] if a + length < n else 0.0
    lab = np.zeros(n, dtype=np.int8)
    lab[a : a + length] = 1
    return out, lab


# -------------------------------------------------------------------- features
def features(xy: np.ndarray) -> np.ndarray:
    """(n, 2) 1 Hz trace -> (n, 7) window features; early seconds reuse the first window."""
    n = len(xy)
    if n < WIN + 1:
        xy = np.vstack([np.repeat(xy[:1], WIN + 1 - n, axis=0), xy])
    step = np.diff(xy, axis=0)
    sp = np.linalg.norm(step, axis=1)
    head = np.arctan2(step[:, 1], step[:, 0])
    turn = np.abs(np.angle(np.exp(1j * np.diff(head))))
    turn = np.where((sp[1:] > 0.3) & (sp[:-1] > 0.3), turn, 0.0)  # ignore GPS jitter when still

    Sw = sliding_window_view(sp, WIN)  # windows of speeds ending at each second
    Tw = sliding_window_view(np.r_[0.0, turn], WIN)
    Pw = sliding_window_view(xy[1:], WIN, axis=0)  # (k, 2, WIN)
    acc = np.abs(np.diff(Sw, axis=1)).mean(1)
    edges = np.linspace(0, np.pi, 5)
    edges[-1] += 1e-9
    bins = np.stack([(Tw >= lo) & (Tw < hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True)], -1)
    p = bins.sum(1) / WIN
    ent = -(np.where(p > 0, p * np.log(p + 1e-12), 0)).sum(1)
    cen = Pw.mean(2, keepdims=True)
    gyr = np.sqrt(((Pw - cen) ** 2).sum(1).mean(1))
    net = np.linalg.norm(Pw[:, :, -1] - Pw[:, :, 0], axis=1)
    straight = net / (Sw.sum(1) + 1e-6)
    dwell = (Sw < 0.3).mean(1)
    F = np.column_stack([Sw.mean(1), Sw.std(1), acc, ent, np.log1p(gyr), straight, dwell])
    F = np.vstack([np.repeat(F[:1], WIN, axis=0), F])  # align: row t describes window ending at t
    return F[-n:]


# ---------------------------------------------------------------------- policy
N_IN, N_HID = 3, 8
N_THETA = N_IN * N_HID + N_HID + N_HID + 1


def policy(theta: np.ndarray, s: np.ndarray) -> np.ndarray:
    """3 -> 8 (tanh) -> 1 (sigmoid) network mapping the policy state (see ``Engine.state``) to a radius."""
    a, b = N_IN * N_HID, N_IN * N_HID + N_HID
    W1, b1, W2, b2 = theta[:a].reshape(N_IN, N_HID), theta[a:b], theta[b : b + N_HID], theta[-1]
    h = np.tanh(s @ W1 + b1)
    return R_MIN + (R_MAX - R_MIN) / (1 + np.exp(-(h @ W2 + b2)))


def smooth(x: np.ndarray, tau: float = 20.0) -> np.ndarray:
    """Exponential moving average with time constant ``tau`` seconds, starting at ``x[0]``."""
    a = 1.0 / tau
    y, _ = lfilter([a], [1.0, a - 1.0], x, zi=[(1.0 - a) * x[0]])
    return y


def reward(theta: np.ndarray, s: np.ndarray, lab: np.ndarray) -> float:
    """Wide membrane in routine, tight in crisis, no flicker (all in units of R_MAX)."""
    r = policy(theta, s) / R_MAX
    return float(r[lab == 0].mean() - 3.0 * r[lab == 1].mean() - 1.0 * np.abs(np.diff(r)).mean())


def evolve(
    s: np.ndarray, lab: np.ndarray, rng: np.random.Generator, iters: int = 50, pop: int = 64, elite: int = 8
) -> np.ndarray:
    """Neuro-evolution by the cross-entropy method: sample weight vectors from a Gaussian,
    keep the elite, refit the Gaussian. (A gradient-style ES collapsed here: once every
    output saturates at R_MIN the rank signal vanishes.)"""
    mu, sd = np.zeros(N_THETA), np.ones(N_THETA)
    for _ in range(iters):
        pop_w = mu + sd * rng.normal(size=(pop, N_THETA))
        f = np.array([reward(w, s, lab) for w in pop_w])
        best = pop_w[np.argsort(f)[-elite:]]
        mu, sd = best.mean(0), best.std(0) + 0.02
    return mu


# ---------------------------------------------------------------------- engine
@dataclass
class Engine:
    """Trained detector, governance policy and benchmark behind the endpoints."""

    scaler: StandardScaler
    ae: MLPRegressor
    err_ref: float  # routine reconstruction error at the 95th percentile
    theta: np.ndarray
    unit: np.ndarray  # (MAX_SECONDS, 2) fixed planar-Laplace unit noise (stable across redraws)
    graph: Graph
    bench: dict

    def score(self, xy: np.ndarray, F: np.ndarray | None = None) -> np.ndarray:
        """Crisis score per second: autoencoder reconstruction error / routine 95th percentile."""
        Z = self.scaler.transform(features(xy) if F is None else F)
        err = ((np.asarray(self.ae.predict(Z)) - Z) ** 2).mean(1)
        return err / self.err_ref

    def state(self, score: np.ndarray) -> np.ndarray:
        """Policy input per second: log score and its moving averages over ~20 s and ~60 s.

        The slow average lets the agent tell a routine wait at a crossing (seconds) from a
        crisis (minutes), which a 20 s view cannot.
        """
        ls = np.log(score + 1e-6)
        return np.column_stack([ls, smooth(ls, 20.0), smooth(ls, 60.0)])

    def run(self, xy: np.ndarray, F: np.ndarray | None = None) -> dict:
        """Detector -> governance policy -> planar-Laplace release for a 1 Hz trace (metres)."""
        score = self.score(xy, F)
        radius = policy(self.theta, self.state(score))
        disp = self.unit[: len(xy)] * (radius / 2.0)[:, None]  # Gamma(2, r/2) radial -> mean r
        return {"score": score, "radius": radius, "obf": xy + disp, "disp": np.linalg.norm(disp, axis=1)}

    def walk_latlng(self, pts: np.ndarray) -> dict:
        """pts: (n, 3) [lat, lng, t_ms] as drawn -> full pipeline output for the viewer."""
        clean = resample(to_xy(pts[:, :2]), pts[:, 2] / 1000.0)
        xy = gps(clean)
        F = features(xy)  # computed once: the detector and the "features now" panel share it
        out = self.run(xy, F)
        alert = out["radius"] < (R_MIN + R_MAX) / 2
        disp = out["disp"]
        return {
            "raw": np.round(to_latlng(clean), 6).tolist(),
            "obf": np.round(to_latlng(out["obf"]), 6).tolist(),
            "score": np.round(out["score"], 3).tolist(),
            "radius": np.round(out["radius"], 1).tolist(),
            "alert": alert.astype(int).tolist(),
            "seconds": len(xy),
            "summary": {
                "shield_m": round(float(disp[~alert].mean()), 1) if (~alert).any() else None,
                "alert_error_m": round(float(disp[alert].mean()), 1) if alert.any() else None,
                "alert_seconds": int(alert.sum()),
                "features_now": dict(zip(FEATURES, np.round(F[-1], 3).tolist(), strict=True)),
            },
        }

    def sample(self, seed: int, crisis: bool = True) -> list[list[float]]:
        """A clean simulated walk (~12 min), with or without one crisis, as [lat, lng, t_ms] points."""
        rng = np.random.default_rng(seed)
        routes = self.graph.random_routes(12, rng, min_len=500)
        lengths = [np.linalg.norm(np.diff(r, axis=0), axis=1).sum() for r in routes]
        xy = walk(routes[int(np.argmin(np.abs(np.array(lengths) - 900)))], rng)  # ~12 min walk
        if crisis:
            xy, _ = inject_crisis(xy, rng)
        ll = to_latlng(xy)
        return [[round(a, 6), round(b, 6), i * 1000] for i, (a, b) in enumerate(ll)]


def resample(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map drawing time to walking time (median moving speed -> WALK_SPEED) and sample at 1 Hz.

    Time is never compressed: that would speed the whole walk up and make it look anomalous.
    Beyond MAX_SECONDS only the most recent stretch is kept (the viewer is a live view)."""
    if len(xy) < 2:
        return np.repeat(xy[:1], 2, axis=0)
    t = t - t[0]
    dt, dd = np.diff(t), np.linalg.norm(np.diff(xy, axis=0), axis=1)
    moving = (dt > 0) & (dd > 0)
    if moving.any():
        t = t * (np.median(dd[moving] / dt[moving]) / WALK_SPEED)
    t = np.maximum.accumulate(t)
    ts = np.arange(max(0, int(t[-1]) + 1 - MAX_SECONDS), int(t[-1]) + 1, dtype=float)
    return np.column_stack([np.interp(ts, t, xy[:, 0]), np.interp(ts, t, xy[:, 1])])


def train() -> Engine:
    """Simulate walks, fit the crisis autoencoder, evolve the policy and benchmark it."""
    rng = np.random.default_rng(SEED)
    graph = load_graph()
    routes = graph.random_routes(240, rng)
    routine = [walk(r, rng) for r in routes]  # clean; GPS noise is added after crisis injection
    train_r, test_r = routine[:160], routine[160:]

    F = np.vstack([features(gps(w, i)) for i, w in enumerate(train_r[:120])])
    scaler = StandardScaler().fit(F)
    Z = scaler.transform(F)
    Z = Z[rng.choice(len(Z), min(len(Z), 40_000), replace=False)]
    ae = MLPRegressor(hidden_layer_sizes=(32, 6, 32), activation="tanh", max_iter=200, random_state=SEED)
    ae.set_params(batch_size=256)  # sklearn's signature types batch_size as str ("auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ae.fit(Z, Z)
    err = ((np.asarray(ae.predict(Z)) - Z) ** 2).mean(1)
    ang = rng.uniform(0, 2 * np.pi, MAX_SECONDS)
    unit = rng.gamma(2.0, 1.0, MAX_SECONDS)[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])
    eng = Engine(scaler, ae, float(np.quantile(err, 0.95)), np.zeros(N_THETA), unit, graph, {})

    def episodes(walks: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Inject a crisis into each walk and score it: (policy state, labels, trace)."""
        eps = []
        for w in walks:
            xy, lab = inject_crisis(w, rng)
            xy = gps(xy, int(rng.integers(1 << 30)))
            eps.append((eng.state(eng.score(xy)), lab, xy))
        return eps

    tr = episodes(train_r[120:])
    eng.theta = evolve(np.vstack([s for s, _, _ in tr]), np.concatenate([lab for _, lab, _ in tr]), rng)

    te = episodes(test_r)
    rs, rc, hit = [], [], []
    for s, lab, _ in te:
        r = policy(eng.theta, s)
        rs.append(r[lab == 0])
        rc.append(r[lab == 1])
        hit.append((r[lab == 1] < (R_MIN + R_MAX) / 2).mean())
    eng.bench = {
        "shield_m": round(float(np.concatenate(rs).mean()), 1),
        "crisis_error_m": round(float(np.concatenate(rc).mean()), 1),
        "crisis_detected": round(float(np.mean(hit)), 3),
        "false_alert": round(float((np.concatenate(rs) < (R_MIN + R_MAX) / 2).mean()), 3),
        "test_walks": len(te),
        "paper": PAPER,
    }
    return eng


def load_or_train() -> Engine:
    """Load the cached engine, or train it once and cache it (``data/cache``).

    An engine trained on the offline grid fallback is not cached, so the next start retries OSM."""
    path = cache_dir() / f"{VERSION}.joblib"
    if path.exists():
        return joblib.load(path)
    eng = train()
    if eng.graph.source == "osm":
        joblib.dump(eng, path)
    return eng
