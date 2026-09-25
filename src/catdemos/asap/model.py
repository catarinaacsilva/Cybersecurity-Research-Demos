"""ASAP Latent Map engine.

Detector (after ASAP 2.0): unsupervised autoencoders on Android permission vectors, one
fitted on benign and one on malicious training apps. The anomaly score is the log ratio
of their reconstruction errors (high = the app looks more like the malicious population);
the decision threshold is the MCC-optimal one on a validation split ("dynamic threshold").

Map: a separate autoencoder with a 2-D tanh bottleneck fitted on all training apps. Its
latent plane is what the viewer navigates; any point decodes to a permission profile.

Labels: permissions are tiered with PsDC (direct personal data, behavioural/contextual,
structural) and combined with the detector probability into an A–E, SCALE-style label.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor

from catdemos.common.datasets import cache_dir, load_naticus

VERSION = "asap-v1"
SEED = 0
GRID = 96
LATENT_LIM = 1.0  # tanh bottleneck -> latent plane is [-1, 1]^2
MAX_DOTS = 3000

# PsDC tiers (paper 5): 1 = direct personal data, 2 = behavioural / contextual, 3 = structural.
TIER1 = {
    "READ_CONTACTS",
    "WRITE_CONTACTS",
    "READ_SMS",
    "RECEIVE_SMS",
    "SEND_SMS",
    "READ_CALENDAR",
    "WRITE_CALENDAR",
    "ACCESS_FINE_LOCATION",
    "RECORD_AUDIO",
    "CAMERA",
    "GET_ACCOUNTS",
    "READ_PROFILE",
    "MANAGE_ACCOUNTS",
    "AUTHENTICATE_ACCOUNTS",
    "USE_CREDENTIALS",
    "READ_EXTERNAL_STORAGE",
    "WRITE_EXTERNAL_STORAGE",
    "CALL_PHONE",
    "PROCESS_INCOMING_CALLS",
    "USE_FINGERPRINT",
    "READ_PHONE_STATE",
}
TIER2 = {
    "ACCESS_COARSE_LOCATION",
    "ACCESS_MOCK_LOCATION",
    "ACCESS_LOCATION_EXTRA_COMMANDS",
    "ACTIVITY_RECOGNITION",
    "GET_TASKS",
    "READ_LOGS",
    "ACCESS_WIFI_STATE",
    "CHANGE_WIFI_STATE",
    "BLUETOOTH",
    "BLUETOOTH_ADMIN",
    "NFC",
    "RECEIVE_BOOT_COMPLETED",
    "BILLING",
    "READ_SYNC_SETTINGS",
    "WRITE_SYNC_SETTINGS",
    "SYSTEM_ALERT_WINDOW",
    "REQUEST_INSTALL_PACKAGES",
    "DOWNLOAD_WITHOUT_NOTIFICATION",
    "KILL_BACKGROUND_PROCESSES",
    "RESTART_PACKAGES",
    "DISABLE_KEYGUARD",
    "READ_GSERVICES",
    "BIND_GET_INSTALL_REFERRER_SERVICE",
    "RECEIVE_USER_PRESENT",
    "CHANGE_CONFIGURATION",
    "MOUNT_UNMOUNT_FILESYSTEMS",
    "WRITE_SETTINGS",
}
TIER_WEIGHT = np.array([0.0, 3.0, 1.5, 0.25], dtype=np.float32)  # indexed by tier 1..3
LABELS = "ABCDE"


def tier_of(perm: str) -> int:
    """PsDC tier of an Android permission: 1 direct personal data, 2 behavioural, 3 structural."""
    short = perm.rsplit(".", 1)[-1]
    if perm.startswith("android.permission.") and short in TIER1:
        return 1
    return 2 if short in TIER2 else 3


def _layers(m: MLPRegressor) -> list[tuple[np.ndarray, np.ndarray]]:
    """Copy an MLP's weights as float32 (weight, bias) pairs for a fast numpy forward pass."""
    return [
        (w.astype(np.float32), b.astype(np.float32)) for w, b in zip(m.coefs_, m.intercepts_, strict=True)
    ]


def _forward(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray, last_linear: bool) -> np.ndarray:
    """Forward pass with tanh hidden units; the last layer is linear when ``last_linear``."""
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if not (last_linear and i == len(layers) - 1):
            x = np.tanh(x)
    return x


@dataclass
class Engine:
    """Trained ASAP models plus the precomputed test-set data behind every endpoint."""

    perms: list[str]
    tiers: np.ndarray  # (n_perms,) 1..3
    ben: list[tuple[np.ndarray, np.ndarray]]
    mal: list[tuple[np.ndarray, np.ndarray]]
    enc: list[tuple[np.ndarray, np.ndarray]]  # map encoder: input -> 2-D latent
    dec: list[tuple[np.ndarray, np.ndarray]]  # map decoder: latent -> input
    calib: tuple[float, float]  # logistic calibration score -> P(malicious)
    threshold: float
    exposure_max: float
    test_score: np.ndarray
    test_y: np.ndarray
    dots: np.ndarray  # (MAX_DOTS, 3): latent x, latent y, label
    grid: np.ndarray  # (GRID, GRID) P(malicious) of decoded profiles, row 0 = top (y = +1)
    knn: NearestNeighbors = field(repr=False)
    knn_y: np.ndarray = field(repr=False)
    knn_nperm: np.ndarray = field(repr=False)

    # ------------------------------------------------------------------ scoring
    def score(self, X: np.ndarray) -> np.ndarray:
        """log(err_benign_AE) - log(err_malicious_AE): high means malicious-like."""
        X = X.astype(np.float32)
        eb = ((_forward(self.ben, X, True) - X) ** 2).mean(1)
        em = ((_forward(self.mal, X, True) - X) ** 2).mean(1)
        return np.log(eb + 1e-6) - np.log(em + 1e-6)

    def p_malicious(self, s: np.ndarray) -> np.ndarray:
        """Calibrated probability of being malicious for anomaly scores ``s``."""
        a, b = self.calib
        return 1.0 / (1.0 + np.exp(-(a * s + b)))

    def exposure(self, X: np.ndarray) -> np.ndarray:
        """PsDC exposure in [0, 1]: requested permissions weighted by tier."""
        return np.minimum((X.astype(np.float32) @ TIER_WEIGHT[self.tiers]) / self.exposure_max, 1.0)

    def label(self, p: float, exposure: float) -> str:
        """SCALE-style A-E label from detector probability (60 %) and PsDC exposure (40 %)."""
        risk = 0.6 * p + 0.4 * exposure
        return LABELS[min(int(risk * 5), 4)]

    # --------------------------------------------------------------- endpoints
    def probe(self, x: float, y: float) -> dict:
        """Decode latent point (x, y) into a permission profile and judge it."""
        z = np.array([[x, y]], dtype=np.float32)
        probs = np.clip(_forward(self.dec, z, True)[0], 0.0, 1.0)
        prof = (probs >= 0.5).astype(np.float32)
        s = float(self.score(prof[None])[0])
        p = float(self.p_malicious(np.array([s]))[0])
        exp = float(self.exposure(prof[None])[0])
        dist, idx = self.knn.kneighbors(z, n_neighbors=7)
        order = np.argsort(-probs)
        perms = [
            {
                "name": self.perms[i].rsplit(".", 1)[-1],
                "full": self.perms[i],
                "p": round(float(probs[i]), 3),
                "tier": int(self.tiers[i]),
                "on": bool(prof[i]),
            }
            for i in order
            if prof[i]  # every requested permission, most certain first
        ]
        return {
            "score": round(s, 4),
            "p_malicious": round(p, 4),
            "flagged": s >= self.threshold,
            "exposure": round(exp, 3),
            "label": self.label(p, exp),
            "n_perms": int(prof.sum()),
            "tiers": {str(t): int(prof[self.tiers == t].sum()) for t in (1, 2, 3)},
            "perms": perms,
            "neighbours": {
                "malicious": int(self.knn_y[idx[0]].sum()),
                "k": int(idx.shape[1]),
                "mean_perms": round(float(self.knn_nperm[idx[0]].mean()), 1),
                "mean_dist": round(float(dist.mean()), 3),
            },
        }

    def at_threshold(self, t: float) -> dict:
        """Confusion matrix, MCC, TPR and FPR on the held-out apps at threshold ``t``."""
        pred = self.test_score >= t
        y = self.test_y.astype(bool)
        tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
        fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return {
            "t": t,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "mcc": round((tp * tn - fp * fn) / den, 4) if den else 0.0,
            "tpr": round(tp / max(tp + fn, 1), 4),
            "fpr": round(fp / max(fp + tn, 1), 4),
        }

    def overview(self) -> dict:
        """Everything the page needs once: map background, dots, score histogram, threshold."""
        lo, hi = np.quantile(self.test_score, [0.005, 0.995])
        bins = np.linspace(lo, hi, 61)
        hb, _ = np.histogram(self.test_score[self.test_y == 0], bins)
        hm, _ = np.histogram(self.test_score[self.test_y == 1], bins)
        return {
            "grid": np.round(self.grid, 3).tolist(),
            "grid_n": GRID,
            "lim": LATENT_LIM,
            "dots": np.round(self.dots, 3).tolist(),
            "hist": {"edges": np.round(bins, 4).tolist(), "benign": hb.tolist(), "malicious": hm.tolist()},
            "threshold": self.threshold,
            "at_threshold": self.at_threshold(self.threshold),
            "n_test": int(self.test_y.size),
            "n_perms": len(self.perms),
        }


def _ae(hidden: tuple[int, ...], X: np.ndarray, max_iter: int) -> MLPRegressor:
    """Fit a tanh autoencoder (MLPRegressor with X as its own target)."""
    m = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="tanh",
        max_iter=max_iter,
        tol=1e-6,
        n_iter_no_change=20,
        random_state=SEED,
    ).set_params(batch_size=256)  # set_params: sklearn's signature types batch_size as str ("auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return m.fit(X, X)


def train(max_iter: int = 300) -> Engine:
    """Train the detectors and the map on NATICUSdroid (60/15/25 split) and precompute the views."""
    d = load_naticus()
    X, y = d.X.astype(np.float32), d.y.astype(np.int8)
    Xtr, Xte, ytr, yte = map(
        np.asarray, train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    )
    Xtr, Xva, ytr, yva = map(
        np.asarray, train_test_split(Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
    )

    ben = _ae((64, 8, 64), Xtr[ytr == 0], max_iter)
    mal = _ae((64, 8, 64), Xtr[ytr == 1], max_iter)
    mp = _ae((32, 2, 32), Xtr, max_iter)
    map_layers = _layers(mp)
    tiers = np.array([tier_of(p) for p in d.perms], dtype=np.int8)
    w = TIER_WEIGHT[tiers]

    eng = Engine(
        perms=d.perms,
        tiers=tiers,
        ben=_layers(ben),
        mal=_layers(mal),
        enc=map_layers[:2],
        dec=map_layers[2:],
        calib=(1.0, 0.0),
        threshold=0.0,
        exposure_max=float(np.quantile(X @ w, 0.99)),
        test_score=np.empty(0),
        test_y=yte,
        dots=np.empty((0, 3)),
        grid=np.empty((0, 0)),
        knn=NearestNeighbors(),
        knn_y=np.empty(0),
        knn_nperm=np.empty(0),
    )

    # Dynamic threshold: MCC-optimal on validation; calibration: 1-D logistic on validation.
    s_va = eng.score(Xva)
    cand = np.quantile(s_va, np.linspace(0.01, 0.99, 400))
    mccs = [matthews_corrcoef(yva, s_va >= t) for t in cand]
    eng.threshold = float(cand[int(np.argmax(mccs))])
    lr = LogisticRegression().fit(s_va[:, None], yva)
    eng.calib = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    eng.test_score = eng.score(Xte)
    zte = np.tanh(_forward(eng.enc, Xte, last_linear=True))  # bottleneck activation is tanh
    eng.knn = NearestNeighbors(n_neighbors=7).fit(zte)
    eng.knn_y, eng.knn_nperm = yte, Xte.sum(1)
    pick = np.random.default_rng(SEED).choice(len(yte), size=min(MAX_DOTS, len(yte)), replace=False)
    eng.dots = np.column_stack([zte[pick], yte[pick]]).astype(np.float32)

    ax = np.linspace(-LATENT_LIM, LATENT_LIM, GRID, dtype=np.float32)
    gx, gy = np.meshgrid(ax, ax[::-1])
    Z = np.column_stack([gx.ravel(), gy.ravel()])
    prof = (np.clip(_forward(eng.dec, Z, True), 0, 1) >= 0.5).astype(np.float32)
    eng.grid = eng.p_malicious(eng.score(prof)).reshape(GRID, GRID).astype(np.float32)
    return eng


def load_or_train() -> Engine:
    """Load the cached engine, or train it once and cache it (``data/cache``)."""
    path = cache_dir() / f"{VERSION}.joblib"
    if path.exists():
        return joblib.load(path)
    eng = train()
    joblib.dump(eng, path)
    return eng
