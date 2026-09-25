"""Download-once loaders for the public datasets used by the demos.

Files land in ``$CATDEMOS_DATA`` (default ``./data``): ``raw/`` for the original
archives, ``cache/`` for derived arrays and trained models. Both are gitignored.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URLS = {
    "naticus.zip": "https://archive.ics.uci.edu/static/public/722/naticusdroid+android+permissions+dataset.zip",
    "adult.zip": "https://archive.ics.uci.edu/static/public/2/adult.zip",
}
# Pinned archive hashes: every result is computed from exactly these bytes.
SHA256 = {
    "naticus.zip": "8d1a47a3a307bf68c3b0a51d58e146eddef941b4e403aeafce03a61fe96cbc82",
    "adult.zip": "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
}


def data_dir() -> Path:
    """Root of the dataset and model cache (``$CATDEMOS_DATA``, default ``./data``)."""
    return Path(os.environ.get("CATDEMOS_DATA", "data")).resolve()


def cache_dir() -> Path:
    """Directory for derived arrays and trained models, created on demand."""
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(name: str) -> Path:
    """Return the local path of a raw archive, downloading and verifying it on first use."""
    raw = data_dir() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / name
    if not path.exists():
        with urllib.request.urlopen(URLS[name], timeout=120) as r:
            blob = r.read()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != SHA256[name]:
            raise RuntimeError(f"{name}: SHA-256 {digest} does not match the pinned {SHA256[name]}")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(blob)
        tmp.rename(path)
    return path


# ---------------------------------------------------------------- NATICUSdroid


@dataclass(frozen=True)
class Naticus:
    """NATICUSdroid permission matrix, labels and permission names."""

    X: np.ndarray  # (n_apps, n_perms) uint8, 1 = permission requested
    y: np.ndarray  # (n_apps,) uint8, 1 = malicious
    perms: list[str]  # full permission names


def load_naticus() -> Naticus:
    """Load NATICUSdroid, parsing the CSV once and caching it as compressed numpy."""
    cached = cache_dir() / "naticus.npz"
    if cached.exists():
        z = np.load(cached, allow_pickle=False)
        return Naticus(z["X"], z["y"], [str(p) for p in z["perms"]])
    with zipfile.ZipFile(fetch("naticus.zip")) as zf, zf.open("data.csv") as fh:
        rows = list(csv.reader(io.TextIOWrapper(fh, encoding="utf-8")))
    header, body = rows[0], rows[1:]
    arr = np.asarray(body, dtype=np.uint8)
    out = Naticus(arr[:, :-1], arr[:, -1], header[:-1])
    np.savez_compressed(cached, X=out.X, y=out.y, perms=np.asarray(out.perms))
    return out


# ---------------------------------------------------------------------- Adult

ADULT_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_adult() -> list[dict[str, str]]:
    """All rows of adult.data + adult.test with missing values ('?') dropped."""
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(fetch("adult.zip")) as zf:
        for member in ("adult.data", "adult.test"):
            for line in zf.read(member).decode("utf-8").splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != len(ADULT_COLS) or "?" in parts:
                    continue  # blank lines, the '|1x3 Cross validator' header, missing values
                parts[-1] = parts[-1].rstrip(".")  # adult.test labels end with '.'
                rows.append(dict(zip(ADULT_COLS, parts, strict=True)))
    return rows
