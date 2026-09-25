"""Walkable street graph around Fábrica Centro Ciência Viva and UA Campus de Santiago.

Fetched once from the Overpass API and cached as JSON; if that fails (offline first run),
a regular grid over the same box is used so the demo still starts.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from catdemos.common.datasets import cache_dir

FABRICA = (40.6381, -8.6577)
CAMPUS = (40.6320, -8.6598)
BBOX = (40.6255, -8.6680, 40.6445, -8.6475)  # south, west, north, east
ORIGIN = (40.6350, -8.6578)
HIGHWAYS = (
    "footway|pedestrian|path|living_street|residential|service|tertiary|secondary|primary|unclassified"
    "|steps|cycleway"
)
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(ORIGIN[0]))


def to_xy(latlng: np.ndarray) -> np.ndarray:
    """(n, 2) lat/lng -> local metres (east, north) around ORIGIN."""
    return np.column_stack(
        [(latlng[:, 1] - ORIGIN[1]) * M_PER_DEG_LON, (latlng[:, 0] - ORIGIN[0]) * M_PER_DEG_LAT]
    )


def to_latlng(xy: np.ndarray) -> np.ndarray:
    """Local metres (east, north) back to lat/lng."""
    return np.column_stack([xy[:, 1] / M_PER_DEG_LAT + ORIGIN[0], xy[:, 0] / M_PER_DEG_LON + ORIGIN[1]])


@dataclass
class Graph:
    """Street graph in local metres."""

    xy: np.ndarray  # (n, 2) node coordinates in metres
    edges: np.ndarray  # (m, 2) node index pairs
    source: str  # "osm" or "grid"

    def route(self, a: int, b: int, pred: np.ndarray) -> np.ndarray:
        """Polyline from ``a`` to ``b`` following Dijkstra predecessors ``pred``."""
        path = [b]
        while path[-1] != a and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        return self.xy[path[::-1]]

    def random_routes(self, n: int, rng: np.random.Generator, min_len: float = 300.0) -> list[np.ndarray]:
        """Shortest-path routes between random node pairs, as polylines in metres."""
        w = np.linalg.norm(self.xy[self.edges[:, 0]] - self.xy[self.edges[:, 1]], axis=1)
        k = len(self.xy)
        adj = coo_matrix(
            (
                np.r_[w, w],
                (np.r_[self.edges[:, 0], self.edges[:, 1]], np.r_[self.edges[:, 1], self.edges[:, 0]]),
            ),
            shape=(k, k),
        ).tocsr()
        out: list[np.ndarray] = []
        while len(out) < n:
            src = rng.integers(k, size=8)
            dist, pred = dijkstra(adj, indices=src, return_predecessors=True)
            for row, a in enumerate(src):
                far = np.flatnonzero((dist[row] > min_len) & np.isfinite(dist[row]))
                if far.size:
                    out.append(self.route(int(a), int(rng.choice(far)), pred[row]))
        return out[:n]


def _fetch_osm() -> dict:
    """Query Overpass for the walkable ways inside BBOX."""
    s, w, n, e = BBOX
    q = f'[out:json][timeout:60];way["highway"~"^({HIGHWAYS})$"]({s},{w},{n},{e});(._;>;);out skel qt;'
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=urllib.parse.urlencode({"data": q}).encode(),
        headers={"User-Agent": "catdemos (IEETA research demo)"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def _from_osm(d: dict) -> Graph:
    """Build the graph from an Overpass response, keeping the largest connected component."""
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in d["elements"] if el["type"] == "node"}
    ids = list(nodes)
    index = {nid: i for i, nid in enumerate(ids)}
    edges = [
        (index[a], index[b])
        for el in d["elements"]
        if el["type"] == "way"
        for a, b in zip(el["nodes"], el["nodes"][1:], strict=False)
        if a in index and b in index
    ]
    xy = to_xy(np.array([nodes[i] for i in ids]))
    E = np.array(edges)
    # Keep the largest connected component so every random route exists.
    k = len(xy)
    adj = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(k, k))
    _, comp = connected_components(adj, directed=False)
    keep = comp == np.bincount(comp).argmax()
    remap = -np.ones(k, dtype=int)
    remap[keep] = np.arange(keep.sum())
    E = remap[E]
    return Graph(xy[keep], E[(E >= 0).all(1)], "osm")


def _grid() -> Graph:
    """Fallback regular street grid over BBOX (used only when OSM is unreachable)."""
    s, w, n, e = BBOX
    lat, lng = np.meshgrid(np.linspace(s, n, 22), np.linspace(w, e, 18), indexing="ij")
    xy = to_xy(np.column_stack([lat.ravel(), lng.ravel()]))
    r, c = lat.shape
    idx = np.arange(r * c).reshape(r, c)
    E = np.vstack(
        [
            np.column_stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),
            np.column_stack([idx[:-1, :].ravel(), idx[1:, :].ravel()]),
        ]
    )
    return Graph(xy, E, "grid")


def load_graph() -> Graph:
    """The street graph, fetching and caching OSM data on first use."""
    path = cache_dir() / "aveiro_osm.json"
    if not path.exists():
        try:
            data = _fetch_osm()
        except Exception:
            return _grid()
        tmp = path.with_suffix(".part")  # write then rename: an interrupted run never leaves half a file
        tmp.write_text(json.dumps(data))
        tmp.rename(path)
    return _from_osm(json.loads(path.read_text()))
