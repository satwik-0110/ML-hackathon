"""Reading the raw trip files (.track, .nodes, .arcs, .route)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3): lon, lat, t [s]
    nodes: np.ndarray            # (V, 2): lon, lat
    arcs: np.ndarray             # (E, 2): from_node, to_node
    route: Optional[np.ndarray]  # (M,) arc ids in travel order (ground truth; may be None)


def trip_dir(data_dir: Path, partition: str, trip_id: str) -> Path:
    return Path(data_dir) / partition / trip_id


def list_trips(data_dir: Path, partition: str) -> list:
    p = Path(data_dir) / partition
    return sorted(d.name for d in p.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())


def load_trip(folder: Path, with_route: bool = True) -> Trip:
    """Load one trip folder. `with_route=False` is what the prediction pipeline uses,
    so the ground truth never enters inference."""
    folder = Path(folder)
    tid = folder.name
    track = np.loadtxt(folder / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(folder / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(folder / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    route = None
    rp = folder / f"{tid}.route"
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=np.int64, ndmin=1)
    # basic cleaning: drop rows with NaNs, sort by time, drop duplicate timestamps
    track = track[np.isfinite(track).all(1)]
    track = track[np.argsort(track[:, 2], kind="stable")]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    track = track[keep]
    return Trip(tid, track, nodes, arcs, route)


def load_route(folder: Path) -> np.ndarray:
    folder = Path(folder)
    return np.loadtxt(folder / f"{folder.name}.route", dtype=np.int64, ndmin=1)
