"""Geometry helpers.

Trips are spread across the globe, so each trip gets its own local
equirectangular projection centred on the trip. Over a few tens of km this
is accurate to well below GPS noise (< 0.1 %), and it lets us use plain
Euclidean geometry in metres.
"""
import numpy as np

EARTH_R = 6371008.8  # mean Earth radius [m]


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * EARTH_R

    def to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_lonlat(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (broadcasting) and the projection
    fraction t in [0, 1] along AB."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    safe = np.where(L2 > 0, L2, 1.0)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / safe, 0.0, 1.0)
    t = np.where(L2 > 0, t, 0.0)
    d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    return d, t
