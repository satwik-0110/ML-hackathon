"""Deriving per-GPS-point labels from the reference route.

The .route file is only an ordered list of arc ids - it does not say which GPS
point lies on which arc. We align the two sequences with dynamic programming:

    choose j_0 <= j_1 <= ... <= j_{N-1}  (positions in the route)
    minimising  sum_i  min(dist(point_i, arc route[j_i]), CAP)

The monotonic constraint encodes "the vehicle travels the route in order", so a
point near a crossing or a road travelled twice is assigned to the right
occurrence. Points still far from their aligned arc (> LABEL_MAX_DIST_M) are
GPS outliers or reference-route errors and receive label -1 (ignored when
training and scoring).
"""
import numpy as np

from .config import LABEL_MAX_DIST_M
from .geo import point_segment

CAP = 100.0


def align_points_to_route(px, py, route, net):
    route = np.asarray(route, dtype=np.int64)
    N, M = len(px), len(route)
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    ar = np.arange(M)
    bp = np.empty((N, M), dtype=np.int32)
    d0, _ = point_segment(px[0], py[0], ax, ay, bx, by)
    D = np.minimum(d0, CAP)
    bp[0] = ar
    for i in range(1, N):
        run = np.minimum.accumulate(D)
        arg = np.maximum.accumulate(np.where(D == run, ar, 0))
        di, _ = point_segment(px[i], py[i], ax, ay, bx, by)
        D = np.minimum(di, CAP) + run
        bp[i] = arg
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(D))
    for i in range(N - 1, 0, -1):
        j[i - 1] = bp[i, j[i]]
    dist, _ = point_segment(px, py, ax[j], ay[j], bx[j], by[j])
    label = np.where(dist <= LABEL_MAX_DIST_M, route[j], -1)
    return label, j, dist
