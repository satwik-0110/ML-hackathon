"""Road network: projected geometry, connectivity, spatial index and shortest paths."""
import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 focus_xy: np.ndarray = None, focus_margin: float = 400.0, sample_step: float = 10.0):
        self.proj = proj
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.nx, self.ny = proj.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        a, b = self.arcs[:, 0], self.arcs[:, 1]
        self.ax, self.ay, self.bx, self.by = self.nx[a], self.ny[a], self.nx[b], self.ny[b]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = np.arctan2(self.by - self.ay, self.bx - self.ax)  # radians, math convention

        # connectivity
        self.out_deg = np.bincount(a, minlength=self.n_nodes)
        self.in_deg = np.bincount(b, minlength=self.n_nodes)
        key = a * self.n_nodes + b
        rkey = b * self.n_nodes + a
        self.has_reverse = np.isin(rkey, key)
        # map (from,to) -> arc id of the reverse arc (for u-turn detection)
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        rev = order[pos]
        self.reverse_arc = np.where(key[rev] == rkey, rev, -1)

        # CSR adjacency as python lists (fast inside the pure-python Dijkstra loop)
        order = np.argsort(a, kind="stable")
        ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(self.out_deg, out=ptr[1:])
        self._ptr = ptr.tolist()
        self._adj_arc = order.tolist()
        self._adj_to = b[order].tolist()
        self._adj_len = self.length[order].tolist()
        self._sp_cache = {}

        self._build_index(focus_xy, focus_margin, sample_step)

    # ------------------------------------------------------------------ spatial index
    def _build_index(self, focus_xy, margin, step):
        idx = np.arange(self.n_arcs)
        if focus_xy is not None and len(focus_xy):
            x0, y0 = focus_xy.min(0) - margin
            x1, y1 = focus_xy.max(0) + margin
            m = (np.maximum(self.ax, self.bx) >= x0) & (np.minimum(self.ax, self.bx) <= x1) & \
                (np.maximum(self.ay, self.by) >= y0) & (np.minimum(self.ay, self.by) <= y1)
            idx = idx[m]
        n = np.maximum(np.ceil(self.length[idx] / step).astype(int), 1) + 1
        arc_of_sample = np.repeat(idx, n)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n]) if len(n) else np.zeros(0)
        sx = self.ax[arc_of_sample] + frac * (self.bx[arc_of_sample] - self.ax[arc_of_sample])
        sy = self.ay[arc_of_sample] + frac * (self.by[arc_of_sample] - self.ay[arc_of_sample])
        self._sample_arc = arc_of_sample
        self._sample_step = step
        self._tree = cKDTree(np.c_[sx, sy]) if len(sx) else None

    def arcs_near(self, px: np.ndarray, py: np.ndarray, radius: float):
        """All (point_index, arc, distance, t) pairs with distance <= radius."""
        if self._tree is None:
            return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
        hits = self._tree.query_ball_point(np.c_[px, py], r=radius + self._sample_step / 2 + 0.01)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
        pi = np.repeat(np.arange(len(hits)), lens)
        arc = self._sample_arc[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if h])]
        pair = np.unique(pi * self.n_arcs + arc)
        pi, arc = pair // self.n_arcs, pair % self.n_arcs
        d, t = point_segment(px[pi], py[pi], self.ax[arc], self.ay[arc], self.bx[arc], self.by[arc])
        keep = d <= radius
        return pi[keep], arc[keep], d[keep], t[keep]

    # ------------------------------------------------------------------ shortest paths
    def dijkstra(self, src: int, limit: float) -> dict:
        """Distances from node `src` to every node reachable within `limit` metres.
        Results are cached per source and reused when a later query needs a smaller limit."""
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1]
        ptr, adj_to, adj_len = self._ptr, self._adj_to, self._adj_len
        dist = {src: 0.0}
        heap = [(0.0, src)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                v = adj_to[k]
                nd = d + adj_len[k]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._sp_cache) > 200_000:
            self._sp_cache.clear()
        self._sp_cache[src] = (limit, dist)
        return dist

    def shortest_path_arcs(self, src: int, dst: int, limit: float = np.inf):
        """Arc ids of the shortest directed path src -> dst (None if unreachable within limit)."""
        if src == dst:
            return []
        ptr, adj_to, adj_len, adj_arc = self._ptr, self._adj_to, self._adj_len, self._adj_arc
        dist = {src: 0.0}
        prev = {}
        heap = [(0.0, src)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                v = adj_to[k]
                nd = d + adj_len[k]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = adj_arc[k]
                    heapq.heappush(heap, (nd, v))
        if dst not in prev:
            return None
        path, v = [], dst
        while v != src:
            e = prev[v]
            path.append(e)
            v = int(self.arcs[e, 0])
        return path[::-1]
