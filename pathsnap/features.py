"""Candidate generation and feature engineering.

Every GPS observation i gets a candidate set: the K nearest *directed* arcs
within CAND_RADIUS_M (widened to CAND_FALLBACK_M if empty). Each
(observation, candidate-arc) pair becomes one row; the classifier learns
P(this candidate is the true arc | features).
"""
import numpy as np
import pandas as pd

from .config import CAND_FALLBACK_M, CAND_MAX_K, CAND_RADIUS_M
from .geo import LocalProjection, point_segment
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "heading_cos", "heading_valid", "speed",
    "ctx_dist", "ctx_lateral",
    "arc_len", "t_proj", "end_dist",
    "oneway", "deg_to", "deg_from",
    "dt_prev", "gps_jitter",
]


def prepare_trip(track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray):
    """Project a trip and build its road network (spatial index limited to the track area)."""
    proj = LocalProjection(np.median(track[:, 0]), np.median(track[:, 1]))
    px, py = proj.to_xy(track[:, 0], track[:, 1])
    net = RoadNetwork(nodes, arcs, proj, focus_xy=np.c_[px, py], focus_margin=CAND_FALLBACK_M + 100)
    return proj, px, py, net


def _topk(pi, arc, d, t, k):
    o = np.lexsort((d, pi))
    pi, arc, d, t = pi[o], arc[o], d[o], t[o]
    first = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    grp_start = np.repeat(first, np.diff(np.r_[first, len(pi)]))
    rank = np.arange(len(pi)) - grp_start
    m = rank < k
    return pi[m], arc[m], d[m], t[m], rank[m]


def generate_candidates(px, py, net: RoadNetwork, radius=CAND_RADIUS_M, k=CAND_MAX_K):
    pi, arc, d, t = net.arcs_near(px, py, radius)
    missing = np.setdiff1d(np.arange(len(px)), pi)
    if len(missing):
        p2, a2, d2, t2 = net.arcs_near(px[missing], py[missing], CAND_FALLBACK_M)
        pi = np.r_[pi, missing[p2]]; arc = np.r_[arc, a2]; d = np.r_[d, d2]; t = np.r_[t, t2]
    return _topk(pi, arc, d, t, k)


def _motion(px, py, ts, w=2):
    n = len(px)
    i0 = np.clip(np.arange(n) - w, 0, n - 1)
    i1 = np.clip(np.arange(n) + w, 0, n - 1)
    dx, dy = px[i1] - px[i0], py[i1] - py[i0]
    move = np.hypot(dx, dy)
    span = np.maximum(ts[i1] - ts[i0], 1e-6)
    heading = np.arctan2(dy, dx)
    valid = (move > 3.0).astype(float)
    speed = np.where(i1 > i0, move / span, 0.0)
    return heading, valid, speed


def build_features(track, px, py, net: RoadNetwork) -> pd.DataFrame:
    ts = track[:, 2]
    n = len(px)
    pi, arc, d, t, rank = generate_candidates(px, py, net)
    df = pd.DataFrame({"obs": pi, "arc": arc, "dist": d, "t_proj": t, "dist_rank": rank})
    g = df.groupby("obs")["dist"]
    dmin = g.transform("min").to_numpy()
    df["n_cands"] = g.transform("size").to_numpy()
    df["dist_minus_min"] = d - dmin
    df["dist_ratio"] = d / (dmin + 1.0)

    heading, valid, speed = _motion(px, py, ts)
    hc = np.cos(heading[pi] - net.bearing[arc])
    df["heading_cos"] = np.where(valid[pi] > 0, hc, 0.0)
    df["heading_valid"] = valid[pi]
    df["speed"] = speed[pi]

    # context: how well do the neighbouring observations fit the same arc?
    ax, ay, bx, by = net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc]
    ctx = np.zeros(len(df)); lat = np.zeros(len(df))
    L = np.maximum(net.length[arc], 1e-6)
    ux, uy = (bx - ax) / L, (by - ay) / L
    offs = [-2, -1, 1, 2]
    for o in offs:
        j = np.clip(pi + o, 0, n - 1)
        dj, _ = point_segment(px[j], py[j], ax, ay, bx, by)
        ctx += np.minimum(dj, 100.0)
        lat += np.minimum(np.abs((px[j] - ax) * uy - (py[j] - ay) * ux), 100.0)
    df["ctx_dist"] = ctx / len(offs)
    df["ctx_lateral"] = lat / len(offs)

    df["arc_len"] = net.length[arc]
    df["end_dist"] = np.minimum(t, 1 - t) * net.length[arc]
    df["oneway"] = (~net.has_reverse[arc]).astype(float)
    df["deg_to"] = net.out_deg[net.arcs[arc, 1]]
    df["deg_from"] = net.in_deg[net.arcs[arc, 0]]
    dtp = np.r_[1.0, np.diff(ts)]
    df["dt_prev"] = dtp[pi]
    ip, inx = np.clip(np.arange(n) - 1, 0, n - 1), np.clip(np.arange(n) + 1, 0, n - 1)
    jitter = np.hypot(px - (px[ip] + px[inx]) / 2, py - (py[ip] + py[inx]) / 2)
    df["gps_jitter"] = jitter[pi]
    return df
