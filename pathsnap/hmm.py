"""HMM + Viterbi route reconstruction.

Hidden state at time i     : which candidate arc (and position on it) the vehicle is on.
Emission  log P(z_i | s)   : w_emit * log of the classifier probability, normalised over
                             the candidates of observation i.
Transition log P(s' | s)   : -|d_route(s, s') - d_gps(i-1, i)| / (beta * sqrt(dt))
                             (Newson & Krumm 2009): a true move along the network has a
                             route length close to the straight-line GPS displacement.
                             d_route = rest of arc s + shortest path + part of arc s'.
                             Unreachable within the search limit -> probability 0.
Breaks                     : a time gap > GAP_BREAK_S, or a step where no transition is
                             possible, ends the current chain and starts a new one; the
                             pieces are later joined with an unrestricted shortest path.
"""
from dataclasses import dataclass, field

import numpy as np

GAP_BREAK_S = 120.0
SAME_ARC_BACKTRACK_M = 15.0   # small backwards jitter on the same arc is treated as noise


@dataclass
class HMMParams:
    beta: float = 5.0
    w_emit: float = 1.0


@dataclass
class Lattice:
    obs: np.ndarray                         # observation index of each lattice step
    arcs: list                              # candidate arc ids per step
    logp: list                              # log classifier probability per candidate (normalised)
    d_route: list = field(default_factory=list)  # (K_prev, K_cur) matrices; None = forced break
    d_gps: np.ndarray = None
    dt: np.ndarray = None


def build_lattice(px, py, ts, cand_obs, cand_arc, cand_t, cand_prob, net) -> Lattice:
    order = np.lexsort((-cand_prob, cand_obs))
    cand_obs, cand_arc, cand_t, cand_prob = cand_obs[order], cand_arc[order], cand_t[order], cand_prob[order]
    obs, start = np.unique(cand_obs, return_index=True)
    ends = np.r_[start[1:], len(cand_obs)]
    arcs, ts_, logp = [], [], []
    for s, e in zip(start, ends):
        p = np.clip(cand_prob[s:e], 1e-6, None)
        arcs.append(cand_arc[s:e]); ts_.append(cand_t[s:e]); logp.append(np.log(p / p.sum()))
    lat = Lattice(obs=obs, arcs=arcs, logp=logp)
    d_gps = np.hypot(np.diff(px[obs]), np.diff(py[obs]))
    dt = np.diff(ts[obs])
    lat.d_gps, lat.dt = d_gps, dt
    L, frm, to = net.length, net.arcs[:, 0], net.arcs[:, 1]
    lat.d_route = [None]
    for k in range(1, len(obs)):
        if dt[k - 1] > GAP_BREAK_S:
            lat.d_route.append(None)
            continue
        A, tA, B, tB = arcs[k - 1], ts_[k - 1], arcs[k], ts_[k]
        limit = max(400.0, 2.0 * d_gps[k - 1] + 150.0)
        M = np.full((len(A), len(B)), np.inf)
        fromB = frm[B]
        headB = tB * L[B]
        for ia, (a, ta) in enumerate(zip(A, tA)):
            rest = (1.0 - ta) * L[a]
            dist = net.dijkstra(int(to[a]), limit)
            for ib, b in enumerate(B):
                if b == a:
                    back = (ta - tB[ib]) * L[a]
                    if back <= SAME_ARC_BACKTRACK_M:
                        M[ia, ib] = abs(back)
                        continue
                dv = dist.get(int(fromB[ib]))
                if dv is not None:
                    M[ia, ib] = rest + dv + headB[ib]
        lat.d_route.append(M)
    return lat


def viterbi(lat: Lattice, prm: HMMParams):
    """Returns the chosen candidate index for each lattice step and the number of chain breaks."""
    n = len(lat.obs)
    choice = np.zeros(n, dtype=np.int64)
    back = [None] * n                       # back[k][j] = best predecessor of state j at step k
    score = prm.w_emit * lat.logp[0]
    chain_start, n_breaks = 0, 0

    def finish_chain(end, final_score):
        c = int(np.argmax(final_score))
        choice[end] = c
        for k in range(end, chain_start, -1):
            c = int(back[k][c])
            choice[k - 1] = c

    for k in range(1, n):
        e = prm.w_emit * lat.logp[k]
        M = lat.d_route[k]
        if M is not None:
            beta = prm.beta * np.sqrt(max(lat.dt[k - 1], 1.0))
            tot = score[:, None] - np.abs(M - lat.d_gps[k - 1]) / beta   # unreachable -> -inf
            bp = np.argmax(tot, axis=0)
            best = tot[bp, np.arange(tot.shape[1])]
            if np.isfinite(best).any():
                score, back[k] = best + e, bp
                continue
        # break: close the current chain and start a new one at step k
        finish_chain(k - 1, score)
        chain_start, n_breaks, score = k, n_breaks + 1, e
    finish_chain(n - 1, score)
    return choice, n_breaks


def decode(lat: Lattice, prm: HMMParams):
    """Per-step chosen arc ids and number of breaks."""
    choice, n_breaks = viterbi(lat, prm)
    arcs = np.array([lat.arcs[k][c] for k, c in enumerate(choice)], dtype=np.int64)
    return arcs, n_breaks


def compress(seq):
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def stitch_route(arc_seq, net, max_gap_m: float = 20000.0):
    """Turn the per-observation arc sequence into a connected arc path by inserting
    shortest paths between consecutive distinct arcs. Returns (route, n_unbridged)."""
    seq = compress(arc_seq)
    if len(seq) == 0:
        return [], 0
    route, unbridged = [int(seq[0])], 0
    for a, b in zip(seq[:-1], seq[1:]):
        a, b = int(a), int(b)
        if net.arcs[a, 1] != net.arcs[b, 0]:
            path = net.shortest_path_arcs(int(net.arcs[a, 1]), int(net.arcs[b, 0]), limit=max_gap_m)
            if path is None:
                unbridged += 1
            else:
                route.extend(path)
        route.append(b)
    return route, unbridged
