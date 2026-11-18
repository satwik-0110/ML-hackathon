"""Evaluation metrics (point level and route level)."""
import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p, thr=0.5):
    """Binary classification metrics on (observation, candidate) rows."""
    yhat = (p >= thr).astype(int)
    return dict(
        macro_f1=f1_score(y, yhat, average="macro"),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1_pos=f1_score(y, yhat, zero_division=0),
        accuracy=float((yhat == y).mean()),
        log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    )


def top1_from_candidates(obs, arc, p, label_per_obs):
    """Observation-level Top-1: argmax candidate per observation vs the label.
    Only observations with a valid label (!= -1) count; a label missing from
    the candidate set counts as an error (candidate miss)."""
    o = np.lexsort((-p, obs))
    obs_s, arc_s = obs[o], arc[o]
    first = np.r_[True, obs_s[1:] != obs_s[:-1]]
    pred = np.full(len(label_per_obs), -1, dtype=np.int64)
    pred[obs_s[first]] = arc_s[first]
    return pred


def point_accuracy(pred, label):
    m = label >= 0
    return float((pred[m] == label[m]).mean()) if m.any() else float("nan")


def candidate_miss_rate(obs, arc, label):
    m = label >= 0
    hit = np.zeros(len(label), bool)
    is_true = arc == label[obs]
    hit[obs[is_true]] = True
    return float(1 - hit[m].mean()) if m.any() else float("nan")


def continuity(seq, net):
    """Fraction of consecutive (distinct) arcs that are actually connected in the network."""
    seq = np.asarray(seq)
    if len(seq) < 2:
        return 1.0
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    if len(seq) < 2:
        return 1.0
    ok = net.arcs[seq[:-1], 1] == net.arcs[seq[1:], 0]
    return float(ok.mean())


def route_metrics(pred_route, ref_route, length):
    """Length-weighted comparison of the predicted arc set with the reference arc set."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    lp = sum(length[a] for a in P)
    lr = sum(length[a] for a in R)
    inter = sum(length[a] for a in P & R)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # Newson & Krumm route mismatch fraction: (wrongly added + missed length) / reference length
    mismatch = ((lp - inter) + (lr - inter)) / lr if lr else float("nan")
    return dict(route_precision=prec, route_recall=rec, route_f1=f1, route_mismatch=mismatch,
                arc_jaccard=len(P & R) / max(len(P | R), 1))
