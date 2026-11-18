"""End-to-end inference: raw trip -> candidates -> classifier -> HMM/Viterbi -> route.

`predict_trip` never sees the reference route. Evaluation against the reference
is a separate step (`evaluate_prediction`) that only the API server calls.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import hmm
from .config import ARTIFACT_DIR
from .features import FEATURES, build_features, prepare_trip
from .labels import align_points_to_route
from .metrics import continuity, point_accuracy, route_metrics


class PathSnap:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        bundle = joblib.load(artifact_dir / "model.joblib")
        self.model = bundle["model"]
        self.model_name = bundle.get("name", "model")
        self.features = bundle.get("features", FEATURES)
        cfg = json.loads((artifact_dir / "hmm_config.json").read_text())
        self.hmm_params = hmm.HMMParams(beta=cfg["beta"], w_emit=cfg["w_emit"])

    def predict_trip(self, track, nodes, arcs):
        t0 = time.perf_counter()
        proj, px, py, net = prepare_trip(track, nodes, arcs)
        df = build_features(track, px, py, net)
        t1 = time.perf_counter()
        prob = self.model.predict_proba(df[self.features].to_numpy(np.float32))[:, 1]
        t2 = time.perf_counter()
        obs, arc, tproj = df["obs"].to_numpy(), df["arc"].to_numpy(), df["t_proj"].to_numpy()

        # classifier only: best candidate per observation
        o = np.lexsort((-prob, obs))
        first = np.r_[True, obs[o][1:] != obs[o][:-1]]
        clf_pred = np.full(len(track), -1, dtype=np.int64)
        clf_pred[obs[o][first]] = arc[o][first]

        # HMM / Viterbi
        lat = hmm.build_lattice(px, py, track[:, 2], obs, arc, tproj, prob, net)
        hmm_arcs, n_breaks = hmm.decode(lat, self.hmm_params)
        hmm_pred = np.full(len(track), -1, dtype=np.int64)
        hmm_pred[lat.obs] = hmm_arcs
        route, unbridged = hmm.stitch_route(hmm_arcs, net)
        t3 = time.perf_counter()

        return dict(
            net=net, proj=proj, px=px, py=py,
            clf_pred=clf_pred, hmm_pred=hmm_pred, route=route,
            n_obs=len(track), n_candidates=int(len(df)),
            n_obs_without_candidates=int(len(track) - len(np.unique(obs))),
            hmm_breaks=int(n_breaks), unbridged_gaps=int(unbridged),
            continuity_clf=continuity(hmm.compress(clf_pred[clf_pred >= 0]), net),
            continuity_hmm=continuity(route, net),
            timing=dict(features_s=t1 - t0, classifier_s=t2 - t1, hmm_s=t3 - t2, total_s=t3 - t0),
        )


def evaluate_prediction(res, ref_route):
    """Server-side scoring against the reference route (never exposed to the predictor)."""
    net = res["net"]
    labels, _, _ = align_points_to_route(res["px"], res["py"], ref_route, net)
    out = dict(
        labelled_points=int((labels >= 0).sum()),
        top1_classifier=point_accuracy(res["clf_pred"], labels),
        top1_hmm=point_accuracy(res["hmm_pred"], labels),
    )
    out.update(route_metrics(res["route"], ref_route, net.length))
    return out


def route_to_lonlat(route, net):
    """Polyline (list of [lat, lon]) following the arcs of the route."""
    if not route:
        return []
    r = np.asarray(route)
    xs = np.r_[net.ax[r[0]], net.bx[r]]
    ys = np.r_[net.ay[r[0]], net.by[r]]
    lon, lat = net.proj.to_lonlat(xs, ys)
    return np.c_[lat, lon].round(7).tolist()
