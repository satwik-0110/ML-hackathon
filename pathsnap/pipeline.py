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
        # Optional comparison models (scripts/07_export_models.py). The final model is the default;
        # the others are loaded from disk the first time they are requested.
        self.model_dir = artifact_dir / "models"
        index_file = self.model_dir / "index.json"
        self.catalog = json.loads(index_file.read_text()) if index_file.exists() else [
            dict(key="lightgbm", name=self.model_name, family="Boosting", file="../model.joblib", default=True)]
        self.default_key = next(m["key"] for m in self.catalog if m.get("default"))
        self._loaded = {self.default_key: self.model}

    def get_model(self, key=None):
        """Return (catalog entry, fitted model) for a model key; None means the final model."""
        key = key or self.default_key
        entry = next((m for m in self.catalog if m["key"] == key), None)
        if entry is None:
            raise KeyError(key)
        if key not in self._loaded:
            # keep memory bounded: only the final model and the most recently used comparison model stay loaded
            for other in [k for k in self._loaded if k != self.default_key]:
                del self._loaded[other]
            self._loaded[key] = joblib.load(self.model_dir / entry["file"])
        return entry, self._loaded[key]

    def predict_trip(self, track, nodes, arcs, model_key=None):
        entry, model = self.get_model(model_key)
        t0 = time.perf_counter()
        proj, px, py, net = prepare_trip(track, nodes, arcs)
        df = build_features(track, px, py, net)
        t1 = time.perf_counter()
        prob = model.predict_proba(df[self.features].to_numpy(np.float32))[:, 1]
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
            net=net, proj=proj, px=px, py=py, model_key=entry["key"], model_name=entry["name"],
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
