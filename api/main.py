"""PathSnap FastAPI backend (Laptop 1).

Hosts the trained classifier, the HMM/Viterbi decoder, the road networks and the
reference routes of the evaluation trips. The UI only sends a trip id; the API runs
the full pipeline, scores it against the reference route *on the server* and
returns the predicted route + metrics. Reference routes are never sent to the client.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR = folder with one sub-folder per evaluation trip
                          (<id>/<id>.track, .nodes, .arcs, .route)
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.config import DATA_DIR  # noqa: E402
from pathsnap.io import load_route, load_trip  # noqa: E402
from pathsnap.pipeline import PathSnap, evaluate_prediction, route_to_lonlat  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", DATA_DIR / "student_test"))
MAX_GPS_POINTS_IN_RESPONSE = 4000

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = PathSnap()
log.info("Loaded model: %s | HMM beta=%.1f w_emit=%.2f", model.model_name, model.hmm_params.beta,
         model.hmm_params.w_emit)
log.info("Evaluation trips folder: %s", EVAL_DIR)


def available_trips():
    return sorted(d.name for d in EVAL_DIR.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())


class PredictRequest(BaseModel):
    trip_id: str
    model: Optional[str] = None   # model key from GET /models; omitted = the final tuned model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, time.perf_counter() - t0)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": model.model_name, "eval_dir": str(EVAL_DIR), "n_trips": len(available_trips())}


@app.get("/models")
def models():
    """Models the API can run. The default is the final tuned model; the others are the comparison models."""
    return {"models": [{k: m.get(k) for k in ("key", "name", "family", "student_test_top1", "default")} for m in model.catalog]}


@app.get("/trips")
def trips():
    out = []
    for tid in available_trips():
        tr = np.loadtxt(EVAL_DIR / tid / f"{tid}.track", ndmin=2)
        out.append(dict(trip_id=tid, n_points=int(len(tr)), duration_s=float(tr[-1, 2] - tr[0, 2]),
                        center=[float(np.median(tr[:, 1])), float(np.median(tr[:, 0]))]))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in available_trips():
        raise HTTPException(404, f"unknown trip id {req.trip_id!r}")
    if req.model is not None and req.model not in {m["key"] for m in model.catalog}:
        raise HTTPException(404, f"unknown model {req.model!r}")
    folder = EVAL_DIR / req.trip_id
    log.info("[%s] request received (model=%s) - loading GPS track and road network", req.trip_id,
             req.model or model.default_key)
    trip = load_trip(folder, with_route=False)           # prediction never sees the reference route
    log.info("[%s] %d GPS points, %d nodes, %d arcs", req.trip_id, len(trip.track), len(trip.nodes), len(trip.arcs))
    res = model.predict_trip(trip.track, trip.nodes, trip.arcs, model_key=req.model)
    t = res["timing"]
    log.info("[%s] %d candidates | features %.2fs, classifier %.2fs, HMM+stitch %.2fs | %d breaks",
             req.trip_id, res["n_candidates"], t["features_s"], t["classifier_s"], t["hmm_s"], res["hmm_breaks"])

    # --- server-side scoring (reference route stays on this machine) ---
    metrics = {}
    route_file = folder / f"{req.trip_id}.route"
    if route_file.exists():
        metrics = evaluate_prediction(res, load_route(folder))
        log.info("[%s] Top-1 classifier=%.4f  HMM=%.4f | route P=%.3f R=%.3f F1=%.3f mismatch=%.3f",
                 req.trip_id, metrics["top1_classifier"], metrics["top1_hmm"], metrics["route_precision"],
                 metrics["route_recall"], metrics["route_f1"], metrics["route_mismatch"])
    metrics.update(continuity_classifier=res["continuity_clf"], continuity_hmm=res["continuity_hmm"])

    track = trip.track
    step = max(1, len(track) // MAX_GPS_POINTS_IN_RESPONSE)
    clf_arcs = np.unique(res["clf_pred"][res["clf_pred"] >= 0])
    net = res["net"]
    lon_a, lat_a = net.proj.to_lonlat(net.ax[clf_arcs], net.ay[clf_arcs])
    lon_b, lat_b = net.proj.to_lonlat(net.bx[clf_arcs], net.by[clf_arcs])
    return {
        "trip_id": req.trip_id,
        "model": res["model_name"],
        "model_key": res["model_key"],
        "hmm_params": {"beta": model.hmm_params.beta, "w_emit": model.hmm_params.w_emit},
        "n_points": int(len(track)),
        "gps": np.c_[track[::step, 1], track[::step, 0]].round(7).tolist(),
        "predicted_route": route_to_lonlat(res["route"], net),
        "predicted_route_arcs": [int(a) for a in res["route"]],
        "classifier_only_segments": np.stack([np.c_[lat_a, lon_a], np.c_[lat_b, lon_b]], 1).round(7).tolist(),
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "diagnostics": {"n_candidates": res["n_candidates"], "hmm_breaks": res["hmm_breaks"],
                        "unbridged_gaps": res["unbridged_gaps"],
                        "obs_without_candidates": res["n_obs_without_candidates"],
                        "route_arcs": len(res["route"]), "route_length_km": round(float(net.length[res["route"]].sum()) / 1000, 3)
                        if res["route"] else 0.0},
        "timing_s": {k: round(v, 3) for k, v in t.items()},
    }
