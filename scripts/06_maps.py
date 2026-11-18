"""Folium maps of the final pipeline on the student-test trips.

Layers: GPS observations, road network near the track, reference route (ground truth),
classifier-only arcs, predicted route (classifier + HMM). Output: results/maps/pred_<trip>.html
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.pipeline import PathSnap, evaluate_prediction  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402


def main():
    out = RESULTS_DIR / "maps"; out.mkdir(parents=True, exist_ok=True)
    ps = PathSnap()
    trips = sys.argv[1:] or list_trips(DATA_DIR, "student_test")
    for tid in trips:
        trip = load_trip(DATA_DIR / "student_test" / tid, with_route=False)
        res = ps.predict_trip(trip.track, trip.nodes, trip.arcs)
        ref = np.loadtxt(DATA_DIR / "student_test" / tid / f"{tid}.route", dtype=np.int64, ndmin=1)
        ev = evaluate_prediction(res, ref)
        title = (f"student_test {tid} | Top-1 classifier {ev['top1_classifier']:.3f} -> HMM {ev['top1_hmm']:.3f} | "
                 f"route F1 {ev['route_f1']:.3f}")
        m = trip_map(trip.track, res["net"], res["px"], res["py"], ref_route=ref, clf_arcs=res["clf_pred"],
                     pred_route=res["route"], title=title)
        m.save(out / f"pred_{tid}.html")
        print(title, flush=True)


if __name__ == "__main__":
    main()
