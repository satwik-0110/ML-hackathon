"""Run the complete pipeline (candidates -> features -> classifier -> HMM/Viterbi -> route) on every
trip in a folder, using one of the saved models, and print the scores per trip.

This is exactly what the API does for one trip, run for a whole folder from the terminal.
Nothing is trained here: the models are loaded from artifacts/.

Usage:
  python scripts/evaluate_trips.py --dir <folder with one sub-folder per trip> [--model KEY] [--save]

Model keys: lightgbm (final model, default), random_forest, mlp, extra_trees, knn, logreg, gnb
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.io import load_route, load_trip  # noqa: E402
from pathsnap.pipeline import PathSnap, evaluate_prediction  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DATA_DIR / "student_test"), help="folder with one sub-folder per trip")
    ap.add_argument("--model", default=None, help="model key (default: the final tuned LightGBM)")
    ap.add_argument("--save", action="store_true", help="also write the table to results/")
    args = ap.parse_args()

    ps = PathSnap()
    entry, _ = ps.get_model(args.model)
    folder = Path(args.dir)
    trips = sorted(d.name for d in folder.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())
    print(f"Model : {entry['name']}   HMM beta={ps.hmm_params.beta} w_emit={ps.hmm_params.w_emit}")
    print(f"Trips : {len(trips)} in {folder}\n")
    print(f"{'trip':<10}{'points':>8}{'classifier':>12}{'with HMM':>10}{'route F1':>10}{'continuity':>16}{'seconds':>9}")

    rows = []
    for tid in trips:
        trip = load_trip(folder / tid, with_route=False)          # prediction never reads the route
        res = ps.predict_trip(trip.track, trip.nodes, trip.arcs, model_key=args.model)
        ev = evaluate_prediction(res, load_route(folder / tid))    # scoring reads it afterwards
        rows.append(dict(trip=tid, points=res["n_obs"], labelled=ev["labelled_points"],
                         top1_classifier=ev["top1_classifier"], top1_hmm=ev["top1_hmm"],
                         route_precision=ev["route_precision"], route_recall=ev["route_recall"],
                         route_f1=ev["route_f1"], route_mismatch=ev["route_mismatch"],
                         continuity_classifier=res["continuity_clf"], continuity_hmm=res["continuity_hmm"],
                         hmm_breaks=res["hmm_breaks"], seconds=res["timing"]["total_s"]))
        r = rows[-1]
        print(f"{tid:<10}{r['points']:>8}{r['top1_classifier']:>12.3f}{r['top1_hmm']:>10.3f}{r['route_f1']:>10.3f}"
              f"{r['continuity_classifier']:>9.2f} -> {r['continuity_hmm']:.2f}{r['seconds']:>9.2f}", flush=True)

    df = pd.DataFrame(rows)
    w = df["labelled"] / df["labelled"].sum()
    print(f"\n{'ALL':<10}{df['points'].sum():>8}{(df['top1_classifier'] * w).sum():>12.3f}{(df['top1_hmm'] * w).sum():>10.3f}"
          f"{df['route_f1'].mean():>10.3f}{df['continuity_classifier'].mean():>9.2f} -> {df['continuity_hmm'].mean():.2f}"
          f"{df['seconds'].sum():>9.2f}")
    print("(classifier / with HMM = share of GPS points on the correct road segment, weighted by points;"
          " route F1 and continuity = mean over trips)")
    if args.save:
        out = RESULTS_DIR / f"eval_{folder.name}_{entry['key']}.csv"
        df.to_csv(out, index=False)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
