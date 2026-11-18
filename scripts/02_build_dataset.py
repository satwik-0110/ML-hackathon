"""Build the candidate-level datasets for the train and student_test partitions.

For every trip: project -> derive per-point labels from the reference route ->
generate candidates -> compute features. Output (in cache/):
  candidates_<partition>.parquet : one row per (trip, observation, candidate arc) + y
  observations_<partition>.parquet : one row per GPS observation with its derived label
and results/dataset_summary.csv with per-trip label coverage and candidate miss rate.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.config import CACHE_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.features import build_features, prepare_trip  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.labels import align_points_to_route  # noqa: E402
from pathsnap.metrics import candidate_miss_rate  # noqa: E402


def main():
    CACHE_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    summary = []
    for part in ["train", "student_test"]:
        cand_parts, obs_parts = [], []
        for tid in list_trips(DATA_DIR, part):
            t0 = time.time()
            trip = load_trip(DATA_DIR / part / tid)
            proj, px, py, net = prepare_trip(trip.track, trip.nodes, trip.arcs)
            label, route_pos, label_dist = align_points_to_route(px, py, trip.route, net)
            df = build_features(trip.track, px, py, net)
            df["y"] = (df["arc"].to_numpy() == label[df["obs"].to_numpy()]).astype(np.int8)
            df.insert(0, "trip", tid)
            cand_parts.append(df)
            obs_parts.append(pd.DataFrame({"trip": tid, "obs": np.arange(len(px)), "t": trip.track[:, 2],
                                           "label": label, "label_dist": label_dist}))
            miss = candidate_miss_rate(df["obs"].to_numpy(), df["arc"].to_numpy(), label)
            summary.append(dict(partition=part, trip=tid, n_obs=len(px), n_rows=len(df),
                                cands_per_obs=len(df) / len(px), labelled_frac=float((label >= 0).mean()),
                                median_label_dist=float(np.median(label_dist)), candidate_miss_rate=miss,
                                seconds=round(time.time() - t0, 2)))
            print(f"{part} {tid}: {len(px):6d} obs  {len(df):7d} rows  labelled={summary[-1]['labelled_frac']:.3f} "
                  f"miss={miss:.4f}  {summary[-1]['seconds']}s", flush=True)
        pd.concat(cand_parts, ignore_index=True).to_parquet(CACHE_DIR / f"candidates_{part}.parquet")
        pd.concat(obs_parts, ignore_index=True).to_parquet(CACHE_DIR / f"observations_{part}.parquet")
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
