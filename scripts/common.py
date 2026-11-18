"""Shared helpers for the experiment scripts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.config import CACHE_DIR, DATA_DIR, EXCLUDED_TRAIN_TRIPS, TRAIN_STRIDE  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402


def load_train(stride=TRAIN_STRIDE):
    """Training rows: labelled observations of the allowed training trips, every `stride`-th point."""
    df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
    obs = pd.read_parquet(CACHE_DIR / "observations_train.parquet")
    df = df[~df["trip"].isin(EXCLUDED_TRAIN_TRIPS)]
    lab = obs.set_index(["trip", "obs"])["label"]
    df = df[(df["obs"] % stride == 0)]
    df = df[lab.reindex(pd.MultiIndex.from_arrays([df["trip"], df["obs"]])).to_numpy() >= 0]
    return df.reset_index(drop=True)


def load_test():
    """All student-test rows plus per-observation labels."""
    df = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
    obs = pd.read_parquet(CACHE_DIR / "observations_student_test.parquet")
    return df.reset_index(drop=True), obs


def XY(df):
    return df[FEATURES].to_numpy(np.float32), df["y"].to_numpy()


def top1_by_obs(df, p):
    """Return a frame (trip, obs, pred_arc) with the highest-probability candidate per observation."""
    tmp = pd.DataFrame({"trip": df["trip"].to_numpy(), "obs": df["obs"].to_numpy(),
                        "arc": df["arc"].to_numpy(), "p": p})
    idx = tmp.groupby(["trip", "obs"])["p"].idxmax()
    return tmp.loc[idx, ["trip", "obs", "arc"]].rename(columns={"arc": "pred"})


def top1_accuracy(df, p, obs_labels):
    """Observation-level Top-1 accuracy over labelled observations (obs_labels: trip, obs, label)."""
    pred = top1_by_obs(df, p)
    m = obs_labels[obs_labels["label"] >= 0].merge(pred, on=["trip", "obs"], how="left")
    return float((m["pred"] == m["label"]).mean())


def train_obs_labels(df):
    """Labels of the observations present in a training frame (the row with y==1 per observation)."""
    pos = df[df["y"] == 1][["trip", "obs", "arc"]].rename(columns={"arc": "label"})
    allobs = df[["trip", "obs"]].drop_duplicates()
    # labelled observation whose true arc is not among its candidates: unreachable label -> always an error
    return allobs.merge(pos, on=["trip", "obs"], how="left").fillna({"label": 10**12}).astype({"label": np.int64})


_ARCS = {}


def trip_arcs(part, tid):
    if (part, tid) not in _ARCS:
        _ARCS[(part, tid)] = np.loadtxt(DATA_DIR / part / tid / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    return _ARCS[(part, tid)]


def classifier_continuity(df, p, part="student_test"):
    """Mean over trips of the fraction of consecutive distinct argmax arcs that are connected."""
    pred = top1_by_obs(df, p).sort_values(["trip", "obs"])
    vals = []
    for tid, g in pred.groupby("trip"):
        s = g["pred"].to_numpy()
        s = s[np.r_[True, s[1:] != s[:-1]]]
        if len(s) < 2:
            continue
        arcs = trip_arcs(part, tid)
        vals.append(float((arcs[s[:-1], 1] == arcs[s[1:], 0]).mean()))
    return float(np.mean(vals))
