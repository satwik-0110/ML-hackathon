"""HMM / Viterbi: tune decoder parameters and compare against classifier-only predictions.

1. Out-of-fold (GroupKFold by trip) probabilities of the tuned model for every point of the
   training trips -> emissions that are as "honest" as on unseen trips.
2. Grid search of (beta, w_emit) on the training trips, objective = point Top-1 accuracy.
   The lattice (candidates + route distances) is built once per trip; only decoding is repeated.
3. Write artifacts/hmm_config.json and evaluate the full pipeline on the 10 student-test trips:
   classifier-only vs classifier+HMM vs HMM with a distance-only emission (no classifier).
"""
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from common import load_train
from pathsnap import hmm
from pathsnap.config import ARTIFACT_DIR, CACHE_DIR, CV_FOLDS, DATA_DIR, EXCLUDED_TRAIN_TRIPS, RESULTS_DIR, SEED
from pathsnap.features import FEATURES, prepare_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.metrics import continuity, point_accuracy, route_metrics
from pathsnap.pipeline import PathSnap, evaluate_prediction

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BETAS = [1, 2, 3, 5, 8, 12, 20, 40]
W_EMITS = [0.5, 1.0, 2.0]
N_TUNE_TRIPS = 30


def oof_probabilities(params):
    """OOF probability for every candidate row of every allowed training trip."""
    full = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
    full = full[~full["trip"].isin(EXCLUDED_TRAIN_TRIPS)].reset_index(drop=True)
    tr_df = load_train()
    trips = np.array(sorted(full["trip"].unique()))
    prob = np.zeros(len(full))
    for f, (_, va_idx) in enumerate(GroupKFold(n_splits=CV_FOLDS).split(trips, groups=trips)):
        va_trips = set(trips[va_idx])
        fit_rows = tr_df[~tr_df["trip"].isin(va_trips)]
        m = LGBMClassifier(random_state=SEED, verbose=-1, **params).fit(
            fit_rows[FEATURES].to_numpy(np.float32), fit_rows["y"].to_numpy())
        rows = full["trip"].isin(va_trips).to_numpy()
        prob[rows] = m.predict_proba(full.loc[rows, FEATURES].to_numpy(np.float32))[:, 1]
        print(f"OOF fold {f} done", flush=True)
    full["p"] = prob
    return full


def main():
    bundle = joblib.load(ARTIFACT_DIR / "model.joblib")
    params = {k: v for k, v in bundle["params"].items()}
    params.setdefault("subsample_freq", 1)
    full = oof_probabilities(params)

    rng = np.random.default_rng(SEED)
    tune_trips = sorted(rng.choice(sorted(full["trip"].unique()), size=N_TUNE_TRIPS, replace=False))
    lattices = []
    for tid in tune_trips:
        trip = load_trip(DATA_DIR / "train" / tid)
        proj, px, py, net = prepare_trip(trip.track, trip.nodes, trip.arcs)
        label, _, _ = align_points_to_route(px, py, trip.route, net)
        g = full[full["trip"] == tid]
        t0 = time.perf_counter()
        lat = hmm.build_lattice(px, py, trip.track[:, 2], g["obs"].to_numpy(), g["arc"].to_numpy(),
                                g["t_proj"].to_numpy(), g["p"].to_numpy(), net)
        clf = np.full(len(px), -1); clf[lat.obs] = [a[np.argmax(lp)] for a, lp in zip(lat.arcs, lat.logp)]
        lattices.append((tid, lat, label, net, clf))
        print(f"lattice {tid}: {len(px)} pts, {time.perf_counter() - t0:.1f}s", flush=True)

    grid = []
    for beta in BETAS:
        for w in W_EMITS:
            prm = hmm.HMMParams(beta=beta, w_emit=w)
            correct = total = 0
            for tid, lat, label, net, clf in lattices:
                arcs, _ = hmm.decode(lat, prm)
                pred = np.full(len(label), -1); pred[lat.obs] = arcs
                m = label >= 0
                correct += int((pred[m] == label[m]).sum()); total += int(m.sum())
            grid.append(dict(beta=beta, w_emit=w, train_oof_top1_hmm=correct / total))
            print(grid[-1], flush=True)
    grid = pd.DataFrame(grid)
    clf_correct = sum(int((c[l >= 0] == l[l >= 0]).sum()) for _, _, l, _, c in lattices)
    clf_total = sum(int((l >= 0).sum()) for _, _, l, _, _ in lattices)
    grid["train_oof_top1_classifier_only"] = clf_correct / clf_total
    grid.to_csv(RESULTS_DIR / "hmm_grid.csv", index=False)
    best = grid.loc[grid["train_oof_top1_hmm"].idxmax()]
    cfg = dict(beta=float(best["beta"]), w_emit=float(best["w_emit"]), gap_break_s=hmm.GAP_BREAK_S,
               tuned_on=f"OOF predictions, {N_TUNE_TRIPS} training trips", objective="point Top-1 accuracy")
    (ARTIFACT_DIR / "hmm_config.json").write_text(json.dumps(cfg, indent=2))
    print("HMM config", cfg, flush=True)

    piv = grid.pivot(index="beta", columns="w_emit", values="train_oof_top1_hmm")
    plt.figure(figsize=(6.5, 4))
    for w in piv.columns:
        plt.plot(piv.index, piv[w], marker="o", label=f"w_emit={w}")
    plt.axhline(clf_correct / clf_total, color="gray", ls="--", label="classifier only")
    plt.xscale("log"); plt.xlabel("beta (m)"); plt.ylabel("Top-1 accuracy (train OOF)")
    plt.title("HMM decoder tuning"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "hmm_tuning.png", dpi=120); plt.close()

    evaluate_student_test(cfg)


def evaluate_student_test(cfg):
    ps = PathSnap()
    rows = []
    for tid in list_trips(DATA_DIR, "student_test"):
        trip = load_trip(DATA_DIR / "student_test" / tid, with_route=False)
        res = ps.predict_trip(trip.track, trip.nodes, trip.arcs)
        ref = np.loadtxt(DATA_DIR / "student_test" / tid / f"{tid}.route", dtype=np.int64, ndmin=1)
        ev = evaluate_prediction(res, ref)
        # classifier-only "route": argmax sequence without HMM (no stitching)
        clf_seq = hmm.compress(res["clf_pred"][res["clf_pred"] >= 0])
        clf_route = route_metrics(clf_seq, ref, res["net"].length)
        # ablation: HMM with a distance-only Gaussian emission (no classifier)
        lat_d = distance_only_lattice(trip, res)
        arcs_d, _ = hmm.decode(lat_d, hmm.HMMParams(beta=cfg["beta"], w_emit=1.0))
        pred_d = np.full(len(trip.track), -1); pred_d[lat_d.obs] = arcs_d
        labels, _, _ = align_points_to_route(res["px"], res["py"], ref, res["net"])
        route_d, _ = hmm.stitch_route(arcs_d, res["net"])
        rows.append(dict(
            trip=tid, n_obs=res["n_obs"],
            top1_classifier=ev["top1_classifier"], top1_hmm=ev["top1_hmm"],
            top1_hmm_distance_only=point_accuracy(pred_d, labels),
            continuity_classifier=res["continuity_clf"], continuity_hmm=res["continuity_hmm"],
            route_f1_classifier=clf_route["route_f1"], route_f1_hmm=ev["route_f1"],
            route_f1_hmm_distance_only=route_metrics(route_d, ref, res["net"].length)["route_f1"],
            route_precision_hmm=ev["route_precision"], route_recall_hmm=ev["route_recall"],
            route_mismatch_hmm=ev["route_mismatch"], hmm_breaks=res["hmm_breaks"],
            unbridged=res["unbridged_gaps"], total_s=res["timing"]["total_s"],
            continuity_hmm_distance_only=continuity(route_d, res["net"])))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    w = df["n_obs"] / df["n_obs"].sum()
    overall = {c: (df[c] * w).sum() if c.startswith("top1") else df[c].mean()
               for c in df.columns if c not in ("trip",)}
    overall["trip"] = "ALL (top1 point-weighted, others trip-mean)"
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    df.to_csv(RESULTS_DIR / "hmm_student_test.csv", index=False)
    print(df.round(4).to_string(), flush=True)

    d = df.iloc[:-1]
    x = np.arange(len(d)); wd = 0.27
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].bar(x - wd, d["top1_classifier"], wd, label="classifier only")
    ax[0].bar(x, d["top1_hmm_distance_only"], wd, label="HMM, distance emission")
    ax[0].bar(x + wd, d["top1_hmm"], wd, label="classifier + HMM")
    ax[0].set_xticks(x, d["trip"], rotation=45); ax[0].set_ylim(0.5, 1.0); ax[0].set_title("Point Top-1 accuracy")
    ax[0].legend(loc="lower right")
    ax[1].bar(x - wd / 2, d["continuity_classifier"], wd, label="classifier only")
    ax[1].bar(x + wd / 2, d["continuity_hmm"], wd, label="classifier + HMM")
    ax[1].set_xticks(x, d["trip"], rotation=45); ax[1].set_title("Route continuity"); ax[1].legend(loc="lower right")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "plots" / "hmm_vs_classifier.png", dpi=120); plt.close()


def distance_only_lattice(trip, res):
    from pathsnap.features import generate_candidates
    net = res["net"]
    pi, arc, d, t, _ = generate_candidates(res["px"], res["py"], net)
    p = np.exp(-0.5 * (d / 5.0) ** 2) + 1e-12   # Gaussian GPS noise, sigma = 5 m
    return hmm.build_lattice(res["px"], res["py"], trip.track[:, 2], pi, arc, t, p, net)


if __name__ == "__main__":
    main()
