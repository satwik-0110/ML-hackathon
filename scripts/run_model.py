"""Train ONE model on the training trips and test it on the student-test trips.

Usage:
  python scripts/run_model.py <model> [--cv] [--hmm]

  <model>  logreg | gnb | knn | random_forest | extra_trees | lightgbm | mlp | lightgbm_tuned
  --cv     also run 5-fold cross-validation grouped by trip on the training trips (slower)
  --hmm    also decode the student-test trips with HMM/Viterbi using this model's probabilities

Run scripts/02_build_dataset.py first. The model definitions are the same ones used by
scripts/03_train_models.py; lightgbm_tuned uses the hyperparameters found by scripts/04_bayes_opt.py.
"""
import argparse
import importlib
import time

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from common import XY, classifier_continuity, load_test, load_train, top1_accuracy, train_obs_labels
from pathsnap import hmm
from pathsnap.config import ARTIFACT_DIR, CV_FOLDS, DATA_DIR, SEED
from pathsnap.features import prepare_trip
from pathsnap.io import load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.metrics import candidate_metrics

KEYS = {"logreg": "LogisticRegression", "gnb": "GaussianNB", "knn": "KNN", "random_forest": "RandomForest",
        "extra_trees": "ExtraTrees", "lightgbm": "LightGBM", "mlp": "MLP"}


def build(key):
    if key == "lightgbm_tuned":
        params = dict(joblib.load(ARTIFACT_DIR / "model.joblib")["params"])
        params.setdefault("subsample_freq", 1)
        return "LightGBM (tuned)", "Boosting", str(params), LGBMClassifier(random_state=SEED, verbose=-1, **params)
    family, config, model = importlib.import_module("03_train_models").make_models()[KEYS[key]]
    return KEYS[key], family, config, model


def hmm_accuracy(test, prob):
    """Point Top-1 on the student-test trips after HMM/Viterbi decoding with these probabilities."""
    cfg = __import__("json").loads((ARTIFACT_DIR / "hmm_config.json").read_text())
    prm = hmm.HMMParams(beta=cfg["beta"], w_emit=cfg["w_emit"])
    correct = total = 0
    for tid, g in test.assign(p=prob).groupby("trip"):
        trip = load_trip(DATA_DIR / "student_test" / tid)
        _, px, py, net = prepare_trip(trip.track, trip.nodes, trip.arcs)
        label, _, _ = align_points_to_route(px, py, trip.route, net)
        lat = hmm.build_lattice(px, py, trip.track[:, 2], g["obs"].to_numpy(), g["arc"].to_numpy(),
                                g["t_proj"].to_numpy(), g["p"].to_numpy(), net)
        arcs, _ = hmm.decode(lat, prm)
        pred = np.full(len(label), -1); pred[lat.obs] = arcs
        m = label >= 0
        correct += int((pred[m] == label[m]).sum()); total += int(m.sum())
    return correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=list(KEYS) + ["lightgbm_tuned"])
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--hmm", action="store_true")
    args = ap.parse_args()

    name, family, config, model = build(args.model)
    train = load_train()
    test, test_obs = load_test()
    X, y = XY(train)
    Xt, yt = XY(test)
    print(f"Model    : {name}  ({family})")
    print(f"Settings : {config}")
    print(f"Training : {len(X):,} rows from {train['trip'].nunique()} trips, {y.mean():.1%} are the true road")
    print(f"Testing  : {len(Xt):,} rows from {test['trip'].nunique()} student-test trips\n")

    if args.cv:
        groups = train["trip"].to_numpy()
        oof = np.zeros(len(y))
        for fold, (tr, va) in enumerate(GroupKFold(n_splits=CV_FOLDS).split(X, y, groups)):
            oof[va] = clone(model).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
            print(f"  cross-validation fold {fold + 1}/{CV_FOLDS} done", flush=True)
        cv = candidate_metrics(y, oof)
        print(f"Cross-validation (5 folds, grouped by trip)")
        print(f"  Top-1 accuracy : {top1_accuracy(train, oof, train_obs_labels(train)):.4f}")
        print(f"  Macro-F1       : {cv['macro_f1']:.4f}")
        print(f"  Log loss       : {cv['log_loss']:.4f}\n")

    t0 = time.perf_counter(); model.fit(X, y); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = model.predict_proba(Xt)[:, 1]; inf_s = time.perf_counter() - t0
    labelled = test_obs.set_index(["trip", "obs"])["label"].reindex(list(zip(test["trip"], test["obs"]))).to_numpy() >= 0
    m = candidate_metrics(yt[labelled], p[labelled])
    print(f"Trained in {fit_s:.1f} s, predicted the test rows in {inf_s:.2f} s")
    print(f"Student-test results")
    print(f"  Top-1 accuracy     : {top1_accuracy(test, p, test_obs):.4f}   (GPS points on the correct road segment)")
    print(f"  Macro-F1           : {m['macro_f1']:.4f}")
    print(f"  Precision / recall : {m['precision']:.4f} / {m['recall']:.4f}")
    print(f"  Log loss           : {m['log_loss']:.4f}")
    print(f"  Route continuity   : {classifier_continuity(test, p):.4f}   (classifier only)")
    if args.hmm:
        print(f"  Top-1 with HMM     : {hmm_accuracy(test, p):.4f}")


if __name__ == "__main__":
    main()
