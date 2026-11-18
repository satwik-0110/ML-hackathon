"""Train and compare candidate classifiers from several algorithm families.

Protocol
  * training data : 78 training trips (2 excluded, see config), every 5th GPS point
  * CV            : 5-fold GroupKFold grouped by trip (no trip in both train and validation)
  * dev test      : all points of the 10 student-test trips
  * primary metric: observation-level Top-1 accuracy (does the highest-scoring candidate equal
                    the reference arc?), log loss as secondary since the probabilities feed the HMM.
Outputs: results/model_comparison.csv, results/cv_folds.csv, plots in results/plots/,
         fitted models in cache/models/.
"""
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from common import XY, classifier_continuity, load_test, load_train, top1_accuracy, train_obs_labels
from pathsnap.config import CACHE_DIR, CV_FOLDS, RESULTS_DIR, SEED
from pathsnap.metrics import candidate_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS = RESULTS_DIR / "plots"


def make_models():
    return {
        "LogisticRegression": ("Linear", "C=1.0, L2, balanced=False, StandardScaler",
                               make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
        "GaussianNB": ("Probabilistic", "var_smoothing=1e-9",
                       GaussianNB()),
        "KNN": ("Instance-based", "k=50, distance-weighted, StandardScaler",
                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1))),
        "RandomForest": ("Ensemble (bagging)", "300 trees, min_samples_leaf=5, max_features=sqrt",
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "ExtraTrees": ("Ensemble (randomised trees)", "300 trees, min_samples_leaf=5",
                       ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "LightGBM": ("Boosting", "400 trees, lr=0.05, num_leaves=31",
                     LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)),
        "MLP": ("Neural network", "hidden=(64,32), relu, adam, early stopping, StandardScaler",
                make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                              max_iter=200, random_state=SEED))),
    }


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "models").mkdir(exist_ok=True)
    train = load_train()
    test, test_obs = load_test()
    X, y = XY(train)
    Xt, yt = XY(test)
    groups = train["trip"].to_numpy()
    # candidate-level test metrics only on labelled observations
    lab = test_obs.set_index(["trip", "obs"])["label"]
    test_lab_mask = lab.reindex(pd.MultiIndex.from_arrays([test["trip"], test["obs"]])).to_numpy() >= 0
    print(f"train rows={len(train)} obs={train[['trip','obs']].drop_duplicates().shape[0]} "
          f"pos_rate={y.mean():.3f} | test rows={len(test)}", flush=True)

    rows, fold_rows = [], []
    gkf = GroupKFold(n_splits=CV_FOLDS)
    for name, (family, cfg, model) in make_models().items():
        oof = np.zeros(len(y))
        fold_f1 = []
        for f, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = clone(model)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
            f1 = f1_score(y[va], (oof[va] >= 0.5).astype(int), average="macro")
            fold_f1.append(f1)
            fold_rows.append(dict(model=name, fold=f, macro_f1=f1,
                                  top1=top1_accuracy(train.iloc[va], oof[va], train_obs_labels(train.iloc[va]))))
        cv = candidate_metrics(y, oof)
        cv_top1 = top1_accuracy(train, oof, train_obs_labels(train))

        t0 = time.perf_counter(); model.fit(X, y); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); pt = model.predict_proba(Xt)[:, 1]; inf_s = time.perf_counter() - t0
        te = candidate_metrics(yt[test_lab_mask], pt[test_lab_mask])
        row = dict(model=name, family=family, config=cfg,
                   cv_macro_f1=cv["macro_f1"], cv_macro_f1_std=float(np.std(fold_f1)), cv_top1=cv_top1,
                   cv_log_loss=cv["log_loss"],
                   test_top1=top1_accuracy(test, pt, test_obs),
                   test_macro_f1=te["macro_f1"], test_precision=te["precision"], test_recall=te["recall"],
                   test_f1_pos=te["f1_pos"], test_log_loss=te["log_loss"],
                   test_route_continuity=classifier_continuity(test, pt),
                   train_time_s=fit_s, infer_ms_per_1k_rows=1000 * inf_s / len(Xt) * 1000)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        joblib.dump(model, CACHE_DIR / "models" / f"{name}.joblib")
        np.save(CACHE_DIR / "models" / f"{name}_test_proba.npy", pt)

        cm = confusion_matrix(yt[test_lab_mask], (pt[test_lab_mask] >= 0.5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["wrong arc", "true arc"]).plot(values_format="d", cmap="Blues")
        plt.title(f"{name} - student test (candidate level)"); plt.tight_layout()
        plt.savefig(PLOTS / f"cm_{name}.png", dpi=110); plt.close()

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    plot_comparison(res, pd.DataFrame(fold_rows))
    learning_curves(train)


def plot_comparison(res, folds):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    folds.boxplot(column="macro_f1", by="model", ax=ax[0], rot=30)
    ax[0].set_title("CV macro-F1 per fold (GroupKFold by trip)"); ax[0].set_xlabel("")
    x = np.arange(len(res)); w = 0.38
    ax[1].bar(x - w / 2, res["test_top1"], w, label="Top-1 accuracy")
    ax[1].bar(x + w / 2, res["test_macro_f1"], w, label="Macro-F1")
    ax[1].set_xticks(x, res["model"], rotation=30); ax[1].set_ylim(0.5, 1); ax[1].legend()
    ax[1].set_title("Student-test performance")
    ax[2].bar(x, res["test_log_loss"], color="tab:red")
    ax[2].set_xticks(x, res["model"], rotation=30); ax[2].set_title("Student-test log loss (lower is better)")
    plt.suptitle(""); plt.tight_layout(); plt.savefig(PLOTS / "model_comparison.png", dpi=120); plt.close()


def learning_curves(train):
    """Epoch/iteration-wise train vs validation log loss for the two models that have training histories.
    Validation = 20% of training trips held out by group."""
    trips = np.array(sorted(train["trip"].unique()))
    rng = np.random.default_rng(SEED)
    val_trips = set(rng.choice(trips, size=len(trips) // 5, replace=False))
    va = train["trip"].isin(val_trips).to_numpy()
    X, y = XY(train)

    lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=SEED, verbose=-1)
    lgb.fit(X[~va], y[~va], eval_set=[(X[~va], y[~va]), (X[va], y[va])], eval_names=["train", "valid"],
            eval_metric="binary_logloss")
    ev = lgb.evals_result_
    plt.figure(figsize=(6, 4))
    plt.plot(ev["train"]["binary_logloss"], label="train"); plt.plot(ev["valid"]["binary_logloss"], label="validation")
    plt.xlabel("boosting iteration"); plt.ylabel("log loss"); plt.title("LightGBM learning curve"); plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS / "curve_lightgbm.png", dpi=120); plt.close()

    sc = StandardScaler().fit(X[~va])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=SEED, learning_rate_init=1e-3)
    tr_l, va_l = [], []
    for ep in range(40):
        mlp.partial_fit(sc.transform(X[~va]), y[~va], classes=[0, 1])
        tr_l.append(log_loss(y[~va], mlp.predict_proba(sc.transform(X[~va]))[:, 1]))
        va_l.append(log_loss(y[va], mlp.predict_proba(sc.transform(X[va]))[:, 1]))
    plt.figure(figsize=(6, 4))
    plt.plot(tr_l, label="train"); plt.plot(va_l, label="validation")
    plt.xlabel("epoch"); plt.ylabel("log loss"); plt.title("MLP learning curve"); plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS / "curve_mlp.png", dpi=120); plt.close()
    pd.DataFrame({"epoch": range(len(tr_l)), "train_log_loss": tr_l, "val_log_loss": va_l}).to_csv(
        RESULTS_DIR / "mlp_history.csv", index=False)


if __name__ == "__main__":
    main()
