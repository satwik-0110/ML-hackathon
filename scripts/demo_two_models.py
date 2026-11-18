"""Quick demo: train and evaluate two models on the prepared dataset.

  * Logistic Regression - simple linear baseline
  * LightGBM            - gradient boosting, the model selected in the full comparison

Usage: python scripts/demo_two_models.py      (run scripts/02_build_dataset.py first)
The full 7-model comparison with cross-validation is scripts/03_train_models.py.
"""
import time

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import XY, classifier_continuity, load_test, load_train, top1_accuracy
from pathsnap.config import SEED
from pathsnap.metrics import candidate_metrics

MODELS = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=SEED, verbose=-1),
}


def main():
    train = load_train()
    test, test_obs = load_test()
    X, y = XY(train)
    Xt, yt = XY(test)
    print(f"Training rows : {len(X):,} (point, candidate) pairs from {train['trip'].nunique()} trips, "
          f"{y.mean():.1%} are the true arc")
    print(f"Test rows     : {len(Xt):,} pairs from {test['trip'].nunique()} student-test trips\n")
    labelled = test_obs.set_index(["trip", "obs"])["label"].reindex(
        list(zip(test["trip"], test["obs"]))).to_numpy() >= 0

    rows = []
    for name, model in MODELS.items():
        t0 = time.perf_counter(); model.fit(X, y); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = model.predict_proba(Xt)[:, 1]; inf_s = time.perf_counter() - t0
        m = candidate_metrics(yt[labelled], p[labelled])
        rows.append({"model": name, "top1_accuracy": top1_accuracy(test, p, test_obs), "macro_f1": m["macro_f1"],
                     "precision": m["precision"], "recall": m["recall"], "log_loss": m["log_loss"],
                     "route_continuity": classifier_continuity(test, p), "train_s": fit_s, "predict_s": inf_s})
        print(f"{name}: trained in {fit_s:.1f}s")
    print("\nStudent-test results")
    print(pd.DataFrame(rows).set_index("model").round(4).T.to_string())


if __name__ == "__main__":
    main()
