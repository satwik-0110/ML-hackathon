"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (LightGBM).

Objective : maximise mean observation-level Top-1 accuracy over 5-fold GroupKFold
            (grouped by trip) on the training trips. CV log loss is recorded per trial too.
Only training trips are used for the search; the student-test trips are used once
at the end to compare the tuned and untuned models.
Outputs: results/optuna_trials.csv, results/tuning_comparison.csv,
         results/plots/optuna_*.png, artifacts/model.joblib
"""
import json
import time

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from common import XY, classifier_continuity, load_test, load_train, top1_accuracy, train_obs_labels
from pathsnap.config import ARTIFACT_DIR, CV_FOLDS, RESULTS_DIR, SEED
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_TRIALS = 30
BASELINE = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8)


def lgbm(**p):
    return LGBMClassifier(random_state=SEED, verbose=-1, subsample_freq=1, **{k: v for k, v in p.items()
                                                                                if k != "subsample_freq"})


def cv_eval(params, train, X, y, groups):
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=CV_FOLDS).split(X, y, groups):
        m = lgbm(**params).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    met = candidate_metrics(y, oof)
    return top1_accuracy(train, oof, train_obs_labels(train)), met


def main():
    train = load_train()
    test, test_obs = load_test()
    X, y = XY(train)
    Xt, yt = XY(test)
    groups = train["trip"].to_numpy()
    lab = test_obs.set_index(["trip", "obs"])["label"]
    lab_mask = lab.reindex(pd.MultiIndex.from_arrays([test["trip"], test["obs"]])).to_numpy() >= 0

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        t0 = time.perf_counter()
        top1, met = cv_eval(params, train, X, y, groups)
        trial.set_user_attr("cv_log_loss", met["log_loss"])
        trial.set_user_attr("cv_macro_f1", met["macro_f1"])
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        print(f"trial {trial.number:2d} top1={top1:.4f} logloss={met['log_loss']:.4f} "
              f"({time.perf_counter() - t0:.0f}s) {params}", flush=True)
        return top1

    t_search = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="lightgbm_top1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.enqueue_trial({**{k: v for k, v in BASELINE.items() if k != "subsample_freq"},
                         "min_child_samples": 20, "reg_lambda": 1e-3, "reg_alpha": 1e-3})  # untuned as trial 0
    study.optimize(objective, n_trials=N_TRIALS)
    search_s = time.perf_counter() - t_search
    trials = study.trials_dataframe()
    trials.to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
    best = study.best_params
    print("best", best, study.best_value, f"search {search_s:.0f}s", flush=True)

    rows = []
    for name, params in [("LightGBM untuned", BASELINE), ("LightGBM tuned (Optuna TPE)", best)]:
        cv_top1, cv_met = cv_eval(params, train, X, y, groups)
        t0 = time.perf_counter(); m = lgbm(**params).fit(X, y); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); pt = m.predict_proba(Xt)[:, 1]; inf_s = time.perf_counter() - t0
        te = candidate_metrics(yt[lab_mask], pt[lab_mask])
        rows.append(dict(model=name, cv_top1=cv_top1, cv_macro_f1=cv_met["macro_f1"], cv_log_loss=cv_met["log_loss"],
                         test_top1=top1_accuracy(test, pt, test_obs), test_macro_f1=te["macro_f1"],
                         test_log_loss=te["log_loss"], test_route_continuity=classifier_continuity(test, pt),
                         train_time_s=fit_s, infer_ms_per_1k_rows=1e6 * inf_s / len(Xt),
                         params=json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in params.items()})))
        if "tuned" in name and "untuned" not in name:
            ARTIFACT_DIR.mkdir(exist_ok=True)
            joblib.dump(dict(model=m, name="LightGBM (Optuna-tuned)", features=FEATURES, params=params),
                        ARTIFACT_DIR / "model.joblib")
    comp = pd.DataFrame(rows)
    comp["search_time_s"] = [np.nan, search_s]
    comp.to_csv(RESULTS_DIR / "tuning_comparison.csv", index=False)
    print(comp.drop(columns="params").round(4).to_string(), flush=True)

    # plots
    plots = RESULTS_DIR / "plots"
    v = trials["value"].to_numpy()
    plt.figure(figsize=(6.5, 4))
    plt.scatter(range(len(v)), v, s=18, label="trial")
    plt.plot(np.maximum.accumulate(v), color="tab:red", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 accuracy"); plt.title("Optuna TPE optimisation history"); plt.legend()
    plt.tight_layout(); plt.savefig(plots / "optuna_history.png", dpi=120); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(6.5, 4)); plt.barh(list(imp)[::-1], list(imp.values())[::-1])
        plt.title("Hyperparameter importance (fANOVA)"); plt.tight_layout()
        plt.savefig(plots / "optuna_importance.png", dpi=120); plt.close()
    except Exception as e:  # importance needs >1 distinct value per param
        print("importance plot skipped:", e)
    plot_feature_importance()


def plot_feature_importance():
    """Total gain per feature (split counts favour continuous features with many split points)."""
    booster = joblib.load(ARTIFACT_DIR / "model.joblib")["model"].booster_
    imp = pd.Series(booster.feature_importance("gain"), index=FEATURES)
    imp = imp / imp.sum()
    imp.to_csv(RESULTS_DIR / "feature_importance_gain.csv", header=["gain_share"])
    imp.sort_values().plot.barh(figsize=(6.5, 5), title="Tuned LightGBM feature importance (share of total gain)")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "plots" / "feature_importance.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
