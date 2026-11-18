"""Export the comparison models so the API can serve any of them (model switch in the UI).

Copies the fitted models written by scripts/03_train_models.py (cache/models/) to
artifacts/models/ with compression. The final tuned LightGBM stays in artifacts/model.joblib.
Also writes artifacts/models/index.json with the display name, family and student-test
Top-1 accuracy of each model (taken from results/model_comparison.csv).
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.config import ARTIFACT_DIR, CACHE_DIR, RESULTS_DIR  # noqa: E402

# key -> (file written by 03_train_models.py, display name)
MODELS = {
    "random_forest": ("RandomForest", "Random Forest"),
    "mlp": ("MLP", "MLP neural network"),
    "extra_trees": ("ExtraTrees", "Extra Trees"),
    "knn": ("KNN", "K-Nearest Neighbours"),
    "logreg": ("LogisticRegression", "Logistic Regression"),
    "gnb": ("GaussianNB", "Gaussian Naive Bayes"),
}


def main():
    out = ARTIFACT_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv").set_index("model")
    tuned = pd.read_csv(RESULTS_DIR / "tuning_comparison.csv").iloc[-1]
    index = [dict(key="lightgbm", name="LightGBM (tuned)", family="Boosting", file="../model.joblib",
                  student_test_top1=round(float(tuned["test_top1"]), 4), default=True)]
    for key, (stem, name) in MODELS.items():
        model = joblib.load(CACHE_DIR / "models" / f"{stem}.joblib")
        path = out / f"{key}.joblib"
        joblib.dump(model, path, compress=("zlib", 6))
        index.append(dict(key=key, name=name, family=comp.loc[stem, "family"], file=path.name,
                          student_test_top1=round(float(comp.loc[stem, "test_top1"]), 4), default=False))
        print(f"{key:14s} {path.stat().st_size / 1e6:7.1f} MB", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
