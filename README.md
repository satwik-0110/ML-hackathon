# PathSnap: map-matching and route reconstruction

PathSnap matches noisy GPS tracks to a road network. For every GPS observation it predicts the
directed road segment (arc) the vehicle was on, and it reconstructs the full connected route.

```
GPS track + road network
   │  local metric projection per trip, cleaning
   ▼
Candidate generation      K ≤ 10 nearest directed arcs within 50 m (widened to 200 m if empty)
   ▼
Feature engineering       18 features per (observation, candidate) pair
   ▼
Classifier (LightGBM)     P(candidate is the true arc), Optuna-tuned
   ▼
HMM + Viterbi             emission = classifier probability, transition = route vs GPS distance
   ▼
Route stitching           shortest paths between matched arcs → continuous arc sequence
```

## Repository layout

| Path | Content |
|---|---|
| `pathsnap/` | Core library: `io` (raw files), `geo` (projection), `network` (graph, spatial index, Dijkstra), `labels` (per-point labels from the route), `features` (candidates + features), `hmm` (lattice, Viterbi, stitching), `metrics`, `pipeline` (end-to-end inference), `viz` (Folium) |
| `scripts/01_eda.py` | Dataset exploration: inventory, quality checks, connectivity, plots, Folium maps |
| `scripts/02_build_dataset.py` | Label derivation, candidate generation and features → `cache/*.parquet` |
| `scripts/03_train_models.py` | 7 models, 5-fold GroupKFold by trip, student-test comparison, confusion matrices, learning curves |
| `scripts/04_bayes_opt.py` | Optuna TPE search on LightGBM, tuned vs untuned comparison, exports `artifacts/model.joblib` |
| `scripts/05_hmm.py` | Out-of-fold emissions, HMM parameter grid, classifier vs HMM evaluation, exports `artifacts/hmm_config.json` |
| `scripts/06_maps.py` | Folium maps of predictions on the student-test trips |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/app.py` | Streamlit evaluator UI (Laptop 2) |
| `artifacts/` | Final model and HMM configuration used by the API |
| `results/` | CSV results, logs, plots (`results/plots`) and interactive maps (`results/maps`) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # Laptop 2 only needs: pip install -r requirements-ui.txt
```

The raw data is not committed. By default the code expects it in `../data/Student folder/{train,student_test}`.
To use another location, set `PATHSNAP_DATA_DIR`.

## Reproducing the experiments

```bash
python scripts/01_eda.py            # ~2 min
python scripts/02_build_dataset.py  # ~1 min  -> cache/
python scripts/03_train_models.py   # ~15 min (KNN and RandomForest are the slow ones)
python scripts/04_bayes_opt.py      # ~15 min (30 Optuna trials x 5 folds)
python scripts/05_hmm.py            # ~3 min
python scripts/06_maps.py           # ~1 min
```

All random seeds are fixed (`SEED = 42` in `pathsnap/config.py`).

## Results

All numbers below come from the CSVs in `results/`.

**Model comparison**

- Training data: 78 training trips, every 5th point, 5-fold GroupKFold.
- Test data: all points of the 10 student-test trips.
- Source: `results/model_comparison.csv`.

| Model | CV macro-F1 | CV Top-1 | Test Top-1 | Test macro-F1 | Test log loss | ms / 1k rows |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.909 | 0.870 | 0.858 | 0.906 | 0.136 | 0.1 |
| Gaussian NB | 0.786 | 0.840 | 0.818 | 0.788 | 0.751 | 0.2 |
| KNN (k=50) | 0.916 | 0.872 | 0.865 | 0.914 | 0.128 | 607 |
| Random Forest | 0.931 | 0.888 | 0.886 | 0.932 | 0.086 | 9.0 |
| Extra Trees | 0.925 | 0.881 | 0.872 | 0.923 | 0.094 | 11.9 |
| **LightGBM** | **0.933** | **0.892** | **0.896** | **0.935** | **0.081** | 11.2 |
| MLP (64-32) | 0.924 | 0.884 | 0.883 | 0.927 | 0.090 | 0.4 |

**Bayesian optimisation** (Optuna TPE, 30 trials; `results/tuning_comparison.csv`)

- CV Top-1 is 0.8916 for both the tuned and the untuned model.
- Test Top-1 is 0.8951 tuned and 0.8958 untuned.
- The difference is far inside the fold spread. Performance is at a plateau, and the tuned model is the more regularised of the two.

**HMM / Viterbi** (β = 5, w_emit = 1; `results/hmm_student_test.csv`)

| Student test | Classifier only | HMM, distance-only emission | Classifier + HMM |
|---|---|---|---|
| Point Top-1 | 0.895 | 0.909 | **0.933** |
| Route continuity | 0.782 | 0.999 | **0.999** |
| Route F1 (length-weighted) | 0.939 | 0.971 | **0.977** |

## Deployment (two laptops)

**Laptop 1: model + API**

```bash
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_trips     # one folder per trip: <id>\<id>.track/.nodes/.arcs/.route
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Find the IP with `ipconfig`.
- Allow inbound TCP 8000 in the Windows firewall, for example (admin):
  `netsh advfirewall firewall add rule name="PathSnap" dir=in action=allow protocol=TCP localport=8000`
- Every request is logged in this console: trip, sizes, timings, metrics.

**Laptop 2: UI**

```bash
set PATHSNAP_API_URL=http://<laptop-1-ip>:8000
streamlit run ui/app.py
```

The UI lists the trips served by `GET /trips` and sends the selected id to `POST /predict`. It then shows
the predicted route on a map together with the metrics computed by the API.

### API

| Endpoint | Description |
|---|---|
| `GET /health` | Status, loaded model, number of evaluation trips |
| `GET /trips` | Trip ids with number of points, duration and centre (no reference data) |
| `POST /predict` `{"trip_id": "..."}` | Runs the full pipeline. Returns the predicted route polyline and arc ids, the GPS points, classifier-only arcs, metrics, diagnostics and timings |

## Leakage and integrity safeguards

- **Prediction never reads the reference route.** `load_trip(..., with_route=False)` is used for
  inference. The `.route` file is only read by `evaluate_prediction` on the API server.
- **The UI only receives metrics and predictions.** It never receives reference routes or labels.
- **Training never touches the student-test or evaluator trips.** Models, Optuna and HMM decoder
  parameters were all fitted on training trips only, using GroupKFold by trip. The student-test trips
  were used only to compare models.
- **No trip-specific logic or hardcoded outputs.** The API processes any trip folder it is given.
