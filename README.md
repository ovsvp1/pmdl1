# PMLDL Assignment 1 — MLOps Deployment Pipeline

An automated MLOps pipeline that cleans and splits the Titanic dataset, trains a
survival-prediction model (logging metrics to MLflow), and serves it through a
FastAPI model API consumed by a Streamlit web app. All three stages run
automatically every 5 minutes.

## Architecture

```
┌─────────────┐   every 5 min    ┌───────────┐        ┌───────────┐
│  scheduler  │ ───────────────▶ │  models/  │ ◀─────▶ │    api    │ ◀── HTTP ── ┌─────┐
│ (retrains)  │   writes model   │ model.pkl │  reads  │ (FastAPI) │             │ app │
└─────────────┘                  └───────────┘         └───────────┘ ──────────▶ │(Streamlit)│
```

* **`scheduler`** container runs the data engineering + model engineering
  stages in a loop (every 5 minutes by default) and writes the trained model
  to the shared `models/` volume.
* **`api`** container (FastAPI) loads the latest model from the shared
  volume and serves `/predict`. It reloads the model automatically whenever
  the file changes, so it always serves the freshest model without a
  restart.
* **`app`** container (Streamlit) provides input fields, a **Predict**
  button, and displays the prediction returned by the API.

All three containers are started with a single `docker compose` command —
this is what makes the whole pipeline (data → model → deployment) run
automatically on a schedule.

## Repository structure

```
├── code
│   ├── datasets/            # Stage 1: data engineering
│   │   └── data_pipeline.py
│   ├── models/               # Stage 2: model engineering
│   │   └── train.py
│   └── deployment/          # Stage 3: deployment
│       ├── api/              # FastAPI model API + Dockerfile
│       ├── app/               # Streamlit app + Dockerfile
│       ├── scheduler/         # Container that runs stages 1+2 every 5 min
│       └── docker-compose.yml
├── data
│   ├── raw/                  # titanic.csv (input dataset)
│   └── processed/            # train.csv / test.csv (generated)
├── models/                    # model.pkl / metrics.json (generated)
├── notebooks/                 # exploratory notebooks (optional)
└── requirements.txt
```

## Dataset

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
(Kaggle competition dataset, `train.csv`, 891 labeled passengers). Only the
labeled `train.csv` is used — the competition's own `test.csv` has no ground
truth (`gender_submission.csv` is just an example baseline submission, not
real labels), so this pipeline performs its own train/test split from
`train.csv` as required by Stage 1. Task: predict whether a passenger
survived from `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and
`Embarked`.

Data engineering (`code/datasets/data_pipeline.py`):
1. Load `data/raw/titanic.csv`.
2. Impute missing `Age`, `Fare` (median) and `Embarked` (mode).
3. Remove outliers in `Age` and `Fare` using the IQR rule.
4. Stratified 80/20 train/test split, saved to `data/processed/`.

## Model

Model engineering (`code/models/train.py`):
1. Feature engineering with a `ColumnTransformer` (scaling numeric
   features, one-hot encoding categorical features).
2. Trains a `RandomForestClassifier`.
3. Evaluates accuracy, precision, recall, and F1 on the test split.
4. Logs parameters, metrics, and the model to MLflow (`./mlruns`).
5. Packages the fitted pipeline to `models/model.pkl` and metrics to
   `models/metrics.json`.

## Running the pipeline

**Requirements:** Docker and Docker Compose.

```bash
git clone <this-repo-url>
cd pmldl_assignment1
docker compose -f code/deployment/docker-compose.yml up --build
```

This starts all three containers. On first startup the `scheduler`
container immediately runs the data + training pipeline (this takes a few
seconds), after which the `api` container starts serving predictions and
the `scheduler` re-runs the pipeline every 5 minutes indefinitely
(configurable via the `PIPELINE_INTERVAL_SECONDS` environment variable in
`docker-compose.yml`; increase it if a run takes longer than 5 minutes).

* **Web app:** http://localhost:8501
* **Model API:** http://localhost:8000 (interactive docs at
  http://localhost:8000/docs)

To stop everything:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

### Running the pipeline stages locally (without Docker)

Useful for development/debugging:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python code/datasets/data_pipeline.py
python code/models/train.py

# inspect MLflow runs
mlflow ui --backend-store-uri ./mlruns
```

## Notes

* The trained model (`models/model.pkl`) and MLflow run data (`mlruns/`)
  are not committed to the repository — they are regenerated automatically
  by the `scheduler` container (or by running the stages locally as shown
  above).
* The API reloads the model file automatically whenever it changes, so the
  `api` container never needs to be rebuilt or restarted after a
  retraining run.
