# PMLDL Assignment 1 — MLOps Deployment Pipeline

An automated MLOps pipeline that cleans and splits the Titanic dataset, trains a
survival-prediction model (logging metrics to MLflow), and serves it through a
FastAPI model API consumed by a Streamlit web app. All three stages run
automatically every 5 minutes.

## Architecture

```
                       every 5 min
┌───────────────┐  runs   ┌────────────────┐  data + train   ┌───────────┐        ┌───────────┐
│  scheduler.py │ ──────▶ │ run_pipeline.sh│ ──────────────▶ │  models/  │ ◀─────▶│    api    │ ◀── HTTP ── ┌─────┐
│ (host process)│         │ (stages 1 + 2) │  writes model   │ model.pkl │  reads │ (FastAPI) │             │ app │
└───────────────┘         └────────────────┘                 └───────────┘        └───────────┘ ──────────▶ │(Streamlit)│
```

* **`scheduler.py`** is the automation entry point — it runs the complete
  pipeline immediately on startup, then every 5 minutes for as long as the
  process is running.
* **`run_pipeline.sh`** runs the data engineering (Stage 1) and model
  engineering (Stage 2) scripts, then makes sure the API/app containers are
  up (`docker compose up -d`).
* **`api`** container (FastAPI) loads the latest model from the shared
  `models/` volume and serves `/predict` (single passenger) and
  `/predict_batch` (a list of passengers, used by the CSV upload feature).
  It reloads the model automatically whenever the file changes, so it
  always serves the freshest model without a restart.
* **`app`** container (Streamlit) has two tabs: **Single passenger** (input
  fields + a **Predict** button showing the result) and **Upload CSV**
  (upload a dataset, run predictions for every row, and download the
  results as a CSV file).

The API and the app (Stage 3) always run in **separate containers**, as
required. They are built from the same image (`code/deployment/Dockerfile`,
containing both `main.py` and `app.py` plus the combined dependencies) —
`docker-compose.yml` just runs a different `command:` in each container
(`uvicorn` for the API, `streamlit run` for the app). One Dockerfile, two
containers.

Stages 1 and 2 run as plain Python scripts orchestrated by `run_pipeline.sh`.
`scheduler.py` uses the lightweight Python `schedule` library to invoke the complete pipeline every five minutes, so the scheduling logic is version-controlled with the project and is easy to demonstrate.

## Repository structure

```
├── code
│   ├── datasets/            # Stage 1: data engineering
│   │   └── data_pipeline.py
│   ├── models/               # Stage 2: model engineering
│   │   └── train.py
│   └── deployment/          # Stage 3: deployment
│       ├── api/              # FastAPI model API (main.py)
│       ├── app/               # Streamlit app (app.py)
│       ├── Dockerfile         # single image, shared by both containers
│       ├── requirements.txt
│       └── docker-compose.yml
├── data
│   ├── raw/                  # titanic.csv (input dataset)
│   └── processed/            # train.csv / test.csv (generated)
├── models/                    # model.pkl / metrics.json (generated)
├── notebooks/                 # exploratory notebooks (optional)
├── scheduler.py                # automatically runs the full pipeline every 5 min
├── run_pipeline.sh             # runs stages 1+2 and (re)deploys stage 3
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

**Requirements:** Python 3.11+, Docker, and Docker Compose.

```bash
git clone <this-repo-url>
cd pmldl_assignment1

python3 -m venv venv
./venv/bin/pip install -r requirements.txt

./run_pipeline.sh   # runs stages 1+2 once and starts the API + app
```

* **Web app:** http://localhost:8501
* **Model API:** http://localhost:8000 (interactive docs at
  http://localhost:8000/docs)

### Scheduling it to run automatically every 5 minutes

Start the repository's Python scheduler:

```bash
./venv/bin/python scheduler.py
```

The scheduler runs `run_pipeline.sh` immediately and then launches another complete pipeline run every 5 minutes while the scheduler process is running. This means the scheduling implementation is part of the repository and does not require a separate cron configuration.

If a pipeline run takes longer than the configured interval, increase the interval in `scheduler.py`, as permitted by the assignment. Stop the scheduler with `Ctrl+C`.

To stop the deployed containers:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

### Inspecting MLflow runs

```bash
./venv/bin/mlflow ui --backend-store-uri ./mlruns
```

## Notes

* The trained model (`models/model.pkl`) and MLflow run data (`mlruns/`)
  are not committed to the repository — they are regenerated by
  `run_pipeline.sh`.
* The API reloads the model file automatically whenever it changes, so it
  never needs to be rebuilt or restarted after a retraining run —
  `run_pipeline.sh` only calls `docker compose up -d`, which is a no-op for
  already-running containers.
# pmdl1
