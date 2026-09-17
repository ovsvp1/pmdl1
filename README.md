# PMLDL Assignment 1 — MLOps Deployment Pipeline

A fully Dockerized MLOps pipeline for Titanic survival prediction. A
scheduler container runs data engineering and model training immediately
and then every 5 minutes; the FastAPI service automatically reloads the
newly generated model, and the Streamlit app talks to the API. No local
Python environment is required — `docker compose up` runs everything.

## Architecture

```
Docker Compose
├── scheduler container       (Stages 1 + 2, every 5 min)
│   ├── code/datasets/data_pipeline.py
│   └── code/models/train.py         ──▶ writes models/model.pkl
├── api container (FastAPI, port 8000)     ◀── reads models/model.pkl
│   └── /predict, /predict_batch
└── app container (Streamlit, port 8501)   ── HTTP ──▶ api
    ├── "Single passenger" tab (form + Predict button)
    └── "Upload CSV" tab (batch predictions, downloadable as CSV)
```

* **`scheduler`** container runs the data engineering (Stage 1) and model
  engineering (Stage 2) scripts back-to-back, immediately on startup and
  then every 5 minutes for as long as the container runs
  (`docker_scheduler.py`, built from `code/pipeline.Dockerfile`).
* **`api`** container (FastAPI) loads the latest model from the shared
  `models/` volume and serves `/predict` (single passenger) and
  `/predict_batch` (a list of passengers, used by the CSV upload feature).
  It reloads the model automatically whenever the file changes, so it
  always serves the freshest model without a restart.
* **`app`** container (Streamlit) has two tabs: **Single passenger** (input
  fields + a **Predict** button showing the result) and **Upload CSV**
  (upload a dataset, run predictions for every row, and download the
  results as a CSV file).

The `data/`, `models/`, and `mlruns/` directories are mounted as volumes
shared between containers, so a model produced by the scheduler is
immediately visible to the API container without rebuilding or
restarting it.

The API and the app (Stage 3) always run in **separate containers**, as
required. They are built from the same image (`code/deployment/Dockerfile`,
containing both `main.py` and `app.py` plus the combined dependencies) —
`docker-compose.yml` just runs a different `command:` in each container
(`uvicorn` for the API, `streamlit run` for the app). One Dockerfile, two
containers.

## Repository structure

```
├── code
│   ├── datasets/            # Stage 1: data engineering
│   │   └── data_pipeline.py
│   ├── models/               # Stage 2: model engineering
│   │   └── train.py
│   ├── pipeline.Dockerfile   # scheduler container image
│   └── deployment/          # Stage 3: deployment
│       ├── api/              # FastAPI model API (main.py)
│       ├── app/               # Streamlit app (app.py)
│       ├── Dockerfile         # single image, shared by api + app
│       ├── requirements.txt
│       └── docker-compose.yml
├── data
│   ├── raw/                  # titanic.csv (input dataset)
│   └── processed/            # train.csv / test.csv (generated)
├── models/                    # model.pkl / metrics.json (generated)
├── notebooks/                 # exploratory notebooks (optional)
├── docker_scheduler.py         # runs stages 1+2 every 5 min (inside the scheduler container)
├── run_pipeline.sh             # convenience wrapper: docker compose up -d --build
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

**Requirements:** Docker and Docker Compose. No local Python environment
or `pip install` is required — everything runs inside containers.

```bash
git clone <this-repo-url>
cd pmldl_assignment1

./run_pipeline.sh   # docker compose up -d --build
```

* **Web app:** http://localhost:8501
* **Model API:** http://localhost:8000 (interactive docs at
  http://localhost:8000/docs)

The scheduler container starts training immediately; the first model is
usually ready within a few seconds. Automation is built in — as long as
the containers are running, the scheduler retrains every 5 minutes with
no extra setup (no host cron, no separate process to start).

Check services:

```bash
docker compose -f code/deployment/docker-compose.yml ps
```

Watch automated pipeline runs:

```bash
docker compose -f code/deployment/docker-compose.yml logs -f scheduler
```

If a pipeline run ever takes longer than 5 minutes, increase
`INTERVAL_SECONDS` in `docker_scheduler.py` accordingly.

Stop everything:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

### Inspecting MLflow runs

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/mlflow ui --backend-store-uri ./mlruns
```

## Notes

* The trained model (`models/model.pkl`) and MLflow run data (`mlruns/`)
  are not committed to the repository — they are regenerated automatically
  by the `scheduler` container.
* The API reloads the model file automatically whenever it changes, so it
  never needs to be rebuilt or restarted after a retraining run.
# pmdl1
