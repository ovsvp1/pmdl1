"""Stage 2: Model engineering — feature engineering, training, evaluation, packaging."""
import json
import os

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
TARGET_COLUMN = "Survived"


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, y_train = train_df[feature_columns], train_df[TARGET_COLUMN]
    x_test, y_test = test_df[feature_columns], test_df[TARGET_COLUMN]

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("titanic-survival")

    with mlflow.start_run():
        pipeline = build_pipeline()
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
        }

        mlflow.log_params({"n_estimators": 200, "max_depth": 6})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        model_path = os.path.join(MODELS_DIR, "model.pkl")
        joblib.dump(pipeline, model_path)

        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Trained model saved to {model_path}")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
