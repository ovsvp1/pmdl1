"""Stage 1: Data engineering — load, clean, and split the Titanic dataset."""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "titanic.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET_COLUMN = "Survived"


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Impute missing values.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Remove outliers in numeric columns using the IQR rule.
    for column in ["Age", "Fare"]:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df[(df[column] >= lower) & (df[column] <= upper)]

    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET_COLUMN]
    )
    return train_df, test_df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data()
    df = clean_data(df)
    train_df, test_df = split_data(df)

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Loaded {len(df)} clean rows from {RAW_PATH}")
    print(f"Saved {len(train_df)} training rows to {train_path}")
    print(f"Saved {len(test_df)} testing rows to {test_path}")


if __name__ == "__main__":
    main()
