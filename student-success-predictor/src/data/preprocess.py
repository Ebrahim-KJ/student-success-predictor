"""
preprocess.py
-------------
Cleans the raw UCI student dataset, encodes categorical variables,
and creates the binary target label (at-risk vs. not at-risk).
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")

# Students with final grade < 10 are considered at-risk (fail threshold in Portuguese system)
FAIL_THRESHOLD = 10

BINARY_COLS = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian"
]

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2"
]


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yes/no columns to 1/0."""
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Label encode categorical columns."""
    df = df.copy()
    os.makedirs(MODELS_DIR, exist_ok=True)

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        encoder_path = os.path.join(MODELS_DIR, f"encoder_{col}.joblib")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, encoder_path)
        else:
            le = joblib.load(encoder_path)
            df[col] = le.transform(df[col].astype(str))
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: 1 = at risk (G3 < 10), 0 = not at risk.
    Drop G3 to avoid data leakage. Keep G1, G2 as predictive features.
    """
    df = df.copy()
    df["at_risk"] = (df["G3"] < FAIL_THRESHOLD).astype(int)
    df = df.drop(columns=["G3"])
    return df


def preprocess(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = encode_binary(df)
    df = encode_categoricals(df, fit=fit)
    df = create_target(df)

    # Drop any remaining nulls
    df = df.dropna()

    print(f"✅ Preprocessed: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   At-risk students: {df['at_risk'].sum()} ({df['at_risk'].mean()*100:.1f}%)")

    return df


def get_features_and_target(df: pd.DataFrame):
    """Split into X and y."""
    X = df.drop(columns=["at_risk"])
    y = df["at_risk"]
    return X, y


def save_processed(df: pd.DataFrame, filename: str = "processed.csv"):
    """Save processed dataframe."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved processed data to {path}")


if __name__ == "__main__":
    from load_data import load_portuguese
    df_raw = load_portuguese()
    df_processed = preprocess(df_raw, fit=True)
    save_processed(df_processed)
    print(df_processed.head())
