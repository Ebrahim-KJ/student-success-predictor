"""
build_features.py
-----------------
Engineers additional features from the preprocessed student dataset
to improve model predictive power.
"""

import pandas as pd
import numpy as np


def add_parental_education_score(df: pd.DataFrame) -> pd.DataFrame:
    """Average of mother and father education level."""
    df = df.copy()
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2
    return df


def add_alcohol_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combined weekday + weekend alcohol consumption."""
    df = df.copy()
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["alcohol_total"] = df["Dalc"] + df["Walc"]
        df["high_alcohol"] = (df["alcohol_total"] >= 6).astype(int)
    return df


def add_social_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite social risk: high going-out + high alcohol + romantic relationship.
    Research shows these correlate with lower academic performance.
    """
    df = df.copy()
    score = 0
    if "goout" in df.columns:
        score += (df["goout"] >= 4).astype(int)
    if "alcohol_total" in df.columns:
        score += (df["alcohol_total"] >= 5).astype(int)
    if "romantic" in df.columns:
        score += df["romantic"]
    df["social_risk_score"] = score
    return df


def add_support_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite support score: family support + school support + paid classes.
    Higher support → lower risk.
    """
    df = df.copy()
    score = 0
    for col in ["famsup", "schoolsup", "paid"]:
        if col in df.columns:
            score += df[col]
    df["support_score"] = score
    return df


def add_grade_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grade change from period 1 to period 2.
    A declining trajectory is a strong at-risk signal.
    """
    df = df.copy()
    if "G1" in df.columns and "G2" in df.columns:
        df["grade_trajectory"] = df["G2"] - df["G1"]
        df["grade_declining"] = (df["grade_trajectory"] < 0).astype(int)
        df["avg_grade"] = (df["G1"] + df["G2"]) / 2
    return df


def add_study_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Study time relative to free time — a proxy for how students use their time.
    """
    df = df.copy()
    if "studytime" in df.columns and "freetime" in df.columns:
        df["study_efficiency"] = df["studytime"] / (df["freetime"] + 1)
    return df


def add_failure_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag for any prior class failures."""
    df = df.copy()
    if "failures" in df.columns:
        df["has_failed_before"] = (df["failures"] > 0).astype(int)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_parental_education_score(df)
    df = add_alcohol_risk_score(df)
    df = add_social_risk_score(df)
    df = add_support_score(df)
    df = add_grade_trajectory(df)
    df = add_study_efficiency(df)
    df = add_failure_flag(df)

    print(f"✅ Feature engineering complete: {df.shape[1]} total features")
    return df


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

    from src.data.load_data import load_portuguese
    from src.data.preprocess import preprocess

    df_raw = load_portuguese()
    df_processed = preprocess(df_raw, fit=True)
    df_featured = build_all_features(df_processed)

    print("\nNew features added:")
    new_cols = [c for c in df_featured.columns if c not in df_processed.columns]
    print(new_cols)
    print(df_featured[new_cols].describe())
