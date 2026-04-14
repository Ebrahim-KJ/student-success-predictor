"""
train.py
--------
Trains multiple ML models on the student success dataset,
tracks all experiments with MLflow, and saves the best model.
"""

import os
import sys
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.load_data import load_portuguese
from src.data.preprocess import preprocess, get_features_and_target
from src.features.build_features import build_all_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "student-success-predictor"


def get_models() -> dict:
    """Define all models to train."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0
        ),
    }


def train_and_log(model, model_name: str, X_train, X_test, y_train, y_test, scaler=None):
    """Train a single model and log everything to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Fit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "cv_roc_auc": cv_auc
        })
        mlflow.sklearn.log_model(model, model_name)

        print(f"\n{'='*50}")
        print(f"  {model_name.upper()}")
        print(f"{'='*50}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  CV AUC   : {cv_auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Not At Risk', 'At Risk'])}")

        return model, auc


def run_training():
    """Full training pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load and prepare data
    print("📥 Loading data...")
    df_raw = load_portuguese()
    df_processed = preprocess(df_raw, fit=True)
    df_featured = build_all_features(df_processed)

    X, y = get_features_and_target(df_featured)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # Save feature names for Streamlit app
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_names.joblib"))

    # Save train/test splits
    joblib.dump((X_train, X_test, y_train, y_test),
                os.path.join(MODELS_DIR, "train_test_split.joblib"))

    # MLflow setup
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    models = get_models()
    results = {}

    print("\n🚀 Training models...\n")
    for name, model in models.items():
        # Use scaled data for logistic regression
        if name == "logistic_regression":
            trained_model, auc = train_and_log(
                model, name, X_train_scaled, X_test_scaled, y_train, y_test
            )
        else:
            trained_model, auc = train_and_log(
                model, name, X_train, X_test, y_train, y_test
            )
        results[name] = (trained_model, auc)
        joblib.dump(trained_model, os.path.join(MODELS_DIR, f"{name}.joblib"))

    # Save the best model (by AUC)
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.joblib"))

    print(f"\n🏆 Best model: {best_name} (AUC: {results[best_name][1]:.4f})")
    print(f"✅ All models saved to {MODELS_DIR}")
    print("📊 Run `mlflow ui` to view experiment tracking dashboard")

    return results


if __name__ == "__main__":
    run_training()
