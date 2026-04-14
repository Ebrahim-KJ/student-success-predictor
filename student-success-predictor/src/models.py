"""
Model Training & Evaluation for Student Success Predictor
Trains multiple classifiers and returns ranked results.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import os
import logging

logger = logging.getLogger(__name__)


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight='balanced', n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=42, learning_rate=0.05
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, random_state=42, eval_metric='logloss',
        scale_pos_weight=3, use_label_encoder=False,
        verbosity=0
    ),
}


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute full evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
    return metrics, y_prob


def train_all_models(X_train, X_test, y_train, y_test,
                     experiment_name: str = "student_dropout") -> tuple:
    """
    Train all models, log to MLflow, return results DataFrame
    and the best model.
    """
    mlflow.set_experiment(experiment_name)
    results = []
    trained_models = {}
    best_auc = 0
    best_model = None
    best_model_name = None

    for name, model in MODELS.items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics, y_prob = evaluate_model(model, X_test, y_test, name)

            # Log to MLflow
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                 if isinstance(v, float)})
            mlflow.sklearn.log_model(model, name.replace(" ", "_"))

            results.append(metrics)
            trained_models[name] = (model, y_prob)

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_model_name = name

            logger.info(f"{name} — AUC: {metrics['roc_auc']} | F1: {metrics['f1']}")

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    logger.info(f"\nBest model: {best_model_name} (AUC: {best_auc})")
    return results_df, trained_models, best_model, best_model_name


def save_model(model, path: str = "models/best_model.pkl"):
    """Persist best model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str = "models/best_model.pkl"):
    """Load persisted model."""
    return joblib.load(path)
