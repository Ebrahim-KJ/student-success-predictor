"""
Hyperparameter Optimisation for Student Success Predictor
Uses Optuna (Bayesian + evolutionary search) — directly extends thesis work
on Genetic Algorithms for Hyperparameter Optimisation.
"""

import optuna
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def optimise_xgboost(X_train, y_train, n_trials: int = 50) -> dict:
    """
    Bayesian hyperparameter search for XGBoost.
    Uses TPE sampler (Tree-structured Parzen Estimator).
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 5),
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
            "use_label_encoder": False,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring='roc_auc', n_jobs=-1
        )
        return scores.mean()

    logger.info(f"Running Optuna optimisation: {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best AUC (CV): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params, study


def optimise_random_forest(X_train, y_train, n_trials: int = 30) -> dict:
    """Bayesian search for Random Forest hyperparameters."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring='roc_auc', n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study


def build_optimised_xgboost(best_params: dict) -> XGBClassifier:
    """Instantiate XGBoost with optimised hyperparameters."""
    return XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        use_label_encoder=False
    )


def get_optimisation_history(study) -> list:
    """Return trial history for plotting in dashboard."""
    history = []
    best_so_far = -np.inf
    for trial in study.trials:
        if trial.value is not None:
            best_so_far = max(best_so_far, trial.value)
        history.append({
            "trial": trial.number,
            "value": trial.value,
            "best_so_far": best_so_far
        })
    return history
