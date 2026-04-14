"""
hyperparameter_opt.py
---------------------
Hyperparameter optimisation using two approaches:
  1. Custom Genetic Algorithm (inspired by Master's thesis work)
  2. Optuna (Tree-structured Parzen Estimator — probabilistic Bayesian optimisation)

Compares both approaches and logs results to MLflow.
"""

import os
import sys
import random
import time
import joblib
import numpy as np
import mlflow
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.load_data import load_portuguese
from src.data.preprocess import preprocess, get_features_and_target
from src.features.build_features import build_all_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")
RANDOM_STATE = 42
CV_FOLDS = 5


def get_X_y():
    """Load data pipeline and return X, y."""
    df_raw = load_portuguese()
    df_processed = preprocess(df_raw, fit=False)
    df_featured = build_all_features(df_processed)
    return get_features_and_target(df_featured)


def evaluate_xgb(params: dict, X, y) -> float:
    """Evaluate an XGBoost config using cross-validated AUC."""
    model = xgb.XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()


# ─────────────────────────────────────────────
# 1. GENETIC ALGORITHM
# ─────────────────────────────────────────────

PARAM_SPACE = {
    "n_estimators":    [50, 100, 150, 200, 300],
    "max_depth":       [3, 4, 5, 6, 7, 8],
    "learning_rate":   [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":[1, 3, 5, 7],
    "scale_pos_weight":[1, 2, 3, 4],
}


def random_individual() -> dict:
    """Generate a random hyperparameter configuration (individual)."""
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}


def crossover(parent1: dict, parent2: dict) -> dict:
    """Single-point crossover between two parents."""
    keys = list(parent1.keys())
    point = random.randint(1, len(keys) - 1)
    child = {}
    for i, k in enumerate(keys):
        child[k] = parent1[k] if i < point else parent2[k]
    return child


def mutate(individual: dict, mutation_rate: float = 0.2) -> dict:
    """Randomly mutate each gene with probability mutation_rate."""
    mutated = individual.copy()
    for k in mutated:
        if random.random() < mutation_rate:
            mutated[k] = random.choice(PARAM_SPACE[k])
    return mutated


def genetic_algorithm(X, y,
                       population_size: int = 10,
                       generations: int = 8,
                       elite_size: int = 2,
                       mutation_rate: float = 0.2):
    """
    Genetic Algorithm for hyperparameter optimisation.

    Inspired by the Master's thesis: 'Genetic Algorithms for Hyperparameter Optimization'.
    Evolves a population of XGBoost hyperparameter configurations over multiple generations,
    selecting the fittest individuals (by ROC-AUC) to produce the next generation.
    """
    print("\n🧬 Running Genetic Algorithm...")
    print(f"   Population: {population_size} | Generations: {generations} | Mutation: {mutation_rate}")

    population = [random_individual() for _ in range(population_size)]
    best_score = 0
    best_params = None
    history = []

    for gen in range(generations):
        # Evaluate fitness
        fitness = [(ind, evaluate_xgb(ind, X, y)) for ind in population]
        fitness.sort(key=lambda x: x[1], reverse=True)

        gen_best = fitness[0][1]
        history.append(gen_best)

        if gen_best > best_score:
            best_score = gen_best
            best_params = fitness[0][0]

        print(f"   Gen {gen+1:02d}/{generations} | Best AUC: {gen_best:.4f} | Pop Best: {fitness[0][1]:.4f}")

        # Elitism — carry top individuals forward
        next_gen = [f[0] for f in fitness[:elite_size]]

        # Fill rest of population with children
        while len(next_gen) < population_size:
            # Tournament selection
            p1 = random.choice(fitness[:5])[0]
            p2 = random.choice(fitness[:5])[0]
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            next_gen.append(child)

        population = next_gen

    print(f"\n🏆 GA Best AUC: {best_score:.4f}")
    print(f"   Best params: {best_params}")
    return best_params, best_score, history


# ─────────────────────────────────────────────
# 2. OPTUNA (BAYESIAN / TPE)
# ─────────────────────────────────────────────

def optuna_objective(trial, X, y):
    """Optuna objective function for XGBoost."""
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 4),
    }
    return evaluate_xgb(params, X, y)


def run_optuna(X, y, n_trials: int = 50):
    """Run Optuna TPE optimisation."""
    print("\n🔬 Running Optuna (TPE / Bayesian)...")
    print(f"   Trials: {n_trials}")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\n🏆 Optuna Best AUC: {best_score:.4f}")
    print(f"   Best params: {best_params}")
    return best_params, best_score, study


# ─────────────────────────────────────────────
# COMPARISON & SAVING
# ─────────────────────────────────────────────

def run_optimisation():
    """Run both methods, compare, log to MLflow, and save best model."""
    X, y = get_X_y()

    mlflow.set_experiment("hyperparameter-optimisation")

    # Genetic Algorithm
    start = time.time()
    ga_params, ga_score, ga_history = genetic_algorithm(X, y, population_size=10, generations=8)
    ga_time = time.time() - start

    # Optuna
    start = time.time()
    optuna_params, optuna_score, optuna_study = run_optuna(X, y, n_trials=50)
    optuna_time = time.time() - start

    # Compare
    print("\n" + "="*55)
    print(f"  {'METHOD':<25} {'AUC':>10} {'TIME (s)':>12}")
    print("="*55)
    print(f"  {'Genetic Algorithm':<25} {ga_score:>10.4f} {ga_time:>12.1f}")
    print(f"  {'Optuna (TPE)':<25} {optuna_score:>10.4f} {optuna_time:>12.1f}")
    print("="*55)

    # Log to MLflow
    with mlflow.start_run(run_name="genetic_algorithm"):
        mlflow.log_metric("best_cv_auc", ga_score)
        mlflow.log_metric("time_seconds", ga_time)
        mlflow.log_params(ga_params)

    with mlflow.start_run(run_name="optuna_tpe"):
        mlflow.log_metric("best_cv_auc", optuna_score)
        mlflow.log_metric("time_seconds", optuna_time)
        mlflow.log_params(optuna_params)

    # Save the best optimised model
    best_params = ga_params if ga_score >= optuna_score else optuna_params
    best_score = max(ga_score, optuna_score)
    best_method = "Genetic Algorithm" if ga_score >= optuna_score else "Optuna (TPE)"

    print(f"\n🏆 Overall winner: {best_method} (AUC: {best_score:.4f})")

    optimised_model = xgb.XGBClassifier(
        **best_params, random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0
    )
    optimised_model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(optimised_model, os.path.join(MODELS_DIR, "optimised_xgboost.joblib"))
    print(f"✅ Optimised model saved to {MODELS_DIR}/optimised_xgboost.joblib")


if __name__ == "__main__":
    run_optimisation()
