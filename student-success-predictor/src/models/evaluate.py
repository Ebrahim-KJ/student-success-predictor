"""
evaluate.py
-----------
Generates evaluation plots, SHAP explainability charts,
and a full model comparison report.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    RocCurveDisplay, ConfusionMatrixDisplay,
    roc_auc_score, f1_score, accuracy_score
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "../../plots")


def load_artefacts():
    """Load saved models, scaler, and data splits."""
    X_train, X_test, y_train, y_test = joblib.load(
        os.path.join(MODELS_DIR, "train_test_split.joblib")
    )
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    best_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.joblib"))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    return X_train, X_test, y_train, y_test, feature_names, best_model, best_name, xgb_model, rf_model


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    """Plot and save confusion matrix."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Not At Risk", "At Risk"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved: {path}")


def plot_roc_curves(models_dict: dict, X_test, y_test):
    """Plot ROC curves for all models on one chart."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved: {path}")


def plot_shap_summary(model, X_train, feature_names: list, model_name: str):
    """Generate SHAP beeswarm plot showing global feature importance."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"🔍 Computing SHAP values for {model_name}...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Handle multi-output (some models return list)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_train,
        feature_names=feature_names,
        show=False, plot_type="beeswarm"
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"shap_summary_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Bar chart of tree-based feature importances."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(indices)), importances[indices], color="steelblue")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Top 15 Feature Importances — {model_name}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✅ Saved: {path}")


def print_model_comparison(models_dict: dict, X_test, y_test):
    """Print a formatted comparison table of all models."""
    print("\n" + "="*65)
    print(f"  {'MODEL':<25} {'ACCURACY':>10} {'F1':>10} {'AUC':>10}")
    print("="*65)
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"  {name:<25} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
    print("="*65)


def run_evaluation():
    """Full evaluation pipeline."""
    X_train, X_test, y_train, y_test, feature_names, best_model, best_name, xgb_model, rf_model = load_artefacts()

    models = {
        "xgboost": xgb_model,
        "random_forest": rf_model,
    }

    print("\n📊 Model Comparison:")
    print_model_comparison(models, X_test, y_test)

    print("\n📈 Generating plots...")
    plot_roc_curves(models, X_test, y_test)

    for name, model in models.items():
        plot_confusion_matrix(model, X_test, y_test, name)
        plot_feature_importance(model, feature_names, name)

    # SHAP on best tree model
    print("\n🔍 SHAP Analysis...")
    plot_shap_summary(xgb_model, X_train, feature_names, "xgboost")
    plot_shap_summary(rf_model, X_train, feature_names, "random_forest")

    print(f"\n✅ All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    run_evaluation()
