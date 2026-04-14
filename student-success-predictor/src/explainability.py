"""
Explainability Module — SHAP-based model interpretation
Explains WHY a student is flagged as at-risk.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
import os

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    'Tuition_fees_up_to_date': 'Tuition Up to Date',
    'Curricular_units_2nd_sem_grade': '2nd Sem Grade',
    'Curricular_units_1st_sem_grade': '1st Sem Grade',
    'Curricular_units_2nd_sem_approved': '2nd Sem Units Passed',
    'Curricular_units_1st_sem_approved': '1st Sem Units Passed',
    'Admission_grade': 'Admission Grade',
    'Age_at_enrollment': 'Age at Enrollment',
    'Scholarship_holder': 'Scholarship Holder',
    'Debtor': 'Has Outstanding Debt',
    'avg_grade': 'Average Grade (Both Sems)',
    'total_units_approved': 'Total Units Passed',
    'grade_progression': 'Grade Trend (Sem1→Sem2)',
    'financial_risk': 'Financial Risk Score',
    'Unemployment_rate': 'Local Unemployment Rate',
    'GDP': 'GDP Growth Rate',
    'Gender': 'Gender',
    'Displaced': 'Displaced Student',
    'International': 'International Student',
    'mature_student': 'Mature Student (23+)',
    'parental_edu_avg': 'Parental Education Level',
    'admission_grade_zscore': 'Admission Grade (Relative)',
}


def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values using TreeExplainer (fast for tree-based models)."""
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def get_top_risk_factors(shap_vals: np.ndarray, feature_names: list,
                          top_n: int = 5) -> list:
    """
    For a single student prediction, return the top N features
    driving the risk score, with human-readable labels.
    """
    feature_impacts = list(zip(feature_names, shap_vals))
    sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)

    results = []
    for feat, impact in sorted_impacts[:top_n]:
        label = FEATURE_LABELS.get(feat, feat.replace('_', ' ').title())
        direction = "increases" if impact > 0 else "decreases"
        results.append({
            "feature": label,
            "impact": round(float(impact), 4),
            "direction": direction,
            "magnitude": abs(round(float(impact), 4))
        })
    return results


def plot_global_importance(shap_values, X: pd.DataFrame,
                           save_path: str = "plots/shap_global.png"):
    """Generate and save global SHAP feature importance bar chart."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    mean_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.Series(mean_shap, index=X.columns)
    feat_importance = feat_importance.sort_values(ascending=True).tail(15)
    feat_importance.index = [FEATURE_LABELS.get(i, i) for i in feat_importance.index]

    colors = ['#e74c3c' if v > feat_importance.median() else '#3498db'
              for v in feat_importance.values]
    feat_importance.plot(kind='barh', ax=ax, color=colors)

    ax.set_title("Top Features Driving Dropout Risk\n(Global Model Explanation)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Mean |SHAP Value| — Average Impact on Dropout Prediction")
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved global SHAP plot to {save_path}")
    return save_path


def plot_shap_waterfall_for_student(explainer, X_student: pd.DataFrame,
                                     save_path: str = "plots/shap_waterfall.png"):
    """Generate waterfall plot for a single student prediction."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shap_vals = explainer.shap_values(X_student)

    fig, ax = plt.subplots(figsize=(10, 6))
    vals = shap_vals[0]
    feats = X_student.columns.tolist()

    # Sort by impact
    pairs = sorted(zip(feats, vals), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels = [FEATURE_LABELS.get(f, f) for f, _ in pairs]
    values = [v for _, v in pairs]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title("What's Driving This Student's Risk?\n(Individual SHAP Explanation)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("SHAP Value (positive = increases dropout risk)")
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path
