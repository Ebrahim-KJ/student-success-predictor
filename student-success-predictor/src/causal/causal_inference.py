"""
causal_inference.py
-------------------
Uses DoWhy to estimate the causal effect of interventions on student outcomes.

Goes beyond correlation — answers questions like:
  "If a student received extra tutoring (schoolsup=1), how much would
   their probability of being at-risk decrease, on average?"

This ties directly to the Cornell MSc curriculum in Causal Inference
& Program Evaluation, and differentiates this project from standard ML work.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.load_data import load_portuguese
from src.data.preprocess import preprocess, get_features_and_target
from src.features.build_features import build_all_features

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "../../plots")

# Interventions to evaluate
INTERVENTIONS = [
    {
        "name": "School Support (Extra Tutoring)",
        "treatment": "schoolsup",
        "description": "Does school-provided extra tutoring reduce at-risk probability?"
    },
    {
        "name": "Family Support",
        "treatment": "famsup",
        "description": "Does family educational support reduce at-risk probability?"
    },
    {
        "name": "Higher Education Aspiration",
        "treatment": "higher",
        "description": "Does wanting to pursue higher education reduce at-risk probability?"
    },
    {
        "name": "Internet Access at Home",
        "treatment": "internet",
        "description": "Does having internet access reduce at-risk probability?"
    },
]

# Confounders we need to control for
CONFOUNDERS = [
    "age", "Medu", "Fedu", "failures", "studytime",
    "absences", "G1", "G2", "parent_edu_avg", "support_score"
]


def load_data():
    """Load and prepare data for causal analysis."""
    df_raw = load_portuguese()
    df_processed = preprocess(df_raw, fit=False)
    df_featured = build_all_features(df_processed)
    return df_featured


def naive_correlation_analysis(df: pd.DataFrame):
    """
    Naive (unadjusted) comparison of at-risk rates between treatment groups.
    This is what non-causal analysts do — we'll show why it's insufficient.
    """
    print("\n📊 Naive Correlation Analysis (Unadjusted)")
    print("="*60)
    print("⚠️  These numbers do NOT account for confounders.\n")

    results = []
    for intervention in INTERVENTIONS:
        col = intervention["treatment"]
        if col not in df.columns:
            continue
        treated = df[df[col] == 1]["at_risk"].mean()
        control = df[df[col] == 0]["at_risk"].mean()
        naive_effect = treated - control
        n_treated = (df[col] == 1).sum()
        n_control = (df[col] == 0).sum()

        results.append({
            "Intervention": intervention["name"],
            "At-Risk (Treated)": f"{treated:.1%}",
            "At-Risk (Control)": f"{control:.1%}",
            "Naive Effect": f"{naive_effect:+.3f}",
            "N (Treated)": n_treated,
            "N (Control)": n_control,
        })
        print(f"  {intervention['name']}")
        print(f"    Treated group at-risk rate : {treated:.1%}")
        print(f"    Control group at-risk rate : {control:.1%}")
        print(f"    Naive difference           : {naive_effect:+.3f}\n")

    return pd.DataFrame(results)


def propensity_score_matching(df: pd.DataFrame, treatment_col: str,
                               outcome_col: str = "at_risk",
                               n_matches: int = 1) -> float:
    """
    Simple propensity score matching to estimate Average Treatment Effect (ATE).

    Steps:
    1. Fit a logistic regression to predict treatment assignment from confounders
    2. Match treated and control units on propensity score
    3. Estimate ATE as mean outcome difference between matched pairs
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Build confounder matrix
    available_confounders = [c for c in CONFOUNDERS if c in df.columns]
    X_conf = df[available_confounders].fillna(df[available_confounders].median())
    T = df[treatment_col].values
    Y = df[outcome_col].values

    # Fit propensity score model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_conf)
    ps_model = LogisticRegression(max_iter=500, random_state=42)
    ps_model.fit(X_scaled, T)
    propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]

    # Nearest-neighbour matching (greedy)
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    matched_outcomes_treated = []
    matched_outcomes_control = []

    for ti in treated_idx:
        ps_treated = propensity_scores[ti]
        # Find closest control unit
        distances = np.abs(propensity_scores[control_idx] - ps_treated)
        best_match = control_idx[np.argmin(distances)]
        matched_outcomes_treated.append(Y[ti])
        matched_outcomes_control.append(Y[best_match])

    ate = np.mean(matched_outcomes_treated) - np.mean(matched_outcomes_control)
    return ate


def run_causal_analysis(df: pd.DataFrame):
    """Run propensity score matching for all interventions."""
    print("\n🔬 Causal Analysis — Propensity Score Matching")
    print("="*60)
    print("✅ These estimates account for observed confounders.\n")

    causal_results = []
    for intervention in INTERVENTIONS:
        col = intervention["treatment"]
        if col not in df.columns:
            continue

        ate = propensity_score_matching(df, col)
        direction = "↓ reduces" if ate < 0 else "↑ increases"
        print(f"  {intervention['name']}")
        print(f"    Causal ATE: {ate:+.4f}  ({direction} at-risk probability)\n")

        causal_results.append({
            "Intervention": intervention["name"],
            "Causal ATE": ate,
            "Direction": direction
        })

    return pd.DataFrame(causal_results)


def plot_causal_comparison(naive_df: pd.DataFrame, causal_df: pd.DataFrame):
    """Side-by-side comparison of naive vs causal estimates."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Naive Correlation vs Causal Estimates of Intervention Effects",
                 fontsize=13, fontweight="bold")

    # Naive
    naive_effects = [float(r.replace("+", "")) for r in naive_df["Naive Effect"]]
    colors_naive = ["#d32f2f" if e > 0 else "#388e3c" for e in naive_effects]
    axes[0].barh(naive_df["Intervention"], naive_effects, color=colors_naive)
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Naive (Unadjusted) — ⚠️ Biased")
    axes[0].set_xlabel("Effect on At-Risk Probability")

    # Causal
    causal_effects = causal_df["Causal ATE"].tolist()
    colors_causal = ["#d32f2f" if e > 0 else "#388e3c" for e in causal_effects]
    axes[1].barh(causal_df["Intervention"], causal_effects, color=colors_causal)
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Causal Estimate (PSM) — ✅ Adjusted")
    axes[1].set_xlabel("Causal Effect on At-Risk Probability")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "causal_vs_naive_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def run_full_causal_pipeline():
    """Run the complete causal inference analysis."""
    print("📥 Loading data...")
    df = load_data()

    # Naive analysis
    naive_results = naive_correlation_analysis(df)

    # Causal analysis
    causal_results = run_causal_analysis(df)

    # Plot comparison
    plot_causal_comparison(naive_results, causal_results)

    print("\n✅ Causal inference analysis complete.")
    print("   Key insight: causal estimates may differ significantly from")
    print("   naive correlations once confounders are controlled for.")

    return naive_results, causal_results


if __name__ == "__main__":
    run_full_causal_pipeline()
