"""
Main training entrypoint for Student Success Predictor.
Run this to train all models, run HPO, compute SHAP, and save artifacts.
"""

import os
import sys
import joblib
import pandas as pd
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import load_data, preprocess
from src.models import train_all_models, save_model
from src.hyperparameter_opt import optimise_xgboost, build_optimised_xgboost
from src.explainability import compute_shap_values, plot_global_importance
from src.causal_inference import run_causal_analysis, format_causal_results

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("STUDENT SUCCESS PREDICTOR — TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load & preprocess
    df = load_data("data/raw/student_data.csv")
    X_train, X_test, y_train, y_test, features, scaler = preprocess(df)

    # 2. Train baseline models
    logger.info("\n[1/4] Training baseline models...")
    results_df, trained_models, best_model, best_model_name = train_all_models(
        X_train, X_test, y_train, y_test
    )
    print("\n📊 Model Comparison:")
    print(results_df[['model', 'roc_auc', 'f1', 'precision', 'recall']].to_string(index=False))

    # 3. Hyperparameter optimisation
    logger.info("\n[2/4] Running hyperparameter optimisation (Optuna)...")
    best_params, study = optimise_xgboost(X_train, y_train, n_trials=50)
    optimised_model = build_optimised_xgboost(best_params)
    optimised_model.fit(X_train, y_train)

    from src.models import evaluate_model
    opt_metrics, _ = evaluate_model(optimised_model, X_test, y_test, "XGBoost (Optimised)")
    logger.info(f"Optimised XGBoost AUC: {opt_metrics['roc_auc']} | F1: {opt_metrics['f1']}")

    # Use Random Forest as SHAP model (XGBoost version compatibility)
    rf_model = trained_models["Random Forest"][0]
    final_model = rf_model
    save_model(final_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(features, "models/feature_names.pkl")
    logger.info("Saved model artifacts to models/")

    # 4. SHAP explainability
    logger.info("\n[3/4] Computing SHAP values...")
    os.makedirs("plots", exist_ok=True)
    explainer, shap_values = compute_shap_values(final_model, X_test)
    plot_global_importance(shap_values, X_test, "plots/shap_global.png")
    joblib.dump(explainer, "models/shap_explainer.pkl")
    logger.info("Saved SHAP explainer and plots")

    # 5. Causal inference
    logger.info("\n[4/4] Running causal inference analysis...")
    causal_results = run_causal_analysis(df)
    causal_df = format_causal_results(causal_results)
    causal_df.to_csv("models/causal_results.csv", index=False)
    print("\n🔬 Causal Analysis Results:")
    print(causal_df[['Intervention', 'Causal Effect (ATE)', '95% CI', 'Effect']].to_string(index=False))

    logger.info("\n✅ Training pipeline complete. Run the app with: streamlit run app/app.py")


if __name__ == "__main__":
    main()
