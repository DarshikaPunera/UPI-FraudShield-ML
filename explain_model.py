import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_DIR = BASE_DIR / "processed_data"
RESULT_DIR = BASE_DIR / "model_results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Explain a small sample first to avoid unnecessary computation.
SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# ==========================================
# 2. LOAD MODEL AND DATA
# ==========================================

def main():
    print("=" * 60)
    print("UPI FRAUD DETECTION - SHAP EXPLAINABILITY")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata not found. Run train_model.py first."
        )

    test_path = DATA_DIR / "X_test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            "Test dataset not found. Run preprocessing.py first."
        )

    print("\nLoading model and metadata...")

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]

    print("Model loaded successfully.")

    # ==========================================
    # 3. SELECT TEST SAMPLES
    # ==========================================

    print("\nLoading test data...")

    X_test = pd.read_csv(test_path)

    X_test = X_test.reindex(
        columns=feature_columns,
        fill_value=0
    )

    sample_size = min(SAMPLE_SIZE, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    print("Total test rows:", len(X_test))
    print("Rows selected for explanation:", len(X_sample))

    # ==========================================
    # 4. INITIALIZE SHAP
    # ==========================================

    print("\nInitializing SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    # ==========================================
    # 5. CALCULATE SHAP VALUES
    # ==========================================

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(X_sample)

    # Handle binary classification output.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    # ==========================================
    # 6. CALCULATE GLOBAL FEATURE IMPORTANCE
    # ==========================================

    print("\nCalculating feature importance...")

    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_absolute_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance.to_csv(
        RESULT_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\nTop 10 influential features:")

    print(
        importance.head(10).to_string(index=False)
    )

    # ==========================================
    # 7. SAVE A SAMPLE OF EXPLANATIONS
    # ==========================================

    print("\nSaving individual explanation results...")

    explanation_rows = []

    for row_index, (sample_index, row) in enumerate(
        X_sample.iterrows()
    ):
        feature_contributions = {}

        for feature_index, feature_name in enumerate(
            X_sample.columns
        ):
            feature_contributions[feature_name] = float(
                shap_values[row_index, feature_index]
            )

        explanation_rows.append({
            "sample_index": int(sample_index),
            "feature_contributions": json.dumps(
                feature_contributions
            )
        })

    pd.DataFrame(explanation_rows).to_csv(
        RESULT_DIR / "shap_sample_explanations.csv",
        index=False
    )

    # ==========================================
    # 8. SAVE RUN INFORMATION
    # ==========================================

    summary = {
        "method": "SHAP TreeExplainer",
        "model": "XGBoost",
        "sample_size": int(sample_size),
        "feature_count": int(X_sample.shape[1]),
        "importance_file": "shap_feature_importance.csv",
        "individual_explanations_file": (
            "shap_sample_explanations.csv"
        ),
        "interpretation": (
            "Positive SHAP values generally push the model "
            "toward the fraud class; negative values push "
            "it toward the genuine class."
        )
    }

    with open(
        RESULT_DIR / "shap_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nSHAP analysis completed successfully.")

    print("\nGenerated files:")
    print("- model_results/shap_feature_importance.csv")
    print("- model_results/shap_sample_explanations.csv")
    print("- model_results/shap_summary.json")


if __name__ == "__main__":
    main()