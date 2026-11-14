import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "model_results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_model.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ==========================================
# 2. LOAD PROCESSED DATA
# ==========================================

def main():
    print("=" * 60)
    print("UPI FRAUD DETECTION - XGBOOST TRAINING")
    print("=" * 60)

    required_files = [
        "X_train.csv",
        "X_validation.csv",
        "X_test.csv",
        "y_train.csv",
        "y_validation.csv",
        "y_test.csv"
    ]

    for filename in required_files:
        if not (DATA_DIR / filename).exists():
            raise FileNotFoundError(
                f"Missing {filename}. Run preprocessing.py first."
            )

    print("\nLoading datasets...")

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_validation = pd.read_csv(DATA_DIR / "X_validation.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["isFraud"]
    y_validation = pd.read_csv(
        DATA_DIR / "y_validation.csv"
    )["isFraud"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["isFraud"]

    y_train = y_train.astype(int)
    y_validation = y_validation.astype(int)
    y_test = y_test.astype(int)

    # Check dataset consistency.
    if not (
        len(X_train) == len(y_train)
        and len(X_validation) == len(y_validation)
        and len(X_test) == len(y_test)
    ):
        raise ValueError("Feature and label row counts differ.")

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError("Training and validation features differ.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing features differ.")

    print("Training rows:", f"{len(X_train):,}")
    print("Validation rows:", f"{len(X_validation):,}")
    print("Testing rows:", f"{len(X_test):,}")
    print("Number of features:", X_train.shape[1])

    # ==========================================
    # 3. HANDLE CLASS IMBALANCE
    # ==========================================

    fraud_count = int(y_train.sum())
    genuine_count = len(y_train) - fraud_count

    if fraud_count == 0 or genuine_count == 0:
        raise ValueError(
            "Training data must contain both classes."
        )

    scale_pos_weight = genuine_count / fraud_count

    print("\nTraining class distribution")
    print("Genuine transactions:", genuine_count)
    print("Fraud transactions:", fraud_count)
    print("Scale positive weight:", round(scale_pos_weight, 2))

    # ==========================================
    # 4. TRAIN XGBOOST
    # ==========================================

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    print("\nTraining XGBoost model...")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(
        f"Training completed in {training_time:.2f} seconds."
    )

    # Save the model.
    model.save_model(str(MODEL_PATH))

    print("Model saved:", MODEL_PATH)

    # ==========================================
    # 5. SELECT THRESHOLD USING VALIDATION DATA
    # ==========================================

    print("\nEvaluating validation data...")

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(
        y_validation,
        validation_probabilities
    )

    if len(thresholds) == 0:
        raise ValueError("No thresholds were generated.")

    # Calculate F1 for each available threshold.
    f1_values = (
        2 * precisions[:-1] * recalls[:-1]
        / (
            precisions[:-1]
            + recalls[:-1]
            + 1e-12
        )
    )

    best_index = int(np.argmax(f1_values))
    selected_threshold = float(thresholds[best_index])

    validation_precision = float(precisions[best_index])
    validation_recall = float(recalls[best_index])
    validation_f1 = float(f1_values[best_index])

    print("\nSelected classification threshold:")
    print(f"Threshold: {selected_threshold:.6f}")
    print(f"Precision: {validation_precision:.4f}")
    print(f"Recall:    {validation_recall:.4f}")
    print(f"F1-score:  {validation_f1:.4f}")

    # Save all validation threshold results.
    threshold_results = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions[:-1],
        "recall": recalls[:-1],
        "f1_score": f1_values
    })

    threshold_results.to_csv(
        RESULT_DIR / "validation_thresholds.csv",
        index=False
    )

    # ==========================================
    # 6. FINAL EVALUATION ON UNTOUCHED TEST DATA
    # ==========================================

    print("\nEvaluating the untouched test dataset...")

    test_probabilities = model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    ).ravel()

    test_metrics = {
        "threshold": selected_threshold,
        "accuracy": float(
            np.mean(test_predictions == y_test.to_numpy())
        ),
        "precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ],
        "average_precision": float(
            average_precision_score(
                y_test,
                test_probabilities
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probabilities
            )
        )
    }

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Threshold:  {selected_threshold:.6f}")
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Precision:  {test_metrics['precision']:.4f}")
    print(f"Recall:     {test_metrics['recall']:.4f}")
    print(f"F1-score:   {test_metrics['f1_score']:.4f}")
    print(
        f"PR-AUC (Average Precision): "
        f"{test_metrics['average_precision']:.4f}"
    )
    print(f"ROC-AUC:    {test_metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Genuine", "Fraud"],
            zero_division=0
        )
    )

    # ==========================================
    # 7. FEATURE IMPORTANCE
    # ==========================================

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False
    )

    # ==========================================
    # 8. SAVE MODEL METADATA
    # ==========================================

    metadata = {
        "algorithm": "XGBoost",
        "dataset": "PaySim synthetic mobile-money dataset",
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "feature_columns": X_train.columns.tolist(),
        "selected_threshold": selected_threshold,
        "threshold_selection_method": (
            "Maximum F1-score on validation data"
        ),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "testing_rows": int(len(X_test)),
        "training_time_seconds": round(training_time, 2),
        "validation_metrics": {
            "precision": validation_precision,
            "recall": validation_recall,
            "f1_score": validation_f1
        },
        "test_metrics": test_metrics
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nSaved files:")
    print("- models/xgboost_model.json")
    print("- models/model_metadata.json")
    print("- model_results/validation_thresholds.csv")
    print("- model_results/feature_importance.csv")

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()