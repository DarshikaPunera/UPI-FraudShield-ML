```python
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "upi_dataset"
    / "upi_fraud_detection_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "processed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
TIME_COLUMN = "step"

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

REQUIRED_COLUMNS = NUMERIC_FEATURES + ["type", TARGET]


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

def main():
    print("=" * 60)
    print("SECUREPAY - FRAUD DETECTION PREPROCESSING")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("\nLoading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("Original shape:", df.shape)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values."
        )

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Target column must contain only 0 and 1."
        )

    if (df[NUMERIC_FEATURES] < 0).any().any():
        raise ValueError(
            "Negative numeric values detected. "
            "Check the dataset before continuing."
        )

    numeric_values = df[NUMERIC_FEATURES].to_numpy(
        dtype=float
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Invalid or infinite numeric values detected."
        )

    if df["type"].astype(str).str.strip().eq("").any():
        raise ValueError(
            "Transaction type contains empty values."
        )

    if not df[TIME_COLUMN].map(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError(
            "Transaction step must contain whole numbers."
        )

    df[TIME_COLUMN] = df[TIME_COLUMN].astype(int)

    print("Data validation passed.")


    # ==========================================
    # 3. SORT CHRONOLOGICALLY
    # ==========================================

    print("\nSorting transactions by step...")

    df = df.sort_values(
        TIME_COLUMN,
        kind="stable"
    ).reset_index(drop=True)

    unique_steps = np.sort(df[TIME_COLUMN].unique())

    if len(unique_steps) < 3:
        raise ValueError(
            "At least three distinct transaction steps "
            "are required."
        )


    # ==========================================
    # 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST
    # ==========================================

    print("\nCreating chronological splits...")

    step_counts = (
        df.groupby(TIME_COLUMN, sort=True)
        .size()
    )

    cumulative = step_counts.cumsum().to_numpy()
    total_rows = len(df)

    # Choose a train boundary nearest to 70%.
    train_candidates = np.arange(0, len(cumulative) - 2)

    train_end_index = train_candidates[
        np.argmin(
            np.abs(
                cumulative[train_candidates]
                - total_rows * 0.70
            )
        )
    ]

    # Choose a validation boundary nearest to 85%
    # of the entire dataset, not 85% of the remainder.
    validation_candidates = np.arange(
        train_end_index + 1,
        len(cumulative) - 1
    )

    if len(validation_candidates) == 0:
        raise ValueError(
            "Could not create valid chronological splits."
        )

    validation_end_index = validation_candidates[
        np.argmin(
            np.abs(
                cumulative[validation_candidates]
                - total_rows * 0.85
            )
        )
    ]

    train_end_step = int(unique_steps[train_end_index])
    validation_end_step = int(
        unique_steps[validation_end_index]
    )

    train_df = df[
        df[TIME_COLUMN] <= train_end_step
    ].copy()

    validation_df = df[
        (df[TIME_COLUMN] > train_end_step)
        & (df[TIME_COLUMN] <= validation_end_step)
    ].copy()

    test_df = df[
        df[TIME_COLUMN] > validation_end_step
    ].copy()

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for name, split in splits.items():
        if split.empty:
            raise ValueError(
                f"The {name} split is empty."
            )

        if split[TARGET].nunique() != 2:
            raise ValueError(
                f"The {name} split does not contain both "
                "fraud and genuine transactions. "
                "Review the dataset and time boundaries."
            )

    print(
        "Split percentages:",
        f"Train={len(train_df) / total_rows * 100:.2f}%,",
        f"Validation={len(validation_df) / total_rows * 100:.2f}%,",
        f"Test={len(test_df) / total_rows * 100:.2f}%"
    )


    # ==========================================
    # 5. SELECT MODEL FEATURES
    # ==========================================

    feature_columns = NUMERIC_FEATURES + ["type"]

    X_train = train_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train = train_df[TARGET].astype(int)
    y_validation = validation_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)


    # ==========================================
    # 6. ENCODE TRANSACTION TYPES
    # ==========================================

    print("\nEncoding transaction types...")

    X_train = pd.get_dummies(
        X_train,
        columns=["type"],
        dtype=int
    )

    X_validation = pd.get_dummies(
        X_validation,
        columns=["type"],
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["type"],
        dtype=int
    )

    # Training columns define the feature schema.
    feature_names = X_train.columns.tolist()

    X_validation = X_validation.reindex(
        columns=feature_names,
        fill_value=0
    )

    X_test = X_test.reindex(
        columns=feature_names,
        fill_value=0
    )


    # ==========================================
    # 7. REPORT CLASS DISTRIBUTION
    # ==========================================

    print("\nSplit summary")

    for name, labels in [
        ("Training", y_train),
        ("Validation", y_validation),
        ("Testing", y_test),
    ]:
        fraud_count = int(labels.sum())
        genuine_count = len(labels) - fraud_count

        print(f"\n{name}:")
        print("Rows:", len(labels))
        print("Genuine:", genuine_count)
        print("Fraud:", fraud_count)
        print(
            "Fraud rate (%):",
            round(labels.mean() * 100, 5)
        )


    # ==========================================
    # 8. SAVE PROCESSED DATA
    # ==========================================

    print("\nSaving processed datasets...")

    output_data = {
        "X_train.csv": X_train,
        "X_validation.csv": X_validation,
        "X_test.csv": X_test,
        "y_train.csv": y_train.to_frame(name=TARGET),
        "y_validation.csv": y_validation.to_frame(name=TARGET),
        "y_test.csv": y_test.to_frame(name=TARGET),
    }

    for filename, data in output_data.items():
        data.to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

    metadata = {
        "target_column": TARGET,
        "source_dataset": DATASET_PATH.name,
        "feature_columns": feature_names,
        "split_method": "chronological_by_step",
        "train_end_step": train_end_step,
        "validation_end_step": validation_end_step,
        "train_rows": len(X_train),
        "validation_rows": len(X_validation),
        "test_rows": len(X_test),
    }

    with open(
        OUTPUT_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nPreprocessing completed successfully.")
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
```