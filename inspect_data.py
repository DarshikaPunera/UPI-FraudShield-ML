import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "upi_dataset"
    / "upi_fraud_detection_dataset.csv"
)

print("=" * 60)
print("UPI FRAUD DETECTION SYSTEM - DATASET INSPECTION")
print("=" * 60)

# ==========================================
# 2. LOAD DATASET
# ==========================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

# ==========================================
# 3. BASIC INFORMATION
# ==========================================

print("\n1. DATASET SHAPE")
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST FIVE RECORDS")
print(df.head())

print("\n4. DATA TYPES")
print(df.dtypes)

# ==========================================
# 4. DATA QUALITY
# ==========================================

print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. TOTAL MISSING VALUES")
print(df.isnull().sum().sum())

print("\n7. DUPLICATE ROWS")
print(df.duplicated().sum())

# ==========================================
# 5. NUMERICAL ANALYSIS
# ==========================================

print("\n8. NUMERICAL SUMMARY")
print(df.describe().T)

# ==========================================
# 6. FRAUD LABEL ANALYSIS
# ==========================================

print("\n9. POSSIBLE FRAUD LABEL COLUMNS")

for column in df.columns:

    if any(
        word in column.lower()
        for word in ["fraud", "label", "target"]
    ):
        print(f"\nColumn: {column}")

        print(
            df[column]
            .value_counts(dropna=False)
        )

# ==========================================
# 7. TRANSACTION TYPE ANALYSIS
# ==========================================

if "type" in df.columns:

    print("\n10. TRANSACTION TYPE DISTRIBUTION")

    print(df["type"].value_counts())

# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)