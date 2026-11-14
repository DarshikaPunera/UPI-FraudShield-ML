import pandas as pd
import matplotlib.pyplot as plt
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

OUTPUT_DIR = BASE_DIR / "eda_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("UPI FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:", df.shape)

# ==========================================
# 3. FRAUD DISTRIBUTION
# ==========================================

print("\n1. FRAUD DISTRIBUTION")

fraud_counts = df["isFraud"].value_counts().sort_index()

print(fraud_counts)

fraud_percentage = (
    df["isFraud"].value_counts(normalize=True) * 100
)

print("\nFraud percentage:")
print(fraud_percentage)

fraud_counts.rename(
    index={0: "Genuine", 1: "Fraud"}
).plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Genuine vs Fraud Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fraud_distribution.png",
    dpi=150
)
plt.close()

# ==========================================
# 4. FRAUD BY TRANSACTION TYPE
# ==========================================

print("\n2. FRAUD BY TRANSACTION TYPE")

type_counts = pd.crosstab(
    df["type"],
    df["isFraud"]
)

type_counts = type_counts.rename(
    columns={0: "Genuine", 1: "Fraud"}
)

print(type_counts)

fraud_rate = (
    df.groupby("type")["isFraud"].mean() * 100
).sort_values(ascending=False)

print("\nFraud rate by transaction type (%):")
print(fraud_rate)

fraud_rate.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fraud_rate_by_type.png",
    dpi=150
)
plt.close()

# ==========================================
# 5. TRANSACTION AMOUNT ANALYSIS
# ==========================================

print("\n3. TRANSACTION AMOUNT ANALYSIS")

amount_summary = df.groupby("isFraud")["amount"].agg(
    ["count", "mean", "median", "min", "max"]
)

amount_summary.index = ["Genuine", "Fraud"]

print(amount_summary)

plt.figure(figsize=(8, 5))

for label, name in [(0, "Genuine"), (1, "Fraud")]:
    values = df.loc[
        df["isFraud"] == label, "amount"
    ]

    plt.hist(
        values,
        bins=60,
        alpha=0.5,
        label=name
    )

plt.yscale("log")
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency (log scale)")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "amount_distribution.png",
    dpi=150
)
plt.close()

# ==========================================
# 6. FRAUD OVER TIME
# ==========================================

print("\n4. FRAUD OVER TIME")

fraud_by_step = df.groupby("step")["isFraud"].agg(
    ["sum", "count", "mean"]
)

fraud_by_step["fraud_rate_percent"] = (
    fraud_by_step["mean"] * 100
)

print("\nFirst 10 time steps:")
print(fraud_by_step.head(10))

fraud_by_step["sum"].plot(
    figsize=(10, 5)
)

plt.title("Fraud Transactions Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Fraud Transactions")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fraud_over_time.png",
    dpi=150
)
plt.close()

# ==========================================
# 7. BALANCE FEATURE ANALYSIS
# ==========================================

print("\n5. BALANCE FEATURES")

balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

balance_summary = df.groupby("isFraud")[
    balance_columns
].median()

balance_summary.index = ["Genuine", "Fraud"]

print("\nMedian balances by transaction class:")
print(balance_summary)

# ==========================================
# 8. CORRELATION ANALYSIS
# ==========================================

print("\n6. NUMERICAL FEATURE CORRELATION")

numeric_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]

correlation = df[numeric_columns].corr()

print(correlation["isFraud"].sort_values(ascending=False))

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=60,
    ha="right"
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title("Numerical Feature Correlation")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_matrix.png",
    dpi=150
)
plt.close()

# ==========================================
# 9. SAVE SUMMARY
# ==========================================

fraud_summary = pd.DataFrame({
    "transaction_class": ["Genuine", "Fraud"],
    "count": [
        int((df["isFraud"] == 0).sum()),
        int((df["isFraud"] == 1).sum())
    ]
})

fraud_summary["percentage"] = (
    fraud_summary["count"] / len(df) * 100
)

fraud_summary.to_csv(
    OUTPUT_DIR / "fraud_summary.csv",
    index=False
)

print("\nEDA completed successfully.")

print("Results saved in:", OUTPUT_DIR)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print("-", path.name)