
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. FOLDERS
# ==========================================
BASE = Path("data/processed/taiwan")
OUT = Path("reports/eda/taiwan_processed")
OUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD PROCESSED DATA
# ==========================================
X_train = pd.read_csv(BASE / "X_train.csv")
X_val = pd.read_csv(BASE / "X_val.csv")
X_test = pd.read_csv(BASE / "X_test.csv")

y_train = pd.read_csv(BASE / "y_train.csv")
y_val = pd.read_csv(BASE / "y_val.csv")
y_test = pd.read_csv(BASE / "y_test.csv")

print("\n--- ORIGINAL FILE COLUMNS ---")
print("X columns:", X_train.columns.tolist())
print("y columns:", y_train.columns.tolist())

# ==========================================
# 3. IDENTIFY THE TARGET CORRECTLY
# ==========================================
# row_id is only an identifier, not a target.
target_columns = [
    col for col in y_train.columns
    if col.lower() != "row_id"
]

if len(target_columns) != 1:
    raise ValueError(
        "Expected exactly one target column besides row_id. "
        f"Found: {target_columns}"
    )

target = target_columns[0]
print("\nDetected target column:", target)

# Validate that target values are 0 and 1.
for name, y in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    if target not in y.columns:
        raise ValueError(f"{name} is missing target: {target}")

    if y[target].isna().any():
        raise ValueError(f"{name} has missing target values")

    if not y[target].isin([0, 1]).all():
        raise ValueError(
            f"{name} has unexpected target values: "
            f"{y[target].unique()}"
        )

# Check row alignment where identifiers are available.
for name, X, y in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    if len(X) != len(y):
        raise ValueError(f"{name}: X and y lengths differ")

    if "row_id" in X.columns and "row_id" in y.columns:
        if not X["row_id"].equals(y["row_id"]):
            raise ValueError(
                f"{name}: X and y row_id values do not match"
            )

# ==========================================
# 4. REMOVE IDENTIFIER FROM FEATURES
# ==========================================
X_train = X_train.drop(columns=["row_id"], errors="ignore")
X_val = X_val.drop(columns=["row_id"], errors="ignore")
X_test = X_test.drop(columns=["row_id"], errors="ignore")

# Combine TRAINING features and target for EDA.
train = X_train.copy()
train[target] = y_train[target].to_numpy()

# ==========================================
# 5. DATASET SHAPES
# ==========================================
print("\n--- PROCESSED DATASET SHAPES ---")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\n--- TRAINING DATA INFO ---")
X_train.info()

print("\n--- MISSING VALUES ---")
print("Training features:", X_train.isna().sum().sum())
print("Training target:", y_train[target].isna().sum())

print("\n--- DUPLICATE FEATURE ROWS ---")
print(X_train.duplicated().sum())
print(
    "Note: identical feature vectors are not "
    "necessarily duplicate customers."
)

# ==========================================
# 6. CORRECT TARGET DISTRIBUTION
# ==========================================
print("\n--- CORRECT TARGET DISTRIBUTION ---")
print(train[target].value_counts().sort_index())

print("\n--- TARGET PERCENTAGES ---")
print(
    (
        train[target]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2)
)

print("\n--- TARGET DISTRIBUTION ACROSS SPLITS ---")
for name, y in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    print(f"\n{name}:")
    print(
        (
            y[target]
            .value_counts(normalize=True)
            .sort_index() * 100
        ).round(2)
    )

# ==========================================
# 7. FEATURE SUMMARY
# ==========================================
print("\n--- FEATURE SUMMARY ---")
print(X_train.describe().round(3).to_string())

print("\n--- CONSTANT FEATURES ---")
constant_cols = [
    col for col in X_train.columns
    if X_train[col].nunique() <= 1
]
print(constant_cols if constant_cols else "None")

# ==========================================
# 8. PLOT: TARGET DISTRIBUTION
# ==========================================
counts = train[target].value_counts().sort_index()

plt.figure(figsize=(7, 5))
counts.plot(kind="bar", color=["steelblue", "coral"])
plt.title("Taiwan: Processed Training Target Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    OUT / "processed_target_distribution.png",
    dpi=150
)
plt.close()

# ==========================================
# 9. PLOT: FEATURE DISTRIBUTIONS
# ==========================================
X_train.hist(figsize=(16, 14), bins=25)
plt.suptitle(
    "Taiwan: Processed Training Feature Distributions",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(
    OUT / "processed_feature_distributions.png",
    dpi=150
)
plt.close()

# ==========================================
# 10. FEATURE CORRELATIONS WITH TARGET
# ==========================================
correlations = (
    train.corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print("\n--- FEATURE CORRELATION WITH TARGET ---")
print(correlations.round(4).to_string())

plt.figure(figsize=(14, 6))
correlations.plot(kind="bar", color="steelblue")
plt.title("Taiwan: Training Feature Correlation with Default")
plt.ylabel("Pearson Correlation")
plt.xlabel("Processed Features")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig(
    OUT / "processed_target_correlations.png",
    dpi=150
)
plt.close()

# ==========================================
# 11. FINISHED
# ==========================================
print("\nProcessed Taiwan EDA completed successfully!")
print("Plots saved to:", OUT.resolve())

print("\nGenerated files:")
for file in sorted(OUT.glob("*.png")):
    print(" -", file.name)
