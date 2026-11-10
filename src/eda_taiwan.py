
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Taiwan CSV
DATA_PATH = Path("data/raw/default of credit card clients.csv")
OUTPUT_DIR = Path("reports/eda/taiwan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 2. Fix the embedded header in this CSV version
if "X1" in df.columns and str(df.iloc[0, 0]).strip() == "ID":
    df.columns = df.iloc[0].tolist()
    df = df.iloc[1:].reset_index(drop=True)

# 3. Convert values to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

target = "default payment next month"

# 4. Basic data overview
print("\nDATASET SHAPE:", df.shape)
print("\nCOLUMNS:\n", df.columns.tolist())
print("\nDATA TYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isna().sum())
print("\nDUPLICATES:", df.duplicated().sum())

print("\nTARGET COUNTS:\n", df[target].value_counts())
print("\nTARGET PERCENTAGES:\n",
      df[target].value_counts(normalize=True) * 100)

print("\nNUMERICAL SUMMARY:\n", df.describe().round(2))

# 5. Target distribution
df[target].value_counts().sort_index().plot(
    kind="bar", title="Taiwan: Default vs Non-default"
)
plt.xlabel("Default next month (0 = No, 1 = Yes)")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png")
plt.close()

# 6. Age distribution
df["AGE"].hist(bins=20)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_distribution.png")
plt.close()

# 7. Credit limit distribution
df["LIMIT_BAL"].plot(kind="box")
plt.title("Credit Limit Distribution")
plt.ylabel("Credit limit")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "credit_limit_boxplot.png")
plt.close()

# 8. Default rates by sex
sex_rates = df.groupby("SEX")[target].mean()
print("\nDEFAULT RATE BY SEX:\n", sex_rates)

sex_rates.plot(kind="bar", title="Default Rate by Sex")
plt.ylabel("Default rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "default_by_sex.png")
plt.close()

# 9. Default rates by age group
df["age_group"] = pd.cut(
    df["AGE"],
    bins=[0, 24, 100],
    labels=["under_25", "25_plus"]
)

age_rates = df.groupby("age_group", observed=False)[target].mean()
print("\nDEFAULT RATE BY AGE GROUP:\n", age_rates)

age_rates.plot(kind="bar", title="Default Rate by Age Group")
plt.ylabel("Default rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "default_by_age_group.png")
plt.close()

# 10. Correlation heatmap (ID excluded)
corr = df.drop(columns=["ID", "age_group"], errors="ignore").corr()

plt.figure(figsize=(14, 10))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns,
           rotation=90, fontsize=7)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
plt.title("Taiwan Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

print("\nEDA completed. Plots saved in:", OUTPUT_DIR)
