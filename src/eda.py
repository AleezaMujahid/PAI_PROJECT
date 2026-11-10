"""
eda.py  -  Exploratory Data Analysis for German Credit (Person A)

Run from the project root (venv activated):
    python src/eda.py

This script only LOOKS at the data. It never changes the raw file.
Plots are saved to reports/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # save plots to files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent      # project root folder
RAW = ROOT / "data" / "raw" / "credit_g.csv"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TARGET = "class"                                   # values: 'good' / 'bad'
SUSPICIOUS = {"", "?", "na", "n/a", "nan", "null", "-1", "unknown"}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ------------------------------------------------------------------ 1. load
df = pd.read_csv(RAW)

section("1. SHAPE, TYPES, FIRST ROWS")
print("shape (rows, columns):", df.shape)
print("\ncolumn types:\n", df.dtypes.to_string())
print("\nfirst 5 rows:\n", df.head().to_string())

numeric_cols = df.select_dtypes(include="number").columns.tolist()
text_cols = [c for c in df.columns if c not in numeric_cols]
print(f"\nnumeric columns ({len(numeric_cols)}):", numeric_cols)
print(f"text columns ({len(text_cols)}):", text_cols)

# ------------------------------------------------------------------ 2. missing
section("2. MISSING VALUES AND DUPLICATES")
missing = df.isna().sum()
print("real missing values per column (only columns with > 0):")
print(missing[missing > 0].to_string() if missing.any() else "  none - 0 missing cells")
print("\nduplicate rows:", int(df.duplicated().sum()))

# fake missing values hidden inside text columns ('?', 'n/a', ...)
fake = {}
for c in text_cols:
    hits = df[c].astype(str).str.strip().str.lower().isin(SUSPICIOUS).sum()
    if hits:
        fake[c] = int(hits)
print("fake-missing placeholders found:", fake if fake else "none")
print("NOTE: 'none' in other_parties / other_payment_plans / own_telephone is a REAL "
      "category, not a missing value.")

# ------------------------------------------------------------------ 3. text columns
section("3. TEXT COLUMNS - every category and its count")
for c in text_cols:
    print(f"\n[{c}]  ({df[c].nunique()} categories)")
    print(df[c].value_counts().to_string())

# ------------------------------------------------------------------ 4. numeric
section("4. NUMERIC COLUMNS - summary statistics")
print(df[numeric_cols].describe().round(2).to_string())
skew = df[numeric_cols].skew().round(2).sort_values(ascending=False)
print("\nskewness (large positive = long right tail):\n", skew.to_string())

# ------------------------------------------------------------------ 5. target
section("5. TARGET BALANCE (bad = default = 1, good = 0)")
df["default"] = (df[TARGET] == "bad").astype(int)
counts = df[TARGET].value_counts()
print(counts.to_string())
print("\nproportions:\n", df[TARGET].value_counts(normalize=True).round(3).to_string())
print("\nA model that always predicts 'good' would score "
      f"{counts['good'] / len(df):.1%} accuracy - so accuracy alone is misleading.")

# ------------------------------------------------------------------ 6. sensitive groups
section("6. DEFAULT RATE PER SENSITIVE GROUP")
df["age_group"] = np.where(df["age"] >= 25, "25+", "under_25")

# CAREFUL: 'female' contains the word 'male', so test for 'female' FIRST.
df["sex"] = np.where(df["personal_status"].str.startswith("female"), "female", "male")

group_tables = {}
for g in ["age_group", "sex", "foreign_worker"]:
    t = df.groupby(g)["default"].agg(n="size", default_rate="mean")
    t["default_rate"] = t["default_rate"].round(3)
    group_tables[g] = t
    print(f"\n[{g}]\n{t.to_string()}")

print("\nNOTE: German Credit has no 'female single' category, so this 'sex' column is "
      "derived from personal_status and is limited. Mention this in the report.")

# ------------------------------------------------------------------ 7. plots
section("7. SAVING PLOTS to " + str(FIG))

# 7a. class balance
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.bar(counts.index, counts.values, color=["#4c78a8", "#e45756"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 8, str(v), ha="center")
ax.set_title("Class balance"); ax.set_ylabel("applicants")
fig.tight_layout(); fig.savefig(FIG / "01_class_balance.png", dpi=150); plt.close(fig)

# 7b. age
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(df["age"], bins=30, color="#4c78a8", edgecolor="white")
ax.axvline(25, color="#e45756", ls="--", label="age 25 cutoff")
ax.set_title("Age distribution"); ax.set_xlabel("age"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "02_age_hist.png", dpi=150); plt.close(fig)

# 7c. credit_amount: raw vs log (why we log-transform later)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].hist(df["credit_amount"], bins=40, color="#4c78a8", edgecolor="white")
axes[0].set_title("credit_amount (raw) - right-skewed")
axes[1].hist(np.log1p(df["credit_amount"]), bins=40, color="#54a24b", edgecolor="white")
axes[1].set_title("log1p(credit_amount) - more symmetric")
fig.tight_layout(); fig.savefig(FIG / "03_credit_amount_hist.png", dpi=150); plt.close(fig)

# 7d. default rate per sensitive group
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
overall = df["default"].mean()
for ax, (g, t) in zip(axes, group_tables.items()):
    ax.bar(t.index, t["default_rate"], color="#4c78a8")
    ax.axhline(overall, color="#e45756", ls="--", label=f"overall {overall:.2f}")
    for i, (idx, row) in enumerate(t.iterrows()):
        ax.text(i, row["default_rate"] + 0.01, f"n={int(row['n'])}", ha="center", fontsize=8)
    ax.set_title(g)
axes[0].set_ylabel("default rate"); axes[0].legend()
fig.tight_layout(); fig.savefig(FIG / "04_default_rate_by_group.png", dpi=150); plt.close(fig)

# 7e. correlation heatmap (numeric columns + target)
corr = df[numeric_cols + ["default"]].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im); ax.set_title("Correlation (numeric features + default)")
fig.tight_layout(); fig.savefig(FIG / "05_correlation.png", dpi=150); plt.close(fig)

# 7f. default rate by checking_status and purpose (strongest categorical signals)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["checking_status", "purpose"]):
    r = df.groupby(col)["default"].mean().sort_values()
    ax.barh(r.index, r.values, color="#4c78a8")
    ax.axvline(overall, color="#e45756", ls="--")
    ax.set_title(f"default rate by {col}")
fig.tight_layout(); fig.savefig(FIG / "06_default_rate_by_category.png", dpi=150); plt.close(fig)

print("saved:", *sorted(p.name for p in FIG.glob("*.png")), sep="\n  ")
print("\nEDA finished. The raw CSV was not modified.")