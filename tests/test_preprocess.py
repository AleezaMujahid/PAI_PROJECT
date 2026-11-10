"""
Tests for src/preprocess.py.  Run from the project root:
    python tests/test_preprocess.py
Writes only to a temporary folder, never to data/processed.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import preprocess as pp                      # noqa: E402

BASE = pp.DATASETS["german"]
TMP = Path(tempfile.mkdtemp(prefix="pp_test_"))
passed = []


def cfg_for(raw, name):
    return {**BASE, "raw": raw, "out": TMP / name}


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok  {name}")


def digest(folder):
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).glob("*.csv"))}


raw = pd.read_csv(BASE["raw"])

# ---- 1. real data: pipeline + built-in verification --------------------------
print("1. real German Credit data")
rep = pp.run(cfg_for(BASE["raw"], "real"), verbose=False)
check("runs end to end and passes verify()", True)
sz = rep["split_sizes"]   # stratified rounding can give 149/151 instead of 150/150
check("split sizes: train 700, val and test ~150 each, total 1000",
      sz["train"] == 700 and abs(sz["val"] - 150) <= 2 and abs(sz["test"] - 150) <= 2
      and sum(sz.values()) == 1000)
check("no duplicates removed, no missing found",
      rep["duplicates_removed"] == 0 and rep["missing_before_imputation"] == {})
check("all 21-1 raw columns minus 3 sensitive -> 17 inputs encoded to >17 features",
      rep["n_features_after_encoding"] > 17)
sx = pd.read_csv(TMP / "real" / "s_test.csv", index_col="row_id")
check("sex split is correct (690 male / 310 female overall)",
      pd.concat([pd.read_csv(TMP / "real" / f"s_{n}.csv")["sex"] for n in ["train", "val", "test"]])
      .value_counts().to_dict() == {"male": 690, "female": 310})

# ---- 2. reproducibility ------------------------------------------------------
print("2. reproducibility")
pp.run(cfg_for(BASE["raw"], "again"), verbose=False)
check("two runs give byte-identical files", digest(TMP / "real") == digest(TMP / "again"))

# ---- 3. leakage proof --------------------------------------------------------
print("3. leakage: changing TEST rows must not change anything learned")
test_ids = pd.read_csv(TMP / "real" / "X_test.csv")["row_id"].to_numpy()
tampered = raw.copy()
tampered.loc[test_ids, "credit_amount"] *= 1000      # wreck every test row
tampered.loc[test_ids, "age"] = 99
tmp_csv = TMP / "tampered.csv"
tampered.to_csv(tmp_csv, index=False)
pp.run(cfg_for(tmp_csv, "tampered"), verbose=False)
a = json.load(open(TMP / "real" / "preprocessing_meta.json"))["preprocessor"]
b = json.load(open(TMP / "tampered" / "preprocessing_meta.json"))["preprocessor"]
check("median/mean/std/categories identical -> nothing leaked from test", a == b)
xt_a = pd.read_csv(TMP / "real" / "X_train.csv")
xt_b = pd.read_csv(TMP / "tampered" / "X_train.csv")
check("train matrix identical", xt_a.equals(xt_b))
check("test matrix DID change (tampering was real)",
      not pd.read_csv(TMP / "real" / "X_test.csv").equals(pd.read_csv(TMP / "tampered" / "X_test.csv")))

# ---- 4. messy data robustness ------------------------------------------------
print("4. messy raw data (what the CCP rubric calls noise / inconsistencies)")
rng = np.random.default_rng(1)
m = raw.copy()
for col in ["duration", "age", "credit_amount"]:
    m.loc[rng.choice(len(m), 25, replace=False), col] = np.nan
for col in ["housing", "savings_status", "job"]:
    m.loc[rng.choice(len(m), 25, replace=False), col] = np.nan
m["credit_amount"] = m["credit_amount"].astype(object)
m.loc[rng.choice(len(m), 5, replace=False), "credit_amount"] = "n/a"     # junk in numeric
m.loc[rng.choice(len(m), 10, replace=False), "purpose"] = "  Radio/TV "  # whitespace + case
m.loc[rng.choice(len(m), 10, replace=False), "own_telephone"] = "?"      # fake missing
m = pd.concat([m, m.sample(15, random_state=2)], ignore_index=True)       # duplicates
messy_csv = TMP / "messy.csv"
m.to_csv(messy_csv, index=False)
r2 = pp.run(cfg_for(messy_csv, "messy"), verbose=False)
check("duplicates removed", r2["duplicates_removed"] >= 15)
check("missing values detected before imputation", sum(r2["missing_before_imputation"].values()) > 0)
mx = pd.read_csv(TMP / "messy" / "X_train.csv")
check("no NaN left after imputation", not mx.isna().any().any())
check("'  Radio/TV ' merged into 'radio/tv' (no separate column)",
      not any("radio/tv " in c or "Radio" in c for c in mx.columns))
check("'none' category kept as real category", "other_parties=none" in mx.columns)
check("outlier tamed by log + scaling (max |z| < 10)",
      mx["credit_amount"].abs().max() < 10)

# ---- 5. unseen category at test time ----------------------------------------
print("5. unseen category")
meta = json.load(open(TMP / "real" / "preprocessing_meta.json"))["preprocessor"]
row = raw.iloc[[0]].copy()
row["purpose"] = "never_seen_before"
for c in BASE["numeric_cols"]:
    row[c] = pd.to_numeric(row[c])
for c in row.columns:
    if c not in BASE["numeric_cols"]:
        row[c] = pp.clean_text(row[c])
Xrow = pp.apply_preprocessor(row, meta)
check("unseen category -> all purpose=* columns are 0",
      Xrow[[c for c in Xrow.columns if c.startswith("purpose=")]].to_numpy().sum() == 0)

# ---- 6. sensitive-column switch ---------------------------------------------
print("6. sensitive columns policy")
cols_default = list(pd.read_csv(TMP / "real" / "X_train.csv", nrows=1).columns)
pp.run(cfg_for(BASE["raw"], "with_sens"), include_sensitive=True, verbose=False)
cols_with = list(pd.read_csv(TMP / "with_sens" / "X_train.csv", nrows=1).columns)
check("default: no age / personal_status / foreign_worker in features",
      not any(c.split("=")[0] in BASE["sensitive_source_cols"] for c in cols_default))
check("--include-sensitive: they ARE in features", "age" in cols_with and any(
    c.startswith("personal_status=") for c in cols_with))

print(f"\nALL {len(passed)} CHECKS PASSED")