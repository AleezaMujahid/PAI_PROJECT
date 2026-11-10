"""
Tests for the Taiwan dataset support in src/preprocess.py.

IMPORTANT: this uses a SYNTHETIC stand-in that copies the layout of the real UCI file
("Default of Credit Card Clients"): same column names, same coding quirks (EDUCATION 0/5/6,
MARRIAGE 0, negative BILL_AMT, PAY_0 then PAY_2), random values. It proves the CODE handles
those shapes. It says nothing about the real data - run the real file yourself.

    python tests/test_taiwan.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import preprocess as pp                      # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="tw_test_"))
passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok  {name}")


def make_synthetic(n=3000, seed=3):
    r = np.random.default_rng(seed)
    d = {"ID": np.arange(1, n + 1),
         "LIMIT_BAL": r.choice([10000, 50000, 100000, 200000, 500000], n),
         "SEX": r.choice([1, 2], n, p=[.4, .6]),
         "EDUCATION": r.choice([0, 1, 2, 3, 4, 5, 6], n, p=[.002, .35, .47, .16, .01, .005, .003]),
         "MARRIAGE": r.choice([0, 1, 2, 3], n, p=[.002, .46, .53, .008]),
         "AGE": r.integers(21, 70, n)}
    pay = {c: r.choice([-2, -1, 0, 1, 2, 3], n, p=[.1, .2, .5, .1, .07, .03])
           for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]}
    d.update(pay)
    for i in range(1, 7):
        d[f"BILL_AMT{i}"] = r.integers(-5000, 200000, n)          # negatives exist in the real file
    for i in range(1, 7):
        d[f"PAY_AMT{i}"] = np.round(np.exp(r.normal(7.5, 1.5, n)))
    z = -1.6 + 0.8 * (pay["PAY_0"] >= 1) + 0.00000 * d["LIMIT_BAL"]
    d["default payment next month"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame(d)


df = make_synthetic()

# --- write the three real-world shapes ------------------------------------------------
kaggle = TMP / "UCI_Credit_Card.csv"                                  # single header, dotted target
df.rename(columns={"default payment next month": "default.payment.next.month"}).to_csv(kaggle, index=False)

uci_csv = TMP / "default of credit card clients.csv"                  # UCI layout: X1..X23,Y first
with open(uci_csv, "w") as f:
    f.write(",".join([""] + [f"X{i}" for i in range(1, 24)] + ["Y"]) + "\n")
    df.to_csv(f, index=False)

try:
    import openpyxl                                                    # noqa: F401
    uci_xlsx = TMP / "default of credit card clients.xlsx"
    with pd.ExcelWriter(uci_xlsx) as w:
        pd.DataFrame([[""] + [f"X{i}" for i in range(1, 24)] + ["Y"]]).to_excel(
            w, index=False, header=False, startrow=0)
        df.to_excel(w, index=False, startrow=1)
    have_xlsx = True
except ImportError:
    have_xlsx = False

# --- 1. readers --------------------------------------------------------------------------
print("1. the reader copes with each file layout")
expected_cols = ["id", "limit_bal", "sex", "education", "marriage", "age", "pay_0", "pay_2", "pay_3",
                 "pay_4", "pay_5", "pay_6"] + [f"bill_amt{i}" for i in range(1, 7)] \
                + [f"pay_amt{i}" for i in range(1, 7)] + ["default"]
for label, path in [("Kaggle-style csv (single header)", kaggle), ("UCI-style csv (X1..Y row first)", uci_csv)] \
        + ([("UCI-style xlsx", uci_xlsx)] if have_xlsx else []):
    out = pp.read_taiwan(path)
    check(f"{label}: columns normalised, target renamed to 'default'", list(out.columns) == expected_cols)
    check(f"{label}: 3000 rows", len(out) == 3000)
if not have_xlsx:
    print("  --  xlsx layout skipped (openpyxl not installed here)")

# --- 2. full pipeline --------------------------------------------------------------------
print("2. full pipeline on the synthetic Taiwan file")
cfg = {**pp.DATASETS["taiwan"], "raw": kaggle, "out": TMP / "proc"}
rep = pp.run(cfg, verbose=False)
check("runs end to end and passes verify()", True)
check("split is ~70/15/15 of 3000", abs(rep["split_sizes"]["train"] - 2100) <= 1
      and abs(rep["split_sizes"]["val"] - 450) <= 2 and abs(rep["split_sizes"]["test"] - 450) <= 2)
check("'inconsistent' codes were fixed and counted (education 0/5/6, marriage 0)",
      rep["value_fixes_applied"]["education"] > 0 and rep["value_fixes_applied"]["marriage"] > 0)
Xtr = pd.read_csv(TMP / "proc" / "X_train.csv")
cols = list(Xtr.columns)
check("education only has categories 1-4 after the fix",
      sorted(c for c in cols if c.startswith("education=")) == [f"education={i}" for i in (1, 2, 3, 4)])
check("marriage and sex columns are not model inputs (sensitive)",
      not any(c.startswith(("marriage", "sex", "age")) for c in cols))
check("'id' column dropped", "id" not in cols and "row_id" in cols)
s = pd.read_csv(TMP / "proc" / "s_test.csv")
check("sensitive table has age_group and sex only, with male/female labels",
      list(s.columns) == ["row_id", "age_group", "sex"] and set(s.sex) <= {"male", "female"})
check("negative bill_amt values pass (not log-transformed)", "bill_amt1" in cols)
check("limit_bal and pay_amt are log-transformed",
      set(rep["assumptions"]["log_transformed"]) == {"limit_bal"} | {f"pay_amt{i}" for i in range(1, 7)})
check("default rate ~ equal across splits",
      max(rep["default_rate"].values()) - min(rep["default_rate"].values()) < 0.02)

print("3. other layouts give the same processed result")
cfg2 = {**cfg, "raw": uci_csv, "out": TMP / "proc2"}
pp.run(cfg2, verbose=False)
check("UCI-style csv -> identical X_train as Kaggle-style csv",
      pd.read_csv(TMP / "proc" / "X_train.csv").equals(pd.read_csv(TMP / "proc2" / "X_train.csv")))
if have_xlsx:
    cfg3 = {**cfg, "raw": uci_xlsx, "out": TMP / "proc3"}
    pp.run(cfg3, verbose=False)
    check("UCI-style xlsx -> identical X_train too",
          pd.read_csv(TMP / "proc" / "X_train.csv").equals(pd.read_csv(TMP / "proc3" / "X_train.csv")))

print("4. guards and helpers")
bad = df.copy()
bad.loc[5, "PAY_AMT3"] = -50
badf = TMP / "bad.csv"
bad.rename(columns={"default payment next month": "default"}).to_csv(badf, index=False)
try:
    pp.run({**cfg, "raw": badf, "out": TMP / "proc_bad"}, verbose=False)
    check("negative value in a log column raises a clear error", False)
except ValueError as e:
    check("negative value in a log column raises a clear error", "non-negative" in str(e))
folder = TMP / "rawdir"; folder.mkdir()
(folder / "notes.txt").write_text("x")
try:
    pp.find_raw(pp.DATASETS["taiwan"]["raw_globs"], folder)
    check("find_raw errors when nothing matches", False)
except FileNotFoundError:
    check("find_raw errors when nothing matches", True)
(folder / "default of credit card clients.xls").write_text("x")
check("find_raw finds the UCI file name",
      pp.find_raw(pp.DATASETS["taiwan"]["raw_globs"], folder).name == "default of credit card clients.xls")

print("5. cv.py and fairness.py also work on Taiwan")
import cv as cvmod                           # noqa: E402
import fairness as fa                        # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
prob, ys, idx = {}, {}, []
for f in cvmod.cv_folds(cfg, scope="all", n_splits=5):
    m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(f.X_train, f.y_train)
    for i, pr, yt in zip(f.val_index, m.predict_proba(f.X_val)[:, 1], f.y_val):
        prob[i], ys[i] = pr, yt
ids = np.array(sorted(prob))
S = pd.concat([pd.read_csv(TMP / "proc" / f"s_{n}.csv", index_col="row_id")
               for n in ("train", "val", "test")]).sort_index().loc[ids].reset_index(drop=True)
R, G = fa.audit(np.array([ys[i] for i in ids]), np.array([prob[i] >= .5 for i in ids]).astype(int),
                S, cfg["reference_groups"], cfg["headline_attrs"], cfg["exploratory_attrs"], n_boot=100)
check("audit runs with age_group and sex, both reliable at n=3000",
      set(G.attribute) == {"age_group", "sex"} and (G.reliability == "ok").all())

print(f"\nALL {len(passed)} CHECKS PASSED")