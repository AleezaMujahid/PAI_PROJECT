"""
Tests for src/cv.py and src/fairness.py.  Run from the project root:
    python tests/test_cv_fairness.py
Writes only to a temporary folder.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import preprocess as pp                      # noqa: E402
import cv                                    # noqa: E402
import fairness as fa                        # noqa: E402

BASE = pp.DATASETS["german"]
TMP = Path(tempfile.mkdtemp(prefix="cvf_test_"))
passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok  {name}")


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


raw = pd.read_csv(BASE["raw"])
N = len(raw)

# =========================== cross-validation ===========================
print("1. CV scope='all': every row validated exactly once")
folds = list(cv.cv_folds(BASE, scope="all", n_splits=5))
vi = np.concatenate([f.val_index for f in folds])
check("5 folds", len(folds) == 5)
check("validation indices are disjoint and cover all 1000 rows", sorted(vi) == list(range(N)))
check("same columns in train and validation of every fold",
      all(list(f.X_train.columns) == list(f.X_val.columns) for f in folds))
check("no NaN in any fold", all(not f.X_train.isna().any().any() and not f.X_val.isna().any().any() for f in folds))
y_all = (raw["class"] == "bad").astype(int).to_numpy()
check("each validation fold keeps the ~30% default rate (stratified)",
      all(abs(f.y_val.mean() - y_all.mean()) < 0.03 for f in folds))
check("sensitive table rows align with validation rows",
      all(list(f.s_val.index) == list(f.X_val.index) for f in folds))
check("fold train mean of numeric columns is 0 (fitted on fold-train only)",
      all(abs(f.X_train["duration"].mean()) < 1e-9 for f in folds))
check("validation mean is NOT exactly 0 (scaler did not see it)",
      all(abs(f.X_val["duration"].mean()) > 1e-6 for f in folds))

print("2. CV scope='dev': the held-out test rows are never touched")
rep = pp.run({**BASE, "out": TMP / "proc"}, verbose=False)
proc_test = pd.read_csv(TMP / "proc" / "X_test.csv")["row_id"].to_numpy()
dev_folds = list(cv.cv_folds(BASE, scope="dev", n_splits=5))
dev_val = np.concatenate([f.val_index for f in dev_folds])
dev_train = np.concatenate([f.X_train.index.to_numpy() for f in dev_folds])
check("dev validation rows == train+val rows of the 70/15/15 split",
      sorted(dev_val) == sorted(set(range(N)) - set(proc_test)))
check("no test row appears in any dev fold (train or validation)",
      not (set(dev_val) | set(dev_train)) & set(proc_test))
fs = cv.final_split(BASE)
check("final_split test rows == data/processed X_test rows", sorted(fs["test_index"]) == sorted(proc_test))
check("final_split: dev + test = all rows, no overlap",
      len(fs["X_dev"]) + len(fs["X_test"]) == N and not set(fs["X_dev"].index) & set(fs["X_test"].index))
check("final_split dev train mean is 0, test mean is not",
      abs(fs["X_dev"]["duration"].mean()) < 1e-9 and abs(fs["X_test"]["duration"].mean()) > 1e-6)

print("3. leakage: wrecking the VALIDATION rows of fold 0 must not change its training matrix")
f0 = folds[0]
tampered = raw.copy()
tampered.loc[f0.val_index, "credit_amount"] *= 1000
tampered.loc[f0.val_index, "duration"] = 999
tcsv = TMP / "tampered.csv"
tampered.to_csv(tcsv, index=False)
t0 = next(iter(cv.cv_folds({**BASE, "raw": tcsv}, scope="all", n_splits=5)))
check("same validation rows (same seed)", list(t0.val_index) == list(f0.val_index))
check("training matrix identical -> nothing leaked from validation rows", t0.X_train.equals(f0.X_train))
check("validation matrix changed (tampering was real)", not t0.X_val.equals(f0.X_val))

print("4. repeated CV")
rep_folds = list(cv.cv_folds(BASE, scope="all", n_splits=5, repeats=3))
check("15 folds for 5 x 3", len(rep_folds) == 15)
check("each row validated exactly 3 times",
      pd.Series(np.concatenate([f.val_index for f in rep_folds])).value_counts().eq(3).all())

# =========================== fairness maths ===========================
print("5. fairness metrics against a hand-calculated example")
yA = np.array([1, 1, 0, 0, 0, 0]); pA = np.array([1, 0, 0, 0, 1, 0])
yB = np.array([1, 1, 1, 0, 0]);    pB = np.array([1, 1, 0, 1, 0])
ra, rb = fa.group_rates(yA, pA), fa.group_rates(yB, pB)
check("group A: denial 2/6, FPR 1/4, FNR 1/2",
      approx(ra["denial_rate"], 2 / 6) and approx(ra["fpr"], .25) and approx(ra["fnr"], .5))
check("group B: denial 3/5, FPR 1/2, FNR 1/3",
      approx(rb["denial_rate"], .6) and approx(rb["fpr"], .5) and approx(rb["fnr"], 1 / 3))
g = fa.gaps(rb, ra)
check("denial_gap = 0.6 - 1/3", approx(g["denial_gap"], 0.6 - 2 / 6))
check("disparate_impact = (1-0.6)/(1-1/3) = 0.6", approx(g["disparate_impact"], 0.6))
check("fpr_gap = +0.25", approx(g["fpr_gap"], .25))
check("fnr_gap = 1/3 - 1/2", approx(g["fnr_gap"], 1 / 3 - .5))
check("group with no defaulters gives NaN FNR, not a crash",
      np.isnan(fa.group_rates(np.array([0, 0, 0]), np.array([0, 1, 0]))["fnr"]))

print("6. audit() on the toy data: structure, flags, errors")
y_toy = np.concatenate([yA, yB]); p_toy = np.concatenate([pA, pB])
grp = pd.DataFrame({"grp": ["A"] * 6 + ["B"] * 5})
rates_df, gaps_df = fa.audit(y_toy, p_toy, grp, {"grp": "A"}, headline=["grp"], n_boot=200)
check("point estimate of fpr_gap in the audit table is +0.25",
      approx(gaps_df[gaps_df.metric == "fpr_gap"].estimate.iloc[0], .25))
check("tiny groups are flagged UNRELIABLE", gaps_df.reliability.str.startswith("UNRELIABLE").all())
check("4 metrics reported per compared group", len(gaps_df) == 4)
try:
    fa.audit(y_toy, p_toy, grp, {"grp": "nope"}, n_boot=10)
    check("unknown reference group raises ValueError", False)
except ValueError:
    check("unknown reference group raises ValueError", True)
grp_u = pd.DataFrame({"grp": ["A"] * 4 + ["unknown"] * 2 + ["B"] * 5})
rd, gd = fa.audit(y_toy, p_toy, grp_u, {"grp": "A"}, n_boot=50)
check("'unknown' group shown in rates but excluded from gap comparisons",
      "unknown" in set(rd.group) and "unknown" not in set(gd.group))

print("7. bootstrap behaviour")
rng = np.random.default_rng(0)


def synth(n, delta):
    g = rng.choice(["a", "b"], n)
    y = (rng.random(n) < .3).astype(int)
    p = (rng.random(n) < np.where(g == "b", .3 + delta, .3)).astype(int)
    return y, p, pd.DataFrame({"grp": g})


y1, p1, g1 = synth(4000, 0.0)
y2, p2, g2 = synth(4000, 0.2)
_, G0 = fa.audit(y1, p1, g1, {"grp": "a"}, n_boot=400, seed=1)
_, G1 = fa.audit(y2, p2, g2, {"grp": "a"}, n_boot=400, seed=1)
den0 = G0[G0.metric == "denial_gap"].iloc[0]; den1 = G1[G1.metric == "denial_gap"].iloc[0]
check("planted +0.20 gap: CI excludes 0 and contains ~0.2",
      den1.ci_excludes_no_difference and den1.ci_low < 0.2 < den1.ci_high)
check("no planted gap: CI includes 0", not den0.ci_excludes_no_difference)
check("point estimate lies inside its own CI",
      all((r.ci_low <= r.estimate <= r.ci_high) for r in G1.itertuples()))
_, Ga = fa.audit(y1, p1, g1, {"grp": "a"}, n_boot=100, seed=7)
_, Gb = fa.audit(y1, p1, g1, {"grp": "a"}, n_boot=100, seed=7)
check("same seed -> identical results", Ga.equals(Gb))
ys, ps, gs = synth(400, 0.0)
_, Gs = fa.audit(ys, ps, gs, {"grp": "a"}, n_boot=400, seed=1)
w_small = (lambda r: r.ci_high - r.ci_low)(Gs[Gs.metric == "denial_gap"].iloc[0])
w_big = (lambda r: r.ci_high - r.ci_low)(den0)
check("10x fewer rows -> clearly wider interval (about sqrt(10) = 3.2x)", w_small / w_big > 2)

print("8. end to end on German Credit out-of-fold predictions")
from sklearn.linear_model import LogisticRegression  # noqa: E402
prob = np.zeros(N)
for f in folds:
    m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000).fit(f.X_train, f.y_train)
    prob[f.val_index] = m.predict_proba(f.X_val)[:, 1]
sens = pd.concat([pd.read_csv(TMP / "proc" / f"s_{n}.csv", index_col="row_id")
                  for n in ("train", "val", "test")]).sort_index()
R, G = fa.audit(y_all, (prob >= .5).astype(int), sens.reset_index(drop=True),
                BASE["reference_groups"], BASE["headline_attrs"], BASE["exploratory_attrs"],
                n_boot=300)
rel = G.groupby("attribute").reliability.first()
check("age_group and sex comparisons are reliable", rel["age_group"] == "ok" and rel["sex"] == "ok")
check("foreign_worker is flagged UNRELIABLE and labelled exploratory",
      rel["foreign_worker"].startswith("UNRELIABLE") and (G[G.attribute == "foreign_worker"].role == "exploratory").all())
check("age_group / sex labelled headline",
      (G[G.attribute == "age_group"].role == "headline").all() and (G[G.attribute == "sex"].role == "headline").all())
chk = R[(R.attribute == "age_group") & (R.group == "under_25")].iloc[0]
manual_fpr = ((prob >= .5) & (y_all == 0) & (sens.age_group.to_numpy() == "under_25")).sum() / \
             ((y_all == 0) & (sens.age_group.to_numpy() == "under_25")).sum()
check("under_25 FPR from audit() equals an independent calculation", approx(chk.fpr, manual_fpr))
check("group sizes in the audit sum to 1000 for every attribute",
      all(R[R.attribute == a].n.sum() == N for a in R.attribute.unique()))

print(f"\nALL {len(passed)} CHECKS PASSED")