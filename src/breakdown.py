import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

D = Path(sys.argv[1])
sp = ["train", "val", "test"]
X = {n: pd.read_csv(D / f"X_{n}.csv", index_col="row_id") for n in sp}
y = {n: pd.read_csv(D / f"y_{n}.csv", index_col="row_id")["default"] for n in sp}
s = {n: pd.read_csv(D / f"s_{n}.csv", index_col="row_id") for n in sp}

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

print("=== A. ROWS PER SPLIT ===")
for n in sp:
    k = int(y[n].sum()); print(f"{n:5s} rows={len(y[n]):4d} defaults={k:3d} non-defaults={len(y[n])-k:3d} rate={k/len(y[n]):.3f}")
n_feat = X["train"].shape[1]
print(f"features after encoding: {n_feat}")
print(f"events-per-variable (train defaults / features): {int(y['train'].sum())}/{n_feat} = {y['train'].sum()/n_feat:.1f}   (rule of thumb: >= 10)")

allS = pd.concat([s[n] for n in sp]); allY = pd.concat([y[n] for n in sp])
allS = allS.assign(default=allY.values, split=sum(([n]*len(y[n]) for n in sp), []))

print("\n=== B. GROUP SIZES per split (rows / defaults) ===")
for g in ["age_group", "sex", "foreign_worker"]:
    print(f"[{g}]")
    for val in sorted(allS[g].unique()):
        row = []
        for n in sp:
            m = allS[(allS.split == n) & (allS[g] == val)]
            row.append(f"{n}: {len(m)}/{int(m.default.sum())}")
        m = allS[allS[g] == val]
        print(f"  {val:9s} all: {len(m)}/{int(m.default.sum())}   " + "   ".join(row))

print("\n=== C. BIAS IN THE LABELS (whole dataset, before any model) ===")
priv = {"age_group": "25+", "sex": "male", "foreign_worker": "no"}
for g, pv in priv.items():
    a = allS[allS[g] == pv]; b = allS[allS[g] != pv]
    ka, na, kb, nb = int(a.default.sum()), len(a), int(b.default.sum()), len(b)
    lo_a, hi_a = wilson(ka, na); lo_b, hi_b = wilson(kb, nb)
    tab = [[ka, na-ka], [kb, nb-kb]]
    p = stats.fisher_exact(tab)[1]
    good_a, good_b = 1-ka/na, 1-kb/nb
    di = good_b / good_a
    print(f"[{g}] reference group = {pv}")
    print(f"  {pv:9s} default {ka/na:.3f}  95% CI [{lo_a:.3f}, {hi_a:.3f}]  n={na}")
    print(f"  other     default {kb/nb:.3f}  95% CI [{lo_b:.3f}, {hi_b:.3f}]  n={nb}")
    print(f"  gap = {kb/na*0 + (kb/nb - ka/na):+.3f} | Fisher exact p = {p:.4f} | "
          f"favourable-outcome ratio (disadvantaged/reference) = {di:.3f}  ({'BELOW 0.8 four-fifths line' if di < 0.8 else 'above 0.8'})")

print("\n=== D. PROXY CHECK: can the sensitive attribute be guessed from the model's inputs? ===")
Xall = pd.concat([X[n] for n in sp]); 
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for g, pv in priv.items():
    t = (allS[g].values != pv).astype(int)
    auc = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), Xall.values, t, cv=cv, scoring="roc_auc")
    print(f"  {g:15s} AUC = {auc.mean():.3f} +/- {auc.std():.3f}   (0.5 = no leakage of the attribute, 1.0 = fully reconstructable)")

print("\n=== E. HOW PRECISE CAN A 150-ROW TEST SET BE? (95% interval half-width) ===")
nt, kt = len(y["test"]), int(y["test"].sum())
for name, p, n in [("accuracy ~0.75", .75, nt), (f"recall (only {kt} defaults in test) ~0.60", .60, kt),
                   (f"precision-ish on ~{int(0.4*nt)} flagged rows ~0.55", .55, int(0.4*nt))]:
    lo, hi = wilson(round(p*n), n); print(f"  {name:55s} +/- {(hi-lo)/2*100:.1f} percentage points")
for g, v in [("age_group", "under_25"), ("foreign_worker", "no")]:
    m = allS[(allS.split == "test") & (allS[g] == v)]
    print(f"  test group {g}={v}: {len(m)} rows, {int(m.default.sum())} defaults, {len(m)-int(m.default.sum())} non-defaults "
          f"-> FPR/FNR rest on {len(m)-int(m.default.sum())} / {int(m.default.sum())} people")