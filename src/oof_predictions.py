"""
oof_predictions.py  -  cross-validated metrics + out-of-fold predictions for the audit.

This is a REFERENCE script so Person A can test the audit pipeline. Person B and Person C
should produce predictions in the same CSV format with their own models.

    python src/oof_predictions.py --dataset german --model logreg --scope dev
        -> 5-fold CV on train+val (850 rows): mean +/- std of every metric (model selection)
    python src/oof_predictions.py --dataset german --model logreg --scope all
        -> every row gets an out-of-fold prediction -> predictions/german_logreg_all_oof.csv
           (audit this with fairness.py; more rows per group than the 151-row test split)
    python src/oof_predictions.py --dataset german --model logreg --final
        -> train on train+val, evaluate ONCE on the held-out test split (do this at the very end)

Decision rule: threshold 0.5 on probabilities from models trained with balanced class weights.
Do not tune the threshold on the same rows you then audit.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import preprocess as P
from cv import cv_folds, final_split

ROOT = Path(__file__).resolve().parent.parent


def make_model(name, seed):
    if name == "logreg":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def metric_row(y, prob, thr):
    pred = (prob >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob)}


def save_preds(path, index, y, prob, thr, model_name):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": index, "y_true": y, "y_pred": (prob >= thr).astype(int),
                  "y_prob": prob, "model_name": model_name}).sort_values("index").to_csv(path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german", choices=sorted(P.DATASETS))
    ap.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    ap.add_argument("--scope", default="dev", choices=["dev", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--final", action="store_true", help="evaluate once on the held-out test split")
    a = ap.parse_args()
    cfg = P.DATASETS[a.dataset]

    if a.final:
        d = final_split(cfg)
        m = make_model(a.model, P.SEED).fit(d["X_dev"], d["y_dev"])
        prob = m.predict_proba(d["X_test"])[:, 1]
        print(f"HELD-OUT TEST ({len(prob)} rows), single evaluation:")
        for k, v in metric_row(d["y_test"].to_numpy(), prob, a.threshold).items():
            print(f"  {k:10s} {v:.3f}")
        out = ROOT / "predictions" / f"{a.dataset}_{a.model}_test.csv"
        save_preds(out, d["y_test"].index, d["y_test"].to_numpy(), prob, a.threshold, a.model)
        print("saved", out)
        raise SystemExit

    rows, sums, counts = [], {}, {}
    ytrue = {}
    for f in cv_folds(cfg, scope=a.scope, n_splits=a.folds, repeats=a.repeats):
        m = make_model(a.model, P.SEED).fit(f.X_train, f.y_train)
        prob = m.predict_proba(f.X_val)[:, 1]
        rows.append(metric_row(f.y_val.to_numpy(), prob, a.threshold))
        for i, pr, yt in zip(f.val_index, prob, f.y_val.to_numpy()):
            sums[i] = sums.get(i, 0.0) + pr
            counts[i] = counts.get(i, 0) + 1
            ytrue[i] = yt

    cvm = pd.DataFrame(rows)
    print(f"{a.model} | scope={a.scope} | {a.folds}-fold x {a.repeats} | {len(ytrue)} rows | "
          f"threshold {a.threshold}")
    for k in cvm.columns:
        print(f"  {k:10s} {cvm[k].mean():.3f} +/- {cvm[k].std(ddof=1):.3f}   (per-fold std)")

    idx = np.array(sorted(ytrue))
    prob = np.array([sums[i] / counts[i] for i in idx])
    y = np.array([ytrue[i] for i in idx])
    pooled = metric_row(y, prob, a.threshold)
    print("  pooled out-of-fold:", {k: round(v, 3) for k, v in pooled.items()})
    out = ROOT / "predictions" / f"{a.dataset}_{a.model}_{a.scope}_oof.csv"
    save_preds(out, idx, y, prob, a.threshold, a.model)
    print("saved", out)