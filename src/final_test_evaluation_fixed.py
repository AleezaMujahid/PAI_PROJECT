"""Evaluate previously locked models using the ORIGINAL saved train/val/test rows.

No re-splitting, no preprocessing refitting: processed features were fitted using
original training rows only. Test membership is taken from X_test.csv.

Usage:
    python src/final_test_evaluation.py --dataset german
    python src/final_test_evaluation.py --dataset taiwan

Note: Earlier provisional test results were already viewed. This is a correction
of the evaluation protocol, not a fresh untouched first look at test outcomes.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve)

ROOT = Path(__file__).resolve().parent.parent
LOCKED = {
    "german": {"name": "Logistic Regression", "threshold": 0.40,
               "params": {"C": 0.1, "class_weight": "balanced"}},
    "taiwan": {"name": "Random Forest", "threshold": 0.55,
               "params": {"n_estimators": 300, "min_samples_leaf": 10,
                          "class_weight": "balanced"}},
}


def read_split(dataset, split):
    folder = ROOT / "data" / "processed" / dataset
    X = pd.read_csv(folder / f"X_{split}.csv")
    y = pd.read_csv(folder / f"y_{split}.csv")
    s = pd.read_csv(folder / f"s_{split}.csv")
    for name, frame in (("X", X), ("y", y), ("s", s)):
        if "row_id" not in frame:
            raise ValueError(f"{dataset}/{split}: {name} lacks row_id")
        if frame.row_id.isna().any() or frame.row_id.duplicated().any():
            raise ValueError(f"{dataset}/{split}: invalid {name} row IDs")
    if not (X.row_id.tolist() == y.row_id.tolist() == s.row_id.tolist()):
        raise ValueError(f"{dataset}/{split}: X/y/s row IDs are not aligned")
    target_cols = [c for c in y if c != "row_id"]
    if len(target_cols) != 1:
        raise ValueError(f"{dataset}/{split}: unexpected y columns {list(y)}")
    target = y[target_cols[0]].to_numpy()
    if not np.isin(target, [0, 1]).all():
        raise ValueError(f"{dataset}/{split}: y is not binary 0/1")
    features = X.drop(columns="row_id")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError(f"{dataset}/{split}: nonnumeric features")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{dataset}/{split}: NaN or infinity in features")
    return features, target.astype(int), s, set(X.row_id.tolist())


def fairness_table(y, pred, s, attribute):
    records = []
    for group in sorted(s[attribute].fillna("unknown").astype(str).unique()):
        mask = (s[attribute].fillna("unknown").astype(str) == group).to_numpy()
        yt, yp = y[mask], pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        records.append({"attribute": attribute, "group": group,
            "n": int(mask.sum()), "actual_default_rate": float(yt.mean()),
            "predicted_default_rate": float(yp.mean()),
            "fpr": float(fp/(fp+tn)) if fp+tn else None,
            "fnr": float(fn/(fn+tp)) if fn+tp else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return pd.DataFrame(records)


def plot_curves(y, prob, pred, out):
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y, prob):.4f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=170)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {average_precision_score(y, prob):.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "precision_recall_curve.png", dpi=170)
    plt.close(fig)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
           title="Confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=170)
    plt.close(fig)


def evaluate(dataset):
    config = LOCKED[dataset]
    out = ROOT / "reports" / "final_test_evaluation_original_splits" / dataset
    if out.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {out}")
    selection_path = ROOT / "reports" / "final_model_selection" / "selected_models.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Locked selection report missing: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("datasets", {}).get(dataset, {})
    if chosen.get("selected_model") != config["name"]:
        raise ValueError(f"Selected model mismatch in {selection_path}: {chosen}")
    if abs(float(chosen.get("selected_threshold", -1)) - config["threshold"]) > 1e-10:
        raise ValueError(f"Selected threshold mismatch in {selection_path}: {chosen}")

    Xtr, ytr, str_, ids_tr = read_split(dataset, "train")
    Xva, yva, sva, ids_va = read_split(dataset, "val")
    Xte, yte, ste, ids_te = read_split(dataset, "test")
    if ids_tr & ids_va or ids_tr & ids_te or ids_va & ids_te:
        raise ValueError("Split row IDs overlap")
    if not (list(Xtr.columns) == list(Xva.columns) == list(Xte.columns)):
        raise ValueError("Feature column order differs between splits")

    expected = {"german": (700, 149, 151), "taiwan": (20975, 4494, 4496)}[dataset]
    if (len(Xtr), len(Xva), len(Xte)) != expected:
        raise ValueError(f"Original split counts changed: expected {expected}, "
                         f"got {(len(Xtr), len(Xva), len(Xte))}")
    metadata_path = ROOT / "data" / "processed" / dataset / "preprocessing_meta.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta["report"]["assumptions"].get("sensitive_columns_used_as_features") is not False:
        raise ValueError("Sensitive columns may be present in model features")
    if meta["report"]["n_features_after_encoding"] != Xtr.shape[1]:
        raise ValueError("Feature count differs from preprocessing metadata")

    Xdev = pd.concat([Xtr, Xva], ignore_index=True)
    ydev = np.concatenate([ytr, yva])
    if config["name"] == "Logistic Regression":
        model = LogisticRegression(**config["params"], max_iter=2000, random_state=42)
    else:
        model = RandomForestClassifier(**config["params"], random_state=42, n_jobs=-1)
    print(f"\n{dataset.upper()}: original split verified: "
          f"train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}")
    print("Preprocessing: original TRAIN-fitted saved features (not refitted)")
    print(f"Locked model: {config['name']}; threshold={config['threshold']}")
    model.fit(Xdev, ydev)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= config["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    metrics = {"dataset": dataset, "model": config["name"],
        "threshold": config["threshold"], "train_rows": len(Xtr),
        "validation_rows": len(Xva), "test_rows": len(Xte),
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, prob)),
        "average_precision": float(average_precision_score(yte, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "protocol_note": "Original saved split IDs; test was viewed in an earlier provisional run."}

    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"row_id": ste.row_id, "y_true": yte,
                  "probability": prob, "y_pred": pred}).to_csv(
        out / "predictions.csv", index=False)
    for attr in ("age_group", "sex", "foreign_worker"):
        if attr in ste.columns:
            fairness_table(yte, pred, ste, attr).to_csv(
                out / f"fairness_{attr}.csv", index=False)
    (out / "classification_report.txt").write_text(
        classification_report(yte, pred, zero_division=0), encoding="utf-8")
    joblib.dump(model, out / "model.joblib")
    plot_curves(yte, prob, pred, out)
    print("\n" + json.dumps(metrics, indent=2))
    print(f"\nSaved corrected original-split results to: {out}")
    print("Do not retune based on these test results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(LOCKED))
    args = parser.parse_args()
    evaluate(args.dataset)