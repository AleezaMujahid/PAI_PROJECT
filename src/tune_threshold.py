
"""
tune_threshold.py

Select prediction thresholds using out-of-fold
probabilities from the DEVELOPMENT split only.

Usage:
    python src/tune_threshold.py --dataset german
    python src/tune_threshold.py --dataset taiwan

No held-out test rows are used.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from cv import cv_folds
from preprocess import DATASETS
#from train import get_models
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent.parent


def get_oof_predictions(cfg, model, n_splits=5, seed=42):
    """Collect one out-of-fold prediction per dev row."""

    records = []

    for fold in cv_folds(
        cfg,
        scope="dev",
        n_splits=n_splits,
        repeats=1,
        seed=seed,
    ):
        fitted_model = clone(model)

        fitted_model.fit(
            fold.X_train,
            fold.y_train,
        )

        probabilities = fitted_model.predict_proba(
            fold.X_val
        )[:, 1]

        for row_id, true_label, probability in zip(
            fold.val_index,
            fold.y_val.to_numpy(),
            probabilities,
        ):
            records.append({
                "row_id": int(row_id),
                "y_true": int(true_label),
                "probability": float(probability),
                "fold": int(fold.k),
            })

    oof = pd.DataFrame(records)

    if oof["row_id"].duplicated().any():
        raise ValueError("Duplicate OOF row IDs detected")

    return oof.sort_values("row_id").reset_index(drop=True)


def evaluate_thresholds(oof, thresholds):
    """Evaluate different thresholds on OOF predictions."""

    y_true = oof["y_true"].to_numpy()
    probabilities = oof["probability"].to_numpy()

    rows = []

    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        rows.append({
            "threshold": round(float(threshold), 3),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

    return pd.DataFrame(rows)

def get_tuned_models(dataset):
    path = (
        ROOT
        / "reports"
        / "hyperparameter_tuning"
        / f"{dataset}_best_models.json"
    )

    with open(path, "r", encoding="utf-8") as f:
        best = json.load(f)

    lr_params = best["Logistic Regression"]["params"]
    rf_params = best["Random Forest"]["params"]

    return {
        "Logistic Regression": LogisticRegression(
            C=lr_params["C"],
            class_weight=lr_params["class_weight"],
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            class_weight=rf_params["class_weight"],
            random_state=42,
            n_jobs=-1,
        ),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        required=True,
    )
    args = parser.parse_args()

    dataset = args.dataset
    cfg = dict(DATASETS[dataset])

    output_dir = ROOT / "reports" / "threshold_tuning_tuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Candidate thresholds
    thresholds = np.round(
        np.arange(0.20, 0.801, 0.05),
        2,
    )

    for model_name, model in get_tuned_models(dataset).items():
        print(f"\n{'=' * 55}")
        print(f"Dataset: {dataset}")
        print(f"Model: {model_name}")
        print(f"{'=' * 55}")

        # Step 1: Generate OOF probabilities
        oof = get_oof_predictions(cfg, model)

        # Step 2: Compare thresholds
        results = evaluate_thresholds(
            oof,
            thresholds,
        )

        # Step 3: Choose highest OOF F1
        best = results.loc[results["f1"].idxmax()]

        print("\nThreshold comparison:")
        print(
            results[
                [
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "false_positives",
                    "false_negatives",
                ]
            ].round(4).to_string(index=False)
        )

        print("\nSelected threshold:")
        print(f"Threshold: {best['threshold']:.2f}")
        print(f"OOF F1: {best['f1']:.4f}")
        print(f"OOF Recall: {best['recall']:.4f}")
        print(f"OOF Precision: {best['precision']:.4f}")

        safe_name = model_name.lower().replace(" ", "_")

        results.to_csv(
            output_dir / f"{dataset}_{safe_name}_thresholds.csv",
            index=False,
        )

        oof.to_csv(
            output_dir / f"{dataset}_{safe_name}_oof.csv",
            index=False,
        )

    print("\nResults saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
