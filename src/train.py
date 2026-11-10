
"""
train.py - Classical ML model comparison.

Trains Logistic Regression and Random Forest using
leak-free cross-validation on the development split.

Usage:
    python src/train.py --dataset taiwan
    python src/train.py --dataset german
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from cv import cv_folds
from preprocess import DATASETS


ROOT = Path(__file__).resolve().parent.parent


def get_models(seed=42):
    """Return baseline classical ML models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Fit a model and calculate validation metrics."""
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    return {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_val, probabilities),
    }


def run_cross_validation(dataset, n_splits=5, seed=42):
    """Evaluate both models on leak-free CV folds."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    cfg = dict(DATASETS[dataset])

    # Use the processed dataset's raw-data source.
    # DATASETS config should already contain its raw path.
    models = get_models(seed)

    results = []

    for fold in cv_folds(
        cfg,
        scope="dev",
        n_splits=n_splits,
        repeats=1,
        seed=seed,
    ):
        print(f"\n--- Fold {fold.k + 1}/{n_splits} ---")

        for model_name, model in models.items():
            metrics = evaluate_model(
                clone(model),
                fold.X_train,
                fold.y_train,
                fold.X_val,
                fold.y_val,
            )

            row = {
                "dataset": dataset,
                "model": model_name,
                "fold": fold.k + 1,
                **metrics,
            }
            results.append(row)

            print(
                f"{model_name}: "
                f"Accuracy={metrics['accuracy']:.3f}, "
                f"Recall={metrics['recall']:.3f}, "
                f"F1={metrics['f1']:.3f}, "
                f"ROC-AUC={metrics['roc_auc']:.3f}"
            )

    results_df = pd.DataFrame(results)

    # Save fold-by-fold results.
    out_dir = ROOT / "reports" / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        out_dir / f"{dataset}_cv_folds.csv",
        index=False,
    )

    # Calculate mean and standard deviation.
    metric_columns = [
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ]

    summary = (
        results_df.groupby("model")[metric_columns]
        .agg(["mean", "std"])
        .round(4)
    )

    summary.to_csv(
        out_dir / f"{dataset}_cv_summary.csv"
    )

    print("\n====================================")
    print(f"{dataset.upper()} MODEL COMPARISON")
    print("====================================")
    print(summary.to_string())

    print("\nSaved results to:", out_dir.resolve())

    return results_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="taiwan",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    run_cross_validation(
        dataset=args.dataset,
        n_splits=args.folds,
    )


if __name__ == "__main__":
    main()
