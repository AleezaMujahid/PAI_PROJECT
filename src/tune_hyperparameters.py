
"""
Hyperparameter tuning for German and Taiwan datasets.

Uses development-set CV only.
Does NOT access the held-out test set.

Usage:
    python src/tune_hyperparameters.py --dataset german
    python src/tune_hyperparameters.py --dataset taiwan
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

from cv import cv_folds
from preprocess import DATASETS


ROOT = Path(__file__).resolve().parent.parent


def get_candidates(seed=42):
    candidates = []

    # Logistic Regression:
    # C controls regularization strength.
    for c in [0.1, 1.0, 10.0]:
        candidates.append((
            "Logistic Regression",
            {"C": c, "class_weight": "balanced"},
            LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        ))

    # Random Forest:
    # Compare class weighting and leaf size.
    for weight in [None, "balanced"]:
        for leaf in [2, 5, 10]:
            candidates.append((
                "Random Forest",
                {
                    "class_weight": weight,
                    "min_samples_leaf": leaf,
                    "n_estimators": 300,
                },
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=leaf,
                    class_weight=weight,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ))

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
    )
    args = parser.parse_args()

    dataset = args.dataset
    cfg = dict(DATASETS[dataset])

    out = ROOT / "reports" / "hyperparameter_tuning"
    out.mkdir(parents=True, exist_ok=True)

    candidates = get_candidates()
    results = []

    # Cache the 5 development folds so every
    # candidate uses exactly the same splits.
    folds = list(cv_folds(
        cfg,
        scope="dev",
        n_splits=5,
        repeats=1,
        seed=42,
    ))

    print(f"\nDataset: {dataset}")
    print(f"Configurations: {len(candidates)}")
    print(f"CV folds: {len(folds)}")

    for number, (name, params, model) in enumerate(
        candidates, start=1
    ):
        auc_scores = []
        f1_scores = []

        for fold in folds:
            fitted = clone(model)
            fitted.fit(fold.X_train, fold.y_train)

            probs = fitted.predict_proba(
                fold.X_val
            )[:, 1]

            preds = (probs >= 0.5).astype(int)

            auc_scores.append(
                roc_auc_score(fold.y_val, probs)
            )
            f1_scores.append(
                f1_score(
                    fold.y_val,
                    preds,
                    zero_division=0,
                )
            )

        row = {
            "model": name,
            "params": json.dumps(params),
            "auc_mean": np.mean(auc_scores),
            "auc_std": np.std(auc_scores, ddof=1),
            "f1_at_050_mean": np.mean(f1_scores),
        }
        results.append(row)

        print(
            f"[{number}/{len(candidates)}] "
            f"{name} {params} "
            f"AUC={row['auc_mean']:.4f} "
            f"F1@0.50={row['f1_at_050_mean']:.4f}"
        )

    df = pd.DataFrame(results)

    df = df.sort_values(
        ["model", "auc_mean"],
        ascending=[True, False],
    )

    csv_path = out / f"{dataset}_tuning_results.csv"
    df.to_csv(csv_path, index=False)

    best_models = {}

    print("\n--- BEST CONFIGURATION PER MODEL ---")

    for name, group in df.groupby("model"):
        best = group.sort_values(
            "auc_mean", ascending=False
        ).iloc[0]

        best_models[name] = {
            "params": json.loads(best["params"]),
            "auc_mean": float(best["auc_mean"]),
            "auc_std": float(best["auc_std"]),
        }

        print(f"\n{name}")
        print("Parameters:", best["params"])
        print(f"Mean ROC-AUC: {best['auc_mean']:.4f}")
        print(f"Std ROC-AUC: {best['auc_std']:.4f}")

    json_path = out / f"{dataset}_best_models.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_models, f, indent=2)

    print("\nResults saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
