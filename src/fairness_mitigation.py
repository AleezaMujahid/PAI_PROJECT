
"""
fairness_mitigation.py

Compare original and reweighed models using development-only
out-of-fold predictions.

Run:
    python src/fairness_mitigation.py --dataset german
    python src/fairness_mitigation.py --dataset taiwan

The held-out test set is NOT used.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import preprocess as P
from cv import cv_folds


ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# Tuned configurations from your hyperparameter search.
BEST_PARAMS = {
    "german": {
        "Logistic Regression": {
            "C": 0.1,
            "class_weight": "balanced",
        },
        "Random Forest": {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
        },
    },
    "taiwan": {
        "Logistic Regression": {
            "C": 1.0,
            "class_weight": "balanced",
        },
        "Random Forest": {
            "n_estimators": 300,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
    },
}

# Previously selected development thresholds.
THRESHOLDS = {
    "german": {
        "Logistic Regression": 0.40,
        "Random Forest": 0.45,
    },
    "taiwan": {
        "Logistic Regression": 0.55,
        "Random Forest": 0.55,
    },
}


def make_model(name, params):
    if name == "Logistic Regression":
        return LogisticRegression(
            **params,
            max_iter=2000,
            random_state=SEED,
        )

    if name == "Random Forest":
        return RandomForestClassifier(
            **params,
            random_state=SEED,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {name}")


def calculate_reweighing_weights(groups, labels):
    """
    Compute Kamiran-Calders style reweighing weights
    using ONLY the current fold's training rows.

    weight(group, label) =
        P(group) * P(label) / P(group, label)
    """
    groups = pd.Series(groups).reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)

    if groups.isna().any():
        raise ValueError("Sensitive groups contain missing values.")

    n = len(labels)

    group_counts = groups.value_counts()
    label_counts = labels.value_counts()

    joint_counts = (
        pd.DataFrame({"group": groups, "label": labels})
        .groupby(["group", "label"])
        .size()
    )

    weights = np.ones(n, dtype=float)

    for i, (group, label) in enumerate(zip(groups, labels)):
        joint_n = joint_counts.loc[(group, label)]

        weights[i] = (
            group_counts.loc[group]
            * label_counts.loc[label]
            / (n * joint_n)
        )

    return weights


def performance_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
    }


def group_metrics(y_true, probability, groups, threshold):
    prediction = (probability >= threshold).astype(int)

    data = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "prediction": prediction,
        "group": np.asarray(groups),
    })

    rows = []

    for group, part in data.groupby("group"):
        y = part["y_true"].to_numpy()
        p = part["prediction"].to_numpy()

        negative = y == 0
        positive = y == 1

        fpr = (
            np.mean(p[negative] == 1)
            if negative.sum() else np.nan
        )
        fnr = (
            np.mean(p[positive] == 0)
            if positive.sum() else np.nan
        )

        rows.append({
            "group": group,
            "n": len(part),
            "predicted_default_rate": np.mean(p == 1),
            "fpr": fpr,
            "fnr": fnr,
        })

    return pd.DataFrame(rows)


def run(dataset, attribute):
    cfg = dict(P.DATASETS[dataset])

    # Ensure CV reads the configured raw dataset.
    # This is the same configuration used by your CV scripts.
    results = []
    group_results = []
    prediction_results = []

    for model_name, params in BEST_PARAMS[dataset].items():
        threshold = THRESHOLDS[dataset][model_name]

        print(f"\nModel: {model_name}")
        print(f"Mitigation attribute: {attribute}")

        collected = {
            "original": [],
            "reweighed": [],
        }

        # Development folds ONLY: train + validation.
        for fold in cv_folds(
            cfg,
            scope="dev",
            n_splits=5,
            repeats=1,
            seed=SEED,
        ):
            # Obtain sensitive training groups by row ID
            # from the already processed sensitive files.
            sensitive_train = pd.concat([
                pd.read_csv(
                    ROOT / "data" / "processed"
                    / dataset / "s_train.csv"
                ),
                pd.read_csv(
                    ROOT / "data" / "processed"
                    / dataset / "s_val.csv"
                ),
            ], ignore_index=True)

            sensitive_train = (
                sensitive_train
                .set_index("row_id")[attribute]
            )

            training_groups = sensitive_train.loc[
                fold.X_train.index
            ].to_numpy()

            validation_groups = fold.s_val[attribute].to_numpy()

            weights = calculate_reweighing_weights(
                training_groups,
                fold.y_train.to_numpy(),
            )

            print(
                f"  Fold {fold.k + 1}: "
                f"weight range {weights.min():.3f}"
                f" - {weights.max():.3f}"
            )

            for version in ["original", "reweighed"]:
                model = make_model(model_name, params)

                if version == "original":
                    model.fit(
                        fold.X_train,
                        fold.y_train,
                    )
                else:
                    model.fit(
                        fold.X_train,
                        fold.y_train,
                        sample_weight=weights,
                    )

                probability = model.predict_proba(
                    fold.X_val
                )[:, 1]

                collected[version].append(
                    pd.DataFrame({
                        "row_id": fold.val_index,
                        "y_true": fold.y_val.to_numpy(),
                        "probability": probability,
                        "group": validation_groups,
                        "fold": fold.k,
                    })
                )

        for version, pieces in collected.items():
            oof = (
                pd.concat(pieces, ignore_index=True)
                .sort_values("row_id")
                .reset_index(drop=True)
            )

            y = oof["y_true"].to_numpy()
            probability = oof["probability"].to_numpy()

            metrics = performance_metrics(
                y, probability, threshold
            )

            results.append({
                "dataset": dataset,
                "model": model_name,
                "version": version,
                "attribute": attribute,
                "threshold": threshold,
                **metrics,
            })

            groups = group_metrics(
                y,
                probability,
                oof["group"],
                threshold,
            )

            groups["dataset"] = dataset
            groups["model"] = model_name
            groups["version"] = version
            groups["attribute"] = attribute

            group_results.append(groups)

            oof["model"] = model_name
            oof["version"] = version
            oof["attribute"] = attribute
            prediction_results.append(oof)

    out = (
        ROOT / "reports" / "fairness_mitigation"
        / dataset / attribute
    )
    out.mkdir(parents=True, exist_ok=True)

    performance = pd.DataFrame(results)
    fairness = pd.concat(group_results, ignore_index=True)
    predictions = pd.concat(
        prediction_results, ignore_index=True
    )

    performance.to_csv(
        out / "performance.csv", index=False
    )
    fairness.to_csv(
        out / "group_metrics.csv", index=False
    )
    predictions.to_csv(
        out / "oof_predictions.csv", index=False
    )

    print("\n--- PERFORMANCE COMPARISON ---")
    print(performance.round(4).to_string(index=False))

    print("\n--- GROUP FAIRNESS METRICS ---")
    print(fairness.round(4).to_string(index=False))

    print("\nResults saved to:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["german", "taiwan"],
    )
    parser.add_argument(
        "--attribute",
        default="age_group",
        choices=["age_group", "sex"],
    )

    args = parser.parse_args()

    run(args.dataset, args.attribute)
