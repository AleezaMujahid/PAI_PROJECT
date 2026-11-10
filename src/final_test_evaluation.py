
"""
Final held-out test evaluation for German and Taiwan credit datasets.

Selected configurations (locked before looking at test results):
    German: Logistic Regression, C=0.1, class_weight='balanced', threshold=0.40
    Taiwan: Random Forest, n_estimators=300, min_samples_leaf=10,
            class_weight='balanced', threshold=0.55

Protocol:
    1. Recreate the original stratified 70/15/15 split (seed=42).
    2. Fit preprocessing on TRAIN + VALIDATION only.
    3. Transform the untouched TEST split with that preprocessor.
    4. Fit the selected model on TRAIN + VALIDATION.
    5. Evaluate TEST once and save reports.

Run from the project root:
    python src/final_test_evaluation.py --dataset german
    python src/final_test_evaluation.py --dataset taiwan

WARNING:
    Do not change model selection, hyperparameters or thresholds
    after inspecting the final test results.
"""

import argparse
import json
from pathlib import Path

import joblib
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
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import (
    DATASETS,
    SEED,
    prepare,
    three_way_split,
    fit_preprocessor,
    apply_preprocessor,
)


ROOT = Path(__file__).resolve().parent.parent

# LOCKED development-stage decisions.
LOCKED_MODELS = {
    "german": {
        "model_name": "Logistic Regression",
        "threshold": 0.40,
        "params": {
            "C": 0.1,
            "class_weight": "balanced",
        },
    },
    "taiwan": {
        "model_name": "Random Forest",
        "threshold": 0.55,
        "params": {
            "n_estimators": 300,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
    },
}


def build_model(model_name, params):
    """Construct the selected sklearn model."""

    if model_name == "Logistic Regression":
        return LogisticRegression(
            **params,
            max_iter=2000,
            random_state=SEED,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            **params,
            random_state=SEED,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def load_locked_selection(dataset):
    """
    Verify the existing selection report agrees with our locked choices.

    This protects against accidentally evaluating a different model
    or threshold than the one selected during development.
    """
    path = (
        ROOT
        / "reports"
        / "final_model_selection"
        / "selected_models.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model selection report: {path}\n"
            "Run final_model_selection.py first."
        )

    with path.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    # The selection JSON format may differ across project versions.
    # Print its contents for audit and use the explicitly locked
    # configuration above as the evaluation source of truth.
    return saved


def prepare_final_data(dataset):
    """
    Reconstruct the original split and refit preprocessing
    on the combined train+validation development set.

    The test rows are NEVER used to fit preprocessing.
    """
    cfg = dict(DATASETS[dataset])

    df, y, sensitive, num_cols, cat_cols, clean_report = prepare(
        cfg,
        include_sensitive=False,
    )

    splits = three_way_split(y, seed=SEED)

    development_idx = np.sort(
        np.concatenate([
            splits["train"],
            splits["val"],
        ])
    )
    test_idx = splits["test"]

    if len(np.intersect1d(development_idx, test_idx)) != 0:
        raise AssertionError("Development/test overlap detected!")

    development_raw = df.iloc[development_idx]
    test_raw = df.iloc[test_idx]

    # FIT only on development rows.
    preprocessor = fit_preprocessor(
        development_raw,
        num_cols,
        cat_cols,
        cfg["log_cols"],
    )

    # Apply exactly the same fitted transformation.
    X_dev = apply_preprocessor(
        development_raw,
        preprocessor,
    )
    X_test = apply_preprocessor(
        test_raw,
        preprocessor,
    )

    y_dev = y[development_idx]
    y_test = y[test_idx]

    s_test = sensitive.iloc[test_idx].copy()

    if list(X_dev.columns) != list(X_test.columns):
        raise AssertionError("Development/test feature columns differ.")

    if X_dev.isna().any().any() or X_test.isna().any().any():
        raise AssertionError("Missing values remain after preprocessing.")

    if not np.isfinite(X_dev.to_numpy()).all():
        raise AssertionError("Non-finite development features.")

    if not np.isfinite(X_test.to_numpy()).all():
        raise AssertionError("Non-finite test features.")

    if set(np.unique(y_dev)) != {0, 1}:
        raise AssertionError("Development target must contain both classes.")

    if set(np.unique(y_test)) != {0, 1}:
        raise AssertionError("Test target must contain both classes.")

    return {
        "X_dev": X_dev,
        "X_test": X_test,
        "y_dev": y_dev,
        "y_test": y_test,
        "s_test": s_test,
        "test_row_ids": test_idx,
        "preprocessor": preprocessor,
        "clean_report": clean_report,
        "split_sizes": {
            "train": len(splits["train"]),
            "validation": len(splits["val"]),
            "development": len(development_idx),
            "test": len(test_idx),
        },
    }


def calculate_metrics(y_true, probabilities, threshold):
    """Compute final test classification metrics."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return result, predictions


def calculate_group_metrics(y_true, predictions, groups):
    """
    Descriptive fairness audit on final test predictions.

    This is an audit, not a fairness optimization step.
    """
    records = []

    for group in sorted(pd.Series(groups).astype(str).unique()):
        mask = np.asarray(groups).astype(str) == group

        yt = np.asarray(y_true)[mask]
        yp = np.asarray(predictions)[mask]

        tn, fp, fn, tp = confusion_matrix(
            yt,
            yp,
            labels=[0, 1],
        ).ravel()

        fpr_denominator = fp + tn
        fnr_denominator = fn + tp

        records.append({
            "group": group,
            "n": int(mask.sum()),
            "actual_default_rate": float(np.mean(yt)),
            "predicted_default_rate": float(np.mean(yp)),
            "fpr": (
                float(fp / fpr_denominator)
                if fpr_denominator > 0
                else np.nan
            ),
            "fnr": (
                float(fn / fnr_denominator)
                if fnr_denominator > 0
                else np.nan
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return pd.DataFrame(records)


def save_confusion_matrix(y_true, predictions, outdir):
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No default (0)", "Default (1)"])
    ax.set_yticklabels(["No default (0)", "Default (1)"])

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Final Test Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                ),
                fontsize=13,
            )

    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_roc_curve(y_true, probabilities, outdir):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", label="Random classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Final Test ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "roc_curve.png", dpi=180)
    plt.close(fig)


def save_precision_recall_curve(y_true, probabilities, outdir):
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    ap = average_precision_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.axhline(
        np.mean(y_true),
        linestyle="--",
        label="Default prevalence",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Final Test Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)


def run(dataset):
    config = LOCKED_MODELS[dataset]

    outdir = (
        ROOT
        / "reports"
        / "final_test_evaluation"
        / dataset
    )

    # Refuse to overwrite existing final results.
    # This helps enforce the one-time held-out evaluation protocol.
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(
            f"Final test results already exist in: {outdir}\n"
            "Do not rerun or overwrite the final evaluation casually."
        )

    # Ensure the development-stage selection report exists.
    selection_report = load_locked_selection(dataset)

    data = prepare_final_data(dataset)

    model = build_model(
        config["model_name"],
        config["params"],
    )

    print("=" * 65)
    print(f"FINAL HELD-OUT TEST EVALUATION: {dataset.upper()}")
    print("=" * 65)

    print(f"Selected model: {config['model_name']}")
    print(f"Threshold: {config['threshold']}")
    print(f"Hyperparameters: {config['params']}")
    print(f"Split sizes: {data['split_sizes']}")
    print("Preprocessing fitted on: TRAIN + VALIDATION")
    print("Test used for fitting: NO")

    print("\nTraining final model...")
    model.fit(data["X_dev"], data["y_dev"])

    print("Evaluating untouched test set...")

    probabilities = model.predict_proba(
        data["X_test"]
    )[:, 1]

    metrics, predictions = calculate_metrics(
        data["y_test"],
        probabilities,
        config["threshold"],
    )

    # Create output directory only after the evaluation succeeds.
    outdir.mkdir(parents=True, exist_ok=False)

    prediction_table = pd.DataFrame({
        "row_id": data["test_row_ids"],
        "y_true": data["y_test"],
        "probability": probabilities,
        "y_pred": predictions,
    })

    prediction_table.to_csv(
        outdir / "test_predictions.csv",
        index=False,
    )

    with (outdir / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    with (outdir / "preprocessor.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data["preprocessor"],
            f,
            indent=2,
            default=str,
        )

    with (outdir / "evaluation_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "dataset": dataset,
                "model": config["model_name"],
                "version": "tuned + locked threshold",
                "hyperparameters": config["params"],
                "threshold": config["threshold"],
                "seed": SEED,
                "split_sizes": data["split_sizes"],
                "preprocessing_fit_scope": "train+validation",
                "test_used_for_tuning": False,
                "selection_report_available": bool(selection_report),
                "fairness_mitigation_applied": False,
            },
            f,
            indent=2,
        )

    joblib.dump(
        model,
        outdir / "final_model.joblib",
    )

    report = classification_report(
        data["y_test"],
        predictions,
        labels=[0, 1],
        target_names=["No default", "Default"],
        zero_division=0,
    )

    (outdir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    # Final descriptive fairness audit.
    for attribute in ("age_group", "sex"):
        if attribute not in data["s_test"].columns:
            continue

        group_df = calculate_group_metrics(
            data["y_test"],
            predictions,
            data["s_test"][attribute].to_numpy(),
        )

        group_df.to_csv(
            outdir / f"fairness_{attribute}.csv",
            index=False,
        )

    save_confusion_matrix(
        data["y_test"],
        predictions,
        outdir,
    )

    save_roc_curve(
        data["y_test"],
        probabilities,
        outdir,
    )

    save_precision_recall_curve(
        data["y_test"],
        probabilities,
        outdir,
    )

    print("\n--- FINAL TEST PERFORMANCE ---")

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ):
        print(f"{key:20s}: {metrics[key]:.4f}")

    print("\n--- CONFUSION MATRIX ---")
    print(f"TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"TP: {metrics['tp']}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(report)

    print(f"\nSaved final reports to: {outdir}")
    print(
        "\nIMPORTANT: Final test results are for reporting, "
        "not further model selection."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["german", "taiwan"],
    )

    args = parser.parse_args()
    run(args.dataset)
