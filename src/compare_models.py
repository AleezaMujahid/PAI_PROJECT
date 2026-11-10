
"""
compare_models.py

Compare baseline and tuned models using saved
development-set out-of-fold (OOF) predictions.

Generates:
  - Performance comparison CSV
  - Confusion matrix PNGs
  - ROC curve PNG
  - Precision-Recall curve PNG
  - F1 comparison PNG

Does NOT train models or use the held-out test set.

Usage:
    python src/compare_models.py --dataset german
    python src/compare_models.py --dataset taiwan
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


ROOT = Path(__file__).resolve().parent.parent

BASELINE_DIR = ROOT / "reports" / "threshold_tuning"
TUNED_DIR = ROOT / "reports" / "threshold_tuning_tuned"
OUT_ROOT = ROOT / "reports" / "model_comparison_final"

# Previously selected thresholds from baseline OOF tuning.
BASELINE_TUNED_THRESHOLDS = {
    "german": {
        "logistic_regression": 0.45,
        "random_forest": 0.25,
    },
    "taiwan": {
        "logistic_regression": 0.55,
        "random_forest": 0.30,
    },
}

# Selected thresholds after hyperparameter tuning.
TUNED_THRESHOLDS = {
    "german": {
        "logistic_regression": 0.40,
        "random_forest": 0.45,
    },
    "taiwan": {
        "logistic_regression": 0.55,
        "random_forest": 0.55,
    },
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def load_oof(folder, dataset, model_key):
    path = folder / f"{dataset}_{model_key}_oof.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing OOF file:\n{path}\n"
            "Run tune_threshold.py for this version first."
        )

    df = pd.read_csv(path)

    required = {"row_id", "y_true", "probability"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError(
            f"Missing values found in {path.name}"
        )

    if df["row_id"].duplicated().any():
        raise ValueError(
            f"Duplicate row IDs in {path.name}"
        )

    if not df["y_true"].isin([0, 1]).all():
        raise ValueError("Targets must contain only 0 and 1")

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            "Probabilities must be between 0 and 1"
        )

    return df.sort_values("row_id").reset_index(drop=True)


def check_alignment(reference, candidate):
    """Ensure both versions predict the same dev rows."""
    if not np.array_equal(
        reference["row_id"].to_numpy(),
        candidate["row_id"].to_numpy(),
    ):
        raise ValueError(
            "OOF row IDs do not match between versions."
        )

    if not np.array_equal(
        reference["y_true"].to_numpy(),
        candidate["y_true"].to_numpy(),
    ):
        raise ValueError(
            "True labels do not match between versions."
        )


def evaluate(df, dataset, version, model_name, threshold):
    y_true = df["y_true"].to_numpy(dtype=int)
    probs = df["probability"].to_numpy(dtype=float)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, preds, labels=[0, 1]
    ).ravel()

    return {
        "dataset": dataset,
        "version": version,
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(
            y_true, preds, zero_division=0
        ),
        "recall": recall_score(
            y_true, preds, zero_division=0
        ),
        "f1": f1_score(
            y_true, preds, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probs),
        "average_precision": average_precision_score(
            y_true, probs
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(
    df, threshold, title, path
):
    y_true = df["y_true"].to_numpy(dtype=int)
    probs = df["probability"].to_numpy(dtype=float)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(
        y_true, preds, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No default (0)", "Default (1)"],
    )

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_roc_curves(entries, dataset, path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, df in entries:
        y = df["y_true"].to_numpy()
        p = df["probability"].to_numpy()

        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        ax.plot(
            fpr, tpr,
            label=f"{label} (AUC={auc:.3f})"
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Random ranking",
    )

    ax.set_title(f"{dataset.title()} - ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_pr_curves(entries, dataset, path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, df in entries:
        y = df["y_true"].to_numpy()
        p = df["probability"].to_numpy()

        precision, recall, _ = precision_recall_curve(
            y, p
        )
        ap = average_precision_score(y, p)

        ax.plot(
            recall, precision,
            label=f"{label} (AP={ap:.3f})"
        )

    prevalence = entries[0][1]["y_true"].mean()
    ax.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"Default prevalence={prevalence:.3f}",
    )

    ax.set_title(
        f"{dataset.title()} - Precision-Recall Curves"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_f1_comparison(results, dataset, path):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = (
        results["model"] + "\n" + results["version"]
    )
    values = results["f1"].to_numpy()

    bars = ax.bar(range(len(values)), values)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_title(
        f"{dataset.title()} - Development OOF F1"
    )
    ax.set_ylim(0, 1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["german", "taiwan"],
        required=True,
    )
    args = parser.parse_args()

    dataset = args.dataset

    out = OUT_ROOT / dataset
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    curve_entries = []
    reference = None

    for model_key, model_name in MODEL_NAMES.items():

        baseline = load_oof(
            BASELINE_DIR, dataset, model_key
        )

        tuned = load_oof(
            TUNED_DIR, dataset, model_key
        )

        check_alignment(baseline, tuned)

        if reference is None:
            reference = baseline
        else:
            check_alignment(reference, baseline)

        configurations = [
            (
                "Baseline @0.50",
                baseline,
                0.50,
            ),
            (
                "Baseline + threshold",
                baseline,
                BASELINE_TUNED_THRESHOLDS[
                    dataset
                ][model_key],
            ),
            (
                "Tuned @0.50",
                tuned,
                0.50,
            ),
            (
                "Tuned + threshold",
                tuned,
                TUNED_THRESHOLDS[
                    dataset
                ][model_key],
            ),
        ]

        for version, df, threshold in configurations:

            row = evaluate(
                df,
                dataset,
                version,
                model_name,
                threshold,
            )
            rows.append(row)

            safe_version = (
                version.lower()
                .replace(" ", "_")
                .replace("@", "at_")
                .replace("+", "plus")
                .replace(".", "_")
            )

            filename = (
                f"{model_key}_{safe_version}_cm.png"
            )

            save_confusion_matrix(
                df,
                threshold,
                (
                    f"{dataset.title()} - {model_name}\n"
                    f"{version}, threshold={threshold:.2f}"
                ),
                out / filename,
            )

        # ROC and PR curves depend on probabilities,
        # not on the classification threshold.
        curve_entries.append(
            (f"{model_name} baseline", baseline)
        )
        curve_entries.append(
            (f"{model_name} tuned", tuned)
        )

    results = pd.DataFrame(rows)

    csv_path = out / "performance_comparison.csv"
    results.to_csv(csv_path, index=False)

    save_roc_curves(
        curve_entries,
        dataset,
        out / "roc_curves.png",
    )

    save_pr_curves(
        curve_entries,
        dataset,
        out / "precision_recall_curves.png",
    )

    save_f1_comparison(
        results,
        dataset,
        out / "f1_comparison.png",
    )

    display_cols = [
        "model",
        "version",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    print("\n" + "=" * 70)
    print(f"{dataset.upper()} - MODEL COMPARISON")
    print("=" * 70)

    print(
        results[display_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\nConfusion matrix counts:")
    print(
        results[
            ["model", "version", "tn", "fp", "fn", "tp"]
        ].to_string(index=False)
    )

    print("\nSaved reports to:")
    print(out.resolve())


if __name__ == "__main__":
    main()
