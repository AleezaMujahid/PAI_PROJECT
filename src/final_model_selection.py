
"""
Final model selection using development-only results.

This script:
1. Reads existing OOF model-comparison reports.
2. Reads fairness and mitigation summaries.
3. Selects a predefined model and threshold for each dataset.
4. Saves a reproducible selection manifest.

IMPORTANT:
- Does not load test data.
- Does not train models.
- Does not calculate new evaluation metrics.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT_DIR = REPORTS / "final_model_selection"

# Provisional decisions based on development OOF performance.
# Change only after reviewing development evidence.
SELECTIONS = {
    "german": {
        "model": "Logistic Regression",
        "version": "Tuned + threshold",
        "threshold": 0.40,
        "reason": (
            "Strong development OOF F1 and recall; "
            "fairness gaps must be documented."
        ),
    },
    "taiwan": {
        "model": "Random Forest",
        "version": "Tuned + threshold",
        "threshold": 0.55,
        "reason": (
            "Strong development OOF F1 and ROC-AUC; "
            "fairness gaps must be documented."
        ),
    },
}


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required report not found: {path}\n"
            "Check your report filenames before proceeding."
        )
    return pd.read_csv(path)


def find_comparison_report(dataset):
    """
    Locate a development model-comparison CSV.

    This script deliberately avoids guessing filenames.
    If multiple CSV files exist, the user must choose the
    correct one after inspecting its columns.
    """
    folder = REPORTS / "model_comparison_final" / dataset

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Model comparison folder not found: {folder}"
        )

    candidates = sorted(folder.glob("*.csv"))

    matches = []
    required = {
        "model",
        "version",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    for path in candidates:
        df = pd.read_csv(path, nrows=5)
        if required.issubset(df.columns):
            matches.append(path)

    if len(matches) != 1:
        found = "\n".join(str(p) for p in matches) or "None"
        raise ValueError(
            f"Expected exactly one comparison CSV for {dataset}, "
            f"but found {len(matches)}:\n{found}\n"
            "Please inspect the files and specify the correct report."
        )

    return matches[0]


def find_selected_row(df, selection, dataset):
    required = {
        "model",
        "version",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset}: comparison report missing columns: "
            f"{sorted(missing)}"
        )

    model_match = (
        df["model"].astype(str).str.strip()
        == selection["model"]
    )

    version_match = (
        df["version"].astype(str).str.strip()
        == selection["version"]
    )

    threshold_values = pd.to_numeric(
        df["threshold"], errors="raise"
    )

    threshold_match = (
        (threshold_values - selection["threshold"]).abs()
        < 1e-8
    )

    selected = df.loc[
        model_match & version_match & threshold_match
    ]

    if len(selected) != 1:
        available = df[
            ["model", "version", "threshold"]
        ].to_string(index=False)

        raise ValueError(
            f"{dataset}: expected one selected model row, "
            f"found {len(selected)}.\n"
            f"Available rows:\n{available}"
        )

    return selected.iloc[0]


def read_fairness_evidence(dataset):
    """
    Load development-only fairness comparison and
    threshold mitigation summaries when available.
    """
    evidence = {}

    fairness_dir = (
        REPORTS / "fairness_comparison" / dataset
    )

    if fairness_dir.exists():
        evidence["fairness_comparison_files"] = [
            str(p.relative_to(ROOT))
            for p in sorted(fairness_dir.glob("*.csv"))
        ]
    else:
        evidence["fairness_comparison_files"] = []

    mitigation_dir = (
        REPORTS / "threshold_fairness_mitigation"
        / dataset
    )

    mitigation = {}

    for attribute in ("age_group", "sex"):
        path = mitigation_dir / attribute / "summary.csv"

        if path.is_file():
            df = load_csv(path)

            expected = {
                "model",
                "version",
                "f1",
                "recall",
                "predicted_default_rate_gap",
                "fpr_gap",
                "fnr_gap",
            }

            if not expected.issubset(df.columns):
                raise ValueError(
                    f"Missing mitigation columns in {path}"
                )

            mitigation[attribute] = {
                "source": str(path.relative_to(ROOT)),
                "results": df.to_dict(orient="records"),
            }

    evidence["threshold_mitigation"] = mitigation

    return evidence


def build_selection(dataset):
    selection = SELECTIONS[dataset]

    comparison_path = find_comparison_report(dataset)
    comparison = load_csv(comparison_path)

    selected_row = find_selected_row(
        comparison, selection, dataset
    )

    performance = {}

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):
        value = float(selected_row[metric])

        if not 0 <= value <= 1:
            raise ValueError(
                f"{dataset}: invalid {metric}={value}"
            )

        performance[metric] = value

    if "average_precision" in selected_row.index:
        performance["average_precision"] = float(
            selected_row["average_precision"]
        )

    return {
        "dataset": dataset,
        "selected_model": selection["model"],
        "selected_version": selection["version"],
        "selected_threshold": selection["threshold"],
        "selection_reason": selection["reason"],
        "development_oof_metrics": performance,
        "comparison_source": str(
            comparison_path.relative_to(ROOT)
        ),
        "fairness_evidence": read_fairness_evidence(
            dataset
        ),
        "final_test_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Lock development-only model selection"
    )

    parser.add_argument(
        "--dataset",
        choices=["german", "taiwan", "both"],
        default="both",
    )

    args = parser.parse_args()

    datasets = (
        ["german", "taiwan"]
        if args.dataset == "both"
        else [args.dataset]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_stage": "development_only",
        "test_set_used": False,
        "datasets": {},
    }

    for dataset in datasets:
        result = build_selection(dataset)
        manifest["datasets"][dataset] = result

        print(f"\n{'=' * 55}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'=' * 55}")

        print(
            f"Selected model: "
            f"{result['selected_model']}"
        )
        print(
            f"Version: {result['selected_version']}"
        )
        print(
            f"Threshold: "
            f"{result['selected_threshold']:.2f}"
        )

        print("\nDevelopment OOF performance:")

        for name, value in (
            result["development_oof_metrics"].items()
        ):
            print(f"  {name}: {value:.4f}")

        print(
            "\nReason:",
            result["selection_reason"]
        )

        print("Test set used: NO")

    output_path = (
        OUTPUT_DIR / "selected_models.json"
    )

    with output_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            allow_nan=False,
        )

    print(f"\nSaved selection to: {output_path}")
    print(
        "IMPORTANT: This script does not evaluate "
        "the final test set."
    )


if __name__ == "__main__":
    main()
