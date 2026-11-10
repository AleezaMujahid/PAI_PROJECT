
"""
fairness_compare.py

Development-only fairness audit for German and Taiwan.

Uses:
- Baseline and tuned OOF probabilities
- Previously selected thresholds
- Sensitive attributes from s_train.csv and s_val.csv

Does NOT use held-out test data.

Run:
    python src/fairness_compare.py --dataset german
    python src/fairness_compare.py --dataset taiwan
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

BASELINE_DIR = ROOT / "reports" / "threshold_tuning"
TUNED_DIR = ROOT / "reports" / "threshold_tuning_tuned"
OUT_ROOT = ROOT / "reports" / "fairness_comparison"

MODELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}

THRESHOLDS = {
    "german": {
        "baseline": {
            "logistic_regression": 0.45,
            "random_forest": 0.25,
        },
        "tuned": {
            "logistic_regression": 0.40,
            "random_forest": 0.45,
        },
    },
    "taiwan": {
        "baseline": {
            "logistic_regression": 0.55,
            "random_forest": 0.30,
        },
        "tuned": {
            "logistic_regression": 0.55,
            "random_forest": 0.55,
        },
    },
}

ATTRIBUTES = {
    "german": ["age_group", "sex", "foreign_worker"],
    "taiwan": ["age_group", "sex"],
}


def load_sensitive(dataset):
    folder = ROOT / "data" / "processed" / dataset

    train = pd.read_csv(folder / "s_train.csv")
    val = pd.read_csv(folder / "s_val.csv")

    sensitive = pd.concat(
        [train, val], ignore_index=True
    )

    if "row_id" not in sensitive.columns:
        raise ValueError("Sensitive data missing row_id")

    if sensitive["row_id"].duplicated().any():
        raise ValueError(
            "Duplicate row_id in sensitive data"
        )

    for attr in ATTRIBUTES[dataset]:
        if attr not in sensitive.columns:
            raise ValueError(
                f"Missing sensitive attribute: {attr}"
            )

    return sensitive


def load_predictions(dataset, model_key, version):
    folder = (
        BASELINE_DIR if version == "baseline"
        else TUNED_DIR
    )

    path = folder / f"{dataset}_{model_key}_oof.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    required = ["row_id", "y_true", "probability"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{path.name}: missing {col}"
            )

    if df["row_id"].duplicated().any():
        raise ValueError(
            f"{path.name}: duplicate row_id"
        )

    if df[required].isna().any().any():
        raise ValueError(
            f"{path.name}: missing prediction data"
        )

    if not df["y_true"].isin([0, 1]).all():
        raise ValueError(
            f"{path.name}: invalid target values"
        )

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            f"{path.name}: invalid probabilities"
        )

    return df


def merge_predictions(predictions, sensitive):
    # Merge by customer row_id, never by row position.
    merged = predictions.merge(
        sensitive,
        on="row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "Some predictions have no matching "
            "sensitive-attribute row."
        )

    if len(merged) != len(sensitive):
        raise ValueError(
            "Prediction rows and development "
            "sensitive rows do not match."
        )

    merged = merged.drop(columns="_merge")

    return merged


def safe_div(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def group_metrics(df, attribute):
    results = []

    for group, data in df.groupby(
        attribute, dropna=False
    ):
        y = data["y_true"].to_numpy()
        pred = data["y_pred"].to_numpy()

        tp = int(((y == 1) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())

        results.append({
            "attribute": attribute,
            "group": str(group),
            "n": len(data),
            "actual_default_rate": float(
                (y == 1).mean()
            ),
            "predicted_default_rate": float(
                (pred == 1).mean()
            ),
            "favorable_prediction_rate": float(
                (pred == 0).mean()
            ),
            "fpr": safe_div(fp, fp + tn),
            "fnr": safe_div(fn, fn + tp),
            "tpr": safe_div(tp, tp + fn),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    return results


def reference_group(dataset, attribute, groups):
    preferences = {
        "age_group": ["25+", "25_plus"],
        "sex": ["male"],
        "foreign_worker": ["yes"],
    }

    for candidate in preferences.get(attribute, []):
        if candidate in groups:
            return candidate

    return sorted(groups)[0]


def calculate_gaps(group_df, dataset):
    """
    Difference = comparison group - reference group.

    Disparate impact here uses favorable outcomes:
    P(predicted non-default | comparison)
    /
    P(predicted non-default | reference)

    This is a prediction-based proxy, NOT an
    observed loan-approval rate.
    """
    rows = []

    for attribute, subset in group_df.groupby("attribute"):
        groups = subset["group"].tolist()

        if len(groups) < 2:
            continue

        ref_name = reference_group(
            dataset, attribute, groups
        )

        ref = subset.loc[
            subset["group"] == ref_name
        ].iloc[0]

        for _, comp in subset.iterrows():
            if comp["group"] == ref_name:
                continue

            ref_favorable = ref[
                "favorable_prediction_rate"
            ]

            comp_favorable = comp[
                "favorable_prediction_rate"
            ]

            rows.append({
                "attribute": attribute,
                "reference_group": ref_name,
                "comparison_group": comp["group"],
                "reference_n": int(ref["n"]),
                "comparison_n": int(comp["n"]),
                "predicted_default_rate_gap":
                    comp["predicted_default_rate"]
                    - ref["predicted_default_rate"],
                "fpr_gap":
                    comp["fpr"] - ref["fpr"],
                "fnr_gap":
                    comp["fnr"] - ref["fnr"],
                "equal_opportunity_gap":
                    comp["tpr"] - ref["tpr"],
                "disparate_impact_favorable":
                    safe_div(
                        comp_favorable,
                        ref_favorable,
                    ),
            })

    return rows


def save_group_plot(
    group_df, dataset, attribute, metric, out
):
    subset = group_df[
        group_df["attribute"] == attribute
    ].copy()

    if subset.empty:
        return

    subset["label"] = (
        subset["model"] + "\n"
        + subset["version"] + "\n"
        + subset["group"]
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(
        range(len(subset)),
        subset[metric],
    )

    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels(
        subset["label"],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"{dataset.title()} - {attribute} - {metric}"
    )
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
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

    sensitive = load_sensitive(dataset)

    group_rows = []
    gap_rows = []
    reference_predictions = None

    for model_key, model_name in MODELS.items():
        for version in ["baseline", "tuned"]:
            predictions = load_predictions(
                dataset, model_key, version
            )

            # Verify all versions cover identical
            # customers with identical true labels.
            current = predictions[
                ["row_id", "y_true"]
            ].sort_values("row_id").reset_index(
                drop=True
            )

            if reference_predictions is None:
                reference_predictions = current
            elif not current.equals(
                reference_predictions
            ):
                raise ValueError(
                    "Prediction row IDs or labels differ "
                    "between models/versions."
                )

            merged = merge_predictions(
                predictions, sensitive
            )

            threshold = THRESHOLDS[
                dataset
            ][version][model_key]

            merged["y_pred"] = (
                merged["probability"] >= threshold
            ).astype(int)

            for attribute in ATTRIBUTES[dataset]:
                if merged[attribute].isna().any():
                    raise ValueError(
                        f"Missing values in {attribute}"
                    )

                metrics = group_metrics(
                    merged, attribute
                )

                for row in metrics:
                    row.update({
                        "dataset": dataset,
                        "model": model_name,
                        "version": version,
                        "threshold": threshold,
                    })

                group_rows.extend(metrics)

                metric_frame = pd.DataFrame(metrics)
                gaps = calculate_gaps(
                    metric_frame, dataset
                )

                for row in gaps:
                    row.update({
                        "dataset": dataset,
                        "model": model_name,
                        "version": version,
                        "threshold": threshold,
                    })

                gap_rows.extend(gaps)

    groups_df = pd.DataFrame(group_rows)
    gaps_df = pd.DataFrame(gap_rows)

    groups_df.to_csv(
        out / "group_metrics.csv",
        index=False,
    )

    gaps_df.to_csv(
        out / "fairness_gaps.csv",
        index=False,
    )

    for attribute in ATTRIBUTES[dataset]:
        for metric in [
            "predicted_default_rate",
            "fpr",
            "fnr",
        ]:
            save_group_plot(
                groups_df,
                dataset,
                attribute,
                metric,
                out / f"{attribute}_{metric}.png",
            )

    print("\n" + "=" * 75)
    print(f"{dataset.upper()} - FAIRNESS GROUP METRICS")
    print("=" * 75)

    columns = [
        "model",
        "version",
        "attribute",
        "group",
        "n",
        "predicted_default_rate",
        "fpr",
        "fnr",
    ]

    print(
        groups_df[columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("FAIRNESS GAPS")
    print("=" * 75)

    if gaps_df.empty:
        print("No group comparisons available.")
    else:
        print(
            gaps_df[
                [
                    "model",
                    "version",
                    "attribute",
                    "comparison_group",
                    "predicted_default_rate_gap",
                    "fpr_gap",
                    "fnr_gap",
                    "disparate_impact_favorable",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    print("\nReports saved to:")
    print(out.resolve())


if __name__ == "__main__":
    main()
