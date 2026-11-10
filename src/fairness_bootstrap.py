
"""
fairness_bootstrap.py

Bootstrap 95% confidence intervals for fairness gaps.

Uses development-only out-of-fold predictions.
Does NOT use the held-out test set.

Run:
    python src/fairness_bootstrap.py --dataset german
    python src/fairness_bootstrap.py --dataset taiwan
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

PREDICTION_DIRS = {
    "baseline": ROOT / "reports" / "threshold_tuning",
    "tuned": ROOT / "reports" / "threshold_tuning_tuned",
}

OUTPUT_ROOT = ROOT / "reports" / "fairness_bootstrap"

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

GROUPS = {
    "german": {
        "age_group": ("under_25", "25+"),
        "sex": ("female", "male"),
        "foreign_worker": ("no", "yes"),
    },
    "taiwan": {
        "age_group": ("under_25", "25+"),
        "sex": ("female", "male"),
    },
}

METRICS = [
    "predicted_default_rate_gap",
    "fpr_gap",
    "fnr_gap",
    "disparate_impact_favorable",
]


def safe_div(a, b):
    return a / b if b != 0 else np.nan


def load_data(dataset, version, model_key):
    """Join OOF predictions to sensitive groups by row_id."""

    path = (
        PREDICTION_DIRS[version]
        / f"{dataset}_{model_key}_oof.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing OOF file: {path}"
        )

    pred = pd.read_csv(path)

    required = {"row_id", "y_true", "probability"}

    if not required.issubset(pred.columns):
        raise ValueError(
            f"OOF file missing columns: "
            f"{required - set(pred.columns)}"
        )

    folder = ROOT / "data" / "processed" / dataset

    sensitive = pd.concat(
        [
            pd.read_csv(folder / "s_train.csv"),
            pd.read_csv(folder / "s_val.csv"),
        ],
        ignore_index=True,
    )

    if pred["row_id"].duplicated().any():
        raise ValueError("Duplicate prediction row IDs")

    if sensitive["row_id"].duplicated().any():
        raise ValueError("Duplicate sensitive row IDs")

    df = pred.merge(
        sensitive,
        on="row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(df) != len(sensitive):
        raise ValueError(
            "Prediction and development row counts differ"
        )

    if not (df["_merge"] == "both").all():
        raise ValueError(
            "Some predictions have no sensitive data"
        )

    df = df.drop(columns="_merge")

    if not df["y_true"].isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1")

    if df["probability"].isna().any():
        raise ValueError("Missing probabilities")

    if not df["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be in [0, 1]")

    for attr in GROUPS[dataset]:
        if df[attr].isna().any():
            raise ValueError(
                f"Missing sensitive values: {attr}"
            )

    threshold = THRESHOLDS[dataset][version][model_key]

    df["y_pred"] = (
        df["probability"] >= threshold
    ).astype(int)

    return df


def rates(df):
    """Compute fairness rates for one demographic group."""

    y = df["y_true"].to_numpy()
    p = df["y_pred"].to_numpy()

    actual_negative = (y == 0)
    actual_positive = (y == 1)

    fp = ((p == 1) & actual_negative).sum()
    fn = ((p == 0) & actual_positive).sum()

    return {
        "predicted_default_rate": float((p == 1).mean()),
        "favorable_rate": float((p == 0).mean()),
        "fpr": safe_div(fp, actual_negative.sum()),
        "fnr": safe_div(fn, actual_positive.sum()),
    }


def calculate_gaps(df, attribute, comparison, reference):
    """Difference = comparison group minus reference group."""

    comp = df[df[attribute] == comparison]
    ref = df[df[attribute] == reference]

    if len(comp) == 0 or len(ref) == 0:
        return {metric: np.nan for metric in METRICS}

    c = rates(comp)
    r = rates(ref)

    return {
        "predicted_default_rate_gap":
            c["predicted_default_rate"]
            - r["predicted_default_rate"],

        "fpr_gap": c["fpr"] - r["fpr"],

        "fnr_gap": c["fnr"] - r["fnr"],

        "disparate_impact_favorable":
            safe_div(
                c["favorable_rate"],
                r["favorable_rate"],
            ),
    }


def bootstrap_gaps(
    df,
    attribute,
    comparison,
    reference,
    n_bootstrap,
    seed,
):
    """
    Stratified bootstrap by sensitive group.

    Sample customers with replacement WITHIN each
    group, preserving group sample sizes.

    The model predictions and thresholds remain fixed.
    This estimates uncertainty conditional on the
    fitted OOF prediction procedure.
    """

    comp = df[df[attribute] == comparison]
    ref = df[df[attribute] == reference]

    if comp.empty or ref.empty:
        raise ValueError(
            f"Missing groups for {attribute}"
        )

    observed = calculate_gaps(
        df, attribute, comparison, reference
    )

    rng = np.random.default_rng(seed)

    comp_indices = np.arange(len(comp))
    ref_indices = np.arange(len(ref))

    bootstrap_values = {
        metric: [] for metric in METRICS
    }

    for _ in range(n_bootstrap):
        comp_sample = comp.iloc[
            rng.choice(
                comp_indices,
                size=len(comp),
                replace=True,
            )
        ]

        ref_sample = ref.iloc[
            rng.choice(
                ref_indices,
                size=len(ref),
                replace=True,
            )
        ]

        c = rates(comp_sample)
        r = rates(ref_sample)

        values = {
            "predicted_default_rate_gap":
                c["predicted_default_rate"]
                - r["predicted_default_rate"],

            "fpr_gap":
                c["fpr"] - r["fpr"],

            "fnr_gap":
                c["fnr"] - r["fnr"],

            "disparate_impact_favorable":
                safe_div(
                    c["favorable_rate"],
                    r["favorable_rate"],
                ),
        }

        for metric, value in values.items():
            if np.isfinite(value):
                bootstrap_values[metric].append(value)

    results = []

    for metric in METRICS:
        values = np.asarray(
            bootstrap_values[metric],
            dtype=float,
        )

        # Undefined bootstrap rates can arise when
        # a resample has no positive or negative labels.
        # We report how many usable draws remain.
        if len(values) < 0.9 * n_bootstrap:
            low = np.nan
            high = np.nan
            warning = "Insufficient valid bootstrap draws"
        else:
            low, high = np.percentile(
                values, [2.5, 97.5]
            )
            warning = ""

        null_value = (
            1.0 if metric == "disparate_impact_favorable"
            else 0.0
        )

        excludes_null = (
            bool(low > null_value or high < null_value)
            if np.isfinite(low) and np.isfinite(high)
            else False
        )

        results.append({
            "attribute": attribute,
            "comparison_group": comparison,
            "reference_group": reference,
            "comparison_n": len(comp),
            "reference_n": len(ref),
            "metric": metric,
            "estimate": observed[metric],
            "ci_lower": low,
            "ci_upper": high,
            "valid_bootstrap_draws": len(values),
            "n_bootstrap": n_bootstrap,
            "ci_excludes_null": excludes_null,
            "warning": warning,
        })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["german", "taiwan"],
        required=True,
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.n_bootstrap < 100:
        raise ValueError(
            "Use at least 100 bootstrap iterations"
        )

    out = OUTPUT_ROOT / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_key, model_name in MODELS.items():
        for version in ["baseline", "tuned"]:

            df = load_data(
                args.dataset,
                version,
                model_key,
            )

            print(
                f"\n{args.dataset} | "
                f"{model_name} | {version}"
            )

            for attribute, (
                comparison, reference
            ) in GROUPS[args.dataset].items():

                results = bootstrap_gaps(
                    df=df,
                    attribute=attribute,
                    comparison=comparison,
                    reference=reference,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )

                for row in results:
                    row.update({
                        "dataset": args.dataset,
                        "model": model_name,
                        "version": version,
                        "threshold": THRESHOLDS[
                            args.dataset
                        ][version][model_key],
                    })

                all_results.extend(results)

    result_df = pd.DataFrame(all_results)

    output_file = out / "fairness_confidence_intervals.csv"

    result_df.to_csv(
        output_file,
        index=False,
    )

    display_cols = [
        "model",
        "version",
        "attribute",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper",
        "ci_excludes_null",
    ]

    print("\n" + "=" * 80)
    print(
        f"{args.dataset.upper()} "
        "- BOOTSTRAP 95% CONFIDENCE INTERVALS"
    )
    print("=" * 80)

    print(
        result_df[display_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved to:")
    print(output_file.resolve())


if __name__ == "__main__":
    main()
