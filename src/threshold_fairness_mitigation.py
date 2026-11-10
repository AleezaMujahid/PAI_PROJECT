"""Research-only group-threshold audit on DEVELOPMENT OOF predictions.

Run from project root:
    python src/threshold_fairness_mitigation.py --dataset german --attribute age_group
    python src/threshold_fairness_mitigation.py --dataset taiwan --attribute sex

Never loads X_test, y_test, or s_test. Group-specific decision rules are NOT
recommended for deployment in lending without legal/ethical review.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)
BASE_THRESHOLDS = {
    "german": {"logistic_regression": 0.40, "random_forest": 0.45},
    "taiwan": {"logistic_regression": 0.55, "random_forest": 0.55},
}


def metrics(y, pred, groups, group_a, group_b):
    y, pred, groups = np.asarray(y, dtype=int), np.asarray(pred, dtype=int), np.asarray(groups)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    def group_rates(g):
        mask = groups == g
        if not mask.any():
            raise ValueError(f"Missing group {g!r} in evaluation data")
        yy, pp = y[mask], pred[mask]
        neg = yy == 0
        pos = yy == 1
        return {
            "n": int(mask.sum()),
            "predicted_default_rate": float(pp.mean()),
            "fpr": float(pp[neg].mean()) if neg.any() else np.nan,
            "fnr": float((1 - pp[pos]).mean()) if pos.any() else np.nan,
        }
    a, b = group_rates(group_a), group_rates(group_b)
    return {
        "accuracy": float((pred == y).mean()), "precision": precision,
        "recall": recall, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "group_a": group_a, "group_b": group_b,
        "group_a_n": a["n"], "group_b_n": b["n"],
        "group_a_predicted_default_rate": a["predicted_default_rate"],
        "group_b_predicted_default_rate": b["predicted_default_rate"],
        "predicted_default_rate_gap": a["predicted_default_rate"] - b["predicted_default_rate"],
        "fpr_gap": a["fpr"] - b["fpr"],
        "fnr_gap": a["fnr"] - b["fnr"],
    }


def load_development_oof(dataset, model_key, attribute):
    path = ROOT / "reports" / "threshold_tuning_tuned" / f"{dataset}_{model_key}_oof.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned OOF file: {path}\nRun tuned threshold tuning first.")
    oof = pd.read_csv(path)
    needed = {"row_id", "y_true", "probability", "fold"}
    if not needed.issubset(oof.columns):
        raise ValueError(f"{path.name} must contain {sorted(needed)}")
    sensitive_parts = []
    for split in ("train", "val"):
        p = ROOT / "data" / "processed" / dataset / f"s_{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing development sensitive file: {p}")
        part = pd.read_csv(p)
        if not {"row_id", attribute}.issubset(part.columns):
            raise ValueError(f"{p} needs row_id and {attribute}")
        sensitive_parts.append(part[["row_id", attribute]])
    sensitive = pd.concat(sensitive_parts, ignore_index=True)
    if sensitive.row_id.duplicated().any() or oof.row_id.duplicated().any():
        raise ValueError("Duplicate row_id: cannot safely align predictions with sensitive groups")
    merged = oof.merge(sensitive, on="row_id", how="left", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any() or merged[attribute].isna().any():
        raise ValueError("OOF rows could not all be matched to train/val sensitive rows")
    if len(merged) != len(sensitive):
        raise ValueError("OOF row count does not equal train+val count; check scope=dev")
    if merged["y_true"].isna().any() or not merged["y_true"].isin([0, 1]).all():
        raise ValueError("y_true must be binary 0/1")
    if merged["probability"].isna().any() or not merged["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be in [0, 1]")
    if merged["fold"].nunique() < 3:
        raise ValueError("At least 3 distinct OOF folds required")
    return merged


def select_thresholds(calibration, attribute, group_a, group_b, base_threshold, max_f1_drop):
    y = calibration.y_true.to_numpy(dtype=int)
    p = calibration.probability.to_numpy(dtype=float)
    g = calibration[attribute].to_numpy()
    baseline = metrics(y, p >= base_threshold, g, group_a, group_b)
    floor = baseline["f1"] - max_f1_drop
    best = None
    for ta in THRESHOLDS:
        for tb in THRESHOLDS:
            pred = np.where(g == group_a, p >= ta, p >= tb)
            m = metrics(y, pred, g, group_a, group_b)
            if m["f1"] < floor:
                continue
            # Minimize disparity; tie-break by higher F1 and smaller deviation
            key = (abs(m["predicted_default_rate_gap"]), -m["f1"],
                   abs(ta - base_threshold) + abs(tb - base_threshold))
            if best is None or key < best[0]:
                best = (key, float(ta), float(tb))
    if best is None:
        return base_threshold, base_threshold, "fallback"
    return best[1], best[2], "selected"


def run(dataset, attribute, max_f1_drop):
    outdir = ROOT / "reports" / "threshold_fairness_mitigation" / dataset / attribute
    outdir.mkdir(parents=True, exist_ok=True)
    all_results, selected = [], []
    for model_key, model_name in (("logistic_regression", "Logistic Regression"),
                                  ("random_forest", "Random Forest")):
        data = load_development_oof(dataset, model_key, attribute)
        groups = sorted(data[attribute].astype(str).unique())
        if len(groups) != 2:
            raise ValueError(f"Expected exactly 2 {attribute} groups; found {groups}")
        if attribute == "age_group":
            group_a, group_b = "under_25", "25+"
        elif attribute == "sex":
            group_a, group_b = "female", "male"
        else:
            raise ValueError("Only age_group and sex are supported")
        if set(groups) != {group_a, group_b}:
            raise ValueError(f"Unexpected {attribute} groups: {groups}")
        base_threshold = BASE_THRESHOLDS[dataset][model_key]
        for fold in sorted(data.fold.unique()):
            calibration = data.loc[data.fold != fold]
            evaluation = data.loc[data.fold == fold]
            ta, tb, status = select_thresholds(calibration, attribute, group_a, group_b,
                                               base_threshold, max_f1_drop)
            y = evaluation.y_true.to_numpy(dtype=int)
            p = evaluation.probability.to_numpy(dtype=float)
            g = evaluation[attribute].to_numpy()
            original = p >= base_threshold
            adjusted = np.where(g == group_a, p >= ta, p >= tb)
            for version, pred in (("original", original), ("group_threshold", adjusted)):
                all_results.append({"dataset": dataset, "model": model_name,
                                    "attribute": attribute, "fold": fold, "version": version,
                                    "threshold_a": base_threshold if version == "original" else ta,
                                    "threshold_b": base_threshold if version == "original" else tb,
                                    **metrics(y, pred, g, group_a, group_b)})
            selected.append({"model": model_name, "fold": fold, "group_a": group_a,
                             "group_b": group_b, "threshold_a": ta, "threshold_b": tb,
                             "selection_status": status})
        print(f"\n{dataset.upper()} | {model_name} | {attribute}")
    results = pd.DataFrame(all_results)
    choices = pd.DataFrame(selected)
    results.to_csv(outdir / "fold_metrics.csv", index=False)
    choices.to_csv(outdir / "selected_thresholds.csv", index=False)
    # Report averages across evaluation folds; never tune based on these averages.
    cols = ["accuracy", "precision", "recall", "f1", "predicted_default_rate_gap", "fpr_gap", "fnr_gap"]
    summary = results.groupby(["model", "version"], as_index=False)[cols].mean()
    summary.to_csv(outdir / "summary.csv", index=False)
    print("\nDEVELOPMENT OOF: held-out-fold results (mean over folds)")
    print(summary.round(4).to_string(index=False))
    print("\nThreshold choices by calibration fold:")
    print(choices.to_string(index=False))
    print(f"\nSaved reports to: {outdir}")
    print("CAUTION: group-specific thresholds are a research simulation, not a deployment recommendation.")
    print("Selection uses OOF predictions from other folds; a fully nested model-training CV is stronger validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["german", "taiwan"])
    parser.add_argument("--attribute", required=True, choices=["age_group", "sex"])
    parser.add_argument("--max-f1-drop", type=float, default=0.02,
                        help="Maximum permitted F1 drop on calibration OOF data (default 0.02)")
    args = parser.parse_args()
    if not 0 <= args.max_f1_drop <= 1:
        parser.error("--max-f1-drop must be between 0 and 1")
    run(args.dataset, args.attribute, args.max_f1_drop)