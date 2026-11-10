"""
fairness.py  -  group-wise fairness audit with bootstrap confidence intervals.

Run from the project root:
    python src/fairness.py --dataset german --preds predictions/german_logreg_all_oof.csv

Input: a predictions CSV with columns  index, y_true, y_pred, y_prob, model_name
       ('index' = row_id, the same ids as in data/processed/<dataset>/s_*.csv)

Vocabulary (positive class = DEFAULT, so a prediction of 1 means the loan is DENIED)
    denial_rate   share of the group predicted to default (loan denied)
    FPR           good applicants wrongly denied      <- the main harm to applicants
    FNR           real defaulters wrongly approved    <- the lender's risk
    denial_gap    denial_rate(group) - denial_rate(reference)         (0 = parity)
    disparate_impact  approval_rate(group) / approval_rate(reference) (1 = parity,
                  below ~0.8 is the common "four-fifths" warning line)
    fpr_gap, fnr_gap  group minus reference                           (0 = parity)
    Equal opportunity (for approval) = small fpr_gap. Equalized odds = small fpr_gap AND fnr_gap.

Reliability
    A comparison is flagged UNRELIABLE when either group has fewer than 50 rows or fewer
    than 10 rows of either outcome class. Attributes listed as 'exploratory' in the
    dataset config (e.g. foreign_worker) are labelled so and are not headline results.

Bootstrap
    Rows are resampled with replacement n_boot times; the 2.5th and 97.5th percentiles of
    each metric give a 95% interval. If the interval excludes the "no difference" value
    (0, or 1 for disparate_impact) the gap is unlikely to be noise.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import preprocess as P

ROOT = Path(__file__).resolve().parent.parent
MIN_GROUP_N = 50
MIN_CLASS_N = 10
METRICS = ["denial_gap", "disparate_impact", "fpr_gap", "fnr_gap"]
NULL_VALUE = {"denial_gap": 0.0, "disparate_impact": 1.0, "fpr_gap": 0.0, "fnr_gap": 0.0}


def group_rates(y, p):
    """Counts and rates for one group. y = actual default, p = predicted default (0/1)."""
    n = len(y)
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tp = int(((y == 1) & (p == 1)).sum())
    return {"n": n, "n_default": tp + fn, "n_good": tn + fp,
            "denial_rate": p.mean() if n else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "fnr": fn / (fn + tp) if fn + tp else np.nan}


def gaps(ra, rr):
    """Metrics of group `ra` compared with reference group `rr`."""
    appr_r = 1 - rr["denial_rate"]
    return {"denial_gap": ra["denial_rate"] - rr["denial_rate"],
            "disparate_impact": (1 - ra["denial_rate"]) / appr_r if appr_r > 0 else np.nan,
            "fpr_gap": ra["fpr"] - rr["fpr"],
            "fnr_gap": ra["fnr"] - rr["fnr"]}


def _reliability(name_a, ra, name_r, rr):
    reasons = []
    for name, r in ((name_a, ra), (name_r, rr)):
        if r["n"] < MIN_GROUP_N:
            reasons.append(f"{name}: only {r['n']} rows")
        elif min(r["n_default"], r["n_good"]) < MIN_CLASS_N:
            few = "defaulters" if r["n_default"] < r["n_good"] else "good applicants"
            reasons.append(f"{name}: only {min(r['n_default'], r['n_good'])} {few}")
    return ("UNRELIABLE (" + "; ".join(reasons) + ")") if reasons else "ok"


def audit(y_true, y_pred, groups, reference_groups, headline=(), exploratory=(),
          n_boot=1000, seed=42):
    """Return (rates_df, gaps_df). `groups` is a DataFrame aligned with y_true / y_pred."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    n = len(y)
    rng = np.random.default_rng(seed)

    attrs = {}
    for attr, ref in reference_groups.items():
        g = np.asarray(groups[attr].astype(str))
        labels = sorted(set(g))
        if ref not in labels:
            raise ValueError(f"reference group '{ref}' not found in '{attr}': {labels}")
        attrs[attr] = (g, ref, [v for v in labels if v not in (ref, "unknown")])

    # point estimates
    rates_rows, point, boot = [], {}, {}
    for attr, (g, ref, others) in attrs.items():
        for lab in [ref] + others + (["unknown"] if "unknown" in set(g) else []):
            r = group_rates(y[g == lab], p[g == lab])
            rates_rows.append({"attribute": attr, "group": lab, "is_reference": lab == ref, **r})
        rr = group_rates(y[g == ref], p[g == ref])
        for lab in others:
            point[(attr, lab)] = gaps(group_rates(y[g == lab], p[g == lab]), rr)
            boot[(attr, lab)] = {m: np.full(n_boot, np.nan) for m in METRICS}

    # bootstrap: the SAME resample is used for every attribute
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        for attr, (g, ref, others) in attrs.items():
            gb = g[idx]
            rr = group_rates(yb[gb == ref], pb[gb == ref])
            for lab in others:
                m = gaps(group_rates(yb[gb == lab], pb[gb == lab]), rr)
                for k in METRICS:
                    boot[(attr, lab)][k][b] = m[k]

    rates_df = pd.DataFrame(rates_rows)
    rates_df["role"] = rates_df["attribute"].map(
        lambda a: "headline" if a in headline else "exploratory" if a in exploratory else "other")

    gap_rows = []
    rate_lookup = {(r.attribute, r.group): r._asdict() for r in rates_df.itertuples()}
    for (attr, lab), est in point.items():
        g, ref, _ = attrs[attr]
        status = _reliability(lab, rate_lookup[(attr, lab)], ref, rate_lookup[(attr, ref)])
        role = "headline" if attr in headline else "exploratory" if attr in exploratory else "other"
        for k in METRICS:
            arr = boot[(attr, lab)][k]
            ok = ~np.isnan(arr)
            lo, hi = (np.percentile(arr[ok], [2.5, 97.5]) if ok.sum() >= 20 else (np.nan, np.nan))
            null = NULL_VALUE[k]
            gap_rows.append({
                "attribute": attr, "group": lab, "reference": ref, "role": role, "metric": k,
                "estimate": est[k], "ci_low": lo, "ci_high": hi,
                "ci_excludes_no_difference": bool(not np.isnan(lo) and (lo > null or hi < null)),
                "valid_bootstrap_share": float(ok.mean()), "reliability": status})
    return rates_df, pd.DataFrame(gap_rows)


def print_report(rates_df, gaps_df, n_boot):
    print(f"\nFAIRNESS AUDIT  (positive class = default = loan denied; 95% bootstrap CI, B={n_boot})")
    order = {"headline": 0, "other": 1, "exploratory": 2}
    for attr in sorted(rates_df.attribute.unique(),
                       key=lambda a: order[rates_df[rates_df.attribute == a].role.iloc[0]]):
        sub = rates_df[rates_df.attribute == attr]
        role = sub.role.iloc[0].upper()
        tag = "  <-- EXPLORATORY, small sample: do not use as a headline result" if role == "EXPLORATORY" else ""
        print(f"\n[{attr}]  ({role}){tag}")
        print(f"  {'group':10s} {'n':>5s} {'defaults':>8s} {'denial':>7s} {'FPR':>7s} {'FNR':>7s}")
        for r in sub.itertuples():
            ref = " (reference)" if r.is_reference else ""
            print(f"  {r.group:10s} {r.n:5d} {r.n_default:8d} {r.denial_rate:7.3f} "
                  f"{r.fpr:7.3f} {r.fnr:7.3f}{ref}")
        for lab in gaps_df[gaps_df.attribute == attr].group.unique():
            g = gaps_df[(gaps_df.attribute == attr) & (gaps_df.group == lab)]
            print(f"  gaps: {lab} vs {g.reference.iloc[0]}   reliability: {g.reliability.iloc[0]}")
            for r in g.itertuples():
                star = "  * CI excludes no-difference" if r.ci_excludes_no_difference else ""
                print(f"    {r.metric:17s} {r.estimate:+.3f}  [{r.ci_low:+.3f}, {r.ci_high:+.3f}]{star}")


def load_for_dataset(dataset, preds_path):
    cfg = P.DATASETS[dataset]
    pdir = Path(cfg["out"])
    s = pd.concat([pd.read_csv(pdir / f"s_{n}.csv", index_col="row_id")
                   for n in ("train", "val", "test")]).sort_index()
    preds = pd.read_csv(preds_path)
    return cfg, preds, s.loc[preds["index"].to_numpy()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german", choices=sorted(P.DATASETS))
    ap.add_argument("--preds", required=True, help="CSV: index, y_true, y_pred, y_prob, model_name")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg, preds, s = load_for_dataset(a.dataset, a.preds)
    rates_df, gaps_df = audit(preds["y_true"], preds["y_pred"], s.reset_index(drop=True),
                              cfg["reference_groups"], cfg["headline_attrs"],
                              cfg["exploratory_attrs"], a.boot, a.seed)
    print_report(rates_df, gaps_df, a.boot)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    stem = Path(a.preds).stem
    rates_df.to_csv(out / f"fairness_rates_{stem}.csv", index=False)
    gaps_df.to_csv(out / f"fairness_gaps_{stem}.csv", index=False)
    print(f"\nsaved: reports/fairness_rates_{stem}.csv, reports/fairness_gaps_{stem}.csv")