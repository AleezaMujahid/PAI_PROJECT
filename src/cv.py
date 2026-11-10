"""
cv.py  -  leak-free cross-validation folds for model selection and fairness auditing.

WHY this file exists
    One 70/15/15 split of 1000 rows is noisy (test accuracy +/- 7 points, tiny groups).
    Cross-validation reuses every row for validation once, so results are more stable.

WHY preprocessing is redone INSIDE every fold
    Medians, means, stds and category lists must be learned from the fold's TRAINING rows
    only. Otherwise the validation rows leak into the scaler.

Two scopes
    scope="dev"  folds over train+val (850 rows). Use this to compare models and tune.
                 The 15% test split from preprocess.py is NEVER touched here.
    scope="all"  folds over all rows (1000). Every row gets an out-of-fold prediction,
                 which is what the fairness audit should use (more rows per group).

Held-out test:  final_split() trains on dev and gives the same untouched test rows as
                data/processed/<dataset>/X_test.csv. Look at it once, at the very end.

Usage
    from cv import cv_folds, final_split
    for fold in cv_folds(cfg, scope="dev", n_splits=5):
        model.fit(fold.X_train, fold.y_train); p = model.predict_proba(fold.X_val)[:, 1]
"""
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import preprocess as P

Fold = namedtuple("Fold", "k X_train y_train X_val y_val s_val val_index")


def _setup(cfg, include_sensitive, seed):
    df, y, sens, num_cols, cat_cols, _ = P.prepare(cfg, include_sensitive)
    sp = P.three_way_split(y, seed)                      # same split as the processed files
    dev_idx = np.sort(np.concatenate([sp["train"], sp["val"]]))
    return df, y, sens, num_cols, cat_cols, dev_idx, sp["test"]


def _frame(df, y, sens, idx, prep):
    return (P.apply_preprocessor(df.iloc[idx], prep),
            pd.Series(y[idx], index=df.index[idx], name="default"),
            sens.iloc[idx])


def cv_folds(cfg, scope="dev", n_splits=5, repeats=1, seed=P.SEED, include_sensitive=False):
    """Yield Fold tuples. With repeats > 1 each row is validated `repeats` times."""
    if scope not in {"dev", "all"}:
        raise ValueError("scope must be 'dev' or 'all'")
    df, y, sens, num_cols, cat_cols, dev_idx, _ = _setup(cfg, include_sensitive, seed)
    pool = dev_idx if scope == "dev" else np.arange(len(y))

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
    for k, (tr, va) in enumerate(splitter.split(pool, y[pool])):
        tr_idx, va_idx = pool[tr], pool[va]
        prep = P.fit_preprocessor(df.iloc[tr_idx], num_cols, cat_cols, cfg["log_cols"])  # fold-train only
        X_tr, y_tr, _ = _frame(df, y, sens, tr_idx, prep)
        X_va, y_va, s_va = _frame(df, y, sens, va_idx, prep)
        yield Fold(k, X_tr, y_tr, X_va, y_va, s_va, va_idx)


def final_split(cfg, seed=P.SEED, include_sensitive=False):
    """Train on dev (train+val); evaluate ONCE on the held-out test rows."""
    df, y, sens, num_cols, cat_cols, dev_idx, test_idx = _setup(cfg, include_sensitive, seed)
    prep = P.fit_preprocessor(df.iloc[dev_idx], num_cols, cat_cols, cfg["log_cols"])
    X_dev, y_dev, s_dev = _frame(df, y, sens, dev_idx, prep)
    X_te, y_te, s_te = _frame(df, y, sens, test_idx, prep)
    return {"X_dev": X_dev, "y_dev": y_dev, "s_dev": s_dev,
            "X_test": X_te, "y_test": y_te, "s_test": s_te, "test_index": test_idx}