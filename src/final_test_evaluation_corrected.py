
"""Final held-out evaluation using ORIGINAL saved split IDs. Run once per dataset.

python src/final_test_evaluation.py --dataset german
python src/final_test_evaluation.py --dataset taiwan
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from preprocess import DATASETS, SEED, prepare, fit_preprocessor, apply_preprocessor

ROOT = Path(__file__).resolve().parents[1]
LOCKED = {
    'german': {'model': 'Logistic Regression', 'threshold': 0.40,
               'params': {'C': 0.1, 'class_weight': 'balanced'}},
    'taiwan': {'model': 'Random Forest', 'threshold': 0.55,
               'params': {'n_estimators': 300, 'min_samples_leaf': 10,
                          'class_weight': 'balanced'}},
}


def load_ids_and_validate(dataset, y, sensitive):
    folder = ROOT / 'data' / 'processed' / dataset
    ids = {}
    for split in ('train', 'val', 'test'):
        tables = {kind: pd.read_csv(folder / f'{kind}_{split}.csv') for kind in ('X', 'y', 's')}
        for kind, table in tables.items():
            if 'row_id' not in table:
                raise ValueError(f'Missing row_id in {kind}_{split}.csv')
            if table['row_id'].isna().any() or table['row_id'].duplicated().any():
                raise ValueError(f'Invalid IDs in {kind}_{split}.csv')
        original = tables['X']['row_id'].to_numpy(dtype=int)
        if not np.array_equal(original, tables['y']['row_id'].to_numpy(dtype=int)) or not np.array_equal(original, tables['s']['row_id'].to_numpy(dtype=int)):
            raise ValueError(f'X/y/s row_id alignment failed for {split}')
        if (original < 0).any() or (original >= len(y)).any():
            raise ValueError(f'Out-of-range row IDs in {split}')
        targets = [c for c in tables['y'].columns if c != 'row_id']
        if len(targets) != 1 or not np.array_equal(tables['y'][targets[0]].to_numpy(dtype=int), np.asarray(y)[original]):
            raise ValueError(f'Original labels do not match reconstructed raw rows for {split}; STOP.')
        for attr in sensitive.columns:
            if attr not in tables['s'].columns:
                raise ValueError(f'Missing sensitive attribute {attr} in s_{split}.csv')
            actual = sensitive.iloc[original][attr].fillna('unknown').astype(str).to_numpy()
            saved = tables['s'][attr].fillna('unknown').astype(str).to_numpy()
            if not np.array_equal(actual, saved):
                raise ValueError(f'Sensitive values differ for {split}/{attr}; STOP.')
        ids[split] = original
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if np.intersect1d(ids[a], ids[b]).size:
            raise ValueError(f'Overlapping {a}/{b} row IDs')
    if sum(len(v) for v in ids.values()) != len(y):
        raise ValueError('Saved splits do not cover the cleaned dataset exactly')
    return ids


def verify_lock(dataset):
    manifest = ROOT / 'reports' / 'final_model_selection' / 'selected_models.json'
    if not manifest.exists():
        raise FileNotFoundError(f'Missing locked selection: {manifest}')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    entry = data.get('datasets', {}).get(dataset)
    if not isinstance(entry, dict):
        raise ValueError(f'No selection for {dataset} in {manifest}')
    lock = LOCKED[dataset]
    if entry.get('selected_model') != lock['model'] or not np.isclose(float(entry.get('selected_threshold', -1)), lock['threshold']):
        raise ValueError(f'Locked model/threshold disagrees with selection manifest: {entry}')
    return lock


def fairness_table(y, pred, sensitive):
    rows = []
    for attr in sensitive.columns:
        for group in sorted(sensitive[attr].fillna('unknown').astype(str).unique()):
            mask = sensitive[attr].fillna('unknown').astype(str).to_numpy() == group
            yt, yp = np.asarray(y)[mask], np.asarray(pred)[mask]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            rows.append({'attribute': attr, 'group': group, 'n': int(mask.sum()),
                         'actual_default_rate': float(yt.mean()),
                         'predicted_default_rate': float(yp.mean()),
                         'fpr': float(fp/(fp+tn)) if fp+tn else np.nan,
                         'fnr': float(fn/(fn+tp)) if fn+tp else np.nan,
                         'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return pd.DataFrame(rows)


def save_plots(y, prob, pred, folder):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicted', ylabel='Actual', title='Test confusion matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.tight_layout(); fig.savefig(folder / 'confusion_matrix.png', dpi=160); plt.close(fig)
    fpr, tpr, _ = roc_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC={roc_auc_score(y, prob):.4f}')
    ax.plot([0, 1], [0, 1], '--')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC')
    ax.legend(); fig.tight_layout(); fig.savefig(folder / 'roc_curve.png', dpi=160); plt.close(fig)
    precision, recall, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f'AP={average_precision_score(y, prob):.4f}')
    ax.set(xlabel='Recall', ylabel='Precision', title='Test precision-recall')
    ax.legend(); fig.tight_layout(); fig.savefig(folder / 'precision_recall_curve.png', dpi=160); plt.close(fig)


def run(dataset):
    folder = ROOT / 'reports' / 'final_test_evaluation' / dataset
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Final results already exist: {folder}. Do not overwrite.')
    lock = verify_lock(dataset)
    cfg = dict(DATASETS[dataset])
    raw, y, sensitive, num_cols, cat_cols, _ = prepare(cfg, include_sensitive=False)
    ids = load_ids_and_validate(dataset, y, sensitive)
    dev_ids = np.concatenate([ids['train'], ids['val']])
    test_ids = ids['test']
    prep = fit_preprocessor(raw.iloc[dev_ids], num_cols, cat_cols, cfg['log_cols'])
    X_dev = apply_preprocessor(raw.iloc[dev_ids], prep)
    X_test = apply_preprocessor(raw.iloc[test_ids], prep)
    if list(X_dev.columns) != list(X_test.columns):
        raise ValueError('Train/test transformed feature columns differ')
    if not np.isfinite(X_dev.to_numpy(dtype=float)).all() or not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError('Non-finite transformed features')
    if dataset == 'german':
        model = LogisticRegression(**lock['params'], max_iter=2000, random_state=SEED)
    else:
        model = RandomForestClassifier(**lock['params'], random_state=SEED, n_jobs=-1)
    print(f'Original split IDs verified for {dataset}: train={len(ids["train"])}, val={len(ids["val"])}, test={len(test_ids)}')
    model.fit(X_dev, np.asarray(y)[dev_ids])
    y_test = np.asarray(y)[test_ids]
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= lock['threshold']).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {'dataset': dataset, 'model': lock['model'], 'threshold': lock['threshold'],
               'n_development': len(dev_ids), 'n_test': len(test_ids), 'n_features': X_dev.shape[1],
               'accuracy': float(accuracy_score(y_test, pred)),
               'precision': float(precision_score(y_test, pred, zero_division=0)),
               'recall': float(recall_score(y_test, pred, zero_division=0)),
               'f1': float(f1_score(y_test, pred, zero_division=0)),
               'roc_auc': float(roc_auc_score(y_test, prob)),
               'average_precision': float(average_precision_score(y_test, prob)),
               'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (folder / 'preprocessor.json').write_text(json.dumps(prep, indent=2, default=str), encoding='utf-8')
    (folder / 'evaluation_metadata.json').write_text(json.dumps({
        'original_saved_split_ids_used': True, 'preprocessing_fit_scope': 'original train+validation',
        'test_used_for_training_or_tuning': False, 'mitigation_applied': False,
        'hyperparameters': lock['params'], 'seed': SEED}, indent=2), encoding='utf-8')
    pd.DataFrame({'row_id': test_ids, 'y_true': y_test, 'probability': prob, 'y_pred': pred}).to_csv(folder / 'test_predictions.csv', index=False)
    fairness_table(y_test, pred, sensitive.iloc[test_ids]).to_csv(folder / 'fairness_groups.csv', index=False)
    report = classification_report(y_test, pred, labels=[0, 1], zero_division=0)
    (folder / 'classification_report.txt').write_text(report, encoding='utf-8')
    joblib.dump(model, folder / 'final_model.joblib')
    save_plots(y_test, prob, pred, folder)
    print('\nFINAL HELD-OUT TEST RESULTS')
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}')
    print('\n' + report)
    print('Saved:', folder)
    print('Do not retune based on test results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(LOCKED), required=True)
    run(parser.parse_args().dataset)
