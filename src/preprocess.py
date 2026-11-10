"""Credit default preprocessing for German Credit and Taiwan Credit Card datasets.

IMPORTANT: This is a reconstructed implementation, NOT a byte-for-byte restoration
of the user's original preprocess.py. Do not use it for final held-out evaluation
until the reconstructed split row IDs are verified against existing processed files.

CLI:
 python src/preprocess.py --dataset german --raw data/raw/credit_g.csv
 python src/preprocess.py --dataset taiwan --raw "data/raw/default of credit card clients.csv"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def read_german(path):
    return pd.read_csv(path)


def read_taiwan(path):
    path = Path(path)
    df = pd.read_excel(path, header=1) if path.suffix.lower() in ('.xls', '.xlsx') else pd.read_csv(path)
    if 'default payment next month' not in df.columns and 'default' not in df.columns:
        # Some Excel exports contain a second header row.
        df2 = pd.read_excel(path, header=1) if path.suffix.lower() in ('.xls', '.xlsx') else pd.read_csv(path, header=1)
        if 'default payment next month' in df2.columns:
            df = df2
    return df


def german_sensitive(df):
    age = pd.to_numeric(df['age'], errors='coerce')
    out = pd.DataFrame({'age_group': np.where(age < 25, 'under_25', '25+')}, index=df.index)
    if 'personal_status' in df.columns:
        # OpenML credit-g encodes sex within personal_status.
        mapping = {'male div/sep': 'male', 'male single': 'male', 'male mar/wid': 'male',
                   'female div/dep/mar': 'female', 'female single': 'female'}
        out['sex'] = df['personal_status'].astype(str).map(mapping).fillna('unknown')
    if 'foreign_worker' in df.columns:
        out['foreign_worker'] = df['foreign_worker'].astype(str)
    return out


def taiwan_sensitive(df):
    sex = pd.to_numeric(df['sex'], errors='coerce')
    age = pd.to_numeric(df['age'], errors='coerce')
    out = pd.DataFrame({'age_group': np.where(age < 25, 'under_25', '25+'),
                        'sex': sex.map({1: 'male', 2: 'female'}).fillna('unknown')}, index=df.index)
    if 'marriage' in df.columns:
        out['marriage'] = df['marriage'].astype(str)
    return out


DATASETS = {
    'german': dict(raw=ROOT / 'data/raw/credit_g.csv', out=ROOT / 'data/processed/german',
                   reader=read_german, target='class', positive_label='bad',
                   id_cols=[], numeric_cols=['duration', 'credit_amount', 'installment_commitment',
                   'residence_since', 'age', 'existing_credits', 'num_dependents'],
                   log_cols=['credit_amount'], value_fixes={},
                   sensitive_source_cols=['age', 'personal_status', 'foreign_worker'],
                   build_sensitive=german_sensitive),
    'taiwan': dict(raw=ROOT / 'data/raw/default of credit card clients.csv',
                   out=ROOT / 'data/processed/taiwan', reader=read_taiwan,
                   target='default', positive_label='1', id_cols=['id'],
                   numeric_cols=(['limit_bal', 'age'] + ['pay_0'] +
                                 [f'pay_{i}' for i in range(2, 7)] +
                                 [f'bill_amt{i}' for i in range(1, 7)] +
                                 [f'pay_amt{i}' for i in range(1, 7)]),
                   log_cols=['limit_bal'] + [f'pay_amt{i}' for i in range(1, 7)],
                   value_fixes={'education': {'0': '4', '5': '4', '6': '4'},
                                'marriage': {'0': '3'}},
                   sensitive_source_cols=['sex', 'age', 'marriage'],
                   build_sensitive=taiwan_sensitive),
}


def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    df = df.rename(columns={'default_payment_next_month': 'default'})
    return df


def prepare(cfg, include_sensitive=False):
    """Return clean predictors, binary target, sensitive groups, numeric/categorical columns, report."""
    raw = cfg['reader'](cfg['raw'])
    raw = _normalize_columns(raw)
    n_raw = len(raw)
    fixes = {}
    for col, mapping in cfg.get('value_fixes', {}).items():
        if col in raw.columns:
            before = raw[col].astype('string')
            fixes[col] = int(before.isin(list(mapping)).sum())
            raw[col] = before.replace(mapping)
    if cfg['target'] not in raw:
        raise ValueError(f"Missing target {cfg['target']}; found {list(raw.columns)}")
    # Deduplicate on all substantive columns; ID is not a predictive variable.
    dedup_cols = [c for c in raw.columns if c not in cfg.get('id_cols', [])]
    before = len(raw)
    raw = raw.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    n_duplicates = before - len(raw)
    target_raw = raw[cfg['target']]
    keep = target_raw.notna()
    missing_target = int((~keep).sum())
    raw = raw.loc[keep].reset_index(drop=True)
    y = (raw[cfg['target']].astype(str).str.strip().str.lower() ==
         str(cfg['positive_label']).lower()).astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError('Target must contain both 0 and 1 classes')
    sens = cfg['build_sensitive'](raw).reset_index(drop=True)
    excluded = {cfg['target'], *cfg.get('id_cols', [])}
    if not include_sensitive:
        excluded.update(cfg.get('sensitive_source_cols', []))
    df = raw.drop(columns=[c for c in excluded if c in raw.columns]).copy()
    numeric = [c for c in cfg['numeric_cols'] if c in df.columns]
    # German categorical features and Taiwan EDUCATION are one-hot encoded.
    categorical = [c for c in df.columns if c not in numeric]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in categorical:
        df[col] = df[col].astype('string')
    report = {'rows_raw': n_raw, 'rows_clean': len(df),
              'duplicates_removed': n_duplicates,
              'rows_missing_target_dropped': missing_target,
              'value_fixes_applied': fixes,
              'missing_before_imputation': {k: int(v) for k, v in df.isna().sum().items() if v}}
    return df, y, sens, numeric, categorical, report


def three_way_split(y, seed=SEED):
    """70/15/15 stratified split; verify row IDs against original before final test."""
    idx = np.arange(len(y))
    tr, rem = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rem, test_size=0.50, stratify=np.asarray(y)[rem], random_state=seed)
    return {'train': np.sort(tr), 'val': np.sort(va), 'test': np.sort(te)}


def _signed_log1p(s):
    return np.sign(s) * np.log1p(np.abs(s))


def fit_preprocessor(df, num_cols, cat_cols, log_cols):
    """Learn all imputation, log, scaling and category rules on fitting rows only."""
    spec = {'numeric_cols': list(num_cols), 'categorical_cols': list(cat_cols),
            'log_cols': [c for c in log_cols if c in num_cols],
            'numeric': {}, 'categorical': {}}
    for c in num_cols:
        values = pd.to_numeric(df[c], errors='coerce')
        med = float(values.median()) if values.notna().any() else 0.0
        transformed = values.fillna(med).astype(float)
        if c in spec['log_cols']:
            transformed = _signed_log1p(transformed)
        mean = float(transformed.mean())
        std = float(transformed.std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        spec['numeric'][c] = {'median': med, 'mean': mean, 'std': std}
    for c in cat_cols:
        values = df[c].astype('string')
        mode = values.mode(dropna=True)
        fill = str(mode.iloc[0]) if len(mode) else '__missing__'
        cats = sorted(values.fillna(fill).astype(str).unique().tolist())
        spec['categorical'][c] = {'mode': fill, 'categories': cats}
    return spec


def apply_preprocessor(df, prep):
    output = pd.DataFrame(index=df.index)
    for c in prep['numeric_cols']:
        info = prep['numeric'][c]
        x = pd.to_numeric(df[c], errors='coerce').fillna(info['median']).astype(float)
        if c in prep['log_cols']:
            x = _signed_log1p(x)
        output[c] = (x - info['mean']) / info['std']
    for c in prep['categorical_cols']:
        info = prep['categorical'][c]
        x = df[c].astype('string').fillna(info['mode']).astype(str)
        for cat in info['categories']:
            output[f'{c}={cat}'] = (x == cat).astype(float)
    return output


def _save_split(out, name, X, y, s, indices):
    pd.concat([pd.Series(indices, name='row_id'), X.reset_index(drop=True)], axis=1).to_csv(
        out / f'X_{name}.csv', index=False)
    pd.DataFrame({'row_id': indices, 'default': y[indices]}).to_csv(out / f'y_{name}.csv', index=False)
    pd.concat([pd.Series(indices, name='row_id'), s.iloc[indices].reset_index(drop=True)], axis=1).to_csv(
        out / f's_{name}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--raw', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--verify-only', action='store_true',
                        help='Compare reconstructed split row IDs to existing files; do not write.')
    args = parser.parse_args()
    cfg = dict(DATASETS[args.dataset])
    if args.raw:
        cfg['raw'] = Path(args.raw)
    if args.out:
        cfg['out'] = Path(args.out)
    df, y, sens, numeric, categorical, report = prepare(cfg)
    sp = three_way_split(y)
    out = Path(cfg['out'])
    if args.verify_only:
        for split, ids in sp.items():
            p = out / f'X_{split}.csv'
            if not p.exists():
                raise FileNotFoundError(p)
            saved = pd.read_csv(p, usecols=['row_id'])['row_id'].to_numpy()
            if not np.array_equal(saved, ids):
                if set(saved) == set(ids):
                    print(f'{split}: same IDs but different order')
                else:
                    print(f'{split}: MISMATCH — reconstructed split differs from original')
                    raise SystemExit(2)
            else:
                print(f'{split}: MATCH')
        print('All split memberships match existing files.')
        return
    # Prevent accidentally overwriting the existing validated split.
    if out.exists() and any(out.glob('X_*.csv')):
        raise FileExistsError(f'{out} already contains processed splits. Use --verify-only first; '
                              'do not overwrite validated project files.')
    out.mkdir(parents=True, exist_ok=True)
    prep = fit_preprocessor(df.iloc[sp['train']], numeric, categorical, cfg['log_cols'])
    for split, ids in sp.items():
        X = apply_preprocessor(df.iloc[ids], prep)
        _save_split(out, split, X, y, sens, ids)
    report.update({'split_sizes': {k: len(v) for k, v in sp.items()},
                   'n_features_after_encoding': apply_preprocessor(df.iloc[sp['train']], prep).shape[1],
                   'seed': SEED, 'preprocessor': prep})
    (out / 'preprocessing_meta.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'preprocessor'}, indent=2))
    print(f'Saved processed data to: {out}')


if __name__ == '__main__':
    main()
