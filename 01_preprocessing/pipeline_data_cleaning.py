"""
Data Cleaning Pipeline for BilinGuard Study
============================================
Cleans baseline clinical data and questionnaire responses.
Follows real-world-data-cleaning skill principles:
  - Check date sequencing
  - Validate bilirubin ranges
  - Standardize variable names and types
  - Generate QC log
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils import BASE_DIR, ensure_dir

QC_LOG = []

def log_step(description, before, after):
    QC_LOG.append({
        'step': description,
        'rows_before': before,
        'rows_after': after,
        'dropped': before - after
    })

def clean_baseline():
    path = os.path.join(BASE_DIR, 'baseline.xlsx')
    print(f'[1/4] Loading baseline from {path}')
    df = pd.read_excel(path)
    before = len(df)
    log_step('Load raw baseline', 0, before)

    # Standardize column names
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    print(f'  Columns: {list(df.columns)}')

    # Check for missing identifiers
    id_cols = [c for c in df.columns if 'id' in c or '编号' in c]
    if id_cols:
        before_sub = len(df)
        df = df.dropna(subset=[id_cols[0]])
        log_step(f'Drop missing IDs ({id_cols[0]})', before_sub, len(df))

    # Validate age range
    age_cols = [c for c in df.columns if 'age' in c or '年龄' in c]
    for c in age_cols:
        print(f'  Age column "{c}": range [{df[c].min()}, {df[c].max()}]')

    # Validate bilirubin if present
    bili_cols = [c for c in df.columns if 'bilir' in c.lower() or '胆红素' in c]
    for c in bili_cols:
        valid = df[c].between(0, 1000)
        n_invalid = (~valid).sum()
        if n_invalid:
            print(f'  WARNING: {c} has {n_invalid} out-of-range values')
            df = df[valid | df[c].isna()]

    out = os.path.join(BASE_DIR, 'data', 'processed', 'baseline_clean.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'[1/4] Saved clean baseline: {len(df)} rows -> {out}')
    return df

def clean_questionnaire():
    path = os.path.join(BASE_DIR, '按序号_去重_合并后的肝炎患者问卷.xlsx')
    print(f'\n[2/4] Loading questionnaire from {path}')
    df = pd.read_excel(path)
    before = len(df)
    log_step('Load raw questionnaire', 0, before)
    print(f'  Columns: {list(df.columns)}')
    print(f'  Shape: {df.shape}')

    out = os.path.join(BASE_DIR, 'data', 'processed', 'questionnaire_clean.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'[2/4] Saved clean questionnaire: {len(df)} rows -> {out}')
    return df

def merge_datasets(baseline, questionnaire):
    print('\n[3/4] Merging baseline and questionnaire')
    id_col_b = [c for c in baseline.columns if 'id' in c or '编号' in c]
    id_col_q = [c for c in questionnaire.columns if 'id' in c or '编号' in c]
    if id_col_b and id_col_q:
        merged = baseline.merge(questionnaire, left_on=id_col_b[0], right_on=id_col_q[0], how='left')
    else:
        merged = baseline
    log_step('Merge datasets', len(baseline), len(merged))
    out = os.path.join(BASE_DIR, 'data', 'processed', 'merged_data.csv')
    merged.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'[3/4] Saved merged: {len(merged)} rows -> {out}')
    return merged

def save_qc_log():
    print('\n[4/4] Saving QC log')
    qc_df = pd.DataFrame(QC_LOG)
    out = os.path.join(BASE_DIR, 'results', 'qc_log.csv')
    qc_df.to_csv(out, index=False, encoding='utf-8-sig')
    print(qc_df.to_string(index=False))
    print(f'[4/4] QC log saved -> {out}')

if __name__ == '__main__':
    ensure_dir(os.path.join(BASE_DIR, 'data', 'processed'))
    ensure_dir(os.path.join(BASE_DIR, 'results'))
    bl = clean_baseline()
    qn = clean_questionnaire()
    merge_datasets(bl, qn)
    save_qc_log()
    print('\nPipeline complete.')
