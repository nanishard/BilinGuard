# -*- coding: utf-8 -*-
"""
Merge returned blinded-review workbooks into one long-form file for analysis.

Each input file (one per reviewer) keeps the distribution format:
sheet '盲评病例' with reviewer-specific case order + filled score columns.
Rows are stacked (long form: pid x reviewer), aligned content verified.

Usage:
  python code/cdss_merge_review_returns.py fileA.xlsx fileB.xlsx [fileC.xlsx ...]

Output: results/face_v4_manuscript_delivery/tables/CDSS_REVIEW_RETURNED_MERGED.xlsx
"""
import sys, os
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
OUT = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables',
                   'CDSS_REVIEW_RETURNED_MERGED.xlsx')

SCORE_COLS = ['合并诊断一致性(1-5)', '建议适当性(1-5)',
              '安全覆盖(1-5)', '营养建议适当性(1-5)']


def load_returned(path):
    xl = pd.ExcelFile(path)
    sheet = '盲评病例' if '盲评病例' in xl.sheet_names else xl.sheet_names[-1]
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if 'pid' not in df.columns or '审阅者' not in df.columns:
        raise ValueError(f'{path}: 缺少 pid/审阅者 列（确认是回收版工作簿）')
    n_scored = int(df[SCORE_COLS[0]].notna().sum()) if SCORE_COLS[0] in df.columns else 0
    print(f'  {os.path.basename(path)}: {len(df)} rows, '
          f'{df["审阅者"].dropna().unique().tolist()}, '
          f'scored {n_scored}/{len(df)}')
    return df


def main(paths):
    frames = [load_returned(p) for p in paths]
    ref_cols = list(frames[0].columns)
    for i, f in enumerate(frames[1:], start=2):
        if list(f.columns) != ref_cols:
            frames[i - 1] = f[ref_cols]  # align column order
    merged = pd.concat(frames, ignore_index=True)
    # basic integrity checks
    dup = merged.duplicated(subset=['pid', '审阅者']).sum()
    if dup:
        print(f'  WARNING: {dup} duplicated (pid, reviewer) rows — keeping first')
        merged = merged.drop_duplicates(subset=['pid', '审阅者'], keep='first')
    n_rev = merged['审阅者'].nunique()
    per_case = merged.groupby('pid')['审阅者'].nunique()
    complete = int((per_case == n_rev).sum())
    print(f'\n  merged: {len(merged)} ratings, {n_rev} reviewers, '
          f'{complete}/{per_case.size} cases rated by all reviewers')
    bad = merged[SCORE_COLS].apply(
        lambda s: s.notna() & (~s.isin([1, 2, 3, 4, 5]))).sum().sum()
    if bad:
        print(f'  WARNING: {int(bad)} scores outside 1-5 — set to NaN')
        for c in SCORE_COLS:
            merged.loc[~merged[c].isin([1, 2, 3, 4, 5]) & merged[c].notna(), c] = pd.NA
    merged.to_excel(OUT, index=False)
    print(f'Saved: {OUT}')
    return OUT


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: cdss_merge_review_returns.py fileA.xlsx fileB.xlsx [...]')
        sys.exit(2)
    main(sys.argv[1:])
