# -*- coding: utf-8 -*-
"""
Reader study FINAL: detect grade columns by CONTENT not position.
Find all columns with >=90% values in {1,2,3,4} among first 100 rows.
First such column = true grade, second = predicted grade.
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from collections import Counter

BASE = r'D:\research\人脸识别营养\传染科'
DOC_DIR = os.path.join(BASE, 'doctor')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'face_v4_manuscript_delivery', 'tables')

files = sorted([f for f in os.listdir(DOC_DIR) if f.endswith('.xlsx')])
rater_names = ['公卫1', '公卫2', '肝外专培', '口腔医生2', 'hao', '口腔医生', '传染科1', '眼科医生']

# First pass: find the true grade pattern from the first valid file
true_reference = None

all_results = {}
rater_idx = 0

for fname in files:
    df = pd.read_excel(os.path.join(DOC_DIR, fname))
    if len(df) < 50: continue  # skip summary

    # Scan ALL columns for grade-like data (values 1-4)
    grade_cols = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors='coerce')
        # Check first 100 rows only
        first_100 = vals.iloc[:100]
        valid = first_100[first_100.notna()]
        if len(valid) < 80: continue
        in_range = ((valid >= 1) & (valid <= 4)).sum()
        if in_range / len(valid) > 0.9 and all(v in [1, 2, 3, 4] for v in valid.unique()):
            grade_cols.append(c)

    if len(grade_cols) < 2:
        # Fallback: also accept columns where most values are 1-4
        for c in df.columns:
            if c in grade_cols: continue
            vals = pd.to_numeric(df[c], errors='coerce').iloc[:100]
            valid = vals[vals.notna()]
            if len(valid) < 80: continue
            in_range = ((valid >= 1) & (valid <= 4)).sum()
            if in_range / len(valid) > 0.7:
                grade_cols.append(c)

    if len(grade_cols) < 2:
        print(f'  SKIP {fname[:25]}: only {len(grade_cols)} grade cols found')
        continue

    # Among grade cols, identify true vs predicted
    # True grade: the column whose values match the reference (from first file)
    # or the one named with 'true/真实'
    true_col = None
    pred_col = None

    # Try to find by checking which column matches across files
    for c in grade_cols:
        vals = pd.to_numeric(df[c], errors='coerce').iloc[:100]
        valid_vals = vals[vals.notna() & (vals >= 1) & (vals <= 4)].astype(int).values
        cs = str(c)
        if true_reference is not None and len(valid_vals) == len(true_reference):
            match = np.mean(valid_vals == true_reference[:len(valid_vals)])
            if match > 0.95:
                true_col = c
            elif pred_col is None:
                pred_col = c
        if true_col is None and ('真' in cs or 'true' in cs.lower()):
            true_col = c
        if pred_col is None and ('医' in cs or '判' in cs or 'pred' in cs.lower() or '分级' in cs):
            if c != true_col:
                pred_col = c

    # Fallback: first grade col = true, second = pred
    if true_col is None:
        true_col = grade_cols[0]
    if pred_col is None:
        for c in grade_cols:
            if c != true_col:
                pred_col = c
                break

    tg = pd.to_numeric(df[true_col], errors='coerce')
    pg = pd.to_numeric(df[pred_col], errors='coerce')
    valid = tg.notna() & pg.notna() & (tg >= 1) & (tg <= 4) & (pg >= 1) & (pg <= 4)
    true_g = tg[valid].astype(int).values
    pred_g = pg[valid].astype(int).values
    n = len(true_g)

    if n < 50:
        print(f'  SKIP {fname[:25]}: only {n} valid rows after filtering')
        continue

    # Set reference from first valid file
    if true_reference is None:
        true_reference = true_g.copy()

    # Binary: 1=normal, 2-4=jaundice
    true_bin = (true_g >= 2).astype(int)
    pred_bin = (pred_g >= 2).astype(int)

    bin_auc = roc_auc_score(true_bin, pred_bin) if len(set(true_bin)) > 1 else 0.5
    bin_acc = accuracy_score(true_bin, pred_bin)
    bin_f1 = f1_score(true_bin, pred_bin, average='macro')

    # Ternary: among jaundiced (grades 2,3,4 → 0,1,2)
    jmask = true_bin == 1
    tern_auc = None; tern_acc = None; tern_f1 = None; tern_n = 0
    if jmask.sum() > 5 and len(set(true_g[jmask])) >= 2:
        true_tern = (true_g[jmask] - 2).astype(int)
        pred_tern = np.clip(pred_g[jmask] - 2, 0, 2).astype(int)
        tern_n = int(jmask.sum())
        classes = sorted(set(true_tern))
        nc = max(len(classes), 2)
        yb = label_binarize(pred_tern, classes=list(range(3)))
        if yb.shape[1] < 3:
            yb = np.hstack([yb] + [np.zeros((tern_n, 1))] * (3 - yb.shape[1]))
        try:
            tern_auc = float(roc_auc_score(true_tern, yb, multi_class='ovr', labels=[0, 1, 2]))
        except:
            tern_auc = None
        tern_acc = float(accuracy_score(true_tern, pred_tern))
        tern_f1 = float(f1_score(true_tern, pred_tern, average='macro', zero_division=0))

    rname = rater_names[rater_idx] if rater_idx < len(rater_names) else f'Reader {rater_idx+1}'
    rater_idx += 1

    all_results[rname] = {
        'n': int(n), 'n_normal': int((true_bin == 0).sum()), 'n_jaundice': int((true_bin == 1).sum()),
        'true_dist': {str(k): int(v) for k, v in Counter(true_g).items()},
        'bin_auc': round(float(bin_auc), 4), 'bin_acc': round(float(bin_acc), 4), 'bin_f1': round(float(bin_f1), 4),
        'tern_auc': round(tern_auc, 4) if tern_auc else None,
        'tern_acc': round(tern_acc, 4) if tern_acc else None,
        'tern_f1': round(tern_f1, 4) if tern_f1 else None,
        'tern_n': int(tern_n),
    }
    ta = f'{tern_auc:.3f}' if tern_auc else 'N/A'
    print(f'  {rname}: n={n} normal={int((true_bin==0).sum())} jaundice={int((true_bin==1).sum())} '
          f'bin_auc={bin_auc:.3f} tern_auc={ta} tern_n={tern_n}')

# Add BilinGuard
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))
model = [r for r in RS['per_rater'] if r['rater'] == 'BilinGuard'][0]
all_results['BilinGuard (Swin)'] = {
    'n': 100, 'n_normal': 25, 'n_jaundice': 75,
    'bin_auc': round(model['auc'], 4), 'bin_acc': round(model['accuracy'], 4), 'bin_f1': round(model['f1_macro'], 4),
    'tern_auc': 0.798, 'tern_acc': 0.714, 'tern_f1': 0.553, 'tern_n': 75,
}

# Stats
humans = {k: v for k, v in all_results.items() if 'BilinGuard' not in k}
bin_aucs = [v['bin_auc'] for v in humans.values()]
tern_aucs = [v['tern_auc'] for v in humans.values() if v['tern_auc'] is not None]

rng = np.random.RandomState(42)
bin_boots = [float(np.mean(rng.choice(bin_aucs, len(bin_aucs)))) for _ in range(1000)]

print(f'\n{"="*60}')
print(f'  READER STUDY FINAL ({len(humans)} human raters)')
print(f'{"="*60}')
print(f'  Binary:  human {np.mean(bin_aucs):.3f} ({np.percentile(bin_boots,2.5):.3f}-{np.percentile(bin_boots,97.5):.3f})')
print(f'           model {model["auc"]:.3f}')
if tern_aucs:
    print(f'  Ternary: human {np.mean(tern_aucs):.3f} +/- {np.std(tern_aucs):.3f} ({len(tern_aucs)} raters)')
    print(f'           model 0.798')
print(f'  Fleiss kappa: {RS["inter_rater"]["fleiss_kappa"]:.3f}')

# Table 4
t4_rows = []
for name, v in sorted(all_results.items(), key=lambda x: (-x[1]['bin_auc'] if 'BilinGuard' not in x[0] else -1)):
    t4_rows.append({
        'Rater': name,
        'n (N/J)': f'{v["n"]} ({v["n_normal"]}/{v["n_jaundice"]})',
        'Binary AUC': f'{v["bin_auc"]:.3f}',
        'Binary Acc': f'{v["bin_acc"]:.3f}',
        'Binary F1': f'{v["bin_f1"]:.3f}',
        'Ternary AUC': f'{v["tern_auc"]:.3f}' if v['tern_auc'] else '-',
        'Ternary F1': f'{v["tern_f1"]:.3f}' if v['tern_f1'] else '-',
    })
# Group mean
t4_rows.append({
    'Rater': 'Human mean', 'n (N/J)': '100 (25/75)',
    'Binary AUC': f'{np.mean(bin_aucs):.3f} ({np.percentile(bin_boots,2.5):.3f}-{np.percentile(bin_boots,97.5):.3f})',
    'Binary Acc': f'{np.mean([v["bin_acc"] for v in humans.values()]):.3f}',
    'Binary F1': f'{np.mean([v["bin_f1"] for v in humans.values()]):.3f}',
    'Ternary AUC': f'{np.mean(tern_aucs):.3f}' if tern_aucs else '-',
    'Ternary F1': f'{np.mean([v["tern_f1"] for v in humans.values() if v["tern_f1"]]):.3f}' if tern_aucs else '-',
})
t4 = pd.DataFrame(t4_rows)
t4.to_csv(os.path.join(TBL, 'Table4_reader_study.csv'), index=False)
print(f'\nTable 4: {len(t4)} rows')

result = {
    'all_raters': all_results,
    'human_binary_mean': float(np.mean(bin_aucs)),
    'human_binary_ci': [float(np.percentile(bin_boots, 2.5)), float(np.percentile(bin_boots, 97.5))],
    'human_ternary_mean': float(np.mean(tern_aucs)) if tern_aucs else None,
    'model_binary_auc': float(model['auc']),
    'fleiss_kappa': float(RS['inter_rater']['fleiss_kappa']),
}
with open(os.path.join(RES, 'reader_study_final.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, default=str)

import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(os.path.join(TBL, 'Table4_reader_study.csv'), os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(RES, 'reader_study_final.json'), os.path.join(dst, 'audit/'))
print('Synced')
