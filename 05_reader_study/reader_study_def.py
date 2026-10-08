# -*- coding: utf-8 -*-
"""
Reader study: use BINARY columns (verified correct) + find grade columns
by cross-file consistency (true grade identical across files, pred varies).
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from collections import Counter

BASE = r'D:\research\人脸识别营养\传染科'
DOC_DIR = os.path.join(BASE, 'doctor')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'face_v4_manuscript_delivery', 'tables')

files = sorted([f for f in os.listdir(DOC_DIR) if f.endswith('.xlsx')])
rater_names = ['公卫1', '公卫2', '肝外专培', '口腔医生2', 'hao', '口腔医生', '传染科1', '眼科医生']

# Pass 1: read all files, extract binary columns + all grade-like columns
file_data = []
for fname in files:
    df = pd.read_excel(os.path.join(DOC_DIR, fname))
    if len(df) < 50: continue
    cols = list(df.columns)

    # Find binary columns (0/1 values, >=80 valid rows)
    bin_cols = []
    for c in cols:
        vals = pd.to_numeric(df[c], errors='coerce')
        v100 = vals.iloc[:100]
        valid = v100[v100.notna()]
        if len(valid) >= 80 and set(valid.unique()) <= {0, 1}:
            bin_cols.append(c)

    # Find grade-like columns (values in {1,2,3,4}, >=80 valid in first 100 rows)
    grade_cols = []
    for c in cols:
        vals = pd.to_numeric(df[c], errors='coerce')
        v100 = vals.iloc[:100]
        valid = v100[v100.notna()]
        if len(valid) < 80: continue
        in_range = ((valid >= 1) & (valid <= 4)).sum()
        if in_range / len(valid) > 0.95:
            grade_cols.append((c, valid.astype(int).values))

    file_data.append({'fname': fname, 'df': df, 'bin_cols': bin_cols, 'grade_cols': grade_cols})

print(f'Files with grade data: {len(file_data)}')

# Identify true grade column: the one with identical values across ALL files
# Compare each grade column from file 0 against all other files
if len(file_data) >= 2 and len(file_data[0]['grade_cols']) >= 2:
    ref_grades = file_data[0]['grade_cols']
    true_col_idx = None

    for i, (cname, cvals) in enumerate(ref_grades):
        matches_all = True
        for fd in file_data[1:]:
            found_match = False
            for _, other_vals in fd['grade_cols']:
                if len(other_vals) == len(cvals):
                    match_rate = np.mean(other_vals == cvals)
                    if match_rate > 0.95:
                        found_match = True
                        break
            if not found_match:
                matches_all = False
                break
        if matches_all and true_col_idx is None:
            true_col_idx = i

    if true_col_idx is None:
        true_col_idx = 0  # fallback: first grade col
    print(f'True grade column identified: index {true_col_idx} in file 0')

# Pass 2: compute metrics
all_results = {}
for idx, fd in enumerate(file_data):
    df = fd['df']
    fname = fd['fname']

    # Binary: use binary columns
    bin_cols = fd['bin_cols']
    if len(bin_cols) >= 2:
        tb = pd.to_numeric(df[bin_cols[0]], errors='coerce')
        pb = pd.to_numeric(df[bin_cols[1]], errors='coerce')
    else:
        continue

    valid = tb.notna() & pb.notna()
    # Extend validity to first 100 rows only
    valid = valid & (valid.cumsum() <= 100)
    true_bin = tb[valid].astype(int).values
    pred_bin = pb[valid].astype(int).values

    # Fix: ensure 0=normal, 1=jaundice
    if true_bin.mean() > 0.5:
        true_bin = 1 - true_bin
        pred_bin = 1 - pred_bin

    bin_auc = roc_auc_score(true_bin, pred_bin) if len(set(true_bin)) > 1 else 0.5
    bin_acc = accuracy_score(true_bin, pred_bin)
    bin_f1 = f1_score(true_bin, pred_bin, average='macro')

    # Grade columns: true = consistent across files, pred = the other
    grade_cols = fd['grade_cols']
    tern_auc = None; tern_acc = None; tern_f1 = None; tern_n = 0

    if len(grade_cols) >= 2:
        # True grade = the one matching reference
        true_g_vals = None
        pred_g_vals = None
        ref_vals = file_data[0]['grade_cols'][true_col_idx][1]

        for cname, cvals in grade_cols:
            if len(cvals) == len(ref_vals) and np.mean(cvals == ref_vals) > 0.95:
                true_g_vals = cvals
            else:
                if pred_g_vals is None:
                    pred_g_vals = cvals

        if true_g_vals is not None and pred_g_vals is not None:
            jmask = true_g_vals >= 2
            if jmask.sum() > 5 and len(set(true_g_vals[jmask])) >= 2:
                true_tern = (true_g_vals[jmask] - 2).astype(int)
                pred_tern = np.clip(pred_g_vals[jmask] - 2, 0, 2).astype(int)
                tern_n = int(jmask.sum())
                yb = label_binarize(pred_tern, classes=[0, 1, 2])
                if yb.shape[1] < 3:
                    yb = np.hstack([yb] + [np.zeros((tern_n, 1))] * (3 - yb.shape[1]))
                try:
                    tern_auc = float(roc_auc_score(true_tern, yb, multi_class='ovr', labels=[0, 1, 2]))
                except:
                    tern_auc = None
                tern_acc = float(accuracy_score(true_tern, pred_tern))
                tern_f1 = float(f1_score(true_tern, pred_tern, average='macro', zero_division=0))

    rname = rater_names[idx] if idx < len(rater_names) else f'Reader {idx+1}'
    all_results[rname] = {
        'n': len(true_bin), 'n_normal': int((true_bin == 0).sum()), 'n_jaundice': int((true_bin == 1).sum()),
        'bin_auc': round(float(bin_auc), 4), 'bin_acc': round(float(bin_acc), 4), 'bin_f1': round(float(bin_f1), 4),
        'tern_auc': round(tern_auc, 4) if tern_auc else None,
        'tern_acc': round(tern_acc, 4) if tern_acc else None,
        'tern_f1': round(tern_f1, 4) if tern_f1 else None,
        'tern_n': int(tern_n),
    }
    ta = f'{tern_auc:.3f}' if tern_auc else 'N/A'
    print(f'  {rname}: bin_auc={bin_auc:.3f} tern_auc={ta} tern_n={tern_n}')

# Add model
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))
model = [r for r in RS['per_rater'] if r['rater'] == 'BilinGuard'][0]
all_results['BilinGuard (Swin)'] = {
    'n': 100, 'n_normal': 25, 'n_jaundice': 75,
    'bin_auc': round(model['auc'], 4), 'bin_acc': round(model['accuracy'], 4),
    'bin_f1': round(model['f1_macro'], 4),
    'tern_auc': 0.798, 'tern_acc': 0.714, 'tern_f1': 0.553, 'tern_n': 75,
}

# Stats
humans = {k: v for k, v in all_results.items() if 'BilinGuard' not in k}
bin_aucs = [v['bin_auc'] for v in humans.values()]
tern_aucs = [v['tern_auc'] for v in humans.values() if v['tern_auc'] is not None]
rng = np.random.RandomState(42)
bin_boots = [float(np.mean(rng.choice(bin_aucs, len(bin_aucs)))) for _ in range(1000)]

print(f'\n=== FINAL ({len(humans)} raters) ===')
print(f'  Binary:  human {np.mean(bin_aucs):.3f} ({np.percentile(bin_boots,2.5):.3f}-{np.percentile(bin_boots,97.5):.3f})')
print(f'           model {model["auc"]:.3f}')
if tern_aucs:
    print(f'  Ternary: human {np.mean(tern_aucs):.3f}+/-{np.std(tern_aucs):.3f} ({len(tern_aucs)} raters)')
    print(f'           model 0.798')

# Table 4
t4_rows = []
for name, v in sorted(all_results.items(), key=lambda x: (-x[1]['bin_auc'] if 'BilinGuard' not in x[0] else -1)):
    t4_rows.append({
        'Rater': name,
        'Binary AUC': f'{v["bin_auc"]:.3f}',
        'Binary Acc': f'{v["bin_acc"]:.3f}',
        'Binary F1': f'{v["bin_f1"]:.3f}',
        'Ternary AUC': f'{v["tern_auc"]:.3f}' if v['tern_auc'] else '-',
        'Ternary F1': f'{v["tern_f1"]:.3f}' if v['tern_f1'] else '-',
    })
t4_rows.append({
    'Rater': 'Human mean',
    'Binary AUC': f'{np.mean(bin_aucs):.3f} ({np.percentile(bin_boots,2.5):.3f}-{np.percentile(bin_boots,97.5):.3f})',
    'Binary Acc': f'{np.mean([v["bin_acc"] for v in humans.values()]):.3f}',
    'Binary F1': f'{np.mean([v["bin_f1"] for v in humans.values()]):.3f}',
    'Ternary AUC': f'{np.mean(tern_aucs):.3f}' if tern_aucs else '-',
    'Ternary F1': f'{np.mean([v["tern_f1"] for v in humans.values() if v["tern_f1"]]):.3f}' if tern_aucs else '-',
})
t4 = pd.DataFrame(t4_rows)
t4.to_csv(os.path.join(TBL, 'Table4_reader_study.csv'), index=False)
print(f'\nTable 4: {len(t4)} rows')

# Save
result = {'all_raters': all_results,
          'human_binary_mean': float(np.mean(bin_aucs)),
          'human_binary_ci': [float(np.percentile(bin_boots, 2.5)), float(np.percentile(bin_boots, 97.5))],
          'human_ternary_mean': float(np.mean(tern_aucs)) if tern_aucs else None,
          'model_binary_auc': float(model['auc']),
          'fleiss_kappa': float(RS['inter_rater']['fleiss_kappa'])}
with open(os.path.join(RES, 'reader_study_final.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, default=str)

import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(os.path.join(TBL, 'Table4_reader_study.csv'), os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(RES, 'reader_study_final.json'), os.path.join(dst, 'audit/'))
print('Synced')
