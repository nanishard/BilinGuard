# -*- coding: utf-8 -*-
"""
P4: Reader study rebuild — read ALL 8 doctor xlsx files by index (bypass filename encoding).
Extract per-case 4-class labels, compute binary + ternary per-rater metrics.
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from collections import Counter

BASE = r'D:\research\人脸识别营养\传染科'
DOC_DIR = os.path.join(BASE, 'doctor')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'face_v4_manuscript_delivery', 'tables')

# List files by os.listdir (handles encoding correctly)
files = sorted([f for f in os.listdir(DOC_DIR) if f.endswith('.xlsx')])
print(f'Files: {len(files)}')

# The summary file has different structure (ROC curves, not per-case)
# Per-case files have: 图片编号, 真实分级, 医生分级, binary columns, ternary columns
# Read all and identify by structure

all_raters = {}
for idx, fname in enumerate(files):
    path = os.path.join(DOC_DIR, fname)
    df = pd.read_excel(path)
    cols = list(df.columns)
    
    # Skip summary file (only ~21 rows)
    if len(df) < 50:
        print(f'  [{idx}] SKIP summary: {fname[:20]} (rows={len(df)})')
        continue
    
    # Find grade columns by position/content
    # Typical: col[0]=图片编号, col[2 or 3]=真实分级, col[3 or 4]=医生分级
    true_col = None
    pred_col = None
    for c in cols:
        cs = str(c)
        if '真实' in cs and ('分级' in cs or '级' in cs or '疸' in cs):
            if true_col is None: true_col = c
        if ('医生' in cs or '判断' in cs) and ('分级' in cs or '级' in cs or '疸' in cs):
            if pred_col is None: pred_col = c
    
    # Fallback: try column indices
    if true_col is None:
        for i, c in enumerate(cols):
            vals = pd.to_numeric(df[c], errors='coerce').dropna()
            if len(vals) > 50 and vals.min() >= 1 and vals.max() <= 4:
                if true_col is None: true_col = c
                elif pred_col is None: pred_col = c
                break
    if pred_col is None:
        for i, c in enumerate(cols):
            if c == true_col: continue
            vals = pd.to_numeric(df[c], errors='coerce').dropna()
            if len(vals) > 50 and vals.min() >= 1 and vals.max() <= 4:
                pred_col = c
                break
    
    if true_col is None or pred_col is None:
        print(f'  [{idx}] SKIP no grade cols: {fname[:20]}')
        continue
    
    true_g = pd.to_numeric(df[true_col], errors='coerce')
    pred_g = pd.to_numeric(df[pred_col], errors='coerce')
    valid = true_g.notna() & pred_g.notna()
    true_g = true_g[valid].astype(int).values
    pred_g = pred_g[valid].astype(int).values
    
    if len(true_g) < 50:
        print(f'  [{idx}] SKIP too few cases: {fname[:20]} ({len(true_g)})')
        continue
    
    # Grade scheme: 1=normal, 2=mild, 3=moderate, 4=severe
    # Binary: grade >= 2 = jaundice
    true_bin = (true_g >= 2).astype(int)
    pred_bin = (pred_g >= 2).astype(int)
    
    # Ternary: among jaundiced only (grades 2,3,4 → remap to 0,1,2)
    jaundiced_mask = true_bin == 1
    tern_auc = float('nan'); tern_acc = float('nan'); tern_f1 = float('nan')
    if jaundiced_mask.sum() > 5:
        true_tern = (true_g[jaundiced_mask] - 2).astype(int)
        pred_tern = np.clip(pred_g[jaundiced_mask] - 2, 0, 2).astype(int)
        nc_tern = len(set(true_tern))
        if nc_tern >= 2:
            from sklearn.preprocessing import label_binarize
            yb_pred = label_binarize(pred_tern, classes=list(range(max(nc_tern, 3))))
            try:
                tern_auc = roc_auc_score(true_tern, yb_pred, multi_class='ovr',
                                         labels=list(range(max(nc_tern, 3)))) if nc_tern > 2 else \
                           roc_auc_score(true_tern, yb_pred[:, 1] if yb_pred.shape[1] > 1 else yb_pred[:, 0])
            except:
                tern_auc = float('nan')
            tern_acc = accuracy_score(true_tern, pred_tern)
            tern_f1 = f1_score(true_tern, pred_tern, average='macro', zero_division=0)
    else:
        tern_auc = float('nan'); tern_acc = float('nan'); tern_f1 = float('nan')
    
    bin_auc = roc_auc_score(true_bin, pred_bin) if len(set(true_bin)) > 1 else float('nan')
    bin_acc = accuracy_score(true_bin, pred_bin)
    bin_f1 = f1_score(true_bin, pred_bin, average='macro')
    
    rater_key = f'Reader_{idx}'
    all_raters[rater_key] = {
        'filename': fname[:30],
        'n_cases': len(true_g),
        'n_normal': int((true_bin == 0).sum()),
        'n_jaundice': int((true_bin == 1).sum()),
        'bin_auc': float(bin_auc), 'bin_acc': float(bin_acc), 'bin_f1': float(bin_f1),
        'tern_auc': float(tern_auc) if not np.isnan(tern_auc) else None,
        'tern_acc': float(tern_acc) if not np.isnan(tern_acc) else None,
        'tern_f1': float(tern_f1) if not np.isnan(tern_f1) else None,
        'tern_n': int(jaundiced_mask.sum()) if jaundiced_mask.sum() > 5 else 0,
        'tern_classes': len(set(true_g[jaundiced_mask])) if jaundiced_mask.sum() > 5 else 0,
    }
    ta_str = f'{tern_auc:.3f}' if not np.isnan(tern_auc) else 'nan'
    print(f'  [{idx}] {rater_key}: n={len(true_g)} bin_auc={bin_auc:.3f} '
          f'tern_auc={ta_str} '
          f'tern_n={int(jaundiced_mask.sum())} tern_classes={len(set(true_g[jaundiced_mask])) if jaundiced_mask.sum() > 0 else 0}')

# Add BilinGuard
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))
model = [r for r in RS['per_rater'] if r['rater'] == 'BilinGuard'][0]
all_raters['BilinGuard'] = {
    'filename': 'AI model (Swin)',
    'n_cases': 100, 'n_normal': 25, 'n_jaundice': 75,
    'bin_auc': model['auc'], 'bin_acc': model['accuracy'], 'bin_f1': model['f1_macro'],
    'tern_auc': 0.798, 'tern_acc': 0.714, 'tern_f1': 0.553,
    'tern_n': 75, 'tern_classes': 3,
}

# Summary
humans = {k: v for k, v in all_raters.items() if k != 'BilinGuard'}
bin_aucs = [v['bin_auc'] for v in humans.values()]
tern_aucs = [v['tern_auc'] for v in humans.values() if v['tern_auc'] is not None]
print(f'\n=== READER STUDY SUMMARY ===')
print(f'  Human raters: {len(humans)}')
print(f'  Binary AUC: mean={np.mean(bin_aucs):.3f} ± {np.std(bin_aucs):.3f}')
print(f'  Model binary AUC: {model["auc"]:.3f}')
if tern_aucs:
    print(f'  Ternary AUC: mean={np.mean(tern_aucs):.3f} ± {np.std(tern_aucs):.3f} (n={len(tern_aucs)} raters)')
    print(f'  Model ternary AUC: 0.798')

# Build Table 4
t4_rows = []
for key, v in sorted(all_raters.items(), key=lambda x: -x[1]['bin_auc']):
    t4_rows.append({
        'Rater': key,
        'n_cases': v['n_cases'],
        'Binary AUC': f"{v['bin_auc']:.3f}",
        'Binary Acc': f"{v['bin_acc']:.3f}",
        'Binary F1': f"{v['bin_f1']:.3f}",
        'Ternary AUC': f"{v['tern_auc']:.3f}" if v['tern_auc'] is not None else '—',
        'Ternary n': v['tern_n'] if v['tern_n'] > 0 else '—',
    })
t4 = pd.DataFrame(t4_rows)
t4.to_csv(os.path.join(TBL, 'Table4_reader_study.csv'), index=False)

result = {'all_raters': all_raters,
          'human_binary_mean': float(np.mean(bin_aucs)),
          'human_binary_std': float(np.std(bin_aucs)),
          'human_ternary_mean': float(np.mean(tern_aucs)) if tern_aucs else None,
          'fleiss_kappa': RS['inter_rater']['fleiss_kappa']}
with open(os.path.join(RES, 'reader_study_rebuilt.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, default=str)

print(f'\nTable 4 saved: {len(t4)} rows')
print(f'Saved: reader_study_rebuilt.json')

# Sync
import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(os.path.join(TBL, 'Table4_reader_study.csv'), os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(RES, 'reader_study_rebuilt.json'), os.path.join(dst, 'audit/'))
print('Synced')
