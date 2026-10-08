# -*- coding: utf-8 -*-
"""P4 final: Use binary columns directly from xlsx (真实是否黄疸 / 医生判断是否黄疸)."""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

BASE = r'D:\research\人脸识别营养\传染科'
DOC_DIR = os.path.join(BASE, 'doctor')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'face_v4_manuscript_delivery', 'tables')

files = sorted([f for f in os.listdir(DOC_DIR) if f.endswith('.xlsx')])
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))

RATER_NAMES = ['Reader 1', 'Reader 2', 'Reader 3', 'Reader 4', 'Reader 5', 'Reader 6', 'Reader 7', 'Reader 8']
all_raters = {}
rater_idx = 0

for fname in files:
    path = os.path.join(DOC_DIR, fname)
    df = pd.read_excel(path)
    if len(df) < 50: continue  # skip summary

    cols = list(df.columns)
    # Find binary columns by checking content (0/1 values in 100+ rows)
    true_bin_col = None
    pred_bin_col = None
    true_grade_col = None
    pred_grade_col = None

    for c in cols:
        vals = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(vals) < 50: continue
        unique_vals = sorted(vals.unique())
        cs = str(c)
        # Binary columns: only 0 and 1
        if set(unique_vals) <= {0, 1} and len(vals) >= 80:
            if '真实' in cs or 'true' in cs.lower():
                if true_bin_col is None: true_bin_col = c
            elif '医生' in cs or '判断' in cs or 'pred' in cs.lower():
                if pred_bin_col is None: pred_bin_col = c
            elif true_bin_col is None:
                true_bin_col = c
            elif pred_bin_col is None:
                pred_bin_col = c
        # Grade columns: >90% of values in {1,2,3,4}
        in_range = ((vals >= 1) & (vals <= 4)).sum()
        if in_range / len(vals) > 0.9 and len(vals) >= 80:
            cs = str(c)
            if '真实' in cs or cs == cols[1]:
                if true_grade_col is None: true_grade_col = c
            elif '医生' in cs or '判断' in cs or cs == cols[3]:
                if pred_grade_col is None: pred_grade_col = c

    if true_bin_col is None or pred_bin_col is None:
        print(f'  SKIP {fname[:20]}: no binary cols found (cols={[str(c)[:8] for c in cols[:8]]})')
        continue

    tb = pd.to_numeric(df[true_bin_col], errors='coerce')
    pb = pd.to_numeric(df[pred_bin_col], errors='coerce')
    valid = tb.notna() & pb.notna()
    true_bin = tb[valid].astype(int).values
    pred_bin = pb[valid].astype(int).values

    bin_auc = roc_auc_score(true_bin, pred_bin) if len(set(true_bin)) > 1 else 0.5
    bin_acc = accuracy_score(true_bin, pred_bin)
    bin_f1 = f1_score(true_bin, pred_bin, average='macro')

    # Ternary: if grade columns found, compute among jaundiced
    tern_auc = None; tern_acc = None; tern_f1 = None; tern_n = 0
    if true_grade_col and pred_grade_col:
        tg = pd.to_numeric(df[true_grade_col], errors='coerce')
        pg = pd.to_numeric(df[pred_grade_col], errors='coerce')
        gvalid = tg.notna() & pg.notna() & valid
        tg = tg[gvalid].astype(int).values
        pg = pg[gvalid].astype(int).values
        tb_g = tb[gvalid].astype(int).values
        jmask = tb_g == 1
        if jmask.sum() > 5 and len(set(tg[jmask])) >= 2:
            t_t = tg[jmask]
            p_t = pg[jmask]
            classes = sorted(set(t_t))
            remap = {v: i for i, v in enumerate(classes)}
            t_r = np.array([remap[v] for v in t_t])
            p_r = np.array([remap.get(v, 0) for v in p_t])
            nc = len(classes)
            if nc >= 2:
                from sklearn.preprocessing import label_binarize
                yb = label_binarize(p_r, classes=list(range(max(nc, 2))))
                if yb.shape[1] < max(nc, 2):
                    yb = np.hstack([yb, np.zeros((len(p_r), max(nc, 2) - yb.shape[1]))])
                try:
                    tern_auc = float(roc_auc_score(t_r, yb, multi_class='ovr',
                                                   labels=list(range(max(nc, 2)))) if nc > 2 else
                               roc_auc_score(t_r, yb[:, min(1, yb.shape[1]-1)]))
                except:
                    tern_auc = None
                tern_acc = float(accuracy_score(t_r, p_r))
                tern_f1 = float(f1_score(t_r, p_r, average='macro', zero_division=0))
                tern_n = int(jmask.sum())

    name = RATER_NAMES[rater_idx] if rater_idx < len(RATER_NAMES) else f'Reader {rater_idx+1}'
    rater_idx += 1
    all_raters[name] = {
        'n': len(true_bin), 'n_normal': int((true_bin == 0).sum()), 'n_jaundice': int((true_bin == 1).sum()),
        'bin_auc': float(bin_auc), 'bin_acc': float(bin_acc), 'bin_f1': float(bin_f1),
        'tern_auc': tern_auc, 'tern_acc': tern_acc, 'tern_f1': tern_f1,
        'tern_n': tern_n,
    }
    ta = f'{tern_auc:.3f}' if tern_auc else '—'
    print(f'  {name}: n={len(true_bin)} bin_auc={bin_auc:.3f} tern_auc={ta} tern_n={tern_n}')

# Add BilinGuard
model = [r for r in RS['per_rater'] if r['rater'] == 'BilinGuard'][0]
all_raters['BilinGuard (Swin)'] = {
    'n': 100, 'n_normal': 25, 'n_jaundice': 75,
    'bin_auc': model['auc'], 'bin_acc': model['accuracy'], 'bin_f1': model['f1_macro'],
    'tern_auc': 0.798, 'tern_acc': 0.714, 'tern_f1': 0.553, 'tern_n': 75,
}

# Stats
humans = {k: v for k, v in all_raters.items() if 'BilinGuard' not in k}
bin_aucs = [v['bin_auc'] for v in humans.values()]
tern_aucs = [v['tern_auc'] for v in humans.values() if v['tern_auc'] is not None]
print(f'\n=== READER STUDY REBUILT ===')
print(f'  Human binary AUC: {np.mean(bin_aucs):.3f} +/- {np.std(bin_aucs):.3f}')
print(f'  Model binary AUC: {model["auc"]:.3f}')
if tern_aucs:
    print(f'  Human ternary AUC: {np.mean(tern_aucs):.3f} +/- {np.std(tern_aucs):.3f} (n={len(tern_aucs)} raters)')
    print(f'  Model ternary AUC: 0.798')

# Table 4
t4_rows = []
for name, v in sorted(all_raters.items(), key=lambda x: -x[1]['bin_auc']):
    t4_rows.append({
        'Rater': name,
        'Binary AUC': f'{v["bin_auc"]:.3f}',
        'Binary Accuracy': f'{v["bin_acc"]:.3f}',
        'Binary F1': f'{v["bin_f1"]:.3f}',
        'Ternary AUC': f'{v["tern_auc"]:.3f}' if v['tern_auc'] else '—',
        'Ternary F1': f'{v["tern_f1"]:.3f}' if v['tern_f1'] else '—',
    })
t4 = pd.DataFrame(t4_rows)
t4.to_csv(os.path.join(TBL, 'Table4_reader_study.csv'), index=False)
print(f'\nTable 4: {len(t4)} rows')

# Save
result = {'all_raters': all_raters,
          'human_binary_mean': float(np.mean(bin_aucs)),
          'human_binary_std': float(np.std(bin_aucs)),
          'human_ternary_mean': float(np.mean(tern_aucs)) if tern_aucs else None,
          'fleiss_kappa': RS['inter_rater']['fleiss_kappa']}
with open(os.path.join(RES, 'reader_study_rebuilt.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, default=str)

import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(os.path.join(TBL, 'Table4_reader_study.csv'), os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(RES, 'reader_study_rebuilt.json'), os.path.join(dst, 'audit/'))
print('Synced')
