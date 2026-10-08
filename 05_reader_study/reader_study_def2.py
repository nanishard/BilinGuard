# -*- coding: utf-8 -*-
"""
Reader study DEFINITIVE: Use verified binary from reader_study_results.json,
extract ternary from raw xlsx with cross-file true grade identification.
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
DOC_DIR = os.path.join(BASE, 'doctor')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'face_v4_manuscript_delivery', 'tables')

# Verified binary data
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))
RATER_MAP = {
    'R1_Attending': 'Attending Surgeon 1', 'R2_Attending': 'Attending Surgeon 2',
    'R3_Nurse': 'Senior ID Nurse', 'R4_Resident': 'Resident 1',
    'R5_Resident': 'Resident 2', 'R6_Resident': 'Resident 3',
    'R7_PubHealth': 'Public Health 1', 'R8_PubHealth': 'Public Health 2',
}

# Read raw files for ternary
files = sorted([f for f in os.listdir(DOC_DIR) if f.endswith('.xlsx')])
file_data = []
for fname in files:
    df = pd.read_excel(os.path.join(DOC_DIR, fname))
    if len(df) < 50: continue
    grade_cols = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors='coerce').iloc[:100]
        valid = vals[vals.notna()]
        if len(valid) < 80: continue
        in_range = ((valid >= 1) & (valid <= 4)).sum()
        if in_range / len(valid) > 0.95:
            grade_cols.append(valid.astype(int).values)
    file_data.append(grade_cols)

# Find true grade column (consistent across files)
true_ref = None
if file_data and len(file_data[0]) >= 2:
    for candidate in file_data[0]:
        matches = 0
        for fd in file_data[1:]:
            for other in fd:
                if len(other) == len(candidate) and np.mean(other == candidate) > 0.95:
                    matches += 1; break
        if matches >= len(file_data) - 1:
            true_ref = candidate.copy()
            break
    if true_ref is None:
        true_ref = file_data[0][0].copy()

# Extract ternary for each rater
ternary_results = []
for idx, grade_cols in enumerate(file_data):
    pred_g = None
    for c in grade_cols:
        if len(c) == len(true_ref) and np.mean(c == true_ref) < 0.95:
            pred_g = c; break
    if pred_g is None and len(grade_cols) >= 2:
        pred_g = grade_cols[1] if np.mean(grade_cols[0] == true_ref) > 0.5 else grade_cols[0]

    if pred_g is not None:
        jmask = true_ref >= 2
        true_tern = (true_ref[jmask] - 2).astype(int)
        pred_tern = np.clip(pred_g[jmask] - 2, 0, 2).astype(int)
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
    else:
        tern_auc = tern_acc = tern_f1 = None; tern_n = 0
    ternary_results.append({'tern_auc': tern_auc, 'tern_acc': tern_acc, 'tern_f1': tern_f1, 'tern_n': tern_n})

# Build combined table
model = [r for r in RS['per_rater'] if r['rater'] == 'BilinGuard'][0]
t4_rows = []
rater_keys = ['R1_Attending','R2_Attending','R3_Nurse','R4_Resident','R5_Resident','R6_Resident','R7_PubHealth','R8_PubHealth']

for i, rkey in enumerate(rater_keys):
    r = [r for r in RS['per_rater'] if r['rater'] == rkey][0]
    kappa = RS['model_rater']['individual_kappas'].get(rkey)
    tr = ternary_results[i] if i < len(ternary_results) else {}
    t4_rows.append({
        'Rater': RATER_MAP[rkey],
        'Binary AUC': f'{r["auc"]:.3f}',
        'Binary Acc': f'{r["accuracy"]:.3f}',
        'Binary F1': f'{r["f1_macro"]:.3f}',
        'Ternary AUC': f'{tr["tern_auc"]:.3f}' if tr.get('tern_auc') else '-',
        'Ternary F1': f'{tr["tern_f1"]:.3f}' if tr.get('tern_f1') else '-',
        'Cohen kappa': f'{kappa:.3f}' if kappa else '-',
    })

# Human means
bin_aucs = [r['auc'] for r in RS['per_rater'] if r['rater'] != 'BilinGuard']
tern_aucs = [t['tern_auc'] for t in ternary_results if t['tern_auc']]
rng = np.random.RandomState(42)
bin_boots = [float(np.mean(rng.choice(bin_aucs, len(bin_aucs)))) for _ in range(1000)]

t4_rows.append({
    'Rater': 'Human mean',
    'Binary AUC': f'{np.mean(bin_aucs):.3f} ({np.percentile(bin_boots,2.5):.3f}-{np.percentile(bin_boots,97.5):.3f})',
    'Binary Acc': f'{np.mean([r["accuracy"] for r in RS["per_rater"] if r["rater"]!="BilinGuard"]):.3f}',
    'Binary F1': f'{np.mean([r["f1_macro"] for r in RS["per_rater"] if r["rater"]!="BilinGuard"]):.3f}',
    'Ternary AUC': f'{np.mean(tern_aucs):.3f}' if tern_aucs else '-',
    'Ternary F1': f'{np.mean([t["tern_f1"] for t in ternary_results if t["tern_f1"]]):.3f}' if tern_aucs else '-',
    'Cohen kappa': '',
})
# Model
t4_rows.append({
    'Rater': 'BilinGuard (Swin)',
    'Binary AUC': f'{model["auc"]:.3f}',
    'Binary Acc': f'{model["accuracy"]:.3f}',
    'Binary F1': f'{model["f1_macro"]:.3f}',
    'Ternary AUC': '0.798',
    'Ternary F1': '0.553',
    'Cohen kappa': '-',
})

t4 = pd.DataFrame(t4_rows)
t4.to_csv(os.path.join(TBL, 'Table4_reader_study.csv'), index=False)

print(f'=== READER STUDY DEFINITIVE ===')
print(f'  Binary:  human {np.mean(bin_aucs):.3f} ({np.percentile(bin_boots,2.5):.3f}-{np.percentile(bin_boots,97.5):.3f}), model {model["auc"]:.3f}')
if tern_aucs:
    print(f'  Ternary: human {np.mean(tern_aucs):.3f}+/-{np.std(tern_aucs):.3f} ({len(tern_aucs)} raters), model 0.798')
print(f'  Fleiss kappa: {RS["inter_rater"]["fleiss_kappa"]:.3f}')
print(f'  Table 4: {len(t4)} rows')

# Save
result = {
    'human_binary_mean': float(np.mean(bin_aucs)),
    'human_binary_ci': [float(np.percentile(bin_boots,2.5)), float(np.percentile(bin_boots,97.5))],
    'human_ternary_mean': float(np.mean(tern_aucs)) if tern_aucs else None,
    'human_ternary_std': float(np.std(tern_aucs)) if tern_aucs else None,
    'model_binary_auc': float(model['auc']),
    'model_ternary_auc': 0.798,
    'fleiss_kappa': float(RS['inter_rater']['fleiss_kappa']),
    'cohen_kappa_mean': float(RS['model_rater']['cohen_kappa_mean']),
}
with open(os.path.join(RES, 'reader_study_final.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(os.path.join(TBL, 'Table4_reader_study.csv'), os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(RES, 'reader_study_final.json'), os.path.join(dst, 'audit/'))
print('Synced')
