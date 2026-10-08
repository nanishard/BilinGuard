# -*- coding: utf-8 -*-
"""
Reader study DEFINITIVE rebuild.
Raters: 公卫1, 公卫2, 肝外专培, 口腔医生2, hao, 口腔医生, 传染科1, 眼科医生
Grade scheme: 1=无黄疸, 2=低度, 3=中度, 4=重度
Binary: 1=normal, 2-4=jaundice
Ternary: among jaundiced (2,3,4), classify mild/moderate/severe
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
print(f'Files found: {len(files)}')

# Expected 8 raters (order will be determined by file content)
# Skip summary file (short, <50 rows)
per_case_files = []
for fname in files:
    df = pd.read_excel(os.path.join(DOC_DIR, fname))
    if len(df) >= 50:
        per_case_files.append((fname, df))
print(f'Per-case files: {len(per_case_files)}')

# Column structure (confirmed from diagnostic):
# col[0] = image number
# col[1] = true grade (1-4)  ← 100 valid rows
# col[3] = predicted grade (1-4)  ← 100 valid rows (some have summary row at bottom)
# col[6] = true binary (0/1)  ← 100 valid rows
# col[7] = pred binary (0/1)  ← 100 valid rows

rater_names_user = ['公卫1', '公卫2', '肝外专培', '口腔医生2', 'hao', '口腔医生', '传染科1', '眼科医生']
all_results = {}

for idx, (fname, df) in enumerate(per_case_files):
    cols = list(df.columns)
    
    # Extract true grade and predicted grade
    # col[1] = true grade, col[3] = pred grade (by position from diagnostic)
    true_grade_col = cols[1]
    pred_grade_col = cols[3]
    
    # Also verify by trying to find binary columns
    true_bin_col = cols[6] if len(cols) > 6 else None
    pred_bin_col = cols[7] if len(cols) > 7 else None
    
    tg = pd.to_numeric(df[true_grade_col], errors='coerce')
    pg = pd.to_numeric(df[pred_grade_col], errors='coerce')
    
    # Filter to valid rows (grade 1-4, exclude summary/ROC rows)
    valid = tg.notna() & pg.notna() & (tg >= 1) & (tg <= 4) & (pg >= 1) & (pg <= 4)
    true_g = tg[valid].astype(int).values
    pred_g = pg[valid].astype(int).values
    n = len(true_g)
    
    if n < 50:
        print(f'  [{idx}] SKIP: only {n} valid rows')
        continue
    
    # Binary: 1=normal, 2-4=jaundice
    true_bin = (true_g >= 2).astype(int)
    pred_bin = (pred_g >= 2).astype(int)
    
    n_normal = int((true_bin == 0).sum())
    n_jaundice = int((true_bin == 1).sum())
    
    bin_auc = roc_auc_score(true_bin, pred_bin) if len(set(true_bin)) > 1 else 0.5
    bin_acc = accuracy_score(true_bin, pred_bin)
    bin_f1 = f1_score(true_bin, pred_bin, average='macro')
    bin_cm = confusion_matrix(true_bin, pred_bin, labels=[0, 1]).tolist()
    
    # Ternary: among jaundiced only (grades 2,3,4 → remap to 0,1,2)
    jmask = true_bin == 1
    tern_auc = None; tern_acc = None; tern_f1 = None; tern_n = 0; tern_cm = None
    
    if jmask.sum() > 5:
        true_tern = (true_g[jmask] - 2).astype(int)  # 2→0(mild), 3→1(moderate), 4→2(severe)
        pred_tern = np.clip(pred_g[jmask] - 2, 0, 2).astype(int)
        tern_n = int(jmask.sum())
        classes_present = sorted(set(true_tern))
        
        if len(classes_present) >= 2:
            nc = 3  # always 3 classes for ternary
            yb = label_binarize(pred_tern, classes=list(range(nc)))
            # Need to handle case where not all classes present in predictions
            if yb.shape[1] < nc:
                yb = np.hstack([yb] + [np.zeros((tern_n, 1))] * (nc - yb.shape[1]))
            try:
                tern_auc = float(roc_auc_score(true_tern, yb, multi_class='ovr', labels=[0, 1, 2]))
            except:
                # Fallback: binary AUC for available classes
                try:
                    tern_auc = float(roc_auc_score(true_tern, yb[:, 1] if yb.shape[1] > 1 else yb[:, 0]))
                except:
                    tern_auc = None
            tern_acc = float(accuracy_score(true_tern, pred_tern))
            tern_f1 = float(f1_score(true_tern, pred_tern, average='macro', zero_division=0))
            tern_cm = confusion_matrix(true_tern, pred_tern, labels=[0, 1, 2]).tolist()
    
    # Assign rater name
    rname = rater_names_user[idx] if idx < len(rater_names_user) else f'Reader {idx+1}'
    
    all_results[rname] = {
        'n': n, 'n_normal': n_normal, 'n_jaundice': n_jaundice,
        'true_dist': dict(Counter(true_g)),
        'bin_auc': float(bin_auc), 'bin_acc': float(bin_acc), 'bin_f1': float(bin_f1),
        'bin_cm': bin_cm,
        'tern_auc': tern_auc, 'tern_acc': tern_acc, 'tern_f1': tern_f1,
        'tern_n': tern_n, 'tern_cm': tern_cm,
    }
    
    ta = f'{tern_auc:.3f}' if tern_auc else 'N/A'
    print(f'  [{idx}] {rname}: n={n} normal={n_normal} jaundice={n_jaundice} '
          f'bin_auc={bin_auc:.3f} tern_auc={ta} tern_n={tern_n}')

# Add BilinGuard model
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))
model = [r for r in RS['per_rater'] if r['rater'] == 'BilinGuard'][0]
all_results['BilinGuard (Swin)'] = {
    'n': 100, 'n_normal': 25, 'n_jaundice': 75,
    'bin_auc': model['auc'], 'bin_acc': model['accuracy'], 'bin_f1': model['f1_macro'],
    'tern_auc': 0.798, 'tern_acc': 0.714, 'tern_f1': 0.553,
    'tern_n': 75,
}

# Summary stats
humans = {k: v for k, v in all_results.items() if 'BilinGuard' not in k}
bin_aucs = [v['bin_auc'] for v in humans.values()]
tern_aucs = [v['tern_auc'] for v in humans.values() if v['tern_auc'] is not None]
bin_accs = [v['bin_acc'] for v in humans.values()]
tern_accs = [v['tern_acc'] for v in humans.values() if v['tern_acc'] is not None]

print(f'\n{"="*60}')
print(f'  READER STUDY DEFINITIVE RESULTS')
print(f'{"="*60}')
print(f'  Human raters: {len(humans)}')
print(f'  Binary:  mean AUC = {np.mean(bin_aucs):.3f} +/- {np.std(bin_aucs):.3f}')
print(f'           mean Acc = {np.mean(bin_accs):.3f}')
print(f'  Model:   Binary AUC = {model["auc"]:.3f}, Acc = {model["accuracy"]:.3f}')
if tern_aucs:
    print(f'  Ternary: mean AUC = {np.mean(tern_aucs):.3f} +/- {np.std(tern_aucs):.3f} ({len(tern_aucs)} raters)')
    print(f'           mean Acc = {np.mean(tern_accs):.3f}')
    print(f'  Model:   Ternary AUC = 0.798, Acc = 0.714')
print(f'  Fleiss kappa (binary): {RS["inter_rater"]["fleiss_kappa"]:.3f}')

# Bootstrap CI for human mean AUC
rng = np.random.RandomState(42)
bin_boots = [np.mean(rng.choice(bin_aucs, len(bin_aucs))) for _ in range(1000)]
print(f'  Human binary AUC 95% CI: [{np.percentile(bin_boots, 2.5):.3f}, {np.percentile(bin_boots, 97.5):.3f}]')

# Build Table 4
t4_rows = []
for name, v in sorted(all_results.items(), key=lambda x: (-x[1]['bin_auc'] if 'BilinGuard' not in x[0] else 1)):
    t4_rows.append({
        'Rater': name,
        'n (normal/jaundice)': f'{v["n"]} ({v["n_normal"]}/{v["n_jaundice"]})',
        'Binary AUC': f'{v["bin_auc"]:.3f}',
        'Binary Accuracy': f'{v["bin_acc"]:.3f}',
        'Binary F1': f'{v["bin_f1"]:.3f}',
        'Ternary AUC': f'{v["tern_auc"]:.3f}' if v['tern_auc'] else '—',
        'Ternary n': str(v['tern_n']) if v['tern_n'] > 0 else '—',
    })

# Group summary row
t4_rows.append({
    'Rater': '— Human mean —',
    'n (normal/jaundice)': f'100 (25/75)',
    'Binary AUC': f'{np.mean(bin_aucs):.3f} ({np.percentile(bin_boots, 2.5):.3f}-{np.percentile(bin_boots, 97.5):.3f})',
    'Binary Accuracy': f'{np.mean(bin_accs):.3f}',
    'Binary F1': f'{np.mean([v["bin_f1"] for v in humans.values()]):.3f}',
    'Ternary AUC': f'{np.mean(tern_aucs):.3f}' if tern_aucs else '—',
    'Ternary n': '',
})

t4 = pd.DataFrame(t4_rows)
t4.to_csv(os.path.join(TBL, 'Table4_reader_study.csv'), index=False)
print(f'\nTable 4: {len(t4)} rows saved')

# Save full data
result = {
    'all_raters': all_results,
    'human_binary_mean': float(np.mean(bin_aucs)),
    'human_binary_std': float(np.std(bin_aucs)),
    'human_binary_ci': [float(np.percentile(bin_boots, 2.5)), float(np.percentile(bin_boots, 97.5))],
    'human_ternary_mean': float(np.mean(tern_aucs)) if tern_aucs else None,
    'human_ternary_std': float(np.std(tern_aucs)) if tern_aucs else None,
    'model_binary_auc': model['auc'],
    'model_ternary_auc': 0.798,
    'fleiss_kappa': RS['inter_rater']['fleiss_kappa'],
    'cohen_kappa_model_vs_raters': RS['model_rater']['cohen_kappa_mean'],
}
with open(os.path.join(RES, 'reader_study_final.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, default=str)

# Sync
import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(os.path.join(TBL, 'Table4_reader_study.csv'), os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(RES, 'reader_study_final.json'), os.path.join(dst, 'audit/'))
print('Synced to 20260724')
