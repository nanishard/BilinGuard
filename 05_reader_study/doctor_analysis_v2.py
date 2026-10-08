# -*- coding: utf-8 -*-
"""
Doctor Performance: Individual metrics + ROC scatter points.
Uses doctor/*.xlsx files (8 individual + 1 summary).
No bootstrap — doctors are plotted as points on model ROC curves.
Computes full metrics: F1, Sens, Spec, Acc, AUC for binary AND ternary.
"""
import os, glob, json, numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, average_precision_score)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\research\人脸识别营养\传染科'
DOCTOR_DIR = os.path.join(BASE, 'doctor')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'tables')
FIG = os.path.join(RES, 'figures')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})

# ── Load summary ─────────────────────────────────────────────
summary = pd.read_excel(os.path.join(DOCTOR_DIR, '医生表现汇总.xlsx'))
summary.columns = ['doc_id', 'task_type', 'FPR', 'TPR', 'roc_coord', 'source', 'ACC', 'col7', 'time_avg']
print('=== Doctor Summary ===')
print(summary.to_string())

# ── Load individual doctor files ─────────────────────────────
doctor_files = sorted([os.path.join(DOCTOR_DIR, f) for f in os.listdir(DOCTOR_DIR)
                       if f.endswith('.xlsx') and not f.startswith('~') and '汇总' not in f and '表现' not in f])
print(f'\nIndividual files: {len(doctor_files)}')
for f in doctor_files:
    print(f'  {os.path.basename(f)}')

# Load GT from the first doctor file (Col3 = correct answer)
ref = pd.read_excel(doctor_files[0])
ref = ref.dropna(subset=[ref.columns[0]])
ref['case_id'] = ref.iloc[:, 0].astype(int)
ref['gt'] = pd.to_numeric(ref.iloc[:, 3], errors='coerce')  # Col3 = 答案
ref = ref.dropna(subset=['gt'])
ref['gt'] = ref['gt'].astype(int)
gt_map = dict(zip(ref['case_id'], ref['gt']))
case_ids = np.array(sorted(gt_map.keys()))
gt = np.array([gt_map[c] for c in case_ids])
print(f'GT: {len(gt)} cases, 1={sum(gt==1)}, 2={sum(gt==2)}, 3={sum(gt==3)}, 4={sum(gt==4)}')

# gt: 1=normal, 2=mild, 3=moderate, 4=severe
gt_binary = (gt > 1).astype(int)  # 0=normal, 1=jaundiced
jaun_mask = gt > 1
gt_ternary = gt[jaun_mask] - 2  # 0=mild, 1=moderate, 2=severe


def compute_metrics(true, pred, nc):
    cm = confusion_matrix(true, pred, labels=list(range(nc)))
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro', labels=list(range(nc)), zero_division=0)
    sens_list, spec_list = [], []
    for c in range(nc):
        tp = cm[c,c]; fn = cm[c,:].sum()-tp; fp = cm[:,c].sum()-tp; tn = cm.sum()-tp-fn-fp
        sens_list.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        spec_list.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    sens = float(np.mean(sens_list))
    spec = float(np.mean(spec_list))
    # FPR = 1 - specificity (for plotting)
    fpr = 1 - spec
    tpr = sens
    # AUC: use one-vs-rest probability approximation
    probs = np.zeros((len(true), nc))
    for i in range(len(pred)): probs[i, pred[i]] = 1.0
    try:
        auc = roc_auc_score(true, probs[:,1]) if nc==2 else roc_auc_score(true, probs, multi_class='ovr', labels=list(range(nc)))
    except: auc = 0.5
    try:
        if nc == 2:
            ap = average_precision_score(true, probs[:,1])
        else:
            from sklearn.preprocessing import label_binarize
            yb = label_binarize(true, classes=list(range(nc)))
            ap = np.mean([average_precision_score(yb[:,c], probs[:,c]) for c in range(nc)])
    except: ap = 0
    return {'f1': f1, 'sens': sens, 'spec': spec, 'acc': acc, 'auc': auc, 'ap': ap,
            'fpr': fpr, 'tpr': tpr, 'cm': cm.tolist()}


# ── Process each doctor ──────────────────────────────────────
all_results = []
doctor_points_binary = []  # (fpr, tpr, label) for ROC plotting
doctor_points_ternary = []

doctor_names_map = {
    '题-hao.xlsx': 'Doctor-Hao',
    '题-传染科1(1).xlsx': 'Doctor-Infectious1',
    '题-公卫1.xlsx': 'Doctor-PubHealth1',
    '题-公卫2.xlsx': 'Doctor-PubHealth2',
    '题-眼科医生.xlsx': 'Doctor-Ophthalmology',
    '题-眼科医生2.xlsx': 'Doctor-Ophthalmology2',
    '题-口腔医生.xlsx': 'Doctor-Stomatology',
    '题-肝外专培.xlsx': 'Doctor-Hepatobiliary',
}

for fpath in doctor_files:
    fname = os.path.basename(fpath)
    dname = doctor_names_map.get(fname, fname.replace('.xlsx',''))
    df = pd.read_excel(fpath)
    df = df.dropna(subset=[df.columns[0]])
    df['case_id'] = df.iloc[:, 0].astype(int)
    df['doc_answer'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')  # Col2 = doctor's answer
    df = df.dropna(subset=['doc_answer'])
    df['doc_answer'] = df['doc_answer'].astype(int)

    case_lookup = dict(zip(df['case_id'], df['doc_answer']))
    preds = np.array([case_lookup.get(c, -1) for c in case_ids])
    valid = preds > 0
    if sum(valid) < 50:
        print(f'  SKIP {dname}: only {sum(valid)} valid')
        continue

    # Binary
    pred_bin = (preds[valid] > 1).astype(int)
    gt_bin = gt_binary[valid]
    m_bin = compute_metrics(gt_bin, pred_bin, 2)
    doctor_points_binary.append((m_bin['fpr'], m_bin['tpr'], dname, m_bin['auc']))

    # Ternary (jaundice only)
    jaun_valid = valid & jaun_mask
    pred_tern = preds[jaun_valid] - 2  # 2→0(mild), 3→1(mod), 4→2(sev)
    pred_tern = np.clip(pred_tern, 0, 2)
    gt_tern = gt_ternary[jaun_valid[jaun_mask]]
    if len(gt_tern) > 5:
        m_tern = compute_metrics(gt_tern, pred_tern, 3)
        doctor_points_ternary.append((m_tern['fpr'], m_tern['tpr'], dname, m_tern['auc']))
    else:
        m_tern = {'f1': 0, 'sens': 0, 'spec': 0, 'acc': 0, 'auc': 0, 'ap': 0, 'cm': []}

    all_results.append({
        'doctor': dname, 'task': 'Binary',
        'f1': m_bin['f1'], 'sens': m_bin['sens'], 'spec': m_bin['spec'],
        'acc': m_bin['acc'], 'auc': m_bin['auc'], 'ap': m_bin['ap'],
        'fpr': m_bin['fpr'], 'tpr': m_bin['tpr'], 'n': int(sum(valid)),
        'cm': str(m_bin['cm']),
    })
    all_results.append({
        'doctor': dname, 'task': 'Ternary',
        'f1': m_tern['f1'], 'sens': m_tern['sens'], 'spec': m_tern['spec'],
        'acc': m_tern['acc'], 'auc': m_tern['auc'], 'ap': m_tern['ap'],
        'fpr': m_tern.get('fpr', 0), 'tpr': m_tern.get('tpr', 0), 'n': int(sum(jaun_valid)),
        'cm': str(m_tern.get('cm', [])),
    })
    print(f'  {dname}: Binary AUC={m_bin["auc"]:.3f} (FPR={m_bin["fpr"]:.3f}, TPR={m_bin["tpr"]:.3f}) | Ternary AUC={m_tern["auc"]:.3f} (FPR={m_tern.get("fpr",0):.3f}, TPR={m_tern.get("tpr",0):.3f})')


# ── Aggregate ────────────────────────────────────────────────
# Majority vote binary
all_preds = []
for fpath in doctor_files:
    fname = os.path.basename(fpath)
    df = pd.read_excel(fpath)
    df = df.dropna(subset=[df.columns[0]])
    df['case_id'] = df.iloc[:, 0].astype(int)
    df['doc_answer'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    df = df.dropna(subset=['doc_answer'])
    lookup = dict(zip(df['case_id'], df['doc_answer'].astype(int)))
    preds = np.array([int(lookup.get(c, -1) > 1) for c in case_ids])
    all_preds.append(preds)
all_preds = np.array(all_preds)  # (n_doctors, n_cases)
valid_mask = np.all(all_preds >= 0, axis=0)
maj_bin = (np.mean(all_preds[:, valid_mask], axis=0) > 0.5).astype(int)
m_agg = compute_metrics(gt_binary[valid_mask], maj_bin, 2)
print(f'\n  Aggregate: Binary AUC={m_agg["auc"]:.3f} F1={m_agg["f1"]:.3f} Sens={m_agg["sens"]:.3f} Spec={m_agg["spec"]:.3f}')
all_results.append({'doctor': 'Aggregate', 'task': 'Binary', **{k: m_agg[k] for k in ['f1','sens','spec','acc','auc','ap','fpr','tpr']},
                     'n': int(sum(valid_mask)), 'cm': str(m_agg['cm'])})


# ── Print manuscript table ───────────────────────────────────
print('\n' + '='*100)
print('  INDIVIDUAL DOCTOR RESULTS (no bootstrap — point estimates)')
print('='*100)

rdf = pd.DataFrame(all_results)
for task in ['Binary', 'Ternary']:
    sub = rdf[rdf['task'] == task]
    print(f'\n--- {task} ---')
    print(f'{"Doctor":<25} {"F1":>8} {"Sens":>8} {"Spec":>8} {"Acc":>8} {"AUC":>8} {"AP":>8} {"FPR":>8} {"TPR":>8}')
    print('-' * 95)
    for _, r in sub.sort_values('auc', ascending=False).iterrows():
        print(f'{r["doctor"]:<25} {r["f1"]:>8.3f} {r["sens"]:>8.3f} {r["spec"]:>8.3f} {r["acc"]:>8.3f} {r["auc"]:>8.3f} {r["ap"]:>8.3f} {r["fpr"]:>8.3f} {r["tpr"]:>8.3f}')

# Save
rdf.to_csv(os.path.join(TBL, 'doctor_individual_metrics.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(TBL, 'doctor_individual_metrics.tsv'), 'w', encoding='utf-8') as f:
    f.write('Task\tDoctor\tF1\tSensitivity\tSpecificity\tAccuracy\tAUC\tAP\tFPR\tTPR\tN\n')
    for _, r in rdf.iterrows():
        f.write(f'{r["task"]}\t{r["doctor"]}\t{r["f1"]:.4f}\t{r["sens"]:.4f}\t{r["spec"]:.4f}\t{r["acc"]:.4f}\t{r["auc"]:.4f}\t{r["ap"]:.4f}\t{r["fpr"]:.4f}\t{r["tpr"]:.4f}\t{int(r["n"])}\n')

# Save points for plotting
with open(os.path.join(RES, 'doctor_roc_points.json'), 'w') as f:
    json.dump({'binary': [(p[0], p[1], p[2], p[3]) for p in doctor_points_binary],
               'ternary': [(p[0], p[1], p[2], p[3]) for p in doctor_points_ternary]}, f, indent=2)


# ── Plot: ROC with doctor points ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Binary ROC with doctor points
ax = ax1
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
colors = plt.cm.Set2(np.linspace(0, 1, len(doctor_points_binary)))
for i, (fpr, tpr, name, auc) in enumerate(doctor_points_binary):
    short = name.replace('Doctor-', 'D')
    ax.scatter(fpr, tpr, s=100, c=[colors[i]], edgecolors='black', lw=0.5, zorder=5)
    ax.annotate(f'{short}\n({auc:.3f})', (fpr, tpr), fontsize=6, ha='left',
                 textcoords='offset points', xytext=(5, 5))
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
ax.set_title('Binary Screening: Doctor Performance Points', fontsize=11, fontweight='bold')
ax.set_xlim([-0.05, 0.6]); ax.set_ylim([0.2, 1.05])

# Ternary
ax = ax2
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
for i, (fpr, tpr, name, auc) in enumerate(doctor_points_ternary):
    short = name.replace('Doctor-', 'D')
    ax.scatter(fpr, tpr, s=100, c=[colors[i]], edgecolors='black', lw=0.5, zorder=5)
    ax.annotate(f'{short}\n({auc:.3f})', (fpr, tpr), fontsize=6, ha='left',
                 textcoords='offset points', xytext=(5, 5))
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
ax.set_title('Ternary Grading: Doctor Performance Points', fontsize=11, fontweight='bold')
ax.set_xlim([-0.05, 0.6]); ax.set_ylim([0.1, 0.8])

plt.tight_layout()
fig_path = os.path.join(FIG, 'Fig_doctor_roc_points.png')
fig.savefig(fig_path); fig.savefig(fig_path.replace('.png', '.svg'))
plt.close()

print(f'\n  CSV: {TBL}/doctor_individual_metrics.csv')
print(f'  TSV: {TBL}/doctor_individual_metrics.tsv')
print(f'  Points: {RES}/doctor_roc_points.json')
print(f'  Figure: {fig_path}')
