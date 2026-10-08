# -*- coding: utf-8 -*-
"""
Supplementary Figures v3 — Lancet style, canonical-v2 data.
  SuppFig1: Forest plot of 95 model AUCs + 95% CI (from MODEL_INVENTORY_105.csv)
  SuppFig4: Confusion matrices for each task's best face model (from cached probs)
Output: face_v4_manuscript_delivery/manuscript/supplementary_figures/ (PNG+SVG)
        BilinGuard_Lancet_v2_20260720/supplementary/figures/
"""
import os, sys, json, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_lancet import set_rc, grid
set_rc()

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
FIG_DEST = os.path.join(RES, 'face_v4_manuscript_delivery', 'manuscript', 'supplementary_figures')
LANCET_FIG = os.path.join(RES, 'BilinGuard_Lancet_v2_20260720', 'supplementary', 'figures')
INV = pd.read_csv(os.path.join(RES, 'face_v4_manuscript_delivery', 'tables', 'MODEL_INVENTORY_complete.csv'))
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))

TASKS = ['Binary Screening', 'Ternary Grading', 'DBIL', 'IBIL', 'Jaundice Type', 'Child-Pugh', 'MELD']
SHORT = {'Binary Screening': 'Binary', 'Ternary Grading': 'Ternary', 'DBIL': 'DBIL', 'IBIL': 'IBIL',
         'Jaundice Type': 'Jaundice Type', 'Child-Pugh': 'Child-Pugh', 'MELD': 'MELD'}
SCOPE_COL = {'Face': '#3b4b63', 'Eyelid': '#c2694f', 'Fusion': '#4e9a8f'}
SCOPE_MRK = {'Face': 'o', 'Eyelid': 's', 'Fusion': 'D'}

os.makedirs(FIG_DEST, exist_ok=True)
os.makedirs(LANCET_FIG, exist_ok=True)

# ═══ SuppFig1: Forest plot ═══
rows = INV[(INV['auc'].notna()) & (~INV['Note'].astype(str).str.contains('N/A'))].copy()
rows['y'] = range(len(rows), 0, -1)
rows['label'] = rows['Task'] + ' ' + rows['Scope'] + ' ' + rows['Architecture']

fig, ax = plt.subplots(figsize=(7.5, 8))
for _, r in rows.iterrows():
    col = SCOPE_COL[r['Scope']]
    y = r['y']
    lo = r['auc'] - r['auc_lo']
    hi = r['auc_hi'] - r['auc']
    ax.errorbar(r['auc'], y, xerr=[[lo], [hi]], fmt=SCOPE_MRK[r['Scope']], color=col, ms=3.5,
                lw=0.7, capsize=1.5, zorder=3)
    ax.text(r['auc'] + hi + 0.008, y, f'{r["auc"]:.3f}', fontsize=3.5, va='center', color=col)

ax.axvline(0.5, color='#999999', lw=0.5, ls='--')
ax.set_xlim(0.3, 1.05); ax.set_xlabel('AUC (95% CI)', fontsize=8)
ax.set_yticks([])
ax.set_title('Supplementary Figure 1. Forest plot of all model AUCs (canonical-v2)',
             fontsize=9, fontweight='bold', loc='left')
grid(ax)
# task group labels
for i, task in enumerate(TASKS):
    sub = rows[rows['Task'] == task]
    if len(sub):
        ymid = sub['y'].mean()
        ax.text(-0.02, ymid, SHORT[task], ha='right', va='center', fontsize=5.5,
                fontweight='bold', transform=ax.get_yaxis_transform())
# scope legend
for s, col in SCOPE_COL.items():
    ax.plot([], [], SCOPE_MRK[s], color=col, label=s, ms=4)
ax.legend(loc='lower right', fontsize=6, frameon=False)
fig.tight_layout(rect=[0.12, 0, 1, 0.97])
for ext in ['png', 'svg']:
    fig.savefig(os.path.join(FIG_DEST, f'SuppFig1_forest_all_models.{ext}'))
plt.close(fig)
print('SuppFig1 saved')

# ═══ SuppFig4: Confusion matrices ═══
# Best face model per task
BEST = {'Binary Screening': ('face', 'convnext', 2),
        'Ternary Grading': ('face', 'convnext', 3),
        'DBIL': ('face', 'convnext', 3),
        'IBIL': ('face', 'convnext', 3),
        'Jaundice Type': ('face', 'convnext', 2),
        'Child-Pugh': ('face', 'convnext', 3),
        'MELD': ('face', 'convnext', 3)}
T2 = json.load(open(os.path.join(RES, 'typev2_probs.json'), encoding='utf-8'))

cls_names = {'Binary Screening': ['Normal', 'Jaundice'], 'Ternary Grading': ['Mild', 'Moderate', 'Severe'],
             'Jaundice Type': ['Hepato.', 'Cholest.']}
fig, axes = plt.subplots(2, 4, figsize=(9, 4.5))
axes = axes.flatten()
for idx, (task, (scope, arch_key, nc)) in enumerate(BEST.items()):
    ax = axes[idx]
    if task == 'Jaundice Type':
        d = T2[scope][arch_key]
    else:
        d = V2[task][scope].get(arch_key)
    if d is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center'); ax.axis('off'); continue
    t = np.array([v[1] for v in d.values()])
    p = np.array([np.argmax(v[0]) for v in d.values()])
    cm = confusion_matrix(t, p, labels=list(range(nc)))
    labels = cls_names.get(task, [f'C{i}' for i in range(nc)])
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(f'{SHORT[task]}', fontsize=7, fontweight='bold', pad=3)
    ax.set_xlabel('Predicted', fontsize=6)
    ax.set_ylabel('True', fontsize=6)
    for t in ax.texts:
        t.set_fontsize(5)
axes[-1].axis('off')
fig.suptitle('Supplementary Figure 4. Confusion matrices — best face model per task (canonical-v2)',
             fontsize=9, fontweight='bold', x=0.02, ha='left')
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ['png', 'svg']:
    fig.savefig(os.path.join(FIG_DEST, f'SuppFig4_confusion_matrices.{ext}'))
plt.close(fig)
print('SuppFig4 saved')

# copy existing design figs (S2, S3, S7-S13) — unchanged by model updates
STATIC_FIGS = ['SuppFig2_summary_3layer', 'SuppFig3_domain_adaptation',
               'SuppFigS7_preprocessing_pipeline', 'SuppFigS8_dataflow',
               'SuppFigS9_domain_adaptation_workflow', 'SuppFigS10_cdss_decision_tree',
               'SuppFigS11_reader_study_protocol', 'SuppFigS12_training_procedure',
               'SuppFigS13_desktop_architecture']
for name in STATIC_FIGS:
    for ext in ['png', 'svg']:
        src = os.path.join(FIG_DEST, f'{name}.{ext}')
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(LANCET_FIG, f'{name}.{ext}'))
# also copy v2 supp figs
for f in ['SuppFig1_forest_all_models.png', 'SuppFig1_forest_all_models.svg',
          'SuppFig4_confusion_matrices.png', 'SuppFig4_confusion_matrices.svg',
          'SuppFig5_binary_models.png', 'SuppFig5_binary_models.svg',
          'SuppFig6_grading_models.png', 'SuppFig6_grading_models.svg']:
    src = os.path.join(FIG_DEST, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(LANCET_FIG, f))
print('Copied to Lancat folder.')
