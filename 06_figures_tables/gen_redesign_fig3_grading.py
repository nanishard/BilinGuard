# -*- coding: utf-8 -*-
"""
Redesigned Figure 3 v2 (CRM/3D-FAICE style): grading, jaundice type, dual-view fusion.
Layout: two independent gridspecs with generous gaps.
  gsTop (2x3): A TBIL face ROCs | B TBIL eyelid ROCs | C TBIL confusion matrix
               D DBIL ROCs      | E IBIL ROCs        | F jaundice-type ROCs
  gsBot (1x2): G dual-view fusion ROC | H summary AUC bars (wide)
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_crm import (set_rc, panel_letter, roc_ax, save, SCOPE_COL,
                          GREY, LGREY, FUS,
                          pd_norm, merge_ens, roc_band, plot_model_rocs, TASK_SHORT)

set_rc()
BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
os.makedirs(OUT, exist_ok=True)

V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
FS = json.load(open(os.path.join(RES, 'fusion_shared_probs.json'), encoding='utf-8'))
T2 = json.load(open(os.path.join(RES, 'typev2_probs.json'), encoding='utf-8'))
PC = json.load(open(os.path.join(RES, 'fusion_paired_comparison.json'), encoding='utf-8'))
INV = pd.read_csv(os.path.join(RES, 'face_v4_manuscript_delivery', 'tables',
                               'MODEL_INVENTORY_complete.csv'))
ARCHS = [('convnext', 'ConvNeXt'), ('vit', 'ViT'), ('swin', 'Swin'), ('efficientnet', 'EfficientNet')]
CLS3 = ['Mild', 'Moderate', 'Severe']

fig = plt.figure(figsize=(6.9, 6.7))
gsTop = fig.add_gridspec(2, 3, left=0.095, right=0.98, top=0.975, bottom=0.40,
                         hspace=0.62, wspace=0.38)
gsBot = fig.add_gridspec(1, 2, width_ratios=[4.2, 5.8], left=0.095, right=0.98,
                         top=0.26, bottom=0.075, wspace=0.22)

# ── A/B: TBIL grading five-model ROCs ───────────────────────────────────────
for i, scope in enumerate(['face', 'eyelid']):
    ax = fig.add_subplot(gsTop[0, i])
    prob_dict = V2['Ternary Grading'][scope]
    plot_model_rocs(ax, prob_dict, 3)
    if i > 0:
        ax.set_ylabel('')
    n = len(next(iter(V2['Ternary Grading'][scope].values())))
    ax.text(0.03, 0.94, f'{scope.capitalize()} (n={n})', transform=ax.transAxes,
            fontsize=6, color=GREY)
    panel_letter(ax, 'A' if i == 0 else 'B')

# ── C: TBIL confusion matrix (facial ensemble) ─────────────────────────────
ax = fig.add_subplot(gsTop[0, 2])
dicts = [pd_norm(V2['Ternary Grading']['face'][k]) for k, _ in ARCHS
         if k in V2['Ternary Grading']['face']]
ens = merge_ens(dicts)
t = np.array([v[1] for v in ens.values()]); pred = np.array([np.argmax(v[0]) for v in ens.values()])
cm = confusion_matrix(t, pred, labels=[0, 1, 2])
im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0)
for r in range(3):
    for c in range(3):
        ax.text(c, r, str(cm[r, c]), ha='center', va='center', fontsize=8, fontweight='bold',
                color='white' if cm[r, c] > cm.max() * 0.6 else '#1A1A1A')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(CLS3, fontsize=5.5)
ax.set_yticklabels(CLS3, fontsize=5.5, rotation=90, va='center')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Facial ensemble TBIL grading (n={len(t)})', fontsize=6, pad=2)
panel_letter(ax, 'C')

# ── D/E: DBIL / IBIL grading ────────────────────────────────────────────────
for i, task in enumerate(['DBIL', 'IBIL']):
    ax = fig.add_subplot(gsTop[1, i])
    for scope in ['face', 'eyelid']:
        dicts = [pd_norm(V2[task][scope][k]) for k, _ in ARCHS if k in V2[task][scope]]
        enc = merge_ens(dicts)
        fpr, tpr, a, lo, hi = roc_band(enc, 3)
        ax.plot(fpr, tpr, '-', color=SCOPE_COL[scope.capitalize()], lw=1.4,
                label=f'{scope.capitalize()} ens ({a:.3f})')
        if lo is not None:
            ax.fill_between(np.linspace(0, 1, len(lo)), lo, hi,
                            color=SCOPE_COL[scope.capitalize()], alpha=0.12, lw=0)
    if i > 0:
        ax.set_ylabel('')
    roc_ax(ax); ax.legend(loc='lower right', fontsize=5.5)
    ax.text(0.03, 0.94, f'{task} grading', transform=ax.transAxes, fontsize=6, color=GREY)
    panel_letter(ax, 'D' if i == 0 else 'E')

# ── F: jaundice type ───────────────────────────────────────────────────────
ax = fig.add_subplot(gsTop[1, 2])
for scope in ['face', 'eyelid']:
    dicts = [pd_norm(T2[scope][k]) for k, _ in ARCHS if k in T2[scope]]
    enc = merge_ens(dicts)
    fpr, tpr, a, lo, hi = roc_band(enc, 2)
    ax.plot(fpr, tpr, '-', color=SCOPE_COL[scope.capitalize()], lw=1.4,
            label=f'{scope.capitalize()} ens ({a:.3f})')
    if lo is not None:
        ax.fill_between(np.linspace(0, 1, len(lo)), lo, hi,
                        color=SCOPE_COL[scope.capitalize()], alpha=0.12, lw=0)
ax.set_ylabel('')
roc_ax(ax); ax.legend(loc='lower right', fontsize=5.5)
n2 = len(next(iter(T2['face'].values())))
ax.text(0.03, 0.94, f'Jaundice type (n={n2})', transform=ax.transAxes, fontsize=6, color=GREY)
panel_letter(ax, 'F')

# ── G: dual-view fusion ROC (binary, shared held-out) ──────────────────────
ax = fig.add_subplot(gsBot[0, 0])
d = FS['Binary Screening']
for scope, lab in [('face', 'Face ensemble'), ('eyelid', 'Eyelid ensemble')]:
    sg = merge_ens([pd_norm(d[scope][k]) for k, _ in ARCHS if d[scope].get(k)])
    fpr, tpr, a, _, _ = roc_band(sg, 2, n_boot=0)
    ax.plot(fpr, tpr, ':', color=SCOPE_COL[scope.capitalize()], lw=1.2,
            label=f'{lab} ({a:.3f})')
dicts = [pd_norm(d[s][k]) for s in ['face', 'eyelid'] for k, _ in ARCHS if d.get(s, {}).get(k)]
fus = merge_ens(dicts)
fpr, tpr, a, lo, hi = roc_band(fus, 2)
ax.plot(fpr, tpr, '-', color=FUS, lw=1.8, label=f'Dual-view fusion ({a:.3f})')
if lo is not None:
    ax.fill_between(np.linspace(0, 1, len(lo)), lo, hi, color=FUS, alpha=0.15, lw=0)
roc_ax(ax); ax.legend(loc='lower right', fontsize=5.5)
pc = PC['Binary Screening']
ax.text(0.03, 0.94, f'Shared held-out n={pc["n"]}', transform=ax.transAxes,
        fontsize=5.5, color=GREY)
panel_letter(ax, 'G')

# ── H: summary AUC bars by input configuration ─────────────────────────────
ax = fig.add_subplot(gsBot[0, 1])
tasks = ['Binary Screening', 'Ternary Grading', 'DBIL', 'IBIL', 'Jaundice Type']
scopes = ['Face', 'Eyelid', 'Fusion']
x = np.arange(len(tasks)); w = 0.26
for j, scope in enumerate(scopes):
    vals = []
    for task in tasks:
        sub = INV[(INV['Task'] == task) & (INV['Scope'] == scope)]
        ens_row = sub[sub['Architecture'] == 'Ensemble']
        v = ens_row.iloc[0]['auc'] if len(ens_row) else sub['auc'].max()
        vals.append(v)
    ax.bar(x + (j - 1) * w, vals, w, color=SCOPE_COL[scope], alpha=0.92,
           edgecolor='white', lw=0.4, label=scope)
    for i, v in enumerate(vals):
        ax.text(x[i] + (j - 1) * w, v + 0.012, f'{v:.2f}', ha='center', fontsize=4.6,
                color=GREY)
for i, task in enumerate(tasks):
    r = PC.get(task)
    if r and (r['delta']['lo'] > 0 or r['delta']['hi'] < 0):
        ax.text(x[i] + w, 1.06, '*', ha='center', fontsize=9, fontweight='bold', color='#1A1A1A')
ax.set_xticks(x); ax.set_xticklabels([TASK_SHORT[t] for t in tasks], fontsize=6)
ax.set_ylim(0.5, 1.24); ax.set_ylabel('AUC (best architecture)')
ax.axhline(0.5, color=LGREY, lw=0.6, ls=':')
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=5.8, ncol=3,
          columnspacing=0.9, handlelength=1.2, borderaxespad=0.25)
ax.text(0.99, 0.03, '*paired fusion gain, 95% CI excludes zero', transform=ax.transAxes,
        fontsize=5, color=GREY, ha='right')
panel_letter(ax, 'H')

save(fig, os.path.join(OUT, 'Figure3_grading_fusion_crm'))
