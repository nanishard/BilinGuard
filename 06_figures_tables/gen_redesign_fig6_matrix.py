# -*- coding: utf-8 -*-
"""
Redesigned Figure 6 (CRM style): 105-cell evaluation matrix.
Layout: wide banner heat-map on top + 2x2 summary grid below.
  A  AUC heat-map, transposed (5 architectures x 21 task/input cells,
     blue-white-amber diverging map)
  B  paired fusion-vs-single-view scatter (task-coloured points)
  C  paired delta-AUC forest plot (fusion minus best single view, 95% CI)
  D  best facial vs best eyelid model scatter (per task, full validation sets)
  E  best-architecture winner grid (7 tasks x 3 input configurations)
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_crm import (set_rc, panel_letter, save, GREY, LGREY, READER,
                          INT, EXT, TASK_COL, TASK_SHORT, ARCH_COL, sig)

set_rc()
BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
os.makedirs(OUT, exist_ok=True)

INV = pd.read_csv(os.path.join(RES, 'face_v4_manuscript_delivery', 'tables',
                               'MODEL_INVENTORY_complete.csv'))
PC = json.load(open(os.path.join(RES, 'fusion_paired_comparison.json'), encoding='utf-8'))
RS = json.load(open(os.path.join(RES, 'reader_study_final.json'), encoding='utf-8'))

TASKS = ['Binary Screening', 'Ternary Grading', 'DBIL', 'IBIL',
         'Jaundice Type', 'Child-Pugh', 'MELD']
SCOPES = ['Face', 'Eyelid', 'Fusion']
ARCHS = ['ConvNeXt', 'ViT', 'Swin', 'EfficientNet', 'Ensemble']
ARCH_SHORT = {'ConvNeXt': 'CNXt', 'ViT': 'ViT', 'Swin': 'Swin',
              'EfficientNet': 'EffN', 'Ensemble': 'Ens'}

mat = np.full((21, 5), np.nan)
for i, task in enumerate(TASKS):
    for j, scope in enumerate(SCOPES):
        c = INV[(INV['Task'] == task) & (INV['Scope'] == scope) &
                (INV['Architecture'].isin(ARCHS))]
        for k, arch in enumerate(ARCHS):
            r = c[c['Architecture'] == arch]
            if len(r) and not np.isnan(r.iloc[0]['auc']):
                mat[i * 3 + j, k] = r.iloc[0]['auc']

CMAP = LinearSegmentedColormap.from_list('bwa', [INT, '#FFFFFF', EXT], N=256)

fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.85, 2.35, 2.35],
                      hspace=0.52, wspace=0.40,
                      left=0.075, right=0.945, top=0.955, bottom=0.075)

# --------------------------------------- A: transposed 105-cell heat-map (banner)
axA = fig.add_subplot(gs[0, :])
im = axA.imshow(mat.T, cmap=CMAP, vmin=0.35, vmax=1.0, aspect='auto')
for r in range(21):
    for k in range(5):
        if not np.isnan(mat[r, k]):
            v = mat[r, k]
            axA.text(r, k, f'{v:.2f}', ha='center', va='center', fontsize=4.2,
                     color='white' if (v > 0.88 or v < 0.5) else '#1A1A1A')
for i in range(1, 7):
    axA.axvline(i * 3 - 0.5, color='#1A1A1A', lw=0.7)
axA.set_yticks(range(5))
axA.set_yticklabels(['CNXt', 'ViT', 'Swin', 'EffN', 'Ens'], fontsize=5.5)
axA.set_xticks(range(21))
axA.set_xticklabels(['F', 'E', 'Fu'] * 7, fontsize=4)
sec = axA.secondary_xaxis('top')
sec.set_xticks([i * 3 + 1 for i in range(7)])
sec.set_xticklabels([TASK_SHORT[t] for t in TASKS], fontsize=6, fontweight='bold')
sec.tick_params(length=0, pad=2)
posA = axA.get_position()
cax = fig.add_axes([posA.x1 + 0.008, posA.y0, 0.011, posA.height])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=4.5, length=1.2, pad=0.8)
cax.set_title('AUC', fontsize=5, pad=1.5)
panel_letter(axA, 'A')
axA.set_title('105 evaluation cells (7 tasks x 3 inputs x 5 models)', fontsize=6, pad=14)

axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])
axD = fig.add_subplot(gs[2, 0])
axE = fig.add_subplot(gs[2, 1])

# --------------------------------------- B: paired fusion-vs-single-view scatter
handlesB = []
for task in TASKS:
    r = PC[task]
    x = r['best_single']['auc']; y = r['fusion']['auc']
    exploratory = r['n'] <= 10
    axB.scatter(x, y, s=48, zorder=3, lw=0.6,
                color=TASK_COL[task], edgecolor='#1A1A1A',
                facecolors='none' if exploratory else TASK_COL[task])
    handlesB.append(Line2D([0], [0], marker='o', color='none', markersize=4.5,
                           markerfacecolor='none' if exploratory else TASK_COL[task],
                           markeredgecolor=TASK_COL[task] if exploratory else '#1A1A1A',
                           markeredgewidth=0.7,
                           label=f"{TASK_SHORT[task]} (n={r['n']})"))
# reader-study means (8 clinicians) marked on the diagonal for reference
for key, lab in [('human_binary_mean', 'Reader mean, binary'),
                 ('human_ternary_mean', 'Reader mean, ternary')]:
    v = RS[key]
    axB.scatter([v], [v], marker='*', s=85, zorder=4,
                facecolors=READER, edgecolors='#1A1A1A', linewidths=0.5)
    handlesB.append(Line2D([0], [0], marker='*', color='none', markersize=6.5,
                           markerfacecolor=READER, markeredgecolor='#1A1A1A',
                           markeredgewidth=0.5, label=lab))
axB.plot([0.5, 1], [0.5, 1], ':', color=LGREY, lw=0.8)
axB.set_xlim(0.6, 1.02); axB.set_ylim(0.55, 1.03)
axB.set_xticks(np.arange(0.6, 1.01, 0.1))
axB.set_yticks(np.arange(0.6, 1.01, 0.1))
axB.set_xlabel('Best single-view AUC (shared patients)', fontsize=6)
axB.set_ylabel('Best fusion AUC (same patients)', fontsize=6)
axB.tick_params(labelsize=5.5)
axB.legend(handles=handlesB, loc='lower right', fontsize=5, frameon=False,
           handletextpad=0.25, borderaxespad=0.25, labelspacing=0.4)
panel_letter(axB, 'B')
axB.set_title('Paired fusion comparison (hollow = exploratory, n\u226410)\n'
              'red star = reader-study mean (8 clinicians)', fontsize=5.5, pad=2)

# --------------------------------------- C: paired delta-AUC forest plot
ys = np.arange(len(TASKS))[::-1].astype(float)
for y, task in zip(ys, TASKS):
    r = PC[task]
    m, lo, hi = r['delta']['mean'], r['delta']['lo'], r['delta']['hi']
    exploratory = r['n'] <= 10
    axC.plot([lo, hi], [y, y], '-', color=TASK_COL[task], lw=1.3, zorder=2)
    for x in (lo, hi):
        axC.plot([x, x], [y - 0.14, y + 0.14], color=TASK_COL[task], lw=1.0, zorder=2)
    axC.scatter([m], [y], s=36, zorder=3, edgecolor='#1A1A1A', lw=0.6,
                color=TASK_COL[task],
                facecolors='none' if exploratory else TASK_COL[task])
axC.axvline(0, color=GREY, ls=':', lw=0.8)
axC.set_yticks(ys)
axC.set_yticklabels([f"{TASK_SHORT[t]} (n={PC[t]['n']})" for t in TASKS], fontsize=5.5)
axC.set_xlim(-0.18, 0.16)
axC.set_xticks(np.arange(-0.15, 0.151, 0.05))
axC.tick_params(axis='x', labelsize=5.5)
axC.set_ylim(-0.8, 6.6)
axC.set_xlabel('Paired \u0394AUC (fusion \u2212 best single view)', fontsize=6)
axC.text(-0.175, -0.55, '\u2190 favours single view', fontsize=4.5, color=GREY,
         ha='left', va='center')
axC.text(0.155, -0.55, 'favours fusion \u2192', fontsize=4.5, color=GREY,
         ha='right', va='center')
panel_letter(axC, 'C')
axC.set_title('Fusion gain with 95% CI\n(hollow = exploratory, n\u226410)', fontsize=5.5, pad=2)

# --------------------------------------- D: best facial vs best eyelid model
handlesD = []
for task in TASKS:
    fa = INV[(INV['Task'] == task) & (INV['Scope'] == 'Face')]['auc'].max()
    ey = INV[(INV['Task'] == task) & (INV['Scope'] == 'Eyelid')]['auc'].max()
    axD.scatter(ey, fa, s=42, color=TASK_COL[task], zorder=3,
                edgecolor='#1A1A1A', lw=0.6)
    handlesD.append(Line2D([0], [0], marker='o', color='none', markersize=4.5,
                           markerfacecolor=TASK_COL[task],
                           markeredgecolor='#1A1A1A', markeredgewidth=0.7,
                           label=TASK_SHORT[task]))
axD.plot([0.6, 1.0], [0.6, 1.0], ':', color=LGREY, lw=0.8)
axD.set_xlim(0.60, 0.92); axD.set_ylim(0.68, 1.02)
axD.set_xticks(np.arange(0.6, 0.91, 0.1))
axD.set_yticks(np.arange(0.7, 1.01, 0.1))
axD.set_xlabel('Best eyelid-model AUC (per task)', fontsize=6)
axD.set_ylabel('Best facial-model AUC (per task)', fontsize=6)
axD.tick_params(labelsize=5.5)
axD.legend(handles=handlesD, loc='lower right', fontsize=5, frameon=False,
           handletextpad=0.25, borderaxespad=0.25, labelspacing=0.4)
panel_letter(axD, 'D')
axD.set_title('Facial vs eyelid best models\n(below diagonal = eyelid superior)',
              fontsize=5.5, pad=2)

# --------------------------------------- E: best-architecture winner grid
codes = np.zeros((7, 3))
wval = np.zeros((7, 3))
warch = [[None] * 3 for _ in range(7)]
for i, task in enumerate(TASKS):
    for j, scope in enumerate(SCOPES):
        c = INV[(INV['Task'] == task) & (INV['Scope'] == scope)]
        b = c.loc[c['auc'].idxmax()]
        codes[i, j] = ARCHS.index(b['Architecture'])
        wval[i, j] = b['auc']
        warch[i][j] = b['Architecture']
# panel-local palette: Ensemble black -> red (keeps global ARCH_COL for curves)
E_COL = dict(ARCH_COL)
E_COL['Ensemble'] = '#E45756'
axE.imshow(codes, cmap=ListedColormap([E_COL[a] for a in ARCHS]),
           vmin=0, vmax=len(ARCHS) - 1, aspect='auto')
for i in range(7):
    for j in range(3):
        axE.text(j, i, f'{ARCH_SHORT[warch[i][j]]}\n{wval[i, j]:.2f}',
                 ha='center', va='center', fontsize=4.2, color='#1A1A1A', linespacing=1.1)
axE.set_xticks(range(3)); axE.set_xticklabels(SCOPES, fontsize=5.5)
axE.set_yticks(range(7)); axE.set_yticklabels([TASK_SHORT[t] for t in TASKS], fontsize=5.5)
axE.set_xticks(np.arange(-0.5, 3, 1), minor=True)
axE.set_yticks(np.arange(-0.5, 7, 1), minor=True)
axE.grid(which='minor', color='#1A1A1A', lw=0.7)
axE.tick_params(which='both', length=0)
panel_letter(axE, 'E')
axE.set_title('Best architecture per cell (by AUC)', fontsize=5.5, pad=2)
handles = [Line2D([0], [0], marker='s', color='none', markerfacecolor=E_COL[a],
                  markeredgecolor='#1A1A1A', markeredgewidth=0.4,
                  markersize=5, label=ARCH_SHORT[a]) for a in ARCHS]
axE.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.16),
           ncol=5, fontsize=4.5, handletextpad=0.15, columnspacing=0.6,
           borderaxespad=0, frameon=False)

save(fig, os.path.join(OUT, 'Figure6_matrix_crm'))
