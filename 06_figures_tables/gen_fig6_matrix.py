# -*- coding: utf-8 -*-
"""Figure 6 v2 — 105-cell matrix heatmap + PAIRED fusion comparison (shared subsets)."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_lancet import set_rc, panel_title, grid, fig_suptitle
set_rc()

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
FIG = os.path.join(RES, 'face_v4_manuscript_delivery', 'main_figures')
INV = pd.read_csv(os.path.join(RES, 'face_v4_manuscript_delivery', 'tables', 'MODEL_INVENTORY_complete.csv'))
PC = json.load(open(os.path.join(RES, 'fusion_paired_comparison.json'), encoding='utf-8'))

TASKS = ['Binary Screening', 'Ternary Grading', 'DBIL', 'IBIL',
         'Jaundice Type', 'Child-Pugh', 'MELD']
TASK_SHORT = {'Binary Screening': 'Binary', 'Ternary Grading': 'Ternary', 'DBIL': 'DBIL',
              'IBIL': 'IBIL', 'Jaundice Type': 'Type', 'Child-Pugh': 'CP', 'MELD': 'MELD'}
SCOPES = ['Face', 'Eyelid', 'Fusion']
ARCHS = ['ConvNeXt', 'ViT', 'Swin', 'EfficientNet', 'Ensemble']

mat = np.full((21, 5), np.nan)
for i, task in enumerate(TASKS):
    for j, scope in enumerate(SCOPES):
        r = i * 3 + j
        for k, arch in enumerate(ARCHS):
            c = INV[(INV['Task'] == task) & (INV['Scope'] == scope) & (INV['Architecture'] == arch)]
            if len(c) and not np.isnan(c.iloc[0]['auc']):
                mat[r, k] = c.iloc[0]['auc']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4.8), gridspec_kw={'width_ratios': [1.15, 1]})
fig_suptitle(fig, 'Figure 6. The 105-Cell Evaluation Matrix (7 tasks × 3 input configurations × 4 backbones + ensemble)')

im = ax1.imshow(mat, cmap='RdYlGn', vmin=0.35, vmax=1.0, aspect='auto')
for r in range(21):
    for k in range(5):
        if not np.isnan(mat[r, k]):
            ax1.text(k, r, f'{mat[r, k]:.2f}', ha='center', va='center', fontsize=3.4, color='black')
for i in range(7):
    ax1.axhline(i * 3 - 0.5, color='black', lw=0.8)
ax1.set_xticks(range(5))
ax1.set_xticklabels(['CNXt', 'ViT', 'Swin', 'EffN', 'Ens'], fontsize=6)
ylabels = []
for t in TASKS:
    for s in SCOPES:
        ylabels.append({'Face': '  F', 'Eyelid': '  E', 'Fusion': '  Fu'}[s])
ax1.set_yticks(range(21))
ax1.set_yticklabels(ylabels, fontsize=4)
sec = ax1.secondary_yaxis('left')
sec.set_yticks([i * 3 + 1 for i in range(7)])
sec.set_yticklabels([TASK_SHORT[t] for t in TASKS], fontsize=6.5, fontweight='bold')
sec.tick_params(length=0, pad=22)
cb = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)
cb.set_label('AUC', fontsize=6)
cb.ax.tick_params(labelsize=5)
panel_title(ax1, 'A', 'AUC across all 105 evaluation cells (all cells complete)')

# B: paired fusion comparison
cmap_t = dict(zip(TASKS, ['#3b4b63', '#8fa876', '#7570b3', '#1b9e77', '#c2694f', '#c8a165', '#4e9a8f']))
for task in TASKS:
    r = PC[task]
    x = r['best_single']['auc']
    y = r['fusion']['auc']
    exploratory = r['n'] <= 10
    ax2.scatter(x, y, s=55, color=cmap_t[task], zorder=3,
                edgecolor='black', lw=0.6,
                marker='o' if not exploratory else 'o',
                facecolors=cmap_t[task] if not exploratory else 'none')
    dx = r['delta']['mean']
    sig = '*' if r['delta']['lo'] > 0 or r['delta']['hi'] < 0 else ''
    lab = f"{TASK_SHORT[task]} (n={r['n']})" + (' expl.' if exploratory else '') + sig
    ax2.annotate(lab, (x, y), textcoords='offset points', xytext=(5, -2), fontsize=5.5, color=cmap_t[task])
ax2.plot([0.5, 1], [0.5, 1], '--', color='grey', lw=0.7)
ax2.set_xlim(0.6, 1.03); ax2.set_ylim(0.55, 1.03)
ax2.set_xlabel('Best single-view AUC (same patients)')
ax2.set_ylabel('Best fusion AUC (same patients)')
grid(ax2)
panel_title(ax2, 'B', 'Paired fusion comparison on shared held-out patients\n(ΔAUC 95% CI crosses 0 for all tasks)')
ax2.text(0.03, 0.97, 'hollow = exploratory (n≤10)', fontsize=5, color='dimgray', transform=ax2.transAxes, va='top')

fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ['png', 'svg']:
    fig.savefig(os.path.join(FIG, f'Figure6_model_matrix.{ext}'))
plt.close(fig)
print('Figure6 saved')
