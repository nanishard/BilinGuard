# -*- coding: utf-8 -*-
"""
Redesigned Figure 4 v3: hepatic-reserve classification + complete attribution gallery.
  A  left column: COMPLETE 7-task Grad-CAM gallery (suptitle strip cropped at y=155,
     detected by row-ink profile), axes sized to the image aspect (no letterboxing)
  B  Child-Pugh macro ROC        C  MELD macro ROC          (right column, stacked)
  D  combined per-class ROC (CP solid / MELD dashed)
  E  benchmark bars (full / restricted / TBIL-only)
  F  inference flow (image -> backbone -> 3-class head -> risk tier)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_crm import (set_rc, panel_letter, roc_ax, save, GREY, LGREY,
                          INT, EXT, POS, pd_norm, roc_band, TASK_COL)

set_rc()
BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
os.makedirs(OUT, exist_ok=True)

V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
RCP = json.load(open(os.path.join(RES, 'restricted_cp_meld_corrected.json'), encoding='utf-8'))
GAL = os.path.join(BASE, '20260911_CRM图表重设计', 'figures', 'Figure4_gallery_crm.png')

# ---- gallery: newly rendered with enlarged labels; use as-is (no crop) -------
# Panel A source priority:
#   1) _fig4_gallery_full.png  -- USER-EDITED asset (rasterised from the user's
#      edited _fig4_gallery_full.svg; NEVER overwritten by this script)
#   2) _fig4_gallery_autogen.png -- auto-generated from Figure4_gallery_crm.png
USER_PNG = os.path.join(OUT, '_fig4_gallery_full.png')
AUTO_PNG = os.path.join(OUT, '_fig4_gallery_autogen.png')
if os.path.exists(USER_PNG) and os.path.getsize(USER_PNG) > 100000:
    STRIP = USER_PNG
else:
    import shutil as _sh
    _sh.copy2(GAL, AUTO_PNG)
    STRIP = AUTO_PNG
gal = Image.open(STRIP)
AR = gal.size[0] / gal.size[1]

# ---- canvas ----
FW, FH = 6.9, 7.9
fig = plt.figure(figsize=(FW, FH))

# A: full gallery left column, exact aspect
gal_w_in = 3.75
gal_h_in = gal_w_in / AR                  # ~4.69 in
m_l = 0.025
gal_top = 0.988
gal_h_frac = gal_h_in / FH
gal_bottom = gal_top - gal_h_frac
axA = fig.add_axes([m_l, gal_bottom, gal_w_in / FW, gal_h_frac])
axA.imshow(mpimg.imread(STRIP), interpolation='lanczos')
axA.axis('off'); axA.set_xticks([]); axA.set_yticks([])
fig.text(m_l - 0.010, gal_top + 0.002, 'A', fontsize=11, fontweight='bold', va='bottom')

# right column: B (CP ROC) over C (MELD ROC)
gsR = fig.add_gridspec(2, 1, left=0.66, right=0.985, top=0.978, bottom=gal_bottom,
                       hspace=0.55)

def full_roc(task):
    t = np.array([v[1] for v in V2[task]['face']['swin'].values()])
    P = np.array([v[0] for v in V2[task]['face']['swin'].values()])
    return {i: (P[i], int(t[i])) for i in range(len(t))}, t, P

pd_cp, t_cp, P_cp = full_roc('Child-Pugh')
pd_m, t_m, P_m = full_roc('MELD')

axB = fig.add_subplot(gsR[0, 0])
fpr, tpr, a, lo, hi = roc_band(pd_cp, 3)
axB.plot(fpr, tpr, '-', color=TASK_COL['Child-Pugh'], lw=1.6)
if lo is not None:
    axB.fill_between(np.linspace(0, 1, len(lo)), lo, hi, color=TASK_COL['Child-Pugh'],
                     alpha=0.15, lw=0)
roc_ax(axB)
axB.text(0.97, 0.05, f'Child-Pugh\nmacro AUC {a:.3f}\nn = {len(t_cp)}',
         transform=axB.transAxes, fontsize=6.5, ha='right', va='bottom',
         color=TASK_COL['Child-Pugh'])
panel_letter(axB, 'B')

axC = fig.add_subplot(gsR[1, 0])
fpr, tpr, a_m, lo, hi = roc_band(pd_m, 3)
axC.plot(fpr, tpr, '-', color=TASK_COL['MELD'], lw=1.6)
if lo is not None:
    axC.fill_between(np.linspace(0, 1, len(lo)), lo, hi, color=TASK_COL['MELD'],
                     alpha=0.15, lw=0)
roc_ax(axC)
axC.text(0.97, 0.05, f'MELD\nmacro AUC {a_m:.3f}\nn = {len(t_m)}',
         transform=axC.transAxes, fontsize=6.5, ha='right', va='bottom',
         color=TASK_COL['MELD'])
panel_letter(axC, 'C')

# bottom row: D combined per-class | E benchmark bars | F flow
gsB = fig.add_gridspec(1, 3, width_ratios=[3.6, 3.6, 2.0], left=0.075, right=0.985,
                       top=0.325, bottom=0.075, wspace=0.42)

axD = fig.add_subplot(gsB[0, 0])
cp_cols = ['#8FBF7F', '#D97706', '#8B2E2E']
m_cols = ['#F0A6A0', '#D9534F', '#7A1F1F']
for c, nm in enumerate(['CP-A', 'CP-B', 'CP-C']):
    y = (t_cp == c).astype(int)
    if y.sum() == 0:
        continue
    f, tt, _ = roc_curve(y, P_cp[:, c])
    axD.plot(f, tt, '-', color=cp_cols[c], lw=1.1, label=f'{nm} ({auc(f, tt):.2f})')
for c, nm in enumerate(['MELD-L', 'MELD-M', 'MELD-H']):
    y = (t_m == c).astype(int)
    if y.sum() == 0:
        continue
    f, tt, _ = roc_curve(y, P_m[:, c])
    axD.plot(f, tt, '--', color=m_cols[c], lw=1.1, label=f'{nm} ({auc(f, tt):.2f})')
roc_ax(axD)
axD.legend(loc='lower right', fontsize=4.6, ncol=2, columnspacing=0.6, handlelength=1.4)
panel_letter(axD, 'D')

axE = fig.add_subplot(gsB[0, 1])
cp_full = roc_auc_score(t_cp, P_cp, multi_class='ovr', labels=[0, 1, 2])
meld_full = roc_auc_score(t_m, P_m, multi_class='ovr', labels=[0, 1, 2])
groups = ['Child-Pugh', 'MELD']
series = [('Full cohort', [cp_full, meld_full], '#4C78A8'),
          ('Restricted', [RCP['cp_restricted']['face_swin_auc'],
                          RCP['meld_restricted']['face_swin_auc']], '#2E8B6E'),
          ('TBIL-only', [RCP['cp_restricted']['tbil_only_auc'],
                         RCP['meld_restricted']['tbil_only_auc']], LGREY)]
x = np.arange(2); w = 0.24
for j, (lab, vals, col) in enumerate(series):
    axE.bar(x + (j - 1) * w, vals, w, color=col, edgecolor='white', lw=0.4, label=lab)
    for i, v in enumerate(vals):
        axE.text(x[i] + (j - 1) * w, v + 0.015, f'{v:.3f}', ha='center', fontsize=5.2,
                 color=GREY)
axE.set_xticks(x); axE.set_xticklabels(groups, fontsize=6.5)
axE.set_ylim(0.5, 1.19); axE.set_ylabel('Macro AUC')
axE.axhline(0.5, color=LGREY, lw=0.6, ls=':')
axE.legend(loc='upper left', fontsize=5.5, ncol=3, columnspacing=0.9,
           handlelength=1.2, borderaxespad=0.4)
axE.text(0.97, 0.03, 'Restricted: jaundiced only\n(CP n=52, MELD n=55)', transform=axE.transAxes,
         fontsize=5, color=GREY, ha='right')
panel_letter(axE, 'E')

# F: inference flow (vertical chips)
axF = fig.add_subplot(gsB[0, 2])
axF.set_xlim(0, 10); axF.set_ylim(0, 10); axF.axis('off')
axF.set_xticks([]); axF.set_yticks([])
chips = [('Facial video +\neyelid photo', '#F3F4F6', GREY),
         ('Swin-Tiny\nbackbone', '#EFF4FA', INT),
         ('3-class softmax\n(CP / MELD)', '#FBF3E3', EXT),
         ('Risk tier\nA/B/C | Low/Hi', '#FBECEA', POS)]
yy = 9.4
for i, (txt, fc, ec) in enumerate(chips):
    axF.add_patch(FancyBboxPatch((0.8, yy - 1.9), 8.4, 1.9, boxstyle='round,pad=0.06',
                                 facecolor=fc, edgecolor=ec, linewidth=0.8))
    axF.text(5.0, yy - 0.95, txt, ha='center', va='center', fontsize=5.4, color=ec,
             linespacing=1.3)
    if i < 3:
        axF.add_patch(FancyArrowPatch((5.0, yy - 2.0), (5.0, yy - 2.6), arrowstyle='-|>',
                                      mutation_scale=8, color=GREY, lw=0.9))
    yy -= 2.65
axF.set_title('F', loc='left', fontsize=11, fontweight='bold', pad=3)

save(fig, os.path.join(OUT, 'Figure4_hepatic_gradcam_crm'))
