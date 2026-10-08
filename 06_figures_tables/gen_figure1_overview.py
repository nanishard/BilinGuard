# -*- coding: utf-8 -*-
"""
Figure 1 — System Overview (programmatic, 4 panels).

A: Three-layer closed-loop architecture schematic
B: CONSORT-style cohort flow diagram
C: Deep learning pipeline (SAM → CLAHE → multi-backbone → patient-level)
D: Desktop application dashboard (rendered schematic)
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(RES, 'figures_v3', 'main')

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})

L1_COLOR = '#1D3557'; L1_BG = '#E0E7FF'
L2_COLOR = '#D97706'; L2_BG = '#FEF3C7'
L3_COLOR = '#DC2626'; L3_BG = '#FEE2E2'
DA_COLOR = '#2A9D8F'; NEUTRAL = '#6B7280'

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
fig.suptitle('Figure 1. BilinGuard Closed-Loop AI System — Architecture, Cohort, and Pipeline',
             fontsize=14, fontweight='bold', y=0.98)


# ═════════════════════════════════════════════════════════════
# Panel A: Three-Layer Architecture
# ═════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(A) Three-Layer Closed-Loop Architecture', fontsize=11, fontweight='bold', loc='left')

# Input box at top
inp = FancyBboxPatch((3.5, 8.5), 3, 1.2, boxstyle='round,pad=0.1',
                      facecolor='#F3F4F6', edgecolor=NEUTRAL, lw=1.5)
ax.add_patch(inp)
ax.text(5, 9.1, 'Facial Video\n(10-20s, bedside)', ha='center', va='center', fontsize=8, fontweight='bold')

# Arrow down
ax.annotate('', xy=(5, 8.0), xytext=(5, 8.5),
             arrowprops=dict(arrowstyle='->', lw=1.5, color=NEUTRAL))

# Layer 1: Recognition (big box with sub-tasks)
l1 = FancyBboxPatch((0.5, 5.5), 9, 2.3, boxstyle='round,pad=0.15',
                     facecolor=L1_BG, edgecolor=L1_COLOR, lw=2)
ax.add_patch(l1)
ax.text(0.8, 7.5, 'Layer 1: Jaundice Recognition', fontsize=10, fontweight='bold', color=L1_COLOR)
# Sub-task boxes
sub_tasks_l1 = [
    ('Screening\nAUC 0.993', 1.2, 6.0),
    ('TBIL Grade\n0.955 / 0.858', 2.8, 6.0),
    ('DBIL Grade\n0.815', 4.4, 6.0),
    ('IBIL Grade\n0.720', 6.0, 6.0),
    ('Jaundice Type\n0.993', 7.6, 6.0),
]
for label, x, y in sub_tasks_l1:
    sub = FancyBboxPatch((x-0.7, y-0.3), 1.4, 1.0, boxstyle='round,pad=0.05',
                          facecolor='white', edgecolor=L1_COLOR, lw=1, alpha=0.9)
    ax.add_patch(sub)
    ax.text(x, y+0.2, label, ha='center', va='center', fontsize=6.5)

ax.annotate('', xy=(5, 5.2), xytext=(5, 5.5),
             arrowprops=dict(arrowstyle='->', lw=1.5, color=NEUTRAL))

# Layer 2: Prognostic Scoring
l2 = FancyBboxPatch((1.5, 3.0), 7, 2.0, boxstyle='round,pad=0.15',
                     facecolor=L2_BG, edgecolor=L2_COLOR, lw=2)
ax.add_patch(l2)
ax.text(1.8, 4.7, 'Layer 2: Prognostic Scoring (Non-Invasive Breakthrough)', fontsize=10,
         fontweight='bold', color=L2_COLOR)
# CP and MELD boxes
cp_box = FancyBboxPatch((2.0, 3.2), 2.5, 1.2, boxstyle='round,pad=0.05',
                         facecolor='white', edgecolor=L2_COLOR, lw=1, alpha=0.9)
ax.add_patch(cp_box)
ax.text(3.25, 3.8, 'Child-Pugh A/B/C\nAUC 0.907\n(no blood draw)', ha='center', va='center',
         fontsize=7, fontweight='bold')
meld_box = FancyBboxPatch((5.5, 3.2), 2.5, 1.2, boxstyle='round,pad=0.05',
                           facecolor='white', edgecolor=L2_COLOR, lw=1, alpha=0.9)
ax.add_patch(meld_box)
ax.text(6.75, 3.8, 'MELD Risk\nAUC 0.927\n(no blood draw)', ha='center', va='center',
         fontsize=7, fontweight='bold')

ax.annotate('', xy=(5, 2.7), xytext=(5, 3.0),
             arrowprops=dict(arrowstyle='->', lw=1.5, color=NEUTRAL))

# Layer 3: Decision Support
l3 = FancyBboxPatch((0.5, 0.5), 9, 2.0, boxstyle='round,pad=0.15',
                     facecolor=L3_BG, edgecolor=L3_COLOR, lw=2)
ax.add_patch(l3)
ax.text(0.8, 2.2, 'Layer 3: Clinical Decision Support', fontsize=10, fontweight='bold', color=L3_COLOR)
# CDSS components
cdss_items = [
    '64\nGuidelines', '17\nSpecialties', '17\nDiseases', '6 Red-flag\nPathways', '9 Lab\nParameters'
]
for i, item in enumerate(cdss_items):
    x = 1.2 + i * 1.8
    sub = FancyBboxPatch((x-0.7, 0.7), 1.4, 1.0, boxstyle='round,pad=0.05',
                          facecolor='white', edgecolor=L3_COLOR, lw=1, alpha=0.9)
    ax.add_patch(sub)
    ax.text(x, 1.2, item, ha='center', va='center', fontsize=6.5)

# Feedback arrow (the "loop")
ax.annotate('', xy=(0.3, 9.1), xytext=(0.3, 1.2),
             arrowprops=dict(arrowstyle='->', lw=2, color=L3_COLOR, ls='--',
                              connectionstyle='arc3,rad=-0.3'))
ax.text(0.05, 5.0, 'Closed\nLoop', ha='center', va='center', fontsize=8,
         fontweight='bold', color=L3_COLOR, rotation=90)


# ═════════════════════════════════════════════════════════════
# Panel B: CONSORT-style Cohort Flow
# ═════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(B) Study Cohort', fontsize=11, fontweight='bold', loc='left')

def consort_box(ax, x, y, w, h, text, color='#F3F4F6', edge=NEUTRAL, fontsize=7):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.1',
                          facecolor=color, edgecolor=edge, lw=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

def consort_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color=NEUTRAL))

# Top: Screened
consort_box(ax, 5, 9.3, 5, 0.9, '1,206 Inpatients Screened\n(West China Hospital, Nov 2022 - Aug 2024)',
             '#DBEAFE', '#1D3557', 8)
consort_arrow(ax, 5, 8.85, 5, 8.3)

# Videos recorded
consort_box(ax, 5, 7.9, 4, 0.8, '942 Facial Videos Recorded', '#E0E7FF', L1_COLOR, 8)
consort_arrow(ax, 5, 7.5, 5, 6.9)

# Exclusions
consort_box(ax, 8.5, 7.0, 2.5, 0.7, 'Excluded (n=3):\n1 motion blur\n2 unmatched labs',
             '#FEE2E2', L3_COLOR, 6)
consort_arrow(ax, 6.5, 7.9, 7.5, 7.2)

# Enrolled
consort_box(ax, 5, 6.5, 4, 0.7, '939 Enrolled (Internal)', '#D1FAE5', DA_COLOR, 8)
consort_arrow(ax, 3, 6.2, 2.5, 5.5)
consort_arrow(ax, 7, 6.2, 7.5, 5.5)

# Two branches: Non-jaundice and Jaundice
consort_box(ax, 2, 5.0, 3.5, 0.8, 'Non-Jaundice Controls\nn = 625\n(Questionnaire)', '#E0E7FF', L1_COLOR, 7)
consort_box(ax, 8, 5.0, 3.5, 0.8, 'Jaundice Patients\nn = 313\n(Internal Validation)', '#FEF3C7', L2_COLOR, 7)

# Below: SAM processed
consort_arrow(ax, 5, 4.5, 5, 4.0)
consort_box(ax, 5, 3.5, 6, 0.8, 'SAM Face Segmentation + CLAHE\n579 Patients with Usable Images',
             '#F3F4F6', NEUTRAL, 8)

# Split
consort_arrow(ax, 3.5, 3.0, 3, 2.5)
consort_arrow(ax, 6.5, 3.0, 7, 2.5)
consort_box(ax, 2.5, 2.0, 3, 0.7, 'Training Set\n~80% (n~465)', '#E0E7FF', L1_COLOR, 7)
consort_box(ax, 7.5, 2.0, 3, 0.7, 'Validation Set\n~20% (n~116)', '#D1FAE5', DA_COLOR, 7)

# External validation
consort_arrow(ax, 8, 4.5, 8.8, 0.8)
consort_box(ax, 8.8, 0.5, 2.5, 0.6, 'External\nn = 117\n(59N + 58J)', '#FEE2E2', L3_COLOR, 7)


# ═════════════════════════════════════════════════════════════
# Panel C: Pipeline Diagram
# ═════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(C) Deep Learning Inference Pipeline', fontsize=11, fontweight='bold', loc='left')

# Horizontal flow
steps = [
    (1.0, 'Video\nInput', '#F3F4F6', NEUTRAL),
    (2.5, '12 Frame\nSampling', '#E0E7FF', L1_COLOR),
    (4.0, 'SAM Face\nSegmentation', '#D1FAE5', DA_COLOR),
    (5.5, 'CLAHE\n+ Grey World', '#D1FAE5', DA_COLOR),
    (7.0, 'Multi-Backbone\nEnsemble', '#FEF3C7', L2_COLOR),
    (8.5, 'Patient-Level\nAggregation', '#FEE2E2', L3_COLOR),
]
for x, label, color, edge in steps:
    box = FancyBboxPatch((x-0.6, 7.5), 1.2, 1.2, boxstyle='round,pad=0.1',
                          facecolor=color, edgecolor=edge, lw=1.5)
    ax.add_patch(box)
    ax.text(x, 8.1, label, ha='center', va='center', fontsize=6.5, fontweight='bold')
    if x < 8.5:
        ax.annotate('', xy=(x+0.9, 8.1), xytext=(x+0.6, 8.1),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color=NEUTRAL))

# Backbones detail
ax.text(7.0, 6.8, 'ConvNeXt  |  ViT-T/16  |  Swin-T  |  EfficientNet-B0',
         ha='center', fontsize=7, color=L2_COLOR, fontweight='bold')

# Output boxes (7 tasks)
ax.text(5, 5.8, '7 Concurrent Clinical Outputs:', ha='center', fontsize=9, fontweight='bold')
outputs = [
    ('Screen', L1_COLOR), ('TBIL', L1_COLOR), ('DBIL', L1_COLOR),
    ('IBIL', L1_COLOR), ('Type', L1_COLOR), ('Child-Pugh', L2_COLOR), ('MELD', L2_COLOR)
]
for i, (name, color) in enumerate(outputs):
    x = 1.0 + i * 1.2
    box = FancyBboxPatch((x-0.45, 4.5), 0.9, 0.8, boxstyle='round,pad=0.05',
                          facecolor='white', edgecolor=color, lw=1.2)
    ax.add_patch(box)
    ax.text(x, 4.9, name, ha='center', va='center', fontsize=6, fontweight='bold', color=color)

# Arrow to CDSS
ax.annotate('', xy=(5, 3.8), xytext=(5, 4.4),
             arrowprops=dict(arrowstyle='->', lw=2, color=L3_COLOR))

# CDSS box
cdss_box = FancyBboxPatch((1.5, 2.0), 7, 1.5, boxstyle='round,pad=0.15',
                           facecolor=L3_BG, edgecolor=L3_COLOR, lw=2)
ax.add_patch(cdss_box)
ax.text(5, 3.0, 'Clinical Decision Support Engine', ha='center', fontsize=10,
         fontweight='bold', color=L3_COLOR)
ax.text(5, 2.4, '64 Guidelines  -  17 Specialties  -  6 Red-flag Pathways  -  9 Lab Parameters',
         ha='center', fontsize=7, color=L3_COLOR)

# Final output
ax.annotate('', xy=(5, 1.5), xytext=(5, 2.0),
             arrowprops=dict(arrowstyle='->', lw=2, color=L3_COLOR))
final = FancyBboxPatch((2.5, 0.5), 5, 0.8, boxstyle='round,pad=0.1',
                        facecolor='#FEE2E2', edgecolor=L3_COLOR, lw=1.5)
ax.add_patch(final)
ax.text(5, 0.9, 'Triage Recommendation + Department Routing + Guideline Citation',
         ha='center', va='center', fontsize=7, fontweight='bold', color=L3_COLOR)


# ═════════════════════════════════════════════════════════════
# Panel D: Performance Summary Table (as figure)
# ═════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('(D) Performance Summary: 10 Tasks, 67 Models', fontsize=11, fontweight='bold', loc='left')

bs = pd.read_csv(os.path.join(RES, 'tables_v3', 'bootstrap_full_metrics_v3.csv'))

# Group by task, show best AUC
task_summary = []
for task in ['Binary-Screening', 'Binary-External-Pure', 'Binary-External',
             'TBIL-Eyelid', 'Face-Ternary', 'DBIL', 'IBIL', 'Jaundice-Type',
             'Child-Pugh', 'MELD-Risk']:
    sub = bs[bs['task'] == task]
    if len(sub) == 0: continue
    best = sub.loc[sub['auc'].idxmax()]
    layer = 1 if task in ['Binary-Screening', 'Binary-External-Pure', 'Binary-External',
                           'TBIL-Eyelid', 'Face-Ternary', 'DBIL', 'IBIL', 'Jaundice-Type'] else 2
    task_summary.append({'task': task, 'best_model': best['name'], 'auc': best['auc'],
                          'auc_lo': best['auc_lo'], 'auc_hi': best['auc_hi'],
                          'n_val': best['n_val'], 'n_models': len(sub), 'layer': layer})

# Draw table
y_start = 8.5
row_h = 0.7
# Header
ax.text(0.5, y_start + 0.5, 'Task', fontsize=8, fontweight='bold', color=NEUTRAL)
ax.text(4.0, y_start + 0.5, 'Best Model', fontsize=8, fontweight='bold', color=NEUTRAL)
ax.text(7.0, y_start + 0.5, 'AUC [95% CI]', fontsize=8, fontweight='bold', color=NEUTRAL)
ax.text(9.0, y_start + 0.5, 'N', fontsize=8, fontweight='bold', color=NEUTRAL)
ax.plot([0.3, 9.8], [y_start + 0.2, y_start + 0.2], color=NEUTRAL, lw=0.5)

for i, r in enumerate(task_summary):
    y = y_start - (i + 1) * row_h
    color = L1_COLOR if r['layer'] == 1 else L2_COLOR
    # Task name
    task_short = (r['task'].replace('Binary-Screening', 'Bin Screen (Int)')
                  .replace('Binary-External-Pure', 'Bin Screen (Ext)')
                  .replace('Binary-External', 'Bin Screen (DA)')
                  .replace('TBIL-Eyelid', 'TBIL (Eyelid)')
                  .replace('Face-Ternary', 'TBIL (Face)')
                  .replace('Jaundice-Type', 'J Type')
                  .replace('Child-Pugh', 'Child-Pugh')
                  .replace('MELD-Risk', 'MELD Risk'))
    ax.text(0.5, y, task_short, fontsize=7, color=color, fontweight='bold')
    ax.text(4.0, y, r['best_model'][:18], fontsize=6.5, color=NEUTRAL)
    auc_str = f'{r["auc"]:.3f} [{r["auc_lo"]:.3f}-{r["auc_hi"]:.3f}]'
    ax.text(7.0, y, auc_str, fontsize=7, color=color, fontweight='bold')
    ax.text(9.0, y, str(int(r['n_val'])), fontsize=7, color=NEUTRAL)
    # Alternating row background
    if i % 2 == 0:
        ax.add_patch(Rectangle((0.2, y - 0.3), 9.6, row_h - 0.1,
                                facecolor='#F9FAFB', lw=0, zorder=-1))

# Layer legend
ax.text(0.5, 0.5, 'Legend:  ', fontsize=8, fontweight='bold')
l1_patch = mpatches.Patch(color=L1_COLOR, label='Layer 1 (8 tasks)')
l2_patch = mpatches.Patch(color=L2_COLOR, label='Layer 2 (2 tasks)')
ax.legend(handles=[l1_patch, l2_patch], fontsize=7, loc='lower left',
           bbox_to_anchor=(0.05, 0.0))


plt.savefig(os.path.join(FIG_DIR, 'Figure1_system_overview.png'))
plt.savefig(os.path.join(FIG_DIR, 'Figure1_system_overview.svg'))
plt.close(fig)
print(f'Saved: {FIG_DIR}/Figure1_system_overview.png/svg')
