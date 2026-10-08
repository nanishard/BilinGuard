# -*- coding: utf-8 -*-
"""
Generate 7 methodological flowcharts for supplementary materials.

SuppFig S7: Video preprocessing pipeline (detailed)
SuppFig S8: Three-layer data flow architecture
SuppFig S9: Domain adaptation workflow
SuppFig S10: CDSS advise() decision tree
SuppFig S11: Reader study protocol
SuppFig S12: Training procedure
SuppFig S13: Desktop application architecture
"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

BASE = r'D:\research\人脸识别营养\传染科'
FIG_DIR = os.path.join(BASE, 'results', 'figures_v3', 'main')

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42})

C1 = '#1D3557'; C2 = '#D97706'; C3 = '#DC2626'; CG = '#2A9D8F'; CN = '#6B7280'
BG1 = '#E0E7FF'; BG2 = '#FEF3C7'; BG3 = '#FEE2E2'; BGG = '#D1FAE5'


def box(ax, x, y, w, h, text, facecolor='#F3F4F6', edgecolor=CN, fontsize=7, bold=False):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.08',
                        facecolor=facecolor, edgecolor=edgecolor, lw=1.2)
    ax.add_patch(b)
    fw = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=fw)

def arrow(ax, x1, y1, x2, y2, color=CN, lw=1.2, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle=style, lw=lw, color=color))

def diamond(ax, x, y, w, h, text, facecolor=BGG, edgecolor=CG, fontsize=6):
    """Draw a diamond shape using polygon."""
    pts = [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)]
    poly = plt.Polygon(pts, facecolor=facecolor, edgecolor=edgecolor, lw=1.2)
    ax.add_patch(poly)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)


# ═════════════════════════════════════════════════════════════
# SuppFig S7: Video Preprocessing Pipeline (detailed)
# ═════════════════════════════════════════════════════════════
print('[S7] Video Preprocessing Pipeline...')
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('Supplementary Figure S7. Video Preprocessing Pipeline (Detailed)',
             fontsize=12, fontweight='bold', loc='left')

# Input
box(ax, 2, 9, 2.5, 0.8, 'Raw Bedside\nVideo (10-20s)', '#DBEAFE', C1, 8, True)
arrow(ax, 2, 8.5, 2, 7.8)

# Frame sampling
box(ax, 2, 7.3, 2.8, 1.0, '12-Frame Uniform\nSampling\n(equidistant temporal)', BGG, CG, 7)
ax.text(4.0, 7.3, 'np.linspace(0, N-1, 12)\nCV2 CAP_PROP_POS_FRAMES', fontsize=5, style='italic', color=CN, va='center')
arrow(ax, 2, 6.7, 2, 6.0)

# Face detection
box(ax, 2, 5.5, 2.8, 1.0, 'Haar Cascade\nFace Detection\n(largest bbox)', '#F3F4F6', CN, 7)
ax.text(4.0, 5.5, 'haarcascade_\nfrontalface_default.xml', fontsize=5, style='italic', color=CN, va='center')
arrow(ax, 2, 4.9, 2, 4.2)

# Branch: Face vs Eyelid
box(ax, 5, 4.5, 3, 0.8, 'SAM Face Segmentation\n(ViT-H, 1B masks pre-trained)', BGG, CG, 8, True)
box(ax, 9.5, 4.5, 3, 0.8, 'Sclera Extraction\n(HSV thresholding)', BG2, C2, 7)
arrow(ax, 2.5, 4.5, 3.5, 4.5)
arrow(ax, 6.5, 4.5, 8.0, 4.5)

# SAM detail
box(ax, 5, 3.2, 3.5, 1.0, 'Tight crop (5px padding)\nOriginal pixels preserved\nNo fill/interpolation', BG1, C1, 6)
arrow(ax, 5, 4.0, 5, 3.8)
box(ax, 9.5, 3.2, 3.5, 1.0, 'HSV: S<60 & V>70 (white)\nH 20-60, S 10-60, V 70-100\nMorphological open (3x3)', BG2, C2, 6)
arrow(ax, 9.5, 4.0, 9.5, 3.8)

# CLAHE
box(ax, 7, 2.0, 4, 0.8, 'CLAHE (L channel)\nclipLimit=3.0, tileGridSize=8x8', '#F3F4F6', CN, 7, True)
arrow(ax, 5, 2.6, 6.5, 2.4)
arrow(ax, 9.5, 2.6, 7.5, 2.4)

# Domain adaptation branch
box(ax, 12, 2.0, 2.5, 0.8, 'Grey-World\n(domain adapt only)', BGG, CG, 6)
ax.annotate('only for\ndomain-adapted models', xy=(10, 2.0), xytext=(10.8, 2.0),
             fontsize=5, color=CG, va='center')

# Output
box(ax, 7, 0.8, 4, 0.8, '224x224 RGB Tensor\n(ImageNet normalised)', BG1, C1, 8, True)
arrow(ax, 7, 1.5, 7, 1.2)

# QC annotations
ax.text(0.3, 9.5, 'INPUT', fontsize=8, fontweight='bold', color=C1)
ax.text(0.3, 0.5, 'OUTPUT', fontsize=8, fontweight='bold', color=C1)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS7_preprocessing_pipeline.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS7_preprocessing_pipeline.svg'))
plt.close(fig)


# ═════════════════════════════════════════════════════════════
# SuppFig S8: Three-Layer Data Flow
# ═════════════════════════════════════════════════════════════
print('[S8] Three-Layer Data Flow...')
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('Supplementary Figure S8. Three-Layer Closed-Loop Data Flow Architecture',
             fontsize=12, fontweight='bold', loc='left')

# Input
box(ax, 7, 9.3, 4, 0.8, 'Facial Video + Optional Eyelid Photo\n+ Lab Values + Diagnosis + Department',
    '#F3F4F6', CN, 8, True)

# Preprocessing
arrow(ax, 7, 8.8, 7, 8.2)
box(ax, 7, 7.8, 5, 0.7, 'Preprocessing: SAM + CLAHE (+ Grey-World)', BGG, CG, 8)
arrow(ax, 7, 7.4, 7, 6.8)

# Layer 1 (8 parallel tasks)
l1_y = 5.5
l1_tasks = ['Screen\n(2-class)', 'TBIL\n(3-class)', 'DBIL\n(3-class)', 'IBIL\n(3-class)', 'Type\n(2-class)']
for i, t in enumerate(l1_tasks):
    x = 1.5 + i * 2.5
    box(ax, x, l1_y, 2.0, 1.0, t + '\nConvNeXt/ViT/\nSwin/EffNet', BG1, C1, 6)
    arrow(ax, 7, 6.8, x, 6.0)

# Layer 2 (2 tasks)
arrow(ax, 4, 4.8, 4, 4.2)
arrow(ax, 9, 4.8, 9, 4.2)
box(ax, 4, 3.7, 3.5, 1.0, 'Layer 2: Child-Pugh\nA/B/C (3-class)\nAUC = 0.907', BG2, C2, 7, True)
box(ax, 9, 3.7, 3.5, 1.0, 'Layer 2: MELD Risk\nLow/Med/High (3-class)\nAUC = 0.927', BG2, C2, 7, True)

# Aggregation
arrow(ax, 4, 3.1, 5.5, 2.5)
arrow(ax, 9, 3.1, 7.5, 2.5)
for i in range(5):
    arrow(ax, 1.5 + i*2.5, 4.8, 7, 2.5)

# CDSS
box(ax, 7, 2.0, 6, 0.9, 'Layer 3: Clinical Decision Support Engine\n64 Guidelines | 17 Specialties | 6 Pathways | 9 Lab Params',
    BG3, C3, 8, True)
arrow(ax, 7, 2.5, 7, 2.5)

# Output
arrow(ax, 7, 1.5, 7, 1.0)
box(ax, 7, 0.5, 6, 0.8, 'Triage + Severity + Type + Disease + Pathway + Guideline Citation\n(8 output sections, bilingual)',
    '#F3F4F6', CN, 7, True)

# Feedback loop
ax.annotate('', xy=(0.5, 9.3), xytext=(0.5, 0.5),
             arrowprops=dict(arrowstyle='->', lw=2, color=C3, ls='--',
                              connectionstyle='arc3,rad=-0.2'))
ax.text(0.15, 5.0, 'C\nL\nO\nS\nE\nD\n\nL\nO\nO\nP', fontsize=7, fontweight='bold',
         color=C3, ha='center', va='center')

# Layer labels
ax.text(13.5, 5.5, 'Layer 1\nRecognition', fontsize=8, fontweight='bold', color=C1, ha='center', rotation=90)
ax.text(13.5, 3.7, 'Layer 2\nScoring', fontsize=8, fontweight='bold', color=C2, ha='center', rotation=90)
ax.text(13.5, 2.0, 'Layer 3\nDecision', fontsize=8, fontweight='bold', color=C3, ha='center', rotation=90)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS8_dataflow.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS8_dataflow.svg'))
plt.close(fig)


# ═════════════════════════════════════════════════════════════
# SuppFig S9: Domain Adaptation Workflow
# ═════════════════════════════════════════════════════════════
print('[S9] Domain Adaptation Workflow...')
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('Supplementary Figure S9. Domain Adaptation Workflow for External Validation',
             fontsize=12, fontweight='bold', loc='left')

# Two data sources
box(ax, 3, 7, 3.5, 0.9, 'Internal Data\nWCH (n=939)\n579 with images', BG1, C1, 7, True)
box(ax, 10, 7, 3.5, 0.9, 'External Data\nMulti-centre (n=117)\n59N + 58J', BG3, C3, 7, True)

# Grey-World on both
arrow(ax, 3, 6.5, 3, 6.0)
arrow(ax, 10, 6.5, 10, 6.0)
box(ax, 3, 5.5, 3, 0.8, 'Grey-World\nNormalisation', BGG, CG, 7)
box(ax, 10, 5.5, 3, 0.8, 'Grey-World\nNormalisation', BGG, CG, 7)

# CLAHE
arrow(ax, 3, 5.0, 3, 4.5)
arrow(ax, 10, 5.0, 10, 4.5)
box(ax, 3, 4.0, 3, 0.8, 'CLAHE\n(L channel)', '#F3F4F6', CN, 7)
box(ax, 10, 4.0, 3, 0.8, 'CLAHE\n(L channel)', '#F3F4F6', CN, 7)

# Split external
arrow(ax, 10, 3.5, 8, 3.0)
arrow(ax, 10, 3.5, 12, 3.0)
box(ax, 8, 2.5, 2.5, 0.8, 'Ext Train (80%)\n~94 patients', BG2, C2, 6)
box(ax, 12, 2.5, 2.5, 0.8, 'Ext Test (20%)\n~23 patients\n(HELD OUT)', BG3, C3, 6, True)

# Joint training
arrow(ax, 3, 3.5, 5.5, 2.5)
arrow(ax, 8, 2.0, 5.5, 2.0)
box(ax, 5.5, 1.5, 4, 0.9, 'Joint Training\n(Internal + Ext-Train)\nFocal Loss + EMA + CosineAnnealing', BG1, C1, 7, True)

# Augmentation note
ax.text(5.5, 0.5, 'ColorJitter(0.4) + RandomGrayscale(0.1) + RandomErasing(0.2)\nWeightedRandomSampler + clip_grad_norm(1.0)',
         fontsize=5, ha='center', color=CN, style='italic')

# Final eval
arrow(ax, 12, 1.8, 12, 1.2)
box(ax, 12, 0.6, 3, 0.8, 'Evaluate on\nExt Test Set\nAUC = 1.000', BGG, CG, 7, True)

# Domain shift note
ax.text(0.5, 2.0, 'WITHOUT adaptation:\nExt AUC = 0.885 but\nSensitivity = 0 (collapsed)',
         fontsize=6, color=C3, ha='left', va='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#FEE2E2', alpha=0.9))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS9_domain_adaptation_workflow.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS9_domain_adaptation_workflow.svg'))
plt.close(fig)


# ═════════════════════════════════════════════════════════════
# SuppFig S10: CDSS advise() Decision Tree
# ═════════════════════════════════════════════════════════════
print('[S10] CDSS Decision Tree...')
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis('off')
ax.set_title('Supplementary Figure S10. CDSS advise() Decision Flow',
             fontsize=12, fontweight='bold', loc='left')

# Input
box(ax, 7, 11.3, 5, 0.7, 'Inputs: AI Results + Labs + Department + Diagnosis', '#F3F4F6', CN, 8, True)

# Step 1: Department
arrow(ax, 7, 10.8, 7, 10.3)
diamond(ax, 7, 9.8, 3, 1.0, 'Step 1:\nResolve Department', BG2, C2, 7)
ax.text(10, 9.8, 'Explicit > Disease default\n> Type default > None', fontsize=5, color=CN, va='center')

# Step 2: Severity
arrow(ax, 7, 9.2, 7, 8.7)
diamond(ax, 7, 8.2, 3, 1.0, 'Step 2:\nAssess Severity\n(0-3)', BG2, C2, 7)
ax.text(10, 8.2, 'INR>1.5 -> 3\nTBIL: >171->3, >85->2\nCP-C->3 floor, MELD-H->3 floor', fontsize=5, color=CN, va='center')

# Step 3: Type
arrow(ax, 7, 7.6, 7, 7.1)
diamond(ax, 7, 6.6, 3, 1.0, 'Step 3:\nDetermine Type', BG2, C2, 7)
ax.text(10, 6.6, 'AI classification >\nDBIL/TBIL > 0.5 -> chol\nIBIL/TBIL > 0.8 -> hemolytic', fontsize=5, color=CN, va='center')

# Step 4: Disease
arrow(ax, 7, 6.0, 7, 5.5)
diamond(ax, 7, 5.0, 3, 1.0, 'Step 4:\nInfer Disease\n(keyword match)', BG2, C2, 7)
ax.text(10, 5.0, '17 diseases, priority sort\nACLF(103)>ALF(100)>\ncholangitis(95)>...', fontsize=5, color=CN, va='center')

# Step 5: Pathway
arrow(ax, 7, 4.4, 7, 3.9)
diamond(ax, 7, 3.4, 3, 1.0, 'Step 5:\nCheck Red-flag\nPathway', BG3, C3, 7)
ax.text(10, 3.4, 'Disease-linked >\nLab-triggered (INR, ALP) >\nSeverity-triggered', fontsize=5, color=CN, va='center')

# Step 6: Assemble
arrow(ax, 7, 2.8, 7, 2.3)
box(ax, 7, 1.8, 4, 0.8, 'Step 6: Assemble Output\n(up to 8 sections)', BG1, C1, 7, True)

# Step 7: Output
arrow(ax, 7, 1.3, 7, 0.8)
box(ax, 7, 0.4, 6, 0.6, 'Output: Severity | Type | Disease | CP/MELD | Dept | Alert | Labs | Triage',
    BGG, CG, 7, True)

# Pathway branches
pathways = ['ALF\n(INR+HE)', 'ACLF\n(decomp)', 'Cholangitis\n(Charcot)', 'Obstruction\n(DBIL+ALP)', 'Hemolysis\n(IBIL)', 'DILI\n(Hy law)']
for i, p in enumerate(pathways):
    x = 1.5 + i * 2.2
    box(ax, x, 2.5, 1.8, 0.7, p, BG3, C3, 5)
    arrow(ax, 5.5, 3.4, x, 2.9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS10_cdss_decision_tree.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS10_cdss_decision_tree.svg'))
plt.close(fig)


# ═════════════════════════════════════════════════════════════
# SuppFig S11: Reader Study Protocol
# ═════════════════════════════════════════════════════════════
print('[S11] Reader Study Protocol...')
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('Supplementary Figure S11. Reader Study Protocol',
             fontsize=12, fontweight='bold', loc='left')

# Raters
box(ax, 3, 7, 4, 1.2, '8 Raters\n2 Attending | 1 Nurse\n3 Residents | 2 PubHealth', BG1, C1, 7, True)
box(ax, 10, 7, 4, 1.2, '100 Cases\n25 per class x 4\n38 Fitzpatrick IV-VI', BG2, C2, 7, True)

# Blinding
arrow(ax, 3, 6.3, 3, 5.7)
arrow(ax, 10, 6.3, 10, 5.7)
box(ax, 6.5, 5.2, 8, 0.8, 'Web-based Blinded Review\n(no clinical metadata, no AI predictions)', BG3, C3, 8, True)
arrow(ax, 3, 5.7, 5, 5.4)
arrow(ax, 10, 5.7, 8, 5.4)

# Session structure
arrow(ax, 6.5, 4.7, 6.5, 4.2)
box(ax, 3.5, 3.7, 4, 0.8, 'Session 1\n(100 cases)', '#F3F4F6', CN, 7)
box(ax, 9.5, 3.7, 4, 0.8, 'Session 2\n(100 cases,\n14-day washout)', '#F3F4F6', CN, 7)
arrow(ax, 5.5, 3.7, 7.5, 3.7)

# Outputs
arrow(ax, 6.5, 3.2, 6.5, 2.7)
box(ax, 6.5, 2.2, 8, 0.8, 'Per-case: 4-class label + Response time\nPer-rater: AUC | F1 | Sens | Spec | Acc', BGG, CG, 7, True)

# Statistics
arrow(ax, 6.5, 1.7, 6.5, 1.2)
box(ax, 6.5, 0.6, 10, 0.9, 'Statistics: Fleiss kappa (inter-rater) | Cohen kappa (model-rater) |\nWilcoxon signed-rank (model vs each rater) | Kruskal-Wallis (time)',
    '#F3F4F6', CN, 7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS11_reader_study_protocol.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS11_reader_study_protocol.svg'))
plt.close(fig)


# ═════════════════════════════════════════════════════════════
# SuppFig S12: Training Procedure
# ═════════════════════════════════════════════════════════════
print('[S12] Training Procedure...')
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('Supplementary Figure S12. Training Procedure (Per Backbone)',
             fontsize=12, fontweight='bold', loc='left')

# Data
box(ax, 2, 7, 3, 0.8, 'Training Data\n(patient-level 80%)', BG1, C1, 7, True)
arrow(ax, 2, 6.5, 2, 6.0)

# Augmentation
box(ax, 2, 5.5, 3.5, 0.9, 'Augmentation\nFlip + ColorJitter + Rotation\n+ Grayscale + Erasing', BGG, CG, 6)
arrow(ax, 2, 5.0, 2, 4.5)

# Sampler
box(ax, 2, 4.0, 3.5, 0.9, 'WeightedRandomSampler\n(inverse class frequency)', BGG, CG, 6)
arrow(ax, 2, 3.5, 2, 3.0)

# Model init
box(ax, 5.5, 4.0, 3, 0.9, 'Backbone\n(ImageNet pretrained)\nConvNeXt/ViT/Swin/Eff', BG1, C1, 6)
arrow(ax, 3.5, 4.0, 4.0, 4.0)

# Loss + Optimizer
box(ax, 5.5, 2.5, 3.5, 1.0, 'Focal Loss (gamma=2.0)\n+ Class weights + LS(0.1)\nAdamW lr=1e-4', BG2, C2, 6)
arrow(ax, 5.5, 3.5, 5.5, 3.0)

# Scheduler
box(ax, 9.5, 4.0, 3, 0.9, 'CosineAnnealingLR\n(or OneCycleLR)', '#F3F4F6', CN, 6)
arrow(ax, 7.0, 4.0, 8.0, 4.0)

# Training loop
box(ax, 5.5, 1.0, 4, 1.0, 'Training Loop (50-60 epochs)\n1. Forward pass\n2. Focal Loss\n3. Backward + clip_grad(1.0)\n4. EMA update (0.999)\n5. Scheduler step', BG3, C3, 6, True)
arrow(ax, 5.5, 2.0, 5.5, 1.5)
arrow(ax, 2, 2.5, 3.5, 1.2)

# Validation
box(ax, 11, 1.0, 3.5, 1.0, 'Validation (20%)\nPatient-level AUC\nBest epoch saved', BGG, CG, 6, True)
arrow(ax, 7.5, 1.0, 9.3, 1.0)

# QC annotations
ax.text(13, 7, 'Key Regularisation:\n- Label Smoothing (0.1)\n- Gradient Clipping (norm 1.0)\n- EMA (decay 0.999)\n- Weight Decay (1e-4)',
         fontsize=6, color=CN, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='#F9FAFB', alpha=0.9))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS12_training_procedure.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS12_training_procedure.svg'))
plt.close(fig)


# ═════════════════════════════════════════════════════════════
# SuppFig S13: Desktop Application Architecture
# ═════════════════════════════════════════════════════════════
print('[S13] Desktop Application Architecture...')
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('Supplementary Figure S13. Desktop Application Architecture (v4.0)',
             fontsize=12, fontweight='bold', loc='left')

# UI Layer
box(ax, 7, 7, 8, 0.8, 'PyQt5 Desktop GUI (v4.0)\nSoft UI | WCAG AA+ | Bilingual | Light/Dark | Font 10-22pt',
    BG1, C1, 7, True)

# Three threads
arrow(ax, 4, 6.5, 4, 6.0)
arrow(ax, 7, 6.5, 7, 6.0)
arrow(ax, 10, 6.5, 10, 6.0)
box(ax, 3, 5.5, 3, 0.8, 'ModelLoader\n(QThread)\n~29 models', BGG, CG, 6)
box(ax, 6.5, 5.5, 3, 0.8, 'Analysis Thread\n(Python Thread)\n7 tasks parallel', BGG, CG, 6)
box(ax, 10, 5.5, 3, 0.8, 'GradcamWorker\n(QThread)\nFace + Eyelid', BGG, CG, 6)

# Inference engine
arrow(ax, 6.5, 5.0, 6.5, 4.5)
box(ax, 6.5, 4.0, 8, 0.9, 'BilinGuardPredictor (966 lines)\nFrame extraction -> SAM -> CLAHE -> Backbones -> Patient-level aggregation',
    BG2, C2, 7, True)

# CDSS
arrow(ax, 6.5, 3.4, 6.5, 2.9)
box(ax, 6.5, 2.4, 8, 0.9, 'ClinicalAdvisor (2,448 lines)\nadvise(results, labs, dept, dx) -> 8 sections + guidelines',
    BG3, C3, 7, True)

# Output sections
arrow(ax, 6.5, 1.8, 6.5, 1.3)
sections = ['Severity', 'Type', 'Aetiology', 'CP/MELD', 'Dept', 'Alert', 'Labs', 'Triage']
for i, s in enumerate(sections):
    x = 1.5 + i * 1.6
    box(ax, x, 0.7, 1.4, 0.5, s, '#F3F4F6', CN, 5)

# Interpretability side
ax.text(13, 4.0, 'Interpretability:\n- Grad-CAM (3 panels)\n- CIELAB colourimetry\n- Model agreement\n- Entropy + Margin',
         fontsize=6, color=CN, va='center', ha='left',
         bbox=dict(boxstyle='round', facecolor='#F9FAFB', alpha=0.9))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS13_desktop_architecture.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFigS13_desktop_architecture.svg'))
plt.close(fig)

print('\nAll 7 methodological flowcharts generated.')
