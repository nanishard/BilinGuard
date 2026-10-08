# -*- coding: utf-8 -*-
"""
SuppFig S7-S13: Lancet/Nature Medicine style flow diagrams.
Regenerate all supplementary design figures as clean, professional method diagrams.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_lancet import set_rc
set_rc()

BASE = r'D:\research\人脸识别营养\传染科'
FIG_DST = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'manuscript', 'supplementary_figures')

# Color palette: low-saturation deep blue / light blue / grey only
DARK_BLUE = '#1a3a5c'
MID_BLUE = '#3b6b9e'
LIGHT_BLUE = '#7ba7c9'
PALE_BLUE = '#c5d8e8'
GREY = '#888888'
LIGHT_GREY = '#d0d0d0'
WHITE = '#ffffff'

def rbox(ax, x, y, w, h, text, fc=WHITE, ec=DARK_BLUE, fontsize=6.5, lw=0.8, text_color=DARK_BLUE, bold=False):
    """Draw a rounded box with text centered."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", 
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, linespacing=1.3, wrap=True)

def arrow(ax, x1, y1, x2, y2, color=DARK_BLUE, style='-', lw=0.7):
    """Draw a thin arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=style))

def module_title(ax, x, y, text, w=1.8):
    ax.text(x + w/2, y, text, ha='center', va='center', fontsize=7.5, 
            color=DARK_BLUE, weight='bold')

# ═══ SuppFig S7: Preprocessing pipeline (v4) ═══
def gen_s7():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Horizontal flow: Video -> Sampling -> Detection -> Segmentation -> Enhancement -> Output
    steps = [
        (0.3, 'Raw\nfacial video\n10-20 s'),
        (2.8, 'Uniform\nframe sampling\n12 frames'),
        (5.3, 'YuNet\ndetection\n(score >= 0.5)'),
        (7.8, 'SAM face\nsegmentation\n+ skin gate\n(>= 0.35)'),
        (10.3, 'Landmark\nalignment\n+ CLAHE'),
        (12.8, '224x224\nface crop\n(patient-level\nmean-softmax)'),
    ]
    for i, (x, text) in enumerate(steps):
        color = DARK_BLUE if i < 2 else (MID_BLUE if i < 4 else LIGHT_BLUE)
        rbox(ax, x, 2.5, 1.8, 1.8, text, fc=WHITE, ec=color, fontsize=6, lw=1.0)
        if i < len(steps) - 1:
            arrow(ax, x + 1.9, 3.4, steps[i+1][0] - 0.1, 3.4)
    
    # Bottom note
    ax.text(7.5, 1.0, 'Acceptance rate: 98.2% (2,649/2,698 frames kept across 228 patients)',
            ha='center', va='center', fontsize=6, color=GREY, style='italic')
    
    fig.suptitle('Supplementary Figure S7. Video Preprocessing Pipeline (v4: YuNet + SAM)',
                 fontsize=8, fontweight='bold', x=0.02, ha='left', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DST, f'SuppFigS7_preprocessing_pipeline.{ext}'))
    plt.close(fig)
    print('S7 saved')

# ═══ SuppFig S8: Dataflow architecture ═══
def gen_s8():
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9.5)
    ax.axis('off')
    
    # 5 modules left-to-right
    module_w = 2.5; gap = 0.5; start_x = 0.3
    # Background boxes must cover ALL inner content: from y=4.2 to y=8.8
    bg_y = 4.0; bg_h = 5.0
    
    modules = [
        ('Cohort', start_x + 0*(module_w+gap)),
        ('Preprocessing', start_x + 1*(module_w+gap)),
        ('Model\nEvaluation', start_x + 2*(module_w+gap)),
        ('Clinical\nTasks', start_x + 3*(module_w+gap)),
        ('CDSS', start_x + 4*(module_w+gap)),
    ]
    
    # Draw background boxes FIRST (so inner boxes sit on top)
    for title, mx in modules:
        bg = FancyBboxPatch((mx, bg_y), module_w, bg_h,
                           boxstyle="round,pad=0.08", facecolor=PALE_BLUE, edgecolor=LIGHT_BLUE, linewidth=0.6, alpha=0.25)
        ax.add_patch(bg)
        module_title(ax, mx, bg_y + bg_h + 0.15, title, w=module_w)
    
    # Inner content boxes (all within bg_y to bg_y+bg_h = 4.0 to 9.0)
    box_x_offset = 0.15; inner_w = module_w - 0.3
    
    # Module 1: Cohort (3 rows)
    rbox(ax, modules[0][1]+box_x_offset, 7.6, inner_w, 0.8, '1,206 screened', fontsize=5.5)
    rbox(ax, modules[0][1]+box_x_offset, 6.5, inner_w, 0.9, '939 enrolled\n625 controls\n314 jaundiced', fontsize=5)
    rbox(ax, modules[0][1]+box_x_offset, 5.2, inner_w, 0.9, '117 external\ncohort', fontsize=5, fc=WHITE, ec='#c2694f', text_color='#c2694f')
    rbox(ax, modules[0][1]+box_x_offset, 4.2, inner_w, 0.7, '579 usable\nimages', fontsize=5)
    
    # Module 2: Preprocessing (3 rows)
    rbox(ax, modules[1][1]+box_x_offset, 7.6, inner_w, 0.8, '10-20 s video\n+ eyelid photo', fontsize=5)
    rbox(ax, modules[1][1]+box_x_offset, 6.4, inner_w, 1.0, 'YuNet + SAM\nCLAHE\n12-frame agg.', fontsize=5)
    rbox(ax, modules[1][1]+box_x_offset, 5.2, inner_w, 0.9, '224x224\ncrops', fontsize=5)
    rbox(ax, modules[1][1]+box_x_offset, 4.2, inner_w, 0.7, '98.2%\naccept', fontsize=5)
    
    # Module 3: Evaluation (3 rows)
    rbox(ax, modules[2][1]+box_x_offset, 7.6, inner_w, 0.8, '4 backbones\n+ ensemble', fontsize=5)
    rbox(ax, modules[2][1]+box_x_offset, 6.4, inner_w, 1.0, '3 configs:\nface / eyelid /\nfusion', fontsize=5)
    rbox(ax, modules[2][1]+box_x_offset, 5.2, inner_w, 0.9, '105-cell\nmatrix', fontsize=5, fc=LIGHT_BLUE, ec=DARK_BLUE)
    rbox(ax, modules[2][1]+box_x_offset, 4.2, inner_w, 0.7, 'bootstrap\nCI', fontsize=5)
    
    # Module 4: Tasks (4 small rows)
    rbox(ax, modules[3][1]+box_x_offset, 7.8, inner_w, 0.6, 'Binary screening', fontsize=4.5)
    rbox(ax, modules[3][1]+box_x_offset, 7.0, inner_w, 0.6, 'TBIL/DBIL/IBIL', fontsize=4.5)
    rbox(ax, modules[3][1]+box_x_offset, 6.2, inner_w, 0.6, 'Jaundice type', fontsize=4.5)
    rbox(ax, modules[3][1]+box_x_offset, 5.4, inner_w, 0.6, 'CP/MELD\n(exploratory)', fontsize=4.5, ec=GREY, text_color=GREY)
    rbox(ax, modules[3][1]+box_x_offset, 4.4, inner_w, 0.7, '105 cells\nevaluated', fontsize=5)
    
    # Module 5: CDSS (3 rows)
    rbox(ax, modules[4][1]+box_x_offset, 7.6, inner_w, 0.8, 'Safety gate\n6 red-flag paths', fontsize=5)
    rbox(ax, modules[4][1]+box_x_offset, 6.4, inner_w, 1.0, 'Dept routing\nUrgency\nGuideline refs', fontsize=5)
    rbox(ax, modules[4][1]+box_x_offset, 5.2, inner_w, 0.9, '51 rules\n64 guidelines', fontsize=5)
    rbox(ax, modules[4][1]+box_x_offset, 4.2, inner_w, 0.7, '150-scenario\ntest', fontsize=5, ec='#c2694f', text_color='#c2694f')
    
    # Arrows between modules
    for i in range(4):
        x1 = modules[i][1] + module_w
        x2 = modules[i+1][1]
        arrow(ax, x1, 6.3, x2, 6.3, lw=1.0)
    
    fig.suptitle('Supplementary Figure S8. Study Design and Data Flow Architecture',
                 fontsize=8, fontweight='bold', x=0.02, ha='left', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DST, f'SuppFigS8_dataflow.{ext}'))
    plt.close(fig)
    print('S8 saved')

# ═══ SuppFig S9: Domain adaptation workflow ═══
def gen_s9():
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6)
    ax.axis('off')
    
    steps = [
        (0.3, 'Internal\ntraining set\n(n=449 train)'),
        (3.3, 'Internal\nvalidation\n(n=112 val,\nseed 42)'),
        (6.3, 'External\ncohort\n(n=117,\n3 centres)'),
        (9.3, 'v4-lite\npreprocessing\n(YuNet bbox\ncrop only)'),
        (12.3, 'Swin model\nfrozen\nevaluation\n(exploratory)'),
    ]
    for i, (x, text) in enumerate(steps):
        ec = DARK_BLUE if i < 2 else ('#c2694f' if i == 2 else MID_BLUE)
        rbox(ax, x, 2.5, 2.4, 2.0, text, ec=ec, fontsize=5.5, lw=1.0)
        if i < len(steps) - 1:
            arrow(ax, x + 2.5, 3.5, steps[i+1][0] - 0.1, 3.5)
    
    # Note
    ax.text(7.5, 0.8, 'Note: v4-lite was selected after comparing 6 preprocessing configurations\n'
            'on the same external cohort; result is exploratory, not independent validation.',
            ha='center', va='center', fontsize=5.5, color='#c2694f', style='italic')
    
    fig.suptitle('Supplementary Figure S9. External Cohort Evaluation Workflow',
                 fontsize=8, fontweight='bold', x=0.02, ha='left', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DST, f'SuppFigS9_domain_adaptation_workflow.{ext}'))
    plt.close(fig)
    print('S9 saved')

# ═══ SuppFig S10: CDSS decision flow ═══
def gen_s10():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.5)
    ax.axis('off')
    
    # Vertical flow - 7 steps
    step_data = [
        (5.0, 11.0, 'Input: AI outputs + optional labs + diagnosis', DARK_BLUE, WHITE),
        (5.0, 9.7, 'Input validation\n(range checks, missing data policy)', DARK_BLUE, WHITE),
        (5.0, 8.3, 'Safety gate\n(INR > 1.5, extreme values)', '#c2694f', '#fdf0ed'),
        (5.0, 6.8, 'Red-flag check\n6 pathways: ALF / ACLF / cholangitis /\nobstruction / hemolysis / DILI', '#c2694f', '#fdf0ed'),
        (5.0, 5.2, 'Severity determination\n(lab values > AI > default)', MID_BLUE, WHITE),
        (5.0, 3.7, 'Jaundice type + disease inference\n+ department routing', MID_BLUE, WHITE),
        (5.0, 2.2, 'Output: recommendation + urgency\n+ guideline citations + provenance', DARK_BLUE, WHITE),
    ]
    
    for i, (cx, cy, text, ec, fc) in enumerate(step_data):
        w = 7.5; h = 1.1
        rbox(ax, cx - w/2, cy - h/2, w, h, text, ec=ec, fontsize=5.5, lw=0.9, fc=fc)
        if i < len(step_data) - 1:
            next_cy = step_data[i+1][1]
            next_h = 1.1
            arrow(ax, cx, cy - h/2, cx, next_cy + next_h/2 - 0.05)
    
    # Clean side annotation for safety override - simplified
    ax.text(9.0, 8.3, 'Priority:\n1. lab values\n2. clinical dx\n3. AI prediction', ha='center', va='center',
            fontsize=5, color='#c2694f',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#c2694f', lw=0.5))
    
    fig.suptitle('Supplementary Figure S10. CDSS Decision Flow\n'
                 '(7 sequential steps; safety gate overrides all downstream logic)',
                 fontsize=8, fontweight='bold', x=0.02, ha='left', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DST, f'SuppFigS10_cdss_decision_tree.{ext}'))
    plt.close(fig)
    print('S10 saved')

# ═══ SuppFig S11: Reader study protocol ═══
def gen_s11():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7)
    ax.axis('off')
    
    # 4 stages left to right
    stages = [
        (0.3, 'Case selection\n100 cases\nstratified by\nbilirubin + skin type'),
        (4.0, '8 raters\n2 attendings\n1 nurse\n3 residents\n2 public health'),
        (7.7, 'Blinded review\n4-class grade\n(normal/mild/\nmod/severe)\nweb-based'),
        (11.4, 'Analysis\nBinary: >=mild\nTernary: mild/mod/sev\namong jaundiced\n(n=75)'),
    ]
    for i, (x, text) in enumerate(stages):
        rbox(ax, x, 2.0, 3.2, 3.0, text, ec=DARK_BLUE, fontsize=6, lw=0.8,
             fc=[WHITE, PALE_BLUE, WHITE, LIGHT_BLUE][i])
        if i < 3:
            arrow(ax, x + 3.3, 3.5, stages[i+1][0] - 0.1, 3.5)
    
    ax.text(7.5, 0.8, 'Inter-rater: Fleiss kappa = 0.691 (substantial)\n'
            'Model exceeded all 8 raters on binary (0.993 vs 0.914) and ternary (0.798 vs 0.656)',
            ha='center', fontsize=5.5, color=GREY, style='italic')
    
    fig.suptitle('Supplementary Figure S11. Reader Study Protocol',
                 fontsize=8, fontweight='bold', x=0.02, ha='left', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DST, f'SuppFigS11_reader_study_protocol.{ext}'))
    plt.close(fig)
    print('S11 saved')

# ═══ SuppFig S12: Training procedure ═══
def gen_s12():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7)
    ax.axis('off')
    
    steps = [
        (0.3, 'Patient-level\nsplit\n80/20, seed 42\nstratified'),
        (3.3, 'Augmentation\nflip, color jitter\nrotation'),
        (6.3, 'Training\nFocal Loss\nWeightedSampler\nEMA (0.999)'),
        (9.3, 'Selection\nSwin-Tiny\n(selected primary)'),
        (12.3, 'Evaluation\n<=12 imgs/patient\n1000 bootstrap\nYouden threshold'),
    ]
    for i, (x, text) in enumerate(steps):
        rbox(ax, x, 2.5, 2.6, 2.0, text, ec=DARK_BLUE, fontsize=5.5, lw=0.8)
        if i < 4:
            arrow(ax, x + 2.7, 3.5, steps[i+1][0] - 0.1, 3.5)
    
    ax.text(7.5, 0.8, 'Backbones: ConvNeXt-Tiny | ViT-Tiny/16 | Swin-Tiny | EfficientNet-B0 (ImageNet pretrained)',
            ha='center', fontsize=5.5, color=GREY, style='italic')
    
    fig.suptitle('Supplementary Figure S12. Training and Evaluation Procedure (canonical-v2 protocol)',
                 fontsize=8, fontweight='bold', x=0.02, ha='left', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DST, f'SuppFigS12_training_procedure.{ext}'))
    plt.close(fig)
    print('S12 saved')

# ═══ SuppFig S13: Desktop application architecture ═══
def gen_s13():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7)
    ax.axis('off')
    
    # 3 layers
    rbox(ax, 0.5, 4.5, 4.0, 1.5, 'User Interface (PyQt5)\nVideo/image input\nLab values (optional)\nResults display', ec=DARK_BLUE, fontsize=6)
    rbox(ax, 5.5, 4.5, 4.0, 1.5, 'Inference Engine\nYuNet + SAM detect\n4 backbones + ensemble\n<=12 frame aggregation', ec=MID_BLUE, fontsize=6)
    rbox(ax, 10.5, 4.5, 4.0, 1.5, 'CDSS Module\nadvise() function\n150-scenario tested\nGuideline-referenced', ec='#4e9a8f', fontsize=6)
    
    arrow(ax, 4.6, 5.25, 5.4, 5.25)
    arrow(ax, 9.6, 5.25, 10.4, 5.25)
    
    # Outputs below
    rbox(ax, 2.0, 1.5, 3.0, 1.5, '7 task results\nSeverity gauge\nType badge\nGrad-CAM overlay', ec=LIGHT_BLUE, fontsize=5.5)
    rbox(ax, 6.0, 1.5, 3.0, 1.5, 'Per-model\nuncertainty\n(entropy, margin,\nagreement)', ec=LIGHT_BLUE, fontsize=5.5)
    rbox(ax, 10.0, 1.5, 3.0, 1.5, 'CDSS output\n8 sections (EN/CN)\nUrgency + dept\nGuideline citations', ec=LIGHT_BLUE, fontsize=5.5)
    
    arrow(ax, 2.5, 4.4, 3.5, 3.1, style='--')
    arrow(ax, 7.5, 4.4, 7.5, 3.1, style='--')
    arrow(ax, 12.5, 4.4, 11.5, 3.1, style='--')
    
    fig.suptitle('Supplementary Figure S13. Desktop Application Architecture (v4.0)',
                 fontsize=8, fontweight='bold', x=0.02, ha='left', y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DST, f'SuppFigS13_desktop_architecture.{ext}'))
    plt.close(fig)
    print('S13 saved')

# Generate all
for func in [gen_s7, gen_s8, gen_s9, gen_s10, gen_s11, gen_s12, gen_s13]:
    func()

# Sync to 20260725
import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260725', '04_Supplementary', 'figures')
for f in os.listdir(FIG_DST):
    if f.endswith('.png') or f.endswith('.svg'):
        shutil.copy2(os.path.join(FIG_DST, f), os.path.join(dst, f))
print('\nAll synced to 20260725')
