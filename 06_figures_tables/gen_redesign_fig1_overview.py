# -*- coding: utf-8 -*-
"""
Redesigned Figure 1 v2 (CRM/3D-FAICE style): full-width schematic, Left/Right,
no panel letters.  v2 fixes: separated title/detail text rows (no overlap),
cohort band moved to a clear bottom strip (no collision with right cards).
  Left  : acquisition -> preprocessing -> multi-task model -> seven clinical heads
  Right : downstream analyses (external validation, Grad-CAM, CDSS, reader study, desktop)
  Bottom: cohort flow strip
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_crm import set_rc, save, INT, EXT, POS, FUS, GREY, LGREY

set_rc()
BASE = r'D:\research\人脸识别营养\传染科'
OUT = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
os.makedirs(OUT, exist_ok=True)

fig = plt.figure(figsize=(6.9, 4.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
ax.set_xticks([]); ax.set_yticks([])

PALE_BLUE, PALE_AMB, PALE_RED = '#EFF4FA', '#FBF3E3', '#FBECEA'
PALE_TEAL, PALE_GREY = '#E9F5F2', '#F3F4F6'
PURP, PALE_PURP = '#6A5F9E', '#F0EDF7'


def box(x, y, w, h, fc, ec, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                facecolor=fc, edgecolor=ec, linewidth=lw))


def arrow(x1, y1, x2, y2, col=GREY, lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9,
                                 color=col, lw=lw, shrinkA=1, shrinkB=1))


# ── LEFT: pipeline ──────────────────────────────────────────────────────────
LX, LW = 2.5, 54
ax.text(LX, 96.5, 'Image acquisition and multi-task modelling', fontsize=7.5,
        fontweight='bold', color=INT, va='center')

box(LX, 84, LW, 9, PALE_GREY, GREY)
ax.text(LX + LW / 2, 88.5, 'Facial video (ward, native illumination)\n+ upper-eyelid photograph',
        ha='center', va='center', fontsize=6.0, color='#1A1A1A', linespacing=1.35)
arrow(LX + LW / 2, 83.8, LX + LW / 2, 81.6)

box(LX, 70, LW, 11, PALE_GREY, GREY)
ax.text(LX + LW / 2, 78.7, 'Pre-processing', ha='center', va='center', fontsize=6.2,
        fontweight='bold', color='#1A1A1A')
ax.text(LX + LW / 2, 73.9, 'YuNet face detection | SAM segmentation | CLAHE\n'
        'patient-level mean softmax over up to 12 frames', ha='center', va='center',
        fontsize=5.5, color=GREY, linespacing=1.4)
arrow(LX + LW / 2, 69.8, LX + LW / 2, 68.1)

box(LX, 55.5, LW, 11.5, PALE_BLUE, INT)
ax.text(LX + LW / 2, 64.4, 'Shared deep backbones (patient-level 80/20 split, seed 42)',
        ha='center', va='center', fontsize=6.0, fontweight='bold', color=INT)
ax.text(LX + LW / 2, 59.6, 'ConvNeXt | ViT | Swin | EfficientNet\n+ probability-averaged ensemble',
        ha='center', va='center', fontsize=5.5, color='#1A1A1A', linespacing=1.4)
arrow(LX + LW / 2, 55.3, LX + LW / 2, 53.2)

heads = [('Binary\nscreening', INT), ('TBIL\ngrading', INT), ('DBIL\ngrading', INT),
         ('IBIL\ngrading', INT), ('Jaundice\ntype', INT), ('Child-Pugh', EXT),
         ('MELD risk', EXT)]
hw, gap = (LW - 6 * 1.2) / 7, 1.2
for i, (lab, col) in enumerate(heads):
    x = LX + i * (hw + gap)
    box(x, 43, hw, 9.5, 'white', col, lw=0.8)
    ax.text(x + hw / 2, 47.7, lab, ha='center', va='center', fontsize=4.9, color=col,
            linespacing=1.2)
for i in range(7):
    arrow(LX + LW / 2, 52.9, LX + i * (hw + gap) + hw / 2, 52.6, col=LGREY, lw=0.5)

# ── RIGHT: downstream analyses ─────────────────────────────────────────────
RX, RW = 60, 37.5
ax.text(RX, 96.5, 'Downstream analyses', fontsize=7.5, fontweight='bold', color='#1A1A1A')
cards = [
    ('Exploratory external validation', 'Independent cohort (n=117)\nfrozen threshold; domain adaptation', EXT, PALE_AMB),
    ('Attribution analysis', 'Grad-CAM maps, facial and eyelid\nmodels across severity strata', FUS, PALE_TEAL),
    ('Guideline-referenced CDSS', '64 guidelines | 51 rules | 6 red-flag\npathways | 17 specialties; abstention', POS, PALE_RED),
    ('Reader study', 'Eight clinicians vs model\n(binary and three-class tasks)', PURP, PALE_PURP),
    ('Desktop deployment', 'BilinGuard.exe offline inference\nwith decision-support output', GREY, PALE_GREY),
]
cy, ch, cgap = 84, 11.2, 1.6
for title, txt, ec, fc in cards:
    box(RX, cy - ch, RW, ch, fc, ec, lw=0.8)
    ax.text(RX + 1.8, cy - 2.6, title, fontsize=5.9, fontweight='bold', color=ec, va='center')
    ax.text(RX + 1.8, cy - 7.4, txt, fontsize=5.1, color='#1A1A1A', va='center',
            linespacing=1.45)
    arrow(LX + LW + 0.8, min(cy - ch / 2, 88), RX - 0.8, cy - ch / 2, col=LGREY, lw=0.7)
    cy -= (ch + cgap)

# ── BOTTOM: cohort flow strip ───────────────────────────────────────────────
ax.text(2.5, 17.6, 'Participant flow', fontsize=7.5, fontweight='bold', color='#1A1A1A',
        va='center')
flow = [('1,206', 'inpatients\nscreened'), ('939', 'enrolled internal\ncohort'),
        ('579', 'usable facial\nimages'), ('112', 'internal test set\n(patient-level)'),
        ('117', 'external cohort\n(59 / 58 jaundiced)')]
fw, fgap, fy = 17.6, 2.0, 4.6
fx = 2.5
for i, (num, lab) in enumerate(flow):
    ec = INT if i < 4 else EXT
    fc = PALE_BLUE if i < 4 else PALE_AMB
    box(fx, fy, fw, 10.6, fc, ec, lw=0.8)
    ax.text(fx + fw / 2, fy + 7.4, num, ha='center', va='center', fontsize=7.5,
            fontweight='bold', color=ec)
    ax.text(fx + fw / 2, fy + 3.3, lab, ha='center', va='center', fontsize=4.8,
            color='#1A1A1A', linespacing=1.25)
    if i < 4:
        arrow(fx + fw + 0.15, fy + 5.3, fx + fw + fgap - 0.15, fy + 5.3, col=GREY, lw=0.9)
    fx += fw + fgap
ax.text(2.5, 2.0, 'Internal cohort: 625 non-jaundiced controls / 314 jaundiced patients; '
        'all tasks scored under a single frozen protocol with 1,000-iteration bootstrap 95% CIs',
        fontsize=5.2, color=GREY, style='italic')

save(fig, os.path.join(OUT, 'Figure1_overview_crm'))
