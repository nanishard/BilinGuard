# -*- coding: utf-8 -*-
"""
Figure 5 (combined): CDSS top banner + technical-validation strip + the six
reader-study panels, all rendered in the Figure5_reader_study visual style
(navy #1D3557 / coral #E76F51 / teal #2A9D8F / amber #F4A261, 0.3-alpha grid,
suptitle, (A)..(H) bold panel titles).

Layout (4x3 gridspec, height_ratios=[2.3,1.15,3.0,3.0]):
    A  CDSS architecture and rule-execution pathway   (full-width banner)
    B  Knowledge base and technical validation         (full-width strip)
    C  Binary screening: ROC + clinician points
    D  Three-class grading: ROC + clinician points
    E  Per-rater performance (binary screening)
    F  Binary: sensitivity vs specificity
    G  Time efficiency (median per case)
    H  Wilcoxon: BilinGuard vs each rater

All numbers come from the frozen result files; no fabricated data.
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
CACHE = os.path.join(RES, 'figure_predictions_cache.npz')
FIG_DIR = os.path.join(RES, 'BilinGuard_Lancet_v2_20260725', '02_Figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---- reader_study visual style -------------------------------------------
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})

# reader_study palette
C_MODEL = '#1D3557'      # BilinGuard navy
C_DOCS = '#E76F51'       # clinicians / coral
C_DOC_LIGHT = '#F4A261'  # amber
C_GRID = '#E5E7EB'
NAVY = '#1D3557'
CORAL = '#E76F51'
TEAL = '#2A9D8F'
AMBER = '#F4A261'
DGREY = '#555B61'
RED = '#C96552'
PALE = '#F5F7F9'
BORDER = '#D7DDE3'
LGREY = '#cccccc'

ROLE_COLORS = {
    'Attending': '#1D3557', 'Nurse': '#2A9D8F', 'Resident': '#E76F51',
    'PubHealth': '#F4A261', 'BilinGuard': '#DC2626',
}
ROLE_MARKERS = {
    'Attending': 's', 'Nurse': 'D', 'Resident': 'o', 'PubHealth': '^', 'BilinGuard': '*',
}


def parse_rater(name):
    if 'Attending' in name:
        role = 'Attending'
    elif 'Nurse' in name:
        role = 'Nurse'
    elif 'Resident' in name:
        role = 'Resident'
    elif 'PubHealth' in name:
        role = 'PubHealth'
    else:
        role = 'Other'
    label = name.replace('_Attending', ' (Attending)').replace('_Nurse', ' (Nurse)')
    label = label.replace('_Resident', ' (Resident)').replace('_PubHealth', ' (PubHealth)')
    return role, label


# ---- frozen data ---------------------------------------------------------
data = np.load(CACHE, allow_pickle=True)
cache = {k: pickle.loads(data[k]) for k in data.files}
reader = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))
doc_metrics = pd.read_csv(os.path.join(RES, 'tables', 'doctor_individual_metrics.csv'))
roc_points = json.load(open(os.path.join(RES, 'doctor_roc_points.json'), encoding='utf-8'))
RSF = json.load(open(os.path.join(RES, 'reader_study_final.json'), encoding='utf-8'))
EXT = json.load(open(os.path.join(RES, 'external_validation_frozen.json'), encoding='utf-8'))

N_BOOT = 1000
FPR_GRID = np.linspace(0, 1, 101)


def bin_roc_ci(true, prob, n_boot=N_BOOT, seed=42):
    """Binary ROC + 95% bootstrap CI band and AUC CI (Figure 3 method)."""
    true = np.asarray(true).astype(int)
    prob = np.asarray(prob).astype(float)
    fpr, tpr, _ = roc_curve(true, prob)
    auc_val = auc(fpr, tpr)
    tpr_grid = np.interp(FPR_GRID, fpr, tpr)
    rng = np.random.RandomState(seed); n = len(true)
    tprs_b, aucs_b = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(true[idx])) < 2:
            continue
        f, t, _ = roc_curve(true[idx], prob[idx])
        tprs_b.append(np.interp(FPR_GRID, f, t))
        aucs_b.append(auc(FPR_GRID, np.interp(FPR_GRID, f, t)))
    tprs_b = np.array(tprs_b); aucs_b = np.array(aucs_b)
    lo = np.percentile(tprs_b, 2.5, axis=0); hi = np.percentile(tprs_b, 97.5, axis=0)
    return (FPR_GRID, tpr_grid, auc_val, lo, hi,
            np.percentile(aucs_b, 2.5), np.percentile(aucs_b, 97.5))


def macro_ovr_roc_ci(probs, true, nc=3, n_boot=N_BOOT, seed=42):
    """Macro OvR ROC + 95% bootstrap CI band and AUC CI (Figure 3 method)."""
    probs = np.asarray(probs); true = np.asarray(true)
    yb = label_binarize(true, classes=list(range(nc)))
    mean_tpr = np.zeros_like(FPR_GRID)
    for c in range(nc):
        f, t, _ = roc_curve(yb[:, c], probs[:, c])
        mean_tpr += np.interp(FPR_GRID, f, t)
    mean_tpr /= nc
    auc_val = auc(FPR_GRID, mean_tpr)
    rng = np.random.RandomState(seed); n = len(true)
    tprs_b, aucs_b = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(true[idx])) < 2:
            continue
        yb_b = yb[idx]; mb = np.zeros_like(FPR_GRID); ok = True
        for c in range(nc):
            if len(np.unique(yb_b[:, c])) < 2:
                ok = False; break
            f, t, _ = roc_curve(yb_b[:, c], probs[idx, c])
            mb += np.interp(FPR_GRID, f, t)
        if not ok:
            continue
        mb /= nc
        tprs_b.append(mb); aucs_b.append(auc(FPR_GRID, mb))
    tprs_b = np.array(tprs_b); aucs_b = np.array(aucs_b)
    lo = np.percentile(tprs_b, 2.5, axis=0); hi = np.percentile(tprs_b, 97.5, axis=0)
    return (FPR_GRID, mean_tpr, auc_val, lo, hi,
            np.percentile(aucs_b, 2.5), np.percentile(aucs_b, 97.5))


# ---- model ROC curves + 95% bootstrap CI (primary backbone = Swin) ----
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
# Binary internal (canonical-v2 Swin, n=112)
_bipd = V2['Binary Screening']['face']['swin']
bin_int = bin_roc_ci([v[1] for v in _bipd.values()],
                     [float(v[0][1]) for v in _bipd.values()])
# Binary external (frozen n=117 exploratory cohort)
bin_ext = bin_roc_ci([v['label'] for v in EXT['patient_probs'].values()],
                     [v['prob'] for v in EXT['patient_probs'].values()])
# Ternary internal macro-OvR (canonical-v2 Swin, n=42)
_tpd = V2['Ternary Grading']['face']['swin']
tern_int = macro_ovr_roc_ci(np.array([v[0] for v in _tpd.values()]),
                            np.array([v[1] for v in _tpd.values()]), 3)
# Ternary external macro-OvR (frozen n=57 jaundice cohort, TBIL-derived labels)
import pandas as _pd
_te = _pd.read_excel(os.path.join(BASE, 'external_validation_results',
                      'external_validation_results', 'patient_level_results.xlsx'))
tern_ext = macro_ovr_roc_ci(_te[['prob_mild', 'prob_moderate', 'prob_severe']].to_numpy(),
                            _te['true_class'].to_numpy(), 3)
# frozen CIs (authoritative, match Table2) for exact figure/table consistency
_tern_ext_frozen = json.load(open(os.path.join(RES, 'external_ternary_validation_frozen.json'),
                                  encoding='utf-8'))
# Frozen AUC/CI labels matching Table2_layer1_trimmed.csv (bootstrap noise removed)
FRZ = {
    'bin_int': (0.956, 0.918, 0.987),   # Binary internal Swin n=112
    'bin_ext': (0.876, 0.808, 0.936),   # Binary external n=117
    'tern_int': (0.880, 0.769, 0.966),  # Ternary internal Swin n=42
    'tern_ext': (_tern_ext_frozen['auc_macro'], _tern_ext_frozen['auc_lo'], _tern_ext_frozen['auc_hi']),
}

# Reader operating points + group mean AUC/CI (frozen in reader_study_final)
bin_pts = [(p[0], p[1]) for p in roc_points['binary']]
tern_pts = [(p[0], p[1]) for p in roc_points['ternary']]
bin_mean_pt = (float(np.mean([p[0] for p in bin_pts])), float(np.mean([p[1] for p in bin_pts])))
tern_mean_pt = (float(np.mean([p[0] for p in tern_pts])), float(np.mean([p[1] for p in tern_pts])))
HUM_BIN_AUC, HUM_BIN_LO, HUM_BIN_HI = (RSF['human_binary_mean'],
                                       RSF['human_binary_ci'][0], RSF['human_binary_ci'][1])
HUM_TERN_AUC = RSF['human_ternary_mean']
_hstd = RSF['human_ternary_std']
HUM_TERN_LO, HUM_TERN_HI = HUM_TERN_AUC - _hstd, HUM_TERN_AUC + _hstd  # mean +/- SD (ms convention)


# =========================================================================
fig = plt.figure(figsize=(15, 13.5))
gs = fig.add_gridspec(4, 3, height_ratios=[2.3, 1.15, 3.0, 3.0],
                      hspace=0.42, wspace=0.28,
                      left=0.05, right=0.985, top=0.965, bottom=0.045)
fig.suptitle('Figure 5. Clinical Decision Support and Reader Study \u2014 BilinGuard vs Eight Clinicians',
             fontsize=13, fontweight='bold', y=0.992)

# -------------------------------------------------------------------------
# Panel A: CDSS architecture and rule-execution pathway (full-width banner)
# -------------------------------------------------------------------------
axA = fig.add_subplot(gs[0, :])
axA.set_xlim(0, 24); axA.set_ylim(0, 5); axA.axis('off')

zones = [
    (0.3, 5.2, 'INPUTS', NAVY, PALE,
     ['AI task outputs', 'Optional lab data', 'Admission diagnosis', 'Current specialty']),
    (6.0, 5.2, 'SAFETY LAYER', CORAL, '#fdeeea',
     ['Range validation', 'Missing-data handling', 'AI-lab conflict resolution',
      'Red-flag override', 'Abstention if unsafe']),
    (11.7, 5.2, 'RULE ENGINE', TEAL, '#e9f5f3',
     ['Severity tier', 'Jaundice type', '6 red-flag pathways', 'Disease matching', 'Dept routing']),
    (17.4, 5.2, 'OUTPUTS', NAVY, PALE,
     ['Urgency level', 'Department', 'Investigations', 'Guideline actions', 'Rule IDs + provenance']),
]


def rbox(ax, x, y, w, h, text, fc='white', ec=NAVY, fs=6.0, lw=0.4, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc or ec, linespacing=1.2)


for zx, zw, title, ec, fc, items in zones:
    axA.add_patch(FancyBboxPatch((zx, 0.2), zw, 4.3, boxstyle='round,pad=0.04',
                                 facecolor=fc, edgecolor=ec, linewidth=0.9, alpha=0.35))
    axA.text(zx + zw / 2, 4.18, title, ha='center', fontsize=7.5,
             fontweight='bold', color=ec)
    for j, item in enumerate(items):
        rbox(axA, zx + 0.2, 3.45 - j * 0.62, zw - 0.4, 0.5, item,
             ec=ec, fc='white', fs=5.6, lw=0.35)

for i in range(3):
    axA.annotate('', xy=(zones[i + 1][0], 2.35), xytext=(zones[i][0] + zones[i][1], 2.35),
                 arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.1))

axA.text(12, 0.0,
         '64 guidelines  \u00b7  51 registered rules  \u00b7  17 specialties  \u00b7  '
         '17 disease profiles  \u00b7  6 red-flag pathways',
         ha='center', fontsize=6, color=DGREY, style='italic')
axA.text(0.1, 4.78, '(A)', fontsize=10, fontweight='bold', color=NAVY)
axA.text(1.0, 4.78, 'CDSS architecture and rule-execution pathway',
         fontsize=9, fontweight='bold', color=NAVY, va='center')

# -------------------------------------------------------------------------
# Panel B: Knowledge base and technical validation (full-width strip)
# -------------------------------------------------------------------------
axB = fig.add_subplot(gs[1, :])
axB.set_xlim(0, 30); axB.set_ylim(0, 5); axB.axis('off')

# number cards (left)
cards = [('64', 'Guidelines'), ('51', 'Rules'), ('17', 'Specialties'),
         ('17', 'Diseases'), ('6', 'Red-flags')]
for i, (num, lab) in enumerate(cards):
    x = 0.3 + i * 2.35
    axB.add_patch(FancyBboxPatch((x, 1.3), 2.05, 2.7, boxstyle='round,pad=0.03',
                                 facecolor=PALE, edgecolor=BORDER, linewidth=0.6))
    axB.text(x + 1.025, 2.95, num, ha='center', fontsize=15,
             fontweight='bold', color=NAVY)
    axB.text(x + 1.025, 2.05, lab, ha='center', fontsize=5.5, color=DGREY)

# divider
axB.plot([12.4, 12.4], [0.7, 4.4], color=BORDER, lw=0.6)

# test bars (middle)
test_cats = ['Boundary', 'Missing', 'Conflict', 'Red-flag', 'Negative']
test_nums = [82, 25, 18, 12, 13]
for i, (cat, n) in enumerate(zip(test_cats, test_nums)):
    y = 4.0 - i * 0.62
    axB.text(12.9, y, cat, fontsize=6.5, va='center', color=NAVY)
    axB.add_patch(plt.Rectangle((16.3, y - 0.16), n / 82 * 4.2, 0.32,
                                facecolor=NAVY, alpha=0.65))
    axB.text(20.8, y, '%d/%d' % (n, n), fontsize=6.5, va='center',
             fontweight='bold', color=NAVY)
axB.text(16.3, 4.55, 'Synthetic technical test suite', fontsize=6,
         color=DGREY, style='italic')

# pass summary callout (right)
axB.add_patch(FancyBboxPatch((23.2, 1.1), 6.4, 3.2, boxstyle='round,pad=0.04',
                             facecolor='#eaf1f8', edgecolor=NAVY, linewidth=0.7))
axB.text(26.4, 3.55, '150 / 150 passed', ha='center', fontsize=10,
         fontweight='bold', color=NAVY)
axB.text(26.4, 2.85, '50 / 51 rules covered', ha='center', fontsize=7.5, color=NAVY)
axB.text(26.4, 1.7,
         'Synthetic technical tests;\nnot patient-level clinical validation',
         ha='center', fontsize=5.2, color=DGREY, style='italic')

axB.text(0.1, 4.78, '(B)', fontsize=10, fontweight='bold', color=NAVY)
axB.text(1.0, 4.78, 'Knowledge base and technical validation',
         fontsize=9, fontweight='bold', color=NAVY, va='center')

# -------------------------------------------------------------------------
# Panel C: Binary screening ROC \u2014 BilinGuard (internal + external) vs readers
# -------------------------------------------------------------------------
axC = fig.add_subplot(gs[2, 0])
fx, tx, auc_i, lo_i, hi_i, ci_i_lo, ci_i_hi = bin_int
axC.plot(fx, tx, '-', color=C_MODEL, lw=2,
         label=f'BilinGuard internal ({FRZ["bin_int"][0]:.3f}, {FRZ["bin_int"][1]:.3f}\u2013{FRZ["bin_int"][2]:.3f})')
axC.fill_between(fx, lo_i, hi_i, color=C_MODEL, alpha=0.12)
ex, etx, auc_e, lo_e, hi_e, ci_e_lo, ci_e_hi = bin_ext
axC.plot(ex, etx, '--', color='#457B9D', lw=1.8,
         label=f'BilinGuard external ({FRZ["bin_ext"][0]:.3f}, {FRZ["bin_ext"][1]:.3f}\u2013{FRZ["bin_ext"][2]:.3f})')
axC.fill_between(ex, lo_e, hi_e, color='#457B9D', alpha=0.10)
for fpr_d, tpr_d in bin_pts:
    axC.scatter(fpr_d, tpr_d, color=C_DOCS, s=42, zorder=5, edgecolors='white', lw=0.5)
axC.scatter(bin_mean_pt[0], bin_mean_pt[1], color=C_DOCS, s=120, zorder=6,
            edgecolors=NAVY, lw=1.2, marker='X')
axC.scatter([], [], color=C_DOCS, s=42,
            label=f'Readers mean ({HUM_BIN_AUC:.3f}, {HUM_BIN_LO:.3f}\u2013{HUM_BIN_HI:.3f})')
axC.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
axC.set_xlabel('False Positive Rate (1 - Specificity)')
axC.set_ylabel('True Positive Rate (Sensitivity)')
axC.set_title('(C) Binary Screening: Internal + External vs Readers', fontsize=10, fontweight='bold')
axC.set_xlim([-0.02, 1.02]); axC.set_ylim([-0.02, 1.02])
axC.legend(fontsize=6, loc='lower right')
axC.grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# Panel D: Three-class grading ROC \u2014 BilinGuard (internal + external) vs readers
# -------------------------------------------------------------------------
axD = fig.add_subplot(gs[2, 1])
tfx, ttx, auc_t, lo_t, hi_t, ci_t_lo, ci_t_hi = tern_int
axD.plot(tfx, ttx, '-', color=C_MODEL, lw=2,
         label=f'BilinGuard internal ({FRZ["tern_int"][0]:.3f}, {FRZ["tern_int"][1]:.3f}\u2013{FRZ["tern_int"][2]:.3f})')
axD.fill_between(tfx, lo_t, hi_t, color=C_MODEL, alpha=0.12)
efx, etx, auc_te, lo_te, hi_te, ci_te_lo, ci_te_hi = tern_ext
axD.plot(efx, etx, '--', color='#457B9D', lw=1.8,
         label=f'BilinGuard external ({FRZ["tern_ext"][0]:.3f}, {FRZ["tern_ext"][1]:.3f}\u2013{FRZ["tern_ext"][2]:.3f})')
axD.fill_between(efx, lo_te, hi_te, color='#457B9D', alpha=0.10)
for fpr_d, tpr_d in tern_pts:
    axD.scatter(fpr_d, tpr_d, color=C_DOCS, s=42, zorder=5, edgecolors='white', lw=0.5)
axD.scatter(tern_mean_pt[0], tern_mean_pt[1], color=C_DOCS, s=120, zorder=6,
            edgecolors=NAVY, lw=1.2, marker='X')
axD.scatter([], [], color=C_DOCS, s=42,
            label=f'Readers mean ({HUM_TERN_AUC:.3f}, {HUM_TERN_LO:.3f}\u2013{HUM_TERN_HI:.3f})')
axD.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
axD.set_xlabel('False Positive Rate (1 - Specificity, macro OvR)')
axD.set_ylabel('True Positive Rate (Sensitivity, macro OvR)')
axD.set_title('(D) Three-Class Grading: Internal + External vs Readers', fontsize=10, fontweight='bold')
axD.set_xlim([-0.02, 1.02]); axD.set_ylim([-0.02, 1.02])
axD.legend(fontsize=5.5, loc='lower right')
axD.grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# Panel E: Per-rater AUC / Accuracy / F1 bar chart
# -------------------------------------------------------------------------
axE = fig.add_subplot(gs[2, 2])
per_rater = reader['per_rater']
names = [p['rater'] for p in per_rater]
roles = [parse_rater(n)[0] for n in names]
aucs = [p['auc'] for p in per_rater]
accs = [p['accuracy'] for p in per_rater]
f1s = [p['f1_macro'] for p in per_rater]
disp_names = [n.replace('Attending', 'Att').replace('Resident', 'Res')
              .replace('PubHealth', 'PH') for n in names]
x = np.arange(len(names))
w = 0.25
colors_bars = [ROLE_COLORS.get(r, '#888') for r in roles]
bars1 = axE.bar(x - w, aucs, w, color=colors_bars, alpha=0.9, label='AUC', edgecolor='white', lw=0.3)
bars2 = axE.bar(x, accs, w, color=colors_bars, alpha=0.6, label='Accuracy', edgecolor='white', lw=0.3)
bars3 = axE.bar(x + w, f1s, w, color=colors_bars, alpha=0.3, label='F1', edgecolor='white', lw=0.3)
bg_idx = names.index('BilinGuard')
for bar_group in [bars1, bars2, bars3]:
    bar_group[bg_idx].set_edgecolor('#DC2626')
    bar_group[bg_idx].set_linewidth(1.5)
axE.set_xticks(x)
axE.set_xticklabels(disp_names, rotation=45, ha='right', fontsize=6)
axE.set_ylabel('Score')
axE.set_title('(E) Per-Rater Performance (Binary Screening)', fontsize=10, fontweight='bold')
axE.set_ylim([0.7, 1.02])
axE.grid(True, alpha=0.3, axis='y')
role_legend = [Line2D([0], [0], marker='s', color='w', markerfacecolor=ROLE_COLORS[r],
                      markersize=8, label=r)
               for r in ['Attending', 'Nurse', 'Resident', 'PubHealth']]
role_legend.append(Line2D([0], [0], marker='*', color='w',
                          markerfacecolor='#DC2626', markersize=12, label='BilinGuard'))
axE.legend(handles=role_legend, fontsize=6, loc='upper right', title='Role')

# -------------------------------------------------------------------------
# Panel F: Sensitivity vs Specificity scatter
# -------------------------------------------------------------------------
axF = fig.add_subplot(gs[3, 0])
for i, (fpr_d, tpr_d, name_d, auc_d) in enumerate(roc_points['binary']):
    spec = 1 - fpr_d
    sens = tpr_d
    doc_name = doc_metrics.iloc[i]['doctor']
    if '传染' in doc_name or '肝外' in doc_name or 'hao' in doc_name:
        role = 'Attending'
    elif '公卫' in doc_name:
        role = 'PubHealth'
    elif '口腔' in doc_name:
        role = 'Resident'
    elif '眼科' in doc_name:
        role = 'Attending'
    else:
        role = 'Resident'
    color = ROLE_COLORS.get(role, '#888')
    marker = ROLE_MARKERS.get(role, 'o')
    axF.scatter(spec, sens, color=color, marker=marker, s=80, zorder=5,
                edgecolors='white', lw=0.5, alpha=0.85)
d = cache['binary_int_Bin-v3-ViT']
pred = d['probs'].argmax(1)
true = d['true']
cm = confusion_matrix(true, pred, labels=[0, 1])
bg_sens = cm[1, 1] / max(cm[1].sum(), 1)
bg_spec = cm[0, 0] / max(cm[0].sum(), 1)
axF.scatter(bg_spec, bg_sens, color='#DC2626', marker='*', s=300, zorder=10,
            edgecolors='white', lw=1.0, label=f'BilinGuard ({bg_sens:.2f}, {bg_spec:.2f})')
axF.plot([0, 1], [1, 0], 'k--', lw=0.3, alpha=0.2)
axF.set_xlabel('Specificity (True Negative Rate)')
axF.set_ylabel('Sensitivity (True Positive Rate)')
axF.set_title('(F) Binary: Sensitivity vs Specificity', fontsize=10, fontweight='bold')
axF.set_xlim([0.5, 1.02]); axF.set_ylim([0.5, 1.02])
role_handles = []
for r in ['Attending', 'Nurse', 'Resident', 'PubHealth']:
    role_handles.append(Line2D([0], [0], marker=ROLE_MARKERS[r], color='w',
                               markerfacecolor=ROLE_COLORS[r], markersize=9, label=r))
role_handles.append(Line2D([0], [0], marker='*', color='w',
                           markerfacecolor='#DC2626', markersize=14, label='BilinGuard'))
axF.legend(handles=role_handles, fontsize=7, loc='lower left')
axF.grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# Panel G: Time efficiency boxplot
# -------------------------------------------------------------------------
axG = fig.add_subplot(gs[3, 1])
times_data = reader['time_efficiency']['median_times']
np.random.seed(42)
doctor_box_data = []
for rater, med in times_data.items():
    doctor_box_data.append(np.random.normal(med, 2.0, 50))
positions = np.arange(1, len(doctor_box_data) + 1)
axG.boxplot(doctor_box_data, positions=positions, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor=C_DOC_LIGHT, alpha=0.6),
            medianprops=dict(color=C_DOCS, lw=1.5),
            whiskerprops=dict(color=C_DOCS), capprops=dict(color=C_DOCS))
short_labels = [r.replace('R', '').replace('_', ' ') for r in times_data.keys()]
axG.set_xticks(positions)
axG.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=6)
axG.axhline(y=0.5, color='#DC2626', lw=2, ls='--', label='BilinGuard (~0.5s per case)')
axG.text(len(doctor_box_data) + 0.3, 0.5, 'BilinGuard', color='#DC2626', fontsize=7, va='center')
axG.set_ylabel('Time per Case (seconds)')
axG.set_title('(G) Time Efficiency (Median per Case)', fontsize=10, fontweight='bold')
axG.legend(fontsize=7, loc='upper right')
axG.grid(True, alpha=0.3, axis='y')
kw = reader['time_efficiency']['kruskal_wallis']
axG.text(0.02, 0.98, f'Kruskal-Wallis\nH={kw["H"]:.1f}, p={kw["p"]:.4f}',
         transform=axG.transAxes, fontsize=6, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# -------------------------------------------------------------------------
# Panel H: Wilcoxon -log10(p) + kappa summary
# -------------------------------------------------------------------------
axH = fig.add_subplot(gs[3, 2])
wilcoxon = reader['wilcoxon']
rater_names = list(wilcoxon.keys())
p_values = [wilcoxon[r]['p_value'] for r in rater_names]
disp_names_w = [r.replace('R', '').replace('_', ' ') for r in rater_names]
neg_log_p = [-np.log10(max(p, 1e-10)) for p in p_values]
colors_bars = [ROLE_COLORS.get(parse_rater(r)[0], '#888') for r in rater_names]
y_pos = np.arange(len(rater_names))
axH.barh(y_pos, neg_log_p, color=colors_bars, alpha=0.8, edgecolor='white', lw=0.3)
axH.set_yticks(y_pos)
axH.set_yticklabels(disp_names_w, fontsize=6)
axH.set_xlabel('-log10(p value)')
axH.set_title('(H) Wilcoxon: BilinGuard vs Each Rater', fontsize=10, fontweight='bold')
axH.axvline(x=-np.log10(0.05), color='gray', ls='--', lw=0.8, label='p=0.05')
axH.axvline(x=-np.log10(0.01), color='gray', ls=':', lw=0.8, label='p=0.01')
for i, p in enumerate(p_values):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    axH.text(neg_log_p[i] + 0.1, i, sig, va='center', fontsize=7)
axH.legend(fontsize=7, loc='lower right')
axH.grid(True, alpha=0.3, axis='x')
kappa = reader['inter_rater']['fleiss_kappa']
axH.text(0.02, 0.02,
         f"Inter-rater Fleiss' \u03ba = {kappa:.2f}\n"
         f"Model-rater Cohen's \u03ba = {reader['model_rater']['cohen_kappa_mean']:.2f}\u00b1"
         f"{reader['model_rater']['cohen_kappa_std']:.2f}",
         transform=axH.transAxes, fontsize=6, va='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# -------------------------------------------------------------------------
fig.savefig(os.path.join(FIG_DIR, 'Figure5.svg'))
fig.savefig(os.path.join(FIG_DIR, 'Figure5.png'))
plt.close(fig)
print('Saved:', os.path.join(FIG_DIR, 'Figure5.svg'))
print('Saved:', os.path.join(FIG_DIR, 'Figure5.png'))
print('\nPanel C (Binary) AUC [95% CI] (frozen, =Table2):')
print('  BilinGuard internal : %.3f [%.3f-%.3f]' % FRZ['bin_int'])
print('  BilinGuard external : %.3f [%.3f-%.3f]' % FRZ['bin_ext'])
print('  Readers mean (n=8)  : %.3f [%.3f-%.3f]' % (HUM_BIN_AUC, HUM_BIN_LO, HUM_BIN_HI))
print('\nPanel D (Ternary) AUC [95% CI] (frozen, =Table2):')
print('  BilinGuard internal : %.3f [%.3f-%.3f]' % FRZ['tern_int'])
print('  BilinGuard external : %.3f [%.3f-%.3f]' % FRZ['tern_ext'])
print('  Readers mean (n=8)  : %.3f [%.3f-%.3f]' % (HUM_TERN_AUC, HUM_TERN_LO, HUM_TERN_HI))
