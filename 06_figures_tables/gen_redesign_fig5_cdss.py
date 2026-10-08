# -*- coding: utf-8 -*-
"""
Redesigned Figure 5 v2 (CRM/3D-FAICE style): CDSS + reader study, compact 3-band layout.
  A  CDSS architecture banner (full width)
  B  grouped bars: BilinGuard vs reader mean (binary + ternary, Wilcoxon stars)
  C  binary ROC (internal/external) + reader operating points
  D  ternary macro ROC + reader points
  E  quantitative bars (AUC / accuracy / macro-F1 vs reader mean)
  F  sensitivity vs specificity scatter (readers by role + model star)
  G  Wilcoxon -log10(p) per reader
Frozen data only. Time-efficiency and kappa stats go to the caption.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_crm import (set_rc, panel_letter, roc_ax, save, FPR_GRID, GREY, LGREY,
                          INT, EXT, POS, FUS, READER, sig, roc_band_arrays,
                          macro_ovr_roc_arrays)

set_rc()
BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
os.makedirs(OUT, exist_ok=True)

V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
EXTJ = json.load(open(os.path.join(RES, 'external_validation_frozen.json'), encoding='utf-8'))
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))
RSF = json.load(open(os.path.join(RES, 'reader_study_final.json'), encoding='utf-8'))
ROCPTS = json.load(open(os.path.join(RES, 'doctor_roc_points.json'), encoding='utf-8'))
TE_FROZ = json.load(open(os.path.join(RES, 'external_ternary_validation_frozen.json'), encoding='utf-8'))
TE_XLSX = pd.read_excel(os.path.join(BASE, 'external_validation_results',
                        'external_validation_results', 'patient_level_results.xlsx'))

FRZ = {'bin_int': (0.956, 0.918, 0.987), 'bin_ext': (0.876, 0.808, 0.936),
       'tern_int': (0.880, 0.769, 0.966),
       'tern_ext': (TE_FROZ['auc_macro'], TE_FROZ['auc_lo'], TE_FROZ['auc_hi'])}

per_rater = RS['per_rater']
readers = [r for r in per_rater if r['rater'] != 'BilinGuard']
bg = next(r for r in per_rater if r['rater'] == 'BilinGuard')
reader_aucs = [r['auc'] for r in readers]
bin_pts = [(p[0], p[1]) for p in ROCPTS['binary']]
tern_pts = [(p[0], p[1]) for p in ROCPTS['ternary']]
bin_mean_pt = (float(np.mean([p[0] for p in bin_pts])), float(np.mean([p[1] for p in bin_pts])))
tern_mean_pt = (float(np.mean([p[0] for p in tern_pts])), float(np.mean([p[1] for p in tern_pts])))

_bipd = V2['Binary Screening']['face']['swin']
bin_int = roc_band_arrays([v[1] for v in _bipd.values()],
                          [float(v[0][1]) for v in _bipd.values()])
bin_ext = roc_band_arrays([v['label'] for v in EXTJ['patient_probs'].values()],
                          [v['prob'] for v in EXTJ['patient_probs'].values()])
_tpd = V2['Ternary Grading']['face']['swin']
tern_int = macro_ovr_roc_arrays(np.array([v[1] for v in _tpd.values()]),
                                np.array([v[0] for v in _tpd.values()]), 3)
tern_ext = macro_ovr_roc_arrays(TE_XLSX['true_class'].to_numpy(),
                                TE_XLSX[['prob_mild', 'prob_moderate', 'prob_severe']].to_numpy(), 3)

ROLE_COL = {'Attending': '#2B6CB0', 'Nurse': '#2E8B6E', 'Resident': '#D97706',
            'PubHealth': '#8A8F98'}
ROLE_MK = {'Attending': 's', 'Nurse': 'D', 'Resident': 'o', 'PubHealth': '^'}

def role_of(name):
    n = str(name)
    if 'Attending' in n: return 'Attending'
    if 'Nurse' in n: return 'Nurse'
    if 'Resident' in n: return 'Resident'
    return 'PubHealth'

# ================= canvas =================
FW, FH = 6.9, 7.4
fig = plt.figure(figsize=(FW, FH))

# ── A: CDSS banner ──────────────────────────────────────────────────────────
ban_h = 1.55 / FH                       # 1.55 in tall
ban_top = 0.985
axA = fig.add_axes([0.02, ban_top - ban_h, 0.96, ban_h])
axA.set_xlim(0, 24); axA.set_ylim(0, 5.6); axA.axis('off')
axA.set_xticks([]); axA.set_yticks([])
zones = [
    (0.3, 5.2, 'INPUTS', INT, '#EFF4FA',
     ['AI task outputs', 'Optional lab data', 'Admission diagnosis', 'Current specialty']),
    (6.0, 5.2, 'SAFETY LAYER', POS, '#FBECEA',
     ['Range validation', 'Missing-data handling', 'AI-lab conflict resolution',
      'Red-flag override', 'Abstention if unsafe']),
    (11.7, 5.2, 'RULE ENGINE', FUS, '#E9F5F2',
     ['Severity tier', 'Jaundice type', '6 red-flag pathways', 'Disease matching',
      'Department routing']),
    (17.4, 6.2, 'OUTPUTS', GREY, '#F3F4F6',
     ['Urgency level', 'Department', 'Investigations', 'Guideline actions',
      'Rule IDs + provenance']),
]
def rbox(ax, x, y, w, h, text, fc, ec, fs=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                facecolor=fc, edgecolor=ec, linewidth=0.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color=ec, linespacing=1.15)
for zx, zw, title, ec, fc, items in zones:
    axA.add_patch(FancyBboxPatch((zx, 0.55), zw, 4.5, boxstyle='round,pad=0.04',
                                 facecolor=fc, edgecolor=ec, linewidth=0.8, alpha=0.5))
    axA.text(zx + zw / 2, 4.72, title, ha='center', fontsize=7, fontweight='bold', color=ec)
    for j, item in enumerate(items):
        rbox(axA, zx + 0.18, 3.62 - j * 0.72, zw - 0.36, 0.56, item, 'white', ec)
for i in range(3):
    axA.annotate('', xy=(zones[i + 1][0], 2.75), xytext=(zones[i][0] + zones[i][1], 2.75),
                 arrowprops=dict(arrowstyle='->', color=GREY, lw=1.1))
axA.text(12, 0.12, '64 guidelines | 51 rules | 17 specialties | 17 disease profiles | '
         '6 red-flag pathways   ·   150/150 synthetic technical scenarios passed',
         ha='center', fontsize=5.8, color=GREY, style='italic')
fig.text(0.008, ban_top + 0.002, 'A', fontsize=11, fontweight='bold', va='bottom')

# ── lower 2x3 grid ──────────────────────────────────────────────────────────
gs = fig.add_gridspec(2, 3, left=0.09, right=0.985, top=ban_top - ban_h - 0.055,
                      bottom=0.075, hspace=0.62, wspace=0.42)

# ── B: grouped bars model vs reader mean ────────────────────────────────────
axB = fig.add_subplot(gs[0, 0])
doc_tern = pd.read_csv(os.path.join(RES, 'tables', 'doctor_individual_metrics.csv')) \
    .query('task=="Ternary"')
tern_aucs = doc_tern['auc'].tolist()
m_vals = [RSF['model_binary_auc'], RSF['model_ternary_auc']]
r_vals = [RSF['human_binary_mean'], RSF['human_ternary_mean']]
r_lo = [r_vals[0] - RSF['human_binary_ci'][0], RSF['human_ternary_std']]
r_hi = [RSF['human_binary_ci'][1] - r_vals[0], RSF['human_ternary_std']]
p_b = wilcoxon(np.array(reader_aucs) - m_vals[0]).pvalue
p_t = wilcoxon(np.array(tern_aucs) - m_vals[1]).pvalue
x = np.arange(2); w = 0.34
axB.bar(x - w / 2, m_vals, w, color=POS, edgecolor='white', lw=0.4, label='BilinGuard')
axB.bar(x + w / 2, r_vals, w, color=READER, alpha=0.45, edgecolor='white', lw=0.4,
        yerr=[r_lo, r_hi], error_kw=dict(lw=0.8, capsize=2.5), label='Reader mean')
for i in range(2):
    axB.text(x[i] - w / 2, m_vals[i] + 0.015, f'{m_vals[i]:.3f}', ha='center', fontsize=5.2,
             color=POS)
    axB.text(x[i] + w / 2, r_vals[i] + r_hi[i] + 0.015, f'{r_vals[i]:.3f}', ha='center',
             fontsize=5.2, color=GREY)
    top = max(m_vals[i], r_vals[i] + r_hi[i]) + 0.07
    axB.plot([x[i] - w / 2, x[i] + w / 2], [top, top], color=GREY, lw=0.7)
    axB.text(x[i], top + 0.012, sig(p_b if i == 0 else p_t), ha='center', fontsize=7.5,
             fontweight='bold', color='#1A1A1A')
axB.set_xticks(x); axB.set_xticklabels(['Binary\nscreening', 'Three-class\ngrading'], fontsize=6)
axB.set_ylim(0.4, 1.24); axB.set_ylabel('AUC')
axB.legend(loc='upper left', fontsize=6, ncol=2, columnspacing=1.0, borderaxespad=0.25)
panel_letter(axB, 'B')

# ── C/D: ROCs + reader points (short legends) ───────────────────────────────
def roc_readers(ax, bands_i, bands_e, frz_i, frz_e, pts, mean_pt, letter):
    fx, tx, a, lo, hi, *_ = bands_i
    ax.plot(fx, tx, '-', color=INT, lw=1.6, label=f'Internal {frz_i[0]:.3f}')
    if lo is not None:
        ax.fill_between(FPR_GRID, lo, hi, color=INT, alpha=0.15, lw=0)
    fx2, tx2, a2, lo2, hi2, *_ = bands_e
    ax.plot(fx2, tx2, '-', color=EXT, lw=1.6, label=f'External {frz_e[0]:.3f}')
    if lo2 is not None:
        ax.fill_between(FPR_GRID, lo2, hi2, color=EXT, alpha=0.15, lw=0)
    for fpr_d, tpr_d in pts:
        ax.scatter(fpr_d, tpr_d, color=READER, s=20, zorder=5, edgecolors='white', lw=0.4)
    ax.scatter(*mean_pt, color=READER, s=70, zorder=6, edgecolors='#1A1A1A', lw=0.9,
               marker='X')
    ax.scatter([], [], color=READER, s=20, label='Readers')
    roc_ax(ax)
    ax.legend(loc='lower right', fontsize=5.5, handlelength=1.3, borderaxespad=0.2)
    panel_letter(ax, letter)

axC = fig.add_subplot(gs[0, 1])
roc_readers(axC, bin_int, bin_ext, FRZ['bin_int'], FRZ['bin_ext'], bin_pts, bin_mean_pt, 'C')
axD = fig.add_subplot(gs[0, 2])
roc_readers(axD, tern_int, tern_ext, FRZ['tern_int'], FRZ['tern_ext'], tern_pts,
            tern_mean_pt, 'D')

# ── E: quantitative bars (AUC / Acc / macro-F1) ─────────────────────────────
axE = fig.add_subplot(gs[1, 0])
acc_r = [r['accuracy'] for r in readers]; f1_r = [r['f1_macro'] for r in readers]
groups = ['AUC', 'Accuracy', 'Macro F1']
m3 = [bg['auc'], bg['accuracy'], bg['f1_macro']]
r3 = [np.mean(reader_aucs), np.mean(acc_r), np.mean(f1_r)]
r3_err = [1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) for v in (reader_aucs, acc_r, f1_r)]
x = np.arange(3); w = 0.34
axE.bar(x - w / 2, m3, w, color=POS, edgecolor='white', lw=0.4, label='BilinGuard')
axE.bar(x + w / 2, r3, w, color=READER, alpha=0.45, edgecolor='white', lw=0.4,
        yerr=r3_err, error_kw=dict(lw=0.8, capsize=2.5), label='Reader mean')
for i in range(3):
    axE.text(x[i] - w / 2, m3[i] + 0.015, f'{m3[i]:.2f}', ha='center', fontsize=5.2, color=POS)
    axE.text(x[i] + w / 2, r3[i] + r3_err[i] + 0.015, f'{r3[i]:.2f}', ha='center',
             fontsize=5.2, color=GREY)
axE.set_xticks(x); axE.set_xticklabels(groups, fontsize=6)
axE.set_ylim(0.5, 1.22); axE.set_ylabel('Score (binary task)')
axE.legend(loc='upper left', fontsize=5.5, ncol=2, columnspacing=1.0, borderaxespad=0.25)
panel_letter(axE, 'E')

# ── F: sensitivity vs specificity scatter ───────────────────────────────────
axF = fig.add_subplot(gs[1, 1])
for i, (fpr_d, tpr_d) in enumerate(bin_pts):
    nm = ROCPTS['binary'][i][2] if len(ROCPTS['binary'][i]) > 2 else ''
    role = role_of(nm)
    axF.scatter(1 - fpr_d, tpr_d, color=ROLE_COL[role], marker=ROLE_MK[role], s=30,
                zorder=5, edgecolors='white', lw=0.5, alpha=0.9)
_cache = np.load(os.path.join(RES, 'figure_predictions_cache.npz'), allow_pickle=True)
_d = pickle.loads(_cache['binary_int_Bin-v3-ViT'])
_pred = np.asarray(_d['probs']).argmax(1); _t = np.asarray(_d['true'])
_cm = confusion_matrix(_t, _pred, labels=[0, 1])
bg_sens = _cm[1, 1] / _cm[1].sum(); bg_spec = _cm[0, 0] / _cm[0].sum()
axF.scatter(bg_spec, bg_sens, color=POS, marker='*', s=190, zorder=10,
            edgecolors='white', lw=0.8)
axF.scatter([], [], color=POS, marker='*', s=90, label='BilinGuard')
axF.plot([0.5, 1], [0.5, 1], ':', color=LGREY, lw=0.7)
axF.set_xlabel('Specificity'); axF.set_ylabel('Sensitivity')
axF.set_xlim(0.55, 1.02); axF.set_ylim(0.55, 1.02)
handles = [Line2D([0], [0], marker=ROLE_MK[r], color='w', markerfacecolor=ROLE_COL[r],
                  markersize=5, label=r) for r in ['Attending', 'Nurse', 'Resident', 'PubHealth']]
handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=POS,
                      markersize=10, label='BilinGuard'))
axF.legend(handles=handles, fontsize=5, loc='lower left', ncol=2, columnspacing=0.8,
           handletextpad=0.2, borderaxespad=0.2)
panel_letter(axF, 'F')

# ── G: Wilcoxon -log10(p) ───────────────────────────────────────────────────
axG = fig.add_subplot(gs[1, 2])
wilcoxon_d = RS['wilcoxon']
names = list(wilcoxon_d.keys()); pvals = [wilcoxon_d[r]['p_value'] for r in names]
nlp = [-np.log10(max(p, 1e-10)) for p in pvals]
ylabels = [n.replace('_', ' ') for n in names]
y = np.arange(len(names))
axG.barh(y, nlp, height=0.58, color='#B279A2', alpha=0.85, edgecolor='white', lw=0.4)
axG.set_yticks(y); axG.set_yticklabels(ylabels, fontsize=5)
axG.invert_yaxis()
for i, p in enumerate(pvals):
    axG.text(nlp[i] + 0.07, i, sig(p), va='center', fontsize=5.2)
axG.axvline(-np.log10(0.05), color=GREY, ls='--', lw=0.7)
axG.axvline(-np.log10(0.01), color=GREY, ls=':', lw=0.7)
axG.set_ylim(len(names) - 0.4, -1.25)          # inverted axis; extra top band for labels
axG.text(-np.log10(0.05), -0.9, 'p=0.05', fontsize=4.8, color=GREY, ha='center')
axG.text(-np.log10(0.01), -0.9, 'p=0.01', fontsize=4.8, color=GREY, ha='center')
axG.set_xlabel(u'\u2212log$_{10}$(p), Wilcoxon vs BilinGuard')
axG.set_xlim(0, max(nlp) * 1.3 + 0.4)
axG.set_xticks(np.arange(0, int(max(nlp)) + 1))
panel_letter(axG, 'G')

save(fig, os.path.join(OUT, 'Figure5_cdss_reader_crm'))
