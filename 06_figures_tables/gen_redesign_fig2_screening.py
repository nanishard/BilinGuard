# -*- coding: utf-8 -*-
"""
Redesigned Figure 2 (CRM/3D-FAICE style): binary jaundice screening performance.
v3 layout: two independent gridspecs with generous gaps (labels never touch neighbors).
  gsA (rows 1-2, 3 cols): A internal ROC | B external ROC | C PR
  _                       D reader forest | E calibration   | F confusion matrix
  gsB (row 3, 4 cols):    G prob dist int | H prob dist ext | I threshold metrics | J DCA
Frozen data only.
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, brier_score_loss, roc_auc_score)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_crm import (set_rc, panel_letter, roc_ax, save, FPR_GRID, SEED,
                          INT, EXT, POS, GREY, LGREY, FUS, READER, sig, roc_band_arrays)

set_rc()
BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
os.makedirs(OUT, exist_ok=True)

V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
EXTJ = json.load(open(os.path.join(RES, 'external_validation_frozen.json'), encoding='utf-8'))
RSF = json.load(open(os.path.join(RES, 'reader_study_final.json'), encoding='utf-8'))
RS = json.load(open(os.path.join(RES, 'reader_study_results.json'), encoding='utf-8'))

_bipd = V2['Binary Screening']['face']['swin']
t_i = np.array([v[1] for v in _bipd.values()]); p_i = np.array([float(v[0][1]) for v in _bipd.values()])
t_e = np.array([v['label'] for v in EXTJ['patient_probs'].values()])
p_e = np.array([v['prob'] for v in EXTJ['patient_probs'].values()])
FRZ_BIN_INT = (0.956, 0.918, 0.987)
FRZ_BIN_EXT = (EXTJ['auc'], EXTJ['auc_lo'], EXTJ['auc_hi'])

fig = plt.figure(figsize=(6.9, 6.9))
gsA = fig.add_gridspec(2, 3, left=0.10, right=0.975, top=0.975, bottom=0.38,
                      hspace=0.62, wspace=0.38)
gsB = fig.add_gridspec(1, 4, left=0.10, right=0.975, top=0.26, bottom=0.075,
                      wspace=0.45)

# ── A: internal ROC ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gsA[0, 0])
fpr, tpr, a_i, lo, hi, ci_i = roc_band_arrays(t_i, p_i)
ax.plot(fpr, tpr, '-', color=INT, lw=1.6,
        label=f'AUC {FRZ_BIN_INT[0]:.3f} ({FRZ_BIN_INT[1]:.3f}\u2013{FRZ_BIN_INT[2]:.3f})')
ax.fill_between(FPR_GRID, lo, hi, color=INT, alpha=0.15, lw=0)
roc_ax(ax); ax.legend(loc='lower right')
ax.text(0.03, 0.94, f'n = {len(t_i)}', transform=ax.transAxes, fontsize=6, color=GREY)
panel_letter(ax, 'A')

# ── B: external ROC ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gsA[0, 1])
fpr_e, tpr_e, a_e, lo_e, hi_e, ci_e = roc_band_arrays(t_e, p_e)
ax.plot(fpr_e, tpr_e, '-', color=EXT, lw=1.6,
        label=f'AUC {FRZ_BIN_EXT[0]:.3f} ({FRZ_BIN_EXT[1]:.3f}\u2013{FRZ_BIN_EXT[2]:.3f})')
ax.fill_between(FPR_GRID, lo_e, hi_e, color=EXT, alpha=0.15, lw=0)
roc_ax(ax); ax.legend(loc='lower right')
ax.set_ylabel('')
ax.text(0.03, 0.94, f'n = {len(t_e)} (exploratory)', transform=ax.transAxes, fontsize=6,
        color=GREY)
panel_letter(ax, 'B')

# ── C: PR curves ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gsA[0, 2])
for t, p, lab, col in [(t_i, p_i, 'Internal', INT), (t_e, p_e, 'External', EXT)]:
    prec, rec, _ = precision_recall_curve(t, p)
    ax.plot(rec, prec, '-', color=col, lw=1.2,
            label=f'{lab} (AP {average_precision_score(t, p):.3f})')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(loc='lower left', fontsize=5.5)
panel_letter(ax, 'C')

# ── D: reader forest (short canonical labels, values consistent with Table 4) ─
ax = fig.add_subplot(gsA[1, 0])
SHORT = {'Attending Surgeon 1': 'Attending 1', 'Attending Surgeon 2': 'Attending 2',
         'Senior ID Nurse': 'Senior nurse', 'Resident 1': 'Resident 1',
         'Resident 2': 'Resident 2', 'Resident 3': 'Resident 3',
         'Public Health 1': 'PubHealth 1', 'Public Health 2': 'PubHealth 2'}
T4 = pd.read_csv(os.path.join(RES, 'BilinGuard_Lancet_v2_20260725',
                              '03_Tables', 'Table4_reader_study.csv'))
t4_map = [(float(r['Binary AUC']), SHORT.get(str(r['Rater']), str(r['Rater'])))
          for _, r in T4.iterrows() if 'mean' not in str(r['Rater']) and 'BilinGuard' not in str(r['Rater'])]
rows = []
for r in RS['per_rater']:
    if r['rater'] == 'BilinGuard':
        continue
    name = min(t4_map, key=lambda c: abs(c[0] - r['auc']))[1]
    rows.append((name, r['auc']))
rows.sort(key=lambda x: x[1])
_cache = np.load(os.path.join(RES, 'figure_predictions_cache.npz'), allow_pickle=True)
_d = pickle.loads(_cache['binary_int_Bin-v3-ViT'])
_t = np.asarray(_d['true']); _p = np.asarray(_d['probs'])[:, 1]
_rng = np.random.RandomState(SEED)
_aucs = []
for _ in range(500):
    _idx = _rng.randint(0, len(_t), len(_t))
    if len(np.unique(_t[_idx])) < 2:
        continue
    _aucs.append(roc_auc_score(_t[_idx], _p[_idx]))
m_auc = float(np.mean(_aucs)); m_lo, m_hi = np.percentile(_aucs, [2.5, 97.5])
hm, (hlo, hhi) = RSF['human_binary_mean'], RSF['human_binary_ci']
for i, (lab, a) in enumerate(rows, start=1):
    ax.plot(a, i, 'o', color=READER, ms=4.5, alpha=0.85, zorder=3)
    ax.text(a, i + 0.22, f'{a:.3f}', ha='center', fontsize=4.6, color=READER)
ax.errorbar(hm, len(rows) + 1.3, xerr=[[hm - hlo], [hhi - hm]], fmt='D', color='#1A1A1A',
            ms=4.5, elinewidth=1.1, capsize=2.5, zorder=4)
ax.axhline(len(rows) + 2.0, color=LGREY, lw=0.6)
ax.errorbar(m_auc, len(rows) + 2.9, xerr=[[m_auc - m_lo], [m_hi - m_auc]],
            fmt='s', color=POS, ms=5, elinewidth=1.2, capsize=2.5, zorder=4)
ax.axvspan(m_lo, m_hi, color=POS, alpha=0.08, lw=0)
labels = [r[0] for r in rows] + ['Reader mean', 'BilinGuard']
ax.set_yticks(list(range(1, len(rows) + 1)) + [len(rows) + 1.3, len(rows) + 2.9])
ax.set_yticklabels(labels, fontsize=5.4)
for tick in ax.get_yticklabels():
    if tick.get_text() in ('Reader mean', 'BilinGuard'):
        tick.set_fontweight('bold')
    if tick.get_text() == 'BilinGuard':
        tick.set_color(POS)
ax.set_ylim(0.2, len(rows) + 3.7)
ax.set_xlim(0.8, 1.02)
ax.set_xlabel('Binary AUC (95% CI for means)')
ax.text(0.02, 0.975, 'BilinGuard on the reader subset (n=110);\n'
        'full-cohort AUC 0.956 (0.918\u20130.987)',
        transform=ax.transAxes, fontsize=5, color=GREY, va='top')
panel_letter(ax, 'D')

# ── E: calibration ──────────────────────────────────────────────────────────
ax = fig.add_subplot(gsA[1, 1])
for t, p, lab, col, mk in [(t_i, p_i, 'Internal', INT, 's'),
                           (t_e, p_e, 'External', EXT, 'o')]:
    fr, mp = calibration_curve(t, p, n_bins=8, strategy='uniform')
    ax.plot(mp, fr, mk + '-', color=col, lw=1.2, ms=3.5,
            label=f'{lab} (Brier {brier_score_loss(t, p):.3f})')
ax.plot([0, 1], [0, 1], ':', color=LGREY, lw=0.7)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
ax.legend(loc='upper left', fontsize=5.5)
panel_letter(ax, 'E')

# ── F: external confusion matrix ────────────────────────────────────────────
ax = fig.add_subplot(gsA[1, 2])
cm = np.array(EXTJ['youden_cm'])
im = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0)
for r in range(2):
    for c in range(2):
        ax.text(c, r, str(cm[r][c]), ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if cm[r][c] > cm.max() * 0.6 else '#1A1A1A')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal', 'Jaundice'], fontsize=6)
ax.set_yticklabels(['Normal', 'Jaundice'], fontsize=6, rotation=90, va='center')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'External, threshold {EXTJ["frozen_threshold"]:.3f} '
             f'(sens {EXTJ["youden_sens"]:.2f}, spec {EXTJ["youden_spec"]:.2f})',
             fontsize=6, pad=2)
panel_letter(ax, 'F')

# ── G/H: probability distributions ─────────────────────────────────────────
for col, (t, p, lab, c) in enumerate([(t_i, p_i, 'Internal', INT), (t_e, p_e, 'External', EXT)]):
    ax = fig.add_subplot(gsB[0, col])
    data = [p[t == 0], p[t == 1]]
    vp = ax.violinplot(data, positions=[0, 1], widths=0.7, showmeans=True)
    for j, body in enumerate(vp['bodies']):
        body.set_facecolor(LGREY if j == 0 else c)
        body.set_alpha(0.55); body.set_edgecolor('none')
    for key in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        vp[key].set_color(GREY); vp[key].set_linewidth(0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Non-jaundiced', 'Jaundiced'], fontsize=5.2)
    ax.set_ylabel('Predicted probability' if col == 0 else '')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'{lab} (n={len(t)})', fontsize=6, pad=2)
    panel_letter(ax, 'G' if col == 0 else 'H')

# ── I: metrics vs threshold (external) ─────────────────────────────────────
ax = fig.add_subplot(gsB[0, 2])
ths = np.linspace(0.01, 0.99, 99)
sens, spec, ppv = [], [], []
for th in ths:
    pred = (p_e >= th).astype(int)
    tp = int(((pred == 1) & (t_e == 1)).sum()); fp = int(((pred == 1) & (t_e == 0)).sum())
    fn = int(((pred == 0) & (t_e == 1)).sum()); tn = int(((pred == 0) & (t_e == 0)).sum())
    sens.append(tp / max(tp + fn, 1)); spec.append(tn / max(tn + fp, 1))
    ppv.append(tp / max(tp + fp, 1))
ax.plot(ths, sens, '-', color=POS, lw=1.2, label='Sensitivity')
ax.plot(ths, spec, '-', color=INT, lw=1.2, label='Specificity')
ax.plot(ths, ppv, '--', color=FUS, lw=1.2, label='PPV')
ax.axvline(EXTJ['frozen_threshold'], color=GREY, lw=0.8, ls=':')
ax.text(EXTJ['frozen_threshold'] + 0.03, 0.06, 'frozen\nthreshold', fontsize=4.6,
        color=GREY, va='bottom')
ax.set_xlabel('Decision threshold'); ax.set_ylabel('Metric')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc='lower left', fontsize=5.2)
panel_letter(ax, 'I')

# ── J: decision-curve analysis ─────────────────────────────────────────────
ax = fig.add_subplot(gsB[0, 3])
pt = np.linspace(0.01, 0.90, 90)
nb_model, nb_all = [], []
n = len(t_e)
for x in pt:
    pred = (p_e >= x).astype(int)
    tp = ((pred == 1) & (t_e == 1)).sum() / n; fp = ((pred == 1) & (t_e == 0)).sum() / n
    nb_model.append(tp - fp * x / (1 - x))
    nb_all.append(t_e.mean() - (1 - t_e.mean()) * x / (1 - x))
ax.plot(pt, nb_model, '-', color=EXT, lw=1.4, label='BilinGuard')
ax.plot(pt, nb_all, '-', color=LGREY, lw=1.0, label='Treat all')
ax.axhline(0, color=GREY, lw=1.0)
ax.text(0.05, -0.235, 'Treat none', fontsize=5.2, color=GREY)
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.set_xlim(0, 0.9); ax.set_ylim(-0.25, max(max(nb_model), t_e.mean()) * 1.15)
ax.set_xticks(np.arange(0, 0.81, 0.2))
ax.set_yticks(np.arange(-0.2, 1.0, 0.2))
ax.legend(loc='upper right', fontsize=5.2)
panel_letter(ax, 'J')

save(fig, os.path.join(OUT, 'Figure2_binary_screening_crm'))
