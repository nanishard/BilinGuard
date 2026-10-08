# -*- coding: utf-8 -*-
"""
Figure 3 — Layer 1: Multiclass grading & jaundice type (9 panels, v3).

Row 1 — Grading ROC + Internal Validation CI bands (bootstrap, n=1000):
  A: TBIL facial ternary ROC + 95% CI
  B: TBIL eyelid ternary ROC + 95% CI
  C: DBIL & IBIL ternary ROC + 95% CI
Row 2 — Per-class detail (F cell split into ROC + confusion matrix):
  D: Per-class TBIL facial OvR ROC + 95% CI
  E: Precision-Recall curves (all grading tasks)
  F: Jaundice type ROC | Confusion matrix
Row 3 — Internal Validation & Calibration:
  G: Internal Validation forest plot — macro AUC + 95% CI (all grading tasks)
  H: Calibration curves (top-label reliability) + Brier score
  I: Internal Validation — bootstrap AUC distributions (box per task)
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, brier_score_loss)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
CACHE = os.path.join(RES, 'figure_predictions_cache.npz')
FIG_DIR = os.path.join(RES, 'figures_v3', 'main')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})

data = np.load(CACHE, allow_pickle=True)
cache = {k: pickle.loads(data[k]) for k in data.files}

C_DBIL = '#0369A1'; C_IBIL = '#7C3AED'; C_TYPE = '#E67E22'
DASH = '--'; DOT = ':'; SOLID = '-'
N_BOOT = 1000


# ── Helpers ──────────────────────────────────────────────────
def macro_ovr_roc(probs, true, nc=3):
    yb = label_binarize(true, classes=list(range(nc)))
    all_fpr = np.linspace(0, 1, 100); mean_tpr = np.zeros_like(all_fpr)
    for c in range(nc):
        fpr_c, tpr_c, _ = roc_curve(yb[:, c], probs[:, c])
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= nc
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def macro_ovr_roc_ci(probs, true, nc=3, n_boot=N_BOOT, seed=42):
    base_fpr, mean_tpr, auc_val = macro_ovr_roc(probs, true, nc)
    rng = np.random.RandomState(seed); n = len(true)
    tprs_boot, aucs_boot = [], []
    yb = label_binarize(true, classes=list(range(nc)))
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(true[idx])) < 2:
            continue
        yb_b = yb[idx]; mb = np.zeros_like(base_fpr); ok = True
        for c in range(nc):
            if len(np.unique(yb_b[:, c])) < 2:
                ok = False; break
            f, t, _ = roc_curve(yb_b[:, c], probs[idx, c])
            mb += np.interp(base_fpr, f, t)
        if not ok:
            continue
        mb /= nc
        tprs_boot.append(mb); aucs_boot.append(auc(base_fpr, mb))
    tprs_boot = np.array(tprs_boot); aucs_boot = np.array(aucs_boot)
    lo = np.percentile(tprs_boot, 2.5, axis=0); hi = np.percentile(tprs_boot, 97.5, axis=0)
    auc_lo = np.percentile(aucs_boot, 2.5); auc_hi = np.percentile(aucs_boot, 97.5)
    return base_fpr, mean_tpr, auc_val, lo, hi, aucs_boot, auc_lo, auc_hi


def bin_roc_ci(probs, true, n_boot=N_BOOT, seed=42):
    base_fpr = np.linspace(0, 1, 100)
    fpr, tpr, _ = roc_curve(true, probs[:, 1]); auc_val = auc(fpr, tpr)
    rng = np.random.RandomState(seed); n = len(true)
    tprs_b, aucs_b = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(true[idx])) < 2:
            continue
        f, t, _ = roc_curve(true[idx], probs[idx, 1])
        t_int = np.interp(base_fpr, f, t)
        tprs_b.append(t_int); aucs_b.append(auc(base_fpr, t_int))
    tprs_b = np.array(tprs_b); aucs_b = np.array(aucs_b)
    lo = np.percentile(tprs_b, 2.5, axis=0); hi = np.percentile(tprs_b, 97.5, axis=0)
    return base_fpr, np.interp(base_fpr, fpr, tpr), auc_val, lo, hi, aucs_b, \
        np.percentile(aucs_b, 2.5), np.percentile(aucs_b, 97.5)


def top_label_calibration(probs, true, n_bins=10):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, conf, n_bins=n_bins, strategy='uniform')
    brier = brier_score_loss(correct, conf)
    return mean_pred, frac_pos, brier


# ══ Build figure with GridSpec (no overlapping subplots) ═════
fig = plt.figure(figsize=(16, 16))
gs = GridSpec(3, 6, figure=fig, hspace=0.40, wspace=0.45,
              left=0.05, right=0.97, top=0.93, bottom=0.04)
axA = fig.add_subplot(gs[0, 0:2])
axB = fig.add_subplot(gs[0, 2:4])
axC = fig.add_subplot(gs[0, 4:6])
axD = fig.add_subplot(gs[1, 0:2])
axE = fig.add_subplot(gs[1, 2:4])
axF_roc = fig.add_subplot(gs[1, 4])    # F cell — left half
axF_cm = fig.add_subplot(gs[1, 5])     # F cell — right half
axG = fig.add_subplot(gs[2, 0:2])
axH = fig.add_subplot(gs[2, 2:4])
axI = fig.add_subplot(gs[2, 4:6])

fig.suptitle('Figure 3. Layer 1 — Multiclass Severity Grading & Jaundice-Type Classification\n'
             '(Internal Validation: 1000-iteration bootstrap, 95% CI)',
             fontsize=13, fontweight='bold', y=0.98)

# Internal-validation registry (used by panels G, H, I)
IV = {}
def register_iv(label, key, nc, color):
    if key not in cache:
        return None
    d = cache[key]; probs, true = d['probs'], d['true']
    if nc == 2:
        fpr, tpr, auc_val, lo, hi, aucs_b, auc_lo, auc_hi = bin_roc_ci(probs, true)
    else:
        fpr, tpr, auc_val, lo, hi, aucs_b, auc_lo, auc_hi = macro_ovr_roc_ci(probs, true, nc)
    IV[label] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_val, 'lo': lo, 'hi': hi,
                 'aucs': aucs_b, 'auc_lo': auc_lo, 'auc_hi': auc_hi,
                 'probs': probs, 'true': true, 'color': color}
    return IV[label]


# ── Panel A: TBIL facial ternary ROC + 95% CI (Internal Validation) ──
ax = axA
for name, key, color, ls in [('ConvNeXt', 'tbil_face_FaceTern-ConvNeXt', '#1D3557', SOLID),
                              ('ViT', 'tbil_face_FaceTern-ViT', '#457B9D', DASH),
                              ('Swin', 'tbil_face_FaceTern-Swin', '#A8DADC', DOT)]:
    if key not in cache:
        continue
    d = cache[key]
    fpr, tpr, auc_val, lo, hi, _, auc_lo, auc_hi = macro_ovr_roc_ci(d['probs'], d['true'], 3)
    if name == 'ConvNeXt':
        ax.fill_between(fpr, lo, hi, alpha=0.15, color=color)
        ax.plot(fpr, tpr, color=color, lw=1.8, ls=ls,
                label=f'{name} (AUC={auc_val:.3f}, 95%CI {auc_lo:.3f}-{auc_hi:.3f})')
    else:
        ax.plot(fpr, tpr, color=color, lw=1.5, ls=ls, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(A) TBIL Facial Ternary — Internal Validation (n=40)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right'); ax.grid(True, alpha=0.3)

# ── Panel B: TBIL eyelid ternary ROC + 95% CI ───────────────
ax = axB
for name, key, color, ls in [('ConvNeXt', 'tbil_eye_EyeTBIL-ConvNeXt', '#2A9D8F', SOLID),
                              ('ViT', 'tbil_eye_EyeTBIL-ViT', '#52B788', DASH)]:
    if key not in cache:
        continue
    d = cache[key]
    fpr, tpr, auc_val, lo, hi, _, auc_lo, auc_hi = macro_ovr_roc_ci(d['probs'], d['true'], 3)
    if name == 'ConvNeXt':
        ax.fill_between(fpr, lo, hi, alpha=0.15, color=color)
        ax.plot(fpr, tpr, color=color, lw=1.8, ls=ls,
                label=f'{name} (AUC={auc_val:.3f}, 95%CI {auc_lo:.3f}-{auc_hi:.3f})')
    else:
        ax.plot(fpr, tpr, color=color, lw=1.5, ls=ls, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(B) TBIL Eyelid Ternary — Internal Validation (n=39)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right'); ax.grid(True, alpha=0.3)

# ── Panel C: DBIL & IBIL ROC + 95% CI ───────────────────────
ax = axC
if 'dbil_eye_ConvNeXt' in cache:
    d = cache['dbil_eye_ConvNeXt']
    fpr, tpr, auc_val, lo, hi, _, auc_lo, auc_hi = macro_ovr_roc_ci(d['probs'], d['true'], 3)
    ax.fill_between(fpr, lo, hi, alpha=0.15, color=C_DBIL)
    ax.plot(fpr, tpr, color=C_DBIL, lw=1.8,
            label=f'DBIL-Eyelid (AUC={auc_val:.3f}, 95%CI {auc_lo:.3f}-{auc_hi:.3f})')
if 'ibil_face_Swin' in cache:
    d = cache['ibil_face_Swin']
    fpr, tpr, auc_val, lo, hi, _, auc_lo, auc_hi = macro_ovr_roc_ci(d['probs'], d['true'], 3)
    ax.fill_between(fpr, lo, hi, alpha=0.12, color=C_IBIL)
    ax.plot(fpr, tpr, color=C_IBIL, lw=1.8, ls=DASH,
            label=f'IBIL-Face (AUC={auc_val:.3f}, 95%CI {auc_lo:.3f}-{auc_hi:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(C) DBIL (n=52) & IBIL (n=55) — Internal Validation', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right'); ax.grid(True, alpha=0.3)

# ── Panel D: Per-class TBIL facial ROC with CI ──────────────
ax = axD
d = cache['tbil_face_FaceTern-ConvNeXt']
probs, true = d['probs'], d['true']
class_names = ['Mild', 'Moderate', 'Severe']
class_colors = ['#F4A261', '#E76F51', '#DC2626']
yb = label_binarize(true, classes=[0, 1, 2])
for c, (cname, cc) in enumerate(zip(class_names, class_colors)):
    fpr, tpr, _ = roc_curve(yb[:, c], probs[:, c])
    auc_val = auc(fpr, tpr)
    rng = np.random.RandomState(42 + c); n = len(true); base_fpr = np.linspace(0, 1, 100)
    tprs_boot = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(yb[idx, c])) < 2:
            continue
        f, t, _ = roc_curve(yb[idx, c], probs[idx, c])
        tprs_boot.append(np.interp(base_fpr, f, t))
    if tprs_boot:
        tprs_boot = np.array(tprs_boot)
        lo = np.percentile(tprs_boot, 2.5, axis=0); hi = np.percentile(tprs_boot, 97.5, axis=0)
        ax.fill_between(base_fpr, lo, hi, alpha=0.12, color=cc)
    ax.plot(fpr, tpr, color=cc, lw=1.5, label=f'{cname} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(D) TBIL Facial Per-Class OvR ROC + 95% CI', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=7, loc='lower right'); ax.grid(True, alpha=0.3)

# ── Panel E: Precision-Recall curves (best of each task) ─────
ax = axE
tasks_pr = [
    ('TBIL-Face', 'tbil_face_FaceTern-ConvNeXt', '#1D3557', SOLID, 3),
    ('TBIL-Eye', 'tbil_eye_EyeTBIL-ConvNeXt', '#2A9D8F', DASH, 3),
    ('DBIL-Eye', 'dbil_eye_ConvNeXt', C_DBIL, SOLID, 3),
    ('IBIL-Face', 'ibil_face_Swin', C_IBIL, DASH, 3),
    ('Type', 'type_Type-EffNet', C_TYPE, SOLID, 2),
]
for name, key, color, ls, nc in tasks_pr:
    if key not in cache:
        continue
    d = cache[key]; probs, true = d['probs'], d['true']
    if nc == 2:
        prec, rec, _ = precision_recall_curve(true, probs[:, 1])
        ap = average_precision_score(true, probs[:, 1])
        ax.plot(rec, prec, color=color, lw=1.5, ls=ls, label=f'{name} (AP={ap:.3f})')
    else:
        yb = label_binarize(true, classes=list(range(nc)))
        aps = []; base_rec = np.linspace(0, 1, 100); mean_prec = np.zeros_like(base_rec)
        for c in range(nc):
            p, r, _ = precision_recall_curve(yb[:, c], probs[:, c])
            aps.append(average_precision_score(yb[:, c], probs[:, c]))
            mean_prec += np.interp(base_rec[::-1], r[::-1], p[::-1])[::-1]
        mean_prec /= nc; ap_macro = np.mean(aps)
        ax.plot(base_rec, mean_prec, color=color, lw=1.5, ls=ls, label=f'{name} (mAP={ap_macro:.3f})')
ax.set_xlabel('Recall (Sensitivity)'); ax.set_ylabel('Precision (PPV)')
ax.set_title('(E) Precision-Recall Curves', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([0, 1.05])
ax.legend(fontsize=6, loc='lower left'); ax.grid(True, alpha=0.3)

# ── Panel F: Jaundice type ROC (left) + confusion matrix (right) ──
d = cache['type_Type-EffNet']
# F-left: ROC
ax = axF_roc
fpr, tpr, _ = roc_curve(d['true'], d['probs'][:, 1])
auc_val = auc(fpr, tpr)
ax.plot(fpr, tpr, color=C_TYPE, lw=1.8, label=f'AUC={auc_val:.3f}')
ax.fill_between(fpr, tpr, alpha=0.1, color=C_TYPE)
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('FPR', fontsize=8); ax.set_ylabel('TPR', fontsize=8)
ax.set_title('Jaundice Type ROC (n=39)', fontsize=9, fontweight='bold')
ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc='lower right')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02]); ax.grid(True, alpha=0.3)
# F-right: confusion matrix
ax = axF_cm
cm = confusion_matrix(d['true'], d['probs'].argmax(1), labels=[0, 1])
ax.imshow(cm, cmap='Oranges', aspect='auto')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Hepato', 'Chol'], fontsize=7, rotation=45)
ax.set_yticklabels(['Hepato', 'Chol'], fontsize=7)
ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('True', fontsize=8)
ax.set_title('Confusion Matrix', fontsize=9, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=11,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
# Panel-F super-title spanning the two F cells
fig.text(0.5 * (axF_roc.get_position().x0 + axF_cm.get_position().x1),
         axF_roc.get_position().y1 + 0.012, '(F) Jaundice-Type Classification',
         ha='center', fontsize=10, fontweight='bold')

# ══ Register internal-validation results ════════════════════
register_iv('TBIL-Face', 'tbil_face_FaceTern-ConvNeXt', 3, '#1D3557')
register_iv('TBIL-Eye', 'tbil_eye_EyeTBIL-ConvNeXt', 3, '#2A9D8F')
register_iv('DBIL-Eye', 'dbil_eye_ConvNeXt', 3, C_DBIL)
register_iv('IBIL-Face', 'ibil_face_Swin', 3, C_IBIL)
register_iv('Jaundice-Type', 'type_Type-EffNet', 2, C_TYPE)

# ── Panel G: Internal Validation forest plot (AUC + 95% CI) ──
ax = axG
order = list(IV.keys())
ypos = np.arange(len(order))[::-1]
for y, name in zip(ypos, order):
    r = IV[name]
    ax.errorbar(r['auc'], y,
                xerr=[[r['auc'] - r['auc_lo']], [r['auc_hi'] - r['auc']]],
                fmt='o', color=r['color'], ecolor=r['color'], elinewidth=1.5,
                capsize=4, markersize=8)
    ax.text(r['auc_hi'] + 0.015, y,
            f"{r['auc']:.3f} [{r['auc_lo']:.3f}-{r['auc_hi']:.3f}]",
            va='center', fontsize=6.5, color=r['color'])
ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=8)
ax.set_xlim(0.5, 1.08); ax.axvline(0.5, color='gray', ls=':', lw=0.6)
ax.set_xlabel('Macro AUC (OvR)')
ax.set_title('(G) Internal Validation — AUC Forest Plot (1000x bootstrap)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# ── Panel H: Calibration curves + Brier ─────────────────────
ax = axH
for name in order:
    r = IV[name]
    try:
        mp, fp, brier = top_label_calibration(r['probs'], r['true'])
        ax.plot(mp, fp, color=r['color'], lw=1.5, marker='o', ms=3,
                label=f"{name} (Brier={brier:.3f})")
    except Exception:
        pass
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.4, label='Perfect')
ax.set_xlabel('Mean Predicted Confidence (top-label)')
ax.set_ylabel('Empirical Accuracy (fraction correct)')
ax.set_title('(H) Calibration Curves — Internal Validation', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
ax.legend(fontsize=6, loc='upper left'); ax.grid(True, alpha=0.3)

# ── Panel I: Bootstrap AUC distributions (box per task) ─────
ax = axI
positions = np.arange(len(order))
box_data = [IV[name]['aucs'] for name in order]
colors = [IV[name]['color'] for name in order]
bp = ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                showfliers=False, medianprops=dict(color='white', lw=1.2),
                whiskerprops=dict(lw=1), capprops=dict(lw=1))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for y, name in zip(positions, order):
    ax.scatter(y, IV[name]['auc'], color='black', zorder=5, s=18)
ax.set_xticks(positions); ax.set_xticklabels(order, fontsize=7, rotation=20, ha='right')
ax.axhline(0.5, color='gray', ls=':', lw=0.6)
ax.set_ylabel('Bootstrap AUC distribution')
ax.set_title('(I) Internal Validation — Bootstrap AUC Stability',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

fig.savefig(os.path.join(FIG_DIR, 'Figure3_grading_v2.png'))
fig.savefig(os.path.join(FIG_DIR, 'Figure3_grading_v2.svg'))
plt.close(fig)
print(f'Saved: {FIG_DIR}/Figure3_grading_v2.png/svg')
print('\nInternal-validation summary (AUC [95% CI]):')
for name in order:
    r = IV[name]
    print(f"  {name:14s}: {r['auc']:.3f} [{r['auc_lo']:.3f}-{r['auc_hi']:.3f}]  (n={len(r['true'])})")
