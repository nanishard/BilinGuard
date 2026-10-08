# -*- coding: utf-8 -*-
"""
Figure 4 — Layer 2: Non-invasive prognostic scoring (CORE INNOVATION) (6 panels).

A: Child-Pugh grade (A/B/C) ROC with 95% CI band — ConvNeXt
B: MELD risk (Low/Med/High) ROC with 95% CI band — ViT
C: Per-class OvR ROC for Child-Pugh (A vs B vs C)
D: Per-class OvR ROC for MELD (Low vs Med vs High)
E: Calibration curves for CP and MELD
F: Model comparison bar chart (CP + MELD all backbones)
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
CACHE = os.path.join(RES, 'figure_predictions_cache.npz')
FIG_DIR = os.path.join(RES, 'figures_v3', 'main')

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})

data = np.load(CACHE, allow_pickle=True)
cache = {k: pickle.loads(data[k]) for k in data.files}

C_CP = '#D97706'   # amber for Child-Pugh
C_MELD = '#DC2626' # red for MELD


def macro_ovr_roc_ci(probs, true, nc=3, n_boot=500, seed=42):
    """Macro OvR ROC with bootstrap CI band."""
    yb = label_binarize(true, classes=list(range(nc)))
    base_fpr = np.linspace(0, 1, 100); mean_tpr = np.zeros_like(base_fpr)
    for c in range(nc):
        fpr_c, tpr_c, _ = roc_curve(yb[:, c], probs[:, c])
        mean_tpr += np.interp(base_fpr, fpr_c, tpr_c)
    mean_tpr /= nc
    auc_val = auc(base_fpr, mean_tpr)
    # Bootstrap CI
    rng = np.random.RandomState(seed); n = len(true)
    tprs_boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(true[idx])) < 2: continue
        mt = np.zeros_like(base_fpr)
        for c in range(nc):
            f, t, _ = roc_curve(yb[idx, c], probs[idx, c])
            mt += np.interp(base_fpr, f, t)
        mt /= nc
        tprs_boot.append(mt)
    if tprs_boot:
        tprs_boot = np.array(tprs_boot)
        lo = np.percentile(tprs_boot, 2.5, axis=0); hi = np.percentile(tprs_boot, 97.5, axis=0)
    else:
        lo = hi = mean_tpr
    return base_fpr, mean_tpr, auc_val, lo, hi


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Figure 4. Layer 2 — Non-Invasive Prediction of Child-Pugh and MELD from Facial Images',
             fontsize=13, fontweight='bold', y=0.98)

# ── Panel A: Child-Pugh ROC with CI ───────────────────────
ax = axes[0, 0]
for name, key, color, ls in [('ConvNeXt', 'cp_CP-ConvNeXt', C_CP, '-'),
                               ('ViT', 'cp_CP-ViT', '#F59E0B', '--')]:
    if key not in cache: continue
    d = cache[key]
    fpr, tpr, auc_val, lo, hi = macro_ovr_roc_ci(d['probs'], d['true'])
    ax.fill_between(fpr, lo, hi, alpha=0.12, color=color)
    ax.plot(fpr, tpr, color=color, lw=1.8, ls=ls, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(A) Child-Pugh A/B/C Prediction (n=110)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=7, loc='lower right'); ax.grid(True, alpha=0.3)
# Annotate significance
ax.text(0.98, 0.02, 'No blood draw\nrequired', transform=ax.transAxes,
        fontsize=7, ha='right', va='bottom', style='italic', color=C_CP,
        bbox=dict(boxstyle='round', facecolor='#FEF3C7', alpha=0.8))

# ── Panel B: MELD ROC with CI ─────────────────────────────
ax = axes[0, 1]
for name, key, color, ls in [('ViT', 'meld_MELD-ViT', C_MELD, '-'),
                               ('ConvNeXt', 'meld_MELD-ConvNeXt', '#EF4444', '--')]:
    if key not in cache: continue
    d = cache[key]
    fpr, tpr, auc_val, lo, hi = macro_ovr_roc_ci(d['probs'], d['true'])
    ax.fill_between(fpr, lo, hi, alpha=0.12, color=color)
    ax.plot(fpr, tpr, color=color, lw=1.8, ls=ls, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(B) MELD Risk Low/Med/High (n=110)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=7, loc='lower right'); ax.grid(True, alpha=0.3)
ax.text(0.98, 0.02, 'No blood draw\nrequired', transform=ax.transAxes,
        fontsize=7, ha='right', va='bottom', style='italic', color=C_MELD,
        bbox=dict(boxstyle='round', facecolor='#FEE2E2', alpha=0.8))

# ── Panel C: Child-Pugh per-class OvR ROC ─────────────────
ax = axes[0, 2]
d = cache['cp_CP-ConvNeXt']
probs, true = d['probs'], d['true']
class_names = ['A (Compensated)', 'B (Decompensated)', 'C (Severe)']
class_colors = ['#FCD34D', '#F59E0B', '#B45309']
yb = label_binarize(true, classes=[0, 1, 2])
for c, (cname, cc) in enumerate(zip(class_names, class_colors)):
    fpr_c, tpr_c, _ = roc_curve(yb[:, c], probs[:, c])
    auc_c = auc(fpr_c, tpr_c)
    ax.plot(fpr_c, tpr_c, color=cc, lw=1.5, label=f'{cname} (AUC={auc_c:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(C) Child-Pugh Per-Class ROC', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right'); ax.grid(True, alpha=0.3)

# ── Panel D: MELD per-class OvR ROC ───────────────────────
ax = axes[1, 0]
d = cache['meld_MELD-ViT']
probs, true = d['probs'], d['true']
class_names = ['Low (<=20)', 'Medium (21-30)', 'High (>30)']
class_colors = ['#FCA5A5', '#EF4444', '#7F1D1D']
yb = label_binarize(true, classes=[0, 1, 2])
for c, (cname, cc) in enumerate(zip(class_names, class_colors)):
    fpr_c, tpr_c, _ = roc_curve(yb[:, c], probs[:, c])
    auc_c = auc(fpr_c, tpr_c)
    ax.plot(fpr_c, tpr_c, color=cc, lw=1.5, label=f'{cname} (AUC={auc_c:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(D) MELD Per-Class ROC', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right'); ax.grid(True, alpha=0.3)

# ── Panel E: Calibration curves ───────────────────────────
ax = axes[1, 1]
# Child-Pugh: use predicted class probability as confidence
for name, key, color, marker in [('Child-Pugh', 'cp_CP-ConvNeXt', C_CP, 'o'),
                                   ('MELD', 'meld_MELD-ViT', C_MELD, 's')]:
    if key not in cache: continue
    d = cache[key]
    probs, true = d['probs'], d['true']
    # For multiclass calibration: use max probability vs correctness
    max_probs = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true).astype(int)
    try:
        frac_pos, mean_pred = calibration_curve(correct, max_probs, n_bins=8, strategy='quantile')
        ax.plot(mean_pred, frac_pos, color=color, lw=1.5, marker=marker, ms=4, label=name)
    except Exception:
        pass
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3, label='Perfect')
ax.set_xlabel('Mean Predicted Confidence (max softmax)')
ax.set_ylabel('Fraction Correctly Classified')
ax.set_title('(E) Calibration Curves', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.3)

# ── Panel F: Confusion matrices (CP and MELD side by side) ─
ax = axes[1, 2]
ax.set_axis_off()
# Child-Pugh CM
ax_cp = fig.add_axes([0.69, 0.06, 0.13, 0.36])
d = cache['cp_CP-ConvNeXt']
cm = confusion_matrix(d['true'], d['probs'].argmax(1), labels=[0, 1, 2])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax_cp.imshow(cm_pct, cmap='Oranges', aspect='auto', vmin=0, vmax=100)
ax_cp.set_xticks([0, 1, 2]); ax_cp.set_yticks([0, 1, 2])
ax_cp.set_xticklabels(['A', 'B', 'C'], fontsize=7)
ax_cp.set_yticklabels(['A', 'B', 'C'], fontsize=7)
ax_cp.set_xlabel('Predicted', fontsize=7); ax_cp.set_ylabel('True', fontsize=7)
ax_cp.set_title('Child-Pugh CM', fontsize=8, fontweight='bold')
for i in range(3):
    for j in range(3):
        val = cm[i, j]; pct = cm_pct[i, j]
        color = 'white' if pct > 50 else 'black'
        ax_cp.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center', fontsize=6, color=color)

# MELD CM
ax_meld = fig.add_axes([0.85, 0.06, 0.13, 0.36])
d = cache['meld_MELD-ViT']
cm = confusion_matrix(d['true'], d['probs'].argmax(1), labels=[0, 1, 2])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax_meld.imshow(cm_pct, cmap='Reds', aspect='auto', vmin=0, vmax=100)
ax_meld.set_xticks([0, 1, 2]); ax_meld.set_yticks([0, 1, 2])
ax_meld.set_xticklabels(['Low', 'Med', 'High'], fontsize=7)
ax_meld.set_yticklabels(['Low', 'Med', 'High'], fontsize=7)
ax_meld.set_xlabel('Predicted', fontsize=7); ax_meld.set_ylabel('True', fontsize=7)
ax_meld.set_title('MELD CM', fontsize=8, fontweight='bold')
for i in range(3):
    for j in range(3):
        val = cm[i, j]; pct = cm_pct[i, j]
        color = 'white' if pct > 50 else 'black'
        ax_meld.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center', fontsize=6, color=color)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG_DIR, 'Figure4_prognostic_scoring.png'))
fig.savefig(os.path.join(FIG_DIR, 'Figure4_prognostic_scoring.svg'))
plt.close(fig)
print(f'Saved: {FIG_DIR}/Figure4_prognostic_scoring.png/svg')
