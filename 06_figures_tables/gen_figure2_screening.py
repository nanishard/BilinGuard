# -*- coding: utf-8 -*-
"""
Figure 2 — Layer 1: Binary screening & domain adaptation (6 panels).

A: Internal ROC with 95% CI band (n=110, best model ViT)
B: External pure ROC multi-model (n=117)
C: Domain-adapted ROC (n=22)
D: Confusion matrices (internal + external side-by-side)
E: 6-metric radar chart (binary backbones)
F: Calibration curves + Brier score
"""
import os, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
CACHE = os.path.join(RES, 'figure_predictions_cache.npz')
FIG_DIR = os.path.join(RES, 'figures_v3', 'main')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})

# Load cache
print('Loading predictions cache...')
data = np.load(CACHE, allow_pickle=True)
cache = {k: pickle.loads(data[k]) for k in data.files}

C_INT = '#1D3557'    # dark blue
C_EXT = '#E76F51'    # coral
C_DA  = '#2A9D8F'    # teal
C_GRID = '#E5E7EB'
MODELS_INT = [('Bin-v3-ViT', C_INT, '-'), ('Bin-v3-ConvNeXt', '#457B9D', '--'), ('Bin-v3-EffNet', '#A8DADC', ':')]
MODELS_EXT = [('V3-ViT', C_EXT, '-'), ('V3-ConvNeXt', '#F4A261', '--'), ('V3-EffNet', '#E9C46A', ':')]
MODELS_DA  = [('DA-ViT', C_DA, '-'), ('DA-ConvNeXt', '#52B788', '--'), ('DA-EffNet', '#95D5B2', ':'), ('DA-Swin', '#74C69D', '-.')]


def roc_with_ci(probs, true, n_boot=1000, seed=42):
    """Compute ROC and bootstrap CI band."""
    fpr, tpr, _ = roc_curve(true, probs[:, 1])
    auc_val = auc(fpr, tpr)
    rng = np.random.RandomState(seed)
    n = len(true)
    base_fpr = np.linspace(0, 1, 100)
    tprs_boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = true[idx], probs[idx]
        if len(np.unique(yt)) < 2: continue
        f, t, _ = roc_curve(yt, yp[:, 1])
        tprs_boot.append(np.interp(base_fpr, f, t))
    tprs_boot = np.array(tprs_boot)
    lo = np.percentile(tprs_boot, 2.5, axis=0)
    hi = np.percentile(tprs_boot, 97.5, axis=0)
    return fpr, tpr, auc_val, base_fpr, lo, hi


def plot_roc_ci(ax, probs, true, color, label, n_boot=1000):
    fpr, tpr, auc_val, base_fpr, lo, hi = roc_with_ci(probs, true, n_boot)
    ax.fill_between(base_fpr, lo, hi, alpha=0.15, color=color)
    ax.plot(fpr, tpr, color=color, lw=1.5, label=f'{label} (AUC={auc_val:.3f})')
    return auc_val


# ═════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Figure 2. Layer 1 — Binary Jaundice Screening & Domain Adaptation',
             fontsize=13, fontweight='bold', y=0.98)

# Panel A: Internal ROC with CI
ax = axes[0, 0]
d = cache['binary_int_Bin-v3-ViT']
auc_val = plot_roc_ci(ax, d['probs'], d['true'], C_INT, 'ViT (best)', n_boot=500)
# Add other models without CI band
for name, color, ls in MODELS_INT[1:]:
    d2 = cache[f'binary_int_{name}']
    fpr, tpr, _ = roc_curve(d2['true'], d2['probs'][:, 1])
    ax.plot(fpr, tpr, color=color, lw=1.0, ls=ls, alpha=0.7, label=f'{name} (AUC={auc(fpr,tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(A) Internal Validation (n=110)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel B: External pure ROC
ax = axes[0, 1]
for name, color, ls in MODELS_EXT:
    key = f'binary_ext_pure_{name}'
    if key not in cache: continue
    d = cache[key]
    fpr, tpr, _ = roc_curve(d['true'], d['probs'][:, 1])
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=1.5, ls=ls, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(B) External Validation, No Adaptation (n=117)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel C: Domain-adapted ROC
ax = axes[0, 2]
for name, color, ls in MODELS_DA:
    key = f'binary_ext_da_{name}'
    if key not in cache: continue
    d = cache[key]
    fpr, tpr, _ = roc_curve(d['true'], d['probs'][:, 1])
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=1.5, ls=ls, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('(C) Domain-Adapted (n=22)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel D: Confusion matrices (internal + external)
ax = axes[1, 0]
d_int = cache['binary_int_Bin-v3-ViT']
d_ext = cache['binary_ext_pure_V3-ViT']
cm_int = confusion_matrix(d_int['true'], d_int['probs'].argmax(1), labels=[0, 1])
cm_ext = confusion_matrix(d_ext['true'], d_ext['probs'].argmax(1), labels=[0, 1])
# Plot side by side as grouped bars
categories = ['TN', 'FP', 'FN', 'TP']
int_vals = [cm_int[0,0], cm_int[0,1], cm_int[1,0], cm_int[1,1]]
ext_vals = [cm_ext[0,0], cm_ext[0,1], cm_ext[1,0], cm_ext[1,1]]
x = np.arange(len(categories))
w = 0.35
ax.bar(x - w/2, int_vals, w, color=C_INT, alpha=0.8, label=f'Internal (n={len(d_int["true"])})')
ax.bar(x + w/2, ext_vals, w, color=C_EXT, alpha=0.8, label=f'External (n={len(d_ext["true"])})')
ax.set_xticks(x); ax.set_xticklabels(categories)
ax.set_ylabel('Count'); ax.set_title('(D) Confusion Matrix Comparison', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)
for i, (iv, ev) in enumerate(zip(int_vals, ext_vals)):
    ax.text(i - w/2, iv + 0.5, str(iv), ha='center', fontsize=7)
    ax.text(i + w/2, ev + 0.5, str(ev), ha='center', fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# Panel E: 6-metric radar chart
ax = axes[1, 1]
ax.set_axis_off()
# Create inset polar plot
ax_polar = fig.add_subplot(2, 3, 5, polar=True)
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score
metrics_names = ['AUC', 'F1', 'Sens', 'Spec', 'Acc', 'AP']
N = len(metrics_names)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for name, color, ls in MODELS_INT:
    key = f'binary_int_{name}'
    if key not in cache: continue
    d = cache[key]
    probs, true = d['probs'], d['true']
    pred = probs.argmax(1)
    try:
        auc_v = auc(*roc_curve(true, probs[:, 1])[:2])
        ap_v = average_precision_score(true, probs[:, 1])
    except: auc_v, ap_v = 0, 0
    f1 = f1_score(true, pred)
    sens = recall_score(true, pred)  # sensitivity for positive class
    cm = confusion_matrix(true, pred, labels=[0, 1])
    spec = cm[0, 0] / max(cm[0].sum(), 1)
    acc = accuracy_score(true, pred)
    vals = [auc_v, f1, sens, spec, acc, ap_v]
    vals += vals[:1]
    ax_polar.plot(angles, vals, color=color, lw=1.5, ls=ls, label=name)
    ax_polar.fill(angles, vals, color=color, alpha=0.05)
ax_polar.set_xticks(angles[:-1])
ax_polar.set_xticklabels(metrics_names, fontsize=7)
ax_polar.set_ylim(0, 1.05)
ax_polar.set_title('(E) 6-Metric Comparison (Internal)', fontsize=10, fontweight='bold', pad=15)
ax_polar.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax_polar.grid(True, alpha=0.3)

# Panel F: Calibration curves
ax = axes[1, 2]
for name, color, ls, key_suffix in [('ViT', C_INT, '-', 'Bin-v3-ViT'),
                                      ('ConvNeXt', '#457B9D', '--', 'Bin-v3-ConvNeXt'),
                                      ('EffNet', '#A8DADC', ':', 'Bin-v3-EffNet')]:
    key = f'binary_int_{key_suffix}'
    if key not in cache: continue
    d = cache[key]
    probs, true = d['probs'], d['true']
    try:
        frac_pos, mean_pred = calibration_curve(true, probs[:, 1], n_bins=10, strategy='uniform')
        brier = brier_score_loss(true, probs[:, 1])
        ax.plot(mean_pred, frac_pos, color=color, lw=1.5, ls=ls, marker='o', ms=3,
                label=f'{name} (Brier={brier:.3f})')
    except Exception:
        pass
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3, label='Perfect')
ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
ax.set_title('(F) Calibration Curves (Internal)', fontsize=10, fontweight='bold')
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.legend(fontsize=6, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG_DIR, 'Figure2_binary_screening.png'))
fig.savefig(os.path.join(FIG_DIR, 'Figure2_binary_screening.svg'))
plt.close(fig)
print(f'Saved: {FIG_DIR}/Figure2_binary_screening.png/svg')
