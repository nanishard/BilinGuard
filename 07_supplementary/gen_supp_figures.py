# -*- coding: utf-8 -*-
"""
Comprehensive Supplementary Figures for BilinGuard manuscript.

Generates:
  SuppFig 1: Forest plot — all 67 models across 10 tasks (with real CIs)
  SuppFig 2: 3-layer summary bar chart (Layer 1=blue, Layer 2=orange, Layer 3=red)
  SuppFig 3: Domain adaptation comparison (probability distributions before/after)
  SuppFig 4: Color signal diagnostic (Lab_b, HSV distributions + Cohen's d)
  SuppFig 5: Confusion matrix grid (all 10 tasks)
  SuppFig 6: All binary models comparison (14 models radar + ROC)
"""
import os, json, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
CACHE = os.path.join(RES, 'figure_predictions_cache.npz')
FIG_DIR = os.path.join(RES, 'figures_v3', 'main')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})

# Load data
bs = pd.read_csv(os.path.join(RES, 'tables_v3', 'bootstrap_full_metrics_v3.csv'))
data = np.load(CACHE, allow_pickle=True)
cache = {k: pickle.loads(data[k]) for k in data.files}

# 3-layer color scheme
LAYER1_COLOR = '#1D3557'  # blue - recognition
LAYER2_COLOR = '#D97706'  # orange - scoring
LAYER3_COLOR = '#DC2626'  # red - decision
DA_COLOR = '#2A9D8F'      # teal - domain adaptation

TASK_LAYER = {
    'Binary-Screening': 1, 'Binary-External-Pure': 1, 'Binary-External': 1,
    'TBIL-Eyelid': 1, 'Face-Ternary': 1, 'DBIL': 1, 'IBIL': 1, 'Jaundice-Type': 1,
    'Child-Pugh': 2, 'MELD-Risk': 2,
}
TASK_DISPLAY = {
    'Binary-Screening': 'Binary Screening\n(Internal, n=110)',
    'Binary-External-Pure': 'Binary Screening\n(External Pure, n=117)',
    'Binary-External': 'Binary Screening\n(Domain-Adapted, n=19)',
    'TBIL-Eyelid': 'TBIL Grading\n(Eyelid, n=42)',
    'Face-Ternary': 'TBIL Grading\n(Face, n=40)',
    'DBIL': 'DBIL Grading\n(n=52)',
    'IBIL': 'IBIL Grading\n(n=55)',
    'Jaundice-Type': 'Jaundice Type\n(n=39)',
    'Child-Pugh': 'Child-Pugh Grade\n(n=110)',
    'MELD-Risk': 'MELD Risk Score\n(n=110)',
}
LAYER_NAMES = {1: 'Layer 1: Recognition', 2: 'Layer 2: Prognostic Scoring', 3: 'Layer 3: Decision Support'}


# ═════════════════════════════════════════════════════════════
# SuppFig 1: Comprehensive Forest Plot (all 67 models)
# ═════════════════════════════════════════════════════════════
print('[1] SuppFig 1: Comprehensive Forest Plot...')
tasks_order = ['Binary-Screening', 'Binary-External-Pure', 'Binary-External',
               'TBIL-Eyelid', 'Face-Ternary', 'DBIL', 'IBIL', 'Jaundice-Type',
               'Child-Pugh', 'MELD-Risk']

# Calculate total rows needed
total_rows = 0
task_bounds = {}
for task in tasks_order:
    sub = bs[bs['task'] == task]
    task_bounds[task] = (total_rows, total_rows + len(sub))
    total_rows += len(sub) + 1  # +1 for gap

fig, ax = plt.subplots(figsize=(14, max(20, total_rows * 0.28)))
y_pos = total_rows
for task in tasks_order:
    sub = bs[bs['task'] == task].sort_values('auc')
    layer = TASK_LAYER.get(task, 1)
    color = LAYER1_COLOR if layer == 1 else LAYER2_COLOR
    for _, r in sub.iterrows():
        ax.errorbar(r['auc'], y_pos, xerr=[[max(r['auc'] - r['auc_lo'], 0)],
                                             [max(r['auc_hi'] - r['auc'], 0)]],
                     fmt='o', ms=4, capsize=2, color=color, lw=1.0)
        ax.text(0.06, y_pos, r['name'], fontsize=5.5, va='center', ha='left',
                transform=ax.get_yaxis_transform())
        y_pos -= 1
    y_pos -= 1  # gap between tasks

ax.axvline(0.5, color='gray', ls='--', lw=0.3)
# Add task separators and labels
y_pos = total_rows
for task in tasks_order:
    sub = bs[bs['task'] == task]
    n_models = len(sub)
    mid_y = y_pos - n_models / 2
    layer = TASK_LAYER.get(task, 1)
    color = LAYER1_COLOR if layer == 1 else LAYER2_COLOR
    ax.text(1.08, mid_y, TASK_DISPLAY.get(task, task), fontsize=7, va='center',
            color=color, fontweight='bold', transform=ax.get_yaxis_transform())
    # Separator line
    if y_pos < total_rows:
        ax.axhline(y_pos + 0.5, color='#E5E7EB', lw=0.3)
    y_pos -= n_models + 1

ax.set_xlim([0.0, 1.15])
ax.set_ylim([-1, total_rows + 1])
ax.set_xlabel('AUC-ROC [95% CI]', fontsize=10)
ax.set_title('Supplementary Figure 1. Forest Plot — All 67 Models Across 10 Tasks\n'
             '(Layer 1 Recognition = blue, Layer 2 Prognostic Scoring = orange)',
             fontsize=11, fontweight='bold')
ax.set_yticks([])
ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFig1_forest_all_models.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFig1_forest_all_models.svg'))
plt.close(fig)
print('  Saved SuppFig1')


# ═════════════════════════════════════════════════════════════
# SuppFig 2: 3-Layer Summary Bar Chart
# ═════════════════════════════════════════════════════════════
print('[2] SuppFig 2: 3-Layer Summary Bar Chart...')
fig, ax = plt.subplots(figsize=(14, 7))
# Best AUC per task, colored by layer
task_best = []
for task in tasks_order:
    sub = bs[bs['task'] == task]
    if len(sub) == 0: continue
    best = sub.loc[sub['auc'].idxmax()]
    layer = TASK_LAYER.get(task, 1)
    task_best.append({'task': task, 'auc': best['auc'], 'auc_lo': best['auc_lo'],
                       'auc_hi': best['auc_hi'], 'model': best['name'],
                       'n_val': best['n_val'], 'layer': layer})

df_best = pd.DataFrame(task_best).sort_values(['layer', 'auc'])

colors = [LAYER1_COLOR if l == 1 else LAYER2_COLOR for l in df_best['layer']]
y_pos = np.arange(len(df_best))

for i, (_, r) in enumerate(df_best.iterrows()):
    color = LAYER1_COLOR if r['layer'] == 1 else LAYER2_COLOR
    ax.barh(i, r['auc'], color=color, height=0.6, edgecolor='white', lw=0.5, alpha=0.85)
    ax.errorbar(r['auc'], i, xerr=[[r['auc'] - r['auc_lo']], [r['auc_hi'] - r['auc']]],
                 fmt='none', color='black', capsize=2, lw=0.8)
    label = f'{r["auc"]:.3f} ({r["model"][:20]}, N={int(r["n_val"])})'
    ax.text(r['auc'] + 0.008, i, label, va='center', fontsize=6.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([TASK_DISPLAY.get(t, t) for t in df_best['task']], fontsize=7)
ax.set_xlabel('Best AUC-ROC [95% CI]', fontsize=10)
ax.set_xlim([0.5, 1.25])
ax.axvline(0.5, color='gray', ls='--', lw=0.3)
ax.set_title('Supplementary Figure 2. Best AUC per Task — 3-Layer Medical Closed Loop\n'
             '67 models across 10 tasks (1000-iteration bootstrap)',
             fontsize=11, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

# Layer legend
legend_elements = [Patch(facecolor=LAYER1_COLOR, alpha=0.85, label='Layer 1: Jaundice Recognition (8 tasks)'),
                    Patch(facecolor=LAYER2_COLOR, alpha=0.85, label='Layer 2: Prognostic Scoring (2 tasks)'),
                    Patch(facecolor=DA_COLOR, alpha=0.85, label='Domain Adaptation')]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFig2_summary_3layer.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFig2_summary_3layer.svg'))
plt.close(fig)
print('  Saved SuppFig2')


# ═════════════════════════════════════════════════════════════
# SuppFig 3: Domain Adaptation Comparison
# ═════════════════════════════════════════════════════════════
print('[3] SuppFig 3: Domain Adaptation Comparison...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Probability distributions before domain adaptation
ax = axes[0]
d = cache['binary_ext_pure_V3-ViT']
probs = d['probs'][:, 1]; true = d['true']
normal_probs = probs[true == 0]; jaun_probs = probs[true == 1]
ax.hist(normal_probs, bins=20, alpha=0.6, color='#2A9D8F', label=f'Normal (n={len(normal_probs)})', density=True)
ax.hist(jaun_probs, bins=20, alpha=0.6, color='#E76F51', label=f'Jaundice (n={len(jaun_probs)})', density=True)
ax.axvline(0.5, color='red', ls='--', lw=1, label='Decision threshold (0.5)')
ax.set_xlabel('Predicted Probability of Jaundice')
ax.set_ylabel('Density')
ax.set_title('(A) Before Domain Adaptation\n(External n=117, V3-ViT)', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)
ax.text(0.02, 0.98, f'Sensitivity: {(jaun_probs > 0.5).mean():.1%}\n'
         f'Specificity: {(normal_probs < 0.5).mean():.1%}\n'
         f'Miscalibration: probabilities\ncollapsed below 0.5',
         transform=ax.transAxes, fontsize=6, va='top',
         bbox=dict(boxstyle='round', facecolor='#FEF3C7', alpha=0.9))

# Panel B: Probability distributions after domain adaptation
ax = axes[1]
d = cache['binary_ext_da_DA-ViT']
probs = d['probs'][:, 1]; true = d['true']
normal_probs = probs[true == 0]; jaun_probs = probs[true == 1]
ax.hist(normal_probs, bins=15, alpha=0.6, color='#2A9D8F', label=f'Normal (n={len(normal_probs)})', density=True)
ax.hist(jaun_probs, bins=15, alpha=0.6, color='#E76F51', label=f'Jaundice (n={len(jaun_probs)})', density=True)
ax.axvline(0.5, color='red', ls='--', lw=1, label='Decision threshold (0.5)')
ax.set_xlabel('Predicted Probability of Jaundice')
ax.set_ylabel('Density')
ax.set_title('(B) After Domain Adaptation\n(Held-out n=22, DA-ViT)', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)
ax.text(0.02, 0.98, f'Sensitivity: {(jaun_probs > 0.5).mean():.1%}\n'
         f'Specificity: {(normal_probs < 0.5).mean():.1%}\n'
         f'Calibration restored:\nprobabilities well-separated',
         transform=ax.transAxes, fontsize=6, va='top',
         bbox=dict(boxstyle='round', facecolor='#D1FAE5', alpha=0.9))

# Panel C: AUC comparison across approaches
ax = axes[2]
approaches = ['Internal\n(n=110)', 'External Pure\n(n=117)', 'Domain\nAdapted (n=22)']
aucs = [0.993, 0.885, 1.000]
auc_los = [0.979, 0.826, 1.000]
auc_his = [1.000, 0.937, 1.000]
colors_bar = [LAYER1_COLOR, '#E76F51', DA_COLOR]
x = np.arange(len(approaches))
bars = ax.bar(x, aucs, color=colors_bar, width=0.5, alpha=0.85, edgecolor='white')
ax.errorbar(x, aucs, yerr=[[aucs[i] - auc_los[i] for i in range(3)],
                            [auc_his[i] - aucs[i] for i in range(3)]],
             fmt='none', color='black', capsize=4, lw=1)
for i, (v, lo, hi) in enumerate(zip(aucs, auc_los, auc_his)):
    ax.text(i, v + 0.005, f'{v:.3f}\n[{lo:.3f}-{hi:.3f}]', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(approaches, fontsize=8)
ax.set_ylabel('AUC-ROC')
ax.set_ylim([0.7, 1.08])
ax.set_title('(C) AUC Comparison\nAcross Validation Approaches', fontsize=9, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Supplementary Figure 3. Domain Adaptation Restores External Performance',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFig3_domain_adaptation.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFig3_domain_adaptation.svg'))
plt.close(fig)
print('  Saved SuppFig3')


# ═════════════════════════════════════════════════════════════
# SuppFig 4: Confusion Matrix Grid (all tasks)
# ═════════════════════════════════════════════════════════════
print('[4] SuppFig 4: Confusion Matrix Grid...')

# Map cache keys to display info
cm_tasks = [
    ('binary_int_Bin-v3-ViT', 'Binary Internal\n(ViT, n=110)', 2, ['#2A9D8F', '#E76F51']),
    ('binary_ext_pure_V3-ViT', 'Binary External\n(V3-ViT, n=117)', 2, ['#2A9D8F', '#E76F51']),
    ('binary_ext_da_DA-ViT', 'Binary DA\n(DA-ViT, n=22)', 2, ['#2A9D8F', '#E76F51']),
    ('tbil_face_FaceTern-ConvNeXt', 'TBIL Face\n(ConvNeXt, n=40)', 3, ['#F4A261', '#E76F51', '#DC2626']),
    ('tbil_eye_EyeTBIL-ConvNeXt', 'TBIL Eyelid\n(ConvNeXt, n=39)', 3, ['#F4A261', '#E76F51', '#DC2626']),
    ('dbil_eye_ConvNeXt', 'DBIL Eyelid\n(ConvNeXt, n=52)', 3, ['#F4A261', '#E76F51', '#DC2626']),
    ('ibil_face_Swin', 'IBIL Face\n(Swin, n=55)', 3, ['#F4A261', '#E76F51', '#DC2626']),
    ('type_Type-EffNet', 'Jaundice Type\n(EffNet, n=39)', 2, ['#2A9D8F', '#E76F51']),
    ('cp_CP-ConvNeXt', 'Child-Pugh\n(ConvNeXt, n=110)', 3, ['#FCD34D', '#F59E0B', '#B45309']),
    ('meld_MELD-ViT', 'MELD Risk\n(ViT, n=110)', 3, ['#FCA5A5', '#EF4444', '#7F1D1D']),
]

n_cols = 5; n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7))
fig.suptitle('Supplementary Figure 4. Confusion Matrices — All 10 Tasks (Best Model per Task)',
             fontsize=12, fontweight='bold', y=1.0)

for idx, (key, title, nc, cmap_colors) in enumerate(cm_tasks):
    ax = axes[idx // n_cols, idx % n_cols]
    if key not in cache:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=8); ax.axis('off'); continue
    d = cache[key]
    cm = confusion_matrix(d['true'], d['probs'].argmax(1), labels=list(range(nc)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    # Custom colormap from the provided colors
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF'] + cmap_colors)
    im = ax.imshow(cm_pct, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    labels_2 = ['Normal', 'Jaundice']
    labels_3 = {0: ['Mild', 'Mod', 'Sev'], 1: ['Normal', 'Mod', 'Sev'],
                 2: ['Low', 'Med', 'High'], 3: ['A', 'B', 'C'], 4: ['Hep', 'Chol']}
    if nc == 2:
        xlabels = ylabels = ['Neg', 'Pos']
    else:
        # Task-specific labels
        if 'cp_' in key: xlabels = ylabels = ['A', 'B', 'C']
        elif 'meld_' in key: xlabels = ylabels = ['Low', 'Med', 'High']
        elif 'dbil' in key or 'ibil' in key: xlabels = ylabels = ['Lo', 'Md', 'Hi']
        else: xlabels = ylabels = ['Mild', 'Mod', 'Sev']

    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel('Predicted', fontsize=7); ax.set_ylabel('True', fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold')

    for i in range(nc):
        for j in range(nc):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.0f}%)', ha='center', va='center',
                    fontsize=6, color=color)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG_DIR, 'SuppFig4_confusion_matrices.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFig4_confusion_matrices.svg'))
plt.close(fig)
print('  Saved SuppFig4')


# ═════════════════════════════════════════════════════════════
# SuppFig 5: All Binary Models Comparison (14 models)
# ═════════════════════════════════════════════════════════════
print('[5] SuppFig 5: All Binary Models Comparison...')
bin_tasks = ['Binary-Screening', 'Binary-External-Pure', 'Binary-External']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: AUC bar chart for all binary models
ax = axes[0]
bin_df = bs[bs['task'].isin(bin_tasks)].sort_values(['task', 'auc'], ascending=[True, False])
colors_bin = []
task_color_map = {'Binary-Screening': LAYER1_COLOR, 'Binary-External-Pure': '#E76F51',
                   'Binary-External': DA_COLOR}
for _, r in bin_df.iterrows():
    colors_bin.append(task_color_map.get(r['task'], '#888'))

y_pos = np.arange(len(bin_df))
ax.barh(y_pos, bin_df['auc'].values, color=colors_bin, height=0.7, alpha=0.85, edgecolor='white', lw=0.3)
ax.errorbar(bin_df['auc'].values, y_pos,
             xerr=[bin_df['auc'].values - bin_df['auc_lo'].values,
                    bin_df['auc_hi'].values - bin_df['auc'].values],
             fmt='none', color='black', capsize=1.5, lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(bin_df['name'].values, fontsize=6)
ax.set_xlabel('AUC-ROC [95% CI]')
ax.set_title('(A) All Binary Screening Models', fontsize=10, fontweight='bold')
ax.set_xlim([0.0, 1.1])
ax.axvline(0.5, color='gray', ls='--', lw=0.3)
ax.grid(True, axis='x', alpha=0.3)
legend_el = [Patch(facecolor=LAYER1_COLOR, alpha=0.85, label='Internal (n=110)'),
              Patch(facecolor='#E76F51', alpha=0.85, label='External Pure (n=117)'),
              Patch(facecolor=DA_COLOR, alpha=0.85, label='Domain-Adapted (n=22)')]
ax.legend(handles=legend_el, fontsize=7, loc='lower right')

# Panel B: Sensitivity vs Specificity scatter for all binary models
ax = axes[1]
for _, r in bin_df.iterrows():
    color = task_color_map.get(r['task'], '#888')
    ax.scatter(r['spec'], r['sens'], color=color, s=60, zorder=5, edgecolors='white', lw=0.5)
    ax.annotate(r['name'][:15], (r['spec'], r['sens']), fontsize=5, ha='left',
                 xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title('(B) Sensitivity vs Specificity (All Binary Models)', fontsize=10, fontweight='bold')
ax.set_xlim([0, 1.05]); ax.set_ylim([0, 1.05])
ax.plot([0, 1], [1, 0], 'k--', lw=0.3, alpha=0.2)
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_el, fontsize=7, loc='lower left')

plt.suptitle('Supplementary Figure 5. All Binary Screening Models Comparison',
             fontsize=12, fontweight='bold', y=1.0)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFig5_binary_models.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFig5_binary_models.svg'))
plt.close(fig)
print('  Saved SuppFig5')


# ═════════════════════════════════════════════════════════════
# SuppFig 6: All Grading Models Comparison (TBIL/DBIL/IBIL/Type)
# ═════════════════════════════════════════════════════════════
print('[6] SuppFig 6: All Grading Models Comparison...')
grade_tasks = ['TBIL-Eyelid', 'Face-Ternary', 'DBIL', 'IBIL', 'Jaundice-Type', 'Child-Pugh', 'MELD-Risk']
fig, ax = plt.subplots(figsize=(14, 8))
grade_df = bs[bs['task'].isin(grade_tasks)].copy()
# Add layer column
grade_df['layer'] = grade_df['task'].map(TASK_LAYER)
grade_df = grade_df.sort_values(['layer', 'task', 'auc'], ascending=[True, True, False])

y_pos = np.arange(len(grade_df))
colors_grade = [LAYER1_COLOR if l == 1 else LAYER2_COLOR for l in grade_df['layer']]

ax.barh(y_pos, grade_df['auc'].values, color=colors_grade, height=0.7, alpha=0.85, edgecolor='white', lw=0.3)
ax.errorbar(grade_df['auc'].values, y_pos,
             xerr=[grade_df['auc'].values - grade_df['auc_lo'].values,
                    grade_df['auc_hi'].values - grade_df['auc'].values],
             fmt='none', color='black', capsize=1.5, lw=0.5)

# Labels with task prefix
labels = [f'[{r["task"][:8]}] {r["name"]}' for _, r in grade_df.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=5.5)
ax.set_xlabel('AUC-ROC [95% CI]')
ax.set_title('Supplementary Figure 6. All Grading & Scoring Models\n'
             '(Layer 1 = blue, Layer 2 = orange)', fontsize=11, fontweight='bold')
ax.set_xlim([0.0, 1.1])
ax.axvline(0.5, color='gray', ls='--', lw=0.3)
ax.grid(True, axis='x', alpha=0.3)
legend_el = [Patch(facecolor=LAYER1_COLOR, alpha=0.85, label='Layer 1: Recognition'),
              Patch(facecolor=LAYER2_COLOR, alpha=0.85, label='Layer 2: Prognostic Scoring')]
ax.legend(handles=legend_el, fontsize=8, loc='lower right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'SuppFig6_grading_models.png'))
fig.savefig(os.path.join(FIG_DIR, 'SuppFig6_grading_models.svg'))
plt.close(fig)
print('  Saved SuppFig6')

print('\n' + '='*70)
print('  All 6 supplementary figures generated.')
print('='*70)
