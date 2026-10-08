# -*- coding: utf-8 -*-
"""
Evaluate Eyelid-Only and Combined (Face+Eyelid) models.
Generates Nature-style tables and figures.
"""
import os, sys, json, random, time, re
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, precision_recall_curve,
                              average_precision_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')
MODEL_DIR = os.path.join(BASE, 'models')
FIG_DIR = os.path.join(BASE, 'results', 'figures')
TBL_DIR = os.path.join(BASE, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.linewidth': 0.6, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

COLORS = {
    'convnext': '#264653', 'vit': '#2A9D8F', 'effnet': '#E9C46A',
    'ensemble': '#1D3557', 'eyelid': '#E76F51', 'combined': '#6A4C93',
    'face': '#457B9D', 'mild': '#F4D03F', 'moderate': '#E67E22', 'severe': '#E74C3C',
}

# ── Data loading ─────────────────────────────────────────────
def find_patient_data(data_root, category):
    cat_dir = os.path.join(data_root, category)
    if not os.path.exists(cat_dir): return []
    patients = []
    for outer in os.listdir(cat_dir):
        p = os.path.join(cat_dir, outer)
        if not os.path.isdir(p): continue
        inner = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
        base = os.path.join(p, inner[0]) if inner else p
        regular, eyelid = [], []
        for f in os.listdir(base):
            fp = os.path.join(base, f)
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            if 'feature' in f.lower(): continue
            if f.startswith('IMG_') and os.path.getsize(fp) > 2_000_000:
                eyelid.append(fp)
            else:
                regular.append(fp)
        for sub in os.listdir(base):
            sp = os.path.join(base, sub)
            if os.path.isdir(sp):
                for f in os.listdir(sp):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        regular.append(os.path.join(sp, f))
        if regular or eyelid:
            patients.append({'id': outer, 'regular': regular, 'eyelid': eyelid,
                             'category': category})
    return patients

print('[1] Loading data...')
all_p = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    ps = find_patient_data(DATA_ROOT, cat)
    print(f'  {cat}: {len(ps)} patients')
    all_p.extend(ps)

j_pats = [p for p in all_p if p['category'] in ('mild', 'moderate', 'severe')]
tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tfn = lambda p: tmap.get(p['category'], -1)

# Stratified split (same as training)
by_l = {}
for p in j_pats:
    by_l.setdefault(tfn(p), []).append(p)
train_p, val_p = [], []
for l, g in by_l.items():
    random.shuffle(g)
    n = max(1, int(len(g) * 0.2))
    val_p.extend(g[:n])
    train_p.extend(g[n:])
print(f'  Validation set: {len(val_p)} jaundice patients')
print(f'    mild={sum(1 for p in val_p if p["category"]=="mild")}, '
      f'moderate={sum(1 for p in val_p if p["category"]=="moderate")}, '
      f'severe={sum(1 for p in val_p if p["category"]=="severe")}')

# Also get normal validation patients for face binary eval
normal_pats = [p for p in all_p if p['category'] == 'normal']
random.shuffle(normal_pats)
val_normal = normal_pats[:len(val_p)]

# ── Transforms ───────────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load models ──────────────────────────────────────────────
print('\n[2] Loading models...')

def load_model(backbone, ckpt_name, num_classes):
    ckpt = os.path.join(MODEL_DIR, ckpt_name)
    if not os.path.exists(ckpt):
        print(f'  Not found: {ckpt_name}')
        return None
    m = timm.create_model(backbone, pretrained=False, num_classes=num_classes).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.eval()
    print(f'  Loaded: {ckpt_name}')
    return m

# Eyelid ternary models
eyelid_models = {}
for bb, ckpt in [('convnext_tiny', 'eyelid_ternary_convnext_tiny.pt'),
                   ('vit_tiny_patch16_224', 'eyelid_ternary_vit_tiny_patch16_224.pt'),
                   ('efficientnet_b0', 'eyelid_ternary_efficientnet_b0.pt')]:
    m = load_model(bb, ckpt, 3)
    if m: eyelid_models[bb] = m

# Face ternary models
face_models = {}
for bb, ckpt in [('vit_tiny_patch16_224', 'ternary_vit_tiny_patch16_224.pt'),
                   ('efficientnet_b0', 'ternary_efficientnet_b0.pt'),
                   ('convnext_tiny', 'ternary_convnext_tiny.pt')]:
    m = load_model(bb, ckpt, 3)
    if m: face_models[bb] = m


# ── Inference functions ──────────────────────────────────────
def predict_eyelid(model, img_path):
    try:
        img = Image.open(img_path).convert('RGB')
    except:
        return None
    t = eval_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        return F.softmax(out, dim=1).cpu().numpy()[0]

def predict_face(model, img_paths):
    if not img_paths: return None
    tensors = []
    for p in img_paths[:8]:
        try:
            img = Image.open(p).convert('RGB')
            tensors.append(eval_tf(img))
        except:
            pass
    if not tensors: return None
    t = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        return F.softmax(out, dim=1).mean(dim=0).cpu().numpy()

def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, ci=95):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    aucs, f1s, accs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]
        if len(np.unique(yt)) < 2: continue
        try:
            aucs.append(roc_auc_score(yt, ypr, multi_class='ovr'))
            f1s.append(f1_score(yt, yp, average='macro'))
            accs.append(accuracy_score(yt, yp))
        except: pass
    def fmt(vals):
        if not vals: return 'N/A'
        m, lo, hi = np.mean(vals), np.percentile(vals, (100-ci)/2), np.percentile(vals, 100-(100-ci)/2)
        return f'{m:.3f} ({lo:.3f}\u2013{hi:.3f})'
    return {'auc': fmt(aucs), 'f1': fmt(f1s), 'acc': fmt(accs),
            'auc_mean': np.mean(aucs) if aucs else 0}


# ── Evaluate on validation set ───────────────────────────────
print('\n[3] Evaluating on validation set...')
results = {}
y_true_all = []
eyelid_preds = {}
face_preds = {}
combined_preds = {}

for p in tqdm(val_p, desc='Predicting'):
    cat = tmap[p['category']]
    y_true_all.append(cat)

    # Eyelid predictions
    if p['eyelid']:
        for bb, model in eyelid_models.items():
            prob = predict_eyelid(model, p['eyelid'][0])
            if bb not in eyelid_preds: eyelid_preds[bb] = []
            eyelid_preds[bb].append(prob)
        # Eyelid ensemble
        all_probs = [predict_eyelid(m, p['eyelid'][0]) for m in eyelid_models.values()]
        ensemble_prob = np.mean(all_probs, axis=0)
        if 'ensemble' not in eyelid_preds: eyelid_preds['ensemble'] = []
        eyelid_preds['ensemble'].append(ensemble_prob)
    else:
        for bb in list(eyelid_models.keys()) + ['ensemble']:
            if bb not in eyelid_preds: eyelid_preds[bb] = []
            eyelid_preds[bb].append(np.array([1/3, 1/3, 1/3]))

    # Face predictions
    if p['regular']:
        for bb, model in face_models.items():
            prob = predict_face(model, p['regular'])
            if bb not in face_preds: face_preds[bb] = []
            face_preds[bb].append(prob)
        all_probs_f = [predict_face(m, p['regular']) for m in face_models.values()]
        ensemble_f = np.mean(all_probs_f, axis=0)
        if 'ensemble' not in face_preds: face_preds['ensemble'] = []
        face_preds['ensemble'].append(ensemble_f)
    else:
        for bb in list(face_models.keys()) + ['ensemble']:
            if bb not in face_preds: face_preds[bb] = []
            face_preds[bb].append(np.array([1/3, 1/3, 1/3]))

    # Combined (average of face ensemble + eyelid ensemble)
    comb = (face_preds['ensemble'][-1] + eyelid_preds['ensemble'][-1]) / 2.0
    combined_preds.setdefault('avg', []).append(comb)

y_true = np.array(y_true_all)

# ── Compute metrics ──────────────────────────────────────────
print('\n[4] Computing metrics...')

DISPLAY = {
    'convnext_tiny': 'ConvNeXt',
    'vit_tiny_patch16_224': 'ViT',
    'efficientnet_b0': 'EfficientNet',
    'ensemble': 'Ensemble',
}

def eval_group(preds_dict, y_true, prefix):
    table_rows = []
    for name, probs in preds_dict.items():
        probs_arr = np.array(probs)
        pred_labels = probs_arr.argmax(axis=1)
        ci = bootstrap_ci(y_true, pred_labels, probs_arr)
        cm = confusion_matrix(y_true, pred_labels, labels=[0,1,2])
        tn_fp_fn_tp = cm
        sens = [cm[i,i]/cm[i,:].sum() if cm[i,:].sum()>0 else 0 for i in range(3)]
        spec = [cm[:,i].sum()-cm[i,i] for i in range(3)]
        spec = [1 - (cm[:,i].sum()-cm[i,i])/(len(y_true)-cm[i,:].sum()) if (len(y_true)-cm[i,:].sum())>0 else 0 for i in range(3)]
        display = DISPLAY.get(name, name)
        table_rows.append({
            'Model': f'{prefix}: {display}',
            'AUC-ROC (OvR)': ci['auc'],
            'Accuracy': ci['acc'],
            'F1 (macro)': ci['f1'],
            'Sens (Mild)': f'{sens[0]:.3f}',
            'Sens (Mod)': f'{sens[1]:.3f}',
            'Sens (Sev)': f'{sens[2]:.3f}',
        })
        print(f'  {prefix} {display:15s}: AUC={ci["auc_mean"]:.4f}')
    return table_rows

table_rows = []
table_rows.extend(eval_group(eyelid_preds, y_true, 'Eyelid'))
table_rows.extend(eval_group(face_preds, y_true, 'Face'))
table_rows.extend(eval_group(combined_preds, y_true, 'Combined'))

df_table = pd.DataFrame(table_rows)
csv_path = os.path.join(TBL_DIR, 'eyelid_face_combined_results.csv')
df_table.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n  Table saved: {csv_path}')


# ═════════════════════════════════════════════════════════════
# FIGURE: Nature-style multi-panel
# ═════════════════════════════════════════════════════════════
print('\n[5] Generating figures...')

from sklearn.preprocessing import label_binarize

# ── Figure 1: Eyelid model ROC + AUC bars + Confusion matrix ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: ROC curves (eyelid models)
ax = axes[0]
y_bin = label_binarize(y_true, classes=[0, 1, 2])
for name, probs in eyelid_preds.items():
    probs_arr = np.array(probs)
    display_name = DISPLAY.get(name, name)
    # micro-average ROC
    fpr, tpr, _ = __import__('sklearn.metrics', fromlist=['roc_curve']).roc_curve(y_bin.ravel(), probs_arr.ravel())
    roc_auc = __import__('sklearn.metrics', fromlist=['auc']).auc(fpr, tpr)
    lw = 2.5 if name == 'ensemble' else 1.2
    color = {'ConvNeXt': COLORS['convnext'], 'ViT': COLORS['vit'],
             'EfficientNet': COLORS['effnet'], 'Ensemble': COLORS['ensemble' if 'ensemble' in COLORS else 'eyelid']}.get(display_name, 'gray')
    ax.plot(fpr, tpr, lw=lw, color=color, label=f'{display_name} ({roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=8)
ax.set_ylabel('True Positive Rate', fontsize=8)
ax.set_title('a | Eyelid-only ROC (OvR)', fontsize=9, fontweight='bold', loc='left')
ax.legend(fontsize=6, loc='lower right', framealpha=0.9)
ax.set_xlim([-0.02, 1]); ax.set_ylim([0, 1.02])
ax.tick_params(labelsize=7)

# Panel B: AUC comparison bar chart
ax = axes[1]
groups = ['Eyelid\nConvNeXt', 'Eyelid\nViT', 'Eyelid\nEffNet', 'Eyelid\nEnsemble',
          'Face\nEnsemble', 'Combined\nEnsemble']
all_aucs = []
for name in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0', 'ensemble']:
    probs_arr = np.array(eyelid_preds[name])
    try: all_aucs.append(roc_auc_score(y_true, probs_arr, multi_class='ovr'))
    except: all_aucs.append(0)
for name in ['ensemble']:
    probs_arr = np.array(face_preds[name])
    try: all_aucs.append(roc_auc_score(y_true, probs_arr, multi_class='ovr'))
    except: all_aucs.append(0)
probs_arr = np.array(combined_preds['avg'])
try: all_aucs.append(roc_auc_score(y_true, probs_arr, multi_class='ovr'))
except: all_aucs.append(0)
bar_colors = [COLORS['convnext'], COLORS['vit'], COLORS['effnet'], COLORS['eyelid'],
              COLORS['face'], COLORS['combined']]
bars = ax.bar(range(len(groups)), all_aucs, color=bar_colors, width=0.6, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups, fontsize=6.5, ha='center')
ax.set_ylabel('AUC-ROC (OvR)', fontsize=8)
ax.set_title('b | AUC Comparison', fontsize=9, fontweight='bold', loc='left')
ax.set_ylim([0.5, 1.0])
ax.axhline(y=0.5, color='gray', ls='--', lw=0.4, alpha=0.5)
for bar, val in zip(bars, all_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.008, f'{val:.3f}',
            ha='center', fontsize=6, fontweight='bold')
ax.tick_params(labelsize=7)

# Panel C: Confusion matrix (Eyelid Ensemble)
ax = axes[2]
probs_arr = np.array(eyelid_preds['ensemble'])
pred_labels = probs_arr.argmax(axis=1)
cm = confusion_matrix(y_true, pred_labels, labels=[0,1,2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(['Mild', 'Moderate', 'Severe'], fontsize=7)
ax.set_yticklabels(['Mild', 'Moderate', 'Severe'], fontsize=7)
ax.set_xlabel('Predicted', fontsize=8)
ax.set_ylabel('True', fontsize=8)
ax.set_title('c | Eyelid Ensemble CM', fontsize=9, fontweight='bold', loc='left')
for i in range(3):
    for j in range(3):
        text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
        ax.text(j, i, f'{cm_norm[i,j]:.2f}\n({cm[i,j]})', ha='center', va='center',
                color=text_color, fontsize=7)

plt.tight_layout(w_pad=2)
fig1_path = os.path.join(FIG_DIR, 'Figure_eyelid_performance.png')
fig.savefig(fig1_path)
fig.savefig(fig1_path.replace('.png', '.svg'))
plt.close(fig)
print(f'  Saved: {fig1_path}')


# ── Figure 2: Per-class OvR ROC comparison ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
class_names = ['Mild', 'Moderate', 'Severe']
class_colors = [COLORS['mild'], COLORS['moderate'], COLORS['severe']]
roc_fpr = __import__('sklearn.metrics', fromlist=['roc_curve']).roc_curve
roc_auc_fn = __import__('sklearn.metrics', fromlist=['auc']).auc

for cls_idx in range(3):
    ax = axes[cls_idx]
    y_binary = (y_true == cls_idx).astype(int)
    for name, probs, style_label, color, lw in [
        ('ensemble', eyelid_preds, 'Eyelid', COLORS['eyelid'], 2.2),
        ('ensemble', face_preds, 'Face', COLORS['face'], 1.8),
        ('avg', combined_preds, 'Combined', COLORS['combined'], 2.5),
    ]:
        probs_arr = np.array(probs[name])
        scores = probs_arr[:, cls_idx]
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_fpr(y_binary, scores)
            a = roc_auc_fn(fpr, tpr)
            ax.plot(fpr, tpr, lw=lw, color=color, label=f'{style_label} ({a:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=8)
    ax.set_ylabel('True Positive Rate', fontsize=8)
    panel = chr(ord('a') + cls_idx)
    ax.set_title(f'{panel} | {class_names[cls_idx]} (OvR)', fontsize=9, fontweight='bold', loc='left')
    ax.legend(fontsize=6, loc='lower right')
    ax.set_xlim([-0.02, 1]); ax.set_ylim([0, 1.02])
    ax.tick_params(labelsize=7)

plt.tight_layout(w_pad=2)
fig2_path = os.path.join(FIG_DIR, 'Figure_per_class_roc.png')
fig.savefig(fig2_path)
fig.savefig(fig2_path.replace('.png', '.svg'))
plt.close(fig)
print(f'  Saved: {fig2_path}')


# ── Figure 3: Combined approach overview (Nature-style) ──────
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

# Panel a: Eyelid ensemble ROC
ax1 = fig.add_subplot(gs[0, 0])
probs_arr = np.array(eyelid_preds['ensemble'])
fpr, tpr, _ = roc_fpr(y_bin.ravel(), probs_arr.ravel())
a = roc_auc_fn(fpr, tpr)
ax1.plot(fpr, tpr, lw=2.5, color=COLORS['eyelid'], label=f'Ensemble ({a:.3f})')
ax1.fill_between(fpr, tpr, alpha=0.1, color=COLORS['eyelid'])
ax1.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
ax1.set_xlabel('FPR', fontsize=7); ax1.set_ylabel('TPR', fontsize=7)
ax1.set_title('a | Eyelid-only ROC', fontsize=8, fontweight='bold', loc='left')
ax1.legend(fontsize=6); ax1.tick_params(labelsize=6)

# Panel b: Combined ROC
ax2 = fig.add_subplot(gs[0, 1])
probs_arr = np.array(combined_preds['avg'])
fpr, tpr, _ = roc_fpr(y_bin.ravel(), probs_arr.ravel())
a = roc_auc_fn(fpr, tpr)
ax2.plot(fpr, tpr, lw=2.5, color=COLORS['combined'], label=f'Combined ({a:.3f})')
ax2.fill_between(fpr, tpr, alpha=0.1, color=COLORS['combined'])
ax2.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
ax2.set_xlabel('FPR', fontsize=7); ax2.set_ylabel('TPR', fontsize=7)
ax2.set_title('b | Face + Eyelid ROC', fontsize=8, fontweight='bold', loc='left')
ax2.legend(fontsize=6); ax2.tick_params(labelsize=6)

# Panel c: Radar chart comparing approaches
ax3 = fig.add_subplot(gs[0, 2], polar=True)
metrics_labels = ['AUC', 'F1', 'Accuracy', 'Sens\n(Mild)', 'Sens\n(Sev)']
angles = np.linspace(0, 2*np.pi, len(metrics_labels), endpoint=False).tolist()
angles += angles[:1]

for name, probs, color, label in [
    ('ensemble', eyelid_preds, COLORS['eyelid'], 'Eyelid'),
    ('ensemble', face_preds, COLORS['face'], 'Face'),
    ('avg', combined_preds, COLORS['combined'], 'Combined'),
]:
    probs_arr = np.array(probs[name])
    pred_l = probs_arr.argmax(axis=1)
    try: auc_v = roc_auc_score(y_true, probs_arr, multi_class='ovr')
    except: auc_v = 0
    f1_v = f1_score(y_true, pred_l, average='macro')
    acc_v = accuracy_score(y_true, pred_l)
    cm = confusion_matrix(y_true, pred_l, labels=[0,1,2])
    s_mild = cm[0,0]/cm[0,:].sum() if cm[0,:].sum()>0 else 0
    s_sev = cm[2,2]/cm[2,:].sum() if cm[2,:].sum()>0 else 0
    vals = [auc_v, f1_v, acc_v, s_mild, s_sev]
    vals += vals[:1]
    ax3.plot(angles, vals, lw=1.5, color=color, label=label)
    ax3.fill(angles, vals, alpha=0.08, color=color)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metrics_labels, fontsize=6)
ax3.set_title('c | Multi-metric', fontsize=8, fontweight='bold', pad=10)
ax3.legend(fontsize=5, loc='upper right', bbox_to_anchor=(1.35, 1.15))

# Panel d: Eyelid CM
ax4 = fig.add_subplot(gs[1, 0])
probs_arr = np.array(eyelid_preds['ensemble'])
pred_l = probs_arr.argmax(axis=1)
cm = confusion_matrix(y_true, pred_l, labels=[0,1,2])
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
ax4.imshow(cm_n, cmap='YlOrRd', vmin=0, vmax=1)
ax4.set_xticks(range(3)); ax4.set_yticks(range(3))
ax4.set_xticklabels(['Mild', 'Mod', 'Sev'], fontsize=6)
ax4.set_yticklabels(['Mild', 'Mod', 'Sev'], fontsize=6)
for i in range(3):
    for j in range(3):
        c = 'white' if cm_n[i,j] > 0.5 else 'black'
        ax4.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center', color=c, fontsize=6)
ax4.set_title('d | Eyelid CM', fontsize=8, fontweight='bold', loc='left')

# Panel e: Combined CM
ax5 = fig.add_subplot(gs[1, 1])
probs_arr = np.array(combined_preds['avg'])
pred_l = probs_arr.argmax(axis=1)
cm = confusion_matrix(y_true, pred_l, labels=[0,1,2])
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
ax5.imshow(cm_n, cmap='Purples', vmin=0, vmax=1)
ax5.set_xticks(range(3)); ax5.set_yticks(range(3))
ax5.set_xticklabels(['Mild', 'Mod', 'Sev'], fontsize=6)
ax5.set_yticklabels(['Mild', 'Mod', 'Sev'], fontsize=6)
for i in range(3):
    for j in range(3):
        c = 'white' if cm_n[i,j] > 0.5 else 'black'
        ax5.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center', color=c, fontsize=6)
ax5.set_title('e | Combined CM', fontsize=8, fontweight='bold', loc='left')

# Panel f: Probability distribution by class
ax6 = fig.add_subplot(gs[1, 2])
probs_arr = np.array(combined_preds['avg'])
for idx, cname, ccolor in [(0,'Mild',COLORS['mild']), (1,'Moderate',COLORS['moderate']), (2,'Severe',COLORS['severe'])]:
    mask = y_true == idx
    ax6.hist(probs_arr[mask, idx], bins=10, alpha=0.5, color=ccolor, label=cname, edgecolor='white', linewidth=0.3)
ax6.set_xlabel('Predicted Probability', fontsize=7)
ax6.set_ylabel('Count', fontsize=7)
ax6.set_title('f | Prob. Distribution', fontsize=8, fontweight='bold', loc='left')
ax6.legend(fontsize=5)
ax6.tick_params(labelsize=6)

fig.suptitle('Eyelid-Only and Combined (Face + Eyelid) Jaundice Severity Grading',
             fontsize=10, fontweight='bold', y=1.01)
fig3_path = os.path.join(FIG_DIR, 'Figure_eyelid_combined_overview.png')
fig.savefig(fig3_path)
fig.savefig(fig3_path.replace('.png', '.svg'))
plt.close(fig)
print(f'  Saved: {fig3_path}')


# ── Save DOCX table ──────────────────────────────────────────
from docx import Document
from docx.shared import Pt

doc = Document()
doc.add_heading('Supplementary Table: Eyelid-Only and Combined Performance', level=2)
p = doc.add_paragraph('Performance metrics for eyelid-only models, face-only models, '
                       'and combined (face + eyelid) approach on the validation cohort '
                       '(n=42 jaundice patients). Values are point estimates with 95% CIs '
                       'from 1,000 bootstrap resamples.')
doc.add_paragraph('')

table = doc.add_table(rows=1, cols=len(df_table.columns))
table.style = 'Light Grid Accent 1'
for i, col in enumerate(df_table.columns):
    table.rows[0].cells[i].text = col
    for pp in table.rows[0].cells[i].paragraphs:
        for r in pp.runs: r.bold = True; r.font.size = Pt(8)
for _, row in df_table.iterrows():
    cells = table.add_row().cells
    for i, val in enumerate(row):
        cells[i].text = str(val)
        for pp in cells[i].paragraphs:
            for r in pp.runs: r.font.size = Pt(8)

docx_path = os.path.join(TBL_DIR, 'Table_eyelid_combined.docx')
doc.save(docx_path)
print(f'\n  DOCX table saved: {docx_path}')

# ── Summary ──────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  Evaluation Complete!')
print('=' * 55)
print(f'\n  Eyelid Ensemble AUC: {roc_auc_score(y_true, np.array(eyelid_preds["ensemble"]), multi_class="ovr"):.4f}')
print(f'  Face Ensemble AUC:   {roc_auc_score(y_true, np.array(face_preds["ensemble"]), multi_class="ovr"):.4f}')
print(f'  Combined AUC:        {roc_auc_score(y_true, np.array(combined_preds["avg"]), multi_class="ovr"):.4f}')
print(f'\n  Figures:')
for f in sorted(os.listdir(FIG_DIR)):
    if 'eyelid' in f or 'per_class' in f:
        print(f'    {f}')
print(f'  Tables:')
for f in sorted(os.listdir(TBL_DIR)):
    if 'eyelid' in f or 'combined' in f:
        print(f'    {f}')
