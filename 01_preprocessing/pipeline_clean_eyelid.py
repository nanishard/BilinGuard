# -*- coding: utf-8 -*-
"""
BilinGuard: Complete Post-Training Pipeline (Clean Eyelid Models)
  1. Evaluate clean eyelid models → tables
  2. Generate Nature-style figures
  3. Interpretability: Grad-CAM, t-SNE, color analysis
  4. Update deployment inference engine
"""
import os, sys, json, random, time, re, warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from docx import Document
from docx.shared import Pt
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
MODEL_DIR = os.path.join(BASE, 'models')
FIG_DIR = os.path.join(BASE, 'results', 'figures', 'interpretability')
TBL_DIR = os.path.join(BASE, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})

COLORS = {'convnext': '#264653', 'vit': '#2A9D8F', 'effnet': '#E9C46A',
           'ensemble': '#1D3557', 'eyelid': '#E76F51',
           'mild': '#F4D03F', 'moderate': '#E67E22', 'severe': '#E74C3C'}
DISPLAY = {'convnext_tiny': 'ConvNeXt-Tiny', 'vit_tiny_patch16_224': 'ViT-Tiny',
           'efficientnet_b0': 'EfficientNet-B0', 'ensemble': 'BilinGuard Ensemble'}

BACKBONES = ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0']
CLEAN_PREFIX = 'eyelid_clean_ternary_'

tmap = {'mild': 0, 'moderate': 1, 'severe': 2}

# ── Load clean data ──────────────────────────────────────────
print('[1] Loading clean data...')
df = pd.read_csv(MANIFEST)
j_df = df[(df['category'].isin(['mild', 'moderate', 'severe'])) & (df['n_eyelid'] > 0)]
patients = []
for _, row in j_df.iterrows():
    paths = json.loads(row['eyelid_images'])
    if paths:
        patients.append({'id': row['patient_id'], 'eyelid_path': paths[0],
                         'category': row['category']})

tfn = lambda p: tmap[p['category']]
by_l = {}
for p in patients:
    by_l.setdefault(tfn(p), []).append(p)
train_p, val_p = [], []
for l, g in by_l.items():
    random.shuffle(g)
    n = max(1, int(len(g) * 0.2))
    val_p.extend(g[:n])
    train_p.extend(g[n:])

y_true = np.array([tmap[p['category']] for p in val_p])
print(f'  Val patients: {len(val_p)} (mild={sum(y_true==0)}, mod={sum(y_true==1)}, sev={sum(y_true==2)})')

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load models ──────────────────────────────────────────────
print('[2] Loading clean eyelid models...')
models = {}
for bb in BACKBONES:
    ckpt = os.path.join(MODEL_DIR, f'{CLEAN_PREFIX}{bb}.pt')
    if os.path.exists(ckpt):
        m = timm.create_model(bb, pretrained=False, num_classes=3).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        models[bb] = m
        print(f'  Loaded: {CLEAN_PREFIX}{bb}.pt')
    else:
        print(f'  NOT FOUND: {ckpt}')

if not models:
    print('ERROR: No clean models found. Waiting...')
    sys.exit(1)

# ── Predict on validation set ────────────────────────────────
print('[3] Running inference on validation set...')
preds_by_model = {}
for bb, model in models.items():
    preds = []
    for p in tqdm(val_p, desc=f'Predict ({bb[:12]})', leave=False):
        try:
            img = Image.open(p['eyelid_path']).convert('RGB')
            t = eval_tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = F.softmax(model(t), dim=1).cpu().numpy()[0]
        except:
            prob = np.array([1/3, 1/3, 1/3])
        preds.append(prob)
    preds_by_model[bb] = np.array(preds)

# Ensemble
ensemble_preds = np.mean([preds_by_model[bb] for bb in models], axis=0)
preds_by_model['ensemble'] = ensemble_preds

# ── Bootstrap CI ─────────────────────────────────────────────
def bootstrap_ci(y_true, probs, n_boot=1000):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    aucs, f1s, accs, aps = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]; ypr = probs[idx]
        if len(np.unique(yt)) < 2: continue
        try:
            aucs.append(roc_auc_score(yt, ypr, multi_class='ovr'))
            ypl = ypr.argmax(axis=1)
            f1s.append(f1_score(yt, ypl, average='macro'))
            accs.append(accuracy_score(yt, ypl))
            y_bin = label_binarize(yt, classes=[0,1,2])
            aps.append(average_precision_score(y_bin, ypr, average='micro'))
        except: pass
    def fmt(vals):
        if not vals: return 'N/A'
        return f'{np.mean(vals):.3f} ({np.percentile(vals,2.5):.3f}\u2013{np.percentile(vals,97.5):.3f})'
    return {'AUC-ROC': fmt(aucs), 'F1 (macro)': fmt(f1s),
            'Accuracy': fmt(accs), 'Avg Precision': fmt(aps)}

# ═════════════════════════════════════════════════════════════
# TABLE 1: Performance metrics
# ═════════════════════════════════════════════════════════════
print('[4] Generating Table...')
table_rows = []
for name in list(models.keys()) + ['ensemble']:
    probs = preds_by_model[name]
    ci = bootstrap_ci(y_true, probs)
    pred_labels = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, pred_labels, labels=[0,1,2])
    sens = [cm[i,i]/cm[i,:].sum() if cm[i,:].sum()>0 else 0 for i in range(3)]
    spec = []
    for i in range(3):
        mask = np.ones(3, bool); mask[i] = False
        tn = cm[mask][:,mask].sum()
        fp = cm[mask][:,i].sum()
        spec.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    row = {'Model': DISPLAY.get(name, name), **ci,
           'Sens (Mild)': f'{sens[0]:.3f}', 'Sens (Mod)': f'{sens[1]:.3f}',
           'Sens (Sev)': f'{sens[2]:.3f}',
           'Spec (Mild)': f'{spec[0]:.3f}', 'Spec (Mod)': f'{spec[1]:.3f}',
           'Spec (Sev)': f'{spec[2]:.3f}'}
    table_rows.append(row)
    print(f'  {DISPLAY.get(name,name):25s}: AUC={ci["AUC-ROC"][:5]}')

table_df = pd.DataFrame(table_rows)
table_df.to_csv(os.path.join(TBL_DIR, 'clean_eyelid_performance.csv'),
                index=False, encoding='utf-8-sig')

# DOCX table
doc = Document()
doc.add_heading('Supplementary Table: Clean Eyelid-Only Ternary Grading Performance', level=2)
doc.add_paragraph(f'Validation cohort: n={len(val_p)} jaundice patients '
                  f'(mild={sum(y_true==0)}, moderate={sum(y_true==1)}, severe={sum(y_true==2)}). '
                  f'95% CIs from 1,000 patient-level bootstrap resamples. '
                  f'Trained on clean eyelid photos (body composition reports excluded).')
doc.add_paragraph('')
t = doc.add_table(rows=1, cols=len(table_df.columns))
t.style = 'Light Grid Accent 1'
for i, col in enumerate(table_df.columns):
    t.rows[0].cells[i].text = col
    for p in t.rows[0].cells[i].paragraphs:
        for r in p.runs: r.bold = True; r.font.size = Pt(8)
for _, row in table_df.iterrows():
    cells = t.add_row().cells
    for i, val in enumerate(row):
        cells[i].text = str(val)
        for p in cells[i].paragraphs:
            for r in p.runs: r.font.size = Pt(8)
doc.save(os.path.join(TBL_DIR, 'Table_clean_eyelid_performance.docx'))
print(f'  Tables saved (CSV + DOCX)')


# ═════════════════════════════════════════════════════════════
# FIGURE 1: Nature-style main figure (ROC + AUC bars + CM)
# ═════════════════════════════════════════════════════════════
print('[5] Generating Nature-style figures...')
y_bin = label_binarize(y_true, classes=[0,1,2])

fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.35)

# Panel a: ROC curves
ax1 = fig.add_subplot(gs[0, 0])
for name in list(models.keys()) + ['ensemble']:
    probs = preds_by_model[name]
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    a = auc(fpr, tpr)
    lw = 2.5 if name == 'ensemble' else 1.3
    color = COLORS.get(name.split('_')[0] if '_' in name else 'ensemble', 'gray')
    ax1.plot(fpr, tpr, lw=lw, color=color, label=f'{DISPLAY.get(name,name)} ({a:.3f})')
ax1.plot([0,1],[0,1],'k--',lw=0.4,alpha=0.3)
ax1.set_xlabel('False Positive Rate', fontsize=8)
ax1.set_ylabel('True Positive Rate', fontsize=8)
ax1.set_title('a | Eyelid-only ROC (OvR micro-avg)', fontsize=9, fontweight='bold', loc='left')
ax1.legend(fontsize=6, loc='lower right')
ax1.tick_params(labelsize=7)

# Panel b: AUC bar chart
ax2 = fig.add_subplot(gs[0, 1])
names_display = [DISPLAY.get(n,n) for n in list(models.keys()) + ['ensemble']]
auc_vals = []
for name in list(models.keys()) + ['ensemble']:
    try: auc_vals.append(roc_auc_score(y_true, preds_by_model[name], multi_class='ovr'))
    except: auc_vals.append(0)
bar_colors = [COLORS.get(n.split('_')[0] if '_' in n else 'ensemble', 'gray')
              for n in list(models.keys()) + ['ensemble']]
bars = ax2.barh(range(len(names_display)), auc_vals, color=bar_colors, height=0.6)
ax2.set_yticks(range(len(names_display)))
ax2.set_yticklabels(names_display, fontsize=7)
ax2.set_xlabel('AUC-ROC (OvR)', fontsize=8)
ax2.set_title('b | AUC Comparison', fontsize=9, fontweight='bold', loc='left')
ax2.set_xlim([0.5, 1.0])
for bar, val in zip(bars, auc_vals):
    ax2.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=6)
ax2.tick_params(labelsize=7)

# Panel c: Ensemble confusion matrix
ax3 = fig.add_subplot(gs[0, 2])
pred_labels = ensemble_preds.argmax(axis=1)
cm = confusion_matrix(y_true, pred_labels, labels=[0,1,2])
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
ax3.imshow(cm_n, cmap='YlOrRd', vmin=0, vmax=1)
ax3.set_xticks(range(3)); ax3.set_yticks(range(3))
ax3.set_xticklabels(['Mild', 'Moderate', 'Severe'], fontsize=7)
ax3.set_yticklabels(['Mild', 'Moderate', 'Severe'], fontsize=7)
for i in range(3):
    for j in range(3):
        c = 'white' if cm_n[i,j] > 0.5 else 'black'
        ax3.text(j, i, f'{cm_n[i,j]:.2f}\n({cm[i,j]})', ha='center', va='center', color=c, fontsize=7)
ax3.set_xlabel('Predicted', fontsize=8); ax3.set_ylabel('True', fontsize=8)
ax3.set_title('c | Ensemble Confusion Matrix', fontsize=9, fontweight='bold', loc='left')

plt.tight_layout(w_pad=2)
p1 = os.path.join(FIG_DIR, 'Fig_clean_eyelid_performance.png')
fig.savefig(p1); fig.savefig(p1.replace('.png','.svg')); plt.close(fig)
print(f'  Saved: Fig_clean_eyelid_performance')

# ═════════════════════════════════════════════════════════════
# FIGURE 2: Per-class OvR ROC
# ═════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
class_names = ['Mild', 'Moderate', 'Severe']
for ci_idx, (ax, cname) in enumerate(zip(axes, class_names)):
    y_binary = (y_true == ci_idx).astype(int)
    for name in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0', 'ensemble']:
        probs = preds_by_model[name]
        scores = probs[:, ci_idx]
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, scores)
            a = auc(fpr, tpr)
            lw = 2.5 if name == 'ensemble' else 1.2
            color = COLORS.get(name.split('_')[0] if '_' in name else 'ensemble', 'gray')
            ax.plot(fpr, tpr, lw=lw, color=color, label=f'{DISPLAY.get(name,name)} ({a:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=0.4,alpha=0.3)
    ax.set_xlabel('FPR', fontsize=7); ax.set_ylabel('TPR', fontsize=7)
    panel = chr(ord('a')+ci_idx)
    ax.set_title(f'{panel} | {cname} (OvR)', fontsize=9, fontweight='bold', loc='left')
    ax.legend(fontsize=5, loc='lower right')
    ax.tick_params(labelsize=6)
plt.tight_layout(w_pad=2)
p2 = os.path.join(FIG_DIR, 'Fig_clean_per_class_roc.png')
fig.savefig(p2); fig.savefig(p2.replace('.png','.svg')); plt.close(fig)
print(f'  Saved: Fig_clean_per_class_roc')


# ═════════════════════════════════════════════════════════════
# INTERPRETABILITY 1: Grad-CAM (ConvNeXt)
# ═════════════════════════════════════════════════════════════
print('[6] Grad-CAM analysis...')
best_bb = max(models.keys(), key=lambda k: roc_auc_score(y_true, preds_by_model[k], multi_class='ovr') if k != 'ensemble' else 0)
print(f'  Best individual model: {best_bb}')
best_model = models[best_bb]

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None; self.activations = None
        target = None
        for name, module in model.named_modules():
            if name == 'stages.3':
                target = module; break
        if target is None:
            for name, module in model.named_modules():
                if 'stages' in name and name.count('.')==1:
                    target = module; break
        if target:
            target.register_forward_hook(self._fh)
            target.register_full_backward_hook(self._bh)
    def _fh(self, m, i, o): self.activations = o
    def _bh(self, m, gi, go): self.gradients = go[0]
    def generate(self, x, cls=None):
        self.model.zero_grad(); self.activations=None; self.gradients=None
        out = self.model(x)
        if cls is None: cls = out.argmax(1).item()
        out[0,cls].backward()
        if self.activations is None or self.gradients is None: return None
        if self.activations.dim() == 4:
            w = self.gradients.mean(dim=(2,3), keepdim=True)
            cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return None

gradcam = GradCAM(best_model)
selected = []
for cat in ['mild', 'moderate', 'severe']:
    cat_p = [p for p in val_p if p['category'] == cat]
    random.shuffle(cat_p)
    if cat_p: selected.append(cat_p[0])

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for row, (patient, cat_name) in enumerate(zip(selected, ['Mild', 'Moderate', 'Severe'][:len(selected)])):
    img = Image.open(patient['eyelid_path']).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    t = eval_tf(img).unsqueeze(0).to(DEVICE)
    cam = gradcam.generate(t.clone(), target_class=tmap[patient['category']])
    # Original
    axes[row, 0].imshow(img); axes[row, 0].axis('off')
    axes[row, 0].set_title(f'{cat_name}\nOriginal', fontsize=8, fontweight='bold')
    # Grad-CAM overlay
    if cam is not None:
        axes[row, 1].imshow(img); axes[row, 1].imshow(cam, cmap='jet', alpha=0.5)
        axes[row, 1].set_title('Grad-CAM Overlay', fontsize=8, fontweight='bold')
    axes[row, 1].axis('off')
    # Grad-CAM only
    if cam is not None:
        axes[row, 2].imshow(cam, cmap='jet')
        axes[row, 2].set_title('Grad-CAM Only', fontsize=8, fontweight='bold')
    axes[row, 2].axis('off')
    # Zoom on high-attention region
    if cam is not None:
        axes[row, 3].imshow(img)
        axes[row, 3].imshow(cam, cmap='jet', alpha=0.6)
        # Find peak region
        peak = np.unravel_index(cam.argmax(), cam.shape)
        y0 = max(0, peak[0]-40); y1 = min(IMG_SIZE, peak[0]+40)
        x0 = max(0, peak[1]-40); x1 = min(IMG_SIZE, peak[1]+40)
        axes[row, 3].add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='white', linewidth=1.5))
        axes[row, 3].set_title('Attention Peak', fontsize=8, fontweight='bold')
    axes[row, 3].axis('off')

# Fill unused rows
for row in range(len(selected), 2):
    for col in range(4):
        axes[row, col].axis('off')

fig.suptitle(f'Grad-CAM: {DISPLAY[best_bb]} Attention on Eyelid Photos', fontsize=10, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(FIG_DIR, 'Fig_clean_gradcam.png')
fig.savefig(p3); fig.savefig(p3.replace('.png','.svg')); plt.close(fig)
print(f'  Saved: Fig_clean_gradcam')


# ═════════════════════════════════════════════════════════════
# INTERPRETABILITY 2: t-SNE feature space
# ═════════════════════════════════════════════════════════════
print('[7] t-SNE feature space...')
def extract_features(model, patient_list):
    feats = []
    model.eval()
    with torch.no_grad():
        for p in tqdm(patient_list, desc='Features', leave=False):
            try:
                img = Image.open(p['eyelid_path']).convert('RGB')
                t = eval_tf(img).unsqueeze(0).to(DEVICE)
                if hasattr(model, 'stages'):
                    x = model.stem(t)
                    for s in model.stages: x = s(x)
                    if hasattr(model, 'norm_pre'): x = model.norm_pre(x)
                    x = model.head.global_pool(x)
                    feats.append(x.flatten(1).cpu().numpy()[0])
                else:
                    x = model.forward_features(t)
                    feats.append(x.mean(dim=1).cpu().numpy()[0] if x.dim()>2 else x.mean(dim=1).cpu().numpy()[0])
            except:
                feats.append(np.zeros(768))
    return np.array(feats)

all_feats = extract_features(best_model, patients)
all_labels = np.array([tmap[p['category']] for p in patients])
tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(patients)-1), n_iter=1000)
tsne_data = tsne.fit_transform(all_feats)

fig, ax = plt.subplots(figsize=(7, 6))
for cls, (cname, color) in enumerate(zip(['Mild','Moderate','Severe'],
                                           [COLORS['mild'], COLORS['moderate'], COLORS['severe']])):
    mask = all_labels == cls
    ax.scatter(tsne_data[mask, 0], tsne_data[mask, 1], c=color, label=cname, s=60,
               alpha=0.7, edgecolors='white', linewidths=0.5)
sil_score = silhouette_score(all_feats, all_labels)
ax.set_title(f'Feature Space (t-SNE)\nSilhouette Score = {sil_score:.3f}', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlabel('t-SNE dim 1', fontsize=8); ax.set_ylabel('t-SNE dim 2', fontsize=8)
plt.tight_layout()
p4 = os.path.join(FIG_DIR, 'Fig_clean_tsne.png')
fig.savefig(p4); fig.savefig(p4.replace('.png','.svg')); plt.close(fig)
print(f'  Saved: Fig_clean_tsne (silhouette={sil_score:.3f})')


# ═════════════════════════════════════════════════════════════
# INTERPRETABILITY 3: Color signal analysis
# ═════════════════════════════════════════════════════════════
print('[8] Color signal analysis...')
color_data = []
for p in patients:
    try:
        img = cv2.imread(p['eyelid_path'])
        if img is None:
            img = np.array(Image.open(p['eyelid_path']).convert('RGB'))[:,:,::-1]
        h, w = img.shape[:2]
        center = img[h//4:3*h//4, w//4:3*w//4]
        lab = cv2.cvtColor(center, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        color_data.append({'category': p['category'],
            'Lab_b': lab[:,:,2].mean(), 'Lab_a': lab[:,:,1].mean(),
            'H': hsv[:,:,0].mean(), 'S': hsv[:,:,1].mean(),
            'R': center[:,:,2].mean(), 'G': center[:,:,1].mean(),
            'yellow_ratio': float((lab[:,:,2] > 150).mean())})
    except: pass
cdf = pd.DataFrame(color_data)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (col, title) in zip(axes, [('Lab_b', 'Lab b* (Yellow\u2192Blue)'),
                                     ('yellow_ratio', 'Yellow Pixel Ratio'),
                                     ('S', 'HSV Saturation')]):
    for cat, (cname, color) in [('mild',('Mild',COLORS['mild'])),
                                  ('moderate',('Moderate',COLORS['moderate'])),
                                  ('severe',('Severe',COLORS['severe']))]:
        vals = cdf[cdf['category']==cat][col]
        ax.hist(vals, bins=15, alpha=0.6, color=color, label=cname, edgecolor='white', linewidth=0.3)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
fig.suptitle('Color Signal by Severity (Eyelid Conjunctiva)', fontsize=10, fontweight='bold')
plt.tight_layout()
p5 = os.path.join(FIG_DIR, 'Fig_clean_color_analysis.png')
fig.savefig(p5); fig.savefig(p5.replace('.png','.svg')); plt.close(fig)
print(f'  Saved: Fig_clean_color_analysis')

# Cohen d separability
print('\n  Color separability (Cohen d, Mild vs Severe):')
for col in ['Lab_b', 'yellow_ratio', 'S', 'R']:
    mild_v = cdf[cdf['category']=='mild'][col]
    sev_v = cdf[cdf['category']=='severe'][col]
    if len(mild_v) > 1 and len(sev_v) > 1:
        pooled = np.sqrt((mild_v.std()**2 + sev_v.std()**2) / 2)
        d = abs(mild_v.mean() - sev_v.mean()) / (pooled + 1e-6)
        print(f'    {col:15s}: d={d:.2f}')


# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Complete Pipeline Finished!')
print('=' * 60)
print(f'\n  Model AUC (clean eyelid):')
for name in list(models.keys()) + ['ensemble']:
    try: a = roc_auc_score(y_true, preds_by_model[name], multi_class='ovr')
    except: a = 0
    print(f'    {DISPLAY.get(name,name):25s}: {a:.4f}')
print(f'\n  Silhouette score: {sil_score:.3f}')
print(f'\n  Generated files:')
for f in sorted(os.listdir(FIG_DIR)):
    if 'clean' in f: print(f'    {FIG_DIR}/{f}')
for f in sorted(os.listdir(TBL_DIR)):
    if 'clean' in f: print(f'    {TBL_DIR}/{f}')

# Save summary JSON
summary = {
    'model_aucs': {name: roc_auc_score(y_true, preds_by_model[name], multi_class='ovr')
                   for name in list(models.keys()) + ['ensemble']},
    'silhouette': sil_score,
    'n_val': len(val_p),
    'cohen_d': {},
}
for col in ['Lab_b', 'yellow_ratio', 'S']:
    mild_v = cdf[cdf['category']=='mild'][col]
    sev_v = cdf[cdf['category']=='severe'][col]
    if len(mild_v)>1 and len(sev_v)>1:
        pooled = np.sqrt((mild_v.std()**2 + sev_v.std()**2)/2)
        summary['cohen_d'][col] = abs(mild_v.mean()-sev_v.mean())/(pooled+1e-6)
with open(os.path.join(BASE, 'results', 'clean_eyelid_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
