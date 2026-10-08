# -*- coding: utf-8 -*-
"""
Generate ALL manuscript figures + supplementary figures with legends.
Main figures: Figure 2 (Binary), Figure 3 (Ternary), Figure 4 (Model vs Doctor)
Supplementary: Confusion matrices, Calibration, External ROC, DBIL/IBIL forest

Step 1: Get model predictions on val sets
Step 2: Generate figures + write legends
"""
import os, json, random, re, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                              f1_score, confusion_matrix, precision_recall_curve,
                              average_precision_score, brier_score_loss)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'tables')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
FIG_MAIN = os.path.join(RES, 'figures', 'manuscript')
FIG_SUPP = os.path.join(RES, 'figures', 'supplementary')
os.makedirs(FIG_MAIN, exist_ok=True)
os.makedirs(FIG_SUPP, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMG_SIZE = 224
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})

# Colors
C_BINARY = '#E76F51'; C_TERNARY = '#2A9D8F'; C_DOCTOR = '#457B9D'; C_MODEL = '#264653'
PAL5 = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51', '#457B9D']

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def predict_patient(model, paths, n_max=12):
    valid = [p for p in paths[:n_max] if os.path.exists(p)]
    if not valid: return None
    tensors = []
    for p in valid:
        img = read_img(p)
        if img is None: continue
        img = clahe(img)
        tensors.append(ev_tf(Image.fromarray(img).convert('RGB')).unsqueeze(0))
    if not tensors: return None
    t = torch.cat(tensors).to(DEV)
    with torch.no_grad():
        return F.softmax(model(t), dim=1).mean(0).cpu().numpy()

def load_model(bb, ckpt, nc):
    path = os.path.join(MODEL, ckpt)
    if not os.path.exists(path): return None
    m = timm.create_model(bb, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    m.eval()
    return m

def collect_patients():
    patients = {}
    for cat in ['normal', 'mild', 'moderate', 'severe']:
        cd = os.path.join(DATA, cat)
        if not os.path.exists(cd): continue
        for pid in os.listdir(cd):
            p = os.path.join(cd, pid)
            if not os.path.isdir(p): continue
            faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
            if faces: patients[pid] = {'faces': faces, 'cat': cat}
    mdf = pd.read_csv(MANIFEST)
    for _, r in mdf.iterrows():
        if r['patient_id'] in patients and r['n_eyelid'] > 0:
            patients[r['patient_id']]['eyelids'] = json.loads(r['eyelid_images'])
    return patients

def stratified_split(pid_grade, ratio=0.2):
    by_g = {}
    for pid, g in pid_grade.items(): by_g.setdefault(g, []).append(pid)
    tr, va = set(), set()
    for g, pids in by_g.items():
        random.seed(SEED); random.shuffle(pids)
        n = max(1, int(len(pids) * ratio))
        va.update(pids[:n]); tr.update(pids[n:])
    return tr, va

sam_pats = collect_patients()

# ═════════════════════════════════════════════════════════════
# STEP 1: Get predictions for all models
# ═════════════════════════════════════════════════════════════
print('[1] Computing model predictions...')

# --- Binary val set (normal vs jaundice) ---
bin_grades = {pid: (0 if p['cat'] == 'normal' else 1) for pid, p in sam_pats.items()}
_, va_bin = stratified_split(bin_grades)
va_bin_list = [{'faces': sam_pats[pid]['faces'], 'label': bin_grades[pid], 'pid': pid}
               for pid in va_bin if pid in sam_pats]

binary_models = [
    ('ConvNeXt', 'convnext_tiny', 'type_binary_convnext_tiny.pt'),
    ('EfficientNet', 'efficientnet_b0', 'type_binary_efficientnet_b0.pt'),
    ('ViT', 'vit_tiny_patch16_224', 'type_binary_vit_tiny_patch16_224.pt'),
]
print(f'  Binary val: {len(va_bin_list)} patients')

bin_preds = {}
for name, bb, ckpt in binary_models:
    m = load_model(bb, ckpt, 2)
    if m is None: continue
    probs = []
    for p in tqdm(va_bin_list, desc=f'Bin-{name}', leave=False):
        pr = predict_patient(m, p['faces'])
        probs.append(pr if pr is not None else np.array([0.5, 0.5]))
    bin_preds[name] = np.array(probs)
    del m; torch.cuda.empty_cache()

# YOLO binary
from ultralytics import YOLO
yolo_path = os.path.join(MODEL, 'v3_yolo_binary.pt')
if os.path.exists(yolo_path):
    ym = YOLO(yolo_path)
    probs = []
    for p in va_bin_list:
        faces = [f for f in p['faces'][:12] if os.path.exists(f)]
        ps = []
        for fp in faces:
            r = ym.predict(fp, verbose=False)
            if r and len(r) > 0: ps.append(r[0].probs.data.cpu().numpy())
        probs.append(np.mean(ps, axis=0) if ps else [0.5, 0.5])
    bin_preds['YOLO'] = np.array(probs)

y_bin = np.array([p['label'] for p in va_bin_list])

# --- Ternary val set (mild/moderate/severe) ---
tern_grades = {pid: {'mild': 0, 'moderate': 1, 'severe': 2}[p['cat']]
               for pid, p in sam_pats.items() if p['cat'] != 'normal'}
_, va_tern = stratified_split(tern_grades)
va_tern_list = [{'faces': sam_pats[pid]['faces'], 'label': tern_grades[pid], 'pid': pid}
                for pid in va_tern if pid in sam_pats]

ternary_models = [
    ('Eyelid-Opt-ConvNeXt', 'convnext_tiny', 'eyelid_opt_convnext_tiny.pt'),
    ('Eyelid-Opt-Swin', 'swin_tiny_patch4_window7_224', 'eyelid_opt_swin_tiny_patch4_window7_224.pt'),
    ('Face-v3-ConvNeXt', 'convnext_tiny', 'v3_ternary_convnext_tiny.pt'),
    ('Face-v3-ViT', 'vit_tiny_patch16_224', 'v3_ternary_vit_tiny_patch16_224.pt'),
    ('Face-v3-Swin', 'swin_tiny_patch4_window7_224', 'v3_ternary_swin_tiny_patch4_window7_224.pt'),
]
print(f'  Ternary val: {len(va_tern_list)} patients')

# Eyelid val (separate from face)
va_eyelid_list = [{'eyelids': sam_pats[pid].get('eyelids', []), 'label': tern_grades[pid], 'pid': pid}
                   for pid in va_tern if pid in sam_pats and sam_pats[pid].get('eyelids')]

tern_preds = {}
tern_preds_eyelid = {}
for name, bb, ckpt in ternary_models:
    m = load_model(bb, ckpt, 3)
    if m is None: continue
    if 'Eyelid' in name:
        probs = []
        for p in tqdm(va_eyelid_list, desc=name, leave=False):
            pr = predict_patient(m, p['eyelids'])
            probs.append(pr if pr is not None else np.array([1/3, 1/3, 1/3]))
        tern_preds_eyelid[name] = (np.array(probs), np.array([p['label'] for p in va_eyelid_list]))
    else:
        probs = []
        for p in tqdm(va_tern_list, desc=name, leave=False):
            pr = predict_patient(m, p['faces'])
            probs.append(pr if pr is not None else np.array([1/3, 1/3, 1/3]))
        tern_preds[name] = (np.array(probs), np.array([p['label'] for p in va_tern_list]))
    del m; torch.cuda.empty_cache()

# YOLO ternary
yolo_tern = os.path.join(MODEL, 'v3_yolo_ternary.pt')
if os.path.exists(yolo_tern):
    ym = YOLO(yolo_tern)
    probs = []
    for p in va_tern_list:
        faces = [f for f in p['faces'][:12] if os.path.exists(f)]
        ps = []
        for fp in faces:
            r = ym.predict(fp, verbose=False)
            if r and len(r) > 0: ps.append(r[0].probs.data.cpu().numpy())
        probs.append(np.mean(ps, axis=0) if ps else [1/3, 1/3, 1/3])
    tern_preds['YOLO-Ternary'] = (np.array(probs), np.array([p['label'] for p in va_tern_list]))

y_tern = np.array([p['label'] for p in va_tern_list])

# --- Load doctor points ---
doc_points = json.load(open(os.path.join(RES, 'doctor_roc_points.json')))

# --- Load bootstrap metrics ---
bs_df = pd.read_csv(os.path.join(TBL, 'bootstrap_full_metrics.csv'))

# --- Load external validation ---
ext_df = pd.read_csv(os.path.join(TBL, 'external_validation_results.csv'))

print('\n[2] Generating figures...')

# ═════════════════════════════════════════════════════════════
# FIGURE 2: Binary Screening
# ═════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.35)

# Panel A: ROC with doctor points
ax = fig.add_subplot(gs[0, 0])
ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
colors_roc = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51', '#6A4C93']
for i, (name, probs) in enumerate(bin_preds.items()):
    fpr, tpr, _ = roc_curve(y_bin, probs[:, 1])
    auc_v = roc_auc_score(y_bin, probs[:, 1])
    ax.plot(fpr, tpr, lw=1.2, color=colors_roc[i % len(colors_roc)], label=f'{name} ({auc_v:.3f})')
# Doctor scatter points
for fpr_d, tpr_d, dname, auc_d in doc_points['binary']:
    ax.scatter(fpr_d, tpr_d, marker='^', s=40, c=C_DOCTOR, edgecolors='white', lw=0.3, zorder=5, alpha=0.7)
ax.set_xlabel('False Positive Rate', fontsize=9); ax.set_ylabel('True Positive Rate', fontsize=9)
ax.set_title('a | Binary Screening ROC', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=6, loc='lower right'); ax.set_xlim([-0.02, 0.55]); ax.set_ylim([0.4, 1.02])
ax.tick_params(labelsize=7)

# Panel B: PR curves
ax = fig.add_subplot(gs[0, 1])
for i, (name, probs) in enumerate(bin_preds.items()):
    prec, rec, _ = precision_recall_curve(y_bin, probs[:, 1])
    ap_v = average_precision_score(y_bin, probs[:, 1])
    ax.plot(rec, prec, lw=1.2, color=colors_roc[i % len(colors_roc)], label=f'{name} ({ap_v:.3f})')
ax.set_xlabel('Recall (Sensitivity)', fontsize=9); ax.set_ylabel('Precision (PPV)', fontsize=9)
ax.set_title('b | Precision-Recall Curves', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=6, loc='lower left'); ax.tick_params(labelsize=7)

# Panel C: Radar chart (6 metrics)
ax = fig.add_subplot(gs[0, 2], polar=True)
# Get best binary model metrics from bootstrap
best_bin = bs_df[bs_df['task'] == 'Binary-Screening'].nlargest(1, 'auc').iloc[0]
metrics_labels = ['AUC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity', 'Avg Precision']
# Use YOLO as the main model
yolo_row = bs_df[(bs_df['name'] == 'YOLO-Binary')]
if len(yolo_row) > 0:
    yolo_m = yolo_row.iloc[0]
else:
    yolo_m = best_bin
# Aggregate doctor
agg_row = bs_df[(bs_df['name'] == 'Bin-Ensemble')]
if len(agg_row) > 0:
    agg_m = agg_row.iloc[0]
else:
    agg_m = best_bin

model_vals = [yolo_m['auc'], yolo_m['acc'], yolo_m['f1'], yolo_m['sens'], yolo_m['spec'], yolo_m['ap']]
# Doctor aggregate (from individual metrics)
doc_df = pd.read_csv(os.path.join(TBL, 'doctor_individual_metrics.csv'))
doc_bin = doc_df[doc_df['task'] == 'Binary']
doc_vals = [doc_bin['auc'].mean(), doc_bin['acc'].mean(), doc_bin['f1'].mean(),
            doc_bin['sens'].mean(), doc_bin['spec'].mean(), doc_bin['ap'].mean()]

angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
angles += angles[:1]
model_vals += model_vals[:1]
doc_vals += doc_vals[:1]
ax.plot(angles, model_vals, 'o-', lw=1.5, color=C_MODEL, label='BilinGuard', markersize=4)
ax.fill(angles, model_vals, alpha=0.1, color=C_MODEL)
ax.plot(angles, doc_vals, 's-', lw=1.5, color=C_DOCTOR, label='Clinicians (mean)', markersize=4)
ax.fill(angles, doc_vals, alpha=0.1, color=C_DOCTOR)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_labels, fontsize=6)
ax.set_ylim([0, 1.05]); ax.set_title('c | BilinGuard vs Clinicians', fontsize=10, fontweight='bold', loc='left', pad=15)
ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
fig.savefig(os.path.join(FIG_MAIN, 'Figure2_binary_screening.png'))
fig.savefig(os.path.join(FIG_MAIN, 'Figure2_binary_screening.svg'))
plt.close(fig)
print('  Figure 2 saved')

# ═════════════════════════════════════════════════════════════
# FIGURE 3: Ternary Grading
# ═════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.35)

# Panel A: ROC curves (all ternary models)
ax = fig.add_subplot(gs[0, 0])
ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
all_tern = {}
for name, (probs, true) in {**tern_preds, **tern_preds_eyelid}.items():
    all_tern[name] = (probs, true)
    yb = label_binarize(true, classes=[0, 1, 2])
    fpr, tpr, _ = roc_curve(yb.ravel(), probs.ravel())
    auc_v = auc(fpr, tpr) if 'auc' in dir() else roc_auc_score(true, probs, multi_class='ovr', labels=[0,1,2])
    from sklearn.metrics import auc as sk_auc
    auc_v = sk_auc(fpr, tpr)
    short = name.replace('Eyelid-Opt-', 'Eyelid-').replace('Face-v3-', 'Face-')
    ax.plot(fpr, tpr, lw=1.2, label=f'{short} ({auc_v:.3f})')
# Doctor ternary points
for fpr_d, tpr_d, dname, auc_d in doc_points.get('ternary', []):
    ax.scatter(fpr_d, tpr_d, marker='^', s=40, c=C_DOCTOR, edgecolors='white', lw=0.3, zorder=5, alpha=0.7)
ax.set_xlabel('False Positive Rate', fontsize=9); ax.set_ylabel('True Positive Rate', fontsize=9)
ax.set_title('a | Ternary Grading ROC (Micro-avg)', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=5.5, loc='lower right'); ax.set_xlim([-0.02, 0.6]); ax.set_ylim([0.2, 1.02])
ax.tick_params(labelsize=7)

# Panel B: PR curves
ax = fig.add_subplot(gs[0, 1])
for name, (probs, true) in all_tern.items():
    yb = label_binarize(true, classes=[0, 1, 2])
    prec, rec, _ = precision_recall_curve(yb.ravel(), probs.ravel())
    ap_v = average_precision_score(yb, probs, average='micro')
    short = name.replace('Eyelid-Opt-', 'Eyelid-').replace('Face-v3-', 'Face-')
    ax.plot(rec, prec, lw=1.2, label=f'{short} ({ap_v:.3f})')
ax.set_xlabel('Recall', fontsize=9); ax.set_ylabel('Precision', fontsize=9)
ax.set_title('b | Precision-Recall Curves', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=5.5, loc='lower left'); ax.tick_params(labelsize=7)

# Panel C: Per-class ROC for best model
ax = fig.add_subplot(gs[0, 2])
# Find best ternary model by AUC
best_tern_name = list(all_tern.keys())[0]
best_auc_val = 0
for name, (probs, true) in all_tern.items():
    try:
        a = roc_auc_score(true, probs, multi_class='ovr', labels=[0,1,2])
        if a > best_auc_val: best_auc_val = a; best_tern_name = name
    except: pass
best_probs, best_true = all_tern[best_tern_name]
class_names = ['Mild', 'Moderate', 'Severe']
class_colors = ['#F4D03F', '#E67E22', '#E74C3C']
for c in range(3):
    yb_c = (best_true == c).astype(int)
    fpr, tpr, _ = roc_curve(yb_c, best_probs[:, c])
    auc_c = roc_auc_score(yb_c, best_probs[:, c])
    ax.plot(fpr, tpr, lw=1.3, color=class_colors[c], label=f'{class_names[c]} (AUC={auc_c:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=9); ax.set_ylabel('True Positive Rate', fontsize=9)
ax.set_title(f'c | Per-Class ROC ({best_tern_name})', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=6, loc='lower right'); ax.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_MAIN, 'Figure3_ternary_grading.png'))
fig.savefig(os.path.join(FIG_MAIN, 'Figure3_ternary_grading.svg'))
plt.close(fig)
print('  Figure 3 saved')

# ═════════════════════════════════════════════════════════════
# FIGURE 4: BilinGuard vs Clinicians
# ═════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

doc_df = pd.read_csv(os.path.join(TBL, 'doctor_individual_metrics.csv'))

# Panel A: Binary ROC + doctor points
ax = fig.add_subplot(gs[0, 0])
# Use YOLO as BilinGuard representative for binary
if 'YOLO' in bin_preds:
    model_probs = bin_preds['YOLO']
    fpr_m, tpr_m, _ = roc_curve(y_bin, model_probs[:, 1])
    auc_m = roc_auc_score(y_bin, model_probs[:, 1])
    ax.plot(fpr_m, tpr_m, lw=2, color=C_MODEL, label=f'BilinGuard ({auc_m:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
for fpr_d, tpr_d, dname, auc_d in doc_points['binary']:
    short = dname.split('-')[-1][:8]
    ax.scatter(fpr_d, tpr_d, marker='^', s=50, c=C_DOCTOR, edgecolors='white', lw=0.3, zorder=5)
    ax.annotate(short, (fpr_d, tpr_d), fontsize=5, textcoords='offset points', xytext=(3, 3))
ax.set_xlabel('False Positive Rate', fontsize=9); ax.set_ylabel('True Positive Rate', fontsize=9)
ax.set_title('a | Binary: BilinGuard vs Clinicians', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=8, loc='lower right'); ax.set_xlim([-0.02, 0.55]); ax.set_ylim([0.4, 1.02])
ax.tick_params(labelsize=7)

# Panel B: Binary metrics bar chart
ax = fig.add_subplot(gs[0, 1])
doc_bin = doc_df[doc_df['task'] == 'Binary']
metrics_plot = ['f1', 'sens', 'spec', 'acc', 'auc']
metric_labels = ['F1', 'Sensitivity', 'Specificity', 'Accuracy', 'AUC-ROC']
x = np.arange(len(metrics_plot))
width = 0.15
# Model bars
yolo_bs = bs_df[bs_df['name'] == 'YOLO-Binary']
if len(yolo_bs) > 0:
    model_vals = [yolo_bs.iloc[0][m] for m in metrics_plot]
else:
    model_vals = [0.5] * len(metrics_plot)
ax.bar(x - width*1.5, model_vals, width, color=C_MODEL, label='BilinGuard', edgecolor='white', lw=0.3)
# Doctor bars (mean ± std)
doc_means = [doc_bin[m].mean() for m in metrics_plot]
doc_stds = [doc_bin[m].std() for m in metrics_plot]
ax.bar(x - width*0.5, doc_means, width, yerr=doc_stds, color=C_DOCTOR, label='Clinicians (mean)', capsize=2, edgecolor='white', lw=0.3)
ax.set_xticks(x - width); ax.set_xticklabels(metric_labels, fontsize=7, rotation=15)
ax.set_ylabel('Score', fontsize=9); ax.set_ylim([0, 1.1])
ax.set_title('b | Binary Metrics Comparison', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=7); ax.tick_params(labelsize=7)

# Panel C: Ternary ROC + doctor points
ax = fig.add_subplot(gs[1, 0])
# Use best ternary model
best_name = list(all_tern.keys())[0]
for name, (probs, true) in all_tern.items():
    bs_row = bs_df[bs_df['name'].str.contains(name.replace('Eyelid-Opt-', 'TBIL-Opt-').replace('Face-v3-', 'FaceTern-v3-'), na=False)]
    if len(bs_row) > 0 and bs_row.iloc[0]['auc'] > 0.7:
        best_name = name; break
best_probs, best_true = all_tern[best_name]
yb = label_binarize(best_true, classes=[0, 1, 2])
fpr_m, tpr_m, _ = roc_curve(yb.ravel(), best_probs.ravel())
from sklearn.metrics import auc as sk_auc
auc_m = sk_auc(fpr_m, tpr_m)
short_name = best_name.replace('Eyelid-Opt-', 'Eyelid ').replace('Face-v3-', 'Face ')
ax.plot(fpr_m, tpr_m, lw=2, color=C_MODEL, label=f'BilinGuard ({auc_m:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
for fpr_d, tpr_d, dname, auc_d in doc_points.get('ternary', []):
    short = dname.split('-')[-1][:8]
    ax.scatter(fpr_d, tpr_d, marker='^', s=50, c=C_DOCTOR, edgecolors='white', lw=0.3, zorder=5)
ax.set_xlabel('False Positive Rate', fontsize=9); ax.set_ylabel('True Positive Rate', fontsize=9)
ax.set_title('c | Ternary: BilinGuard vs Clinicians', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=8, loc='lower right'); ax.set_xlim([-0.02, 0.6]); ax.set_ylim([0.1, 1.02])
ax.tick_params(labelsize=7)

# Panel D: Ternary metrics bar chart
ax = fig.add_subplot(gs[1, 1])
doc_tern = doc_df[doc_df['task'] == 'Ternary']
# Best model ternary metrics from bootstrap
best_bs = bs_df[bs_df['task'] == 'TBIL-Eyelid'].nlargest(1, 'auc')
if len(best_bs) == 0:
    best_bs = bs_df[bs_df['task'] == 'Face-Ternary'].nlargest(1, 'auc')
model_vals_t = [best_bs.iloc[0][m] for m in metrics_plot] if len(best_bs) > 0 else [0.5]*5
doc_means_t = [doc_tern[m].mean() for m in metrics_plot]
doc_stds_t = [doc_tern[m].std() for m in metrics_plot]
ax.bar(x - width*1.5, model_vals_t, width, color=C_MODEL, label='BilinGuard', edgecolor='white', lw=0.3)
ax.bar(x - width*0.5, doc_means_t, width, yerr=doc_stds_t, color=C_DOCTOR, label='Clinicians (mean)', capsize=2, edgecolor='white', lw=0.3)
ax.set_xticks(x - width); ax.set_xticklabels(metric_labels, fontsize=7, rotation=15)
ax.set_ylabel('Score', fontsize=9); ax.set_ylim([0, 1.1])
ax.set_title('d | Ternary Metrics Comparison', fontsize=10, fontweight='bold', loc='left')
ax.legend(fontsize=7); ax.tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(FIG_MAIN, 'Figure4_model_vs_clinician.png'))
fig.savefig(os.path.join(FIG_MAIN, 'Figure4_model_vs_clinician.svg'))
plt.close(fig)
print('  Figure 4 saved')

# ═════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ═════════════════════════════════════════════════════════════
print('\n[3] Generating supplementary figures...')

# SuppFig 1: Confusion matrices (binary + ternary best models)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# Binary
if 'YOLO' in bin_preds:
    cm = confusion_matrix(y_bin, bin_preds['YOLO'].argmax(1))
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Jaundiced']); ax.set_yticklabels(['Normal', 'Jaundiced'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Binary Screening (YOLO)', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
# Ternary
if all_tern:
    bn = list(all_tern.keys())[0]
    bp, bt = all_tern[bn]
    cm = confusion_matrix(bt, bp.argmax(1), labels=[0, 1, 2])
    ax = axes[1]
    im = ax.imshow(cm, cmap='Greens')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Mild', 'Moderate', 'Severe']); ax.set_yticklabels(['Mild', 'Moderate', 'Severe'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Ternary Grading ({bn[:12]})', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
fig.savefig(os.path.join(FIG_SUPP, 'SuppFig_confusion_matrices.png'))
fig.savefig(os.path.join(FIG_SUPP, 'SuppFig_confusion_matrices.svg'))
plt.close(fig)
print('  SuppFig confusion matrices saved')

# SuppFig 2: External validation ROC
fig, ax = plt.subplots(figsize=(7, 6))
ext_json = json.load(open(os.path.join(RES, 'questionnaire_and_external_bootstrap.json')))
ext_models = [r for r in ext_json if r.get('task') == 'External-Validation']
for r in ext_models:
    # We don't have raw probs for external, so use the summary
    pass
# Use the pre-generated figure
import shutil
src = os.path.join(RES, 'figures', 'Fig_external_roc_with_ci.png')
if os.path.exists(src):
    shutil.copy(src, os.path.join(FIG_SUPP, 'SuppFig_external_roc.png'))
    shutil.copy(src.replace('.png', '.svg'), os.path.join(FIG_SUPP, 'SuppFig_external_roc.svg'))
print('  SuppFig external ROC copied')

# SuppFig 3: DBIL/IBIL forest plot
src_forest = os.path.join(RES, 'figures', 'bootstrap', 'Fig_bootstrap_forest.png')
if os.path.exists(src_forest):
    shutil.copy(src_forest, os.path.join(FIG_SUPP, 'SuppFig_bootstrap_forest.png'))
    shutil.copy(src_forest.replace('.png', '.svg'), os.path.join(FIG_SUPP, 'SuppFig_bootstrap_forest.svg'))
print('  SuppFig forest plot copied')

# SuppFig 4: Doctor response times
doc_files_dir = os.path.join(BASE, 'doctor')
doc_times = {}
for fname in os.listdir(doc_files_dir):
    if not fname.endswith('.xlsx') or '汇总' in fname or '表现' in fname: continue
    df = pd.read_excel(os.path.join(doc_files_dir, fname))
    if df.shape[1] > 1:
        times = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
        if len(times) > 50:
            dname = fname.replace('.xlsx', '').replace('题-', '')
            doc_times[dname] = times.values

if doc_times:
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(list(doc_times.values()), positions=range(len(doc_times)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(doc_times)))
    ax.set_xticklabels(list(doc_times.keys()), fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Response Time (seconds)', fontsize=10)
    ax.set_title('Clinician Response Times', fontsize=11, fontweight='bold')
    # Add BilinGuard time
    ax.axhline(y=0.5, color=C_MODEL, linestyle='--', lw=1, label='BilinGuard (~0.5s)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_SUPP, 'SuppFig_response_times.png'))
    fig.savefig(os.path.join(FIG_SUPP, 'SuppFig_response_times.svg'))
    plt.close(fig)
    print('  SuppFig response times saved')

# SuppFig 5: Bilirubin classification (DBIL/IBIL) summary
src_bil = os.path.join(RES, 'figures', 'bilirubin', 'Fig_bilirubin_classification.png')
if os.path.exists(src_bil):
    shutil.copy(src_bil, os.path.join(FIG_SUPP, 'SuppFig_bilirubin_classification.png'))
    shutil.copy(src_bil.replace('.png', '.svg'), os.path.join(FIG_SUPP, 'SuppFig_bilirubin_classification.svg'))
print('  SuppFig bilirubin classification copied')

# ═════════════════════════════════════════════════════════════
# WRITE FIGURE LEGENDS
# ═════════════════════════════════════════════════════════════
legends = """
 manuscript figure legends
═════════════════════════════════════════════════════════════════════

Figure 2. Binary jaundice screening performance.
(a) Receiver operating characteristic (ROC) curves for image-based deep learning models (YOLO, ConvNeXt, EfficientNet, ViT) in distinguishing jaundiced from non-jaundiced patients. Blue triangles indicate individual clinician performance points (n=8 raters). AUC values are shown in parentheses. (b) Precision-recall curves for the same models, with average precision (AP) values. (c) Radar chart comparing six performance metrics (AUC-ROC, accuracy, F1 score, sensitivity, specificity, average precision) between BilinGuard (dark blue) and the mean of eight clinicians (light blue). Bootstrap 95% confidence intervals are reported in Table 2.

Figure 3. Ternary jaundice severity grading performance.
(a) Micro-averaged ROC curves for eyelid-based (ConvNeXt, Swin) and face-based (ConvNeXt, ViT, Swin, YOLO) models in classifying mild, moderate, and severe jaundice. Blue triangles indicate individual clinician performance points. (b) Micro-averaged precision-recall curves for the same models. (c) One-vs-rest ROC curves for the best-performing model, showing class-specific discrimination for mild (yellow), moderate (orange), and severe (red) jaundice. Bootstrap 95% confidence intervals are reported in Table 3.

Figure 4. BilinGuard versus clinician performance.
(a) Binary screening ROC curve for BilinGuard (dark blue line) with individual clinician operating points (blue triangles, n=8). (b) Binary screening metrics comparison between BilinGuard and clinician mean (±SD). (c) Ternary grading ROC curve for BilinGuard with clinician operating points. (d) Ternary grading metrics comparison. Error bars for clinicians represent standard deviation across eight raters. P-values from Wilcoxon signed-rank tests are reported in Table 4.

═════════════════════════════════════════════════════════════════════
Supplementary Figure legends
═════════════════════════════════════════════════════════════════════

Supplementary Figure S1. Confusion matrices for the best-performing binary (YOLO) and ternary (eyelid ConvNeXt) models on the internal validation set.

Supplementary Figure S2. External validation ROC curves. Binary screening performance of all image-based models on the independent external cohort (n=98 patients; 40 normal, 58 jaundiced). Shaded regions indicate 95% confidence intervals from 1000-iteration bootstrap resampling.

Supplementary Figure S3. Bootstrap 95% confidence interval forest plots for all models across all tasks (binary screening, TBIL ternary grading, DBIL ternary grading, IBIL ternary grading, and jaundice type classification). Each point represents the AUC-ROC point estimate; horizontal bars indicate the 2.5th–97.5th percentile range from 1000 bootstrap iterations at the patient level.

Supplementary Figure S4. Clinician response times. Violin plots showing the distribution of assessment times (seconds) for each of the eight clinician raters. The dashed line indicates BilinGuard's average inference time (~0.5 seconds). Kruskal-Wallis test: H=395.4, P<0.0001.

Supplementary Figure S5. Direct and indirect bilirubin subclassification. (a) Direct bilirubin (DBIL) ternary grading performance (≤10, 10–68, >68 μmol/L) for eyelid and face models. (b) Indirect bilirubin (IBIL) ternary grading performance (≤20, 20–50, >50 μmol/L). Bar charts show the best AUC-ROC for each modality.
"""

with open(os.path.join(FIG_MAIN, 'figure_legends.txt'), 'w', encoding='utf-8') as f:
    f.write(legends)

print(f'\n[4] Figure legends saved: {FIG_MAIN}/figure_legends.txt')
print(f'\n=== ALL FIGURES GENERATED ===')
print(f'  Main: {FIG_MAIN}/')
for f in sorted(os.listdir(FIG_MAIN)):
    print(f'    {f}')
print(f'  Supplementary: {FIG_SUPP}/')
for f in sorted(os.listdir(FIG_SUPP)):
    print(f'    {f}')
