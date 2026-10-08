# -*- coding: utf-8 -*-
"""
Comprehensive comparison v2: evaluates ALL models (v3 face, type classification,
YOLO, eyelid) and produces a unified master table + Nature-style figure.

Data sources:
  - Face models (v3/type/YOLO): data/face_v4/{cat}/{pid}/_face.jpg
  - Eyelid models: results/tables/master_comparison.csv (already computed)
  - External: results/tables/external_validation_results.csv (already computed)
"""
import os, json, random, re, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from docx import Document
from docx.shared import Pt

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL_DIR = os.path.join(BASE, 'models')
FIG_DIR = os.path.join(BASE, 'results', 'figures')
TBL_DIR = os.path.join(BASE, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TBL_DIR, exist_ok=True)
DEVICE = torch.device('cuda')
IMG_SIZE = 224; SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_img(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))


def safe_auc(y, p, nc=2):
    try:
        if nc == 2:
            return roc_auc_score(y, p[:, 1]) if len(np.unique(y)) > 1 else 0
        return roc_auc_score(y, p, multi_class='ovr') if len(np.unique(y)) > 1 else 0
    except Exception:
        return 0


# ── Build patient lists from face_v4 ────────────────
print('[1] Building patient lists from face_v4...')

def collect_patients(cat_dir, cat_name):
    patients = {}
    if not os.path.exists(cat_dir):
        return patients
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p):
            continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            patients[pid] = faces
    return patients

normal_pats = collect_patients(os.path.join(DATA, 'normal'), 'normal')
mild_pats = collect_patients(os.path.join(DATA, 'mild'), 'mild')
mod_pats = collect_patients(os.path.join(DATA, 'moderate'), 'moderate')
sev_pats = collect_patients(os.path.join(DATA, 'severe'), 'severe')
all_jaundice = {**mild_pats, **mod_pats, **sev_pats}

print(f'  Normal: {len(normal_pats)}, Mild: {len(mild_pats)}, Moderate: {len(mod_pats)}, Severe: {len(sev_pats)}')

# ── Binary split (normal vs all jaundice) ────────────────────
def split_keys(d, ratio=0.2):
    keys = sorted(d.keys()); random.seed(SEED); random.shuffle(keys)
    n = max(1, int(len(keys) * ratio))
    return keys[n:], keys[:n]

tr_n, va_n = split_keys(normal_pats)
all_jaundice_keys = sorted(all_jaundice.keys()); random.seed(SEED); random.shuffle(all_jaundice_keys)
nj = max(1, int(len(all_jaundice_keys) * 0.2))
tr_j, va_j = all_jaundice_keys[nj:], all_jaundice_keys[:nj]

# Binary val patients
bin_val = [{'id': k, 'label': 0, 'faces': normal_pats[k]} for k in va_n] + \
          [{'id': k, 'label': 1, 'faces': all_jaundice[k]} for k in va_j]
y_binary = np.array([p['label'] for p in bin_val])
print(f'  Binary val: {len(bin_val)} (normal={sum(y_binary==0)}, jaundice={sum(y_binary==1)})')

# ── Ternary split (mild/moderate/severe) ─────────────────────
_, va_mild = split_keys(mild_pats)
_, va_mod = split_keys(mod_pats)
_, va_sev = split_keys(sev_pats)

ternary_map = {'mild': 0, 'moderate': 1, 'severe': 2}
tern_val = [{'id': k, 'label': 0, 'faces': mild_pats[k]} for k in va_mild] + \
           [{'id': k, 'label': 1, 'faces': mod_pats[k]} for k in va_mod] + \
           [{'id': k, 'label': 2, 'faces': sev_pats[k]} for k in va_sev]
y_ternary = np.array([p['label'] for p in tern_val])
y_tern_bin = label_binarize(y_ternary, classes=[0, 1, 2])
print(f'  Ternary val: {len(tern_val)} (mild={sum(y_ternary==0)}, mod={sum(y_ternary==1)}, sev={sum(y_ternary==2)})')

# ── Jaundice type (hepato vs chol) ───────────────────────────
print('[2] Loading jaundice type labels...')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
df_excel = pd.read_excel(EXCEL)
df_excel['hid_str'] = df_excel['2、患者住院号'].astype(str)

def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None

def strip_zeros(s):
    return s.lstrip('0') if s else s

def get_jtype(patient_id):
    hid = extract_hid(patient_id)
    if not hid:
        return None
    hid_norm = strip_zeros(hid)
    rows = df_excel[df_excel['hid_str'].apply(strip_zeros) == hid_norm]
    if len(rows) == 0:
        rows = df_excel[df_excel['hid_str'].apply(strip_zeros).str.endswith(hid_norm[-6:])]
    if len(rows) > 0:
        return int(rows.iloc[0]['25、黄疸类型'])
    return None

hep_pats, chol_pats = {}, {}
for pid, faces in all_jaundice.items():
    jt = get_jtype(pid)
    if jt in (1, 2):
        hep_pats[pid] = faces
    elif jt == 3:
        chol_pats[pid] = faces

_, va_hep = split_keys(hep_pats)
_, va_chol = split_keys(chol_pats)
type_val = [{'id': k, 'label': 0, 'faces': hep_pats[k]} for k in va_hep] + \
           [{'id': k, 'label': 1, 'faces': chol_pats[k]} for k in va_chol]
y_type = np.array([p['label'] for p in type_val])
print(f'  Type val: {len(type_val)} (hepato={sum(y_type==0)}, chol={sum(y_type==1)})')


# ── Prediction helper ────────────────────────────────────────
def predict_patient_probs(model, patient, n_max=12):
    faces = [f for f in patient['faces'][:n_max] if os.path.exists(f)]
    if not faces:
        return None
    ts = [eval_tf(load_img(f)).unsqueeze(0) for f in faces]
    t = torch.cat(ts).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        return F.softmax(out, dim=1).mean(0).cpu().numpy()


def eval_model(ckpt_name, backbone, nc, val_set, y_true):
    ckpt = os.path.join(MODEL_DIR, ckpt_name)
    if not os.path.exists(ckpt):
        return None, None
    try:
        m = timm.create_model(backbone, pretrained=False, num_classes=nc).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
    except Exception as e:
        print(f'    SKIP {ckpt_name}: {e}'); return None, None
    probs = []
    for p in tqdm(val_set, desc=ckpt_name, leave=False):
        pr = predict_patient_probs(m, p)
        if pr is None:
            pr = np.array([1.0 / nc] * nc)
        probs.append(pr)
    probs = np.array(probs)
    del m; torch.cuda.empty_cache()
    pred = probs.argmax(1)
    return probs, {'auc': safe_auc(y_true, probs, nc),
                    'acc': accuracy_score(y_true, pred),
                    'f1': f1_score(y_true, pred, average='macro')}


# ── Evaluate all face models ─────────────────────────────────
print('\n[3] Evaluating face models on internal validation...')
results = []

# Binary face models
binary_face_models = [
    ('v3_binary_convnext', 'convnext_tiny', 'v3_binary_convnext_tiny.pt'),
    ('v3_binary_vit', 'vit_tiny_patch16_224', 'v3_binary_vit_tiny_patch16_224.pt'),
    ('v3_binary_efficientnet', 'efficientnet_b0', 'v3_binary_efficientnet_b0.pt'),
    ('type_binary_convnext', 'convnext_tiny', 'type_binary_convnext_tiny.pt'),
    ('type_binary_vit', 'vit_tiny_patch16_224', 'type_binary_vit_tiny_patch16_224.pt'),
    ('type_binary_efficientnet', 'efficientnet_b0', 'type_binary_efficientnet_b0.pt'),
    ('type_binary_swin', 'swin_tiny_patch4_window7_224', 'type_binary_swin_tiny_patch4_window7_224.pt'),
]

bin_preds = {}
for name, bb, ckpt in binary_face_models:
    probs, met = eval_model(ckpt, bb, 2, bin_val, y_binary)
    if met:
        bin_preds[name] = probs
        results.append({'Task': 'Binary Screening (Face)', 'Algorithm': name.replace('_', ' ').title(),
                         'Input': 'Face (SAM v3)', 'AUC': met['auc'], 'Accuracy': met['acc'], 'F1': met['f1']})
        print(f'  {name}: AUC={met["auc"]:.4f}')

# Binary ensemble
if len(bin_preds) >= 2:
    ens = np.mean(list(bin_preds.values()), axis=0)
    pred = ens.argmax(1)
    results.append({'Task': 'Binary Screening (Face)', 'Algorithm': 'Face Ensemble',
                     'Input': 'Face (SAM v3)', 'AUC': safe_auc(y_binary, ens, 2),
                     'Accuracy': accuracy_score(y_binary, pred), 'F1': f1_score(y_binary, pred, average='macro')})

# Ternary face models
ternary_face_models = [
    ('v3_ternary_convnext', 'convnext_tiny', 'v3_ternary_convnext_tiny.pt'),
    ('v3_ternary_vit', 'vit_tiny_patch16_224', 'v3_ternary_vit_tiny_patch16_224.pt'),
    ('v3_ternary_efficientnet', 'efficientnet_b0', 'v3_ternary_efficientnet_b0.pt'),
    ('v3_ternary_swin', 'swin_tiny_patch4_window7_224', 'v3_ternary_swin_tiny_patch4_window7_224.pt'),
    ('v3_ternary_yf', 'yf_custom', 'v3_ternary_yf.pt'),
]

tern_preds = {}
for name, bb, ckpt in ternary_face_models:
    if bb == 'yf_custom':
        ckpt_path = os.path.join(MODEL_DIR, ckpt)
        if not os.path.exists(ckpt_path):
            continue
        # YellowFeatures model
        import torch.nn as nn
        class YF(nn.Module):
            def __init__(self, nc=3):
                super().__init__()
                self.c1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
                self.c2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
                self.c3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
                self.p = nn.AdaptiveAvgPool2d(1)
                self.f = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, nc))
            def forward(self, x):
                return self.f(self.p(self.c3(self.c2(self.c1(x)))).flatten(1))
        m = YF(3).to(DEVICE)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        m.eval()
        probs = []
        for p in tqdm(tern_val, desc=ckpt, leave=False):
            pr = predict_patient_probs(m, p)
            if pr is None:
                pr = np.array([1/3, 1/3, 1/3])
            probs.append(pr)
        probs = np.array(probs)
        del m; torch.cuda.empty_cache()
    else:
        probs, met = eval_model(ckpt, bb, 3, tern_val, y_ternary)
        if met is None:
            continue
    tern_preds[name] = probs
    met = {'auc': safe_auc(y_ternary, probs, 3),
           'acc': accuracy_score(y_ternary, probs.argmax(1)),
           'f1': f1_score(y_ternary, probs.argmax(1), average='macro')}
    results.append({'Task': 'Ternary Grading (Face)', 'Algorithm': name.replace('_', ' ').title(),
                     'Input': 'Face (SAM v3)', 'AUC': met['auc'], 'Accuracy': met['acc'], 'F1': met['f1']})
    print(f'  {name}: AUC={met["auc"]:.4f}')

# Ternary ensemble
if len(tern_preds) >= 2:
    ens = np.mean(list(tern_preds.values()), axis=0)
    pred = ens.argmax(1)
    results.append({'Task': 'Ternary Grading (Face)', 'Algorithm': 'Face Ensemble',
                     'Input': 'Face (SAM v3)', 'AUC': safe_auc(y_ternary, ens, 3),
                     'Accuracy': accuracy_score(y_ternary, pred), 'F1': f1_score(y_ternary, pred, average='macro')})
    print(f'  Face Ensemble: AUC={safe_auc(y_ternary, ens, 3):.4f}')

# Type classification models
type_models = [
    ('type_class_convnext', 'convnext_tiny', 'type_class_convnext_tiny.pt'),
    ('type_class_vit', 'vit_tiny_patch16_224', 'type_class_vit_tiny_patch16_224.pt'),
    ('type_class_efficientnet', 'efficientnet_b0', 'type_class_efficientnet_b0.pt'),
    ('type_class_swin', 'swin_tiny_patch4_window7_224', 'type_class_swin_tiny_patch4_window7_224.pt'),
]

type_preds = {}
for name, bb, ckpt in type_models:
    probs, met = eval_model(ckpt, bb, 2, type_val, y_type)
    if met:
        type_preds[name] = probs
        results.append({'Task': 'Jaundice Type (Hepato vs Chol)', 'Algorithm': name.replace('_', ' ').title(),
                         'Input': 'Face (SAM v3)', 'AUC': met['auc'], 'Accuracy': met['acc'], 'F1': met['f1']})
        print(f'  {name}: AUC={met["auc"]:.4f}')

if len(type_preds) >= 2:
    ens = np.mean(list(type_preds.values()), axis=0)
    pred = ens.argmax(1)
    results.append({'Task': 'Jaundice Type (Hepato vs Chol)', 'Algorithm': 'Type Ensemble',
                     'Input': 'Face (SAM v3)', 'AUC': safe_auc(y_type, ens, 2),
                     'Accuracy': accuracy_score(y_type, pred), 'F1': f1_score(y_type, pred, average='macro')})

# ── Load pre-computed results (eyelid + external) ────────────
print('\n[4] Loading pre-computed eyelid and external results...')

# Eyelid results
eyelid_csv = os.path.join(TBL_DIR, 'master_comparison.csv')
if os.path.exists(eyelid_csv):
    edf = pd.read_csv(eyelid_csv)
    for _, r in edf.iterrows():
        results.append({'Task': str(r['Task']), 'Algorithm': str(r['Algorithm']),
                         'Input': str(r.get('Input', 'Eyelid')), 'AUC': r['AUC'],
                         'Accuracy': r['Accuracy'], 'F1': r['F1']})

# External results
ext_csv = os.path.join(TBL_DIR, 'external_validation_results.csv')
if os.path.exists(ext_csv):
    edf2 = pd.read_csv(ext_csv)
    for _, r in edf2.iterrows():
        results.append({'Task': 'External Validation (Binary)', 'Algorithm': str(r['Algorithm']),
                         'Input': 'Face (External)', 'AUC': r['AUC'],
                         'Accuracy': r['Accuracy'], 'F1': r['F1']})

# ── Save master table ────────────────────────────────────────
rdf = pd.DataFrame(results)
rdf = rdf.round(4)
rdf = rdf.sort_values(['Task', 'AUC'], ascending=[True, False])
rdf.to_csv(os.path.join(TBL_DIR, 'master_comparison_v2.csv'), index=False, encoding='utf-8-sig')

# DOCX
doc = Document()
doc.add_heading('Table: Comprehensive Model Comparison (All Tasks)', level=2)
doc.add_paragraph(f'Internal validation: Binary n={len(bin_val)}, Ternary n={len(tern_val)}, Type n={len(type_val)}. '
                   f'Split SEED={SEED}. External validation: n=98 (40 normal, 58 jaundice).')
t = doc.add_table(rows=1, cols=len(rdf.columns))
t.style = 'Light Grid Accent 1'
for i, c in enumerate(rdf.columns):
    t.rows[0].cells[i].text = c
    for p in t.rows[0].cells[i].paragraphs:
        for r in p.runs:
            r.bold = True; r.font.size = Pt(8)
for _, row in rdf.iterrows():
    cells = t.add_row().cells
    for i, v in enumerate(row):
        cells[i].text = str(v)
        for p in cells[i].paragraphs:
            for r in p.runs:
                r.font.size = Pt(8)
doc.save(os.path.join(TBL_DIR, 'Table_master_comparison_v2.docx'))

# ── Print summary ────────────────────────────────────────────
print('\n' + '=' * 80)
print('  COMPREHENSIVE MODEL COMPARISON (ALL TASKS)')
print('=' * 80)
print(f'\n{"Task":<35} {"Algorithm":<28} {"AUC":>6} {"Acc":>6} {"F1":>6}')
print('-' * 80)
for _, r in rdf.iterrows():
    print(f'{str(r["Task"]):<35} {str(r["Algorithm"]):<28} {r["AUC"]:>6.4f} {r["Accuracy"]:>6.4f} {r["F1"]:>6.4f}')
print('-' * 80)

# ── Figure: Nature-style comprehensive ───────────────────────
print('\n[5] Generating Nature-style figure...')
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# Panel a: Ternary face ROC (v3 models)
ax = fig.add_subplot(gs[0, 0])
if tern_preds:
    for name, probs in tern_preds.items():
        fpr, tpr, _ = roc_curve(y_tern_bin.ravel(), probs.ravel())
        a = auc(fpr, tpr)
        lbl = name.replace('v3_ternary_', '').replace('_', ' ').title()
        ax.plot(fpr, tpr, lw=1.3, label=f'{lbl} ({a:.3f})')
    if len(tern_preds) >= 2:
        ens = np.mean(list(tern_preds.values()), axis=0)
        fpr, tpr, _ = roc_curve(y_tern_bin.ravel(), ens.ravel())
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, color='#1D3557', label=f'Ensemble ({a:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
ax.set_title('a | Ternary Grading — Face ROC', fontsize=9, fontweight='bold', loc='left')
ax.set_xlabel('False Positive Rate', fontsize=8); ax.set_ylabel('True Positive Rate', fontsize=8)
ax.legend(fontsize=6, loc='lower right'); ax.tick_params(labelsize=7)

# Panel b: Jaundice type ROC
ax = fig.add_subplot(gs[0, 1])
if type_preds:
    for name, probs in type_preds.items():
        fpr, tpr, _ = roc_curve(y_type, probs[:, 1])
        a = auc(fpr, tpr)
        lbl = name.replace('type_class_', '').replace('_', ' ').title()
        ax.plot(fpr, tpr, lw=1.3, label=f'{lbl} ({a:.3f})')
    if len(type_preds) >= 2:
        ens = np.mean(list(type_preds.values()), axis=0)
        fpr, tpr, _ = roc_curve(y_type, ens[:, 1])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, color='#E76F51', label=f'Ensemble ({a:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.4, alpha=0.3)
ax.set_title('b | Jaundice Type (Hepato vs Chol) ROC', fontsize=9, fontweight='bold', loc='left')
ax.set_xlabel('False Positive Rate', fontsize=8); ax.set_ylabel('True Positive Rate', fontsize=8)
ax.legend(fontsize=6, loc='lower right'); ax.tick_params(labelsize=7)

# Panel c: External validation ROC
ax = fig.add_subplot(gs[0, 2])
ext_json = os.path.join(BASE, 'results', 'external_validation_results.json')
if os.path.exists(ext_json):
    ext_df = pd.read_json(ext_json)
    task_groups = ext_df.groupby('Algorithm')
    colors = plt.cm.Set2(np.linspace(0, 1, len(task_groups)))
    for (alg, grp), col in zip(task_groups, colors):
        ax.barh(range(len(grp)), grp['AUC'].values, color=col, height=0.6, label=alg)
    ax.set_yticks(range(len(ext_df)))
    ax.set_yticklabels([f'{r["Algorithm"]}' for _, r in ext_df.iterrows()], fontsize=6)
    ax.set_xlabel('AUC-ROC', fontsize=8)
    ax.set_title('c | External Validation (Binary)', fontsize=9, fontweight='bold', loc='left')
    ax.set_xlim([0, 1]); ax.tick_params(labelsize=7)

# Panel d: All AUCs bar chart (top 20)
ax = fig.add_subplot(gs[1, :2])
top = rdf.nlargest(25, 'AUC')
colors_map = {'Binary': '#457B9D', 'Face': '#2A9D8F', 'Eyelid': '#E76F51',
              'Type': '#6A4C93', 'External': '#F4A261', 'Combined': '#264653'}
def get_color(task):
    for k, v in colors_map.items():
        if k.lower() in task.lower():
            return v
    return 'gray'
bar_colors = [get_color(str(t)) for t in top['Task']]
bars = ax.barh(range(len(top)), top['AUC'], color=bar_colors, height=0.7, edgecolor='white', linewidth=0.4)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f'{r["Algorithm"]}'[:25] for _, r in top.iterrows()], fontsize=6)
ax.set_xlabel('AUC-ROC', fontsize=8)
ax.set_title('d | Top 25 Models by AUC', fontsize=9, fontweight='bold', loc='left')
ax.set_xlim([0.4, 1.0]); ax.invert_yaxis()
for bar, val in zip(bars, top['AUC']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=5.5)

# Panel e: Task summary (best AUC per task)
ax = fig.add_subplot(gs[1, 2])
task_best = rdf.groupby('Task')['AUC'].max().sort_values(ascending=True)
bars = ax.barh(range(len(task_best)), task_best.values,
               color=[get_color(t) for t in task_best.index], height=0.6)
ax.set_yticks(range(len(task_best)))
ax.set_yticklabels([t.replace(' (', '\n(').replace('Grading ', '') for t in task_best.index], fontsize=6)
ax.set_xlabel('Best AUC-ROC', fontsize=8)
ax.set_title('e | Best AUC per Task', fontsize=9, fontweight='bold', loc='left')
ax.set_xlim([0.4, 1.0])
for bar, val in zip(bars, task_best.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=7)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'Fig_master_comparison_v2.png')
fig.savefig(fig_path); fig.savefig(fig_path.replace('.png', '.svg'))
plt.close(fig)
print(f'  Saved: {fig_path}')

print(f'\n  Table: {TBL_DIR}/master_comparison_v2.csv')
print(f'  Table: {TBL_DIR}/Table_master_comparison_v2.docx')
print(f'  Figure: {fig_path}')
