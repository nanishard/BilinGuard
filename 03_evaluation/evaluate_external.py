# -*- coding: utf-8 -*-
"""
External Validation Evaluation
Evaluates binary (normal vs jaundiced) models on the external validation set.
External set: data/external_processed_v3/{jaundice,normal}/{pid}/frame_*_face.jpg

Reports patient-level AUC/Acc/F1 for each model + ensemble.
"""
import os, json, random, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\research\人脸识别营养\传染科'
EXT = os.path.join(BASE, 'data', 'external_processed_v3')
MODEL_DIR = os.path.join(BASE, 'models')
FIG_DIR = os.path.join(BASE, 'results', 'figures')
TBL_DIR = os.path.join(BASE, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TBL_DIR, exist_ok=True)
DEVICE = torch.device('cuda')
IMG_SIZE = 224; SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_img(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))


# ── Load external data ───────────────────────────────────────
print('[1] Loading external validation data...')
patients = []
for label, lbl_int in [('normal', 0), ('jaundice', 1)]:
    cat_dir = os.path.join(EXT, label)
    if not os.path.exists(cat_dir):
        print(f'  WARNING: {cat_dir} not found'); continue
    for pid in sorted(os.listdir(cat_dir)):
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f and f.endswith('.jpg')])
        if faces:
            patients.append({'id': pid, 'label': lbl_int, 'faces': faces})

print(f'  External: {len(patients)} patients ({sum(p["label"]==0 for p in patients)} normal, {sum(p["label"]==1 for p in patients)} jaundice)')
y_true = np.array([p['label'] for p in patients])


def predict_patient(model, patient, n_max=12):
    faces = [p for p in patient['faces'][:n_max] if os.path.exists(p)]
    if not faces:
        return np.array([0.5, 0.5])
    ts = [eval_tf(load_img(f)).unsqueeze(0) for f in faces]
    t = torch.cat(ts).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        return F.softmax(out, dim=1).mean(0).cpu().numpy()


# ── Model configs (binary face models only) ──────────────────
MODEL_CONFIGS = {
    'type_binary_convnext': {'backbone': 'convnext_tiny', 'ckpt': 'type_binary_convnext_tiny.pt'},
    'type_binary_vit': {'backbone': 'vit_tiny_patch16_224', 'ckpt': 'type_binary_vit_tiny_patch16_224.pt'},
    'type_binary_efficientnet': {'backbone': 'efficientnet_b0', 'ckpt': 'type_binary_efficientnet_b0.pt'},
    'type_binary_swin': {'backbone': 'swin_tiny_patch4_window7_224', 'ckpt': 'type_binary_swin_tiny_patch4_window7_224.pt'},
    'v3_binary_convnext': {'backbone': 'convnext_tiny', 'ckpt': 'v3_binary_convnext_tiny.pt'},
    'v3_binary_vit': {'backbone': 'vit_tiny_patch16_224', 'ckpt': 'v3_binary_vit_tiny_patch16_224.pt'},
    'v3_binary_efficientnet': {'backbone': 'efficientnet_b0', 'ckpt': 'v3_binary_efficientnet_b0.pt'},
}

# ── Evaluate ─────────────────────────────────────────────────
print('\n[2] Evaluating models on external set...')
results = []
all_preds = {}

for name, cfg in MODEL_CONFIGS.items():
    ckpt = os.path.join(MODEL_DIR, cfg['ckpt'])
    if not os.path.exists(ckpt):
        print(f'  SKIP {name}: checkpoint not found')
        continue
    try:
        m = timm.create_model(cfg['backbone'], pretrained=False, num_classes=2).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
    except Exception as e:
        print(f'  SKIP {name}: {e}')
        continue

    probs = np.array([predict_patient(m, p) for p in tqdm(patients, desc=name, leave=False)])
    all_preds[name] = probs
    pred = probs.argmax(1)
    try:
        auc_v = roc_auc_score(y_true, probs[:, 1])
    except Exception:
        auc_v = 0
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    bb_display = name.replace('type_binary_', 'Type-').replace('v3_binary_', 'V3-').replace('_', '-').title()
    results.append({'Model': name, 'Algorithm': bb_display, 'AUC': auc_v, 'Accuracy': acc, 'F1': f1, 'CM': str(cm)})
    print(f'  {name}: AUC={auc_v:.4f} Acc={acc:.4f} F1={f1:.4f}')
    del m
    torch.cuda.empty_cache()

# ── YOLO evaluation ──────────────────────────────────────────
yolo_ckpt = os.path.join(MODEL_DIR, 'v3_yolo_binary.pt')
if os.path.exists(yolo_ckpt):
    print('\n  Evaluating YOLO11...')
    from ultralytics import YOLO
    ym = YOLO(yolo_ckpt)
    yprobs = []
    for p in tqdm(patients, desc='YOLO', leave=False):
        faces = [fp for fp in p['faces'][:12] if os.path.exists(fp)]
        if not faces:
            yprobs.append([0.5, 0.5]); continue
        ps = []
        for fp in faces:
            r = ym.predict(fp, verbose=False)
            if r and len(r) > 0:
                ps.append(r[0].probs.data.cpu().numpy())
        yprobs.append(np.mean(ps, axis=0) if ps else [0.5, 0.5])
    yprobs = np.array(yprobs)
    all_preds['yolo'] = yprobs
    ypred = yprobs.argmax(1)
    try:
        yauc = roc_auc_score(y_true, yprobs[:, 1])
    except Exception:
        yauc = 0
    results.append({'Model': 'yolo', 'Algorithm': 'YOLO11n', 'AUC': yauc,
                     'Accuracy': accuracy_score(y_true, ypred), 'F1': f1_score(y_true, ypred),
                     'CM': str(confusion_matrix(y_true, ypred, labels=[0, 1]).tolist())})
    print(f'  YOLO11: AUC={yauc:.4f} Acc={accuracy_score(y_true, ypred):.4f} F1={f1_score(y_true, ypred):.4f}')

# ── Ensemble ─────────────────────────────────────────────────
if len(all_preds) >= 2:
    ens = np.mean(list(all_preds.values()), axis=0)
    epred = ens.argmax(1)
    try:
        eauc = roc_auc_score(y_true, ens[:, 1])
    except Exception:
        eauc = 0
    results.append({'Model': 'ensemble', 'Algorithm': 'All-Ensemble', 'AUC': eauc,
                     'Accuracy': accuracy_score(y_true, epred), 'F1': f1_score(y_true, epred),
                     'CM': str(confusion_matrix(y_true, epred, labels=[0, 1]).tolist())})
    print(f'  Ensemble: AUC={eauc:.4f} Acc={accuracy_score(y_true, epred):.4f} F1={f1_score(y_true, epred):.4f}')

# ── Save results ─────────────────────────────────────────────
rdf = pd.DataFrame(results).round(4)
rdf.to_csv(os.path.join(TBL_DIR, 'external_validation_results.csv'), index=False, encoding='utf-8-sig')

with open(os.path.join(BASE, 'results', 'external_validation_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'\n[3] Results saved to {TBL_DIR}/external_validation_results.csv')

# ── Figure: ROC curves ───────────────────────────────────────
print('[4] Generating ROC figure...')
fig, ax = plt.subplots(figsize=(6, 5.5))
colors = plt.cm.tab10(np.linspace(0, 0.8, len(all_preds)))
for (name, probs), col in zip(all_preds.items(), colors):
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    a = roc_auc_score(y_true, probs[:, 1]) if len(np.unique(y_true)) > 1 else 0
    lbl = name.replace('type_binary_', 'Type-').replace('v3_binary_', 'V3-').replace('_', '-').title()
    if name == 'yolo': lbl = 'YOLO11n'
    ax.plot(fpr, tpr, lw=1.4, color=col, label=f'{lbl} ({a:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('External Validation: Binary Screening ROC', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim([-0.02, 1.0]); ax.set_ylim([-0.02, 1.02])
fig.tight_layout()
fig_path = os.path.join(FIG_DIR, 'Fig_external_validation_roc.png')
fig.savefig(fig_path); fig.savefig(fig_path.replace('.png', '.svg'))
plt.close(fig)
print(f'  Saved: {fig_path}')

# ── Print summary ────────────────────────────────────────────
print('\n' + '=' * 65)
print('  EXTERNAL VALIDATION RESULTS')
print('=' * 65)
print(f'\n{"Model":<30} {"AUC":>7} {"Acc":>7} {"F1":>7}')
print('-' * 65)
for _, r in rdf.iterrows():
    print(f'{r["Algorithm"]:<30} {r["AUC"]:>7.4f} {r["Accuracy"]:>7.4f} {r["F1"]:>7.4f}')
print('-' * 65)
print(f'\n  N = {len(patients)} patients ({sum(y_true==0)} normal, {sum(y_true==1)} jaundice)')
