# -*- coding: utf-8 -*-
"""
P1: External validation with canonical-v2 four-backbone ensemble.
Run final_binary_face_{convnext,vit,swin,efficientnet}.pt on all external patients.
Output: results/external_validation_canonical_v2.json
"""
import os, json, numpy as np, pandas as pd, cv2
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, brier_score_loss, roc_curve)
from sklearn.calibration import calibration_curve
from scipy.stats import norm

BASE = r'D:\research\人脸识别营养\传染科'
EXT = os.path.join(BASE, 'data', 'external_processed_v3')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42; IMG_SIZE = 224; BS = 32
np.random.seed(SEED); torch.manual_seed(SEED)

# Load internal val probs to get Youden threshold from ensemble
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
val_pd = V2['Binary Screening']['face']
# Build ensemble val probs
val_pids = sorted(set.intersection(*[set(val_pd[a].keys()) for a in ['convnext','vit','swin','efficientnet']]))
val_ens_prob = np.array([np.mean([val_pd[a][p][0] for a in ['convnext','vit','swin','efficientnet']], 0) for p in val_pids])
val_true = np.array([val_pd['convnext'][p][1] for p in val_pids])
# Youden threshold from internal val ensemble
fpr_v, tpr_v, thr_v = roc_curve(val_true, val_ens_prob[:, 1])
youden_thr = thr_v[np.argmax(tpr_v - fpr_v)]
print(f'Internal val Youden threshold (ensemble): {youden_thr:.4f}')

# Preprocessing
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
    l, a, b = cv2.split(lab); l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid

ARCHS = [('convnext', 'convnext_tiny', 'final_binary_face_convnext.pt'),
         ('vit', 'vit_tiny_patch16_224', 'final_binary_face_vit.pt'),
         ('swin', 'swin_tiny_patch4_window7_224', 'final_binary_face_swin.pt'),
         ('efficientnet', 'efficientnet_b0', 'final_binary_face_efficientnet.pt')]

# Build external patient list
patients = []
for label, cat in [(0, 'normal'), (1, 'jaundice')]:
    cat_dir = os.path.join(EXT, cat)
    if not os.path.isdir(cat_dir): continue
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if f.endswith('_face.jpg')])
        if faces:
            patients.append({'pid': pid, 'label': label, 'faces': faces[:12], 'cat': cat})

print(f'External patients: {len(patients)} (normal={sum(1 for p in patients if p["label"]==0)}, jaundiced={sum(1 for p in patients if p["label"]==1)})')

# Build image items
items = []
for p in patients:
    for fp in p['faces']:
        items.append((fp, p['label'], p['pid']))
print(f'External images: {len(items)}')

# Run each backbone
all_probs = {}
for name, bb, ckpt in ARCHS:
    print(f'  Loading {ckpt}...')
    m = timm.create_model(bb, pretrained=False, num_classes=2).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, ckpt), map_location=DEV))
    m.eval()
    dl = DataLoader(DS(items), BS, shuffle=False, num_workers=0)
    d = {}
    with torch.no_grad():
        for im, la, pi in dl:
            o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
            for p, l, pid in zip(o, la.numpy(), pi):
                d.setdefault(pid, ([], int(l))); d[pid][0].append(p)
    del m; torch.cuda.empty_cache()
    all_probs[name] = {pid: (np.mean(v[0], 0), v[1]) for pid, v in d.items()}
    print(f'    {name}: {len(d)} patients')

# Ensemble
ens_probs = {}
for pid in all_probs['convnext']:
    ps = [all_probs[a][pid][0] for a in ['convnext','vit','swin','efficientnet'] if pid in all_probs[a]]
    if len(ps) == 4:
        ens_probs[pid] = (np.mean(ps, 0), all_probs['convnext'][pid][1])

# Compute metrics
pids = sorted(ens_probs.keys())
true = np.array([ens_probs[p][1] for p in pids])
P1 = np.array([ens_probs[p][0][1] for p in pids])  # P(jaundice)
pred_youden = (P1 >= youden_thr).astype(int)
pred_05 = (P1 >= 0.5).astype(int)

# Bootstrap CI
rng = np.random.RandomState(SEED)
n = len(true)
auc_boots, ap_boots = [], []
for _ in range(1000):
    idx = rng.randint(0, n, n)
    if len(np.unique(true[idx])) < 2: continue
    auc_boots.append(roc_auc_score(true[idx], P1[idx]))
    ap_boots.append(average_precision_score(true[idx], P1[idx]))

# Metrics at Youden threshold
tp_y = int(((pred_youden == 1) & (true == 1)).sum())
tn_y = int(((pred_youden == 0) & (true == 0)).sum())
fp_y = int(((pred_youden == 1) & (true == 0)).sum())
fn_y = int(((pred_youden == 0) & (true == 1)).sum())

# Metrics at 0.5
tp_5 = int(((pred_05 == 1) & (true == 1)).sum())
tn_5 = int(((pred_05 == 0) & (true == 0)).sum())
fp_5 = int(((pred_05 == 1) & (true == 0)).sum())
fn_5 = int(((pred_05 == 0) & (true == 1)).sum())

# Calibration
brier = brier_score_loss(true, P1)
frac_pos, mean_pred = calibration_curve(true, P1, n_bins=10, strategy='uniform')
# Calibration slope/intercept (logistic regression on predicted vs observed)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1e9)  # unregularized
lr.fit(P1.reshape(-1, 1), true)
cal_slope = float(lr.coef_[0][0])
cal_intercept = float(lr.intercept_[0])

result = {
    'n_total': len(pids),
    'n_normal': int((true == 0).sum()),
    'n_jaundice': int((true == 1).sum()),
    'model': 'canonical-v2 four-backbone facial ensemble (final_binary_face_*)',
    'auc': float(roc_auc_score(true, P1)),
    'auc_lo': float(np.percentile(auc_boots, 2.5)),
    'auc_hi': float(np.percentile(auc_boots, 97.5)),
    'ap': float(average_precision_score(true, P1)),
    'ap_lo': float(np.percentile(ap_boots, 2.5)),
    'ap_hi': float(np.percentile(ap_boots, 97.5)),
    'youden_threshold': float(youden_thr),
    'youden_sens': tp_y / (tp_y + fn_y) if (tp_y + fn_y) else 0,
    'youden_spec': tn_y / (tn_y + fp_y) if (tn_y + fp_y) else 0,
    'youden_acc': (tp_y + tn_y) / n,
    'youden_f1': 2 * tp_y / (2 * tp_y + fp_y + fn_y) if (2 * tp_y + fp_y + fn_y) else 0,
    'youden_cm': [[tn_y, fp_y], [fn_y, tp_y]],
    'threshold_0.5_sens': tp_5 / (tp_5 + fn_5) if (tp_5 + fn_5) else 0,
    'threshold_0.5_spec': tn_5 / (tn_5 + fp_5) if (tn_5 + fp_5) else 0,
    'threshold_0.5_acc': (tp_5 + tn_5) / n,
    'threshold_0.5_cm': [[tn_5, fp_5], [fn_5, tp_5]],
    'brier_score': float(brier),
    'calibration_slope': cal_slope,
    'calibration_intercept': cal_intercept,
    'calibration_curve': {'mean_predicted': mean_pred.tolist(), 'fraction_positive': frac_pos.tolist()},
    'patient_probs': {pid: {'prob': float(ens_probs[pid][0][1]), 'label': int(ens_probs[pid][1])} for pid in pids},
}

with open(os.path.join(RES, 'external_validation_canonical_v2.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(f'\n=== EXTERNAL VALIDATION (canonical-v2 ensemble) ===')
print(f'  N={result["n_total"]} (normal={result["n_normal"]}, jaundice={result["n_jaundice"]})')
print(f'  AUC={result["auc"]:.4f} [{result["auc_lo"]:.4f}-{result["auc_hi"]:.4f}]')
print(f'  AP={result["ap"]:.4f} [{result["ap_lo"]:.4f}-{result["ap_hi"]:.4f}]')
print(f'  Youden thr={youden_thr:.4f}: sens={result["youden_sens"]:.4f} spec={result["youden_spec"]:.4f} acc={result["youden_acc"]:.4f} F1={result["youden_f1"]:.4f}')
print(f'  CM (Youden): {result["youden_cm"]}')
print(f'  0.5 thr: sens={result["threshold_0.5_sens"]:.4f} spec={result["threshold_0.5_spec"]:.4f} acc={result["threshold_0.5_acc"]:.4f}')
print(f'  CM (0.5): {result["threshold_0.5_cm"]}')
print(f'  Brier={result["brier_score"]:.4f}  Cal slope={result["calibration_slope"]:.4f}  Cal intercept={result["calibration_intercept"]:.4f}')
print(f'\nSaved: results/external_validation_canonical_v2.json')
