# -*- coding: utf-8 -*-
"""
Final frozen external validation: v4-lite preprocessing + Swin (pre-specified primary).
Compute ALL requested metrics: AUC, AP, sens/spec at frozen Youden + 0.5,
Brier score, calibration slope/intercept, reliability curve, bootstrap CIs.
"""
import os, json, numpy as np, cv2, pandas as pd
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, brier_score_loss, roc_curve, confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

BASE = r'D:\research\人脸识别营养\传染科'
EXT = os.path.join(BASE, 'data', 'external_v4_lite')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42; IMG_SIZE = 224; BS = 32

# 1. Get frozen Youden threshold from internal validation Swin
val_pd = V2['Binary Screening']['face']['swin']
val_true = np.array([v[1] for v in val_pd.values()])
val_prob = np.array([float(v[0][1]) for v in val_pd.values()])
fpr_v, tpr_v, thr_v = roc_curve(val_true, val_prob)
youden_thr = float(thr_v[np.argmax(tpr_v - fpr_v)])
print(f'Internal val Swin Youden threshold: {youden_thr:.4f}')

# 2. Load external v4-lite data
ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe_fn(img):
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
        img = clahe_fn(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid

patients = []
for label, cat in [(0, 'normal'), (1, 'jaundice')]:
    cat_dir = os.path.join(EXT, cat)
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if f.endswith('_face.jpg')])
        if faces: patients.append({'pid': pid, 'label': label, 'faces': faces[:12]})
items = [(fp, p['label'], p['pid']) for p in patients for fp in p['faces']]
print(f'External v4-lite: {len(patients)} patients, {len(items)} images')

# 3. Run Swin
m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2).to(DEV)
m.load_state_dict(torch.load(os.path.join(MODEL, 'final_binary_face_swin.pt'), map_location=DEV))
m.eval()
dl = DataLoader(DS(items), BS, shuffle=False, num_workers=0)
d = {}
with torch.no_grad():
    for im, la, pi in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        for p, l, pid in zip(o, la.numpy(), pi):
            d.setdefault(pid, ([], int(l))); d[pid][0].append(p)
del m; torch.cuda.empty_cache()
patient_probs = {pid: (np.mean(v[0], 0), v[1]) for pid, v in d.items()}

# 4. Compute ALL metrics
pids = sorted(patient_probs.keys())
true = np.array([patient_probs[p][1] for p in pids])
P1 = np.array([float(patient_probs[p][0][1]) for p in pids])

# Bootstrap
rng = np.random.RandomState(SEED)
n = len(true)
auc_b, ap_b = [], []
for _ in range(1000):
    idx = rng.randint(0, n, n)
    if len(np.unique(true[idx])) < 2: continue
    auc_b.append(roc_auc_score(true[idx], P1[idx]))
    ap_b.append(average_precision_score(true[idx], P1[idx]))

# Frozen Youden threshold metrics
pred_y = (P1 >= youden_thr).astype(int)
cm_y = confusion_matrix(true, pred_y, labels=[0, 1])
tp_y, fp_y, fn_y, tn_y = cm_y[1,1], cm_y[0,1], cm_y[1,0], cm_y[0,0]

# 0.5 threshold
pred_5 = (P1 >= 0.5).astype(int)
cm_5 = confusion_matrix(true, pred_5, labels=[0, 1])
tp_5, fp_5, fn_5, tn_5 = cm_5[1,1], cm_5[0,1], cm_5[1,0], cm_5[0,0]

# Brier + calibration
brier = brier_score_loss(true, P1)
frac_pos, mean_pred = calibration_curve(true, P1, n_bins=10, strategy='uniform')
lr_cal = LogisticRegression(C=1e9).fit(P1.reshape(-1, 1), true)
cal_slope = float(lr_cal.coef_[0][0])
cal_intercept = float(lr_cal.intercept_[0])

result = {
    'model': 'canonical-v2 Swin-Tiny (pre-specified primary single backbone)',
    'preprocessing': 'v4-lite: YuNet bbox crop, no SAM/alignment, CLAHE at inference',
    'n_total': len(pids), 'n_normal': int((true==0).sum()), 'n_jaundice': int((true==1).sum()),
    'frozen_threshold': youden_thr,
    'auc': float(roc_auc_score(true, P1)),
    'auc_lo': float(np.percentile(auc_b, 2.5)), 'auc_hi': float(np.percentile(auc_b, 97.5)),
    'ap': float(average_precision_score(true, P1)),
    'ap_lo': float(np.percentile(ap_b, 2.5)), 'ap_hi': float(np.percentile(ap_b, 97.5)),
    'youden_sens': float(tp_y/(tp_y+fn_y)) if (tp_y+fn_y) else 0,
    'youden_spec': float(tn_y/(tn_y+fp_y)) if (tn_y+fp_y) else 0,
    'youden_acc': float((tp_y+tn_y)/n),
    'youden_f1': float(2*tp_y/(2*tp_y+fp_y+fn_y)) if (2*tp_y+fp_y+fn_y) else 0,
    'youden_cm': [[int(tn_y), int(fp_y)], [int(fn_y), int(tp_y)]],
    '0.5_sens': float(tp_5/(tp_5+fn_5)) if (tp_5+fn_5) else 0,
    '0.5_spec': float(tn_5/(tn_5+fp_5)) if (tn_5+fp_5) else 0,
    '0.5_acc': float((tp_5+tn_5)/n),
    '0.5_cm': [[int(tn_5), int(fp_5)], [int(fn_5), int(tp_5)]],
    'brier_score': float(brier),
    'calibration_slope': cal_slope, 'calibration_intercept': cal_intercept,
    'calibration_curve': {'mean_pred': mean_pred.tolist(), 'frac_pos': frac_pos.tolist()},
    'patient_probs': {pid: {'prob': float(patient_probs[pid][0][1]), 'label': int(patient_probs[pid][1])} for pid in pids},
}

with open(os.path.join(RES, 'external_validation_frozen.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(f'\n=== FROZEN EXTERNAL VALIDATION (v4-lite + Swin) ===')
print(f'  N={result["n_total"]} (normal={result["n_normal"]}, jaundice={result["n_jaundice"]})')
print(f'  AUC = {result["auc"]:.4f} [{result["auc_lo"]:.4f}-{result["auc_hi"]:.4f}]')
print(f'  AP  = {result["ap"]:.4f} [{result["ap_lo"]:.4f}-{result["ap_hi"]:.4f}]')
print(f'  Frozen thr={youden_thr:.4f}: sens={result["youden_sens"]:.4f} spec={result["youden_spec"]:.4f} acc={result["youden_acc"]:.4f} F1={result["youden_f1"]:.4f}')
print(f'  CM (Youden): {result["youden_cm"]}')
print(f'  0.5 thr: sens={result["0.5_sens"]:.4f} spec={result["0.5_spec"]:.4f} acc={result["0.5_acc"]:.4f}')
print(f'  CM (0.5): {result["0.5_cm"]}')
print(f'  Brier={result["brier_score"]:.4f}  Cal slope={result["calibration_slope"]:.4f}  Cal int={result["calibration_intercept"]:.4f}')
print(f'\nSaved: results/external_validation_frozen.json')
