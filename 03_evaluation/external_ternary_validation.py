# -*- coding: utf-8 -*-
"""
External ternary-grading validation (canonical-v2 Swin) on the 57 jaundice
patients of the external cohort. Severity ground truth is derived from total
bilirubin (TBIL): Mild<171, Moderate 171-340, Severe>340 (umol/L).

Mirrors external_validation_final.py (binary) but swaps in the 3-class face
Swin model and computes macro-OvR AUC / macro-AP / accuracy / macro-F1 with
1000-iteration bootstrap 95% CIs (seed 42).
"""
import os, json, re
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
EXT_DIR = os.path.join(BASE, 'data', 'external_v4_lite', 'jaundice')
MODEL_DIR = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
LABELS_XLSX = os.path.join(BASE, 'external_validation_results',
                           'external_validation_results', 'patient_level_results.xlsx')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42; IMG_SIZE = 224; BS = 32; N_BOOT = 1000
CLASS_NAMES = ['Mild', 'Moderate', 'Severe']

# ---- ground-truth severity from TBIL (frozen in patient_level_results.xlsx) ----
lab = pd.read_excel(LABELS_XLSX)
name2cls = {str(r['patient_name']).strip(): int(r['true_class']) for _, r in lab.iterrows()}
name2tbil = {str(r['patient_name']).strip(): float(r['bilirubin']) for _, r in lab.iterrows()}
print('TBIL-derived labels: %d patients, class counts %s' %
      (len(name2cls), {c: sum(1 for v in name2cls.values() if v == c) for c in range(3)}))

# ---- map external face folders to patient names ----
folders = [d for d in os.listdir(EXT_DIR) if os.path.isdir(os.path.join(EXT_DIR, d))]
folder2name, unmatched = {}, []
for f in folders:
    hit = [n for n in name2cls if n and n in f]
    if hit:
        folder2name[f] = hit[0]
    else:
        unmatched.append(f)
print('matched folders: %d / %d ; unmatched: %d' % (len(folder2name), len(folders), len(unmatched)))
if unmatched:
    print('  unmatched examples:', unmatched[:5])

# ---- image read + CLAHE (same preprocessing as binary external) ----
def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None

def clahe_fn(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab); l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe_fn(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), pid

items = []
for folder, name in folder2name.items():
    faces = sorted([os.path.join(EXT_DIR, folder, f)
                    for f in os.listdir(os.path.join(EXT_DIR, folder)) if f.endswith('_face.jpg')])
    for fp in faces[:12]:
        items.append((fp, folder))
print('inference items (face images):', len(items))

# ---- run canonical-v2 Swin ternary ----
m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=3).to(DEV)
m.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'face_ternary_swin_tiny_patch4_window7_224.pt'),
                             map_location=DEV))
m.eval()
dl = DataLoader(DS(items), BS, shuffle=False, num_workers=0)
agg = {}
with torch.no_grad():
    for im, pid in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        for p, folder in zip(o, pid):
            agg.setdefault(folder, []).append(p)
del m; torch.cuda.empty_cache() if DEV == 'cuda' else None

patient = {}
for folder, probs in agg.items():
    name = folder2name[folder]
    patient[name] = {'probs': np.mean(probs, 0).tolist(),
                     'true': name2cls[name], 'tbil': name2tbil[name]}

names = sorted(patient.keys())
true = np.array([patient[n]['true'] for n in names])
P = np.array([patient[n]['probs'] for n in names])
print('\nEvaluated patients: %d ; true-class counts %s' %
      (len(names), {c: int((true == c).sum()) for c in range(3)}))

# ---- metrics ----
yb = label_binarize(true, classes=[0, 1, 2])
# macro AUC (OvR)
aucs_c = [roc_auc_score(yb[:, c], P[:, c]) for c in range(3)]
macro_auc = float(np.mean(aucs_c))
# macro AP
aps_c = [average_precision_score(yb[:, c], P[:, c]) for c in range(3)]
macro_ap = float(np.mean(aps_c))
pred = P.argmax(1)
acc = float(accuracy_score(true, pred))
f1m = float(f1_score(true, pred, average='macro'))

# ---- bootstrap CIs ----
rng = np.random.RandomState(SEED); n = len(true)
b_auc, b_ap, b_acc, b_f1 = [], [], [], []
for _ in range(N_BOOT):
    idx = rng.randint(0, n, n)
    if len(np.unique(true[idx])) < 2:
        continue
    yb_b = yb[idx]; Pb = P[idx]
    try:
        a = np.mean([roc_auc_score(yb_b[:, c], Pb[:, c]) for c in range(3)])
        ap = np.mean([average_precision_score(yb_b[:, c], Pb[:, c]) for c in range(3)])
    except Exception:
        continue
    b_auc.append(a); b_ap.append(ap)
    pb = Pb.argmax(1)
    b_acc.append(accuracy_score(true[idx], pb)); b_f1.append(f1_score(true[idx], pb, average='macro'))

def ci(arr):
    arr = np.array(arr)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


result = {
    'model': 'canonical-v2 face_ternary_swin_tiny (primary ternary backbone)',
    'preprocessing': 'v4-lite: YuNet bbox crop face jpgs, CLAHE@inference, 224',
    'n_total': int(len(names)),
    'class_counts': {CLASS_NAMES[c]: int((true == c).sum()) for c in range(3)},
    'tbil_thresholds_umol': 'Mild<171, Moderate 171-340, Severe>340',
    'auc_macro': macro_auc, 'auc_lo': ci(b_auc)[0], 'auc_hi': ci(b_auc)[1],
    'ap_macro': macro_ap, 'ap_lo': ci(b_ap)[0], 'ap_hi': ci(b_ap)[1],
    'accuracy': acc, 'acc_lo': ci(b_acc)[0], 'acc_hi': ci(b_acc)[1],
    'f1_macro': f1m, 'f1_lo': ci(b_f1)[0], 'f1_hi': ci(b_f1)[1],
    'per_class_auc': {CLASS_NAMES[c]: float(aucs_c[c]) for c in range(3)},
    'confusion_matrix': [[int((np.array((true == i) & (pred == j))).sum())
                          for j in range(3)] for i in range(3)],
    'patients': {n: {'probs': patient[n]['probs'], 'true': int(patient[n]['true']),
                     'tbil': patient[n]['tbil']} for n in names},
}
out = os.path.join(RES, 'external_ternary_validation_frozen.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print('\n=== EXTERNAL TERNARY VALIDATION (canonical-v2 Swin, n=%d) ===' % len(names))
print('  Macro AUC : %.3f [%.3f-%.3f]' % (macro_auc, result['auc_lo'], result['auc_hi']))
print('  Macro AP  : %.3f [%.3f-%.3f]' % (macro_ap, result['ap_lo'], result['ap_hi']))
print('  Accuracy  : %.3f [%.3f-%.3f]' % (acc, result['acc_lo'], result['acc_hi']))
print('  F1 (macro): %.3f [%.3f-%.3f]' % (f1m, result['f1_lo'], result['f1_hi']))
print('  Per-class AUC:', {CLASS_NAMES[c]: round(aucs_c[c], 3) for c in range(3)})
print('  Confusion matrix (rows=true Mild/Mod/Sev):', result['confusion_matrix'])
print('\nSaved:', out)
