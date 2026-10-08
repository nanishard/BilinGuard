# -*- coding: utf-8 -*-
"""
A2: Training-set Youden threshold for binary tasks.
  For each binary model, compute probs on training patients,
  find Youden threshold, apply frozen to validation patients,
  recompute sens/spec/acc/F1 with 100-iter bootstrap.
Output: results/threshold_trainset.json
"""
import os, re, json, random, numpy as np, pandas as pd, cv2
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_curve, f1_score, accuracy_score, confusion_matrix

BASE = r'D:\research\人脸识别营养\传染科'
DATA_DIR = os.path.join(BASE, 'data', 'face_v4')
MODEL_DIR = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42; IMG_SIZE = 224; BS = 32
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Cache: already have patient-level val probs. Need training probs for Youden.
# Rebuild the face_v4 patient sets used in train_binary_final (face) and
# eval_eyelid_binary_all (eyelid with reconstruction).

V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))

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

class ImgDS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid

@torch.no_grad()
def patient_probs(ckpt, bb, nc, items):
    m = timm.create_model(bb, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, ckpt), map_location=DEV)); m.eval()
    dl = DataLoader(ImgDS(items), BS, shuffle=False, num_workers=0)
    d = {}
    for im, la, pi in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        for p, l, pid in zip(o, la.numpy(), pi):
            d.setdefault(pid, ([], int(l))); d[pid][0].append(p)
    del m; torch.cuda.empty_cache()
    return {pid: (np.mean(v[0], 0), v[1]) for pid, v in d.items()}

# Build binary training patient sets
# Face: train_binary_final split (sam_pats, per-label split seed 42)
sam_pats = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(DATA_DIR, cat)
    if not os.path.isdir(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces: sam_pats[pid] = {'faces': faces, 'cat': cat}

bin_pids = {pid: (0 if p['cat'] == 'normal' else 1) for pid, p in sam_pats.items()}
by_label = {}
for pid, l in bin_pids.items(): by_label.setdefault(l, []).append(pid)
tr_b, va_b = set(), set()
for g, pids in by_label.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_b.update(pids[:n]); tr_b.update(pids[n:])

# Eyelid: reconstruction split B (same as eval_eyelid_binary_all)
mdf = pd.read_csv(MANIFEST)
eye_items_all = []
for _, r in mdf.iterrows():
    if r['n_eyelid'] > 0:
        for img_path in json.loads(r['eyelid_images'])[:1]:
            eye_items_all.append((img_path, 0 if r['category'] == 'normal' else 1, r['patient_id']))
normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
for pid in sorted(os.listdir(normal_ext)):
    pp = os.path.join(normal_ext, pid)
    if not os.path.isdir(pp): continue
    cand = None
    for root, dirs, files in os.walk(pp):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                if os.path.getsize(os.path.join(root, f)) > 1000000:
                    cand = os.path.join(root, f); break
        if cand: break
    if cand: eye_items_all.append((cand, 0, pid))

by_l = {}
for it in eye_items_all: by_l.setdefault(it[1], []).append(it)
random.seed(SEED)
tr_e, va_e = [], []
for label, its in by_l.items():
    random.shuffle(its)
    n = max(1, int(len(its) * 0.2))
    va_e.extend(its[:n]); tr_e.extend(its[n:])

def build_face_items(pids):
    out = []
    for pid in pids:
        for ip in sam_pats[pid].get('faces', [])[:4]:
            if os.path.exists(ip): out.append((ip, bin_pids[pid], pid))
    return out

print(f'Face: train patients={len(tr_b)}, val={len(va_b)}')
print(f'Eyelid: train items={len(tr_e)}, val items={len(va_e)}')

tr_face = build_face_items(tr_b)

# Models to evaluate
models = [
    ('Binary Screening', 'face', 'final_binary_face_convnext.pt', 'convnext_tiny', tr_face),
    ('Binary Screening', 'face', 'final_binary_face_vit.pt', 'vit_tiny_patch16_224', tr_face),
    ('Binary Screening', 'face', 'final_binary_face_swin.pt', 'swin_tiny_patch4_window7_224', tr_face),
    ('Binary Screening', 'face', 'final_binary_face_efficientnet.pt', 'efficientnet_b0', tr_face),
    ('Binary Screening', 'eyelid', 'eyelid_binary_convnext.pt', 'convnext_tiny', tr_e),
    ('Binary Screening', 'eyelid', 'eyelid_binary_vit_tiny_patch16_224.pt', 'vit_tiny_patch16_224', tr_e),
    ('Binary Screening', 'eyelid', 'eyelid_binary_swin_tiny_patch4_window7_224.pt', 'swin_tiny_patch4_window7_224', tr_e),
    ('Binary Screening', 'eyelid', 'eyelid_binary_efficientnet_b0.pt', 'efficientnet_b0', tr_e),
]

results = {}
for task, scope, ckpt, bb, tr_items in models:
    print(f'  Computing training probs: {scope}_{ckpt} ({len(tr_items)} items)...')
    tr_pd = patient_probs(ckpt, bb, 2, tr_items)
    # Find Youden on training
    t_tr = np.array([tr_pd[p][1] for p in tr_pd])
    P_tr = np.array([tr_pd[p][0][1] for p in tr_pd])
    fpr, tpr, thresholds = roc_curve(t_tr, P_tr)
    youden = thresholds[np.argmax(tpr - fpr)]

    # Apply frozen threshold to validation probs
    arch = ckpt.replace('.pt', '').replace('final_binary_face_', '').replace('eyelid_binary_', '')
    val_key = arch.replace('vit_tiny_patch16_224', 'vit').replace('swin_tiny_patch4_window7_224', 'swin').replace('efficientnet_b0', 'efficientnet')
    val_pd = V2[task][scope].get(val_key)
    if not val_pd:
        # try different arch key mapping
        for k in V2[task][scope]:
            if k in val_key or val_key in k:
                val_pd = V2[task][scope][k]; val_key = k; break
    if not val_pd: continue
    t_v = np.array([v[1] for v in val_pd.values()])
    P_v = np.array([v[0][1] for v in val_pd.values()])
    pred_v = (P_v >= youden).astype(int)
    
    # Bootstrap CI on val with frozen threshold
    rng = np.random.RandomState(SEED)
    n = len(t_v)
    boots = {'sens': [], 'spec': [], 'acc': [], 'f1': []}
    for _ in range(100):
        idx = rng.randint(0, n, n)
        if len(np.unique(t_v[idx])) < 2: continue
        pred = (P_v[idx] >= youden).astype(int)
        cm = confusion_matrix(t_v[idx], pred, labels=[0, 1])
        tp = cm[1, 1]; tn = cm[0, 0]; fp = cm[0, 1]; fn = cm[1, 0]
        boots['sens'].append(tp / (tp+fn) if (tp+fn) else 0)
        boots['spec'].append(tn / (tn+fp) if (tn+fp) else 0)
        boots['acc'].append((tp+tn)/(tp+tn+fp+fn))
        boots['f1'].append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0)
    
    cm_v = confusion_matrix(t_v, pred_v, labels=[0, 1])
    tp = cm_v[1,1]; tn = cm_v[0,0]; fp = cm_v[0,1]; fn = cm_v[1,0]
    pt = {
        'sens': round(tp/(tp+fn), 4) if (tp+fn) else 0,
        'spec': round(tn/(tn+fp), 4) if (tn+fp) else 0,
        'acc': round((tp+tn)/(tp+tn+fp+fn), 4),
        'f1': round(2*tp/(2*tp+fp+fn), 4) if (2*tp+fp+fn) else 0,
    }
    for k in boots:
        arr = boots[k]; pt[f'{k}_lo'] = round(np.percentile(arr, 2.5), 4) if arr else 0
        pt[f'{k}_hi'] = round(np.percentile(arr, 97.5), 4) if arr else 0
    pt['threshold'] = float(youden); pt['n_val'] = len(t_v)
    results[f'{scope}_{val_key}'] = pt
    print(f'    {scope}_{val_key}: threshold={youden:.3f}, sens={pt["sens"]:.3f}[{pt.get("sens_lo",0):.3f}-{pt.get("sens_hi",0):.3f}]')

with open(os.path.join(RES, 'threshold_trainset.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('\nSaved: results/threshold_trainset.json')
