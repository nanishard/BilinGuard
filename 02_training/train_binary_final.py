# -*- coding: utf-8 -*-
"""
Retrain binary screening with SAME configuration as DBIL (strong augmentation).
This fixes both: (1) v3 leakage (AUC=1.0 from confound) and (2) type collapse (AUC=0.5).
Expected: honest AUC ~0.65-0.75.
"""
import os, json, random, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'tables')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32; EP = 80; LR = 1e-4; NC = 2

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

# SAME augmentation as DBIL — this is the key fix
tr_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomGrayscale(p=0.15),
    transforms.RandomRotation(10), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items = items; self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE,IMG_SIZE,3), 128, dtype=np.uint8)
        img = clahe(img)
        return self.tf(Image.fromarray(img).convert('RGB')), label, pid

class FocalLoss(nn.Module):
    def __init__(self, a=None, g=2.0, ls=0.1):
        super().__init__(); self.a=a; self.g=g; self.ls=ls
    def forward(self, lo, t):
        st = torch.zeros_like(lo).scatter_(1, t.unsqueeze(1), 1.0)
        st = st*(1-self.ls)+self.ls/lo.size(1)
        lp = F.log_softmax(lo,1); p = torch.exp(lp)
        fw = (1-p.gather(1,t.unsqueeze(1)).clamp(min=1e-8))**self.g
        l = -(st*lp).sum(1)*fw.squeeze()
        return (l*self.a[t]).mean() if self.a is not None else l.mean()

class EMA:
    def __init__(self, m, d=0.999):
        self.d = d; self.s = {n: p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
    def update(self, m):
        for n,p in m.named_parameters():
            if n in self.s: self.s[n] = self.d*self.s[n]+(1-self.d)*p.data
    def apply(self, m):
        b = {}
        for n,p in m.named_parameters():
            if n in self.s: b[n]=p.data.clone(); p.data=self.s[n].clone()
        return b
    def restore(self, m, b):
        for n,p in m.named_parameters():
            if n in b: p.data=b[n].clone()

def patient_agg(ps, ls, pis):
    d = {}
    for p,l,pi in zip(ps,ls,pis): d.setdefault(pi,([],l)); d[pi][0].append(p)
    pis = list(d.keys())
    avg = np.array([np.mean(d[p][0],0) for p in pis])
    true = np.array([d[p][1] for p in pis])
    return avg, true, pis

# ── Build dataset: Normal vs Jaundice (face images) ──────────
print('[1] Building face binary dataset...')
patients = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            patients[pid] = {'faces': faces, 'label': 0 if cat == 'normal' else 1}

# Also build eyelid binary dataset
mdf = pd.read_csv(MANIFEST)
for _, r in mdf.iterrows():
    pid = r['patient_id']
    if pid in patients and r['n_eyelid'] > 0:
        patients[pid]['eyelids'] = json.loads(r['eyelid_images'])

# ── Patient-level stratified split (SAME as DBIL) ────────────
by_label = {}
for pid, p in patients.items():
    by_label.setdefault(p['label'], []).append(pid)
tr_pids, va_pids = set(), set()
for g, pids in by_label.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_pids.update(pids[:n]); tr_pids.update(pids[n:])

# Face items
face_items = []
for pid, p in patients.items():
    for fp in p['faces'][:4]:
        if os.path.exists(fp):
            face_items.append((fp, p['label'], pid))
tr_face = [x for x in face_items if x[2] in tr_pids]
va_face = [x for x in face_items if x[2] in va_pids]

# Eyelid items
eyelid_items = []
for pid, p in patients.items():
    for ep in p.get('eyelids', [])[:1]:
        if os.path.exists(ep):
            eyelid_items.append((ep, p['label'], pid))
tr_eyelid = [x for x in eyelid_items if x[2] in tr_pids]
va_eyelid = [x for x in eyelid_items if x[2] in va_pids]

from collections import Counter
va_labels = [patients[pid]['label'] for pid in va_pids]
print(f'  Total: {len(patients)} (normal={sum(p["label"]==0 for p in patients.values())}, jaundiced={sum(p["label"]==1 for p in patients.values())})')
print(f'  Val: {len(va_pids)} patients {Counter(va_labels)}')
print(f'  Face: train={len(tr_face)}, val={len(va_face)}')
print(f'  Eyelid: train={len(tr_eyelid)}, val={len(va_eyelid)}')


def train_model(name, backbone_id, tr_items, va_items):
    print(f'\n{"="*60}')
    print(f'  Training: {name} ({backbone_id})')
    print(f'{"="*60}')
    tr_ds = ImgDS(tr_items, tr_tf); va_ds = ImgDS(va_items, ev_tf)
    # Weighted sampler
    train_labels = [l for _, l, _ in tr_items]
    class_counts = np.bincount(train_labels, minlength=NC)
    weights_arr = 1.0 / class_counts
    sample_weights = [weights_arr[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    trl = DataLoader(tr_ds, BS, sampler=sampler, num_workers=0)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)

    m = timm.create_model(backbone_id, pretrained=True, num_classes=NC).to(DEV)
    cnt = np.bincount([l for _, l, _ in tr_items], minlength=NC)
    al = torch.FloatTensor((1./cnt) / (1./cnt).sum()).to(DEV)
    cr = FocalLoss(a=al)
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=EP)
    em = EMA(m); best_auc = 0; best_ep = 0

    for ep in range(1, EP+1):
        m.train()
        for im, la, _ in tqdm(trl, desc=f'{name} E{ep}', leave=False):
            im, la = im.to(DEV), la.to(DEV)
            op.zero_grad(); lo = cr(m(im), la); lo.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            op.step(); em.update(m)
        sc.step()
        bk = em.apply(m); m.eval()
        ps, ls, pids = [], [], []
        with torch.no_grad():
            for im, la, pi in vrl:
                o = m(im.to(DEV)); ps.extend(F.softmax(o,1).cpu().numpy()); ls.extend(la.numpy()); pids.extend(pi)
        avg, true, _ = patient_agg(ps, ls, pids)
        pred = avg.argmax(1)
        try: auc = roc_auc_score(true, avg[:,1])
        except: auc = 0
        acc = accuracy_score(true, pred)
        f1 = f1_score(true, pred, average='macro')
        if ep % 5 == 0 or ep == 1:
            print(f'  E{ep:3d}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
        if auc > best_auc:
            best_auc = auc; best_ep = ep
            torch.save(m.state_dict(), os.path.join(MODEL, f'{name}.pt'))
        em.restore(m, bk)
    print(f'  → Best AUC: {best_auc:.4f} (epoch {best_ep})')
    del m; torch.cuda.empty_cache()
    return best_auc

# ── Train face binary models ─────────────────────────────────
print('\n\n' + '█'*60)
print('  FACE BINARY (strong augmentation, same as DBIL)')
print('█'*60)

backbones = [('convnext', 'convnext_tiny'), ('vit', 'vit_tiny_patch16_224'),
             ('efficientnet', 'efficientnet_b0'), ('swin', 'swin_tiny_patch4_window7_224')]
results = {}
for name, bb in backbones:
    sn = f'final_binary_face_{name}'
    results[sn] = train_model(sn, bb, tr_face, va_face)

# ── Train eyelid binary models ───────────────────────────────
print('\n\n' + '█'*60)
print('  EYELID BINARY (strong augmentation)')
print('█'*60)

for name, bb in backbones:
    sn = f'final_binary_eyelid_{name}'
    results[sn] = train_model(sn, bb, tr_eyelid, va_eyelid)

# ── Summary ──────────────────────────────────────────────────
print('\n' + '='*60)
print('  BINARY SCREENING RETRAIN RESULTS')
print('='*60)
print(f'\n  {"Model":<35} {"AUC":>8}')
print('  ' + '-'*45)
for n, a in sorted(results.items(), key=lambda x: -x[1]):
    print(f'    {n:<33} {a:>8.4f}')

with open(os.path.join(RES, 'final_binary_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
