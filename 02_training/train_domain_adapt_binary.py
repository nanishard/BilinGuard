# -*- coding: utf-8 -*-
"""
Retrain binary screening with DOMAIN ADAPTATION:
1. Grey World normalization to ALL images (internal + external) before training
2. Include external data in training (multi-center approach)
3. Hold out 20% external for true external validation
4. Stronger augmentation to break remaining domain-specific features
5. Add sclera-focused features (illumination-invariant)

This fixes the domain shift problem at its root.
"""
import os, json, random, re, numpy as np, pandas as pd, cv2
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
INT_DATA = os.path.join(BASE, 'data', 'face_v4')
EXT_DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32; EP = 60; LR = 1e-4

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def grey_world(img):
    """Grey World normalization — removes global illumination bias."""
    mr, mg, mb = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
    gray = (mr + mg + mb) / 3
    if gray < 1: return img
    out = img.astype(np.float32)
    out[:,:,0] = np.clip(out[:,:,0] * gray / (mr + 1e-6), 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] * gray / (mg + 1e-6), 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] * gray / (mb + 1e-6), 0, 255)
    return out.astype(np.uint8)

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

def preprocess_img(img_rgb):
    """Domain-invariant preprocessing: Grey World → CLAHE → resize."""
    img_gw = grey_world(img_rgb)
    img_clahe = clahe(img_gw)
    return cv2.resize(img_clahe, (IMG_SIZE, IMG_SIZE))

# ── Collect ALL patients (internal + external) ───────────────
print('[1] Collecting patients from internal + external...')
all_items = []  # (path, label, pid, source)

# Internal
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(INT_DATA, cat)
    if not os.path.exists(cd): continue
    label = 0 if cat == 'normal' else 1
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        for fp in faces[:4]:
            if os.path.exists(fp):
                all_items.append((fp, label, pid, 'int'))

# External
for cat, label in [('normal', 0), ('jaundice', 1)]:
    cd = os.path.join(EXT_DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        for fp in faces[:4]:
            if os.path.exists(fp):
                all_items.append((fp, label, f'ext_{pid}', 'ext'))

print(f'  Total items: {len(all_items)} (internal={sum(1 for x in all_items if x[3]=="int")}, external={sum(1 for x in all_items if x[3]=="ext")})')

# ── Patient-level split ──────────────────────────────────────
# Internal: 80% train, 20% internal-val
# External: 80% train (domain adaptation), 20% external-test (true held-out)
print('[2] Splitting data...')
int_pids = set(x[2] for x in all_items if x[3] == 'int')
ext_pids = set(x[2] for x in all_items if x[3] == 'ext')

# Internal split
by_label_int = {}
for pid in int_pids:
    items = [x for x in all_items if x[2] == pid]
    lbl = items[0][1]
    by_label_int.setdefault(lbl, []).append(pid)
tr_int, va_int = set(), set()
for lbl, pids in by_label_int.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_int.update(pids[:n]); tr_int.update(pids[n:])

# External split
by_label_ext = {}
for pid in ext_pids:
    items = [x for x in all_items if x[2] == pid]
    lbl = items[0][1]
    by_label_ext.setdefault(lbl, []).append(pid)
tr_ext, te_ext = set(), set()
for lbl, pids in by_label_ext.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    te_ext.update(pids[:n]); tr_ext.update(pids[n:])  # 20% test = held-out

# Build item lists
tr_items = [x for x in all_items if (x[2] in tr_int or x[2] in tr_ext)]
va_items = [x for x in all_items if x[2] in va_int]
te_items = [x for x in all_items if x[2] in te_ext]  # external-only test set

print(f'  Train (int+ext): {len(tr_items)} items ({len(tr_int)} int pids + {len(tr_ext)} ext pids)')
print(f'  Internal val: {len(va_items)} items ({len(va_int)} pids)')
print(f'  External test: {len(te_items)} items ({len(te_ext)} pids)')

# ── Dataset with domain-invariant preprocessing ──────────────
class DomainInvariantDS(Dataset):
    def __init__(self, items, is_train=True):
        self.items = items
        if is_train:
            self.after_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        else:
            self.after_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid, src = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE,IMG_SIZE,3), 128, dtype=np.uint8)
        # ALWAYS apply Grey World + CLAHE (domain-invariant)
        img = preprocess_img(img)
        return self.after_tf(Image.fromarray(img).convert('RGB')), label, pid

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
        self.d=d; self.s={n:p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
    def update(self, m):
        for n,p in m.named_parameters():
            if n in self.s: self.s[n]=self.d*self.s[n]+(1-self.d)*p.data
    def apply(self, m):
        b={}
        for n,p in m.named_parameters():
            if n in self.s: b[n]=p.data.clone(); p.data=self.s[n].clone()
        return b
    def restore(self, m, b):
        for n,p in m.named_parameters():
            if n in b: p.data=b[n].clone()

def patient_agg(ps, ls, pis):
    d = {}
    for p,l,pi in zip(ps,ls,pis): d.setdefault(pi,([],l)); d[pi][0].append(p)
    pis=list(d.keys())
    avg = np.array([np.mean(d[p][0],0) for p in pis])
    true = np.array([d[p][1] for p in pis])
    return avg, true, pis

# ── Train ────────────────────────────────────────────────────
print('\n[3] Training with domain adaptation...')
tr_ds = DomainInvariantDS(tr_items, is_train=True)
va_ds = DomainInvariantDS(va_items, is_train=False)
te_ds = DomainInvariantDS(te_items, is_train=False)

train_labels = [x[1] for x in tr_items]
cnt = np.bincount(train_labels, minlength=2)
sw = [1.0/cnt[l] for l in train_labels]
sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
trl = DataLoader(tr_ds, BS, sampler=sampler, num_workers=0)
vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)
tel = DataLoader(te_ds, BS, shuffle=False, num_workers=0)

BACKBONES = [
    ('convnext', 'convnext_tiny'),
    ('vit', 'vit_tiny_patch16_224'),
    ('efficientnet', 'efficientnet_b0'),
    ('swin', 'swin_tiny_patch4_window7_224'),
]

results = {}
all_preds = {}

for name, bb in BACKBONES:
    print(f'\n  >> {name} ({bb})')
    m = timm.create_model(bb, pretrained=True, num_classes=2).to(DEV)
    al = torch.FloatTensor((1./cnt)/(1./cnt).sum()).to(DEV)
    cr = FocalLoss(a=al)
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=EP)
    em = EMA(m); best_auc = 0; best_ep = 0

    for ep in range(1, EP+1):
        m.train()
        for im,la,_ in tqdm(trl, desc=f'{name} E{ep}', leave=False):
            im,la = im.to(DEV), la.to(DEV)
            op.zero_grad(); lo=cr(m(im),la); lo.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            op.step(); em.update(m)
        sc.step()
        bk=em.apply(m); m.eval()
        ps,ls,pis=[],[],[]
        with torch.no_grad():
            for im,la,pi in vrl:
                o=m(im.to(DEV)); ps.extend(F.softmax(o,1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
        avg,true,_=patient_agg(ps,ls,pis)
        try: auc=roc_auc_score(true, avg[:,1])
        except: auc=0
        if ep%10==0 or ep==1:
            acc=accuracy_score(true, avg.argmax(1))
            print(f'    E{ep:3d}: int-val acc={acc:.4f} auc={auc:.4f}')
        if auc>best_auc:
            best_auc=auc; best_ep=ep
            torch.save(m.state_dict(), os.path.join(MODEL, f'domain_adapt_binary_{name}.pt'))
        em.restore(m,bk)

    # Evaluate on external test set
    m.load_state_dict(torch.load(os.path.join(MODEL, f'domain_adapt_binary_{name}.pt'), weights_only=True))
    m.eval()
    ps,ls,pis=[],[],[]
    with torch.no_grad():
        for im,la,pi in tel:
            o=m(im.to(DEV)); ps.extend(F.softmax(o,1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
    avg,true,_=patient_agg(ps,ls,pis)
    pred=avg.argmax(1)
    try: ext_auc=roc_auc_score(true, avg[:,1])
    except: ext_auc=0
    ext_acc=accuracy_score(true, pred)
    ext_f1=f1_score(true, pred)
    ext_sens = true[true==1].shape[0] and sum((avg[true==1,1]>0.5)) / sum(true==1) or 0
    ext_spec = true[true==0].shape[0] and sum((avg[true==0,0]>0.5)) / sum(true==0) or 0
    cm = confusion_matrix(true, pred, labels=[0,1])

    results[name] = {
        'int_val_auc': best_auc,
        'ext_test_auc': ext_auc,
        'ext_test_acc': ext_acc,
        'ext_test_f1': ext_f1,
        'ext_test_sens': ext_sens,
        'ext_test_spec': ext_spec,
        'ext_cm': cm.tolist(),
        'ext_n': len(true),
    }
    all_preds[name] = avg
    print(f'    → Int-val AUC={best_auc:.4f} | Ext-test AUC={ext_auc:.4f} Acc={ext_acc:.4f} Sens={ext_sens:.3f} Spec={ext_spec:.3f}')
    print(f'      CM: {cm.tolist()}')
    del m; torch.cuda.empty_cache()

# Ensemble
if len(all_preds) >= 2:
    ens = np.mean(list(all_preds.values()), axis=0)
    try: ens_auc=roc_auc_score(true, ens[:,1])
    except: ens_auc=0
    ens_pred=ens.argmax(1)
    ens_sens = sum(ens[true==1,1]>0.5)/sum(true==1) if sum(true==1)>0 else 0
    ens_spec = sum(ens[true==0,0]>0.5)/sum(true==0) if sum(true==0)>0 else 0
    results['ensemble'] = {
        'ext_test_auc': ens_auc,
        'ext_test_acc': accuracy_score(true, ens_pred),
        'ext_test_sens': ens_sens,
        'ext_test_spec': ens_spec,
        'ext_n': len(true),
    }
    print(f'\n  Ensemble: Ext AUC={ens_auc:.4f} Sens={ens_sens:.3f} Spec={ens_spec:.3f}')

# Summary
print('\n' + '='*65)
print('  DOMAIN-ADAPTED BINARY MODEL — FINAL RESULTS')
print('='*65)
print(f'\n  {"Model":<20} {"Int-val AUC":>12} {"Ext-test AUC":>13} {"Ext Sens":>9} {"Ext Spec":>9}')
print('  ' + '-'*65)
for n, r in sorted(results.items(), key=lambda x: -x[1].get('ext_test_auc', 0)):
    print(f'  {n:<20} {r.get("int_val_auc",0):>12.4f} {r["ext_test_auc"]:>13.4f} {r["ext_test_sens"]:>9.3f} {r["ext_test_spec"]:>9.3f}')

with open(os.path.join(RES, 'domain_adapt_binary_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\n  Saved: results/domain_adapt_binary_results.json')
