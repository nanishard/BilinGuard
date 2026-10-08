# -*- coding: utf-8 -*-
"""
BilinGuard binary training on SAM v4 (color) data.

Labels: normal=0, jaundice(mild+moderate+severe)=1  (unchanged task definition)

Inputs compared (to test whether relative color breaks the confound):
  A. _face   : Grey-World + CLAHE corrected face   (color-temp normalized)
  B. _relface: skin-referenced relative Lab map     (illumination-invariant)
  C. concat  : 6-channel = face(3) + relface(3)

Augmentation: moderate (Grey World already removed color-temp variance, so we
avoid hue jitter that would re-introduce the signal we want the model to learn).
"""
import os, json, random, copy
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'sam_processed_color')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
os.makedirs(MODEL, exist_ok=True); os.makedirs(RES, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S = 42; IS = 224; BS = 32; EP = 60; LR = 5e-5; MF = 12
random.seed(S); np.random.seed(S); torch.manual_seed(S); torch.cuda.manual_seed_all(S)
print(f'GPU: {torch.cuda.get_device_name(0) if DEV.type=="cuda" else "CPU"}')


class FocalLoss(nn.Module):
    def __init__(self, a=None, g=2.0, ls=0.1):
        super().__init__(); self.a = a; self.g = g; self.ls = ls
    def forward(self, lo, t):
        nc = lo.size(1)
        st = torch.zeros_like(lo).scatter_(1, t.unsqueeze(1), 1.0)
        st = st * (1 - self.ls) + self.ls / nc
        lp = F.log_softmax(lo, 1); p = torch.exp(lp)
        fw = (1 - p.gather(1, t.unsqueeze(1)).clamp(min=1e-8)) ** self.g
        l = -(st * lp).sum(1) * fw.squeeze()
        return (l * self.a[t]).mean() if self.a is not None else l.mean()


class EMA:
    def __init__(self, m, d=0.999):
        self.d = d; self.s = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def update(self, m):
        for n, p in m.named_parameters():
            if n in self.s: self.s[n] = self.d * self.s[n] + (1 - self.d) * p.data
    def apply(self, m):
        b = {}
        for n, p in m.named_parameters():
            if n in self.s: b[n] = p.data.clone(); p.data = self.s[n].clone()
        return b
    def restore(self, m, b):
        for n, p in m.named_parameters():
            if n in b: p.data = b[n].clone()


# Moderate augmentation — NO hue jitter (preserve the yellow signal we want).
tr_tf = transforms.Compose([
    transforms.Resize((IS, IS)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ev_tf = transforms.Compose([
    transforms.Resize((IS, IS)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class JaundiceDS(Dataset):
    """mode: 'face' | 'relface' | 'both'."""
    def __init__(self, items, mode, tf, mx=MF):
        self.s = []; self.mode = mode; self.tf = tf
        for pid, face_paths, rel_paths, label in items:
            fps = face_paths[:mx]; rps = rel_paths[:mx]
            if mode == 'face':
                for p in fps:
                    self.s.append((p, None, label, pid))
            elif mode == 'relface':
                for p in rps:
                    self.s.append((None, p, label, pid))
            else:  # both — pair by frame name
                fmap = {os.path.basename(p).replace('_face.jpg', ''): p for p in fps}
                rmap = {os.path.basename(p).replace('_relface.jpg', ''): p for p in rps}
                for k in sorted(set(fmap) & set(rmap)):
                    self.s.append((fmap[k], rmap[k], label, pid))

    def __len__(self): return len(self.s)

    def _read(self, p):
        try:
            return Image.open(p).convert('RGB')
        except Exception:
            return Image.new('RGB', (IS, IS), (128, 128, 128))

    def __getitem__(self, i):
        fp, rp, label, pid = self.s[i]
        if self.mode == 'both':
            im = self.tf(self._read(fp)); rm = self.tf(self._read(rp))
            return torch.cat([im, rm], dim=0), label, pid
        p = fp if fp is not None else rp
        return self.tf(self._read(p)), label, pid


def find_patients(root, cats):
    ps = []
    for c in cats:
        cd = os.path.join(root, c)
        if not os.path.exists(cd):
            continue
        for d in sorted(os.listdir(cd)):
            p = os.path.join(cd, d)
            if not os.path.isdir(p):
                continue
            fs = os.listdir(p)
            faces = sorted([os.path.join(p, f) for f in fs if f.endswith('_face.jpg')])
            rels = sorted([os.path.join(p, f) for f in fs if f.endswith('_relface.jpg')])
            if faces:
                ps.append({'id': d, 'faces': faces, 'rels': rels,
                           'cat': c, 'label': 0 if c == 'normal' else 1})
    return ps


def split(patients, ratio=0.2):
    random.seed(S)
    by_lab = {}
    for p in patients:
        by_lab.setdefault(p['label'], []).append(p)
    tr, va = [], []
    for lab, g in by_lab.items():
        g = g[:]; random.shuffle(g)
        n = max(1, int(len(g) * ratio))
        va.extend(g[:n]); tr.extend(g[n:])
    return tr, va


def pagg(probs, labels, pids):
    d = {}
    for p, l, pi in zip(probs, labels, pids):
        d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pis = list(d.keys())
    avg = np.array([np.mean(d[pi][0], 0) for pi in pis])
    true = np.array([d[pi][1] for pi in pis])
    return avg, true, pis


def train_one(name, backbone, in_ch, mode, tr_items, va_items):
    print(f'\n{"=" * 60}\n  {name}  backbone={backbone}  mode={mode}  in_ch={in_ch}\n{"=" * 60}')
    tr_ds = JaundiceDS(tr_items, mode, tr_tf); va_ds = JaundiceDS(va_items, mode, ev_tf)
    trl = DataLoader(tr_ds, BS, shuffle=True, num_workers=0)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)
    print(f'  train imgs={len(tr_ds)}  val imgs={len(va_ds)}')

    m = timm.create_model(backbone, pretrained=True, num_classes=2, in_chans=in_ch).to(DEV)
    cnt = np.bincount([s[2] for s in tr_ds.s], minlength=2)
    al = torch.FloatTensor((1. / cnt) / (1. / cnt).sum()).to(DEV)
    cr = FocalLoss(a=al)
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=EP)
    em = EMA(m); best_auc = 0; best_ep = 0

    for ep in range(1, EP + 1):
        m.train()
        for im, la, _ in tqdm(trl, desc=f'{name} E{ep}', leave=False):
            im, la = im.to(DEV), la.to(DEV)
            op.zero_grad(); lo = cr(m(im), la); lo.backward(); op.step(); em.update(m)
        sc.step()
        bk = em.apply(m); m.eval()
        ps, ls, pids = [], [], []
        with torch.no_grad():
            for im, la, pi in vrl:
                o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy())
                ls.extend(la.numpy()); pids.extend(pi)
        avg, true, _ = pagg(ps, ls, pids)
        try:
            auc = roc_auc_score(true, avg[:, 1])
        except Exception:
            auc = 0
        if ep % 5 == 0 or ep == 1:
            acc = accuracy_score(true, avg.argmax(1))
            print(f'  E{ep:3d}: acc={acc:.4f} auc={auc:.4f}')
        if auc > best_auc:
            best_auc = auc; best_ep = ep
            torch.save(m.state_dict(), os.path.join(MODEL, f'{name}.pt'))
        em.restore(m, bk)

    # final confusion matrix at best checkpoint
    m.load_state_dict(torch.load(os.path.join(MODEL, f'{name}.pt'), weights_only=True))
    m.eval()
    ps, ls, pids = [], [], []
    with torch.no_grad():
        for im, la, pi in vrl:
            o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy())
            ls.extend(la.numpy()); pids.extend(pi)
    avg, true, _ = pagg(ps, ls, pids)
    cm = confusion_matrix(true, avg.argmax(1), labels=[0, 1]).tolist()
    print(f'  -> Best AUC: {best_auc:.4f} (ep {best_ep})  CM(normal,jaundice)={cm}')
    del m; torch.cuda.empty_cache()
    return {'auc': best_auc, 'best_epoch': best_ep, 'cm': cm}


# ── Load data ────────────────────────────────────────────────
print('\n[1] Loading SAM v4 (color) data...')
allp = find_patients(DATA, ['normal', 'mild', 'moderate', 'severe'])
for c in ['normal', 'mild', 'moderate', 'severe']:
    sub = [p for p in allp if p['cat'] == c]
    nface = sum(len(p['faces']) for p in sub)
    nrel = sum(len(p['rels']) for p in sub)
    print(f'  {c:12s}: {len(sub):3d} patients | face={nface} | relface={nrel}')

# patient-level split (label-stratified)
tr_pat, va_pat = split(allp)
tr_items = [(p['id'], p['faces'], p['rels'], p['label']) for p in tr_pat]
va_items = [(p['id'], p['faces'], p['rels'], p['label']) for p in va_pat]
n_tr_n = sum(1 for it in tr_items if it[3] == 0); n_tr_j = sum(1 for it in tr_items if it[3] == 1)
n_va_n = sum(1 for it in va_items if it[3] == 0); n_va_j = sum(1 for it in va_items if it[3] == 1)
print(f'\n[2] Split (patient-level): train={len(tr_items)} ({n_tr_n}N/{n_tr_j}J)  '
      f'val={len(va_items)} ({n_va_n}N/{n_va_j}J)')

# ── Experiments ──────────────────────────────────────────────
results = {}

# A. face only (Grey-World + CLAHE)
results['face_convnext'] = train_one('color_bin_face_convnext', 'convnext_tiny', 3, 'face', tr_items, va_items)

# B. relface only (illumination-invariant relative color)
results['relface_convnext'] = train_one('color_bin_relface_convnext', 'convnext_tiny', 3, 'relface', tr_items, va_items)

# C. 6-channel concat (face + relface)
results['both_convnext'] = train_one('color_bin_both_convnext', 'convnext_tiny', 6, 'both', tr_items, va_items)

# D. best single-input with EfficientNet
best_mode = max(['face', 'relface'], key=lambda k: results[f'{k}_convnext']['auc'])
results[f'{best_mode}_effnet'] = train_one(
    f'color_bin_{best_mode}_effnet', 'efficientnet_b0', 3, best_mode, tr_items, va_items)

# ── Summary ──────────────────────────────────────────────────
print('\n' + '=' * 65)
print('  BINARY (color v4) RESULTS — normal vs jaundice')
print('=' * 65)
print(f'  {"Experiment":<28} {"AUC":>7} {"ep":>4} {"CM":>18}')
print('  ' + '-' * 65)
for n, r in results.items():
    print(f'  {n:<28} {r["auc"]:>7.4f} {r["best_epoch"]:>4} {str(r["cm"]):>18}')

mx = max(r['auc'] for r in results.values())
verdict = ('SIGNAL RECOVERED (color pipeline works!)' if mx > 0.70
           else 'weak signal' if mx > 0.60
           else 'still no signal — face binary not feasible')
print(f'\n  Best AUC={mx:.4f}  -> {verdict}')

out = {'experiments': results, 'best_auc': mx,
       'data': DATA, 'note': 'SAM v4: Grey World + skin-referenced relative Lab'}
with open(os.path.join(RES, 'binary_color_results.json'), 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\n  Saved: results/binary_color_results.json')
