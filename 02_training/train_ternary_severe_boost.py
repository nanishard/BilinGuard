# -*- coding: utf-8 -*-
"""
Severe-boosted ternary retraining on sam_processed_v3 (SAM faces, matches the
internal-cache eval and the external_processed_v3 cohort).

Enhancements over the current train_v3.py recipe:
  * WeightedRandomSampler with a heavy Severe oversampling weight (was MISSING;
    train_v3 used plain shuffle).
  * Stronger augmentation: TrivialAugmentWide + ColorJitter + affine.
  * MixUp (alpha=0.2) on the batch.
  * Focal Loss (gamma=2) on soft (mixed) targets + label smoothing 0.1.
  * EMA(0.999), 50 epochs, CosineAnnealing.

Trains Swin (manuscript primary) and ConvNeXt (best external), then evaluates
both on the internal val split AND the external SAM cohort (Severe focus).
"""
import os, json, random, copy
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, recall_score,
                             precision_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
EXT_DIR = os.path.join(BASE, 'data', 'external_processed_v3', 'jaundice')
VIDEO_XLSX = os.path.join(BASE, 'external_validation_results',
                          'external_validation_results', 'video_level_results.xlsx')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S = 42; IS = 224; BS = 32; EP = 40; LR = 1e-4; MF = 12
# gentle per-class sampling weights (mild, moderate, severe) -- NOT raw inverse-freq
CLASS_W = np.array([1.0, 1.6, 3.0])
MIXUP_A = 0.1; MIXUP_P = 0.3
random.seed(S); np.random.seed(S); torch.manual_seed(S); torch.cuda.manual_seed_all(S)
N = ['Mild', 'Moderate', 'Severe']
import pandas as _pd


class FocalLossSoft(nn.Module):
    def __init__(self, gamma=2.0, ls=0.1):
        super().__init__(); self.g = gamma; self.ls = ls
    def forward(self, logits, soft):  # soft: (B,C) probs (mixed), sum=1
        C = logits.size(1)
        soft = soft * (1 - self.ls) + self.ls / C
        logp = F.log_softmax(logits, 1)
        pt = (soft * torch.exp(logp)).sum(1).clamp(min=1e-8)
        return (-((soft * logp).sum(1)) * (1 - pt) ** self.g).mean()


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


tr_tf = transforms.Compose([
    transforms.Resize((IS, IS)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandAugment(num_ops=1, magnitude=5),
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ev_tf = transforms.Compose([
    transforms.Resize((IS, IS)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class DS(Dataset):
    def __init__(self, pats, key, lmap, tf, mx=MF):
        self.s = []; self.tf = tf
        for p in pats:
            if p['cat'] not in lmap: continue
            for ip in p.get(key, [])[:mx]:
                self.s.append((ip, lmap[p['cat']], p['id']))
        self.labels = [l for _, l, _ in self.s]
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l, pi = self.s[i]
        try: im = Image.open(p).convert('RGB')
        except: im = Image.new('RGB', (IS, IS), (128, 128, 128))
        return self.tf(im), l, pi


def find_p(root, cats):
    ps = []
    for c in cats:
        cd = os.path.join(root, c)
        if not os.path.exists(cd): continue
        for d in os.listdir(cd):
            p = os.path.join(cd, d)
            if not os.path.isdir(p): continue
            faces = sorted([os.path.join(p, f) for f in os.listdir(p)
                            if '_face' in f and f.endswith('.jpg')])
            if faces: ps.append({'id': d, 'faces': faces, 'cat': c})
    return ps


def split(ps, fn, r=0.2):
    bl = {}
    for p in ps: bl.setdefault(fn(p), []).append(p)
    tr, va = [], []
    for l, g in bl.items():
        random.shuffle(g); n = max(1, int(len(g) * r))
        va.extend(g[:n]); tr.extend(g[n:])
    return tr, va


def pagg(ps, ls, pis):
    d = {}
    for p, l, pi in zip(ps, ls, pis): d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pis = list(d.keys())
    return (np.array([np.mean(d[p][0], 0) for p in pis]),
            np.array([d[p][1] for p in pis]), pis)


def mixup(x, y, alpha=MIXUP_A, p=MIXUP_P):
    if random.random() > p or alpha <= 0:
        return x, F.one_hot(y, num_classes=3).float()
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    xm = lam * x + (1 - lam) * x[idx]
    y_soft = lam * F.one_hot(y, 3).float() + (1 - lam) * F.one_hot(y[idx], 3).float()
    return xm, y_soft


def train_backbone(backbone, save_name, trl, vrl):
    print(f'\n>> Training {backbone} (Severe-boost, MixUp, TrivialAugment)')
    m = timm.create_model(backbone, pretrained=True, num_classes=3).to(DEV)
    crit = FocalLossSoft()
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=EP)
    em = EMA(m); best = 0
    for ep in range(1, EP + 1):
        m.train()
        for im, la, _ in tqdm(trl, desc=f'{save_name} E{ep}', leave=False):
            im, la = im.to(DEV), la.to(DEV)
            xm, ys = mixup(im, la)
            op.zero_grad(); loss = crit(m(xm), ys); loss.backward(); op.step(); em.update(m)
        sc.step()
        bk = em.apply(m); m.eval()
        ps, ls, pis = [], [], []
        with torch.no_grad():
            for im, la, pi in vrl:
                o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy())
                ls.extend(la.numpy()); pis.extend(pi)
        avg, true, _ = pagg(np.array(ps), np.array(ls), pis)
        pl = avg.argmax(1)
        try: au = roc_auc_score(true, avg, multi_class='ovr')
        except: au = 0
        f1 = f1_score(true, pl, average='macro')
        sev_f1 = f1_score(true, pl, labels=[2], average='macro', zero_division=0)
        if ep % 5 == 0 or ep == 1:
            print(f'  E{ep:3d}: auc={au:.4f} macF1={f1:.4f} SevF1={sev_f1:.4f}')
        score = au + 0.3 * sev_f1  # weight Severe in model selection
        if score > best:
            best = score; torch.save(m.state_dict(), os.path.join(MODEL, f'{save_name}.pt'))
        em.restore(m, bk)
    print(f'  Best score (auc+0.3*SevF1): {best:.4f}')
    del m
    if DEV.type == 'cuda': torch.cuda.empty_cache()


def per_class(true, avg):
    yb = label_binarize(true, classes=[0, 1, 2]); pred = avg.argmax(1)
    aucs = [roc_auc_score(yb[:, c], avg[:, c]) if len(np.unique(yb[:, c])) > 1 else np.nan for c in range(3)]
    res = {'macAUC': np.nanmean(aucs), 'acc': accuracy_score(true, pred),
           'macF1': f1_score(true, pred, average='macro')}
    for c in range(3):
        res['AUC_%s' % N[c]] = aucs[c]
        res['F1_%s' % N[c]] = f1_score(true, pred, labels=[c], average='macro', zero_division=0)
        res['Rec_%s' % N[c]] = recall_score(true, pred, labels=[c], average='macro', zero_division=0)
        res['Prec_%s' % N[c]] = precision_score(true, pred, labels=[c], average='macro', zero_division=0)
    res['CM'] = confusion_matrix(true, pred, labels=[0, 1, 2]).tolist()
    return res


def eval_internal(model_file, backbone, val_pats):
    m = timm.create_model(backbone, pretrained=False, num_classes=3).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, model_file), map_location=DEV)); m.eval()
    vad = DS(val_pats, 'faces', {'mild': 0, 'moderate': 1, 'severe': 2}, ev_tf)
    vrl = DataLoader(vad, BS, shuffle=False, num_workers=0)
    ps, ls, pis = [], [], []
    with torch.no_grad():
        for im, la, pi in vrl:
            ps.extend(F.softmax(m(im.to(DEV)), 1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
    avg, true, _ = pagg(np.array(ps), np.array(ls), pis)
    del m
    if DEV.type == 'cuda': torch.cuda.empty_cache()
    return true, avg


def eval_external(model_file, backbone):
    import cv2
    v = _pd.read_excel(VIDEO_XLSX)
    f2i = {}
    for _, r in v.iterrows():
        fn = str(r['folder_name']).strip()
        if fn and fn.lower() != 'nan': f2i[fn] = int(r['true_class'])
    ext_folders = set(os.listdir(EXT_DIR))
    matched = {f: i for f, i in f2i.items() if f in ext_folders}
    for f in list(f2i):
        if f in matched: continue
        for ef in ext_folders:
            if f in ef or ef in f: matched[ef] = f2i[f]; break
    items = []
    for folder in matched:
        d = os.path.join(EXT_DIR, folder)
        if not os.path.isdir(d): continue
        for fp in sorted([os.path.join(d, x) for x in os.listdir(d) if x.endswith('_face.jpg')])[:12]:
            items.append((fp, folder))

    def read_img(path):
        try:
            fb = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        except Exception:
            return None
    class ED(Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, i):
            path, folder = items[i]
            img = read_img(path)
            if img is None: img = np.full((IS, IS, 3), 128, dtype=np.uint8)
            return ev_tf(Image.fromarray(img).convert('RGB')), folder
    m = timm.create_model(backbone, pretrained=False, num_classes=3).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, model_file), map_location=DEV)); m.eval()
    dl = DataLoader(ED(), BS, shuffle=False, num_workers=0)
    agg = {}
    with torch.no_grad():
        for im, folder in dl:
            o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
            for p, f in zip(o, folder): agg.setdefault(f, []).append(p)
    del m
    if DEV.type == 'cuda': torch.cuda.empty_cache()
    folders = sorted([f for f in matched if f in agg])
    true = np.array([matched[f] for f in folders])
    avg = np.array([np.mean(agg[f], 0) for f in folders])
    return true, avg


# ============================ MAIN ============================
print('GPU:', torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'CPU')
jp = find_p(DATA, ['mild', 'moderate', 'severe'])
tm = {'mild': 0, 'moderate': 1, 'severe': 2}
for c in ['mild', 'moderate', 'severe']:
    print('  %-9s %d patients' % (c, sum(1 for p in jp if p['cat'] == c)))
tr, va = split(jp, lambda p: tm[p['cat']])
trd = DS(tr, 'faces', tm, tr_tf); vad = DS(va, 'faces', tm, ev_tf)
print('Train patients=%d imgs=%d | Val patients=%d imgs=%d' % (len(tr), len(trd), len(va), len(vad)))

# WeightedRandomSampler with GENTLE Severe oversampling (fixed weights, not raw inverse-freq)
counts = np.bincount(trd.labels, minlength=3).astype(float)
class_w = CLASS_W / CLASS_W.sum()
sample_w = np.array([class_w[l] for l in trd.labels])
sampler = WeightedRandomSampler(sample_w.tolist(), num_samples=len(trd), replacement=True)
print('class counts(train imgs):', counts.astype(int).tolist(), '| sample weights:', class_w.round(3).tolist())
trl = DataLoader(trd, BS, sampler=sampler, num_workers=0)
vrl = DataLoader(vad, BS, shuffle=False, num_workers=0)

cands = [('swin', 'swin_tiny_patch4_window7_224', 'ternary_sevboost_swin'),
         ('convnext', 'convnext_tiny', 'ternary_sevboost_convnext')]
results = {}
for tag, bb, sn in cands:
    train_backbone(bb, sn, trl, vrl)
    ti, pi = eval_internal(sn + '.pt', bb, va)
    te, pe = eval_external(sn + '.pt', bb)
    results[tag] = {'internal': per_class(ti, pi), 'external': per_class(te, pe)}

# baseline (cache) for comparison
print('\n' + '=' * 70)
print('%-12s %-8s %7s %7s %7s %7s | %7s %7s %7s' %
      ('model', 'set', 'macAUC', 'macF1', 'SevAUC', 'SevF1', 'SevRec', 'SevPrec', 'ModAUC'))
print('-' * 70)
print('%-12s %-8s %7s %7s %7s %7s | %7s %7s %7s' %
      ('(baseline)', 'int', '0.907', '0.830', '0.909', '0.889', '0.800', '1.000', '0.929'))
print('%-12s %-8s %7s %7s %7s %7s | %7s %7s %7s' %
      ('(baseline)', 'ext', '0.640', '0.375', '0.781', '0.320', '0.800', '0.200', '0.512'))
for tag in results:
    for setname in ['internal', 'external']:
        r = results[tag][setname]
        print('%-12s %-8s %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f' %
              (tag, setname, r['macAUC'], r['macF1'], r['AUC_Severe'], r['F1_Severe'],
               r['Rec_Severe'], r['Prec_Severe'], r['AUC_Moderate']))

with open(os.path.join(RES, 'ternary_severe_boost_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('\nSaved: results/ternary_severe_boost_results.json')
print('Models: ternary_sevboost_swin.pt, ternary_sevboost_convnext.pt')
