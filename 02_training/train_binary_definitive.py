# -*- coding: utf-8 -*-
"""
DEFINITIVE binary screening test: Can we break the lighting confound?

Root cause: Normal patients photographed in brighter environments.
Model learns 'dark = jaundice' instead of 'yellow = jaundice'.

Fix strategy:
1. Grey World normalization — equalize color temperature per image
2. Reinhard color transfer — match all images to common color statistics  
3. EXTREME augmentation — brightness/contrast/saturation ±0.5 to break confound
4. If AUC > 0.65 → signal exists. If ~0.5 → no signal in data.
"""
import os, json, random, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32; EP = 60; LR = 5e-5  # Lower LR for stability


def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None


def grey_world(img_rgb):
    """Grey World: scale channels so mean R=G=B."""
    mr, mg, mb = img_rgb[:, :, 0].mean(), img_rgb[:, :, 1].mean(), img_rgb[:, :, 2].mean()
    gray = (mr + mg + mb) / 3
    out = img_rgb.astype(np.float32)
    out[:, :, 0] *= gray / (mr + 1e-6)
    out[:, :, 1] *= gray / (mg + 1e-6)
    out[:, :, 2] *= gray / (mb + 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def reinhard_transfer(img_rgb, ref_mean, ref_std):
    """Reinhard color transfer to reference statistics in Lab space."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean = lab.mean(axis=(0, 1))
    src_std = lab.std(axis=(0, 1))
    lab = (lab - src_mean) / (src_std + 1e-6) * ref_std + ref_mean
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


class JaundiceDS(Dataset):
    def __init__(self, items, mode='train', normalize='grey_world'):
        self.items = items
        self.mode = mode
        self.normalize = normalize
        # EXTREME augmentation to break lighting confound
        if mode == 'train':
            self.after_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                transforms.RandomGrayscale(p=0.1),  # Sometimes force color-invariant
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.after_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None:
            img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        # Preprocessing: normalize then resize
        if self.normalize == 'grey_world':
            img = grey_world(img)
        elif self.normalize == 'reinhard':
            # Use fixed reference (mean global image stats)
            img = reinhard_transfer(img, np.array([168, 128, 138]), np.array([30, 8, 10]))
        img = clahe(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        pil = Image.fromarray(img)
        return self.after_tf(pil), label, pid


class FocalLoss(nn.Module):
    def __init__(self, a=None, g=2.0):
        super().__init__(); self.a = a; self.g = g
    def forward(self, lo, t):
        nc = lo.size(1); st = torch.zeros_like(lo).scatter_(1, t.unsqueeze(1), 1.0)
        st = st * 0.9 + 0.1 / nc
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


def split_items(items, ratio=0.2):
    random.seed(SEED); random.shuffle(items)
    n = max(1, int(len(items) * ratio))
    return items[n:], items[:n]


def run_experiment(name, backbone_id, normalize, tr_items, va_items):
    print(f'\n{"="*60}')
    print(f'  Experiment: {name} (normalize={normalize})')
    print(f'{"="*60}')

    tr_ds = JaundiceDS(tr_items, mode='train', normalize=normalize)
    va_ds = JaundiceDS(va_items, mode='val', normalize=normalize)
    trl = DataLoader(tr_ds, BS, shuffle=True, num_workers=0)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)

    m = timm.create_model(backbone_id, pretrained=True, num_classes=2).to(DEV)
    cnt = np.bincount([l for _, l, _ in tr_items], minlength=2)
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
                o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pids.extend(pi)
        d = {}
        for p, l, pi in zip(ps, ls, pids):
            d.setdefault(pi, ([], l)); d[pi][0].append(p)
        pis = list(d.keys())
        avg = np.array([np.mean(d[p][0], 0) for p in pis])
        true = np.array([d[p][1] for p in pis])
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

    print(f'  → Best AUC: {best_auc:.4f} (epoch {best_ep})')
    del m; torch.cuda.empty_cache()
    return best_auc


# ── Build dataset ────────────────────────────────────────────
print('[1] Building face binary dataset...')
items = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    label = 0 if cat == 'normal' else 1
    for pid in sorted(os.listdir(cat_dir)):
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f and f.endswith('.jpg')])
        for fp in faces[:4]:
            items.append((fp, label, pid))

tr_items, va_items = split_items(items)
n_norm_tr = sum(1 for _, l, _ in tr_items if l == 0)
n_jaun_tr = sum(1 for _, l, _ in tr_items if l == 1)
n_norm_va = sum(1 for _, l, _ in va_items if l == 0)
n_jaun_va = sum(1 for _, l, _ in va_items if l == 1)
print(f'  Train: {len(tr_items)} ({n_norm_tr} normal, {n_jaun_tr} jaundice)')
print(f'  Val:   {len(va_items)} ({n_norm_va} normal, {n_jaun_va} jaundice)')

# ── Experiments ──────────────────────────────────────────────
results = {}

# Exp 1: Control — no normalization, weak augmentation (reproduce failure)
results['control_raw'] = run_experiment(
    'binary_control', 'convnext_tiny', None, tr_items, va_items)

# Exp 2: Grey World + extreme augmentation
results['grey_world_extremeaug'] = run_experiment(
    'binary_gw_extreme', 'convnext_tiny', 'grey_world', tr_items, va_items)

# Exp 3: Reinhard color transfer + extreme augmentation
results['reinhard_extremeaug'] = run_experiment(
    'binary_reinhard_extreme', 'convnext_tiny', 'reinhard', tr_items, va_items)

# Exp 4: Best config with EfficientNet
best_norm = 'grey_world' if results['grey_world_extremeaug'] > results['reinhard_extremeaug'] else 'reinhard'
results[f'best_efficientnet'] = run_experiment(
    f'binary_best_effnet', 'efficientnet_b0', best_norm, tr_items, va_items)

# ── Summary ──────────────────────────────────────────────────
print('\n' + '=' * 65)
print('  DEFINITIVE BINARY SCREENING RESULTS')
print('=' * 65)
print(f'\n  {"Experiment":<35} {"Best AUC":>8} {"Verdict":>20}')
print('  ' + '-' * 65)
for name, auc in results.items():
    if auc > 0.70:
        verdict = '✓ SIGNAL EXISTS!'
    elif auc > 0.60:
        verdict = '~ weak signal'
    else:
        verdict = '✗ no signal'
    print(f'  {name:<35} {auc:>8.4f} {verdict:>20}')
print('  ' + '-' * 65)
print(f'\n  Conclusion: {"Jaundice signal IS learnable with proper normalization!" if max(results.values()) > 0.65 else "No learnable signal even after normalization — face-based binary screening is not feasible with current data."}')

with open(os.path.join(RES, 'binary_definitive_test.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
