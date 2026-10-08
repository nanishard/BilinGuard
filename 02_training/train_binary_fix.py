# -*- coding: utf-8 -*-
"""
Binary Screening on EYELID photos (normal vs jaundiced).
Hypothesis: eyelid ternary grading works (AUC 0.94) via texture features.
Binary on eyelids should also work.

Also trains FACE binary with Grey World color normalization (Approach B).
"""
import os, json, random, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import cv2

BASE = r'D:\research\人脸识别营养\传染科'
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
FACE_DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

IMG_SIZE = 224; BS = 32; EP = 50; LR = 1e-4


class FocalLoss(nn.Module):
    def __init__(self, a=None, g=2.0, ls=0.1):
        super().__init__(); self.a = a; self.g = g; self.ls = ls
    def forward(self, lo, t):
        nc = lo.size(1); st = torch.zeros_like(lo).scatter_(1, t.unsqueeze(1), 1.0)
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


def load_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
    except Exception:
        return Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))


def grey_world_normalize(img_rgb):
    """Grey World color constancy: scale channels so mean R=G=B."""
    mean_r = img_rgb[:, :, 0].mean()
    mean_g = img_rgb[:, :, 1].mean()
    mean_b = img_rgb[:, :, 2].mean()
    gray = (mean_r + mean_g + mean_b) / 3
    img = img_rgb.astype(np.float32)
    img[:, :, 0] *= gray / (mean_r + 1e-6)
    img[:, :, 1] *= gray / (mean_g + 1e-6)
    img[:, :, 2] *= gray / (mean_b + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


def clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


tr_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.03),
    transforms.RandomRotation(10), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class PreprocessedDS(Dataset):
    """Dataset with optional preprocessing (grey world + clahe)."""
    def __init__(self, items, tf, preprocess=None):
        self.items = items; self.tf = tf; self.preprocess = preprocess
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = load_img(path)
        if isinstance(img, np.ndarray):
            if self.preprocess == 'grey_world':
                img = grey_world_normalize(img)
            img = clahe(img)
            img = Image.fromarray(img)
        try:
            img = img.convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.tf(img), label, pid


def split_items(items, ratio=0.2):
    random.seed(SEED); random.shuffle(items)
    n = max(1, int(len(items) * ratio))
    return items[n:], items[:n]


def train_model(backbone_id, save_name, tr_items, va_items, preprocess=None):
    print(f'\n  >> {save_name} (preprocess={preprocess})...')
    tr_ds = PreprocessedDS(tr_items, tr_tf, preprocess)
    va_ds = PreprocessedDS(va_items, ev_tf, preprocess)
    trl = DataLoader(tr_ds, BS, shuffle=True, num_workers=0)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)

    m = timm.create_model(backbone_id, pretrained=True, num_classes=2).to(DEV)
    cnt = np.bincount([l for _, l, _ in tr_items], minlength=2)
    al = torch.FloatTensor((1. / cnt) / (1. / cnt).sum()).to(DEV)
    cr = FocalLoss(a=al)
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=EP)
    em = EMA(m); best_auc = 0

    for ep in range(1, EP + 1):
        m.train()
        for im, la, _ in tqdm(trl, desc=f'{save_name} E{ep}', leave=False):
            im, la = im.to(DEV), la.to(DEV)
            op.zero_grad(); lo = cr(m(im), la); lo.backward(); op.step(); em.update(m)
        sc.step()
        bk = em.apply(m); m.eval()
        ps, ls, pids = [], [], []
        with torch.no_grad():
            for im, la, pi in vrl:
                o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pids.extend(pi)
        # Patient-level aggregation
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
        acc = accuracy_score(true, avg.argmax(1))
        if ep % 10 == 0 or ep == 1:
            print(f'    E{ep:3d}: acc={acc:.4f} auc={auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            torch.save(m.state_dict(), os.path.join(MODEL, f'{save_name}.pt'))
        em.restore(m, bk)
    print(f'    Best AUC: {best_auc:.4f}')
    del m; torch.cuda.empty_cache()
    return best_auc


# ═════════════════════════════════════════════════════════════
# APPROACH A: Eyelid binary (normal vs jaundice)
# ═════════════════════════════════════════════════════════════
print('=' * 60)
print('  APPROACH A: Binary Screening on EYELID Photos')
print('=' * 60)

df = pd.read_csv(MANIFEST)
df = df[df['n_eyelid'] > 0].copy()
print(f'  Total eyelid patients: {len(df)}')

eyelid_items = []
for _, r in df.iterrows():
    imgs = json.loads(r['eyelid_images'])
    for img_path in imgs[:1]:  # Use first eyelid photo
        label = 0 if r['category'] == 'normal' else 1
        eyelid_items.append((img_path, label, r['patient_id']))

n_normal = sum(1 for _, l, _ in eyelid_items if l == 0)
n_jaundice = sum(1 for _, l, _ in eyelid_items if l == 1)
print(f'  Normal: {n_normal}, Jaundice: {n_jaundice}')

tr_items, va_items = split_items(eyelid_items)
print(f'  Train: {len(tr_items)}, Val: {len(va_items)}')

eyelid_results = {}
for name, bb in [('convnext', 'convnext_tiny'), ('vit', 'vit_tiny_patch16_224'), ('efficientnet', 'efficientnet_b0')]:
    sn = f'eyelid_binary_{name}'
    eyelid_results[sn] = train_model(bb, sn, tr_items, va_items, preprocess=None)

# ═════════════════════════════════════════════════════════════
# APPROACH B: Face binary with Grey World normalization
# ═════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  APPROACH B: Binary Screening on FACE Photos (Grey World)')
print('=' * 60)

face_items = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(FACE_DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    label = 0 if cat == 'normal' else 1
    for pid in sorted(os.listdir(cat_dir)):
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f and f.endswith('.jpg')])
        for fp in faces[:4]:
            face_items.append((fp, label, pid))

n_normal_f = sum(1 for _, l, _ in face_items if l == 0)
n_jaundice_f = sum(1 for _, l, _ in face_items if l == 1)
print(f'  Normal: {n_normal_f}, Jaundice: {n_jaundice_f}')

tr_f, va_f = split_items(face_items)
print(f'  Train: {len(tr_f)}, Val: {len(va_f)}')

face_results = {}
for name, bb in [('convnext', 'convnext_tiny'), ('vit', 'vit_tiny_patch16_224'), ('efficientnet', 'efficientnet_b0')]:
    sn = f'face_binary_gw_{name}'
    face_results[sn] = train_model(bb, sn, tr_f, va_f, preprocess='grey_world')

# Also train WITHOUT grey world for comparison (should reproduce poor results)
print('\n  >> Control: Face WITHOUT grey world...')
face_results['face_binary_raw_convnext'] = train_model('convnext_tiny', 'face_binary_raw_convnext', tr_f, va_f, preprocess=None)

# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  BINARY SCREENING COMPARISON')
print('=' * 60)
print('\n  Approach A: EYELID photos')
for n, a in eyelid_results.items():
    print(f'    {n:<30}: AUC={a:.4f}')
print('\n  Approach B: FACE photos (Grey World normalized)')
for n, a in face_results.items():
    flag = ' [control]' if 'raw' in n else ''
    print(f'    {n:<30}: AUC={a:.4f}{flag}')

with open(os.path.join(RES, 'binary_screening_fix_results.json'), 'w') as f:
    json.dump({'eyelid': eyelid_results, 'face_grey_world': {k: v for k, v in face_results.items() if 'raw' not in k},
               'face_raw_control': {k: v for k, v in face_results.items() if 'raw' in k}}, f, indent=2, default=str)
