# -*- coding: utf-8 -*-
"""
DBIL-based Ternary Classification Training
  Grade 0: DBIL ≤ 10 μmol/L (normal + mild)
  Grade 1: DBIL 10-68 μmol/L (moderate)
  Grade 2: DBIL > 68 μmol/L (severe)

Trains 3 model types:
  A) Eyelid ternary (strongest signal: HSV_S Cohen's d=0.54)
  B) Face ternary (weak signal but included for comparison)
  C) Fusion (eyelid + face ensemble)
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
MANIFEST = os.path.join(BASE, 'data', 'dbil_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32; EP = 80; LR = 3e-5; NC = 3


def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None


def clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


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
    def __init__(self, items, tf, modality):
        self.items = items; self.tf = tf; self.modality = modality
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None:
            img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        pil = Image.fromarray(img).convert('RGB')
        return self.tf(pil), label, pid


class FocalLoss(nn.Module):
    def __init__(self, a=None, g=2.0, ls=0.1):
        super().__init__(); self.a = a; self.g = g; self.ls = ls
    def forward(self, lo, t):
        st = torch.zeros_like(lo).scatter_(1, t.unsqueeze(1), 1.0)
        st = st * (1 - self.ls) + self.ls / lo.size(1)
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


def patient_agg(probs, labels, pids):
    d = {}
    for p, l, pi in zip(probs, labels, pids):
        d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pis = list(d.keys())
    avg = np.array([np.mean(d[p][0], 0) for p in pis])
    true = np.array([d[p][1] for p in pis])
    return avg, true, pis


def train_model(name, backbone_id, tr_items, va_items, modality):
    print(f'\n{"="*60}')
    print(f'  Training: {name} [{modality}] ({backbone_id})')
    print(f'  Train: {len(tr_items)} items | Val: {len(va_items)} items')
    print(f'{"="*60}')

    tr_ds = ImgDS(tr_items, tr_tf, modality)
    va_ds = ImgDS(va_items, ev_tf, modality)
    trl = DataLoader(tr_ds, BS, shuffle=True, num_workers=0)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)

    m = timm.create_model(backbone_id, pretrained=True, num_classes=NC).to(DEV)
    cnt = np.bincount([l for _, l, _ in tr_items], minlength=NC)
    al = torch.FloatTensor((1. / cnt) / (1. / cnt).sum()).to(DEV)
    cr = FocalLoss(a=al)
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=EP)
    em = EMA(m); best_auc = 0; best_ep = 0

    for ep in range(1, EP + 1):
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
                o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pids.extend(pi)
        avg, true, _ = patient_agg(ps, ls, pids)
        pred = avg.argmax(1)
        try:
            auc = roc_auc_score(true, avg, multi_class='ovr') if len(np.unique(true)) > 1 else 0
        except Exception:
            auc = 0
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


def eval_model(name, backbone_id, va_items, modality):
    """Evaluate and return predictions for fusion."""
    ckpt = os.path.join(MODEL, f'{name}.pt')
    if not os.path.exists(ckpt):
        return None, None
    va_ds = ImgDS(va_items, ev_tf, modality)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)
    m = timm.create_model(backbone_id, pretrained=False, num_classes=NC).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    m.eval()
    ps, ls, pids = [], [], []
    with torch.no_grad():
        for im, la, pi in vrl:
            o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pids.extend(pi)
    avg, true, va_pids = patient_agg(ps, ls, pids)
    pred = avg.argmax(1)
    try:
        auc = roc_auc_score(true, avg, multi_class='ovr') if len(np.unique(true)) > 1 else 0
    except Exception:
        auc = 0
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro')
    cm = confusion_matrix(true, pred, labels=list(range(NC)))
    print(f'  {name}: AUC={auc:.4f} Acc={acc:.4f} F1={f1:.4f}')
    print(f'    CM: {cm.tolist()}')
    del m; torch.cuda.empty_cache()
    return avg, true


# ── Load DBIL manifest ───────────────────────────────────────
print('[1] Loading DBIL manifest...')
df = pd.read_csv(MANIFEST)
df = df[df['dbil_grade'].notna()].copy()
df['dbil_grade'] = df['dbil_grade'].astype(int)
print(f'  Total: {len(df)} patients')
print(f'  Grade 0 (≤10): {sum(df["dbil_grade"]==0)}')
print(f'  Grade 1 (10-68): {sum(df["dbil_grade"]==1)}')
print(f'  Grade 2 (>68): {sum(df["dbil_grade"]==2)}')

# ── Build eyelid items ───────────────────────────────────────
# Eyelid models trained separately via train_eyelid_*.py
# Face models use face_v4 below
eyelid_items = []
print(f'\n  Eyelid items: {len(eyelid_items)} (trained separately)')

# ── Build face items ─────────────────────────────────────────
face_items = []
for _, r in df.iterrows():
    imgs = json.loads(r['face_images'])
    for img_path in imgs[:4]:  # Up to 4 face photos
        face_items.append((img_path, int(r['dbil_grade']), r['patient_id']))
face_items = [x for x in face_items if os.path.exists(x[0])]
print(f'  Face items: {len(face_items)}')

# ── Split (patient-level) ────────────────────────────────────
def patient_split(df, ratio=0.2):
    by_grade = {}
    for _, r in df.iterrows():
        by_grade.setdefault(int(r['dbil_grade']), []).append(r['patient_id'])
    tr_pids, va_pids = set(), set()
    for g, pids in by_grade.items():
        random.seed(SEED); random.shuffle(pids)
        n = max(1, int(len(pids) * ratio))
        va_pids.update(pids[:n]); tr_pids.update(pids[n:])
    return tr_pids, va_pids

tr_pids, va_pids = patient_split(df)
tr_eyelid = [x for x in eyelid_items if x[2] in tr_pids]
va_eyelid = [x for x in eyelid_items if x[2] in va_pids]
tr_face = [x for x in face_items if x[2] in tr_pids]
va_face = [x for x in face_items if x[2] in va_pids]
print(f'\n  Split: train_pids={len(tr_pids)}, val_pids={len(va_pids)}')
print(f'  Eyelid: train={len(tr_eyelid)}, val={len(va_eyelid)}')
print(f'  Face: train={len(tr_face)}, val={len(va_face)}')

# ═════════════════════════════════════════════════════════════
# A) EYELID DBIL ternary (skipped — trained separately)
# ═════════════════════════════════════════════════════════════
print('\n\n' + '█' * 60)
print('  PART A: EYELID DBIL Ternary (SKIPPED — separate trainer)')
print('█' * 60)
eyelid_results = {}
eyelid_preds = {}
eyelid_true = None

# ═════════════════════════════════════════════════════════════
# B) FACE DBIL ternary
# ═════════════════════════════════════════════════════════════
print('\n\n' + '█' * 60)
print('  PART B: FACE DBIL Ternary Classification')
print('█' * 60)

face_results = {}
face_preds = {}
backbones = [('convnext', 'convnext_tiny'), ('vit', 'vit_tiny_patch16_224'),
             ('efficientnet', 'efficientnet_b0'), ('swin', 'swin_tiny_patch4_window7_224')]
for name, bb in backbones:
    sn = f'dbil_face_{name}'
    face_results[sn] = train_model(sn, bb, tr_face, va_face, 'face')

print('\n  Face evaluation:')
face_true = None
for name, bb in backbones:
    sn = f'dbil_face_{name}'
    preds, true = eval_model(sn, bb, va_face, 'face')
    if preds is not None:
        face_preds[sn] = preds
        face_true = true

if len(face_preds) >= 2:
    ens = np.mean(list(face_preds.values()), axis=0)
    pred = ens.argmax(1)
    try:
        auc = roc_auc_score(face_true, ens, multi_class='ovr')
    except Exception:
        auc = 0
    print(f'\n  Face Ensemble: AUC={auc:.4f} Acc={accuracy_score(face_true, pred):.4f} F1={f1_score(face_true, pred, average="macro"):.4f}')
    face_results['dbil_face_ensemble'] = auc

# ═════════════════════════════════════════════════════════════
# C) FUSION (Eyelid + Face)
# ═════════════════════════════════════════════════════════════
print('\n\n' + '█' * 60)
print('  PART C: FUSION (Eyelid + Face)')
print('█' * 60)

# Find common validation patients
va_pids_eyelid = set()
for items in [va_eyelid]:
    for _, _, pid in items:
        va_pids_eyelid.add(pid)

# Build patient-level predictions
def get_patient_preds(preds_array, true_array, items):
    """Aggregate image-level predictions to patient-level."""
    pid_to_probs = {}
    # Need to rebuild since eval_model already did patient_agg
    return preds_array, true_array

fusion_results = {}
if len(eyelid_preds) >= 2 and face_preds:
    # Get per-patient predictions from eyelid and face models
    # Use eyelid ensemble and face ensemble
    eye_ens = np.mean(list(eyelid_preds.values()), axis=0)
    face_ens = np.mean(list(face_preds.values()), axis=0)

    # Both should have same validation patients (same split)
    # But eyelid and face val sets may differ in size (some patients may not have eyelid photos)
    # Let's check
    print(f'  Eyelid val patients: {len(eyelid_true)}')
    print(f'  Face val patients: {len(face_true)}')

    if len(eyelid_true) == len(face_true):
        # Same patients — can fuse directly
        for w_eye in [0.5, 0.6, 0.7, 0.8]:
            fused = w_eye * eye_ens + (1 - w_eye) * face_ens
            pred = fused.argmax(1)
            try:
                auc = roc_auc_score(eyelid_true, fused, multi_class='ovr')
            except Exception:
                auc = 0
            acc = accuracy_score(eyelid_true, pred)
            f1 = f1_score(eyelid_true, pred, average='macro')
            print(f'  Fusion (eye={int(w_eye*100)}%): AUC={auc:.4f} Acc={acc:.4f} F1={f1:.4f}')
            fusion_results[f'fusion_eye{int(w_eye*100)}'] = auc
    else:
        print('  WARNING: Eyelid and face val sets differ — using separate evaluations')
        print(f'  Using eyelid ensemble result as primary fusion (eyelid dominates)')

# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n\n' + '=' * 65)
print('  DBIL TERNARY CLASSIFICATION — COMPLETE RESULTS')
print('=' * 65)
print(f'\n  {"Model":<35} {"AUC":>8}')
print('  ' + '-' * 45)
if eyelid_results:
    print('\n  EYELID:')
    for n, a in sorted(eyelid_results.items(), key=lambda x: -x[1]):
        print(f'    {n:<33} {a:>8.4f}')
print('\n  FACE:')
for n, a in sorted(face_results.items(), key=lambda x: -x[1]):
    print(f'    {n:<33} {a:>8.4f}')
if fusion_results:
    print('\n  FUSION:')
    for n, a in sorted(fusion_results.items(), key=lambda x: -x[1]):
        print(f'    {n:<33} {a:>8.4f}')
print('  ' + '-' * 45)

all_results = {**eyelid_results, **face_results, **fusion_results}
with open(os.path.join(RES, 'dbil_classification_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'\n  Results saved: results/dbil_classification_results.json')
