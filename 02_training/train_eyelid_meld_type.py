# -*- coding: utf-8 -*-
"""
Train eyelid models for MELD and Jaundice Type tasks.
MELD: Low/Mid/High (3-class)
Type: Hepatocellular vs Cholestatic (2-class)
"""

import os, re, json, random, math, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
MODEL = os.path.join(BASE, 'models')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 16; EP = 40; LR = 3e-5


def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


tr_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.03),
    transforms.RandomRotation(8), transforms.ToTensor(),
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
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        return self.tf(Image.fromarray(img).convert('RGB')), label, pid


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
        self.d = d
        self.s = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
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


def patient_agg(ps, ls, pis):
    d = {}
    for p, l, pi in zip(ps, ls, pis): d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pis = list(d.keys())
    return np.array([np.mean(d[p][0], 0) for p in pis]), np.array([d[p][1] for p in pis]), pis


def train_model(name, bb_id, nc, tr_items, va_items):
    print(f'\n  >> {name} ({bb_id})')
    tr_ds = ImgDS(tr_items, tr_tf); va_ds = ImgDS(va_items, ev_tf)
    labels = [l for _, l, _ in tr_items]
    cnt = np.bincount(labels, minlength=nc)
    sw = [1.0 / cnt[l] for l in labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    trl = DataLoader(tr_ds, BS, sampler=sampler, num_workers=0)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)

    m = timm.create_model(bb_id, pretrained=True, num_classes=nc).to(DEV)
    al = torch.FloatTensor((1. / cnt) / (1. / cnt).sum()).to(DEV)
    cr = FocalLoss(a=al)
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=EP)
    em = EMA(m); best = 0; best_ep = 0

    for ep in range(1, EP + 1):
        m.train()
        for im, la, _ in tqdm(trl, desc=f'{name} E{ep}', leave=False):
            im, la = im.to(DEV), la.to(DEV)
            op.zero_grad(); lo = cr(m(im), la); lo.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            op.step(); em.update(m)
        sc.step()
        bk = em.apply(m); m.eval()
        ps, ls, pis = [], [], []
        with torch.no_grad():
            for im, la, pi in vrl:
                o = m(im.to(DEV))
                ps.extend(F.softmax(o, 1).cpu().numpy())
                ls.extend(la.numpy()); pis.extend(pi)
        avg, true, _ = patient_agg(ps, ls, pis)
        pred = avg.argmax(1)
        try:
            if nc > 2:
                auc = roc_auc_score(true, avg, multi_class='ovr', labels=list(range(nc)))
            else:
                auc = roc_auc_score(true, avg[:, 1])
        except: auc = 0
        acc = accuracy_score(true, pred)
        em.restore(m, bk)
        if auc > best:
            best = auc; best_ep = ep
            torch.save(m.state_dict(), os.path.join(MODEL, f'{name}.pt'))
    print(f'    Best AUC={best:.4f} @ ep{best_ep}')
    return best


def main():
    print("=" * 60)
    print("  Eyelid Model Training: MELD + Jaundice Type")
    print("=" * 60)

    df_ex = pd.read_excel(EXCEL)
    jtype_col = '25、黄疸类型'
    TBIL_COL_IDX = 51  # column index (same as train_cp_meld_images.py)
    INR_COL_IDX = 61   # column index

    mdf = pd.read_csv(MANIFEST)
    print(f'\n[1] Building eyelid dataset...')

    # Load eyelid photos
    eyelid_patients = {}
    for _, r in mdf.iterrows():
        if r['n_eyelid'] > 0:
            imgs = json.loads(r['eyelid_images'])
            valid = [p for p in imgs if os.path.exists(p)]
            if valid:
                eyelid_patients[r['patient_id']] = valid

    print(f'  Patients with eyelid photos: {len(eyelid_patients)}')

    # Match labels
    type_items = []  # (path, label, pid) for jaundice type
    meld_items = []  # (path, meld_class, pid) for MELD

    for pid, e_paths in eyelid_patients.items():
        nums = re.findall(r'\d{6,}', pid)
        hid = nums[-1] if nums else None
        if not hid: continue
        rows = df_ex[df_ex['2、患者住院号'].astype(str).str.strip().str.lstrip('0') == hid.lstrip('0')]
        if len(rows) == 0:
            rows = df_ex[df_ex['2、患者住院号'].astype(str).str.strip().str.lstrip('0').str.endswith(hid[-6:])]
        if len(rows) == 0: continue
        r = rows.iloc[0]

        # Jaundice type
        jt = pd.to_numeric(r[jtype_col], errors='coerce')
        if pd.notna(jt) and jt in (2, 3):
            label = 0 if jt == 2 else 1  # 2=hepatocellular→0, 3=cholestatic→1
            type_items.append((e_paths[0], label, pid))

        # MELD score
        tbil = pd.to_numeric(r.iloc[TBIL_COL_IDX], errors='coerce')
        inr_val = pd.to_numeric(r.iloc[INR_COL_IDX], errors='coerce') if INR_COL_IDX else None
        # MELD score: IDENTICAL to train_cp_meld_images.py
        # Uses raw Excel TBIL (μmol/L) directly — matches meld_face_convnext training
        if pd.notna(tbil) and pd.notna(inr_val) and tbil > 0 and inr_val > 0:
            meld_score = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr_val, 1)) + 9.57 * math.log(1.0) + 6.43
            meld_score = round(max(meld_score, 6))
            if meld_score <= 20: meld_cls = 0  # Low (normal ~8)
            elif meld_score <= 30: meld_cls = 1  # Medium
            else: meld_cls = 2  # High
            meld_items.append((e_paths[0], meld_cls, pid))

    # Add NORMAL controls with default MELD=8 (Low)
    # Scan data/extracted/normal for eyelid photos
    added_low = 0
    normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
    if os.path.isdir(normal_ext):
        for pid in sorted(os.listdir(normal_ext)):
            pp = os.path.join(normal_ext, pid)
            if not os.path.isdir(pp): continue
            if any(item[2] == pid for item in meld_items): continue
            # Find eyelid photo
            candidates = []
            for root, dirs, files in os.walk(pp):
                for f in files:
                    if f.lower().endswith(('.jpg','.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                        fp = os.path.join(root, f)
                        if os.path.getsize(fp) > 1000000:
                            candidates.append(fp)
            if candidates:
                meld_items.append((candidates[0], 0, pid))
                added_low += 1
                if added_low >= 150: break
    print(f'  Added normal controls (Low): {added_low}')

    print(f'  Jaundice Type: {len(type_items)} items')
    print(f'    Type 2 (hepatocellular): {sum(1 for _, l, _ in type_items if l==0)}')
    print(f'    Type 3 (cholestatic): {sum(1 for _, l, _ in type_items if l==1)}')
    print(f'  MELD: {len(meld_items)} items')
    print(f'    Low (<15): {sum(1 for _, l, _ in meld_items if l==0)}')
    print(f'    Mid (15-25): {sum(1 for _, l, _ in meld_items if l==1)}')
    print(f'    High (>25): {sum(1 for _, l, _ in meld_items if l==2)}')

    # Split
    def split(items, ratio=0.2):
        by_label = {}
        for item in items:
            by_label.setdefault(item[1], []).append(item)
        tr, va = [], []
        for label, its in by_label.items():
            random.shuffle(its)
            n = max(1, int(len(its) * ratio))
            va.extend(its[:n]); tr.extend(its[n:])
        random.shuffle(tr); random.shuffle(va)
        return tr, va

    # ── Train Type eyelid ──
    print('\n' + '=' * 60)
    print('  Task A: Jaundice Type (Eyelid)')
    print('=' * 60)
    if len(type_items) >= 10:
        tr_t, va_t = split(type_items)
        print(f'  Split: train={len(tr_t)}, val={len(va_t)}')
        for name, bb_id in [('type_eyelid_convnext_tiny', 'convnext_tiny')]:
            train_model(name, bb_id, 2, tr_t, va_t)
    else:
        print('  SKIP: insufficient data')

    # ── Train MELD eyelid ──
    print('\n' + '=' * 60)
    print('  Task B: MELD Risk (Eyelid)')
    print('=' * 60)
    if len(meld_items) >= 10:
        tr_m, va_m = split(meld_items)
        print(f'  Split: train={len(tr_m)}, val={len(va_m)}')
        for name, bb_id in [('meld_eyelid_convnext', 'convnext_tiny')]:
            train_model(name, bb_id, 3, tr_m, va_m)
    else:
        print('  SKIP: insufficient data')

    # ── Binary Screening (Eyelid) ──
    print('\n' + '=' * 60)
    print('  Task C: Binary Screening (Eyelid)')
    print('=' * 60)
    bin_items = list(hep_bin_items) + list(normal_bin_items)
    print(f'  Total: {len(bin_items)} (jaundice={len(hep_bin_items)}, normal={len(normal_bin_items)})')
    if len(bin_items) >= 20:
        tr_b, va_b = split(bin_items)
        print(f'  Split: train={len(tr_b)}, val={len(va_b)}')
        for name, bb_id in [('eyelid_binary_convnext', 'convnext_tiny')]:
            train_model(name, bb_id, 2, tr_b, va_b)
    else:
        print('  SKIP: insufficient data')

    print('\nDone.')


if __name__ == '__main__':
    main()
