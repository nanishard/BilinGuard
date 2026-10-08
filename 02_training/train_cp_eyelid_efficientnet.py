# -*- coding: utf-8 -*-
"""
Train cp_eyelid_efficientnet (EfficientNet-B0, 3-class Child-Pugh from eyelid).
Pipeline / split / hyperparameters identical to train_cp_meld_images.py
(split reproduction already verified: convnext 0.7186≈0.7187, vit 0.5311=0.5311).
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
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32; EP = 50; LR = 1e-4

print('[1] Loading clinical data...')
df_ex = pd.read_excel(EXCEL)
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)

def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None
def strip_zeros(s): return s.lstrip('0') if s else s

def get_clinical(pid):
    hid = extract_hid(pid)
    if not hid: return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    if len(rows) == 0: return None
    r = rows.iloc[0]
    cp_grade = str(r.iloc[15]).strip().upper() if pd.notna(r.iloc[15]) else None
    return {'cp_grade': cp_grade}

print('[2] Matching images with clinical data...')
mdf = pd.read_csv(MANIFEST)
patients = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            patients[pid] = {'faces': faces, 'cat': cat}
for _, r in mdf.iterrows():
    pid = r['patient_id']
    if pid in patients and r['n_eyelid'] > 0:
        patients[pid]['eyelids'] = json.loads(r['eyelid_images'])

matched = {}
for pid, p in patients.items():
    if p['cat'] == 'normal':
        p.update({'cp_grade': 'A'})
        matched[pid] = p
    else:
        clin = get_clinical(pid)
        if clin and clin['cp_grade'] in ('A', 'B', 'C'):
            p.update(clin)
            matched[pid] = p
print(f'  Matched: {len(matched)} patients')

pids = list(matched.keys())
grades = {pid: matched[pid]['cp_grade'] for pid in pids}
by_grade = {}
for pid in pids: by_grade.setdefault(grades[pid], []).append(pid)
tr_pids, va_pids = set(), set()
for g, ps in by_grade.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_pids.update(ps[:n]); tr_pids.update(ps[n:])
print(f'  Split: train={len(tr_pids)}, val={len(va_pids)}')

grade_map = {'A': 0, 'B': 1, 'C': 2}
tr_e, va_e = [], []
for pid in pids:
    p = matched[pid]
    label = grade_map.get(p['cp_grade'])
    if label is None: continue
    for ip in p.get('eyelids', [])[:1]:
        if os.path.exists(ip):
            (tr_e if pid in tr_pids else va_e).append((ip, label, pid))
print(f'  Eyelid -> CP: train={len(tr_e)}, val={len(va_e)}')

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

tr_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    transforms.RandomRotation(10), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ImgDS(Dataset):
    def __init__(self, items, tf): self.items = items; self.tf = tf
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

def patient_agg(ps, ls, pis):
    d = {}
    for p, l, pi in zip(ps, ls, pis): d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pis = list(d.keys())
    avg = np.array([np.mean(d[p][0], 0) for p in pis])
    true = np.array([d[p][1] for p in pis])
    return avg, true, pis

def train_classifier(name, backbone, tr_items, va_items, nc=3):
    print(f'\n  >> {name} ({backbone})')
    tr_ds = ImgDS(tr_items, tr_tf); va_ds = ImgDS(va_items, ev_tf)
    labels = [l for _, l, _ in tr_items]
    cnt = np.bincount(labels, minlength=nc)
    weights_arr = 1.0 / cnt
    sw = [weights_arr[l] for l in labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    trl = DataLoader(tr_ds, BS, sampler=sampler, num_workers=0)
    vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)
    m = timm.create_model(backbone, pretrained=True, num_classes=nc).to(DEV)
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
                o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
        avg, true, _ = patient_agg(ps, ls, pis)
        try: auc = roc_auc_score(true, avg, multi_class='ovr', labels=[0, 1, 2])
        except: auc = 0
        if ep % 10 == 0 or ep == 1:
            acc = accuracy_score(true, avg.argmax(1)); print(f'    E{ep:3d}: acc={acc:.4f} auc={auc:.4f}')
        if auc > best:
            best = auc; best_ep = ep
            torch.save(m.state_dict(), os.path.join(MODEL, f'{name}.pt'))
        em.restore(m, bk)
    print(f'    -> Best AUC: {best:.4f} (epoch {best_ep})')
    del m; torch.cuda.empty_cache(); return best

auc = train_classifier('cp_eyelid_efficientnet', 'efficientnet_b0', tr_e, va_e, 3)

out_path = os.path.join(RES, 'cp_eyelid_efficientnet_result.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({'model': 'cp_eyelid_efficientnet', 'backbone': 'efficientnet_b0',
               'task': 'Child-Pugh', 'scope': 'eyelid', 'auc': float(auc)}, f, indent=2)
print(f'  Saved: {out_path}')
