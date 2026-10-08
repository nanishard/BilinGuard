# -*- coding: utf-8 -*-
"""
Fusion-dedicated eyelid models for Binary & Ternary tasks, trained on the
SAME patient split as the face models, so a shared held-out val exists for
fusion evaluation.

Binary:  split = train_binary_final (face_v4 patients, per-label seed42)
         eyelid train = tr_b patients with eyelid images
         eval = va_b patients with eyelid images
Ternary: split = bootstrap G3 (non-normal sam patients, stratified_split)
         eyelid train = tr_t3 patients with eyelid images
         eval = va_t3 patients with eyelid images

Output:
  models/eyelid_{binary,ternary}_fusion_{arch}.pt  (4 archs each)
  results/fusion_shared_probs.json  patient-level probs for BOTH scopes on shared val
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
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32; EP = 50; LR = 1e-4

BB = [('convnext', 'convnext_tiny'), ('vit', 'vit_tiny_patch16_224'),
      ('efficientnet', 'efficientnet_b0'), ('swin', 'swin_tiny_patch4_window7_224')]


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
    return np.array([np.mean(d[p][0], 0) for p in pis]), np.array([d[p][1] for p in pis]), pis


def train_model(name, bb_id, nc, tr_items, va_items):
    print(f'\n  >> {name} ({bb_id}) train={len(tr_items)} val={len(va_items)}')
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
                o = m(im.to(DEV)); ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
        avg, true, _ = patient_agg(ps, ls, pis)
        try:
            auc = roc_auc_score(true, avg[:, 1]) if nc == 2 else roc_auc_score(true, avg, multi_class='ovr', labels=list(range(nc)))
        except: auc = 0
        if ep % 10 == 0 or ep == 1:
            print(f'    E{ep:3d}: auc={auc:.4f}')
        if auc > best:
            best = auc; best_ep = ep
            torch.save(m.state_dict(), os.path.join(MODEL, f'{name}.pt'))
        em.restore(m, bk)
    print(f'    -> Best AUC: {best:.4f} (ep{best_ep})')
    del m; torch.cuda.empty_cache(); return best


@torch.no_grad()
def probs_for(ckpt, bb_id, nc, items):
    m = timm.create_model(bb_id, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, ckpt), map_location=DEV)); m.eval()
    dl = DataLoader(ImgDS(items, ev_tf), BS, shuffle=False, num_workers=0)
    ps, ls, pis = [], [], []
    for im, la, pi in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        ps.extend(o); ls.extend(la.numpy()); pis.extend(pi)
    del m; torch.cuda.empty_cache()
    avg, true, pids = patient_agg(ps, ls, pis)
    return {pid: [avg[i].tolist(), int(true[i])] for i, pid in enumerate(pids)}


# ── data ──
print('[init] loading...')
mdf = pd.read_csv(MANIFEST)
sam_pats = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            sam_pats[pid] = {'faces': faces, 'old_cat': cat}
for _, r in mdf.iterrows():
    if r['patient_id'] in sam_pats and r['n_eyelid'] > 0:
        sam_pats[r['patient_id']]['eyelids'] = json.loads(r['eyelid_images'])

# normal eyelid from extracted scan (for binary task)
normal_eye = {}
normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
for pid in sorted(os.listdir(normal_ext)):
    pp = os.path.join(normal_ext, pid)
    if not os.path.isdir(pp): continue
    cands = []
    for root, dirs, files in os.walk(pp):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                fp = os.path.join(root, f)
                if os.path.getsize(fp) > 1000000:
                    cands.append(fp)
    if cands:
        normal_eye[pid] = cands
for pid, cands in normal_eye.items():
    if pid in sam_pats:
        sam_pats[pid]['eyelids'] = cands

out_probs = {}

# ═══ Binary: shared split with face ═══
print('\n' + '=' * 60)
print('  BINARY: fusion-dedicated eyelid models (shared split)')
print('=' * 60)
bin_pids = {pid: (0 if p['old_cat'] == 'normal' else 1) for pid, p in sam_pats.items()}
by_label = {}
for pid, l in bin_pids.items(): by_label.setdefault(l, []).append(pid)
tr_b, va_b = set(), set()
for g, pids in by_label.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_b.update(pids[:n]); tr_b.update(pids[n:])

def eye_items(pids, labels, n_img=1):
    out = []
    for pid in pids:
        if pid not in sam_pats or pid not in labels: continue
        for ip in sam_pats[pid].get('eyelids', [])[:n_img]:
            if os.path.exists(ip):
                out.append((ip, labels[pid], pid))
    return out

tr_be = eye_items(tr_b, bin_pids, 1)
va_be = eye_items(va_b, bin_pids, 12)
va_be_pids = sorted(set(p for _, _, p in va_be))
print(f'  eyelid: train={len(tr_be)} ({len(set(p for _,_,p in tr_be))} p), val={len(va_be)} ({len(va_be_pids)} p)')
from collections import Counter
print(f'  val labels: {dict(Counter(l for _, l, _ in va_be))}')

out_probs['Binary Screening'] = {'face': {}, 'eyelid': {}, 'val_pids': va_be_pids}
if tr_be and va_be:
    for name, bb in BB:
        sn = f'eyelid_binary_fusion_{name}'
        train_model(sn, bb, 2, tr_be, va_be)
    # probs: eyelid (new fusion models) + face (existing final_binary models) on shared val
    for name, bb in BB:
        out_probs['Binary Screening']['eyelid'][name] = probs_for(f'eyelid_binary_fusion_{name}.pt', bb, 2, va_be)
        out_probs['Binary Screening']['face'][name] = probs_for(f'final_binary_face_{name}.pt', bb, 2,
                                                                [(ip, bin_pids[pid], pid) for pid in va_be_pids
                                                                 for ip in sam_pats[pid]['faces'][:12] if os.path.exists(ip)])

# ═══ Ternary: shared split with face (bootstrap G3) ═══
print('\n' + '=' * 60)
print('  TERNARY: fusion-dedicated eyelid models (shared split)')
print('=' * 60)
tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tern_pids = {pid: tmap[p['old_cat']] for pid, p in sam_pats.items() if p['old_cat'] != 'normal'}
by_grade = {}
for pid, g in tern_pids.items(): by_grade.setdefault(g, []).append(pid)
tr_t3, va_t3 = set(), set()
for g, pids in by_grade.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_t3.update(pids[:n]); tr_t3.update(pids[n:])

tr_te = eye_items(tr_t3, tern_pids, 1)
va_te = eye_items(va_t3, tern_pids, 12)
va_te_pids = sorted(set(p for _, _, p in va_te))
print(f'  eyelid: train={len(tr_te)} ({len(set(p for _,_,p in tr_te))} p), val={len(va_te)} ({len(va_te_pids)} p)')
print(f'  val labels: {dict(Counter(l for _, l, _ in va_te))}')

out_probs['Ternary Grading'] = {'face': {}, 'eyelid': {}, 'val_pids': va_te_pids}
if tr_te and va_te:
    for name, bb in BB:
        sn = f'eyelid_ternary_fusion_{name}'
        train_model(sn, bb, 3, tr_te, va_te)
    for name, bb in BB:
        out_probs['Ternary Grading']['eyelid'][name] = probs_for(f'eyelid_ternary_fusion_{name}.pt', bb, 3, va_te)
        out_probs['Ternary Grading']['face'][name] = probs_for(f'v3_ternary_{bb}.pt', bb, 3,
                                                               [(ip, tern_pids[pid], pid) for pid in va_te_pids
                                                                for ip in sam_pats[pid]['faces'][:12] if os.path.exists(ip)])

with open(os.path.join(RES, 'fusion_shared_probs.json'), 'w', encoding='utf-8') as f:
    json.dump(out_probs, f, ensure_ascii=False)
print('\n  Saved: results/fusion_shared_probs.json')
print('Done.')
