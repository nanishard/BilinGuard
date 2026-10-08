# -*- coding: utf-8 -*-
"""
Train Jaundice Type models on the RAW-REBUILT dataset (data/type_v2).
Binary: hepatocellular (0) vs cholestatic (1). Shared canonical split (seed 42)
for face + eyelid so fusion is computable on the full shared val.
Output: models/typev2_{face,eyelid}_{arch}.pt (8), results/typev2_probs.json
"""
import os, json, random, math, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import Counter

BASE = r'D:\research\人脸识别营养\传染科'
TDIR = os.path.join(BASE, 'data', 'type_v2')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
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
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.03),
    transforms.RandomRotation(8), transforms.ToTensor(),
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


def train_model(name, bb_id, tr_items, va_items, nc=2):
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
        try: auc = roc_auc_score(true, avg[:, 1])
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
def probs_for(ckpt, bb_id, items):
    m = timm.create_model(bb_id, pretrained=False, num_classes=2).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, ckpt), map_location=DEV)); m.eval()
    dl = DataLoader(ImgDS(items, ev_tf), BS, shuffle=False, num_workers=0)
    ps, ls, pis = [], [], []
    for im, la, pi in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        ps.extend(o); ls.extend(la.numpy()); pis.extend(pi)
    del m; torch.cuda.empty_cache()
    avg, true, pids = patient_agg(ps, ls, pis)
    return {pid: [avg[i].tolist(), int(true[i])] for i, pid in enumerate(pids)}


def main():
    mdf = pd.read_csv(os.path.join(TDIR, 'manifest.csv'))
    mdf = mdf[mdf['n_faces'] >= 3].reset_index(drop=True)
    labels = {r['patient_id']: int(r['label']) for _, r in mdf.iterrows()}
    print(f'  patients: {len(labels)} {dict(Counter(labels.values()))}')

    by_grade = {}
    for pid, g in labels.items(): by_grade.setdefault(g, []).append(pid)
    tr_p, va_p = set(), set()
    for g, pids in by_grade.items():
        random.seed(SEED); random.shuffle(pids)
        n = max(1, int(len(pids) * 0.2))
        va_p.update(pids[:n]); tr_p.update(pids[n:])
    print(f'  split: train={len(tr_p)} val={len(va_p)} dist={dict(Counter(labels[p] for p in va_p))}')

    def face_items(pids, n_img):
        out = []
        for pid in pids:
            d = os.path.join(TDIR, 'faces', pid)
            fs = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('_face.jpg')])
            for ip in fs[:n_img]:
                out.append((ip, labels[pid], pid))
        return out

    def eyelid_items(pids):
        out = []
        for pid in pids:
            ep = os.path.join(TDIR, 'eyelids', pid, 'eyelid.jpg')
            if os.path.exists(ep):
                out.append((ep, labels[pid], pid))
        return out

    tr_f, va_f = face_items(tr_p, 4), face_items(va_p, 12)
    tr_e, va_e = eyelid_items(tr_p), eyelid_items(va_p)
    print(f'  face: train={len(tr_f)} val={len(va_f)} | eyelid: train={len(tr_e)} val={len(va_e)}')

    print('\n=== TYPE v2 FACE ===')
    for name, bb in BB:
        train_model(f'typev2_face_{name}', bb, tr_f, va_f)
    print('\n=== TYPE v2 EYELID ===')
    for name, bb in BB:
        train_model(f'typev2_eyelid_{name}', bb, tr_e, va_e)

    out = {'face': {}, 'eyelid': {}, 'val_pids': sorted(va_p)}
    for name, bb in BB:
        out['face'][name] = probs_for(f'typev2_face_{name}.pt', bb, va_f)
        out['eyelid'][name] = probs_for(f'typev2_eyelid_{name}.pt', bb, va_e)
        for scope in ['face', 'eyelid']:
            pd_ = out[scope][name]
            P = np.array([v[0] for v in pd_.values()]); t = np.array([v[1] for v in pd_.values()])
            print(f'  {scope:<7} {name:<13} val AUC={roc_auc_score(t, P[:, 1]):.4f} n={len(pd_)}')

    with open(os.path.join(RES, 'typev2_probs.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print('\n  Saved: results/typev2_probs.json')


if __name__ == '__main__':
    main()
