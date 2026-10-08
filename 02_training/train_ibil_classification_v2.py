# -*- coding: utf-8 -*-
"""
IBIL Classification — Literature-based thresholds.

References:
1. CTCAE v5.0: bilirubin graded by ULN multiples (1.5x, 3x, 10x)
2. IBIL ULN ≈ 17.1 μmol/L (1.0 mg/dL)
3. Clinical thresholds:
   - 20 μmol/L ≈ 1.2x ULN: upper boundary of "normal-range" IBIL
   - 50 μmol/L ≈ 3x ULN: threshold for marked hyperbilirubinemia

Grade 0 (Normal/Borderline): ≤20 μmol/L — clinically insignificant
Grade 1 (Moderate):          20-50 μmol/L — 1.2-3x ULN, hepatic dysfunction
Grade 2 (Severe):            >50 μmol/L — >3x ULN, marked pathology

Distribution: 109 / 115 / 89 (balanced ratio 1:1.1:0.8)

Same training pattern as DBIL: Eyelid + Face + Fusion, strong augmentation.
"""
import os, re, json, random, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST_PATH = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32; EP = 80; LR = 1e-4; NC = 3

# ── Literature-based IBIL grading ────────────────────────────
# Grade 0: ≤20 μmol/L (normal/borderline, <1.5x ULN)
# Grade 1: 20-50 μmol/L (moderate, 1.5-3x ULN)
# Grade 2: >50 μmol/L (severe, >3x ULN)
def ibil_grade(v):
    if v is None or pd.isna(v): return None
    if v <= 20: return 0
    elif v <= 50: return 1
    else: return 2

# ── Load Excel ───────────────────────────────────────────────
df_ex = pd.read_excel(EXCEL)
ibil_col = [c for c in df_ex.columns if '间接胆红素' in str(c)][0]
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)
df_ex['ibil'] = pd.to_numeric(df_ex[ibil_col], errors='coerce')

def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None

def strip_zeros(s):
    return s.lstrip('0') if s else s

def get_ibil(patient_id):
    hid = extract_hid(patient_id)
    if not hid: return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    if len(rows) > 0:
        return float(rows.iloc[0]['ibil'])
    return None


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
    def __init__(self, items, tf):
        self.items = items; self.tf = tf
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


def train_model(name, backbone_id, tr_items, va_items):
    print(f'\n{"="*60}')
    print(f'  Training: {name} ({backbone_id})')
    print(f'  Train: {len(tr_items)} | Val: {len(va_items)}')
    print(f'{"="*60}')
    tr_ds = ImgDS(tr_items, tr_tf)
    va_ds = ImgDS(va_items, ev_tf)
    # Weighted sampling for class balance
    train_labels = [l for _, l, _ in tr_items]
    class_counts = np.bincount(train_labels, minlength=NC)
    weights_arr = 1.0 / class_counts
    sample_weights = [weights_arr[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    trl = DataLoader(tr_ds, BS, sampler=sampler, num_workers=0)
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
            auc = roc_auc_score(true, avg, multi_class='ovr', labels=[0,1,2]) if len(np.unique(true)) > 1 else 0
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


def eval_and_predict(name, backbone_id, va_items):
    ckpt = os.path.join(MODEL, f'{name}.pt')
    if not os.path.exists(ckpt):
        return None, None, None
    va_ds = ImgDS(va_items, ev_tf)
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
        auc = roc_auc_score(true, avg, multi_class='ovr', labels=[0,1,2])
    except Exception:
        auc = 0
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro')
    cm = confusion_matrix(true, pred, labels=[0,1,2])
    print(f'  {name}: AUC={auc:.4f} Acc={acc:.4f} F1={f1:.4f}  CM={cm.tolist()}')
    del m; torch.cuda.empty_cache()
    return avg, true, va_pids


# ═════════════════════════════════════════════════════════════
# Build manifest
# ═════════════════════════════════════════════════════════════
print('[1] Building IBIL manifest (literature-based thresholds: ≤20/20-50/>50 μmol/L)...')

patients = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p):
            continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            ibil_val = get_ibil(pid)
            grade = ibil_grade(ibil_val)
            if grade is None:
                continue
            patients[pid] = {'faces': faces, 'old_cat': cat, 'ibil_grade': grade}

# Add eyelid images
mdf = pd.read_csv(MANIFEST_PATH)
for _, r in mdf.iterrows():
    pid = r['patient_id']
    if pid in patients and r['n_eyelid'] > 0:
        patients[pid]['eyelids'] = json.loads(r['eyelid_images'])

grades = [p['ibil_grade'] for p in patients.values()]
from collections import Counter
print(f'  Total: {len(patients)}')
print(f'  Grade 0 (≤20, normal/borderline): {grades.count(0)}')
print(f'  Grade 1 (20-50, moderate): {grades.count(1)}')
print(f'  Grade 2 (>50, severe): {grades.count(2)}')

# Save manifest
manifest_rows = []
for pid, pdata in patients.items():
    manifest_rows.append({
        'patient_id': pid, 'ibil_grade': pdata['ibil_grade'],
        'old_category': pdata['old_cat'],
        'n_face': len(pdata['faces']),
        'n_eyelid': len(pdata.get('eyelids', [])),
        'face_images': json.dumps(pdata['faces'][:12]),
        'eyelid_images': json.dumps(pdata.get('eyelids', [])[:3]),
    })
pd.DataFrame(manifest_rows).to_csv(os.path.join(BASE, 'data', 'ibil_manifest.csv'), index=False, encoding='utf-8-sig')

# ── Build items ──────────────────────────────────────────────
# Eyelid models trained separately via train_eyelid_*.py
eyelid_items = []
face_items = []
for pid, p in patients.items():
    for fp in p['faces'][:4]:
        if os.path.exists(fp):
            face_items.append((fp, p['ibil_grade'], pid))

# Patient-level stratified split
by_grade = {}
for pid, p in patients.items():
    by_grade.setdefault(p['ibil_grade'], []).append(pid)
tr_pids, va_pids = set(), set()
for g, pids in by_grade.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_pids.update(pids[:n]); tr_pids.update(pids[n:])

tr_eyelid = [x for x in eyelid_items if x[2] in tr_pids]
va_eyelid = [x for x in eyelid_items if x[2] in va_pids]
tr_face = [x for x in face_items if x[2] in tr_pids]
va_face = [x for x in face_items if x[2] in va_pids]

print(f'\n  Split: train={len(tr_pids)}p, val={len(va_pids)}p')
print(f'  Eyelid: train={len(tr_eyelid)}, val={len(va_eyelid)}')
print(f'  Face: train={len(tr_face)}, val={len(va_face)}')

# ═════════════════════════════════════════════════════════════
# A) EYELID (skipped — trained separately)
# ═════════════════════════════════════════════════════════════
print('\n\n' + '█' * 60)
print('  PART A: EYELID IBIL Ternary (SKIPPED — separate trainer)')
print('█' * 60)

backbones = [('convnext', 'convnext_tiny'), ('vit', 'vit_tiny_patch16_224'),
             ('efficientnet', 'efficientnet_b0'), ('swin', 'swin_tiny_patch4_window7_224')]
eyelid_results = {}
eyelid_preds = {}
eyelid_true = None

# ═════════════════════════════════════════════════════════════
# B) FACE
# ═════════════════════════════════════════════════════════════
print('\n\n' + '█' * 60)
print('  PART B: FACE IBIL Ternary')
print('█' * 60)

face_results = {}
face_preds = {}

for name, bb in backbones:
    sn = f'ibil_face_{name}'
    face_results[sn] = train_model(sn, bb, tr_face, va_face)

print('\n  Face evaluation:')
face_true = None
for name, bb in backbones:
    sn = f'ibil_face_{name}'
    preds, true, vpids = eval_and_predict(sn, bb, va_face)
    if preds is not None:
        face_preds[sn] = (preds, vpids)
        face_true = true

if len(face_preds) >= 2:
    common = set(list(face_preds.values())[0][1])
    for _, (_, vp) in face_preds.items():
        common &= set(vp)
    all_p = []
    for _, (p, vp) in face_preds.items():
        im = {pid: i for i, pid in enumerate(vp)}
        all_p.append(p[[im[pid] for pid in common]])
    ens = np.mean(all_p, axis=0)
    first_vp = list(face_preds.values())[0][1]
    true_a = np.array([face_true[list(first_vp).index(pid)] for pid in common])
    try:
        auc = roc_auc_score(true_a, ens, multi_class='ovr', labels=[0,1,2])
    except Exception:
        auc = 0
    print(f'\n  Face Ensemble: AUC={auc:.4f} Acc={accuracy_score(true_a, ens.argmax(1)):.4f} F1={f1_score(true_a, ens.argmax(1), average="macro"):.4f}')
    face_results['ibil_face_ensemble'] = auc

# ═════════════════════════════════════════════════════════════
# C) FUSION
# ═════════════════════════════════════════════════════════════
print('\n\n' + '█' * 60)
print('  PART C: FUSION (Eyelid + Face)')
print('█' * 60)

fusion_results = {}
if eyelid_preds and len(eyelid_preds) >= 2 and face_preds:
    common = set()
    for _, (_, vp) in eyelid_preds.items():
        if not common:
            common = set(vp)
        else:
            common &= set(vp)
    for _, (_, vp) in face_preds.items():
        common &= set(vp)

    if len(common) > 5:
        common_list = sorted(common)
        eye_ens_list = []
        for _, (p, vp) in eyelid_preds.items():
            im = {pid: i for i, pid in enumerate(vp)}
            eye_ens_list.append(p[[im[pid] for pid in common_list]])
        eye_ens = np.mean(eye_ens_list, axis=0)

        face_ens_list = []
        for _, (p, vp) in face_preds.items():
            im = {pid: i for i, pid in enumerate(vp)}
            face_ens_list.append(p[[im[pid] for pid in common_list]])
        face_ens = np.mean(face_ens_list, axis=0)

        first_vp = list(eyelid_preds.values())[0][1]
        true_f = np.array([eyelid_true[list(first_vp).index(pid)] for pid in common_list])

        for w_eye in [0.5, 0.6, 0.7, 0.8]:
            fused = w_eye * eye_ens + (1 - w_eye) * face_ens
            pred = fused.argmax(1)
            try:
                auc = roc_auc_score(true_f, fused, multi_class='ovr', labels=[0,1,2])
            except Exception:
                auc = 0
            acc = accuracy_score(true_f, pred)
            f1 = f1_score(true_f, pred, average='macro')
            print(f'  Fusion (eye={int(w_eye*100)}%): AUC={auc:.4f} Acc={acc:.4f} F1={f1:.4f}')
            fusion_results[f'fusion_eye{int(w_eye*100)}'] = auc

# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n\n' + '=' * 65)
print('  IBIL TERNARY CLASSIFICATION (≤20/20-50/>50 μmol/L)')
print('=' * 65)
print(f'\n  {"Model":<35} {"AUC":>8}')
print('  ' + '-' * 45)
for section, rd in [('EYELID', eyelid_results), ('FACE', face_results), ('FUSION', fusion_results)]:
    if section == 'EYELID' and not rd: continue
    if rd:
        print(f'\n  {section}:')
        for n, a in sorted(rd.items(), key=lambda x: -x[1]):
            print(f'    {n:<33} {a:>8.4f}')
print('  ' + '-' * 45)

all_results = {**face_results, **fusion_results}
with open(os.path.join(RES, 'ibil_classification_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f'\n  Saved: results/ibil_classification_results.json')
