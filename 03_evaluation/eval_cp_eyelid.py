# -*- coding: utf-8 -*-
"""
Evaluate CP eyelid models on the EXACT val split from train_cp_meld_images.py.
Sanity check: convnext must reproduce ~0.7187 and vit ~0.5311.
Then reads AUC for cp_eyelid_swin (file exists, AUC unknown).
"""
import os, re, json, random, math, numpy as np, pandas as pd, cv2
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32

# ── identical data matching to train_cp_meld_images.py ──
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
    cp_score = pd.to_numeric(r.iloc[14], errors='coerce')
    tbil = pd.to_numeric(r.iloc[51], errors='coerce')
    inr = pd.to_numeric(r.iloc[61], errors='coerce')
    alb = pd.to_numeric(r.iloc[67], errors='coerce')
    dbil = pd.to_numeric(r.iloc[53], errors='coerce')
    meld = None
    if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
        meld = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr, 1)) + 9.57 * math.log(1.0) + 6.43
        meld = round(max(meld, 6))
    return {'cp_grade': cp_grade, 'cp_score': cp_score, 'meld': meld,
            'tbil': tbil, 'inr': inr, 'alb': alb, 'dbil': dbil}

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
        p.update({'cp_grade': 'A', 'cp_score': 5, 'meld': 8,
                  'tbil': 15, 'inr': 1.0, 'alb': 42, 'dbil': 3})
        matched[pid] = p
    else:
        clin = get_clinical(pid)
        if clin and clin['cp_grade'] in ('A', 'B', 'C'):
            p.update(clin)
            matched[pid] = p

print(f'  Matched: {len(matched)} patients')

# ── identical split ──
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
def build_items(img_key, label_fn):
    tr, va = [], []
    for pid in pids:
        p = matched[pid]
        label = label_fn(p)
        if label is None: continue
        imgs = p.get(img_key, [])
        for ip in imgs[:4 if img_key == 'faces' else 1]:
            if os.path.exists(ip):
                target = (ip, label, pid)
                if pid in tr_pids: tr.append(target)
                else: va.append(target)
    return tr, va

tr_e, va_e = build_items('eyelids', lambda p: grade_map.get(p['cp_grade']))
print(f'  Eyelid → Child-Pugh: train={len(tr_e)}, val={len(va_e)}')

# ── eval infra ──
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

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ImgDS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid

@torch.no_grad()
def evaluate(fname, backbone, nc=3):
    m = timm.create_model(backbone, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, fname), map_location=DEV))
    m.eval()
    vrl = DataLoader(ImgDS(va_e), BS, shuffle=False, num_workers=0)
    ps, ls, pis = [], [], []
    for im, la, pi in vrl:
        o = m(im.to(DEV))
        ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
    d = {}
    for p, l, pi in zip(ps, ls, pis): d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pids_v = list(d.keys())
    avg = np.array([np.mean(d[p][0], 0) for p in pids_v])
    true = np.array([d[p][1] for p in pids_v])
    auc = roc_auc_score(true, avg, multi_class='ovr', labels=[0, 1, 2])
    acc = accuracy_score(true, avg.argmax(1))
    del m; torch.cuda.empty_cache()
    return auc, acc, len(pids_v), true, avg, pids_v

MODELS = [
    ('cp_eyelid_convnext.pt', 'convnext_tiny', 'ConvNeXt', 0.7187),
    ('cp_eyelid_vit.pt', 'vit_tiny_patch16_224', 'ViT', 0.5311),
    ('cp_eyelid_swin.pt', 'swin_tiny_patch4_window7_224', 'Swin', None),
]

results = {}
for fname, bb, arch, reported in MODELS:
    if not os.path.exists(os.path.join(MODEL, fname)):
        print(f'  {arch:<12} SKIP (no file)'); continue
    auc, acc, npat, true, avg, pids_v = evaluate(fname, bb)
    rep = f'reported={reported}' if reported else 'NEW'
    print(f'  {arch:<12} AUC={auc:.4f}  ACC={acc:.4f}  n={npat}  ({rep})')
    results[arch] = {'auc': float(auc), 'acc': float(acc), 'n_val': npat, 'reported': reported,
                     'val_pids': pids_v, 'val_true': [int(x) for x in true],
                     'val_prob': avg.tolist()}

with open(os.path.join(RES, 'eval_cp_eyelid.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\n  Saved: results/eval_cp_eyelid.json')
