# -*- coding: utf-8 -*-
"""
Complete Training Pipeline: Jaundice Type Classification
Tasks:
  A) Binary: normal vs jaundiced (ALL models: YOLO, ConvNeXt, ViT, EfficientNet, Swin, YellowFeatures)
  B) Type: hepatocellular vs cholestatic (main clinical distinction, sufficient data)
  C) External validation on both tasks

Note: Type 1 (hemolytic, n=1) merged into hepatocellular. 
      Type 4 (other, n=4) excluded due to insufficient data.
"""
import os, json, random, shutil, cv2, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')  # Internal face data
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
DEV = 'cuda'
SEED = 42
random.seed(SEED); np.random.seed(SEED)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

print(f'GPU: {torch.cuda.get_device_name(0)}')

# ═════════════════════════════════════════════════════════════
# STEP 1: Build datasets
# ═════════════════════════════════════════════════════════════
print('\n[1] Building datasets...')

df_excel = pd.read_excel(EXCEL)
df_excel['hid_str'] = df_excel['2、患者住院号'].astype(str)

import re
def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None
def strip_zeros(s):
    return s.lstrip('0') if s else s

def get_jtype(patient_id):
    hid = extract_hid(patient_id)
    if not hid: return None
    hid_norm = strip_zeros(hid)
    rows = df_excel[df_excel['hid_str'].apply(strip_zeros) == hid_norm]
    if len(rows) == 0:
        rows = df_excel[df_excel['hid_str'].apply(strip_zeros).str.endswith(hid_norm[-6:])]
    if len(rows) > 0:
        return int(rows.iloc[0]['25、黄疸类型'])
    return None

# Collect patients with face images
def find_face_imgs(cat_dir):
    patients = {}
    if not os.path.exists(cat_dir): return patients
    for d in os.listdir(cat_dir):
        p = os.path.join(cat_dir, d)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        scleras = sorted([os.path.join(p, f) for f in os.listdir(p) if '_sclera' in f and f.endswith('.jpg')])
        if faces:
            patients[d] = {'faces': faces, 'scleras': scleras}
    return patients

# Normal patients
normal_pats = find_face_imgs(os.path.join(DATA, 'normal'))
print(f'  Normal: {len(normal_pats)} patients')

# Jaundice patients by type
jaundice_pats = {}
for cat in ['mild', 'moderate', 'severe']:
    jaundice_pats.update(find_face_imgs(os.path.join(DATA, cat)))

# Classify by type
type_patients = {'hepatocellular': {}, 'cholestatic': {}}
for pid, data in jaundice_pats.items():
    jt = get_jtype(pid)
    if jt == 1 or jt == 2:  # hemolytic(1) merged into hepatocellular(2)
        type_patients['hepatocellular'][pid] = data
    elif jt == 3:  # cholestatic
        type_patients['cholestatic'][pid] = data
    # Skip type 4 (other, only 4 patients)

print(f'  Hepatocellular: {len(type_patients["hepatocellular"])} patients')
print(f'  Cholestatic: {len(type_patients["cholestatic"])} patients')
print(f'  Total jaundice: {sum(len(v) for v in type_patients.values())} patients')

# ═════════════════════════════════════════════════════════════
# DATASET CLASS
# ═════════════════════════════════════════════════════════════
IMG_SIZE = 224; BS = 32; EP = 30; LR = 1e-4

tr_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.03),
    transforms.RandomRotation(10), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class ImgDS(Dataset):
    def __init__(self, pats_dict, label, img_key, tf, mx=12):
        self.s = []; self.tf = tf
        for pid, data in pats_dict.items():
            for ip in data.get(img_key, [])[:mx]:
                self.s.append((ip, label, pid))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l, pi = self.s[i]
        try: im = Image.open(p).convert('RGB')
        except: im = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.tf(im), l, pi

def pagg(ps, ls, pis):
    d = {}
    for p, l, pi in zip(ps, ls, pis): d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pis = list(d.keys())
    return np.array([np.mean(d[p][0], 0) for p in pis]), np.array([d[p][1] for p in pis])

def split_dict(d, ratio=0.2):
    keys = list(d.keys()); random.shuffle(keys)
    n = max(1, int(len(keys) * ratio))
    return {k: d[k] for k in keys[n:]}, {k: d[k] for k in keys[:n]}

class FocalLoss(nn.Module):
    def __init__(self, a=None, g=2.0, ls=0.1):
        super().__init__(); self.a=a; self.g=g; self.ls=ls
    def forward(self, lo, t):
        nc=lo.size(1); st=torch.zeros_like(lo).scatter_(1,t.unsqueeze(1),1.0)
        st=st*(1-self.ls)+self.ls/nc
        lp=F.log_softmax(lo,1); p=torch.exp(lp)
        fw=(1-p.gather(1,t.unsqueeze(1)).clamp(min=1e-8))**self.g
        l=-(st*lp).sum(1)*fw.squeeze()
        return (l*self.a[t]).mean() if self.a is not None else l.mean()

class EMA:
    def __init__(self,m,d=0.999):
        self.d=d; self.s={n:p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
    def update(self,m):
        for n,p in m.named_parameters():
            if n in self.s: self.s[n]=self.d*self.s[n]+(1-self.d)*p.data
    def apply(self,m):
        b={}
        for n,p in m.named_parameters():
            if n in self.s: b[n]=p.data.clone(); p.data=self.s[n].clone()
        return b
    def restore(self,m,b):
        for n,p in m.named_parameters():
            if n in b: p.data=b[n].clone()

class YellowFeatures(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.c1=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.c2=nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.c3=nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2))
        self.p=nn.AdaptiveAvgPool2d(1)
        self.f=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,nc))
    def forward(self,x): return self.f(self.p(self.c3(self.c2(self.c1(x)))).flatten(1))

def train(m, trl, vrl, nc, task, sn):
    cnt=np.bincount([l for _,l,_ in trl.dataset.s],minlength=nc)
    al=torch.FloatTensor((1./cnt)/(1./cnt).sum()).to(DEV)
    cr=FocalLoss(a=al)
    op=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(op,T_max=EP)
    em=EMA(m); ba=0
    for ep in range(1,EP+1):
        m.train()
        for im,la,_ in tqdm(trl,desc=f'{task} E{ep}',leave=False):
            im,la=im.to(DEV),la.to(DEV)
            op.zero_grad(); lo=cr(m(im),la); lo.backward(); op.step(); em.update(m)
        sc.step()
        bk=em.apply(m); m.eval()
        ps,ls,pis=[],[],[]
        with torch.no_grad():
            for im,la,pi in vrl:
                o=m(im.to(DEV)); ps.extend(F.softmax(o,1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
        avg,true=pagg(np.array(ps),np.array(ls),pis)
        pl=avg.argmax(1)
        try: au=roc_auc_score(true,avg[:,1]) if nc==2 else roc_auc_score(true,avg,multi_class='ovr')
        except: au=0
        ac=accuracy_score(true,pl); f1=f1_score(true,pl,average='macro')
        if ep%10==0 or ep==1: print(f'  {task} E{ep:3d}: acc={ac:.4f} f1={f1:.4f} auc={au:.4f}')
        if au>ba: ba=au; torch.save(m.state_dict(),os.path.join(MODEL,f'{sn}.pt'))
        em.restore(m,bk)
    print(f'  {task} Best: {ba:.4f}')
    del m; torch.cuda.empty_cache()
    return ba

# ═════════════════════════════════════════════════════════════
# TASK A: Binary (normal vs jaundiced) — ALL MODELS
# ═════════════════════════════════════════════════════════════
print('\n[2] TASK A: Binary Screening (ALL models)')
all_jaundice = {}
for t in type_patients.values(): all_jaundice.update(t)
all_patients = {'normal': {k: v for k, v in normal_pats.items()}, 'jaundiced': all_jaundice}

tr_n, va_n = split_dict(normal_pats)
tr_j, va_j = split_dict(all_jaundice)

# Face datasets
tr_dict = {}; tr_dict.update({k: (v, 0) for k, v in tr_n.items()}); tr_dict.update({k: (v, 1) for k, v in tr_j.items()})
va_dict = {}; va_dict.update({k: (v, 0) for k, v in va_n.items()}); va_dict.update({k: (v, 1) for k, v in va_j.items()})

tr_ds = ImgDS({k: v for k, (v, _) in tr_dict.items()}, None, 'faces', tr_tf)  # Need to fix label handling
# Actually, simpler approach:
class BinDS(Dataset):
    def __init__(self, normal_dict, jaundice_dict, img_key, tf, mx=12):
        self.s = []; self.tf = tf
        for pid, data in normal_dict.items():
            for ip in data.get(img_key, [])[:mx]: self.s.append((ip, 0, pid))
        for pid, data in jaundice_dict.items():
            for ip in data.get(img_key, [])[:mx]: self.s.append((ip, 1, pid))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l, pi = self.s[i]
        try: im = Image.open(p).convert('RGB')
        except: im = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.tf(im), l, pi

tr_ds = BinDS(tr_n, tr_j, 'faces', tr_tf)
va_ds = BinDS(va_n, va_j, 'faces', ev_tf)
trl = DataLoader(tr_ds, BS, shuffle=True, num_workers=0)
vrl = DataLoader(va_ds, BS, shuffle=False, num_workers=0)
print(f'  Binary: train={len(tr_ds)} val={len(va_ds)} images')

# YellowFeatures use v3 sclera — skip for v4 (use old type_binary_yf.pt)
trl_s, vrl_s = trl, vrl

binary_results = {}
for bb_name, bb_id in [('ConvNeXt', 'convnext_tiny'), ('ViT', 'vit_tiny_patch16_224'),
                         ('EfficientNet', 'efficientnet_b0'), ('Swin', 'swin_tiny_patch4_window7_224')]:
    print(f'\n  >> Binary {bb_name}...')
    m = timm.create_model(bb_id, pretrained=True, num_classes=2).to(DEV)
    binary_results[bb_name] = train(m, trl, vrl, 2, f'Bin-{bb_name}', f'type_binary_{bb_id}')

# YellowFeatures for binary
print(f'\n  >> Binary YellowFeatures (sclera)...')
m = YellowFeatures(2).to(DEV)
binary_results['YellowFeatures'] = train(m, trl_s, vrl_s, 2, 'Bin-YF', 'type_binary_yf')

# ═════════════════════════════════════════════════════════════
# TASK B: Jaundice Type (hepatocellular=0 vs cholestatic=1)
# ═════════════════════════════════════════════════════════════
print('\n[3] TASK B: Jaundice Type Classification')

tr_h, va_h = split_dict(type_patients['hepatocellular'])
tr_c, va_c = split_dict(type_patients['cholestatic'])

class TypeDS(Dataset):
    def __init__(self, hep_dict, chol_dict, img_key, tf, mx=12):
        self.s = []; self.tf = tf
        for pid, data in hep_dict.items():
            for ip in data.get(img_key, [])[:mx]: self.s.append((ip, 0, pid))
        for pid, data in chol_dict.items():
            for ip in data.get(img_key, [])[:mx]: self.s.append((ip, 1, pid))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l, pi = self.s[i]
        try: im = Image.open(p).convert('RGB')
        except: im = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.tf(im), l, pi

tr_ds_t = TypeDS(tr_h, tr_c, 'faces', tr_tf)
va_ds_t = TypeDS(va_h, va_c, 'faces', ev_tf)
trl_t = DataLoader(tr_ds_t, BS, shuffle=True, num_workers=0)
vrl_t = DataLoader(va_ds_t, BS, shuffle=False, num_workers=0)
print(f'  Type: train={len(tr_ds_t)} val={len(va_ds_t)} images')

tr_ds_ts = TypeDS(tr_h, tr_c, 'faces', tr_tf)
va_ds_ts = TypeDS(va_h, va_c, 'faces', ev_tf)
trl_ts = DataLoader(tr_ds_ts, BS, shuffle=True, num_workers=0)
vrl_ts = DataLoader(va_ds_ts, BS, shuffle=False, num_workers=0)

type_results = {}
for bb_name, bb_id in [('ConvNeXt', 'convnext_tiny'), ('ViT', 'vit_tiny_patch16_224'),
                         ('EfficientNet', 'efficientnet_b0'), ('Swin', 'swin_tiny_patch4_window7_224')]:
    print(f'\n  >> Type {bb_name}...')
    m = timm.create_model(bb_id, pretrained=True, num_classes=2).to(DEV)
    type_results[bb_name] = train(m, trl_t, vrl_t, 2, f'Type-{bb_name}', f'type_class_{bb_id}')

# YellowFeatures for type
print(f'\n  >> Type YellowFeatures (sclera)...')
m = YellowFeatures(2).to(DEV)
type_results['YellowFeatures'] = train(m, trl_ts, vrl_ts, 2, 'Type-YF', 'type_class_yf')

# ═════════════════════════════════════════════════════════════
# ENSEMBLE
# ═════════════════════════════════════════════════════════════
print('\n[4] Ensemble...')
# Binary ensemble
bin_preds = {}
for bb_name, bb_id in [('ConvNeXt', 'convnext_tiny'), ('ViT', 'vit_tiny_patch16_224'),
                         ('EfficientNet', 'efficientnet_b0'), ('Swin', 'swin_tiny_patch4_window7_224')]:
    ckpt = os.path.join(MODEL, f'type_binary_{bb_id}.pt')
    if os.path.exists(ckpt):
        m = timm.create_model(bb_id, pretrained=False, num_classes=2).to(DEV)
        m.load_state_dict(torch.load(ckpt, weights_only=True)); m.eval()
        ps = []
        with torch.no_grad():
            for im, _, _ in vrl: ps.extend(F.softmax(m(im.to(DEV)), 1).cpu().numpy())
        bin_preds[bb_name] = np.array(ps)
        del m
torch.cuda.empty_cache()

# YF binary
yf_ckpt = os.path.join(MODEL, 'type_binary_yf.pt')
if os.path.exists(yf_ckpt):
    m = YellowFeatures(2).to(DEV)
    m.load_state_dict(torch.load(yf_ckpt, weights_only=True)); m.eval()
    ps = []
    with torch.no_grad():
        for im, _, _ in vrl_s: ps.extend(F.softmax(m(im.to(DEV)), 1).cpu().numpy())
    bin_preds['YellowFeatures'] = np.array(ps)
    del m
torch.cuda.empty_cache()

if len(bin_preds) >= 2:
    ens = np.mean(list(bin_preds.values()), axis=0)
    # Patient-level
    pids = []
    for im, la, pi in vrl: pids.extend(pi)
    avg, true = pagg(ens, np.array([0 if pi in va_n else 1 for pi in pids]), pids)
    try: ens_auc = roc_auc_score(true, avg[:, 1])
    except: ens_auc = 0
    binary_results['Ensemble'] = ens_auc
    print(f'  Binary Ensemble AUC: {ens_auc:.4f}')

# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  COMPLETE RESULTS')
print('='*60)
print('\n  Binary (normal vs jaundiced):')
for n, a in binary_results.items(): print(f'    {n:20s}: AUC={a:.4f}')
print('\n  Jaundice Type (hepato vs cholestatic):')
for n, a in type_results.items(): print(f'    {n:20s}: AUC={a:.4f}')

with open(os.path.join(RES, 'type_classification_results.json'), 'w') as f:
    json.dump({'binary': binary_results, 'type': type_results}, f, indent=2, default=str)

print('\n  Models saved in models/')
print('  External validation pending (need SAM processing first)')
