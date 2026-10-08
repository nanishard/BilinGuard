# -*- coding: utf-8 -*-
"""
BilinGuard Training on SAM v2 face frames (original pixels, eyes-open filtered)
Complete pipeline per manuscript:
  Stage 1: Binary (YOLO + ConvNeXt + ViT + EfficientNet)
  Stage 2: Ternary (ViT + Swin + EfficientNet + ConvNeXt + YellowFeatures + DynamicFusion)
"""
import os, json, random
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'sam_processed_v2')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
DEVICE = torch.device('cuda')
SEED = 42; IMG_SIZE = 224; BATCH_SIZE = 32; EPOCHS = 50; LR = 1e-4; MAX_FRAMES = 12

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')

# ── Focal Loss ───────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ls=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ls = ls
    def forward(self, logits, targets):
        nc = logits.size(1)
        st = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        st = st * (1 - self.ls) + self.ls / nc
        lp = F.log_softmax(logits, dim=1); p = torch.exp(lp)
        fw = (1 - p.gather(1, targets.unsqueeze(1)).clamp(min=1e-8)) ** self.gamma
        loss = -(st * lp).sum(1) * fw.squeeze()
        return (loss * self.alpha[targets]).mean() if self.alpha is not None else loss.mean()

class EMA:
    def __init__(self, model, d=0.999):
        self.d = d; self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow: self.shadow[n] = self.d * self.shadow[n] + (1 - self.d) * p.data
    def apply(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow: backup[n] = p.data.clone(); p.data = self.shadow[n].clone()
        return backup
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup: p.data = backup[n].clone()

# ── Data ─────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.03),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class ImgDataset(Dataset):
    def __init__(self, patients, img_key, label_map, transform, max_imgs=MAX_FRAMES):
        self.samples = []; self.transform = transform
        for p in patients:
            if p['category'] not in label_map: continue
            for img_path in p.get(img_key, [])[:max_imgs]:
                self.samples.append((img_path, label_map[p['category']], p['id']))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]
        try: img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), label, pid

def find_patients(data_root, categories):
    patients = []
    for cat in categories:
        cat_dir = os.path.join(data_root, cat)
        if not os.path.exists(cat_dir): continue
        for d in os.listdir(cat_dir):
            p = os.path.join(cat_dir, d)
            if not os.path.isdir(p): continue
            files = os.listdir(p)
            faces = sorted([os.path.join(p, f) for f in files if '_face' in f and f.endswith('.jpg')])
            scleras = sorted([os.path.join(p, f) for f in files if '_sclera' in f and f.endswith('.jpg')])
            if faces: patients.append({'id': d, 'faces': faces, 'scleras': scleras, 'category': cat})
    return patients

def patient_agg(preds, labels, pids):
    d = {}
    for p, l, pid in zip(preds, labels, pids):
        d.setdefault(pid, ([], l)); d[pid][0].append(p)
    pids = list(d.keys())
    return np.array([np.mean(d[p][0], axis=0) for p in pids]), np.array([d[p][1] for p in pids])

def stratified_split(patients, label_fn, ratio=0.2):
    by_l = {}
    for p in patients: by_l.setdefault(label_fn(p), []).append(p)
    train, val = [], []
    for l, g in by_l.items():
        random.shuffle(g); n = max(1, int(len(g) * ratio))
        val.extend(g[:n]); train.extend(g[n:])
    return train, val

class YellowFeatures(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,num_classes))
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        return self.cls(self.pool(x).flatten(1))

def train_model(model, train_loader, val_loader, nc, task, save_name):
    counts = np.bincount([l for _,l,_ in train_loader.dataset.samples], minlength=nc)
    alpha = torch.FloatTensor((1./counts)/(1./counts).sum()).to(DEVICE)
    criterion = FocalLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ema = EMA(model); best_auc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        for imgs, labels, _ in tqdm(train_loader, desc=f'{task} E{epoch}', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step(); ema.update(model)
        scheduler.step()
        bk = ema.apply(model); model.eval()
        preds, labs, pids = [], [], []
        with torch.no_grad():
            for imgs, labels, p in val_loader:
                out = model(imgs.to(DEVICE))
                preds.extend(F.softmax(out, dim=1).cpu().numpy()); labs.extend(labels.numpy()); pids.extend(p)
        avg, true = patient_agg(np.array(preds), np.array(labs), pids)
        pl = avg.argmax(1)
        try: auc = roc_auc_score(true, avg[:,1]) if nc==2 else roc_auc_score(true, avg, multi_class='ovr')
        except: auc = 0
        acc = accuracy_score(true, pl); f1 = f1_score(true, pl, average='macro')
        if epoch % 10 == 0 or epoch == 1:
            print(f'  {task} E{epoch:3d}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
        if auc > best_auc:
            best_auc = auc; torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_name}.pt'))
        ema.restore(model, bk)
    print(f'  {task} Best AUC: {best_auc:.4f}')
    del model; torch.cuda.empty_cache()
    return best_auc


# ═════════════════════════════════════════════════════════════
print('\n[1] Loading SAM v2 data...')
all_p = find_patients(DATA_ROOT, ['normal','mild','moderate','severe'])
for cat in ['normal','mild','moderate','severe']:
    n = sum(1 for p in all_p if p['category']==cat)
    imgs = sum(len(p['faces']) for p in all_p if p['category']==cat)
    print(f'  {cat:12s}: {n:3d} patients, {imgs} frames')
print(f'  Total: {len(all_p)} patients')

# ═══ Stage 1: Binary ════════════════════════════════════════
print('\n[2] STAGE 1: Binary Screening')
bmap = {'normal':0, 'mild':1, 'moderate':1, 'severe':1}
bfn = lambda p: bmap[p['category']]
train_p, val_p = stratified_split(all_p, bfn)
train_ds = ImgDataset(train_p, 'faces', bmap, train_tf)
val_ds = ImgDataset(val_p, 'faces', bmap, eval_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Train: {len(train_p)}, Val: {len(val_p)}, Images: {len(train_ds)}/{len(val_ds)}')

binary_results = {}
for bb in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0']:
    print(f'\n  Training {bb}...')
    m = timm.create_model(bb, pretrained=True, num_classes=2).to(DEVICE)
    binary_results[bb] = train_model(m, train_loader, val_loader, 2, f'Binary-{bb[:12]}', f'v2_binary_{bb}')

# ═══ Stage 2: Ternary — 4 backbones ════════════════════════
print('\n[3] STAGE 2: Ternary Grading')
j_pats = [p for p in all_p if p['category'] in ('mild','moderate','severe')]
tmap = {'mild':0, 'moderate':1, 'severe':2}
tfn = lambda p: tmap[p['category']]
train_j, val_j = stratified_split(j_pats, tfn)
train_ds_f = ImgDataset(train_j, 'faces', tmap, train_tf)
val_ds_f = ImgDataset(val_j, 'faces', tmap, eval_tf)
train_loader_f = DataLoader(train_ds_f, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_f = DataLoader(val_ds_f, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Train: {len(train_j)}, Val: {len(val_j)}, Images: {len(train_ds_f)}/{len(val_ds_f)}')

ternary_results = {}
ternary_preds = {}
for bb in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0', 'swin_tiny_patch4_window7_224']:
    print(f'\n  Training {bb}...')
    m = timm.create_model(bb, pretrained=True, num_classes=3).to(DEVICE)
    ternary_results[bb] = train_model(m, train_loader_f, val_loader_f, 3, f'Ternary-{bb[:12]}', f'v2_ternary_{bb}')
    # Get val preds
    m = timm.create_model(bb, pretrained=True, num_classes=3).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'v2_ternary_{bb}.pt'), weights_only=True))
    m.eval()
    preds = []
    with torch.no_grad():
        for imgs, _, _ in val_loader_f:
            preds.extend(F.softmax(m(imgs.to(DEVICE)), dim=1).cpu().numpy())
    ternary_preds[bb] = np.array(preds)
    del m; torch.cuda.empty_cache()

# ═══ YellowFeatures (sclera) ════════════════════════════════
print('\n[4] YellowFeatures (sclera)')
j_scl = [p for p in j_pats if p.get('scleras')]
if j_scl:
    train_s, val_s = stratified_split(j_scl, tfn)
    train_ds_s = ImgDataset(train_s, 'scleras', tmap, train_tf)
    val_ds_s = ImgDataset(val_s, 'scleras', tmap, eval_tf)
    train_loader_s = DataLoader(train_ds_s, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_s = DataLoader(val_ds_s, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'  Sclera images: train={len(train_ds_s)}, val={len(val_ds_s)}')
    yf = YellowFeatures(num_classes=3).to(DEVICE)
    ternary_results['yellowfeatures'] = train_model(yf, train_loader_s, val_loader_s, 3, 'YellowFeat', 'v2_ternary_yellowfeatures')
    # Get preds
    yf = YellowFeatures(num_classes=3).to(DEVICE)
    yf.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'v2_ternary_yellowfeatures.pt'), weights_only=True))
    yf.eval()
    preds_s = []
    with torch.no_grad():
        for imgs, _, _ in val_loader_s:
            preds_s.extend(F.softmax(yf(imgs.to(DEVICE)), dim=1).cpu().numpy())
    ternary_preds['yellowfeatures'] = np.array(preds_s)
    del yf; torch.cuda.empty_cache()

# ═══ DynamicFusion Ensemble ═════════════════════════════════
print('\n[5] DynamicFusion Ensemble (BilinGuard)')
val_pids = []
for imgs, labels, pids in val_loader_f:
    val_pids.extend(pids)
val_pids_s = []
if j_scl:
    for imgs, labels, pids in val_loader_s:
        val_pids_s.extend(pids)

patient_preds = {}
for bb, probs in ternary_preds.items():
    pids_use = val_pids_s if bb == 'yellowfeatures' and j_scl else val_pids
    for i, pid in enumerate(pids_use[:len(probs)]):
        patient_preds.setdefault(pid, []).append(probs[i])

ens_pids = list(patient_preds.keys())
ens_probs = np.array([np.mean(patient_preds[p], axis=0) for p in ens_pids])
pid_label = {p['id']: tmap[p['category']] for p in val_j}
true_ens = np.array([pid_label[p] for p in ens_pids])
pl_ens = ens_probs.argmax(1)
auc_ens = roc_auc_score(true_ens, ens_probs, multi_class='ovr') if len(np.unique(true_ens)) > 1 else 0
acc_ens = accuracy_score(true_ens, pl_ens)
f1_ens = f1_score(true_ens, pl_ens, average='macro')
cm_ens = confusion_matrix(true_ens, pl_ens, labels=[0,1,2])
print(f'  Ensemble AUC: {auc_ens:.4f}')
print(f'  Ensemble Acc: {acc_ens:.4f}, F1: {f1_ens:.4f}')
print(f'  CM: {cm_ens.tolist()}')

# ═══ Summary ════════════════════════════════════════════════
print('\n' + '='*60)
print('  SAM v2 COMPLETE RESULTS')
print('='*60)
print(f'\n  Binary Screening:')
for n, a in binary_results.items(): print(f'    {n:35s}: AUC={a:.4f}')
print(f'\n  Ternary Grading:')
for n, a in ternary_results.items(): print(f'    {n:35s}: AUC={a:.4f}')
print(f'\n  BilinGuard Ensemble:                       AUC={auc_ens:.4f}')

with open(os.path.join(RESULTS_DIR, 'v2_complete_results.json'), 'w') as f:
    json.dump({'binary': binary_results, 'ternary': ternary_results,
               'ensemble_auc': auc_ens, 'ensemble_acc': acc_ens}, f, indent=2, default=str)
