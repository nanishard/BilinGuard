# -*- coding: utf-8 -*-
"""
BilinGuard Training on SAM-processed face frames (per manuscript Section 6-7)
  Stage 1: Binary screening (normal vs jaundiced) — all categories
  Stage 2: Ternary grading (mild/moderate/severe) — jaundice only
  
Input: data/sam_processed/{category}/{patient}/frame_XXX_face.jpg
Pipeline: SAM segment → mean-color fill → CLAHE → 224x224 (already done)
Training: Optimized (Focal Loss + MixUp + OneCycleLR + EMA + WeightedSampler)
"""
import os, json, random, copy
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
DATA_ROOT = os.path.join(BASE, 'data', 'sam_processed')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60
LR_MAX = 1e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2
FOCAL_GAMMA = 2.0
MAX_FRAMES = 12

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')


# ═════════════════════════════════════════════════════════════
# LOSSES & AUGS
# ═════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, ls=LABEL_SMOOTH):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ls = ls
    def forward(self, logits, targets):
        nc = logits.size(1)
        st = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        st = st * (1 - self.ls) + self.ls / nc
        lp = F.log_softmax(logits, dim=1)
        p = torch.exp(lp)
        fw = (1 - p.gather(1, targets.unsqueeze(1)).clamp(min=1e-8)) ** self.gamma
        loss = -(st * lp).sum(1) * fw.squeeze()
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()

def mixup(x, y, a=MIXUP_ALPHA):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

class EMA:
    def __init__(self, model, d=0.999):
        self.d = d
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n] = self.d * self.shadow[n] + (1 - self.d) * p.data
    def apply(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data = self.shadow[n].clone()
        return backup
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data = backup[n].clone()


# ═════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1, 0.1, 0.05, 0.03),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class FaceDataset(Dataset):
    def __init__(self, patients, label_map, transform, max_frames=MAX_FRAMES):
        self.samples = []; self.transform = transform
        for p in patients:
            if p['category'] not in label_map: continue
            label = label_map[p['category']]
            for img_path in p['images'][:max_frames]:
                self.samples.append((img_path, label, p['id']))
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
            imgs = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
            if imgs:
                patients.append({'id': d, 'images': imgs, 'category': cat})
    return patients

def patient_agg(preds, labels, pids):
    d = {}
    for p, l, pid in zip(preds, labels, pids):
        d.setdefault(pid, ([], l))
        d[pid][0].append(p)
    pids = list(d.keys())
    return np.array([np.mean(d[p][0], axis=0) for p in pids]), np.array([d[p][1] for p in pids])

def stratified_split(patients, label_fn, ratio=0.2):
    by_l = {}
    for p in patients:
        by_l.setdefault(label_fn(p), []).append(p)
    train, val = [], []
    for l, g in by_l.items():
        random.shuffle(g)
        n = max(1, int(len(g) * ratio))
        val.extend(g[:n]); train.extend(g[n:])
    return train, val


# ═════════════════════════════════════════════════════════════
# TRAIN
# ═════════════════════════════════════════════════════════════
def train_model(backbone, train_loader, val_loader, num_classes, task, save_name):
    print(f'\n{"="*55}')
    print(f'  {task}: {backbone} ({num_classes} classes)')
    print(f'{"="*55}')
    
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes).to(DEVICE)
    counts = np.bincount([l for _, l, _ in train_loader.dataset.samples], minlength=num_classes)
    alpha = torch.FloatTensor((1./counts) / (1./counts).sum()).to(DEVICE)
    criterion = FocalLoss(alpha=alpha)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, epochs=EPOCHS,
        steps_per_epoch=len(train_loader.dataset)//BATCH_SIZE+1, pct_start=0.1)
    
    ema = EMA(model)
    best_auc = 0
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        tloss = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f'E{epoch}/{EPOCHS}', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            mx, ya, yb, lam = mixup(imgs, labels)
            optimizer.zero_grad()
            out = model(mx)
            loss = lam * criterion(out, ya) + (1-lam) * criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            tloss += loss.item()
        
        # Eval with EMA
        bk = ema.apply(model)
        model.eval()
        preds, labs, pids = [], [], []
        with torch.no_grad():
            for imgs, labels, p in val_loader:
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                preds.extend(F.softmax(out, dim=1).cpu().numpy())
                labs.extend(labels.numpy())
                pids.extend(p)
        avg, true = patient_agg(np.array(preds), np.array(labs), pids)
        pl = avg.argmax(1)
        acc = accuracy_score(true, pl)
        f1 = f1_score(true, pl, average='macro')
        try:
            auc = roc_auc_score(true, avg[:,1]) if num_classes==2 else roc_auc_score(true, avg, multi_class='ovr')
        except: auc = 0
        
        if epoch % 5 == 0 or epoch == 1:
            cm = confusion_matrix(true, pl, labels=list(range(num_classes)))
            print(f'  E{epoch:3d}: loss={tloss/len(train_loader):.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
            print(f'        CM={cm.tolist()}')
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_name}.pt'))
        
        ema.restore(model, bk)
    
    print(f'  Best AUC: {best_auc:.4f}')
    del model; torch.cuda.empty_cache()
    return best_auc


# ═════════════════════════════════════════════════════════════
# STAGE 1: Binary Screening
# ═════════════════════════════════════════════════════════════
print('\n[1] Loading SAM-processed data...')
all_patients = find_patients(DATA_ROOT, ['normal', 'mild', 'moderate', 'severe'])
for cat in ['normal', 'mild', 'moderate', 'severe']:
    n = sum(1 for p in all_patients if p['category']==cat)
    imgs = sum(len(p['images']) for p in all_patients if p['category']==cat)
    print(f'  {cat:12s}: {n:3d} patients, {imgs} face frames')
print(f'  Total: {len(all_patients)} patients')

binary_map = {'normal': 0, 'mild': 1, 'moderate': 1, 'severe': 1}
binary_fn = lambda p: binary_map[p['category']]

print('\n[2] Stage 1: Binary Screening (normal vs jaundiced)')
print('-' * 55)
train_p, val_p = stratified_split(all_patients, binary_fn)
print(f'  Train: {len(train_p)} patients, Val: {len(val_p)} patients')

train_ds = FaceDataset(train_p, binary_map, train_tf)
val_ds = FaceDataset(val_p, binary_map, eval_tf)
counts = np.bincount([l for _, l, _ in train_ds.samples], minlength=2)
sw = [1./counts[l] for _, l, _ in train_ds.samples]
sampler = WeightedRandomSampler(sw, len(sw))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Images: train={len(train_ds)}, val={len(val_ds)}')

binary_results = {}
for bb in ['convnext_tiny', 'vit_tiny_patch16_224']:
    auc = train_model(bb, train_loader, val_loader, 2, 'Binary', f'face_binary_sam_{bb}')
    binary_results[bb] = auc


# ═════════════════════════════════════════════════════════════
# STAGE 2: Ternary Grading
# ═════════════════════════════════════════════════════════════
print('\n[3] Stage 2: Ternary Grading (mild/moderate/severe)')
print('-' * 55)
j_pats = [p for p in all_patients if p['category'] in ('mild', 'moderate', 'severe')]
ternary_map = {'mild': 0, 'moderate': 1, 'severe': 2}
ternary_fn = lambda p: ternary_map[p['category']]

train_j, val_j = stratified_split(j_pats, ternary_fn)
print(f'  Train: {len(train_j)} patients, Val: {len(val_j)} patients')

train_ds_j = FaceDataset(train_j, ternary_map, train_tf)
val_ds_j = FaceDataset(val_j, ternary_map, eval_tf)
counts_j = np.bincount([l for _, l, _ in train_ds_j.samples], minlength=3)
sw_j = [1./counts_j[l] for _, l, _ in train_ds_j.samples]
sampler_j = WeightedRandomSampler(sw_j, len(sw_j))
train_loader_j = DataLoader(train_ds_j, batch_size=BATCH_SIZE, sampler=sampler_j, num_workers=0)
val_loader_j = DataLoader(val_ds_j, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Images: train={len(train_ds_j)}, val={len(val_ds_j)}')

ternary_results = {}
for bb in ['convnext_tiny', 'vit_tiny_patch16_224', 'swin_tiny_patch4_window7_224']:
    auc = train_model(bb, train_loader_j, val_loader_j, 3, 'Ternary', f'face_ternary_sam_{bb}')
    ternary_results[bb] = auc


# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n' + '=' * 55)
print('  SAM Face Training Complete!')
print('=' * 55)
print(f'\n  Binary Screening:')
for bb, auc in binary_results.items():
    print(f'    {bb:30s}: AUC={auc:.4f}')
print(f'\n  Ternary Grading:')
for bb, auc in ternary_results.items():
    print(f'    {bb:30s}: AUC={auc:.4f}')

with open(os.path.join(RESULTS_DIR, 'sam_face_training_results.json'), 'w') as f:
    json.dump({'binary': binary_results, 'ternary': ternary_results}, f, indent=2)

print(f'\n  Models:')
for f in sorted(os.listdir(MODEL_DIR)):
    if 'face_' in f and 'sam' in f:
        print(f'    {f} ({os.path.getsize(os.path.join(MODEL_DIR, f))//1024//1024} MB)')
