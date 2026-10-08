# -*- coding: utf-8 -*-
"""
BilinGuard Eyelid Training v3 — Optimized (Single Split)
Optimizations:
  1. Focal Loss (γ=2.0) with label smoothing (ε=0.1)
  2. MixUp (α=0.2) 
  3. OneCycleLR (max_lr=2e-4, pct_start=0.1)
  4. SWA (from epoch 50)
  5. EMA (decay=0.999)
  6. WeightedRandomSampler
  7. RandomErasing + RandomPerspective augmentation
  8. Gradient clipping
Backbones: ConvNeXt-Tiny, ViT-Tiny, Swin-Tiny (replaces EfficientNet)
"""
import os, json, random, time, copy
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')

DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 80
LR_MAX = 2e-4
WD = 1e-4
LS = 0.1
MIXUP_A = 0.2
GAMMA = 2.0
N_AUG = 10
SWA_START = 50
EMA_DECAY = 0.999

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE}')

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=GAMMA, ls=LS):
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
        return (loss * self.alpha[targets]).mean()

def mixup(x, y, a=MIXUP_A):
    lam = np.random.beta(a, a) if a > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

class EMA:
    def __init__(self, model, d=EMA_DECAY):
        self.d = d; self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
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

# Data
print('\n[1] Loading data...')
df = pd.read_csv(MANIFEST)
jdf = df[(df['category'].isin(['mild','moderate','severe'])) & (df['n_eyelid']>0)]
patients = []
for _, r in jdf.iterrows():
    paths = json.loads(r['eyelid_images'])
    if paths:
        patients.append({'id': r['patient_id'], 'eyelid_path': paths[0], 'category': r['category']})

tmap = {'mild':0, 'moderate':1, 'severe':2}
tfn = lambda p: tmap[p['category']]

by_l = {}
for p in patients:
    by_l.setdefault(tfn(p), []).append(p)
train_p, val_p = [], []
for l, g in by_l.items():
    random.shuffle(g)
    n = max(1, int(len(g)*0.2))
    val_p.extend(g[:n]); train_p.extend(g[n:])

print(f'  Train: {len(train_p)}, Val: {len(val_p)}')

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE*2, IMG_SIZE*2)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.15,0.15), scale=(0.85,1.15), shear=10),
    transforms.RandomPerspective(0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02,0.1)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class DS(Dataset):
    def __init__(self, pats, lmap, tf, n=1):
        self.s = []; self.tf = tf
        for p in pats:
            if p['category'] in lmap:
                for _ in range(n): self.s.append((p['eyelid_path'], lmap[p['category']], p['id']))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        path, label, pid = self.s[i]
        try: img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE,IMG_SIZE), (128,128,128))
        return self.tf(img), label, pid

def pagg(preds, labels, pids):
    d = {}
    for p, l, pid in zip(preds, labels, pids):
        d.setdefault(pid, ([], l))
        d[pid][0].append(p)
    pids = list(d.keys())
    return np.array([np.mean(d[p][0], axis=0) for p in pids]), np.array([d[p][1] for p in pids])

BACKBONES = ['convnext_tiny', 'vit_tiny_patch16_224', 'swin_tiny_patch4_window7_224']
results = {}

for bb in BACKBONES:
    print(f'\n{"="*55}')
    print(f'  Optimized Training: {bb}')
    print(f'{"="*55}')
    
    model = timm.create_model(bb, pretrained=True, num_classes=3).to(DEVICE)
    counts = np.bincount([tmap[p['category']] for p in train_p], minlength=3)
    alpha = torch.FloatTensor((1./counts)/(1./counts).sum()).to(DEVICE)
    criterion = FocalLoss(alpha=alpha)
    
    train_ds = DS(train_p, tmap, train_tf, n=N_AUG)
    val_ds = DS(val_p, tmap, eval_tf, n=1)
    sw = [1./counts[tmap[p['category']]] for p in train_p for _ in range(N_AUG)]
    sampler = WeightedRandomSampler(sw, len(sw))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, epochs=EPOCHS,
        steps_per_epoch=len(train_ds)//BATCH_SIZE+1, pct_start=0.1)
    
    ema = EMA(model)
    swa_model = copy.deepcopy(model)
    swa_n = 0
    best_auc = 0
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        tloss = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f'E{epoch}/{EPOCHS}', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            mx, ya, yb, lam = mixup(imgs, labels)
            optimizer.zero_grad()
            out = model(mx)
            lp = F.log_softmax(out, dim=1)
            loss = lam * criterion(out, ya) + (1-lam) * criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            tloss += loss.item()
        
        # SWA
        if epoch >= SWA_START:
            swa_n += 1
            with torch.no_grad():
                for ps, pm in zip(swa_model.parameters(), model.parameters()):
                    ps.data = (ps.data*(swa_n-1) + pm.data) / swa_n
        
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
        avg, true = pagg(np.array(preds), np.array(labs), pids)
        pl = avg.argmax(1)
        acc = accuracy_score(true, pl)
        f1 = f1_score(true, pl, average='macro')
        try: auc = roc_auc_score(true, avg, multi_class='ovr')
        except: auc = 0
        
        if epoch % 5 == 0 or epoch == 1:
            cm = confusion_matrix(true, pl, labels=[0,1,2])
            print(f'  E{epoch:3d}: loss={tloss/len(train_loader):.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
            print(f'        CM={cm.tolist()}')
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'eyelid_opt_{bb}.pt'))
        
        ema.restore(model, bk)
    
    # SWA eval
    if swa_n > 0:
        swa_model.eval()
        preds, labs, pids = [], [], []
        with torch.no_grad():
            for imgs, labels, p in val_loader:
                out = swa_model(imgs.to(DEVICE))
                preds.extend(F.softmax(out, dim=1).cpu().numpy())
                labs.extend(labels.numpy()); pids.extend(p)
        avg, true = pagg(np.array(preds), np.array(labs), pids)
        try: swa_auc = roc_auc_score(true, avg, multi_class='ovr')
        except: swa_auc = 0
        print(f'  SWA AUC: {swa_auc:.4f}')
        if swa_auc > best_auc:
            best_auc = swa_auc
            torch.save(swa_model.state_dict(), os.path.join(MODEL_DIR, f'eyelid_opt_{bb}.pt'))
            print('  SWA model saved (best)')
    
    print(f'  Best AUC: {best_auc:.4f}')
    results[bb] = best_auc
    del model, swa_model
    torch.cuda.empty_cache()

# Ensemble eval
print('\n[3] Ensemble evaluation...')
ens_models = {}
for bb in BACKBONES:
    ckpt = os.path.join(MODEL_DIR, f'eyelid_opt_{bb}.pt')
    if os.path.exists(ckpt):
        m = timm.create_model(bb, pretrained=False, num_classes=3).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        ens_models[bb] = m

val_ds = DS(val_p, tmap, eval_tf, n=1)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
preds, labs, pids = [], [], []
with torch.no_grad():
    for imgs, labels, p in val_loader:
        imgs = imgs.to(DEVICE)
        ps = []
        for m in ens_models.values():
            ps.append(F.softmax(m(imgs), dim=1))
        avg_p = torch.stack(ps).mean(0)
        preds.extend(avg_p.cpu().numpy())
        labs.extend(labels.numpy()); pids.extend(p)
avg, true = pagg(np.array(preds), np.array(labs), pids)
try: ens_auc = roc_auc_score(true, avg, multi_class='ovr')
except: ens_auc = 0
results['ensemble'] = ens_auc

# Save ensemble model info
torch.save({bb: os.path.join(MODEL_DIR, f'eyelid_opt_{bb}.pt') for bb in ens_models},
           os.path.join(MODEL_DIR, 'eyelid_opt_ensemble_info.pt'))

print('\n' + '='*55)
print('  Optimized Training Complete!')
print('='*55)
for n, a in results.items():
    print(f'  {n:35s}: AUC={a:.4f}')
with open(os.path.join(RESULTS_DIR, 'eyelid_optimized_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
