# -*- coding: utf-8 -*-
"""Continue eyelid training: EfficientNet + ConvNeXt (ViT already done)."""
import os, json, random, time
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')
MODEL_DIR = os.path.join(BASE, 'models')
DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4
N_AUG = 6

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')

def find_eyelid_photos(data_root, category):
    cat_dir = os.path.join(data_root, category)
    if not os.path.exists(cat_dir): return []
    patients = []
    for outer in os.listdir(cat_dir):
        p = os.path.join(cat_dir, outer)
        if not os.path.isdir(p): continue
        inner = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
        base = os.path.join(p, inner[0]) if inner else p
        for f in os.listdir(base):
            if (f.lower().endswith('.jpg') and f.startswith('IMG_')
                    and 'feature' not in f.lower()
                    and os.path.getsize(os.path.join(base, f)) > 2_000_000):
                patients.append({'patient_id': outer, 'eyelid_path': os.path.join(base, f), 'category': category})
                break
    return patients

print('\n[1] Loading eyelid photos...')
all_p = []
for cat in ['mild', 'moderate', 'severe']:
    ps = find_eyelid_photos(DATA_ROOT, cat)
    print(f'  {cat}: {len(ps)} patients')
    all_p.extend(ps)
print(f'  Total: {len(all_p)}')

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE*2, IMG_SIZE*2)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.15, 0.15, 0.08, 0.04),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class DS(Dataset):
    def __init__(self, pats, lmap, tf, n_aug=1):
        self.s = []; self.tf = tf
        for p in pats:
            if p['category'] not in lmap: continue
            for _ in range(n_aug):
                self.s.append((p['eyelid_path'], lmap[p['category']], p['patient_id']))
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

def train_one(backbone, train_loader, val_loader, save_name):
    print(f'\n=== Training {backbone} ===')
    model = timm.create_model(backbone, pretrained=True, num_classes=3).to(DEVICE)
    counts = np.bincount([l for _,l,_ in train_loader.dataset.s], minlength=3)
    w = torch.FloatTensor(1./counts).to(DEVICE); w = w/w.sum()
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_auc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        tloss = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f'E{epoch}/{EPOCHS}', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        scheduler.step()
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
        pl = avg.argmax(axis=1)
        acc = accuracy_score(true, pl)
        f1 = f1_score(true, pl, average='macro')
        try: auc = roc_auc_score(true, avg, multi_class='ovr')
        except: auc = 0
        if epoch % 5 == 0 or epoch == 1:
            cm = confusion_matrix(true, pl, labels=[0,1,2])
            print(f'  E{epoch:3d}: loss={tloss/len(train_loader):.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
            print(f'        CM: {cm.tolist()}')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_name}.pt'))
    print(f'  Best AUC: {best_auc:.4f}')
    del model; torch.cuda.empty_cache()
    return best_auc

# Split
tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tfn = lambda p: tmap.get(p['category'], -1)
by_l = {}
for p in all_p:
    by_l.setdefault(tfn(p), []).append(p)
train_p, val_p = [], []
for l, g in by_l.items():
    random.shuffle(g)
    n = max(1, int(len(g)*0.2))
    val_p.extend(g[:n])
    train_p.extend(g[n:])
print(f'\nTrain: {len(train_p)}, Val: {len(val_p)}')

train_ds = DS(train_p, tmap, train_tf, n_aug=N_AUG)
val_ds = DS(val_p, tmap, eval_tf, n_aug=1)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'Samples: train={len(train_ds)}, val={len(val_ds)}')

results = {}
for bb in ['efficientnet_b0', 'convnext_tiny']:
    ckpt = os.path.join(MODEL_DIR, f'eyelid_ternary_{bb}.pt')
    if os.path.exists(ckpt):
        print(f'\n  {bb} already trained, skipping')
        continue
    auc = train_one(bb, train_loader, val_loader, f'eyelid_ternary_{bb}')
    results[bb] = auc

print('\n' + '='*50)
print('  Eyelid Training Complete!')
print('='*50)
print(f'  ViT: AUC=0.8626 (already saved)')
for bb, auc in results.items():
    print(f'  {bb}: AUC={auc:.4f}')
print(f'\n  All eyelid models:')
for f in sorted(os.listdir(MODEL_DIR)):
    if f.startswith('eyelid_'):
        print(f'    {f} ({os.path.getsize(os.path.join(MODEL_DIR,f))//1024//1024} MB)')
