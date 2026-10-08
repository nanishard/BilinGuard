# -*- coding: utf-8 -*-
"""Train only ternary grading (binary already done)."""
import os, sys, json, random, time
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def find_images(data_root, category):
    cat_dir = os.path.join(data_root, category)
    if not os.path.exists(cat_dir): return []
    patients = []
    for outer in os.listdir(cat_dir):
        p = os.path.join(cat_dir, outer)
        if not os.path.isdir(p): continue
        inner = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
        base = os.path.join(p, inner[0]) if inner else p
        imgs = [os.path.join(base, f) for f in os.listdir(base)
                if f.lower().endswith(('.jpg','.jpeg','.png')) and 'feature' not in f.lower()]
        for sub in os.listdir(base):
            sp = os.path.join(base, sub)
            if os.path.isdir(sp):
                imgs += [os.path.join(sp, f) for f in os.listdir(sp) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if imgs: patients.append({'id': outer, 'images': imgs, 'category': category})
    return patients

print('Loading data...')
all_p = []
for cat in ['normal','mild','moderate','severe']:
    ps = find_images(DATA_ROOT, cat)
    print(f'  {cat}: {len(ps)} patients')
    all_p.extend(ps)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1,0.1,0.05,0.03),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class DS(Dataset):
    def __init__(self, pats, lmap, transform, max_i=12):
        self.s = []
        self.transform = transform
        for p in pats:
            if p['category'] not in lmap: continue
            for img in p['images'][:max_i]:
                self.s.append((img, lmap[p['category']], p['id']))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        path, label, pid = self.s[i]
        try: img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE,IMG_SIZE), (128,128,128))
        return self.transform(img), label, pid

def patient_agg(preds, labels, pids):
    d = {}
    for p, l, pid in zip(preds, labels, pids):
        d.setdefault(pid, ([], l))
        d[pid][0].append(p)
    pids = list(d.keys())
    avg = np.array([np.mean(d[p][0], axis=0) for p in pids])
    true = np.array([d[p][1] for p in pids])
    return avg, true

def train_one(backbone, num_classes, train_loader, val_loader, save_name):
    print(f'\n=== Training {backbone} ({num_classes} classes) ===')
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes).to(DEVICE)
    counts = np.bincount([l for _,l,_ in train_loader.dataset.s], minlength=num_classes)
    w = torch.FloatTensor(1./counts).to(DEVICE)
    w = w / w.sum()
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
                probs = F.softmax(out, dim=1)
                preds.extend(probs.cpu().numpy())
                labs.extend(labels.numpy())
                pids.extend(p)
        avg, true = patient_agg(np.array(preds), np.array(labs), pids)
        pl = avg.argmax(axis=1)
        acc = accuracy_score(true, pl)
        f1 = f1_score(true, pl, average='macro')
        try:
            auc = roc_auc_score(true, avg, multi_class='ovr') if num_classes > 2 else roc_auc_score(true, avg[:,1])
        except: auc = 0
        if epoch % 5 == 0 or epoch == 1:
            print(f'  E{epoch}: loss={tloss/len(train_loader):.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_name}.pt'))
    print(f'  Best AUC: {best_auc:.4f}')
    del model
    torch.cuda.empty_cache()
    return best_auc

# ── Ternary ──
print('\n[Stage 2] Ternary grading: mild/moderate/severe')
j_pats = [p for p in all_p if p['category'] in ('mild','moderate','severe')]
lmap = {'mild':0, 'moderate':1, 'severe':2}
by_l = {}
for p in j_pats:
    by_l.setdefault(lmap[p['category']], []).append(p)
train_p, val_p = [], []
for l, g in by_l.items():
    random.shuffle(g)
    n = max(1, int(len(g)*0.2))
    val_p.extend(g[:n])
    train_p.extend(g[n:])
print(f'  Train: {len(train_p)}, Val: {len(val_p)}')
train_ds = DS(train_p, lmap, train_tf)
val_ds = DS(val_p, lmap, eval_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Images: train={len(train_ds)}, val={len(val_ds)}')

results = {}
for bb in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0']:
    auc = train_one(bb, 3, train_loader, val_loader, f'ternary_{bb}')
    results[bb] = auc

print('\n' + '='*50)
print('Training complete!')
print('='*50)
for bb, auc in results.items():
    print(f'  {bb}: AUC={auc:.4f}')
with open(os.path.join(RESULTS_DIR, 'ternary_training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nModels in {MODEL_DIR}:')
for f in os.listdir(MODEL_DIR):
    print(f'  {f}')
