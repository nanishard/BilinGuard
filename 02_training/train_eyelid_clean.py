# -*- coding: utf-8 -*-
"""
BilinGuard Eyelid Ternary Training v2 (Clean Data)
Uses ONLY verified eyelid photos from clean_dataset_manifest.csv.
Body composition reports have been excluded.

Input: everted eyelid (翻上眼睑) photos only
Output: mild (0) / moderate (1) / severe (2)
"""
import os, json, random, time
import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-4
N_AUG = 8

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')


# ── Load clean data from manifest ────────────────────────────
print('\n[1] Loading CLEAN dataset from manifest...')
df = pd.read_csv(MANIFEST)

# Only jaundice patients with eyelid photos
j_df = df[(df['category'].isin(['mild', 'moderate', 'severe'])) & (df['n_eyelid'] > 0)]
print(f'  Jaundice patients with clean eyelid photos: {len(j_df)}')
for cat in ['mild', 'moderate', 'severe']:
    n = len(j_df[j_df['category'] == cat])
    total_imgs = j_df[j_df['category'] == cat]['n_eyelid'].sum()
    print(f'    {cat:12s}: {n:3d} patients, {total_imgs} eyelid photos')

# Build patient list with eyelid paths
patients = []
for _, row in j_df.iterrows():
    eyelid_paths = json.loads(row['eyelid_images'])
    if eyelid_paths:
        patients.append({
            'patient_id': row['patient_id'],
            'eyelid_path': eyelid_paths[0],  # Use first eyelid photo
            'category': row['category'],
        })

tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tfn = lambda p: tmap.get(p['category'], -1)


# ── Augmentation ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE * 2, IMG_SIZE * 2)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.15, 0.15, 0.08, 0.04),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class EyelidDataset(Dataset):
    def __init__(self, patients, label_map, transform, n_aug=1):
        self.samples = []
        self.transform = transform
        for p in patients:
            if p['category'] not in label_map:
                continue
            label = label_map[p['category']]
            for _ in range(n_aug):
                self.samples.append((p['eyelid_path'], label, p['patient_id']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
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


def create_model(num_classes, backbone='convnext_tiny'):
    return timm.create_model(backbone, pretrained=True, num_classes=num_classes).to(DEVICE)


def train_model(model, train_loader, val_loader, num_classes, task, save_name):
    counts = np.bincount([l for _, l, _ in train_loader.dataset.samples], minlength=num_classes)
    w = 1.0 / counts
    w = torch.FloatTensor(w / w.sum()).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_auc = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tloss = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f'E{epoch}/{EPOCHS} [{task}]', leave=False):
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

        avg, true = patient_agg(np.array(preds), np.array(labs), pids)
        pl = avg.argmax(axis=1)
        acc = accuracy_score(true, pl)
        f1 = f1_score(true, pl, average='macro')
        try:
            auc = roc_auc_score(true, avg, multi_class='ovr')
        except:
            auc = 0

        if epoch % 5 == 0 or epoch == 1:
            cm = confusion_matrix(true, pl, labels=[0, 1, 2])
            print(f'  E{epoch:3d}: loss={tloss/len(train_loader):.4f} acc={acc:.4f} '
                  f'f1={f1:.4f} auc={auc:.4f}')
            print(f'        CM: {cm.tolist()}')

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_name}.pt'))

    print(f'  Best AUC: {best_auc:.4f}')
    return best_auc


def stratified_split(patients, label_fn, val_ratio=0.2):
    by_l = {}
    for p in patients:
        l = label_fn(p)
        by_l.setdefault(l, []).append(p)
    train, val = [], []
    for l, g in by_l.items():
        random.shuffle(g)
        n = max(1, int(len(g) * val_ratio))
        val.extend(g[:n])
        train.extend(g[n:])
    return train, val


# ══════════════════════════════════════════════════════════════
# Ternary grading: mild / moderate / severe (clean eyelid only)
# ══════════════════════════════════════════════════════════════
print('\n[2] Ternary Grading: mild / moderate / severe (CLEAN eyelid)')
print('-' * 60)

train_p, val_p = stratified_split(patients, tfn)
print(f'  Train: {len(train_p)} patients')
print(f'  Val:   {len(val_p)} patients')
print(f'  Train dist: '
      f'mild={sum(1 for p in train_p if p["category"]=="mild")}, '
      f'moderate={sum(1 for p in train_p if p["category"]=="moderate")}, '
      f'severe={sum(1 for p in train_p if p["category"]=="severe")}')

train_ds = EyelidDataset(train_p, tmap, train_transform, n_aug=N_AUG)
val_ds = EyelidDataset(val_p, tmap, eval_transform, n_aug=1)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Samples: train={len(train_ds)} (x{N_AUG} aug), val={len(val_ds)}')

results = {}
for backbone in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0']:
    print(f'\n  Training {backbone}...')
    model = create_model(num_classes=3, backbone=backbone)
    save_name = f'eyelid_clean_ternary_{backbone}'
    auc = train_model(model, train_loader, val_loader, 3, backbone[:15], save_name)
    results[backbone] = auc
    del model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  CLEAN Eyelid Training Complete!')
print('=' * 60)
for bb, auc in results.items():
    print(f'  {bb:30s} AUC: {auc:.4f}')

with open(os.path.join(RESULTS_DIR, 'eyelid_clean_training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n  Models:')
for f in sorted(os.listdir(MODEL_DIR)):
    if f.startswith('eyelid_clean'):
        print(f'    {f} ({os.path.getsize(os.path.join(MODEL_DIR, f)) // 1024 // 1024} MB)')
