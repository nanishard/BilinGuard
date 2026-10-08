# -*- coding: utf-8 -*-
"""
BilinGuard Eyelid-Only Ternary Training
Trains a ternary classifier (mild / moderate / severe) using ONLY
everted eyelid (翻上眼睑) photos. Normal patients do not have eyelid photos.

Input: 1 high-resolution IMG_*.jpg per jaundice patient
Output: mild (0) / moderate (1) / severe (2)
"""
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
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-4
N_AUG = 8  # augmented views per eyelid photo

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')


# ── Find eyelid photos (jaundice patients only) ──────────────
def find_eyelid_photos(data_root, category):
    cat_dir = os.path.join(data_root, category)
    if not os.path.exists(cat_dir):
        return []
    patients = []
    for outer in os.listdir(cat_dir):
        outer_path = os.path.join(cat_dir, outer)
        if not os.path.isdir(outer_path):
            continue
        inner_dirs = [d for d in os.listdir(outer_path)
                      if os.path.isdir(os.path.join(outer_path, d))]
        inner_path = os.path.join(outer_path, inner_dirs[0]) if inner_dirs else outer_path
        for f in os.listdir(inner_path):
            if (f.lower().endswith('.jpg') and f.startswith('IMG_')
                    and 'feature' not in f.lower()
                    and os.path.getsize(os.path.join(inner_path, f)) > 2_000_000):
                patients.append({
                    'patient_id': outer,
                    'eyelid_path': os.path.join(inner_path, f),
                    'category': category,
                })
                break
    return patients


print('\n[1] Loading eyelid photos (jaundice patients only)...')
all_patients = []
for cat in ['mild', 'moderate', 'severe']:
    patients = find_eyelid_photos(DATA_ROOT, cat)
    print(f'  {cat:12s}: {len(patients):3d} patients')
    all_patients.extend(patients)
print(f'  Total: {len(all_patients)} jaundice patients with eyelid photos')


# ── Augmentation ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE * 2, IMG_SIZE * 2)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.04),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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


def create_model(num_classes, backbone='vit_tiny_patch16_224'):
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
            cm = confusion_matrix(true, pl, labels=list(range(num_classes)))
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
# Ternary grading: mild / moderate / severe (eyelid only)
# ══════════════════════════════════════════════════════════════
print('\n[2] Ternary Grading: mild / moderate / severe (eyelid only)')
print('-' * 55)

tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tfn = lambda p: tmap.get(p['category'], -1)

train_p, val_p = stratified_split(all_patients, tfn)
print(f'  Train: {len(train_p)} patients')
print(f'  Val:   {len(val_p)} patients')
print(f'  Train distribution: '
      f'mild={sum(1 for p in train_p if p["category"]=="mild")}, '
      f'moderate={sum(1 for p in train_p if p["category"]=="moderate")}, '
      f'severe={sum(1 for p in train_p if p["category"]=="severe")}')

train_ds = EyelidDataset(train_p, tmap, train_transform, n_aug=N_AUG)
val_ds = EyelidDataset(val_p, tmap, eval_transform, n_aug=1)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Samples: train={len(train_ds)} (x{N_AUG} aug), val={len(val_ds)}')

results = {}
for backbone in ['vit_tiny_patch16_224', 'efficientnet_b0', 'convnext_tiny']:
    print(f'\n  Training {backbone}...')
    model = create_model(num_classes=3, backbone=backbone)
    save_name = f'eyelid_ternary_{backbone}'
    auc = train_model(model, train_loader, val_loader, 3,
                      backbone[:15], save_name)
    results[backbone] = auc
    del model; torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 55)
print('  Eyelid-Only Ternary Training Complete!')
print('=' * 55)
for bb, auc in results.items():
    print(f'  {bb:30s} AUC: {auc:.4f}')

with open(os.path.join(RESULTS_DIR, 'eyelid_training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n  Models:')
for f in sorted(os.listdir(MODEL_DIR)):
    if f.startswith('eyelid_'):
        print(f'    {f} ({os.path.getsize(os.path.join(MODEL_DIR, f))//1024//1024} MB)')
