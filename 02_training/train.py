# -*- coding: utf-8 -*-
"""
BilinGuard Training Script — works directly with extracted patient folders.
  Stage 1: Binary screening (normal vs jaundiced)
  Stage 2: Ternary grading (mild / moderate / severe)
"""
import os, sys, re, random, time, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix)
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
N_FRAMES = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Data discovery ────────────────────────────────────────────
def find_patient_images(data_root, category):
    """Find all JPG images for patients in a category folder."""
    cat_dir = os.path.join(data_root, category)
    if not os.path.exists(cat_dir):
        return []
    patients = []
    for outer in os.listdir(cat_dir):
        outer_path = os.path.join(cat_dir, outer)
        if not os.path.isdir(outer_path):
            continue
        # Find inner folder
        inner_dirs = [d for d in os.listdir(outer_path)
                      if os.path.isdir(os.path.join(outer_path, d))]
        if inner_dirs:
            inner_path = os.path.join(outer_path, inner_dirs[0])
        else:
            inner_path = outer_path
        # Collect JPG images (exclude feature subfolder images)
        imgs = []
        for f in os.listdir(inner_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 'feature' not in f.lower():
                imgs.append(os.path.join(inner_path, f))
        # Also check feature subfolder for additional images
        for sub in os.listdir(inner_path):
            sub_path = os.path.join(inner_path, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        imgs.append(os.path.join(sub_path, f))
        if imgs:
            patients.append({'patient_id': outer, 'images': imgs, 'category': category})
    return patients

print('\n[1] Loading data...')
all_patients = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    patients = find_patient_images(DATA_ROOT, cat)
    print(f'  {cat:12s}: {len(patients)} patients, {sum(len(p["images"]) for p in patients)} images')
    all_patients.extend(patients)
print(f'  Total: {len(all_patients)} patients')

# ── Dataset ───────────────────────────────────────────────────
CATEGORY_TO_IDX = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_IDX.items()}

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.03),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class JaundiceDataset(Dataset):
    """Image-level dataset for jaundice classification."""
    def __init__(self, patients, label_map, transform=None, max_images=12):
        self.samples = []
        self.transform = transform
        for p in patients:
            cat = p['category']
            if cat not in label_map:
                continue
            label = label_map[cat]
            imgs = p['images'][:max_images]
            for img_path in imgs:
                self.samples.append((img_path, label, p['patient_id']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, pid = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label, pid

def patient_level_predict(image_preds, image_labels, image_pids):
    """Aggregate image-level predictions to patient-level."""
    pid_scores = {}
    pid_labels = {}
    for pred, label, pid in zip(image_preds, image_labels, image_pids):
        if pid not in pid_scores:
            pid_scores[pid] = []
            pid_labels[pid] = label
        pid_scores[pid].append(pred)
    pids = list(pid_scores.keys())
    avg_preds = np.array([np.mean(pid_scores[p], axis=0) for p in pids])
    true_labels = np.array([pid_labels[p] for p in pids])
    return avg_preds, true_labels, pids

# ── Model ─────────────────────────────────────────────────────
def create_model(num_classes, backbone='convnext_tiny'):
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)

# ── Training ──────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, num_classes, task_name, save_name):
    criterion = nn.CrossEntropyLoss()
    if num_classes > 2:
        class_counts = np.bincount([label for _, label, _ in train_loader.dataset.samples],
                                     minlength=num_classes)
        weights = 1.0 / class_counts
        weights = torch.FloatTensor(weights / weights.sum()).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [{task_name}]',
                                      leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        all_preds, all_labels, all_pids = [], [], []
        with torch.no_grad():
            for imgs, labels, pids in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                probs = F.softmax(out, dim=1)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_pids.extend(pids)

        # Patient-level metrics
        avg_preds, true_labels, _ = patient_level_predict(
            np.array(all_preds), np.array(all_labels), all_pids)
        pred_labels = avg_preds.argmax(axis=1)
        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro')
        try:
            if num_classes == 2:
                auc = roc_auc_score(true_labels, avg_preds[:, 1])
            else:
                auc = roc_auc_score(true_labels, avg_preds, multi_class='ovr')
        except Exception:
            auc = 0

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                          'accuracy': acc, 'f1': f1, 'auc': auc})

        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
                  f'acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}')

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_name}.pt'))

    print(f'  Best AUC: {best_auc:.4f}')
    return best_auc, history

# ── Split ─────────────────────────────────────────────────────
def stratified_split(patients, label_fn, val_ratio=0.2, seed=SEED):
    """Patient-level stratified split."""
    random.seed(seed)
    by_label = {}
    for p in patients:
        label = label_fn(p)
        by_label.setdefault(label, []).append(p)
    train, val = [], []
    for label, group in by_label.items():
        random.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val

# ══════════════════════════════════════════════════════════════
# STAGE 1: Binary screening
# ══════════════════════════════════════════════════════════════
print('\n[2] Stage 1: Binary Screening (normal vs jaundiced)')
print('-' * 50)

binary_label_map = {'normal': 0, 'mild': 1, 'moderate': 1, 'severe': 1}
binary_label_fn = lambda p: binary_label_map[p['category']]

train_pats, val_pats = stratified_split(all_patients, binary_label_fn)
print(f'  Train: {len(train_pats)} patients')
print(f'  Val:   {len(val_pats)} patients')

train_ds = JaundiceDataset(train_pats, binary_label_map, train_transform)
val_ds = JaundiceDataset(val_pats, binary_label_map, eval_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Train images: {len(train_ds)}, Val images: {len(val_ds)}')

binary_model = create_model(num_classes=2, backbone='convnext_tiny')
binary_auc, binary_hist = train_model(
    binary_model, train_loader, val_loader, num_classes=2,
    task_name='Binary', save_name='binary_screening')

# ══════════════════════════════════════════════════════════════
# STAGE 2: Ternary grading
# ══════════════════════════════════════════════════════════════
print('\n[3] Stage 2: Ternary Grading (mild vs moderate vs severe)')
print('-' * 50)

jaundice_patients = [p for p in all_patients if p['category'] in ('mild', 'moderate', 'severe')]
ternary_label_map = {'mild': 0, 'moderate': 1, 'severe': 2}
ternary_label_fn = lambda p: ternary_label_map.get(p['category'], -1)

train_j, val_j = stratified_split(jaundice_patients, ternary_label_fn)
print(f'  Train: {len(train_j)} jaundice patients')
print(f'  Val:   {len(val_j)} jaundice patients')

train_ds_j = JaundiceDataset(train_j, ternary_label_map, train_transform)
val_ds_j = JaundiceDataset(val_j, ternary_label_map, eval_transform)
train_loader_j = DataLoader(train_ds_j, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_j = DataLoader(val_ds_j, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Train images: {len(train_ds_j)}, Val images: {len(val_ds_j)}')

ternary_model = create_model(num_classes=3, backbone='convnext_tiny')
ternary_auc, ternary_hist = train_model(
    ternary_model, train_loader_j, val_loader_j, num_classes=3,
    task_name='Ternary', save_name='ternary_grading')

# ══════════════════════════════════════════════════════════════
# Also train additional backbones for ensemble
# ══════════════════════════════════════════════════════════════
print('\n[4] Training additional backbones for ternary ensemble')
print('-' * 50)

backbone_results = {}
for backbone in ['vit_tiny_patch16_224', 'efficientnet_b0']:
    print(f'\n  Training {backbone}...')
    model = create_model(num_classes=3, backbone=backbone)
    save_name = f'ternary_{backbone}'
    auc, hist = train_model(
        model, train_loader_j, val_loader_j, num_classes=3,
        task_name=backbone[:15], save_name=save_name)
    backbone_results[backbone] = {'auc': auc}
    del model
    torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Training Complete!')
print('=' * 60)
print(f'\n  Binary screening AUC:  {binary_auc:.4f}')
print(f'  Ternary grading AUC:   {ternary_auc:.4f}')
for bb, res in backbone_results.items():
    print(f'  {bb} AUC:              {res["auc"]:.4f}')

# Save training history
history = {
    'binary': binary_hist,
    'ternary_convnext': ternary_hist,
}
with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)
print(f'\n  Models saved to: {MODEL_DIR}')
print(f'  History saved to: {RESULTS_DIR}/training_history.json')
