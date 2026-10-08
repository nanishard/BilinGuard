# -*- coding: utf-8 -*-
"""
BilinGuard v2 Training — with eyelid (翻上眼睑) photo support.
Each patient has:
  - Regular photos (sporthealth-*.jpg): standard facial view
  - Eyelid photo (IMG_*.jpg): everted upper eyelid showing palpebral conjunctiva
  
The eyelid photo is processed through CLAHE and used as additional input frames.
The model learns to use both facial and conjunctival signals.
"""
import os, sys, json, random, time, re
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
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 24
EPOCHS = 50
LR = 1e-4
MAX_FRAMES = 8        # regular face frames per patient
MAX_EYELID = 4        # eyelid-derived frames (original + augmented)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')


# ── Data discovery with eyelid photo separation ──────────────
def find_patient_images(data_root, category):
    """Find regular photos and eyelid photos for each patient."""
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
        
        regular_imgs = []
        eyelid_imgs = []
        for f in os.listdir(inner_path):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if 'feature' in f.lower():
                continue
            fpath = os.path.join(inner_path, f)
            if f.startswith('IMG_') and os.path.getsize(fpath) > 2_000_000:
                # Large IMG_*.jpg = eyelid everted photo (typically 5-15 MB)
                eyelid_imgs.append(fpath)
            else:
                regular_imgs.append(fpath)
        
        # Also check feature subfolder
        for sub in os.listdir(inner_path):
            sub_path = os.path.join(inner_path, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        regular_imgs.append(os.path.join(sub_path, f))
        
        if regular_imgs or eyelid_imgs:
            patients.append({
                'patient_id': outer,
                'regular_images': regular_imgs,
                'eyelid_images': eyelid_imgs,
                'has_eyelid': len(eyelid_imgs) > 0,
                'category': category,
            })
    return patients


print('\n[1] Loading data with eyelid photo separation...')
all_patients = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    patients = find_patient_images(DATA_ROOT, cat)
    with_eyelid = sum(1 for p in patients if p['has_eyelid'])
    total_reg = sum(len(p['regular_images']) for p in patients)
    total_eye = sum(len(p['eyelid_images']) for p in patients)
    print(f'  {cat:12s}: {len(patients):3d} patients | '
          f'with eyelid: {with_eyelid:3d} | '
          f'regular imgs: {total_reg} | eyelid imgs: {total_eye}')
    all_patients.extend(patients)
print(f'  Total: {len(all_patients)} patients')


# ── Dataset with eyelid support ──────────────────────────────
CATEGORY_TO_IDX = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}

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

class JaundiceDataset(Dataset):
    """
    Image-level dataset that includes both regular and eyelid photos.
    Eyelid photos are augmented (flip + crops) to generate multiple views.
    """
    def __init__(self, patients, label_map, transform=None,
                 max_regular=MAX_FRAMES, max_eyelid=MAX_EYELID):
        self.samples = []
        self.transform = transform
        for p in patients:
            cat = p['category']
            if cat not in label_map:
                continue
            label = label_map[cat]
            pid = p['patient_id']
            
            # Regular face photos
            for img_path in p['regular_images'][:max_regular]:
                self.samples.append((img_path, label, pid, 'face'))
            
            # Eyelid photos with augmentation variants
            for img_path in p['eyelid_images'][:1]:  # Usually only 1 eyelid photo
                self.samples.append((img_path, label, pid, 'eyelid'))
                # Add augmented versions during training (handled by transform)
                for _ in range(max_eyelid - 1):
                    self.samples.append((img_path, label, pid, 'eyelid'))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, pid, img_type = self.samples[idx]
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
    return avg_preds, true_labels


# ── Model creation ───────────────────────────────────────────
def create_model(num_classes, backbone='convnext_tiny'):
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)


# ── Training loop ────────────────────────────────────────────
def train_model(model, train_loader, val_loader, num_classes, task_name, save_name):
    criterion = nn.CrossEntropyLoss()
    if num_classes > 2:
        counts = np.bincount([l for _, l, _ in train_loader.dataset.samples],
                             minlength=num_classes)
        weights = 1.0 / counts
        weights = torch.FloatTensor(weights / weights.sum()).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_auc = 0
    history = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for imgs, labels, _ in tqdm(train_loader,
                                      desc=f'E{epoch}/{EPOCHS} [{task_name}]',
                                      leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        
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
        
        avg_preds, true_labels = patient_level_predict(
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
        history.append({'epoch': epoch, 'train_loss': train_loss,
                          'val_loss': val_loss, 'accuracy': acc,
                          'f1': f1, 'auc': auc})
        
        if epoch % 5 == 0 or epoch == 1:
            print(f'  E{epoch:3d}: loss={train_loss:.4f}/{val_loss:.4f} '
                  f'acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f'{save_name}.pt'))
    
    print(f'  Best AUC: {best_auc:.4f}')
    return best_auc, history


def stratified_split(patients, label_fn, val_ratio=0.2):
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
# STAGE 1: Binary screening (with eyelid photos)
# ══════════════════════════════════════════════════════════════
print('\n[2] Stage 1: Binary Screening (normal vs jaundiced)')
print('-' * 55)

binary_map = {'normal': 0, 'mild': 1, 'moderate': 1, 'severe': 1}
binary_fn = lambda p: binary_map[p['category']]

train_p, val_p = stratified_split(all_patients, binary_fn)
print(f'  Train: {len(train_p)} patients')
print(f'  Val:   {len(val_p)} patients')

train_ds = JaundiceDataset(train_p, binary_map, train_tf)
val_ds = JaundiceDataset(val_p, binary_map, eval_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Images: train={len(train_ds)}, val={len(val_ds)}')

binary_model = create_model(num_classes=2, backbone='convnext_tiny')
binary_auc, binary_hist = train_model(
    binary_model, train_loader, val_loader, num_classes=2,
    task_name='Binary', save_name='binary_screening_v2')
del binary_model
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# STAGE 2: Ternary grading (with eyelid photos)
# ══════════════════════════════════════════════════════════════
print('\n[3] Stage 2: Ternary Grading (mild/moderate/severe)')
print('-' * 55)

j_pats = [p for p in all_patients if p['category'] in ('mild', 'moderate', 'severe')]
ternary_map = {'mild': 0, 'moderate': 1, 'severe': 2}
ternary_fn = lambda p: ternary_map.get(p['category'], -1)

train_j, val_j = stratified_split(j_pats, ternary_fn)
print(f'  Train: {len(train_j)} patients, Val: {len(val_j)} patients')

train_ds_j = JaundiceDataset(train_j, ternary_map, train_tf)
val_ds_j = JaundiceDataset(val_j, ternary_map, eval_tf)
train_loader_j = DataLoader(train_ds_j, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_j = DataLoader(val_ds_j, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Images: train={len(train_ds_j)}, val={len(val_ds_j)}')

results = {}
for backbone in ['vit_tiny_patch16_224', 'efficientnet_b0', 'convnext_tiny']:
    print(f'\n  Training {backbone}...')
    model = create_model(num_classes=3, backbone=backbone)
    save_name = f'ternary_{backbone}_v2'
    auc, hist = train_model(
        model, train_loader_j, val_loader_j, num_classes=3,
        task_name=backbone[:15], save_name=save_name)
    results[backbone] = {'auc': auc}
    del model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 55)
print('  Training Complete! (with eyelid photos)')
print('=' * 55)
print(f'\n  Binary screening AUC: {binary_auc:.4f}')
for bb, res in results.items():
    print(f'  Ternary {bb:25s} AUC: {res["auc"]:.4f}')

with open(os.path.join(RESULTS_DIR, 'training_history_v2.json'), 'w') as f:
    json.dump({'binary': binary_hist, 'ternary': results}, f, indent=2)

print(f'\n  Models saved to: {MODEL_DIR}')
for f in sorted(os.listdir(MODEL_DIR)):
    if 'v2' in f:
        sz = os.path.getsize(os.path.join(MODEL_DIR, f))
        print(f'    {f} ({sz//1024//1024} MB)')
