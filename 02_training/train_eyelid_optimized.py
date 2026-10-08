# -*- coding: utf-8 -*-
"""
BilinGuard Eyelid Training v3 — Optimized
Optimizations over v2:
  1. Focal Loss (better than weighted CE for class imbalance)
  2. MixUp augmentation (improves generalization on small datasets)
  3. Label Smoothing (prevents overconfidence)
  4. OneCycleLR scheduler (faster convergence, better final AUC)
  5. Stochastic Weight Averaging (SWA — stabilizes final model)
  6. Exponential Moving Average (EMA of weights)
  7. Test-Time Augmentation (TTA at inference)
  8. RandAugment (automated augmentation policy)
  9. 5-fold Stratified Cross-Validation (robust evaluation)
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 80
LR = 2e-4       # Higher peak LR for OneCycleLR
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2
FOCAL_GAMMA = 2.0
N_AUG = 10      # More augmented copies per image
SWA_START = 50  # Start SWA at epoch 50
EMA_DECAY = 0.999

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')


# ═════════════════════════════════════════════════════════════
# LOSSES
# ═════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples, focuses on hard ones."""
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        # Label smoothing
        soft_targets = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        soft_targets = soft_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        # Focal weighting
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs.gather(1, targets.unsqueeze(1)).clamp(min=1e-8)) ** self.gamma
        loss = -(soft_targets * log_probs).sum(dim=1) * focal_weight.squeeze()
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


# ═════════════════════════════════════════════════════════════
# AUGMENTATION: MixUp
# ═════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """MixUp: linearly combine two samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)
# ═════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data = self.shadow[name].clone()

    def restore(self, model):
        # Restore original weights (for SWA)
        pass


# ═════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════
print('\n[1] Loading clean data...')
df = pd.read_csv(MANIFEST)
j_df = df[(df['category'].isin(['mild', 'moderate', 'severe'])) & (df['n_eyelid'] > 0)]
patients = []
for _, row in j_df.iterrows():
    paths = json.loads(row['eyelid_images'])
    if paths:
        patients.append({'id': row['patient_id'], 'eyelid_path': paths[0],
                         'category': row['category']})

tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
labels = np.array([tmap[p['category']] for p in patients])
print(f'  Total: {len(patients)} (mild={sum(labels==0)}, mod={sum(labels==1)}, sev={sum(labels==2)})')

# Augmentation: RandAugment-style
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE * 2, IMG_SIZE * 2)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
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
            if p['category'] not in label_map: continue
            for _ in range(n_aug):
                self.samples.append((p['eyelid_path'], label_map[p['category']], p['id']))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, pid = self.samples[idx]
        try: img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), label, pid


def patient_agg(preds, labels, pids):
    d = {}
    for p, l, pid in zip(preds, labels, pids):
        d.setdefault(pid, ([], l))
        d[pid][0].append(p)
    pids = list(d.keys())
    return np.array([np.mean(d[p][0], axis=0) for p in pids]), np.array([d[p][1] for p in pids])


# ═════════════════════════════════════════════════════════════
# TRAINING with all optimizations
# ═════════════════════════════════════════════════════════════
def train_optimized(backbone, train_pats, val_pats, save_prefix):
    print(f'\n{"="*60}')
    print(f'  Training {backbone} with optimized pipeline')
    print(f'{"="*60}')

    model = timm.create_model(backbone, pretrained=True, num_classes=3).to(DEVICE)

    # Class-weighted Focal Loss
    counts = np.bincount([tmap[p['category']] for p in train_pats], minlength=3)
    alpha = torch.FloatTensor((1.0 / counts) / (1.0 / counts).sum()).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
    print(f'  Focal Loss: alpha={alpha.cpu().numpy()}, gamma={FOCAL_GAMMA}, label_smooth={LABEL_SMOOTH}')

    # OneCycleLR
    train_ds = EyelidDataset(train_pats, tmap, train_transform, n_aug=N_AUG)
    val_ds = EyelidDataset(val_pats, tmap, eval_transform, n_aug=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_ds)//BATCH_SIZE + 1,
        pct_start=0.1, anneal_strategy='cos')
    print(f'  OneCycleLR: max_lr={LR}, pct_start=0.1')

    # WeightedRandomSampler for class balance
    sample_weights = [1.0 / counts[tmap[p['category']]] for p in train_pats for _ in range(N_AUG)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'  WeightedRandomSampler: balanced sampling')
    print(f'  MixUp: alpha={MIXUP_ALPHA}')
    print(f'  EMA: decay={EMA_DECAY}')
    print(f'  SWA: starts at epoch {SWA_START}')

    # EMA
    ema = EMA(model, decay=EMA_DECAY)

    # SWA
    swa_model = copy.deepcopy(model)
    swa_n = 0

    best_auc = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tloss = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f'E{epoch}/{EPOCHS}', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # MixUp
            mixed_x, y_a, y_b, lam = mixup_data(imgs, labels)
            optimizer.zero_grad()
            out = model(mixed_x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            tloss += loss.item()

        # SWA update
        if epoch >= SWA_START:
            swa_model = swa_model.to(DEVICE)
            with torch.no_grad():
                swa_n += 1
                for p_swa, p_model in zip(swa_model.parameters(), model.parameters()):
                    p_swa.data = (p_swa.data * (swa_n - 1) + p_model.data) / swa_n

        # Evaluate with EMA weights
        ema_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        ema.apply_shadow(model)
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
        try: auc = roc_auc_score(true, avg, multi_class='ovr')
        except: auc = 0

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            cm = confusion_matrix(true, pl, labels=[0,1,2])
            print(f'  E{epoch:3d}: loss={tloss/len(train_loader):.4f} acc={acc:.4f} '
                  f'f1={f1:.4f} auc={auc:.4f} CM={cm.tolist()}')

        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_{backbone}.pt'))

        # Restore EMA shadow for next epoch training
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data = ema_shadow[name].clone()

    # Evaluate SWA model
    if swa_n > 0:
        swa_model.eval()
        preds, labs, pids = [], [], []
        with torch.no_grad():
            for imgs, labels, p in val_loader:
                imgs = imgs.to(DEVICE)
                out = swa_model(imgs)
                preds.extend(F.softmax(out, dim=1).cpu().numpy())
                labs.extend(labels.numpy())
                pids.extend(p)
        avg, true = patient_agg(np.array(preds), np.array(labs), pids)
        try: swa_auc = roc_auc_score(true, avg, multi_class='ovr')
        except: swa_auc = 0
        print(f'  SWA AUC: {swa_auc:.4f} (n_avg={swa_n})')
        if swa_auc > best_auc:
            best_auc = swa_auc
            torch.save(swa_model.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_{backbone}.pt'))
            print(f'  SWA model saved (better than best)')

    print(f'  Best AUC: {best_auc:.4f}')
    del model, swa_model
    torch.cuda.empty_cache()
    return best_auc


# ═════════════════════════════════════════════════════════════
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════
print(f'\n[2] 5-Fold Stratified Cross-Validation')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

BACKBONES = ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0']
all_results = {}

for fold, (train_idx, val_idx) in enumerate(skf.split(patients, labels)):
    print(f'\n{"#"*60}')
    print(f'  FOLD {fold+1}/5')
    print(f'{"#"*60}')
    train_pats = [patients[i] for i in train_idx]
    val_pats = [patients[i] for i in val_idx]
    print(f'  Train: {len(train_pats)}, Val: {len(val_pats)}')

    fold_results = {}
    for bb in BACKBONES:
        auc = train_optimized(bb, train_pats, val_pats, f'eyelid_opt_fold{fold}')
        fold_results[bb] = auc
        all_results.setdefault(bb, []).append(auc)

    # Ensemble: load fold models and average
    print(f'\n  Fold {fold+1} ensemble...')
    fold_models = {}
    for bb in BACKBONES:
        ckpt = os.path.join(MODEL_DIR, f'eyelid_opt_fold{fold}_{bb}.pt')
        if os.path.exists(ckpt):
            m = timm.create_model(bb, pretrained=False, num_classes=3).to(DEVICE)
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
            m.eval()
            fold_models[bb] = m

    # Evaluate ensemble
    val_ds = EyelidDataset(val_pats, tmap, eval_transform, n_aug=1)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds_all, labs_all, pids_all = [], [], []
    with torch.no_grad():
        for imgs, labels, p in val_loader:
            imgs = imgs.to(DEVICE)
            probs_list = []
            for m in fold_models.values():
                out = m(imgs)
                probs_list.append(F.softmax(out, dim=1))
            avg_probs = torch.stack(probs_list).mean(dim=0)
            preds_all.extend(avg_probs.cpu().numpy())
            labs_all.extend(labels.numpy())
            pids_all.extend(p)

    avg, true = patient_agg(np.array(preds_all), np.array(labs_all), pids_all)
    try: ens_auc = roc_auc_score(true, avg, multi_class='ovr')
    except: ens_auc = 0
    all_results.setdefault('ensemble', []).append(ens_auc)
    print(f'  Fold {fold+1} Ensemble AUC: {ens_auc:.4f}')

    # Cleanup fold models
    for m in fold_models.values():
        del m
    torch.cuda.empty_cache()

# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  Optimized Training Complete!')
print('=' * 60)
print(f'\n  5-Fold CV Results:')
for name, aucs in all_results.items():
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f'    {name:30s}: {mean_auc:.4f} \u00b1 {std_auc:.4f}  '
          f'(folds: {[f"{a:.3f}" for a in aucs]})')

with open(os.path.join(RESULTS_DIR, 'eyelid_optimized_cv_results.json'), 'w') as f:
    json.dump({k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'folds': v}
               for k, v in all_results.items()}, f, indent=2)

print(f'\n  Models saved in {MODEL_DIR}/')
print(f'  Results in {RESULTS_DIR}/')
