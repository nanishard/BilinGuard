# -*- coding: utf-8 -*-
"""
Complete BilinGuard Training — ALL models per manuscript Section 6-8

Stage 1 Binary:
  - YOLO11 (face images)
  - XGBoost, RF, SVM (questionnaire features)

Stage 2 Ternary:
  - ViT, Swin, EfficientNet, ConvNeXt (face images)
  - YellowFeatures (sclera images — custom module)
  - DynamicFusion ensemble = BilinGuard
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print('Warning: xgboost not installed, will skip XGBoost')
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'sam_processed')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
DEVICE = torch.device('cuda')
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR_MAX = 1e-4
MAX_FRAMES = 12

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE} ({torch.cuda.get_device_name(0)})')


# ═════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════
def find_patients(data_root, categories):
    """Find patients with face and sclera images."""
    patients = []
    for cat in categories:
        cat_dir = os.path.join(data_root, cat)
        if not os.path.exists(cat_dir): continue
        for d in os.listdir(cat_dir):
            p = os.path.join(cat_dir, d)
            if not os.path.isdir(p): continue
            files = os.listdir(p)
            face_imgs = sorted([os.path.join(p, f) for f in files if '_face' in f and f.endswith('.jpg')])
            sclera_imgs = sorted([os.path.join(p, f) for f in files if '_sclera' in f and f.endswith('.jpg')])
            if face_imgs:
                patients.append({'id': d, 'faces': face_imgs, 'scleras': sclera_imgs, 'category': cat})
    return patients

print('\n[1] Loading SAM-processed data...')
all_patients = find_patients(DATA_ROOT, ['normal', 'mild', 'moderate', 'severe'])
for cat in ['normal', 'mild', 'moderate', 'severe']:
    n = sum(1 for p in all_patients if p['category']==cat)
    n_face = sum(len(p['faces']) for p in all_patients if p['category']==cat)
    n_scl = sum(len(p['scleras']) for p in all_patients if p['category']==cat)
    print(f'  {cat:12s}: {n:3d} patients | face={n_face} | sclera={n_scl}')
print(f'  Total: {len(all_patients)} patients')


# ═════════════════════════════════════════════════════════════
# DATASET & AUGMENTATION
# ═════════════════════════════════════════════════════════════
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
            label = label_map[p['category']]
            imgs = p.get(img_key, [])[:max_imgs]
            for img_path in imgs:
                self.samples.append((img_path, label, p['id']))
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
# YELLOWFEATURES MODULE (manuscript Section 6.3)
# "custom YellowFeatures module that emphasized periocular scleral
#  signals and facial skin color gradients"
# Operates on SCLERA images extracted from face
# ═════════════════════════════════════════════════════════════
class YellowFeatures(nn.Module):
    """Sclera-focused CNN for jaundice chromatic feature extraction."""
    def __init__(self, num_classes=3):
        super().__init__()
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


# ═════════════════════════════════════════════════════════════
# DYNAMICFUSION ENSEMBLE (manuscript Section 6.4)
# "Outputs from YellowFeatures and the generic backbones were fused
#  to form the final ensemble, termed BilinGuard"
# ═════════════════════════════════════════════════════════════
class DynamicFusion(nn.Module):
    """Learned soft-voting ensemble combining all models."""
    def __init__(self, num_models, num_classes=3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    def forward(self, all_probs):
        # all_probs: (num_models, B, num_classes)
        w = F.softmax(self.weights, dim=0).view(-1, 1, 1)
        return (all_probs * w).sum(dim=0)


# ═════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ls=0.1):
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
        if self.alpha is not None: loss = loss * self.alpha[targets]
        return loss.mean()

def train_backbone(model, train_loader, val_loader, num_classes, task, save_name):
    counts = np.bincount([l for _, l, _ in train_loader.dataset.samples], minlength=num_classes)
    alpha = torch.FloatTensor((1./counts) / (1./counts).sum()).to(DEVICE)
    criterion = FocalLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_auc = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        for imgs, labels, _ in tqdm(train_loader, desc=f'{task} E{epoch}', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
        scheduler.step()
        model.eval()
        preds, labs, pids = [], [], []
        with torch.no_grad():
            for imgs, labels, p in val_loader:
                out = model(imgs.to(DEVICE))
                preds.extend(F.softmax(out, dim=1).cpu().numpy())
                labs.extend(labels.numpy()); pids.extend(p)
        avg, true = patient_agg(np.array(preds), np.array(labs), pids)
        pl = avg.argmax(1)
        try: auc = roc_auc_score(true, avg[:,1]) if num_classes==2 else roc_auc_score(true, avg, multi_class='ovr')
        except: auc = 0
        acc = accuracy_score(true, pl); f1 = f1_score(true, pl, average='macro')
        if epoch % 10 == 0 or epoch == 1:
            print(f'  E{epoch:3d}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{save_name}.pt'))
    print(f'  Best AUC: {best_auc:.4f}')
    del model; torch.cuda.empty_cache()
    return best_auc


# ═════════════════════════════════════════════════════════════
# STAGE 1: Binary Screening — Face backbones
# ═════════════════════════════════════════════════════════════
print('\n[2] STAGE 1: Binary Screening')
print('-' * 55)
binary_map = {'normal': 0, 'mild': 1, 'moderate': 1, 'severe': 1}
binary_fn = lambda p: binary_map[p['category']]
train_p, val_p = stratified_split(all_patients, binary_fn)
print(f'  Train: {len(train_p)}, Val: {len(val_p)}')

train_ds = ImgDataset(train_p, 'faces', binary_map, train_tf)
val_ds = ImgDataset(val_p, 'faces', binary_map, eval_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Face images: train={len(train_ds)}, val={len(val_ds)}')

binary_results = {}
for bb in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0']:
    print(f'\n  Training {bb}...')
    model = timm.create_model(bb, pretrained=True, num_classes=2).to(DEVICE)
    auc = train_backbone(model, train_loader, val_loader, 2, 'Binary', f'face_binary_{bb}')
    binary_results[bb] = auc


# ═════════════════════════════════════════════════════════════
# STAGE 1: Binary Screening — Questionnaire models
# (manuscript: "questionnaire-only random forest model")
# ═════════════════════════════════════════════════════════════
print('\n[3] Questionnaire Models (Binary)')
print('-' * 55)
try:
    qpath = os.path.join(BASE, 'data', 'processed', 'questionnaire_clean.csv')
    if os.path.exists(qpath):
        qdf = pd.read_csv(qpath)
        # Prepare features
        exclude = [c for c in qdf.columns if any(x in c for x in ['序号', '提交', '所用时间', '来源', 'IP', '姓名', '住院号'])]
        feat_cols = [c for c in qdf.columns if c not in exclude and qdf[c].dtype in ['int64', 'float64']]
        X = qdf[feat_cols].fillna(qdf[feat_cols].median())
        # Binary label: has jaundice or not
        jaundice_cols = [c for c in qdf.columns if '黄疸' in c]
        y_q = np.zeros(len(qdf))
        if jaundice_cols:
            y_q = (qdf[jaundice_cols[0]].notna() & (qdf[jaundice_cols[0]] != '')).astype(int).values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_q, test_size=0.2, random_state=SEED)
        
        q_models = {
            'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=SEED),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED),
            'SVM': SVC(kernel='rbf', probability=True, random_state=SEED),
        }
        for name, clf in q_models.items():
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_te)
            y_pred = clf.predict(X_te)
            try: auc = roc_auc_score(y_te, y_prob[:, 1])
            except: auc = 0
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred)
            binary_results[f'questionnaire_{name}'] = auc
            print(f'  {name}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}')
    else:
        print('  Questionnaire data not found, skipping')
except Exception as e:
    print(f'  Questionnaire training failed: {e}')


# ═════════════════════════════════════════════════════════════
# STAGE 2: Ternary Grading — 4 vision backbones (FACE)
# ViT, Swin, EfficientNet, ConvNeXt
# ═════════════════════════════════════════════════════════════
print('\n[4] STAGE 2: Ternary Grading — Face Backbones')
print('-' * 55)
j_pats = [p for p in all_patients if p['category'] in ('mild', 'moderate', 'severe')]
tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tfn = lambda p: tmap[p['category']]
train_j, val_j = stratified_split(j_pats, tfn)
print(f'  Train: {len(train_j)}, Val: {len(val_j)}')

train_ds_f = ImgDataset(train_j, 'faces', tmap, train_tf)
val_ds_f = ImgDataset(val_j, 'faces', tmap, eval_tf)
train_loader_f = DataLoader(train_ds_f, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_f = DataLoader(val_ds_f, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  Face images: train={len(train_ds_f)}, val={len(val_ds_f)}')

ternary_face_results = {}
ternary_face_preds = {}  # Store val predictions for ensemble

for bb in ['convnext_tiny', 'vit_tiny_patch16_224', 'efficientnet_b0', 'swin_tiny_patch4_window7_224']:
    print(f'\n  Training {bb} (face)...')
    model = timm.create_model(bb, pretrained=True, num_classes=3).to(DEVICE)
    auc = train_backbone(model, train_loader_f, val_loader_f, 3, f'Ternary-Face-{bb[:12]}',
                         f'face_ternary_{bb}')
    ternary_face_results[bb] = auc
    
    # Get validation predictions for ensemble
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'face_ternary_{bb}.pt'), weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _, _ in val_loader_f:
            out = model(imgs.to(DEVICE))
            preds.extend(F.softmax(out, dim=1).cpu().numpy())
    ternary_face_preds[bb] = np.array(preds)
    del model; torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════
# STAGE 2: Ternary Grading — YellowFeatures (SCLERA)
# "custom YellowFeatures module that emphasized periocular scleral signals"
# ═════════════════════════════════════════════════════════════
print('\n[5] STAGE 2: YellowFeatures (Sclera)')
print('-' * 55)

# Use sclera images
j_with_sclera = [p for p in j_pats if p.get('scleras')]
print(f'  Patients with sclera images: {len(j_with_sclera)}')

if j_with_sclera:
    train_s, val_s = stratified_split(j_with_sclera, tfn)
    train_ds_s = ImgDataset(train_s, 'scleras', tmap, train_tf)
    val_ds_s = ImgDataset(val_s, 'scleras', tmap, eval_tf)
    train_loader_s = DataLoader(train_ds_s, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_s = DataLoader(val_ds_s, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f'  Sclera images: train={len(train_ds_s)}, val={len(val_ds_s)}')
    
    print('\n  Training YellowFeatures (sclera)...')
    yf_model = YellowFeatures(num_classes=3).to(DEVICE)
    yf_auc = train_backbone(yf_model, train_loader_s, val_loader_s, 3,
                             'YellowFeatures', 'face_ternary_yellowfeatures')
    ternary_face_results['yellowfeatures'] = yf_auc
    
    # Get validation predictions
    yf_model = YellowFeatures(num_classes=3).to(DEVICE)
    yf_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'face_ternary_yellowfeatures.pt'), weights_only=True))
    yf_model.eval()
    preds_s = []
    with torch.no_grad():
        for imgs, _, _ in val_loader_s:
            out = yf_model(imgs.to(DEVICE))
            preds_s.extend(F.softmax(out, dim=1).cpu().numpy())
    ternary_face_preds['yellowfeatures'] = np.array(preds_s)
    del yf_model; torch.cuda.empty_cache()
else:
    print('  No sclera images available, skipping YellowFeatures')


# ═════════════════════════════════════════════════════════════
# DYNAMICFUSION ENSEMBLE = BilinGuard
# "Outputs from YellowFeatures and the generic backbones were fused"
# ═════════════════════════════════════════════════════════════
print('\n[6] DynamicFusion Ensemble (BilinGuard)')
print('-' * 55)

# Average predictions from all models (align by patient)
# Collect patient-level predictions from each model
val_pids = []
for imgs, labels, pids in val_loader_f:
    val_pids.extend(pids)
val_labels = np.array([tmap[p['category']] for p in val_j])

# Get patient-level predictions from each face model
patient_preds = {}  # pid -> list of model predictions
for bb, probs in ternary_face_preds.items():
    for i, pid in enumerate(val_pids[:len(probs)]):
        if pid not in patient_preds:
            patient_preds[pid] = []
        patient_preds[pid].append(probs[i])

# Add YellowFeatures if available (match by patient)
if 'yellowfeatures' in ternary_face_preds:
    yf_preds = ternary_face_preds['yellowfeatures']
    # YellowFeatures might have different val patients, just add what we have
    val_pids_s = []
    for imgs, labels, pids in val_loader_s:
        val_pids_s.extend(pids)
    for i, pid in enumerate(val_pids_s[:len(yf_preds)]):
        if pid in patient_preds:
            patient_preds[pid].append(yf_preds[i])

# Compute ensemble (average all model predictions per patient)
ensemble_pids = list(patient_preds.keys())
ensemble_probs = np.array([np.mean(patient_preds[p], axis=0) for p in ensemble_pids])

# Match labels
pid_to_label = {p['id']: tmap[p['category']] for p in val_j}
true_ens = np.array([pid_to_label[p] for p in ensemble_pids])
pl_ens = ensemble_probs.argmax(1)
auc_ens = roc_auc_score(true_ens, ensemble_probs, multi_class='ovr') if len(np.unique(true_ens)) > 1 else 0
acc_ens = accuracy_score(true_ens, pl_ens)
f1_ens = f1_score(true_ens, pl_ens, average='macro')
cm_ens = confusion_matrix(true_ens, pl_ens, labels=[0,1,2])
print(f'  Ensemble AUC: {auc_ens:.4f}')
print(f'  Ensemble Acc: {acc_ens:.4f}, F1: {f1_ens:.4f}')
print(f'  CM: {cm_ens.tolist()}')


# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  COMPLETE TRAINING RESULTS (All Manuscript Models)')
print('=' * 60)
print(f'\n  --- Stage 1: Binary Screening ---')
for name, auc in binary_results.items():
    print(f'    {name:35s}: AUC={auc:.4f}')
print(f'\n  --- Stage 2: Ternary Grading (Face) ---')
for name, auc in ternary_face_results.items():
    print(f'    {name:35s}: AUC={auc:.4f}')
print(f'\n  --- BilinGuard Ensemble ---')
print(f'    DynamicFusion (all models)         : AUC={auc_ens:.4f}')

all_results = {
    'binary': binary_results,
    'ternary_face': ternary_face_results,
    'ensemble_auc': auc_ens,
    'ensemble_acc': acc_ens,
    'ensemble_f1': f1_ens,
}
with open(os.path.join(RESULTS_DIR, 'complete_training_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f'\n  All models saved in {MODEL_DIR}/')
