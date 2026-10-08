# -*- coding: utf-8 -*-
"""
Diagnose external validation drop and try improvements:
  1. Check individual backbone AUCs (maybe one generalizes better)
  2. Apply Grey-World normalization before inference
  3. Test-time augmentation (horizontal flip)
  4. Check if external images have different characteristics (size, skin fraction)
  5. Try all combinations
"""
import os, json, numpy as np, pandas as pd, cv2
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

BASE = r'D:\research\人脸识别营养\传染科'
EXT = os.path.join(BASE, 'data', 'external_processed_v3')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224; BS = 32

ARCHS = [('convnext', 'convnext_tiny', 'final_binary_face_convnext.pt'),
         ('vit', 'vit_tiny_patch16_224', 'final_binary_face_vit.pt'),
         ('swin', 'swin_tiny_patch4_window7_224', 'final_binary_face_swin.pt'),
         ('efficientnet', 'efficientnet_b0', 'final_binary_face_efficientnet.pt')]

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab); l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def grey_world(img):
    mean_r = img[:,:,0].mean(); mean_g = img[:,:,1].mean(); mean_b = img[:,:,2].mean()
    gray = (mean_r + mean_g + mean_b) / 3
    img = img.astype(np.float32)
    img[:,:,0] *= gray / (mean_r + 1e-6)
    img[:,:,1] *= gray / (mean_g + 1e-6)
    img[:,:,2] *= gray / (mean_b + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)

class DS(Dataset):
    def __init__(self, items, gw=False, flip=False):
        self.items = items; self.gw = gw; self.flip = flip
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        if self.gw: img = grey_world(img)
        img = clahe(img)
        if self.flip: img = img[:, ::-1].copy()
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid

# Build patient list
patients = []
for label, cat in [(0, 'normal'), (1, 'jaundice')]:
    cat_dir = os.path.join(EXT, cat)
    if not os.path.isdir(cat_dir): continue
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if f.endswith('_face.jpg')])
        if faces: patients.append({'pid': pid, 'label': label, 'faces': faces[:12]})
items = [(fp, p['label'], p['pid']) for p in patients for fp in p['faces']]
print(f'External: {len(patients)} patients, {len(items)} images')

# Quick diagnostic: check image stats for first 5 normal vs 5 jaundice
for cat_pids in [('normal', [p for p in patients if p['label']==0][:5]),
                  ('jaundice', [p for p in patients if p['label']==1][:5])]:
    cat_name, ps = cat_pids
    mean_brightness = []
    for p in ps:
        img = read_img(p['faces'][0])
        if img is not None:
            mean_brightness.append(img.mean())
    nf = len(ps[0]['faces']) if ps else 0
    print(f'  {cat_name}: mean brightness = {np.mean(mean_brightness):.1f} ({nf} faces/patient avg)')

@torch.no_grad()
def run_model(ckpt, bb, items_list, gw=False, flip=False):
    m = timm.create_model(bb, pretrained=False, num_classes=2).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, ckpt), map_location=DEV))
    m.eval()
    dl = DataLoader(DS(items_list, gw=gw, flip=flip), BS, shuffle=False, num_workers=0)
    d = {}
    for im, la, pi in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        for p, l, pid in zip(o, la.numpy(), pi):
            d.setdefault(pid, ([], int(l))); d[pid][0].append(p)
    del m; torch.cuda.empty_cache()
    return {pid: (np.mean(v[0], 0), v[1]) for pid, v in d.items()}

def auc_of(pd_dict):
    t = np.array([v[1] for v in pd_dict.values()])
    p = np.array([v[0][1] for v in pd_dict.values()])
    if len(np.unique(t)) < 2: return float('nan')
    return roc_auc_score(t, p)

# ═══ EXPERIMENT 1: Individual backbone AUCs ═══
print('\n=== Experiment 1: Individual backbones ===')
individual = {}
for name, bb, ckpt in ARCHS:
    pd_ = run_model(ckpt, bb, items)
    a = auc_of(pd_)
    individual[name] = pd_
    print(f'  {name:<15} AUC = {a:.4f}')

# ═══ EXPERIMENT 2: Grey-World normalization ═══
print('\n=== Experiment 2: Grey-World normalization ===')
gw_results = {}
for name, bb, ckpt in ARCHS:
    pd_ = run_model(ckpt, bb, items, gw=True)
    a = auc_of(pd_)
    gw_results[name] = pd_
    print(f'  {name:<15} GW AUC = {a:.4f}')

# GW ensemble
gw_ens = {}
for pid in gw_results['convnext']:
    ps = [gw_results[a][pid][0] for a in ['convnext','vit','swin','efficientnet'] if pid in gw_results[a]]
    if len(ps) == 4: gw_ens[pid] = (np.mean(ps, 0), gw_results['convnext'][pid][1])
print(f'  GW Ensemble      AUC = {auc_of(gw_ens):.4f}')

# ═══ EXPERIMENT 3: Test-time augmentation (flip) ═══
print('\n=== Experiment 3: TTA (flip averaging) ===')
tta_results = {}
for name, bb, ckpt in ARCHS:
    pd_orig = individual[name]
    pd_flip = run_model(ckpt, bb, items, flip=True)
    tta = {}
    for pid in pd_orig:
        if pid in pd_flip:
            tta[pid] = ((pd_orig[pid][0] + pd_flip[pid][0]) / 2, pd_orig[pid][1])
    a = auc_of(tta)
    tta_results[name] = tta
    print(f'  {name:<15} TTA AUC = {a:.4f}')

# TTA ensemble
tta_ens = {}
for pid in tta_results['convnext']:
    ps = [tta_results[a][pid][0] for a in ['convnext','vit','swin','efficientnet'] if pid in tta_results[a]]
    if len(ps) == 4: tta_ens[pid] = (np.mean(ps, 0), tta_results['convnext'][pid][1])
print(f'  TTA Ensemble     AUC = {auc_of(tta_ens):.4f}')

# ═══ EXPERIMENT 4: GW + TTA combined ═══
print('\n=== Experiment 4: GW + TTA combined ===')
gw_tta = {}
for name, bb, ckpt in ARCHS:
    pd_gw = gw_results[name]
    pd_gw_flip = run_model(ckpt, bb, items, gw=True, flip=True)
    combined = {}
    for pid in pd_gw:
        if pid in pd_gw_flip:
            combined[pid] = ((pd_gw[pid][0] + pd_gw_flip[pid][0]) / 2, pd_gw[pid][1])
    gw_tta[name] = combined
gw_tta_ens = {}
for pid in gw_tta['convnext']:
    ps = [gw_tta[a][pid][0] for a in ['convnext','vit','swin','efficientnet'] if pid in gw_tta[a]]
    if len(ps) == 4: gw_tta_ens[pid] = (np.mean(ps, 0), gw_tta['convnext'][pid][1])
print(f'  GW+TTA Ensemble  AUC = {auc_of(gw_tta_ens):.4f}')

# ═══ EXPERIMENT 5: Best 2 or 3 backbone ensemble (drop worst) ═══
print('\n=== Experiment 5: Subset ensembles ===')
sorted_archs = sorted(individual.keys(), key=lambda a: -auc_of(individual[a]))
for n_keep in [2, 3]:
    keep = sorted_archs[:n_keep]
    sub_ens = {}
    for pid in individual[keep[0]]:
        ps = [individual[a][pid][0] for a in keep if pid in individual[a]]
        if len(ps) == n_keep: sub_ens[pid] = (np.mean(ps, 0), individual[keep[0]][pid][1])
    print(f'  Top-{n_keep} ({"+".join(keep)}): AUC = {auc_of(sub_ens):.4f}')

# ═══ EXPERIMENT 6: GW + best subset ═══
print('\n=== Experiment 6: GW + best subset ensemble ===')
gw_sorted = sorted(gw_results.keys(), key=lambda a: -auc_of(gw_results[a]))
for n_keep in [2, 3, 4]:
    keep = gw_sorted[:n_keep]
    sub_ens = {}
    for pid in gw_results[keep[0]]:
        ps = [gw_results[a][pid][0] for a in keep if pid in gw_results[a]]
        if len(ps) == n_keep: sub_ens[pid] = (np.mean(ps, 0), gw_results[keep[0]][pid][1])
    print(f'  GW Top-{n_keep} ({"+".join(keep)}): AUC = {auc_of(sub_ens):.4f}')

# ═══ SUMMARY ═══
print('\n=== SUMMARY (ranked by AUC) ===')
all_configs = {
    'Vanilla Ensemble': auc_of({pid: (np.mean([individual[a][pid][0] for a in individual if pid in individual[a]], 0), individual['convnext'][pid][1]) for pid in individual['convnext'] if all(pid in individual[a] for a in individual)}),
    'GW Ensemble': auc_of(gw_ens),
    'TTA Ensemble': auc_of(tta_ens),
    'GW+TTA Ensemble': auc_of(gw_tta_ens),
}
for name, auc in sorted(all_configs.items(), key=lambda x: -x[1]):
    print(f'  {name:<25} AUC = {auc:.4f}')

# Save best
best_name = max(all_configs, key=all_configs.get)
print(f'\nBest: {best_name} (AUC={all_configs[best_name]:.4f})')

# Save the best probs
if 'GW' in best_name:
    best_probs = gw_ens
elif 'TTA' in best_name:
    best_probs = tta_ens
else:
    best_probs = {pid: (np.mean([individual[a][pid][0] for a in ['convnext','vit','swin','efficientnet']], 0), individual['convnext'][pid][1]) for pid in individual['convnext']}

out = {
    'best_config': best_name,
    'best_auc': float(all_configs[best_name]),
    'all_aucs': {k: float(v) for k, v in all_configs.items()},
    'individual_aucs': {a: float(auc_of(individual[a])) for a in individual},
    'gw_individual_aucs': {a: float(auc_of(gw_results[a])) for a in gw_results},
}
with open(os.path.join(RES, 'external_validation_experiments.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: results/external_validation_experiments.json')
