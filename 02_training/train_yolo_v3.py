# -*- coding: utf-8 -*-
"""
YOLO11 Training on SAM v3 data (occlusion-filtered, original pixels)
  Stage 1: Binary screening (yolo11n-cls)
  Stage 2: Ternary grading (yolo11n-cls)
"""
import os, shutil, json, random, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
YOLO_DIR = os.path.join(BASE, 'data', 'yolo_v3')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
SEED = 42
random.seed(SEED); np.random.seed(SEED)

def build_yolo(task='binary'):
    if task == 'binary':
        cls_map = {'normal': 'normal', 'mild': 'jaundiced', 'moderate': 'jaundiced', 'severe': 'jaundiced'}
    else:
        cls_map = {'mild': 'mild', 'moderate': 'moderate', 'severe': 'severe'}
    
    patients = []
    for cat in cls_map:
        cd = os.path.join(DATA, cat)
        if not os.path.exists(cd): continue
        for d in os.listdir(cd):
            p = os.path.join(cd, d)
            if not os.path.isdir(p): continue
            faces = [os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')]
            if faces:
                patients.append({'id': d, 'images': faces, 'label': cls_map[cat]})
    
    # Split
    by_l = {}
    for p in patients: by_l.setdefault(p['label'], []).append(p)
    train_p, val_p = [], []
    for l, g in by_l.items():
        random.shuffle(g); n = max(1, int(len(g) * 0.2))
        val_p.extend(g[:n]); train_p.extend(g[n:])
    
    # Copy to YOLO format
    ds_dir = os.path.join(YOLO_DIR, task)
    if os.path.exists(ds_dir): shutil.rmtree(ds_dir)
    
    for split, pats in [('train', train_p), ('val', val_p)]:
        for p in pats:
            out = os.path.join(ds_dir, split, p['label'])
            os.makedirs(out, exist_ok=True)
            for img_path in p['images'][:12]:
                dst = os.path.join(out, f"{p['id']}_{os.path.basename(img_path)}")
                try:
                    fb = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                    if img is not None:
                        ok, buf = cv2.imencode('.jpg', img)
                        if ok: buf.tofile(dst)
                except: pass
    
    # Count
    for split in ['train', 'val']:
        n = sum(len(os.listdir(os.path.join(ds_dir, split, d)))
                for d in os.listdir(os.path.join(ds_dir, split)))
        print(f'    {task} {split}: {n} images')
    return ds_dir

print('='*55)
print('  YOLO11 on SAM v3 Data')
print('='*55)

print('\n[1] Building datasets...')
binary_dir = build_yolo('binary')
ternary_dir = build_yolo('ternary')

print('\n[2] Training YOLO11 Binary...')
from ultralytics import YOLO

binary_model = YOLO('yolo11n-cls.pt')
binary_model.train(
    data=binary_dir, epochs=60, batch=32, imgsz=224, device=0,
    project=os.path.join(RES, 'yolo_v3_runs'), name='binary', seed=SEED,
    patience=20, pretrained=True, augment=True, lr0=0.01, cos_lr=True, workers=0
)

# Copy best model
best = os.path.join(RES, 'yolo_v3_runs', 'binary', 'weights', 'best.pt')
if os.path.exists(best):
    shutil.copy(best, os.path.join(MODEL, 'v3_yolo_binary.pt'))
    print('  Saved: v3_yolo_binary.pt')

print('\n[3] Training YOLO11 Ternary...')
ternary_model = YOLO('yolo11n-cls.pt')
ternary_model.train(
    data=ternary_dir, epochs=60, batch=32, imgsz=224, device=0,
    project=os.path.join(RES, 'yolo_v3_runs'), name='ternary', seed=SEED,
    patience=20, pretrained=True, augment=True, lr0=0.01, cos_lr=True, workers=0
)

best = os.path.join(RES, 'yolo_v3_runs', 'ternary', 'weights', 'best.pt')
if os.path.exists(best):
    shutil.copy(best, os.path.join(MODEL, 'v3_yolo_ternary.pt'))
    print('  Saved: v3_yolo_ternary.pt')

# ── Evaluate ─────────────────────────────────────────────────
print('\n[4] Evaluation...')
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def eval_yolo(model_path, ds_dir, task):
    model = YOLO(model_path)
    val_dir = os.path.join(ds_dir, 'val')
    classes = sorted(os.listdir(val_dir))
    y_true, y_pred, y_prob = [], [], []
    for ci, cn in enumerate(classes):
        cd = os.path.join(val_dir, cn)
        if not os.path.exists(cd): continue
        for f in tqdm(os.listdir(cd), desc=f'Eval {cn}', leave=False):
            r = model.predict(os.path.join(cd, f), verbose=False)
            if r and len(r) > 0:
                p = r[0].probs.data.cpu().numpy()
                y_true.append(ci); y_pred.append(p.argmax()); y_prob.append(p)
    yt = np.array(y_true); yp = np.array(y_pred); ypr = np.array(y_prob)
    acc = accuracy_score(yt, yp)
    f1 = f1_score(yt, yp, average='macro')
    nc = len(classes)
    try: auc = roc_auc_score(yt, ypr[:, 1]) if nc == 2 else roc_auc_score(yt, ypr, multi_class='ovr')
    except: auc = 0
    cm = confusion_matrix(yt, yp, labels=list(range(nc)))
    print(f'\n  {task}: AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}')
    print(f'  CM: {cm.tolist()}')
    return {'auc': auc, 'acc': acc, 'f1': f1, 'cm': cm.tolist()}

b_metrics = eval_yolo(os.path.join(MODEL, 'v3_yolo_binary.pt'), binary_dir, 'Binary (YOLO11)')
t_metrics = eval_yolo(os.path.join(MODEL, 'v3_yolo_ternary.pt'), ternary_dir, 'Ternary (YOLO11)')

# Summary
print('\n' + '='*55)
print('  YOLO11 Results')
print('='*55)
print(f'\n  Binary:  AUC={b_metrics["auc"]:.4f}  Acc={b_metrics["acc"]:.4f}  F1={b_metrics["f1"]:.4f}')
print(f'  Ternary: AUC={t_metrics["auc"]:.4f}  Acc={t_metrics["acc"]:.4f}  F1={t_metrics["f1"]:.4f}')

with open(os.path.join(RES, 'v3_yolo_results.json'), 'w') as f:
    json.dump({'binary': b_metrics, 'ternary': t_metrics}, f, indent=2, default=str)
