# -*- coding: utf-8 -*-
"""
YOLO11 Training on SAM-processed face frames.
Per manuscript: "Binary classification was first performed using a YOLO11-based image model"

  Stage 1: Binary screening (yolo11n-cls, normal vs jaundiced)
  Stage 2: Ternary grading (yolo11n-cls, mild/moderate/severe)
"""
import os, shutil, json, random, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'sam_processed')
YOLO_DIR = os.path.join(BASE, 'data', 'yolo_dataset')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ── 1. Build YOLO classification dataset ─────────────────────
def build_yolo_dataset(task='binary'):
    """
    Create YOLO cls directory structure:
      dataset/{train,val}/{class_name}/*.jpg
    """
    if task == 'binary':
        classes = {'normal': 'normal', 'mild': 'jaundiced',
                   'moderate': 'jaundiced', 'severe': 'jaundiced'}
    else:
        classes = {'mild': 'mild', 'moderate': 'moderate', 'severe': 'severe'}
    
    # Collect patients
    patients = []
    for cat in classes.keys():
        cat_dir = os.path.join(DATA_ROOT, cat)
        if not os.path.exists(cat_dir): continue
        for d in os.listdir(cat_dir):
            p = os.path.join(cat_dir, d)
            if not os.path.isdir(p): continue
            imgs = [os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')]
            if imgs:
                patients.append({'id': d, 'images': imgs, 'category': cat,
                                 'label': classes[cat]})
    
    print(f'  [{task}] {len(patients)} patients')
    for lbl in set(classes.values()):
        n = sum(1 for p in patients if p['label'] == lbl)
        print(f'    {lbl}: {n} patients')
    
    # Stratified split
    by_label = {}
    for p in patients:
        by_label.setdefault(p['label'], []).append(p)
    
    train_patients, val_patients = [], []
    for lbl, group in by_label.items():
        random.shuffle(group)
        n_val = max(1, int(len(group) * 0.2))
        val_patients.extend(group[:n_val])
        train_patients.extend(group[n_val:])
    
    # Create directories and copy images
    dataset_dir = os.path.join(YOLO_DIR, task)
    for split, pats in [('train', train_patients), ('val', val_patients)]:
        for p in pats:
            lbl = p['label']
            out_dir = os.path.join(dataset_dir, split, lbl)
            os.makedirs(out_dir, exist_ok=True)
            for i, img_path in enumerate(p['images'][:12]):
                dst = os.path.join(out_dir, f"{p['id']}_{i:03d}.jpg")
                if not os.path.exists(dst):
                    # Copy via numpy (handles non-ASCII)
                    import cv2
                    fb = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                    if img is not None:
                        ok, buf = cv2.imencode('.jpg', img)
                        if ok: buf.tofile(dst)
    
    # Count
    for split in ['train', 'val']:
        n = sum(len(os.listdir(os.path.join(dataset_dir, split, d)))
                for d in os.listdir(os.path.join(dataset_dir, split)))
        print(f'    {split}: {n} images')
    
    return dataset_dir


print('=' * 55)
print('  YOLO11 Training on SAM Face Frames')
print('=' * 55)

print('\n[1] Building datasets...')
binary_dir = build_yolo_dataset('binary')
ternary_dir = build_yolo_dataset('ternary')


# ── 2. Train YOLO11 ──────────────────────────────────────────
from ultralytics import YOLO

def train_yolo(dataset_dir, task_name, epochs=60):
    print(f'\n{"="*55}')
    print(f'  Training YOLO11: {task_name}')
    print(f'{"="*55}')
    
    model = YOLO('yolo11n-cls.pt')
    results = model.train(
        data=dataset_dir,
        epochs=epochs,
        batch=32,
        imgsz=224,
        device=0,
        project=os.path.join(RESULTS_DIR, 'yolo_runs'),
        name=task_name,
        seed=SEED,
        patience=20,
        pretrained=True,
        augment=True,
    )
    
    # Load best model
    best_path = os.path.join(RESULTS_DIR, 'yolo_runs', task_name, 'weights', 'best.pt')
    if os.path.exists(best_path):
        # Copy to models dir
        dst = os.path.join(MODEL_DIR, f'yolo_{task_name}.pt')
        shutil.copy(best_path, dst)
        print(f'  Best model saved: {dst}')
    
    return model, results


# Stage 1: Binary
binary_model, binary_results = train_yolo(binary_dir, 'binary_sam', epochs=60)

# Stage 2: Ternary
ternary_model, ternary_results = train_yolo(ternary_dir, 'ternary_sam', epochs=60)


# ── 3. Evaluate ──────────────────────────────────────────────
print('\n[4] Evaluation...')

def evaluate_yolo(model, dataset_dir, task_name):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    
    val_dir = os.path.join(dataset_dir, 'val')
    class_names = sorted(os.listdir(val_dir))
    
    y_true, y_pred, y_prob = [], [], []
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(val_dir, cls_name)
        if not os.path.exists(cls_dir): continue
        for img_file in tqdm(os.listdir(cls_dir), desc=f'Eval {cls_name}', leave=False):
            img_path = os.path.join(cls_dir, img_file)
            results = model.predict(img_path, verbose=False)
            if results and len(results) > 0:
                probs = results[0].probs.data.cpu().numpy()
                pred = probs.argmax()
                y_true.append(cls_idx)
                y_pred.append(pred)
                y_prob.append(probs)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    n_classes = len(class_names)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except:
        auc = 0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    print(f'\n  {task_name}:')
    print(f'    Accuracy: {acc:.4f}')
    print(f'    F1 (macro): {f1:.4f}')
    print(f'    AUC-ROC: {auc:.4f}')
    print(f'    Confusion Matrix: {cm.tolist()}')
    
    return {'accuracy': acc, 'f1': f1, 'auc': auc, 'cm': cm.tolist()}

binary_metrics = evaluate_yolo(binary_model, binary_dir, 'Binary Screening')
ternary_metrics = evaluate_yolo(ternary_model, ternary_dir, 'Ternary Grading')


# ── Summary ──────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  YOLO11 Training Complete!')
print('=' * 55)
print(f'\n  Binary Screening:  AUC={binary_metrics["auc"]:.4f}  Acc={binary_metrics["accuracy"]:.4f}  F1={binary_metrics["f1"]:.4f}')
print(f'  Ternary Grading:   AUC={ternary_metrics["auc"]:.4f}  Acc={ternary_metrics["accuracy"]:.4f}  F1={ternary_metrics["f1"]:.4f}')

results_json = {'binary': binary_metrics, 'ternary': ternary_metrics}
with open(os.path.join(RESULTS_DIR, 'yolo_sam_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

print(f'\n  Models: yolo_binary_sam.pt, yolo_ternary_sam.pt')
print(f'  Results: {RESULTS_DIR}/yolo_sam_results.json')
