# -*- coding: utf-8 -*-
"""
Master retraining script: runs all face model training scripts
using face_v4 data, then restores eyelid models from backup.
"""

import os, shutil, subprocess, time, sys
from datetime import datetime

BASE = r'D:\research\人脸识别营养\传染科'
MODEL_DIR = os.path.join(BASE, 'models')
BACKUP_DIR = os.path.join(BASE, 'models_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
CODE = os.path.join(BASE, 'code')
LOG = os.path.join(BASE, 'results')

os.makedirs(BACKUP_DIR, exist_ok=True)

# Step 0: Backup all models
print("=" * 60)
print("  Master Retraining Pipeline - All Face Models")
print("=" * 60)
print(f"\n[0] Backing up current models to {BACKUP_DIR}")
for f in os.listdir(MODEL_DIR):
    if f.endswith('.pt'):
        shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(BACKUP_DIR, f))
pt_count = len([f for f in os.listdir(BACKUP_DIR) if f.endswith('.pt')])
print(f"  Backed up {pt_count} models")

# Scripts to run (in order)
scripts = [
    ('train_binary_final.py', 'final_binary_face_*'),
    ('train_cp_meld_images.py', 'cp_face_* + meld_face_*'),
    ('train_dbil_classification.py', 'dbil_face_*'),
    ('train_ibil_classification_v2.py', 'ibil_face_*'),
    ('train_type_classification.py', 'type_class_* + type_binary_*'),
    ('train_v3.py', 'v3_ternary_* + v3_binary_*'),
]

results = []
for script, description in scripts:
    fp = os.path.join(CODE, script)
    if not os.path.exists(fp):
        print(f"\n[SKIP] {script} (not found)")
        results.append((script, 'SKIP', None))
        continue
    
    log_path = os.path.join(LOG, f'{script.replace(".py","")}_retrain.log')
    print(f"\n[{scripts.index((script, description))+1}] Running {script} ({description})")
    print(f"  Log: {log_path}")
    start = time.time()
    
    with open(log_path, 'w', encoding='utf-8') as f:
        r = subprocess.run(['python', fp], stdout=f, stderr=subprocess.STDOUT,
                          timeout=14400, cwd=CODE)  # 4 hours max
    
    elapsed = time.time() - start
    if r.returncode == 0:
        print(f"  ✅ Completed in {elapsed/60:.1f} min")
        results.append((script, 'OK', elapsed))
    else:
        print(f"  ⚠️  Exit code {r.returncode} after {elapsed/60:.1f} min")
        results.append((script, f'EXIT_{r.returncode}', elapsed))

# Restore eyelid models from backup
print(f"\n[Restore] Restoring eyelid models from backup...")
restored = 0
for f in os.listdir(BACKUP_DIR):
    if 'eyelid' in f.lower() and f.endswith('.pt'):
        src = os.path.join(BACKUP_DIR, f)
        dst = os.path.join(MODEL_DIR, f)
        shutil.copy2(src, dst)
        restored += 1
print(f"  Restored {restored} eyelid models")

# Summary
print("\n" + "=" * 60)
print("  Training Summary")
print("=" * 60)
for script, status, elapsed in results:
    elapsed_str = f'{elapsed/60:.1f}min' if elapsed else ''
    print(f"  {script:40s} {status:10s} {elapsed_str}")

# Count new models
print(f"\n  Face models retrained on face_v4:")
face_count = 0
for f in sorted(os.listdir(MODEL_DIR)):
    if f.endswith('.pt') and 'eyelid' not in f.lower():
        face_count += 1
print(f"    {face_count} face models in models/")

print(f"\n  Eyelid models restored from backup: {restored}")
print(f"\n  Backup: {BACKUP_DIR}")
print("=" * 60)
