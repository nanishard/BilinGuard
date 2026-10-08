# -*- coding: utf-8 -*-
"""
Clean training data: remove body composition reports and non-face images.
  1. Exclude all sporthealth-*.jpg (body composition reports)
  2. For face images: keep only those with detectable skin/face pixels
  3. For eyelid images: verify they show conjunctival tissue
  4. Write clean dataset manifest
"""
import os, json, numpy as np, cv2
from PIL import Image

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')

def load_pil(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        return None

def is_body_report(path):
    """Detect body composition report (white bg, text-heavy, no skin)."""
    img = load_pil(path)
    if img is None:
        return True  # can't read = exclude
    arr = np.array(img.resize((224, 224)))
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    white_mask = (r > 230) & (g > 230) & (b > 230)
    white_ratio = white_mask.mean()
    
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    
    skin_mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 12)
    skin_ratio = skin_mask.mean()
    
    # Report: high white ratio + high edge density + low skin
    return white_ratio > 0.35 and skin_ratio < 0.15

def has_face_features(path):
    """Check if image has face-like content (skin pixels in center region)."""
    img = load_pil(path)
    if img is None:
        return False
    arr = np.array(img.resize((224, 224)))
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    # Check center region for skin
    center = arr[56:168, 56:168]
    cr, cg, cb = center[:,:,0], center[:,:,1], center[:,:,2]
    skin_mask = (cr > 95) & (cg > 40) & (cb > 20) & (cr > cg) & (cr > cb) & ((cr - cg) > 12)
    skin_ratio = skin_mask.mean()
    
    return skin_ratio > 0.10

def is_eyelid_photo(path):
    """Check if image looks like an eyelid photo (reddish/pink tissue)."""
    img = load_pil(path)
    if img is None:
        return False
    arr = np.array(img.resize((224, 224)))
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    
    # Reddish/pink tissue (conjunctiva)
    red_mask = ((hsv[:,:,0] < 25) | (hsv[:,:,0] > 155)) & (hsv[:,:,1] > 20) & (r > 80)
    red_ratio = red_mask.mean()
    
    # Also check for skin tones
    skin_mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 12)
    skin_ratio = skin_mask.mean()
    
    # Eyelid photo: should have reddish/pink tissue OR significant skin
    return red_ratio > 0.10 or skin_ratio > 0.15


# ── Scan and classify all images ─────────────────────────────
print('Cleaning dataset...')
clean_records = []
stats = {'total': 0, 'excluded_report': 0, 'excluded_noface': 0,
         'kept_face': 0, 'kept_eyelid': 0, 'excluded_bad_eyelid': 0}

for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA_ROOT, cat)
    if not os.path.exists(cat_dir):
        continue
    for outer in os.listdir(cat_dir):
        outer_path = os.path.join(cat_dir, outer)
        if not os.path.isdir(outer_path):
            continue
        inner_dirs = [d for d in os.listdir(outer_path) if os.path.isdir(os.path.join(outer_path, d))]
        base = os.path.join(outer_path, inner_dirs[0]) if inner_dirs else outer_path
        
        face_imgs = []
        eyelid_imgs = []
        
        for f in os.listdir(base):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if 'feature' in f.lower():
                continue
            fp = os.path.join(base, f)
            stats['total'] += 1
            sz = os.path.getsize(fp)
            is_eyelid_name = f.startswith('IMG_') and sz > 2_000_000
            
            # Skip body composition reports
            if is_body_report(fp):
                stats['excluded_report'] += 1
                continue
            
            if is_eyelid_name:
                # Verify it's actually an eyelid photo
                if is_eyelid_photo(fp):
                    eyelid_imgs.append(fp)
                    stats['kept_eyelid'] += 1
                else:
                    stats['excluded_bad_eyelid'] += 1
            else:
                # Face photo: verify it has face features
                if has_face_features(fp):
                    face_imgs.append(fp)
                    stats['kept_face'] += 1
                else:
                    stats['excluded_noface'] += 1
        
        if face_imgs or eyelid_imgs:
            clean_records.append({
                'patient_id': outer,
                'category': cat,
                'face_images': json.dumps(face_imgs),
                'n_face': len(face_imgs),
                'eyelid_images': json.dumps(eyelid_imgs),
                'n_eyelid': len(eyelid_imgs),
                'has_face': len(face_imgs) > 0,
                'has_eyelid': len(eyelid_imgs) > 0,
            })

# Save manifest
import pandas as pd
df = pd.DataFrame(clean_records)
df.to_csv(MANIFEST, index=False, encoding='utf-8-sig')

print(f'\n{"="*60}')
print(f'  Dataset Cleaning Results')
print(f'{"="*60}')
print(f'  Total images scanned:     {stats["total"]}')
print(f'  Excluded (body reports):  {stats["excluded_report"]}')
print(f'  Excluded (no face):       {stats["excluded_noface"]}')
print(f'  Excluded (bad eyelid):     {stats["excluded_bad_eyelid"]}')
print(f'  Kept (face photos):       {stats["kept_face"]}')
print(f'  Kept (eyelid photos):     {stats["kept_eyelid"]}')
print(f'\n  Clean patients: {len(df)}')
print(f'  With face photos: {df["has_face"].sum()}')
print(f'  With eyelid photos: {df["has_eyelid"].sum()}')

print(f'\n  Per-category:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    sub = df[df['category'] == cat]
    print(f'    {cat:12s}: {len(sub):3d} patients | '
          f'face={sub["n_face"].sum():4d} imgs | '
          f'eyelid={sub["n_eyelid"].sum():4d} imgs')

print(f'\n  Manifest saved: {MANIFEST}')
