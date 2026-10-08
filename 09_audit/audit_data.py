# -*- coding: utf-8 -*-
"""
Audit extracted data for contamination:
  1. sporthealth-*.jpg that are NOT face photos (e.g. body composition reports)
  2. IMG_*.jpg that are NOT eyelid photos
Uses: file size, image dimensions, face detection
"""
import os, sys, cv2, numpy as np
from PIL import Image

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')

# Load face detector
try:
    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
    HAS_MP = True
except:
    HAS_MP = False
    print('MediaPipe not available, using size/dimension heuristics only')

def has_face(img_path):
    """Check if image contains a detectable face."""
    if not HAS_MP:
        return None
    try:
        file_bytes = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)
        return len(results.detections) > 0 if results.detections else False
    except:
        return None

def get_image_info(path):
    """Get basic image properties."""
    try:
        file_bytes = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        return {'width': w, 'height': h, 'aspect': w/h if h > 0 else 0, 'size_kb': os.path.getsize(path)//1024}
    except:
        return None

# Scan all patients
print('Scanning extracted data for contamination...')
print('=' * 70)

stats = {
    'total_face_imgs': 0,
    'face_no_face_detected': 0,
    'face_is_report': 0,
    'total_eyelid_imgs': 0,
    'eyelid_no_face': 0,
    'eyelid_wrong_size': 0,
}

contaminated = []
patients_with_issues = set()

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
        
        all_files = os.listdir(base)
        # Check each image
        for f in all_files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if 'feature' in f.lower():
                continue
            fp = os.path.join(base, f)
            info = get_image_info(fp)
            if info is None:
                continue
            
            is_eyelid = f.startswith('IMG_') and info['size_kb'] > 2000
            is_face = 'sporthealth' in f.lower() or (f.lower().endswith('.jpg') and not is_eyelid)
            
            if is_eyelid:
                stats['total_eyelid_imgs'] += 1
                # Eyelid photos should be large, portrait orientation (taller than wide or roughly square)
                # But body composition reports are also large...
                face_detected = has_face(fp)
                # Check aspect ratio: eyelid photos are usually portrait (aspect < 1.0)
                # Body composition reports are also portrait
                # Key difference: eyelid photos have reddish/pinkish tones (conjunctiva)
                img = cv2.imread(fp)
                if img is not None:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mean_hue = hsv[:,:,0].mean()
                    mean_sat = hsv[:,:,1].mean()
                    # Conjunctiva: reddish-pink (hue 0-20 or 340-360), moderate saturation
                    is_reddish = mean_hue < 30 or mean_hue > 170
                    if not is_reddish and mean_sat < 30:
                        stats['eyelid_wrong_size'] += 1
                        contaminated.append({
                            'file': f, 'path': fp, 'category': cat,
                            'type': 'eyelid_not_reddish',
                            'size_kb': info['size_kb'], 'hue': mean_hue, 'sat': mean_sat,
                        })
                        patients_with_issues.add(outer)
            elif is_face:
                stats['total_face_imgs'] += 1
                # Check if it actually has a face
                face_detected = has_face(fp)
                if face_detected is False:
                    stats['face_no_face_detected'] += 1
                    contaminated.append({
                        'file': f, 'path': fp, 'category': cat,
                        'type': 'face_no_face',
                        'size_kb': info['size_kb'],
                    })
                    patients_with_issues.add(outer)
                # Check if it's a body composition report (usually white background, text-heavy)
                if face_detected is False and info['size_kb'] < 500:
                    stats['face_is_report'] += 1

# Also check feature subfolders
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA_ROOT, cat)
    if not os.path.exists(cat_dir):
        continue
    for outer in os.listdir(cat_dir)[:5]:  # sample 5 per category
        outer_path = os.path.join(cat_dir, outer)
        if not os.path.isdir(outer_path):
            continue
        inner_dirs = [d for d in os.listdir(outer_path) if os.path.isdir(os.path.join(outer_path, d))]
        base = os.path.join(outer_path, inner_dirs[0]) if inner_dirs else outer_path
        for sub in os.listdir(base):
            sp = os.path.join(base, sub)
            if os.path.isdir(sp):
                sub_files = [f for f in os.listdir(sp) if f.lower().endswith('.jpg')]
                if sub_files:
                    print(f'  Feature subfolder {sub} ({len(sub_files)} files)')

print()
print('=' * 70)
print('AUDIT RESULTS')
print('=' * 70)
print(f'Face images scanned:           {stats["total_face_imgs"]}')
print(f'  - No face detected:           {stats["face_no_face_detected"]}')
print(f'  - Likely body comp report:    {stats["face_is_report"]}')
print(f'Eyelid images scanned:          {stats["total_eyelid_imgs"]}')
print(f'  - Not reddish (possible report): {stats["eyelid_wrong_size"]}')
print(f'\nTotal contaminated files:       {len(contaminated)}')
print(f'Patients with issues:           {len(patients_with_issues)}')

# Categorize contamination types
types = {}
for c in contaminated:
    types[c['type']] = types.get(c['type'], 0) + 1
print(f'\nContamination breakdown:')
for t, n in sorted(types.items(), key=lambda x: -x[1]):
    print(f'  {t:30s}: {n}')

# Show samples
print(f'\nSample contaminated files:')
for c in contaminated[:10]:
    print(f'  [{c["category"]:8s}] {c["file"]:40s} type={c["type"]} size={c.get("size_kb","")}KB')
