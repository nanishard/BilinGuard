# -*- coding: utf-8 -*-
"""
Deep audit: use PIL + pixel analysis to classify each image.
  - Face photos: contain skin-tone pixels in upper-center region
  - Body composition reports: white background, text-heavy, charts
  - Eyelid photos: reddish/pink tissue, high saturation in center
"""
import os, numpy as np
from PIL import Image

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')

def load_img_pil(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        return None

def classify_image(path, filename):
    """Classify an image as face/eyelid/report/unknown based on pixel analysis."""
    img = load_img_pil(path)
    if img is None:
        return 'unreadable', {}
    arr = np.array(img.resize((224, 224)))
    
    # Basic stats
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    
    # HSV conversion
    import cv2
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mean_h = hsv[:,:,0].mean()
    mean_s = hsv[:,:,1].mean()
    
    # Edge density (text-heavy reports have high edge density)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    
    # White pixel ratio (reports have white backgrounds)
    white_mask = (r > 230) & (g > 230) & (b > 230)
    white_ratio = white_mask.mean()
    
    # Skin-tone pixel ratio (face photos have skin tones)
    # Skin: R > G > B, R in [95, 255], G in [40, 255], |R-G| > 15
    skin_mask = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 12)
    skin_ratio = skin_mask.mean()
    
    # Red tissue (conjunctiva/eyelid): high R, moderate-high S, hue near 0 or 170-180
    red_mask = ((hsv[:,:,0] < 20) | (hsv[:,:,0] > 160)) & (hsv[:,:,1] > 30) & (r > 100)
    red_ratio = red_mask.mean()
    
    info = {
        'size_kb': os.path.getsize(path) // 1024,
        'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
        'mean_h': mean_h, 'mean_s': mean_s,
        'edge_density': edge_density,
        'white_ratio': white_ratio,
        'skin_ratio': skin_ratio,
        'red_ratio': red_ratio,
        'width': img.size[0], 'height': img.size[1],
    }
    
    # Classification logic
    is_large = info['size_kb'] > 2000
    is_img_prefix = filename.startswith('IMG_')
    
    # Body composition report: high white ratio, high edge density, low skin ratio
    if white_ratio > 0.4 and edge_density > 0.08 and skin_ratio < 0.15:
        return 'report', info
    
    # Eyelid photo: large file, reddish/pink tissue, moderate skin ratio
    if is_large and is_img_prefix:
        if red_ratio > 0.15 or (skin_ratio > 0.2 and mean_s > 25):
            return 'eyelid', info
        elif white_ratio > 0.3:
            return 'report_large', info
        else:
            return 'eyelid', info  # default for large IMG_ files
    
    # Face photo: has skin-tone pixels, moderate size
    if skin_ratio > 0.15:
        return 'face', info
    
    # Unknown
    if white_ratio > 0.5:
        return 'report', info
    
    return 'unknown', info


# ── Scan all images ──────────────────────────────────────────
print('Deep image audit using pixel analysis...')
print('=' * 70)

results = {}
contamination = []

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
        
        for f in os.listdir(base):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if 'feature' in f.lower():
                continue
            fp = os.path.join(base, f)
            img_type, info = classify_image(fp, f)
            key = f'{cat}/{f}'
            results[key] = {'type': img_type, 'category': cat, 'filename': f, 'path': fp, **info}
            
            # Flag contamination
            is_eyelid_name = f.startswith('IMG_') and info['size_kb'] > 2000
            is_face_name = not is_eyelid_name
            
            if is_eyelid_name and img_type in ('report', 'report_large'):
                contamination.append(f'[EYELID-IS-REPORT] {cat}/{f} ({info["size_kb"]}KB, white={info["white_ratio"]:.2f})')
            if is_face_name and img_type in ('report', 'report_large'):
                contamination.append(f'[FACE-IS-REPORT] {cat}/{f} ({info["size_kb"]}KB, white={info["white_ratio"]:.2f})')
            if is_face_name and img_type == 'unknown' and info['skin_ratio'] < 0.05:
                contamination.append(f'[FACE-IS-NOT-FACE] {cat}/{f} ({info["size_kb"]}KB, skin={info["skin_ratio"]:.2f})')

# Summary
type_counts = {}
for r in results.values():
    t = r['type']
    type_counts[t] = type_counts.get(t, 0) + 1

print(f'\nImage type distribution:')
for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f'  {t:20s}: {n:4d} images')

print(f'\nTotal images scanned: {len(results)}')
print(f'Contamination issues found: {len(contamination)}')

# Show contamination samples
if contamination:
    print(f'\nContamination details (first 20):')
    for c in contamination[:20]:
        print(f'  {c}')
else:
    print('\nNo contamination detected by pixel analysis.')

# Breakdown by category
print(f'\nPer-category breakdown:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_results = [r for r in results.values() if r['category'] == cat]
    print(f'  {cat}:')
    for t in ['face', 'eyelid', 'report', 'report_large', 'unknown', 'unreadable']:
        n = sum(1 for r in cat_results if r['type'] == t)
        if n > 0:
            print(f'    {t:20s}: {n}')
