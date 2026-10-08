# -*- coding: utf-8 -*-
"""
Occlusion Detection on ORIGINAL video frames.
Runs BEFORE SAM processing to identify and flag occluded frames.

Detection strategy:
  1. Detect face region in original 1920x1080 frame
  2. Within face region, detect:
     a. White/bright patches (gauze, tape) — HSV: V>200, S<40
     b. Large smooth regions (gauze texture is uniform) — low local variance
     c. Non-skin colored patches in central face
  3. Flag frames with >15% occlusion as 'occluded'
"""
import os, cv2, numpy as np, pandas as pd, json
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
VF = os.path.join(BASE, 'data', 'video_frames')
OUTPUT_CSV = os.path.join(BASE, 'data', 'occlusion_report.csv')
OUTPUT_V2 = os.path.join(BASE, 'data', 'sam_processed_v3')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ── Face detection ───────────────────────────────────────────
def get_face_region(frame_gray):
    """Get face bounding box from original frame."""
    try:
        cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        alt_path = r'C:\Users\o\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
        path = cascade_path if os.path.exists(cascade_path) else alt_path
        if os.path.exists(path):
            fc = cv2.CascadeClassifier(path)
            faces = fc.detectMultiScale(frame_gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                areas = [w*h for x,y,w,h in faces]
                x,y,w,h = faces[np.argmax(areas)]
                return (x, y, x+w, y+h)
    except:
        pass
    # Fallback: center crop
    h, w = frame_gray.shape
    return (int(w*0.25), int(h*0.1), int(w*0.75), int(h*0.8))


def detect_occlusion_original(frame_bgr, face_bbox):
    """
    Detect occlusion in original frame within face region.
    Returns: (occlusion_ratio, is_occluded, details)
    """
    x1, y1, x2, y2 = face_bbox
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return 0, False, {}
    
    h, w = face.shape[:2]
    face_area = h * w
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # 1. White/bright detection (gauze, medical tape, bandages)
    #    Very white: V > 200, S < 50
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    
    # 2. Light-colored non-skin (beige/skin-colored gauze)
    #    Gauze can be skin-colored but has LOW TEXTURE (smooth)
    #    Compute local variance to find smooth regions
    kernel_size = 15
    local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((gray.astype(np.float32))**2, (kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Smooth + bright = likely gauze
    smooth_bright = ((local_std < 8) & (gray > 120)).astype(np.uint8) * 255
    
    # 3. Combine candidate occlusion masks
    candidate = cv2.bitwise_or(white_mask, smooth_bright)
    
    # Clean up
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k11)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k5)
    
    # 4. Filter: only large connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    occlusion_mask = np.zeros_like(candidate)
    largest_area = 0
    n_patches = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > face_area * 0.01:  # >1% of face
            occlusion_mask[labels == i] = 255
            largest_area = max(largest_area, area)
            n_patches += 1
    
    # Dilate to cover edges of gauze
    occlusion_mask = cv2.dilate(occlusion_mask, k11, iterations=1)
    
    ratio = occlusion_mask.sum() / (255 * face_area)
    
    details = {
        'white_area_pct': white_mask.sum() / (255 * face_area) * 100,
        'smooth_bright_pct': smooth_bright.sum() / 255 * 100 / face_area * 255,
        'n_occlusion_patches': n_patches,
        'largest_patch_pct': largest_area / face_area * 100,
    }
    
    return ratio, ratio > 0.15, details  # >15% = occluded


# ── Scan all video frames ────────────────────────────────────
print('[1] Scanning video frames for occlusion...')
df_manifest = pd.read_csv(MANIFEST)
report_rows = []

for _, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc='Occlusion scan'):
    cat = row['category']
    pid = row['patient_id']
    vf_dir = os.path.join(VF, cat, pid)
    
    if not os.path.exists(vf_dir):
        continue
    
    frames = sorted([f for f in os.listdir(vf_dir) if f.endswith('.jpg')])
    n_clean = 0
    n_occluded = 0
    
    for fname in frames:
        fpath = os.path.join(vf_dir, fname)
        try:
            fb = np.fromfile(fpath, dtype=np.uint8)
            frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        except:
            continue
        if frame is None:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = get_face_region(gray)
        ratio, is_occ, details = detect_occlusion_original(frame, bbox)
        
        report_rows.append({
            'category': cat, 'patient_id': pid, 'frame': fname,
            'occlusion_ratio': round(ratio, 3),
            'is_occluded': is_occ,
            'white_pct': round(details['white_area_pct'], 1),
            'n_patches': details['n_occlusion_patches'],
        })
        
        if is_occ:
            n_occluded += 1
        else:
            n_clean += 1
    
    # Patient-level: mark if patient has enough clean frames
    total = n_clean + n_occluded

# ── Save report ──────────────────────────────────────────────
rdf = pd.DataFrame(report_rows)
rdf.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

# ── Summary ──────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'  OCCLUSION DETECTION REPORT')
print(f'{"="*60}')
total_frames = len(rdf)
occ_frames = rdf['is_occluded'].sum()
clean_frames = total_frames - occ_frames
print(f'  Total frames scanned:  {total_frames}')
print(f'  Clean (usable):        {clean_frames} ({clean_frames/total_frames*100:.1f}%)')
print(f'  Occluded (>15%):       {occ_frames} ({occ_frames/total_frames*100:.1f}%)')

print(f'\n  Per-category occlusion rate:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    sub = rdf[rdf['category'] == cat]
    if len(sub) > 0:
        occ_rate = sub['is_occluded'].mean() * 100
        print(f'    {cat:12s}: {occ_rate:.1f}% occluded ({sub["is_occluded"].sum()}/{len(sub)} frames)')

# Patient-level summary
print(f'\n  Patient-level:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    sub = rdf[rdf['category'] == cat]
    if len(sub) == 0: continue
    patients = sub.groupby('patient_id').agg(
        n_frames=('frame', 'count'),
        n_occ=('is_occluded', 'sum'),
    ).reset_index()
    heavily = (patients['n_occ'] / patients['n_frames'] > 0.5).sum()
    print(f'    {cat:12s}: {len(patients)} patients, {heavily} with >50% frames occluded')

print(f'\n  Report saved: {OUTPUT_CSV}')

# ── Flag patients with heavy occlusion ───────────────────────
patient_summary = rdf.groupby(['category', 'patient_id']).agg(
    total_frames=('frame', 'count'),
    occluded_frames=('is_occluded', 'sum'),
    avg_ratio=('occlusion_ratio', 'mean'),
).reset_index()
patient_summary['occlusion_rate'] = patient_summary['occluded_frames'] / patient_summary['total_frames']
patient_summary['has_enough_clean'] = (patient_summary['total_frames'] - patient_summary['occluded_frames']) >= 4
patient_summary.to_csv(os.path.join(BASE, 'data', 'occlusion_patient_summary.csv'),
                        index=False, encoding='utf-8-sig')

not_enough = patient_summary[~patient_summary['has_enough_clean']]
print(f'\n  Patients with <4 clean frames: {len(not_enough)}')
if len(not_enough) > 0:
    for _, r in not_enough.head(10).iterrows():
        print(f'    {r["category"]:8s} {r["patient_id"][:25]}: {r["occluded_frames"]}/{r["total_frames"]} occluded ({r["occlusion_rate"]*100:.0f}%)')
