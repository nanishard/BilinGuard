# -*- coding: utf-8 -*-
"""
Occlusion Detection v2 — tuned for ward video frames.
Focus: detect gauze/bandage ONLY in periocular region (around eyes)
where jaundice signal is strongest. Body/forehead occlusion doesn't
affect sclera analysis.

Detection: bright white patches (V>210, S<25) that are LARGE (>3% of face area)
AND located in the eye/mid-face region.
"""
import os, cv2, numpy as np, pandas as pd, json
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
VF = os.path.join(BASE, 'data', 'video_frames')

def get_face_bbox(frame_gray):
    try:
        paths = [
            os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
            r'C:\Users\o\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml',
        ]
        for p in paths:
            if os.path.exists(p):
                fc = cv2.CascadeClassifier(p)
                faces = fc.detectMultiScale(frame_gray, 1.1, 5, minSize=(80,80))
                if len(faces) > 0:
                    areas = [w*h for x,y,w,h in faces]
                    x,y,w,h = faces[np.argmax(areas)]
                    return (x, y, x+w, y+h)
    except: pass
    h, w = frame_gray.shape
    return (int(w*0.25), int(h*0.1), int(w*0.75), int(h*0.85))

def detect_occlusion_v2(frame_bgr, face_bbox):
    """Detect gauze/bandage in eye region of face."""
    x1, y1, x2, y2 = face_bbox
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0: return 0, False
    
    fh, fw = face.shape[:2]
    face_area = fh * fw
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    
    # Eye region: upper-middle of face (where jaundice signal lives)
    eye_region = face[int(fh*0.2):int(fh*0.5), int(fw*0.1):int(fw*0.9)]
    if eye_region.size == 0: return 0, False
    eh, ew = eye_region.shape[:2]
    eye_area = eh * ew
    eye_hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
    
    # Strict white detection: V > 215, S < 25 (true gauze white)
    # Not just any bright pixel
    gauze_mask = cv2.inRange(eye_hsv, (0, 0, 215), (180, 25, 255))
    
    # Clean up: only large connected patches
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    gauze_mask = cv2.morphologyEx(gauze_mask, cv2.MORPH_CLOSE, k9)
    gauze_mask = cv2.morphologyEx(gauze_mask, cv2.MORPH_OPEN, k5)
    
    # Find large connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gauze_mask, 8)
    final_mask = np.zeros_like(gauze_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > eye_area * 0.03:  # >3% of eye region
            final_mask[labels == i] = 255
    
    ratio = final_mask.sum() / (255 * eye_area)
    return ratio, ratio > 0.10  # >10% of eye region = occluded


# ── Scan ─────────────────────────────────────────────────────
print('Scanning video frames for periocular occlusion...')
df_manifest = pd.read_csv(os.path.join(BASE, 'data', 'clean_dataset_manifest.csv'))
report = []

for _, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc='Scan'):
    cat = row['category']; pid = row['patient_id']
    vf_dir = os.path.join(VF, cat, pid)
    if not os.path.exists(vf_dir): continue
    
    for fname in sorted(os.listdir(vf_dir)):
        if not fname.endswith('.jpg'): continue
        fpath = os.path.join(vf_dir, fname)
        try:
            fb = np.fromfile(fpath, dtype=np.uint8)
            frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        except: continue
        if frame is None: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = get_face_bbox(gray)
        ratio, is_occ = detect_occlusion_v2(frame, bbox)
        
        report.append({'category': cat, 'patient_id': pid, 'frame': fname,
                       'eye_occlusion_ratio': round(ratio, 3), 'is_occluded': is_occ})

rdf = pd.DataFrame(report)
rdf.to_csv(os.path.join(BASE, 'data', 'occlusion_report_v2.csv'), index=False, encoding='utf-8-sig')

# Summary
total = len(rdf)
occ = rdf['is_occluded'].sum()
print(f'\n{"="*60}')
print(f'  OCCLUSION DETECTION (Eye Region)')
print(f'{"="*60}')
print(f'  Total frames:   {total}')
print(f'  Occluded:       {occ} ({occ/total*100:.1f}%)')
print(f'  Clean:          {total-occ} ({(total-occ)/total*100:.1f}%)')

for cat in ['normal','mild','moderate','severe']:
    sub = rdf[rdf['category']==cat]
    if len(sub)>0:
        r = sub['is_occluded'].mean()*100
        print(f'    {cat:12s}: {r:.1f}% occluded')

# Patient-level
ps = rdf.groupby(['category','patient_id']).agg(
    total=('frame','count'), occ=('is_occluded','sum'),
).reset_index()
ps['clean_rate'] = 1 - ps['occ']/ps['total']
ps['usable'] = (ps['total']-ps['occ']) >= 4
ps.to_csv(os.path.join(BASE, 'data', 'occlusion_patient_v2.csv'), index=False, encoding='utf-8-sig')

not_usable = ps[~ps['usable']]
print(f'\n  Patients with <4 clean frames: {len(not_usable)}/{len(ps)}')
for _, r in not_usable.head(10).iterrows():
    print(f'    {r["category"]:8s} {r["patient_id"][:25]}: {r["occ"]}/{r["total"]} occluded')
