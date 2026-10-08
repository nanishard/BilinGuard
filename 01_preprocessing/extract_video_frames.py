# -*- coding: utf-8 -*-
"""
Extract frames from patient videos (VID_*.mp4) for face model training.
Each video yields 12 uniformly sampled frames.
Saves to data/extracted_video/{category}/{patient_id}/
"""
import os, cv2, numpy as np, json, pandas as pd
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
DATA_ROOT = os.path.join(BASE, 'data', 'extracted')
OUTPUT_ROOT = os.path.join(BASE, 'data', 'video_frames')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
N_FRAMES = 12

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def extract_video_frames(video_path, output_dir, n_frames=12):
    """Extract N uniformly sampled frames from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return 0
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f'frame_{i:03d}.jpg')
            # Use imencode to handle non-ASCII paths
            ok, buf = cv2.imencode('.jpg', frame)
            if ok:
                buf.tofile(out_path)
                saved += 1
    cap.release()
    return saved

# Load manifest
df = pd.read_csv(MANIFEST)
print(f'Total patients in manifest: {len(df)}')

stats = {'total': 0, 'extracted': 0, 'no_video': 0, 'already': 0}

for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting videos'):
    cat = row['category']
    pid = row['patient_id']
    
    # Find the patient's extracted folder
    cat_dir = os.path.join(DATA_ROOT, cat)
    patient_dir = None
    for d in os.listdir(cat_dir):
        if d == pid or pid in d:
            patient_dir = os.path.join(cat_dir, d)
            break
    if not patient_dir:
        continue
    
    # Find inner folder
    inner_dirs = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
    base = os.path.join(patient_dir, inner_dirs[0]) if inner_dirs else patient_dir
    
    # Find video file
    video_path = None
    for f in os.listdir(base):
        if f.lower().endswith('.mp4'):
            video_path = os.path.join(base, f)
            break
    
    if not video_path:
        stats['no_video'] += 1
        continue
    
    # Output directory
    out_dir = os.path.join(OUTPUT_ROOT, cat, pid)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= N_FRAMES:
        stats['already'] += 1
        continue
    
    os.makedirs(out_dir, exist_ok=True)
    stats['total'] += 1
    n = extract_video_frames(video_path, out_dir, N_FRAMES)
    if n > 0:
        stats['extracted'] += 1

print(f'\nResults:')
print(f'  Total videos found:     {stats["total"]}')
print(f'  Frames extracted:       {stats["extracted"]}')
print(f'  Already extracted:      {stats["already"]}')
print(f'  No video file:          {stats["no_video"]}')

# Summary per category
print(f'\nPer-category:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(OUTPUT_ROOT, cat)
    if os.path.exists(cat_dir):
        n_patients = len(os.listdir(cat_dir))
        n_frames = sum(len(os.listdir(os.path.join(cat_dir, d)))
                       for d in os.listdir(cat_dir))
        print(f'  {cat:12s}: {n_patients} patients, {n_frames} frames')
