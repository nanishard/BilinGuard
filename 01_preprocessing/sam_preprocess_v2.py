# -*- coding: utf-8 -*-
"""
SAM Preprocessing v2 — per user requirements:
  1. Use ORIGINAL video frames (not mean-color filled)
  2. SAM segmentation only for face crop (no fill, no zero mask)
  3. Filter: keep only frames where EYES ARE OPEN (detect eye openness)
  4. Pipeline: frame → face detect → SAM crop → CLAHE → 224x224
  
Eye openness detection: use eye aspect ratio (EAR) via OpenCV landmarks
or brightness/variance analysis of eye region.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2, numpy as np, json, pandas as pd, torch, glob
from PIL import Image
from transformers import SamModel, SamProcessor
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
VIDEO_FRAMES = os.path.join(BASE, 'data', 'video_frames')
OUTPUT_ROOT = os.path.join(BASE, 'data', 'sam_processed_v2')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ── Load SAM ─────────────────────────────────────────────────
print('[1] Loading SAM...')
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to('cuda')
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model.eval()

# ── Eye openness detection ───────────────────────────────────
# Load Haar cascade for eye detection (ships with opencv-python-headless)
def get_eye_cascade():
    """Try multiple paths to find eye cascade XML."""
    # Try to find in opencv data directory
    candidates = [
        os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_eye.xml'),
        os.path.join(os.path.dirname(cv2.__file__), '..', 'data', 'haarcascade_eye.xml'),
    ]
    # Also check standard locations
    for cv_dir in [r'C:\Users\o\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data']:
        candidates.append(os.path.join(cv_dir, 'haarcascade_eye.xml'))
    
    for path in candidates:
        if os.path.exists(path):
            return cv2.CascadeClassifier(path)
    return None

EYE_CASCADE = get_eye_cascade()

def is_eyes_open(frame_gray):
    """Check if eyes are open using eye cascade detection.
    Returns True if at least one eye is detected (open eyes are detectable)."""
    if EYE_CASCADE is None:
        # Fallback: check brightness/variance in upper face region
        h, w = frame_gray.shape
        eye_region = frame_gray[int(h*0.2):int(h*0.45), int(w*0.15):int(w*0.85)]
        # If eye region has enough variance and brightness, assume open
        return eye_region.std() > 20 and eye_region.mean() > 40
    
    eyes = EYE_CASCADE.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    return len(eyes) >= 1

def get_face_bbox_simple(frame_gray):
    """Detect face using simple Haar cascade."""
    face_cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
    if not os.path.exists(face_cascade_path):
        for p in [r'C:\Users\o\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml']:
            if os.path.exists(p):
                face_cascade_path = p
                break
    
    if os.path.exists(face_cascade_path):
        fc = cv2.CascadeClassifier(face_cascade_path)
        faces = fc.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            # Return largest face
            areas = [w*h for (x,y,w,h) in faces]
            idx = np.argmax(areas)
            x, y, w, h = faces[idx]
            return (x, y, x+w, y+h)
    
    # Fallback: center crop
    h, w = frame_gray.shape
    return (int(w*0.15), int(h*0.05), int(w*0.85), int(h*0.95))


def process_frame(frame_bgr):
    """
    Process one frame:
      1. Check eyes open
      2. Detect face bbox
      3. SAM segment face
      4. Crop to face (ORIGINAL pixels, no fill)
      5. CLAHE
      Returns: (face_224, sclera_224) or (None, None) if eyes closed
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Check eyes open — skip closed-eye frames
    if not is_eyes_open(gray):
        return None, None
    
    # 2. Get face bbox
    bbox = get_face_bbox_simple(gray)
    if bbox is None:
        return None, None
    x1, y1, x2, y2 = bbox
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    
    # 3. SAM segmentation (box prompt)
    try:
        pil_img = Image.fromarray(frame_rgb)
        inputs = sam_processor(pil_img, input_boxes=[[[list(bbox)]]], return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())
        mask = masks[0][0][0].numpy()
    except:
        mask = np.ones((h, w), dtype=bool)
    
    # 4. Crop to mask bbox — KEEP ORIGINAL PIXELS (no fill, no zero mask)
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ys = np.where(rows)[0]; xs = np.where(cols)[0]
        rmin = max(0, ys[0] - 5); rmax = min(h, ys[-1] + 5)
        cmin = max(0, xs[0] - 5); cmax = min(w, xs[-1] + 5)
    else:
        rmin, rmax, cmin, cmax = y1, y2, x1, x2
    
    face_crop = frame_rgb[rmin:rmax, cmin:cmax].copy()
    
    # 5. CLAHE on L* channel
    lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    face_clahe = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    # 6. Resize to 224x224
    face_224 = cv2.resize(face_clahe, (224, 224))
    
    # 7. Sclera extraction (from CLAHE face, eye region)
    fh, fw = face_clahe.shape[:2]
    eye_region = face_clahe[int(fh*0.20):int(fh*0.45), int(fw*0.10):int(fw*0.90)].copy()
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_RGB2HSV)
    sclera_mask = cv2.inRange(hsv, (20, 10, 70), (60, 60, 100))
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    sclera_mask = cv2.bitwise_or(sclera_mask, white_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sclera_mask = cv2.morphologyEx(sclera_mask, cv2.MORPH_OPEN, kernel)
    sclera = cv2.bitwise_and(eye_region, eye_region, mask=sclera_mask)
    if sclera_mask.sum() < 50:
        sclera = eye_region
    sclera_224 = cv2.resize(sclera, (224, 224))
    
    return face_224, sclera_224


# ── Process all frames ───────────────────────────────────────
print('\n[2] Processing with SAM v2 (original pixels, eyes-open only)...')
df = pd.read_csv(MANIFEST)

stats = {'total': 0, 'processed': 0, 'eyes_closed': 0, 'failed': 0, 'skipped': 0}

for _, row in tqdm(df.iterrows(), total=len(df), desc='SAM v2'):
    cat = row['category']
    pid = row['patient_id']
    
    in_dir = os.path.join(VIDEO_FRAMES, cat, pid)
    out_dir = os.path.join(OUTPUT_ROOT, cat, pid)
    
    if not os.path.exists(in_dir):
        stats['skipped'] += 1
        continue
    
    # Skip if already done
    if os.path.exists(out_dir):
        existing = [f for f in os.listdir(out_dir) if '_face' in f]
        if len(existing) >= 6:  # At least 6 open-eye frames
            stats['skipped'] += 1
            continue
    
    os.makedirs(out_dir, exist_ok=True)
    
    frames = sorted([f for f in os.listdir(in_dir) if f.endswith('.jpg')])
    stats['total'] += len(frames)
    
    for fname in frames:
        fpath = os.path.join(in_dir, fname)
        try:
            fb = np.fromfile(fpath, dtype=np.uint8)
            frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        except:
            frame = None
        if frame is None:
            stats['failed'] += 1
            continue
        
        face, sclera = process_frame(frame)
        
        if face is None:
            stats['eyes_closed'] += 1
            continue
        
        # Save face (original pixels, SAM-cropped, CLAHE)
        face_path = os.path.join(out_dir, fname.replace('.jpg', '_face.jpg'))
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        if ok: buf.tofile(face_path)
        
        # Save sclera
        sclera_path = os.path.join(out_dir, fname.replace('.jpg', '_sclera.jpg'))
        ok2, buf2 = cv2.imencode('.jpg', cv2.cvtColor(sclera, cv2.COLOR_RGB2BGR))
        if ok2: buf2.tofile(sclera_path)
        
        stats['processed'] += 1

print(f'\n{"="*55}')
print(f'  SAM v2 Processing Results')
print(f'{"="*55}')
print(f'  Total frames:       {stats["total"]}')
print(f'  Processed (open):   {stats["processed"]}')
print(f'  Skipped (closed):   {stats["eyes_closed"]}')
print(f'  Failed:             {stats["failed"]}')
print(f'  Already done:       {stats["skipped"]}')
print(f'  Open-eye ratio:     {stats["processed"]/(stats["processed"]+stats["eyes_closed"])*100:.1f}%')

print(f'\n  Per-category:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(OUTPUT_ROOT, cat)
    if os.path.exists(cat_dir):
        n_pat = len(os.listdir(cat_dir))
        n_files = sum(len(os.listdir(os.path.join(cat_dir, d))) for d in os.listdir(cat_dir))
        print(f'    {cat:12s}: {n_pat} patients, {n_files} files')
