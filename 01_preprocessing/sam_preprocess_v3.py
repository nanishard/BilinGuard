# -*- coding: utf-8 -*-
"""
SAM Processing v3 — with occlusion filtering.
Pipeline:
  1. Read original video frame
  2. Check eyes open + occlusion (skip if closed or occluded)
  3. SAM face segmentation → tight crop (original pixels)
  4. CLAHE
  5. Sclera extraction
  6. Save 224x224 face + sclera
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2, numpy as np, pandas as pd, torch, json
from PIL import Image
from transformers import SamModel, SamProcessor
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
VF = os.path.join(BASE, 'data', 'video_frames')
OUTPUT = os.path.join(BASE, 'data', 'sam_processed_v3')
OCC_CSV = os.path.join(BASE, 'data', 'occlusion_report_v2.csv')
os.makedirs(OUTPUT, exist_ok=True)

# Load occlusion report
occ_df = pd.read_csv(OCC_CSV)
occluded_set = set()
for _, r in occ_df[occ_df['is_occluded']].iterrows():
    occluded_set.add((r['patient_id'], r['frame']))
print(f'[1] Loaded occlusion report: {len(occluded_set)} occluded frame-patient pairs will be skipped')

# Load SAM
print('[2] Loading SAM...')
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to('cuda')
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model.eval()

def get_face_bbox(gray):
    try:
        for p in [os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
                  r'C:\Users\o\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml']:
            if os.path.exists(p):
                fc = cv2.CascadeClassifier(p)
                faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
                if len(faces) > 0:
                    areas = [w*h for x,y,w,h in faces]
                    x,y,w,h = faces[np.argmax(areas)]
                    return (x, y, x+w, y+h)
    except: pass
    h, w = gray.shape
    return (int(w*0.25), int(h*0.1), int(w*0.75), int(h*0.85))

def get_eye_cascade():
    for p in [os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_eye.xml'),
              r'C:\Users\o\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_eye.xml']:
        if os.path.exists(p):
            return cv2.CascadeClassifier(p)
    return None

EYE_CASCADE = get_eye_cascade()

def is_eyes_open(gray):
    if EYE_CASCADE is None:
        h, w = gray.shape
        eye_r = gray[int(h*0.2):int(h*0.45), int(w*0.15):int(w*0.85)]
        return eye_r.std() > 15
    eyes = EYE_CASCADE.detectMultiScale(gray, 1.1, 3, minSize=(20,20))
    return len(eyes) >= 1

def apply_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


# ── Process ──────────────────────────────────────────────────
print('\n[3] Processing frames (v3: original pixels + occlusion filter)...')
df = pd.read_csv(os.path.join(BASE, 'data', 'clean_dataset_manifest.csv'))
stats = {'total': 0, 'kept': 0, 'eyes_closed': 0, 'occluded': 0, 'failed': 0, 'skipped': 0}

for _, row in tqdm(df.iterrows(), total=len(df), desc='SAM v3'):
    cat = row['category']; pid = row['patient_id']
    in_dir = os.path.join(VF, cat, pid)
    out_dir = os.path.join(OUTPUT, cat, pid)
    
    if not os.path.exists(in_dir):
        stats['skipped'] += 1; continue
    
    if os.path.exists(out_dir):
        existing = [f for f in os.listdir(out_dir) if '_face' in f]
        if len(existing) >= 4:
            stats['skipped'] += 1; continue
    
    os.makedirs(out_dir, exist_ok=True)
    
    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith('.jpg'): continue
        
        # Check occlusion
        if (pid, fname) in occluded_set:
            stats['occluded'] += 1; continue
        
        fpath = os.path.join(in_dir, fname)
        try:
            fb = np.fromfile(fpath, dtype=np.uint8)
            frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        except:
            stats['failed'] += 1; continue
        if frame is None:
            stats['failed'] += 1; continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats['total'] += 1
        
        # Check eyes open
        if not is_eyes_open(gray):
            stats['eyes_closed'] += 1; continue
        
        # Get face bbox
        bbox = get_face_bbox(gray)
        x1, y1, x2, y2 = bbox
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # SAM segmentation
        try:
            pil = Image.fromarray(frame_rgb)
            inputs = sam_processor(pil, input_boxes=[[[list(bbox)]]], return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = sam_model(**inputs)
            masks = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())
            mask = masks[0][0][0].numpy()
        except:
            mask = np.ones((h, w), dtype=bool)
        
        # Crop to SAM mask bbox — original pixels
        if mask.any():
            rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
            ys = np.where(rows)[0]; xs = np.where(cols)[0]
            rmin = max(0, ys[0]-5); rmax = min(h, ys[-1]+5)
            cmin = max(0, xs[0]-5); cmax = min(w, xs[-1]+5)
        else:
            rmin, rmax, cmin, cmax = y1, y2, x1, x2
        
        face_crop = frame_rgb[rmin:rmax, cmin:cmax].copy()
        
        # CLAHE
        face_clahe = apply_clahe(face_crop)
        
        # Resize
        face_224 = cv2.resize(face_clahe, (224, 224))
        
        # Sclera extraction
        fh, fw = face_clahe.shape[:2]
        eye_region = face_clahe[int(fh*0.20):int(fh*0.45), int(fw*0.10):int(fw*0.90)].copy()
        hsv_eye = cv2.cvtColor(eye_region, cv2.COLOR_RGB2HSV)
        sclera_mask = cv2.inRange(hsv_eye, (20,10,70), (60,60,100))
        white_m = cv2.inRange(hsv_eye, (0,0,180), (180,40,255))
        sclera_mask = cv2.bitwise_or(sclera_mask, white_m)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        sclera_mask = cv2.morphologyEx(sclera_mask, cv2.MORPH_OPEN, k3)
        sclera = cv2.bitwise_and(eye_region, eye_region, mask=sclera_mask)
        if sclera_mask.sum() < 50: sclera = eye_region
        sclera_224 = cv2.resize(sclera, (224, 224))
        
        # Save
        face_path = os.path.join(out_dir, fname.replace('.jpg', '_face.jpg'))
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(face_224, cv2.COLOR_RGB2BGR))
        if ok: buf.tofile(face_path)
        
        sclera_path = os.path.join(out_dir, fname.replace('.jpg', '_sclera.jpg'))
        ok2, buf2 = cv2.imencode('.jpg', cv2.cvtColor(sclera_224, cv2.COLOR_RGB2BGR))
        if ok2: buf2.tofile(sclera_path)
        
        stats['kept'] += 1

print(f'\n{"="*55}')
print(f'  SAM v3 Results (occlusion-filtered)')
print(f'{"="*55}')
print(f'  Total evaluated:    {stats["total"]}')
print(f'  Kept (clean+open):  {stats["kept"]}')
print(f'  Eyes closed:        {stats["eyes_closed"]}')
print(f'  Occluded (gauze):   {stats["occluded"]}')
print(f'  Failed:             {stats["failed"]}')
print(f'  Already done:       {stats["skipped"]}')

print(f'\n  Per-category:')
for cat in ['normal','mild','moderate','severe']:
    cd = os.path.join(OUTPUT, cat)
    if os.path.exists(cd):
        n = len(os.listdir(cd))
        files = sum(len(os.listdir(os.path.join(cd,d))) for d in os.listdir(cd))
        print(f'    {cat:12s}: {n} patients, {files} files')
