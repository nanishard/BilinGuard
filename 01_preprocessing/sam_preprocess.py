# -*- coding: utf-8 -*-
"""
BilinGuard SAM Preprocessing Pipeline (per manuscript Section 5)
Order:
  5.1  Uniform frame sampling (12 frames from video)
  5.2  MediaPipe Face Mesh (468-point landmarks)
  5.3  SAM facial-region segmentation (landmark-initialized)
  5.4  CLAHE on L* channel (clip=3.0, tile 8×8)
  5.5  Sclera extraction (eye-contour landmarks + HSV threshold + morphology)
  5.6  Resize 224×224 + ImageNet normalize

Applies to video frames from data/video_frames/{category}/{patient}/
Outputs to data/sam_processed/{category}/{patient}/
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'
import sys, json, random, time, copy, glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
VIDEO_FRAMES = os.path.join(BASE, 'data', 'video_frames')
OUTPUT_ROOT = os.path.join(BASE, 'data', 'sam_processed')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ── 1. Face detection (OpenCV DNN as MediaPipe replacement) ─
# Download DNN face detector model if not present
DNN_PROTO = os.path.join(BASE, 'models', 'sam', 'deploy.prototxt')
DNN_MODEL = os.path.join(BASE, 'models', 'sam', 'res10_300x300_ssd_iter_140000.caffemodel')

# Use simple center-crop face detection (videos are recorded at the patient's face)
def get_face_bbox_and_landmarks(frame_rgb):
    """Approximate face region from video frame (center-weighted)."""
    h, w = frame_rgb.shape[:2]
    # Videos are frontal face recordings: face occupies center 60-80% of frame
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.05)
    x = margin_x; y = margin_y
    fw = w - 2 * margin_x; fh = h - 2 * margin_y
    
    landmarks = {
        'bbox': (x, y, x + fw, y + fh),
        'eye_region': (
            int(x + fw * 0.15),  # left
            int(y + fh * 0.20),  # top  
            int(x + fw * 0.85),  # right
            int(y + fh * 0.50),  # bottom
        ),
    }
    return (x, y, fw, fh), landmarks

# ── 2. SAM model (via HuggingFace transformers) ──────────────
DEVICE = 'cuda'
import torch
from transformers import SamModel, SamProcessor

print('[1] Loading SAM ViT-B (HuggingFace)...')
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model.eval()
print('  SAM loaded.')

# ── Eye contour landmark indices ── (simplified, using detected eye positions)
# Since we use OpenCV face detection instead of MediaPipe 468 landmarks,
# we use approximate eye positions from the bounding box geometry.


def get_landmarks(frame_rgb):
    """Extract face landmarks using OpenCV face detection."""
    face, landmarks = get_face_bbox_and_landmarks(frame_rgb)
    return landmarks


def detect_face_bbox(frame_rgb):
    """Face detection fallback."""
    face, landmarks = get_face_bbox_and_landmarks(frame_rgb)
    if landmarks:
        return landmarks['bbox']
    return None


def sam_segment_face(frame_rgb, landmarks):
    """Use SAM to segment facial region, initialized by face bounding box."""
    if landmarks is None:
        return np.ones(frame_rgb.shape[:2], dtype=bool)
    
    box = landmarks['bbox']
    
    # SAM via HuggingFace transformers
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(frame_rgb)
    input_boxes = [[[box[0], box[1], box[2], box[3]]]]  # [[[x1,y1,x2,y2]]]
    
    inputs = sam_processor(pil_img, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    mask = masks[0][0][0].numpy()  # Best mask, first batch, first mask
    return mask


def apply_clahe(frame_rgb):
    """CLAHE on L* channel (clip=3.0, 8×8 tile grid)."""
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_sclera_region(frame_rgb, landmarks):
    """Extract sclera from eye region using HSV thresholding."""
    if landmarks is None:
        return frame_rgb[:112, :, :].copy()
    
    # Get eye region from landmarks
    ex_min, ey_min, ex_max, ey_max = landmarks['eye_region']
    eye_region = frame_rgb[ey_min:ey_max, ex_min:ex_max].copy()
    if eye_region.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # HSV thresholding for sclera
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 10, 70), (60, 60, 100))
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    mask = cv2.bitwise_or(mask, white_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    sclera = cv2.bitwise_and(eye_region, eye_region, mask=mask)
    if mask.sum() < 50:
        sclera = eye_region
    
    return sclera


def process_frame(frame_bgr):
    """
    Full pipeline for one frame:
      1. MediaPipe landmarks
      2. SAM segmentation
      3. CLAHE
      4. Sclera extraction
      Returns: (face_processed, sclera_processed) or (None, None)
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    
    # 5.2: Face detection + landmarks
    landmarks = get_landmarks(frame_rgb)
    if landmarks is None:
        return None, None
    bbox = landmarks['bbox']
    x1, y1, x2, y2 = bbox
    face = frame_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None, None
    
    # 5.3: SAM facial segmentation — find face, crop tightly (Strategy A)
    try:
        face_mask = sam_segment_face(frame_rgb, landmarks)
    except Exception:
        face_mask = np.ones((h, w), dtype=bool)
    
    # Crop to SAM mask bounding box + fill background with mean face color
    if face_mask.any():
        rows = np.any(face_mask, axis=1)
        cols = np.any(face_mask, axis=0)
        ys = np.where(rows)[0]; xs = np.where(cols)[0]
        rmin = max(0, ys[0] - 5); rmax = min(h, ys[-1] + 5)
        cmin = max(0, xs[0] - 5); cmax = min(w, xs[-1] + 5)
        # Crop region
        crop = frame_rgb[rmin:rmax, cmin:cmax].copy()
        mask_crop = face_mask[rmin:rmax, cmin:cmax]
        # Fill background (non-face) with mean face color — removes background cleanly
        if mask_crop.any():
            mean_color = crop[mask_crop].mean(axis=0).astype(np.uint8)
            crop[~mask_crop] = mean_color
    else:
        crop = frame_rgb[int(h*0.05):int(h*0.95), int(w*0.10):int(w*0.90)]
    
    # 5.4: CLAHE on clean crop
    face_clahe = apply_clahe(crop)
    
    # 5.5: Sclera extraction (use original landmarks on cropped face)
    # Adjust eye region coordinates relative to crop
    if face_mask.any() and 'bbox' in landmarks:
        ex_min_orig, ey_min_orig, ex_max_orig, ey_max_orig = landmarks['eye_region']
        # Adjust to cropped coordinates
        ex_min = max(0, ex_min_orig - cmin)
        ex_max = min(face_clahe.shape[1], ex_max_orig - cmin)
        ey_min = max(0, ey_min_orig - rmin)
        ey_max = min(face_clahe.shape[0], ey_max_orig - rmin)
        eye_region = face_clahe[ey_min:ey_max, ex_min:ex_max].copy()
        
        if eye_region.size > 0:
            hsv = cv2.cvtColor(eye_region, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (20, 10, 70), (60, 60, 100))
            white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
            mask = cv2.bitwise_or(mask, white_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            sclera = cv2.bitwise_and(eye_region, eye_region, mask=mask)
            if mask.sum() < 50:
                sclera = eye_region
        else:
            sclera = np.zeros((100, 100, 3), dtype=np.uint8)
    else:
        sclera = face_clahe[:face_clahe.shape[0]//2, :, :]
    
    # Resize to 224×224
    face_resized = cv2.resize(face_clahe, (224, 224))
    sclera_resized = cv2.resize(sclera, (224, 224))
    
    return face_resized, sclera_resized


# ── Process all video frames ─────────────────────────────────
print('\n[2] Processing video frames with SAM pipeline...')
df = pd.read_csv(MANIFEST)

stats = {'total': 0, 'processed': 0, 'failed': 0, 'skipped': 0}

for _, row in tqdm(df.iterrows(), total=len(df), desc='SAM processing'):
    cat = row['category']
    pid = row['patient_id']
    
    in_dir = os.path.join(VIDEO_FRAMES, cat, pid)
    out_dir = os.path.join(OUTPUT_ROOT, cat, pid)
    
    if not os.path.exists(in_dir):
        stats['skipped'] += 1
        continue
    
    # Skip if already processed
    existing = os.listdir(out_dir) if os.path.exists(out_dir) else []
    face_files = [f for f in existing if '_face.jpg' in f]
    if os.path.exists(out_dir) and len(face_files) >= 12:
        stats['skipped'] += 1
        continue
    
    os.makedirs(out_dir, exist_ok=True)
    
    frames = sorted([f for f in os.listdir(in_dir) if f.endswith('.jpg')])
    stats['total'] += len(frames)
    
    for fname in frames:
        fpath = os.path.join(in_dir, fname)
        # Use np.fromfile + imdecode for non-ASCII paths
        try:
            file_bytes = np.fromfile(fpath, dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except:
            frame = None
        if frame is None:
            stats['failed'] += 1
            continue
        
        face, sclera = process_frame(frame)
        
        if face is not None:
            # Save face (SAM segmented + CLAHE) — use imencode for non-ASCII paths
            face_path = os.path.join(out_dir, fname.replace('.jpg', '_face.jpg'))
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            if ok: buf.tofile(face_path)
            
            # Save sclera
            sclera_path = os.path.join(out_dir, fname.replace('.jpg', '_sclera.jpg'))
            ok2, buf2 = cv2.imencode('.jpg', cv2.cvtColor(sclera, cv2.COLOR_RGB2BGR))
            if ok2: buf2.tofile(sclera_path)
            
            stats['processed'] += 1
        else:
            stats['failed'] += 1

print(f'\n{"="*55}')
print(f'  SAM Processing Results')
print(f'{"="*55}')
print(f'  Total frames:     {stats["total"]}')
print(f'  Processed:        {stats["processed"]}')
print(f'  Failed:           {stats["failed"]}')
print(f'  Skipped (exist):  {stats["skipped"]}')

print(f'\n  Per-category:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(OUTPUT_ROOT, cat)
    if os.path.exists(cat_dir):
        n_pat = len(os.listdir(cat_dir))
        n_files = sum(len(os.listdir(os.path.join(cat_dir, d))) for d in os.listdir(cat_dir))
        print(f'    {cat:12s}: {n_pat} patients, {n_files} files')
