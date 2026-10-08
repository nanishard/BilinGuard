# -*- coding: utf-8 -*-
"""
SAM Processing v4 (color) — breaks the lighting/environment confound.

Pipeline:
  1. Read original video frame
  2. Skip if eyes closed or occluded (reuse v3 occlusion report)
  3. SAM face segmentation -> tight crop (original pixels)
  4. Grey World on SAM-masked face pixels (color-temperature correction)
  5. CLAHE (lightness only)
  6. Skin detection -> skin mean in Lab
  7. Relative-color map: pixel_Lab - skin_Lab_mean + 128  (lighting-invariant)
  8. Sclera extraction (from Grey-World + CLAHE image)
  9. Save 224x224: _face (GW+CLAHE), _relface (skin-referenced), _sclera

Rationale (from diagnose_sam_lighting.py / diagnose_root_cause.py):
  - Normal vs jaundice differ mainly by ENVIRONMENT (background brightness),
    not just color temperature. So Grey World alone is insufficient.
  - Within-image relative color (sclera-skin) is invariant to illumination and
    camera; it is the feature with the best chance of carrying real signal.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import cv2, numpy as np, pandas as pd, torch, json
from PIL import Image
from transformers import SamModel, SamProcessor
from tqdm import tqdm

BASE = r'D:\research\人脸识别营养\传染科'
VF = os.path.join(BASE, 'data', 'video_frames')
OUTPUT = os.path.join(BASE, 'data', 'sam_processed_color')
OCC_CSV = os.path.join(BASE, 'data', 'occlusion_report_v2.csv')
os.makedirs(OUTPUT, exist_ok=True)

# ── Occlusion report ─────────────────────────────────────────
occ_df = pd.read_csv(OCC_CSV)
occluded_set = set()
for _, r in occ_df[occ_df['is_occluded']].iterrows():
    occluded_set.add((r['patient_id'], r['frame']))
print(f'[1] Occlusion report: {len(occluded_set)} occluded pairs will be skipped')

# ── SAM ──────────────────────────────────────────────────────
print('[2] Loading SAM...')
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to('cuda')
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model.eval()


def _cascade_path(name):
    cands = [
        os.path.join(os.path.dirname(cv2.__file__), 'data', name),
        r'C:\Users\o\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data',
    ]
    cands[1] = os.path.join(cands[1], name)
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def get_face_bbox(gray):
    p = _cascade_path('haarcascade_frontalface_default.xml')
    if p:
        fc = cv2.CascadeClassifier(p)
        faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) > 0:
            areas = [w * h for x, y, w, h in faces]
            x, y, w, h = faces[np.argmax(areas)]
            return (x, y, x + w, y + h)
    h, w = gray.shape
    return (int(w * 0.25), int(h * 0.1), int(w * 0.75), int(h * 0.85))


EYE_CASCADE_PATH = _cascade_path('haarcascade_eye.xml')
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH) if EYE_CASCADE_PATH else None


def is_eyes_open(gray):
    if EYE_CASCADE is None:
        h, w = gray.shape
        eye_r = gray[int(h * 0.2):int(h * 0.45), int(w * 0.15):int(w * 0.85)]
        return eye_r.std() > 15
    eyes = EYE_CASCADE.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
    return len(eyes) >= 1


def grey_world_masked(rgb, mask):
    """Grey World using only masked (face) pixels: scale channels so
    masked-mean R = G = B. Removes illumination chromaticity (color temp)."""
    m = mask & (mask.sum(-1) > 0) if mask.ndim == 3 else mask
    fg = rgb[m]
    if len(fg) < 50:
        # fall back to whole image
        fg = rgb.reshape(-1, 3)
    mr, mg, mb = fg[:, 0].mean(), fg[:, 1].mean(), fg[:, 2].mean()
    gray = (mr + mg + mb) / 3.0
    out = rgb.astype(np.float32)
    out[:, :, 0] *= gray / (mr + 1e-6)
    out[:, :, 1] *= gray / (mg + 1e-6)
    out[:, :, 2] *= gray / (mb + 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_rgb(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def skin_mask_hsv(rgb):
    """Skin pixels in HSV. Used to estimate per-image skin reference color."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (h > 5) & (h < 25) & (s > 30) & (s < 150) & (v > 60)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return m.astype(bool)


def relative_color_map(rgb):
    """Skin-referenced relative-color map in Lab.
    Each pixel = pixel_Lab - skin_Lab_mean + 128  -> skin becomes neutral (128),
    sclera deviation (yellowness) is preserved, illumination is factored out."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    sm = skin_mask_hsv(rgb)
    if sm.sum() < 100:
        # whole-face fallback
        sm = np.ones(rgb.shape[:2], dtype=bool)
    skin_mean = lab[sm].mean(axis=0)  # (L, a, b)
    rel = lab - skin_mean + 128.0
    rel = np.clip(rel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rel, cv2.COLOR_LAB2RGB)


def extract_sclera(rgb):
    """Sclera extraction from the eye region (GW+CLAHE corrected)."""
    h, w = rgb.shape[:2]
    eye_region = rgb[int(h * 0.20):int(h * 0.45), int(w * 0.10):int(w * 0.90)].copy()
    hsv = cv2.cvtColor(eye_region, cv2.COLOR_RGB2HSV)
    sclera_mask = cv2.inRange(hsv, (20, 10, 70), (60, 60, 100))
    white_m = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    sclera_mask = cv2.bitwise_or(sclera_mask, white_m)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sclera_mask = cv2.morphologyEx(sclera_mask, cv2.MORPH_OPEN, k3)
    sclera = cv2.bitwise_and(eye_region, eye_region, mask=sclera_mask)
    if sclera_mask.sum() < 50:
        sclera = eye_region
    return cv2.resize(sclera, (224, 224))


def save_jpg(rgb_arr, path):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
    if ok:
        buf.tofile(path)


# ── Process ──────────────────────────────────────────────────
print('\n[3] Processing frames (v4: Grey World + relative color)...')
df = pd.read_csv(os.path.join(BASE, 'data', 'clean_dataset_manifest.csv'))
stats = {'total': 0, 'kept': 0, 'eyes_closed': 0, 'occluded': 0,
         'failed': 0, 'skipped': 0}

for _, row in tqdm(df.iterrows(), total=len(df), desc='SAM v4 color'):
    cat = row['category']; pid = row['patient_id']
    in_dir = os.path.join(VF, cat, pid)
    out_dir = os.path.join(OUTPUT, cat, pid)

    if not os.path.exists(in_dir):
        stats['skipped'] += 1; continue

    # resume support: skip if already has >=4 face outputs
    if os.path.exists(out_dir):
        existing = [f for f in os.listdir(out_dir) if '_face' in f]
        if len(existing) >= 4:
            stats['skipped'] += 1; continue

    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith('.jpg'):
            continue
        if (pid, fname) in occluded_set:
            stats['occluded'] += 1; continue

        fpath = os.path.join(in_dir, fname)
        try:
            fb = np.fromfile(fpath, dtype=np.uint8)
            frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        except Exception:
            stats['failed'] += 1; continue
        if frame is None:
            stats['failed'] += 1; continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats['total'] += 1

        if not is_eyes_open(gray):
            stats['eyes_closed'] += 1; continue

        bbox = get_face_bbox(gray)
        x1, y1, x2, y2 = bbox
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # SAM segmentation
        try:
            pil = Image.fromarray(frame_rgb)
            inputs = sam_processor(pil, input_boxes=[[[list(bbox)]]],
                                   return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = sam_model(**inputs)
            masks = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())
            mask = masks[0][0][0].numpy()
        except Exception:
            mask = np.ones((h, w), dtype=bool)

        # tight crop to SAM mask bbox
        if mask.any():
            rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
            ys = np.where(rows)[0]; xs = np.where(cols)[0]
            rmin = max(0, ys[0] - 5); rmax = min(h, ys[-1] + 5)
            cmin = max(0, xs[0] - 5); cmax = min(w, xs[-1] + 5)
        else:
            rmin, rmax, cmin, cmax = y1, y2, x1, x2

        face_crop = frame_rgb[rmin:rmax, cmin:cmax].copy()
        face_mask = mask[rmin:rmax, cmin:cmax].copy()

        # [NEW] Grey World on masked face pixels (color-temperature correction)
        gw = grey_world_masked(face_crop, face_mask)

        # CLAHE (lightness only)
        face_clahe = clahe_rgb(gw)

        # [NEW] Relative-color map (skin-referenced, illumination-invariant)
        rel_map = relative_color_map(face_clahe)

        # resize to 224
        face_224 = cv2.resize(face_clahe, (224, 224))
        rel_224 = cv2.resize(rel_map, (224, 224))

        # sclera (from GW+CLAHE)
        sclera_224 = extract_sclera(face_clahe)

        # save
        base = fname.replace('.jpg', '')
        save_jpg(face_224, os.path.join(out_dir, f'{base}_face.jpg'))
        save_jpg(rel_224, os.path.join(out_dir, f'{base}_relface.jpg'))
        save_jpg(sclera_224, os.path.join(out_dir, f'{base}_sclera.jpg'))

        stats['kept'] += 1

# ── Summary ──────────────────────────────────────────────────
print(f'\n{"=" * 55}')
print(f'  SAM v4 (color) Results')
print(f'{"=" * 55}')
for k, v in stats.items():
    print(f'  {k:14s}: {v}')

print(f'\n  Per-category:')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(OUTPUT, cat)
    if os.path.exists(cd):
        n = len(os.listdir(cd))
        files = sum(len(os.listdir(os.path.join(cd, d)))
                    for d in os.listdir(cd) if os.path.isdir(os.path.join(cd, d)))
        print(f'    {cat:12s}: {n} patients, {files} files')
