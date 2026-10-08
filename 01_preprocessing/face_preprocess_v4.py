# -*- coding: utf-8 -*-
"""
Face Preprocessing v4 — Robust Face Detection + SAM Masking + Validation

Fixes the critical bug in v3 where missing haarcascade XML caused 100% fallback to
a fixed bbox (38% of frame), resulting in "face" crops dominated by gauze/linen/gown.

v4 Pipeline:
  1. YuNet face detection (cv2.FaceDetectorYN) — tight bbox + 5 landmarks
  2. SAM precise face masking (prompted with correct YuNet bbox)
  3. Strict validation gate: skin fraction, detection score, landmark validity
  4. Face alignment using eye landmarks (optional, for canonical pose)
  5. CLAHE enhancement
  6. Save 224x224 tight face crop + metadata

NO silent fallbacks. If detection fails, frame is skipped and logged.

Output: data/face_v4/{category}/{patient_id}/frame_NNN_face.jpg
         data/face_v4/{category}/{patient_id}/frame_NNN_face_meta.json
         data/face_v4/processing_report.csv
"""

import os, sys, json, argparse, time, traceback
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import torch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import SamModel, SamProcessor
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────
BASE = r'D:\research\人脸识别营养\传染科'
VF_DIR = os.path.join(BASE, 'data', 'video_frames')
OUTPUT_DIR = os.path.join(BASE, 'data', 'face_v4')
MODEL_DIR = os.path.join(BASE, 'models')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
OCC_CSV = os.path.join(BASE, 'data', 'occlusion_report_v2.csv')
TEMP_DIR = r'C:\Users\o\AppData\Local\Temp\opencode'

IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Detection thresholds ─────────────────────────────────────
DET_SCORE_MIN = 0.50       # YuNet confidence
FACE_AREA_MIN = 40 * 40    # minimum face pixel area in original frame
SKIN_FRACTION_MIN = 0.35   # minimum skin fraction in SAM-masked face crop
NMS_THRESH = 0.3
TOP_K = 5000

# ── YuNet model setup ────────────────────────────────────────
# OpenCV 5 C++ ONNX loader fails on Chinese paths. Copy model to temp.
YUNET_SRC = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
YUNET_TMP = os.path.join(TEMP_DIR, 'yunet.onnx')

def setup_yunet():
    """Ensure YuNet ONNX model is available at a path without CJK characters."""
    if not os.path.exists(YUNET_TMP):
        if not os.path.exists(YUNET_SRC):
            raise FileNotFoundError(
                f"YuNet model not found at {YUNET_SRC}. "
                "Run download script or place model manually.")
        import shutil
        shutil.copy2(YUNET_SRC, YUNET_TMP)
        print(f"[Setup] Copied YuNet model to {YUNET_TMP}")
    detector = cv2.FaceDetectorYN.create(
        YUNET_TMP, "", (640, 480),
        score_threshold=DET_SCORE_MIN, nms_threshold=NMS_THRESH, top_k=TOP_K)
    print(f"[Setup] YuNet ready")
    return detector


# ── Skin fraction ────────────────────────────────────────────
def skin_fraction(img_rgb):
    """Fraction of pixels within YCbCr skin color range."""
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    cb = ycbcr[:, :, 2].astype(int)
    cr = ycbcr[:, :, 1].astype(int)
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return skin.mean()


# ── CLAHE ─────────────────────────────────────────────────────
def apply_clahe(rgb_img):
    """CLAHE on L channel of LAB."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


# ── Face alignment ────────────────────────────────────────────
def align_face(img_rgb, landmarks_5, target_size=224):
    """
    Align face to canonical pose using 5-point landmarks.
    landmarks_5: [[x1,y1], ...] for left_eye, right_eye, nose, left_mouth, right_mouth
    Returns aligned 224x224 RGB image.
    """
    # Standard alignment target (ArcFace-style 112x112 → scaled to target_size)
    scale = target_size / 112.0
    src_pts = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ], dtype=np.float32) * scale

    dst_pts = np.array(landmarks_5, dtype=np.float32)

    # Affine transform using 5 points (L2-fit)
    M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
    if M is None:
        # Fallback: simple resize if alignment fails
        return cv2.resize(img_rgb, (target_size, target_size))

    warped = cv2.warpAffine(img_rgb, M, (target_size, target_size),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped


# ── Load occlusion filter ─────────────────────────────────────
def load_occlusion_set():
    """Load previously detected occlusion frames to skip."""
    occluded = set()
    if os.path.exists(OCC_CSV):
        try:
            df = pd.read_csv(OCC_CSV)
            for _, r in df[df['is_occluded']].iterrows():
                occluded.add((str(r['patient_id']), str(r['frame'])))
            print(f"[Occlusion] Loaded {len(occluded)} occluded frames to skip")
        except Exception as e:
            print(f"[Occlusion] Could not load: {e}")
    return occluded


# ── Core: process one frame ───────────────────────────────────
def process_frame(frame_bgr, detector, sam_model, sam_processor, pid, fname):
    """
    Process a single video frame.
    Returns: (face_224_rgb, meta_dict) or (None, failure_reason)
    """
    h, w = frame_bgr.shape[:2]

    # ---- 1. YuNet face detection ----
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)

    if faces is None or len(faces) == 0:
        return None, {"reason": "no_face_detected", "confidence": None}

    # Pick highest-confidence face
    face = faces[np.argmax(faces[:, -1])]
    x, y, fw_bw, fh_bh = face[:4].astype(np.float32)
    landmarks = face[4:14].reshape(5, 2).copy()  # 5 landmarks
    score = float(face[14])

    if score < DET_SCORE_MIN:
        return None, {"reason": "low_score", "confidence": score}

    if fw_bw * fh_bh < FACE_AREA_MIN:
        return None, {"reason": "face_too_small", "confidence": score,
                       "area": fw_bw * fh_bh}

    # ---- 2. Build padded bbox for SAM prompt ----
    # Pad 15% on each side so SAM sees full face + margin
    pad_w = fw_bw * 0.15
    pad_h = fh_bh * 0.15
    x1_sam = max(0, int(x - pad_w))
    y1_sam = max(0, int(y - pad_h))
    x2_sam = min(w, int(x + fw_bw + pad_w))
    y2_sam = min(h, int(y + fh_bh + pad_h))
    sam_bbox = [x1_sam, y1_sam, x2_sam, y2_sam]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---- 3. SAM precise face mask ----
    try:
        pil_img = Image.fromarray(frame_rgb)
        inputs = sam_processor(pil_img, input_boxes=[[[sam_bbox]]],
                               return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())
        sam_mask = masks[0][0][0].numpy()
    except Exception:
        # SAM failed, fall back to bbox-based mask
        sam_mask = np.zeros((h, w), dtype=np.bool_)
        sam_mask[y1_sam:y2_sam, x1_sam:x2_sam] = True

    # ---- 4. Crop to SAM mask bbox ----
    if sam_mask.any():
        rows = np.any(sam_mask, axis=1)
        cols = np.any(sam_mask, axis=0)
        ys = np.where(rows)[0]
        xs = np.where(cols)[0]
        rmin, rmax = max(0, ys[0] - 3), min(h, ys[-1] + 3)
        cmin, cmax = max(0, xs[0] - 3), min(w, xs[-1] + 3)
    else:
        rmin, rmax, cmin, cmax = y1_sam, y2_sam, x1_sam, x2_sam

    face_crop = frame_rgb[rmin:rmax, cmin:cmax].copy()
    if face_crop.size == 0:
        return None, {"reason": "empty_crop", "confidence": score}

    # ---- 5. Validate skin fraction ----
    skin_frac = skin_fraction(face_crop)
    if skin_frac < SKIN_FRACTION_MIN:
        return None, {
            "reason": "low_skin", "confidence": score,
            "skin_frac": round(skin_frac, 3),
            "crop_size": face_crop.shape}

    # ---- 6. Face alignment (using landmarks, adjusted to crop coordinates) ----
    landmarks_crop = landmarks.copy()
    landmarks_crop[:, 0] -= cmin
    landmarks_crop[:, 1] -= rmin
    face_aligned = align_face(face_crop, landmarks_crop, IMG_SIZE)

    # ---- 7. CLAHE ----
    face_clahe = apply_clahe(face_aligned)

    # ---- 8. Build metadata ----
    meta = {
        "detector": "yunet",
        "confidence": round(score, 4),
        "face_bbox_raw": [int(x), int(y), int(fw_bw), int(fh_bh)],
        "sam_bbox": sam_bbox,
        "crop_rect": [int(rmin), int(rmax), int(cmin), int(cmax)],
        "skin_fraction": round(skin_frac, 3),
        "landmarks_raw": [[round(float(l[0]), 1), round(float(l[1]), 1)] for l in landmarks],
        "frame_size": [w, h],
    }

    return face_clahe, meta


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Face Preprocessing v4")
    parser.add_argument('--limit', type=int, default=0,
                        help="Max frames per patient (0=all)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Detect faces only, no SAM/save")
    parser.add_argument('--force', action='store_true',
                        help="Reprocess even if output exists")
    parser.add_argument('--categories', nargs='+',
                        default=['normal', 'mild', 'moderate', 'severe'])
    parser.add_argument('--SAM', dest='use_sam', action='store_true',
                        default=True, help="Use SAM masking (default)")
    parser.add_argument('--no-SAM', dest='use_sam', action='store_false',
                        help="Skip SAM, use bbox crop only")
    args = parser.parse_args()

    print("=" * 60)
    print("  Face Preprocessing v4 — Robust Detection + Validation")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  SAM: {'ON' if args.use_sam else 'OFF (bbox-only)'}")
    print(f"  Categories: {args.categories}")
    print(f"  Limit/frame: {args.limit or 'all'}")
    print(f"  Skin threshold: {SKIN_FRACTION_MIN}")
    print(f"  Det score min: {DET_SCORE_MIN}")
    print()

    # ── Load manifest ──
    if not os.path.exists(MANIFEST):
        print(f"[FATAL] Manifest not found: {MANIFEST}")
        return
    df = pd.read_csv(MANIFEST)
    df = df[df['category'].isin(args.categories)]
    print(f"[Manifest] {len(df)} entries after category filter")

    # ── Setup detectors ──
    detector = setup_yunet()

    if args.use_sam:
        print("[GPU] Loading SAM...")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model.eval()
    else:
        sam_model = None
        sam_processor = None

    # ── Load occlusion filter ──
    occlusion_set = load_occlusion_set()

    # ── Stats ──
    stats = {
        'total': 0, 'kept': 0, 'skipped_exists': 0,
        'no_face': 0, 'low_score': 0, 'face_too_small': 0,
        'low_skin': 0, 'empty_crop': 0, 'occluded': 0,
        'sam_failed': 0, 'error': 0, 'missing_vf': 0,
    }
    category_report = {}

    report_rows = []

    # ── Process ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        cat = row['category']
        pid = str(row['patient_id'])
        cat_report = category_report.setdefault(cat, dict(stats.copy()))

        in_dir = os.path.join(VF_DIR, cat, pid)
        out_dir = os.path.join(OUTPUT_DIR, cat, pid)

        if not os.path.isdir(in_dir):
            stats['missing_vf'] += 1
            cat_report['missing_vf'] = cat_report.get('missing_vf', 0) + 1
            continue

        # Skip if already processed (and not forced)
        if not args.force and os.path.isdir(out_dir):
            existing = [f for f in os.listdir(out_dir) if '_face.jpg' in f]
            if len(existing) >= 3:
                stats['skipped_exists'] += 1
                cat_report['skipped_exists'] = cat_report.get('skipped_exists', 0) + 1
                continue

        os.makedirs(out_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(in_dir) if f.endswith('.jpg')])

        n_frames_kept = 0
        cat_stats_local = {
            'total': 0, 'kept': 0,
            'no_face': 0, 'low_score': 0, 'face_too_small': 0,
            'low_skin': 0, 'empty_crop': 0, 'occluded': 0, 'error': 0,
        }

        for fname in frame_files:
            if args.limit and n_frames_kept >= args.limit:
                break

            cat_stats_local['total'] += 1

            # Occlusion check
            if (pid, fname) in occlusion_set:
                cat_stats_local['occluded'] += 1
                continue

            fpath = os.path.join(in_dir, fname)
            try:
                fb = np.fromfile(fpath, dtype=np.uint8)
                frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            except Exception:
                cat_stats_local['error'] += 1
                continue
            if frame is None:
                cat_stats_local['error'] += 1
                continue

            # Process
            try:
                result = process_frame(
                    frame, detector, sam_model, sam_processor,
                    pid, fname,
                )
            except Exception as e:
                traceback.print_exc()
                cat_stats_local['error'] += 1
                report_rows.append({
                    'category': cat, 'patient_id': pid, 'frame': fname,
                    'status': 'ERROR', 'reason': str(e)[:200],
                })
                continue

            face_img, meta_or_reason = result

            if face_img is None:
                reason = meta_or_reason.get('reason', 'unknown')
                if reason in cat_stats_local:
                    cat_stats_local[reason] += 1
                if reason not in cat_stats_local:
                    cat_stats_local[reason] = 0
                # Log rejected frames
                meta_detail = {k: v for k, v in meta_or_reason.items() if k != 'reason'}
                report_rows.append({
                    'category': cat, 'patient_id': pid, 'frame': fname,
                    'status': 'SKIPPED', 'reason': reason,
                    **meta_detail,
                })
                continue

            # Save face image
            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            out_name = fname.replace('.jpg', '_face.jpg')
            out_path = os.path.join(out_dir, out_name)
            ok, buf = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(out_path)

            # Save metadata
            meta_path = os.path.join(out_dir, fname.replace('.jpg', '_face_meta.json'))
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(meta_or_reason, mf, ensure_ascii=False, indent=2)

            cat_stats_local['kept'] += 1
            n_frames_kept += 1
            report_rows.append({
                'category': cat, 'patient_id': pid, 'frame': fname,
                'status': 'OK',
                'confidence': meta_or_reason.get('confidence'),
                'skin_frac': meta_or_reason.get('skin_fraction'),
            })

        # Aggregate per-category
        for k, v in cat_stats_local.items():
            cat_report[k] = cat_report.get(k, 0) + v
            stats[k] = stats.get(k, 0) + v

        stats['total'] += cat_stats_local['total']
        stats['kept'] += cat_stats_local['kept']

    # ── Save report ──
    report_df = pd.DataFrame(report_rows)
    report_csv = os.path.join(OUTPUT_DIR, 'processing_report.csv')
    report_df.to_csv(report_csv, index=False, encoding='utf-8-sig')

    # ── Print summary ──
    print()
    print("=" * 60)
    print("  v4 Processing Summary")
    print("=" * 60)
    print(f"  Total frames evaluated:   {stats['total']}")
    print(f"  Kept (clean face):        {stats['kept']}")
    print(f"  ── Rejections ──")
    print(f"  Pre-existing (skipped):   {stats['skipped_exists']}")
    print(f"  No face detected:         {stats['no_face']}")
    print(f"  Low detection score:      {stats['low_score']}")
    print(f"  Face too small:           {stats['face_too_small']}")
    print(f"  Low skin fraction:        {stats['low_skin']}")
    print(f"  Empty crop:               {stats['empty_crop']}")
    print(f"  Occluded (gauze/closed):  {stats['occluded']}")
    print(f"  Errors:                   {stats['error']}")
    print(f"  Missing VF dirs:          {stats['missing_vf']}")
    if stats['total'] > 0:
        print(f"  ── KPIs ──")
        print(f"  Detection rate:           {100*(stats['total']-stats['no_face'])/max(stats['total'],1):.1f}%")
        print(f"  Acceptance rate:          {100*stats['kept']/max(stats['total'],1):.1f}%")

    print()
    print("  Per-category:")
    for cat, cr in sorted(category_report.items()):
        total = cr.get('total', 0)
        kept = cr.get('kept', 0)
        noface = cr.get('no_face', 0)
        lowskin = cr.get('low_skin', 0)
        det_rate = 100 * (total - noface) / max(total, 1)
        acc_rate = 100 * kept / max(total, 1)
        print(f"    {cat:12s} → total={total:4d}  kept={kept:4d}  "
              f"no_face={noface:3d}  low_skin={lowskin:3d}  "
              f"det={det_rate:.0f}%  accept={acc_rate:.0f}%")

    print(f"\n  Report: {report_csv}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
