# -*- coding: utf-8 -*-
"""
Occlusion Detection Module
Detects and masks facial occlusions (gauze, bandages, masks, hands, etc.)
before feeding images to the classification model.

Approach:
  1. Skin-tone detection: identify pixels that match human skin color
  2. Non-skin patches in face region = occlusion candidates
  3. Large connected non-skin regions within face bbox = occlusion
  4. Frames with >30% occlusion in face region are flagged

This module runs AFTER SAM face crop, BEFORE CLAHE/classification.
Occluded pixels are flagged but NOT removed from training.
Instead, the occlusion mask is passed to the model so it can
learn to ignore occluded regions.
"""
import cv2
import numpy as np


def detect_skin_mask(face_rgb):
    """
    Detect skin-colored pixels in a face image.
    Tuned for Asian skin tones in ward lighting conditions.
    """
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCR_CB)
    
    # HSV skin rule — relaxed for CLAHE-processed images
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    hsv_mask = (h >= 0) & (h <= 60) & (s >= 5) & (s <= 200) & (v >= 50)
    
    # YCbCr skin rule — widened Cr/Cb range
    y_ch, cr, cb = ycrcb[:,:,0], ycrcb[:,:,1], ycrcb[:,:,2]
    ycbcr_mask = (cr >= 125) & (cr <= 180) & (cb >= 70) & (cb <= 140) & (y_ch >= 30)
    
    # Combine — use either rule (OR) for inclusiveness
    skin_mask = (hsv_mask | ycbcr_mask).astype(np.uint8) * 255
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    # Fill holes
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(skin_mask, contours, -1, 255, -1)
    
    return skin_mask


def detect_occlusion_mask(face_rgb):
    """
    Detect occlusions (gauze, bandages, masks, medical tape, etc.)
    Strategy: focus on WHITE BRIGHT patches (gauze is distinctly white)
    Returns: occlusion_mask (255=occluded, 0=clear) and occlusion_ratio
    """
    h, w = face_rgb.shape[:2]
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    
    occlusion_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Detect bright white patches (gauze, medical tape)
    #    Gauze: V > 190, S < 40 (very white, low saturation)
    white_mask = cv2.inRange(hsv, (0, 0, 190), (180, 40, 255))
    # Exclude eye sclera (which is also white but small)
    # Only large white patches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    # 2. Detect colored non-skin patches (green/blue medical tape, bandages)
    skin = detect_skin_mask(face_rgb)
    non_skin = (skin == 0)
    # Exclude dark background (already mostly removed by SAM crop)
    non_skin[(gray < 30)] = False
    # Exclude eyes/mouth area (natural non-skin)
    # Check for colored (high saturation) non-skin = medical items
    colored_mask = cv2.inRange(hsv, (35, 50, 50), (180, 255, 255))
    medical_colored = non_skin & (colored_mask > 0)
    
    # 3. Combine
    candidate_mask = cv2.bitwise_or(white_mask, (medical_colored.astype(np.uint8)) * 255)
    
    # 4. Filter by size: only large connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    face_area = h * w
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > face_area * 0.02:  # >2% of face
            occlusion_mask[labels == i] = 255
    
    # Clean up
    occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    occlusion_ratio = occlusion_mask.sum() / (face_area * 255)
    return occlusion_mask, occlusion_ratio


def classify_frame_quality(face_rgb):
    """
    Classify a face frame's quality.
    Returns dict with:
      - 'occlusion_ratio': float (0-1)
      - 'is_heavily_occluded': bool (>30% occluded)
      - 'usable': bool (can be used for training)
      - 'occlusion_type': str ('none', 'gauze', 'mask', 'hand', 'unknown')
    """
    occ_mask, occ_ratio = detect_occlusion_mask(face_rgb)
    
    # Determine type
    if occ_ratio < 0.05:
        occ_type = 'none'
    else:
        # Check occlusion location for type estimation
        h, w = face_rgb.shape[:2]
        if occ_mask[int(h*0.4):int(h*0.7), int(w*0.2):int(w*0.8)].sum() > occ_mask.sum() * 0.5:
            occ_type = 'mask_or_lower_face'  # Lower face occlusion
        elif occ_mask[int(h*0.1):int(h*0.4), :].sum() > occ_mask.sum() * 0.5:
            occ_type = 'gauze_or_forehead'  # Forehead/upper face
        else:
            occ_type = 'patch'
    
    return {
        'occlusion_ratio': occ_ratio,
        'is_heavily_occluded': occ_ratio > 0.30,
        'usable': occ_ratio < 0.30,
        'occlusion_type': occ_type,
        'occlusion_mask': occ_mask,
    }


def visualize_occlusion(face_rgb, occ_mask, occ_ratio):
    """Create visualization with occlusion highlighted in red."""
    vis = face_rgb.copy()
    vis[occ_mask > 0] = [255, 50, 50]  # Red overlay
    # Add text
    cv2.putText(vis, f'Occlusion: {occ_ratio:.1%}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


# ── Test on known cases ──────────────────────────────────────
if __name__ == '__main__':
    import os
    
    BASE = r'D:\research\人脸识别营养\传染科'
    OUT = os.path.join(BASE, 'data', 'occlusion_test')
    os.makedirs(OUT, exist_ok=True)
    
    # Test on specific patients
    test_cases = [
        ('severe', '18严美君0035460805'),  # Known gauze case
        ('severe', '139李平源0038278210'),  # Normal case
        ('normal', '102徐永洪0038127495'),  # Normal
        ('mild', '100王兴元0038117813'),    # Mild
    ]
    
    for cat, pid in test_cases:
        v2_dir = os.path.join(BASE, 'data', 'sam_processed_v2', cat, pid)
        if not os.path.exists(v2_dir):
            print(f'{cat}/{pid}: NOT FOUND')
            continue
        
        faces = sorted([f for f in os.listdir(v2_dir) if '_face' in f])
        print(f'\n{cat}/{pid}: {len(faces)} frames')
        
        for ff in faces[:4]:
            fp = os.path.join(v2_dir, ff)
            fb = np.fromfile(fp, dtype=np.uint8)
            img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if img is None: continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            result = classify_frame_quality(rgb)
            
            status = 'USABLE' if result['usable'] else 'OCCLUDED'
            print(f'  {ff}: {status} ratio={result["occlusion_ratio"]:.1%} type={result["occlusion_type"]}')
            
            # Save visualization
            vis = visualize_occlusion(rgb, result['occlusion_mask'], result['occlusion_ratio'])
            vis_path = os.path.join(OUT, f'{cat}_{pid[:10]}_{ff}')
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if ok: buf.tofile(vis_path)
    
    # Scan all severe patients for occlusion stats
    print(f'\n{"="*55}')
    print('Scanning all severe patients for occlusion...')
    print('='*55)
    
    severe_dir = os.path.join(BASE, 'data', 'sam_processed_v2', 'severe')
    if os.path.exists(severe_dir):
        occluded_count = 0
        total_frames = 0
        for d in sorted(os.listdir(severe_dir)):
            pd = os.path.join(severe_dir, d)
            faces = [f for f in os.listdir(pd) if '_face' in f]
            for ff in faces:
                fp = os.path.join(pd, ff)
                fb = np.fromfile(fp, dtype=np.uint8)
                img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                if img is None: continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = classify_frame_quality(rgb)
                total_frames += 1
                if not result['usable']:
                    occluded_count += 1
        
        print(f'  Severe: {occluded_count}/{total_frames} frames heavily occluded ({occluded_count/total_frames*100:.1f}%)')
    
    print(f'\n  Visualization saved to: {OUT}/')
