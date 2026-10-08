"""Verify SAM processing was actually applied to new external normal patients.

Checks:
1. Each new patient has _face.jpg files in external_processed_v3/normal/
2. Face images are 224x224 (SAM crop + resize signature)
3. Face images differ from raw frames (not just copied)
4. Compare a raw frame vs its SAM-processed counterpart
"""
import os, re, numpy as np, cv2
from collections import Counter

BASE = r'D:\research\人脸识别营养\传染科'
NEW_DIR = os.path.join(BASE, 'external_validation_results',
                        'external_validation_results', '外部验证正常人')
FRAMES_DIR = os.path.join(BASE, 'data', 'external_frames', 'normal')
PROCESSED_DIR = os.path.join(BASE, 'data', 'external_processed_v3', 'normal')

# Build set of new patient PIDs
new_pids = set()
for f in os.listdir(NEW_DIR):
    if f.lower().endswith('.zip'):
        new_pids.add(os.path.splitext(f)[0])

print(f'New patient PIDs: {len(new_pids)}')
print(f'='*70)

# Check each new patient in BOTH directories
results = []
missing_processed = []
missing_face = []
wrong_size = []

new_in_frames = [d for d in os.listdir(FRAMES_DIR) if d in new_pids]
print(f'New patients in frames dir: {len(new_in_frames)}')
print(f'New patients in processed dir: '
      f'{len([d for d in os.listdir(PROCESSED_DIR) if d in new_pids])}')

for pid in sorted(new_in_frames):
    frames_p = os.path.join(FRAMES_DIR, pid)
    proc_p = os.path.join(PROCESSED_DIR, pid)
    
    # Check processed directory exists
    if not os.path.exists(proc_p):
        missing_processed.append(pid)
        results.append({'pid': pid, 'has_processed': False, 'n_face': 0})
        continue
    
    # Check face files
    face_files = sorted([f for f in os.listdir(proc_p) if '_face' in f and f.endswith('.jpg')])
    if not face_files:
        missing_face.append(pid)
        results.append({'pid': pid, 'has_processed': True, 'n_face': 0})
        continue
    
    # Check first face image size (should be 224x224)
    first_face = os.path.join(proc_p, face_files[0])
    try:
        fb = np.fromfile(first_face, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
    except Exception:
        h, w = 0, 0
    
    # Check if face differs from raw frame
    raw_frames = sorted([f for f in os.listdir(frames_p) if f.endswith('.jpg')])
    raw_diff = None
    face_diff_from_raw = None
    if raw_frames and h == 224:
        raw_path = os.path.join(frames_p, raw_frames[0])
        try:
            rb = np.fromfile(raw_path, dtype=np.uint8)
            raw_img = cv2.imdecode(rb, cv2.IMREAD_COLOR)
            rh, rw = raw_img.shape[:2]
            raw_diff = (rh, rw)
            # Compare: face should be a CROP (different content from full raw frame)
            face_resized = cv2.resize(img, (rw, rh))
            diff = np.mean(np.abs(face_resized.astype(float) - raw_img.astype(float)))
            face_diff_from_raw = diff
        except Exception:
            pass
    
    is_correct = (h == 224 and w == 224)
    if not is_correct:
        wrong_size.append((pid, f'{w}x{h}'))
    
    results.append({'pid': pid, 'has_processed': True, 'n_face': len(face_files),
                     'face_size': (w, h), 'raw_size': raw_diff,
                     'face_vs_raw_diff': face_diff_from_raw, 'correct': is_correct})

# Summary
print(f'\n{"="*70}')
print(f'  SAM PROCESSING VERIFICATION ({len(results)} patients)')
print(f'{"="*70}')

n_with_processed = sum(1 for r in results if r['has_processed'])
n_with_face = sum(1 for r in results if r.get('n_face', 0) > 0)
n_correct_size = sum(1 for r in results if r.get('correct', False))

print(f'  Has processed directory: {n_with_processed}/{len(results)}')
print(f'  Has _face.jpg files:     {n_with_face}/{len(results)}')
print(f'  Face size = 224x224:     {n_correct_size}/{len(results)}')

# Size distribution
sizes = Counter(r.get('face_size') for r in results if r.get('face_size'))
print(f'\n  Face image size distribution:')
for sz, c in sizes.most_common():
    print(f'    {sz}: {c} patients')

# Frame-vs-face difference (proof of SAM cropping)
diffs = [r['face_vs_raw_diff'] for r in results if r.get('face_vs_raw_diff') is not None]
if diffs:
    print(f'\n  Face-vs-raw-frame pixel difference (proof of SAM crop, not copy):')
    print(f'    Mean: {np.mean(diffs):.2f}')
    print(f'    Min:  {min(diffs):.2f}')
    print(f'    Max:  {max(diffs):.2f}')
    print(f'    (High value = face was cropped/segmented, not just copied)')

# Raw frame sizes (should be various video resolutions)
raw_sizes = Counter(r.get('raw_size') for r in results if r.get('raw_size'))
print(f'\n  Raw frame size distribution (video resolutions):')
for sz, c in raw_sizes.most_common(5):
    print(f'    {sz}: {c} patients')

# Errors
if missing_processed:
    print(f'\n  MISSING PROCESSED DIR ({len(missing_processed)}):')
    for p in missing_processed[:5]:
        print(f'    {p[:50]}')
if missing_face:
    print(f'\n  MISSING FACE FILES ({len(missing_face)}):')
    for p in missing_face[:5]:
        print(f'    {p[:50]}')
if wrong_size:
    print(f'\n  WRONG SIZE ({len(wrong_size)}):')
    for p, s in wrong_size[:5]:
        print(f'    {p[:40]} {s}')

if not missing_processed and not missing_face and not wrong_size:
    print(f'\n  [OK] ALL {len(results)} patients correctly SAM-processed (224x224 face images)')

# Sample detail
print(f'\n{"="*70}')
print(f'  SAMPLE PATIENT DETAIL (first 3)')
print(f'{"="*70}')
for r in results[:3]:
    pid = r['pid']
    print(f'\n  {pid[:45]}:')
    print(f'    Raw frames:   {r.get("raw_size", "N/A")}')
    print(f'    Processed:    {"YES" if r["has_processed"] else "NO"}')
    print(f'    Face files:   {r.get("n_face", 0)}')
    print(f'    Face size:    {r.get("face_size", "N/A")}')
    print(f'    Face-vs-raw:  {r.get("face_vs_raw_diff", "N/A"):.2f}' if r.get('face_vs_raw_diff') else f'    Face-vs-raw:  N/A')
