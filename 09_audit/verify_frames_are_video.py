"""Verify that extracted frames are real VIDEO frames (not duplicates of screenshots).

Checks:
1. Frame file sizes — video frames should vary; duplicate screenshots would be identical
2. Image hash uniqueness — 12 frames from a video should all be different
3. Per-pixel difference between consecutive frames — video frames have motion
"""
import os, re, hashlib, numpy as np, cv2
from collections import Counter

BASE = r'D:\research\人脸识别营养\传染科'
NEW_DIR = os.path.join(BASE, 'external_validation_results',
                        'external_validation_results', '外部验证正常人')
FRAMES_DIR = os.path.join(BASE, 'data', 'external_frames', 'normal')

# Build set of new patient PIDs (from zip filenames)
new_pids = set()
for f in os.listdir(NEW_DIR):
    if f.lower().endswith('.zip'):
        new_pids.add(os.path.splitext(f)[0])

print(f'New patient PIDs (from zips): {len(new_pids)}')
print(f'Frames directory: {FRAMES_DIR}')
print(f'='*70)

def img_hash(path):
    """Perceptual hash (downsampled to 16x16 grayscale)."""
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.resize(img, (16, 16))
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None

def img_array(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(fb, cv2.IMREAD_COLOR)
    except Exception:
        return None

# Check each new patient
all_results = []
problem_pids = []

new_in_frames = [d for d in os.listdir(FRAMES_DIR) if d in new_pids]
print(f'New patients found in frames dir: {len(new_in_frames)}\n')

for pid in sorted(new_in_frames):  # Check ALL patients
    pdir = os.path.join(FRAMES_DIR, pid)
    frames = sorted([f for f in os.listdir(pdir) if f.endswith('.jpg')])
    if len(frames) < 2:
        problem_pids.append((pid, f'only {len(frames)} frames'))
        continue
    
    # 1. File sizes
    sizes = [os.path.getsize(os.path.join(pdir, f)) for f in frames]
    size_unique = len(set(sizes))
    
    # 2. Image hashes
    hashes = [img_hash(os.path.join(pdir, f)) for f in frames]
    hash_unique = len(set(h for h in hashes if h))
    
    # 3. Pixel difference between consecutive frames
    arrays = [img_array(os.path.join(pdir, f)) for f in frames]
    diffs = []
    for i in range(1, len(arrays)):
        if arrays[i] is not None and arrays[i-1] is not None:
            d = np.mean(np.abs(arrays[i].astype(float) - arrays[i-1].astype(float)))
            diffs.append(d)
    mean_diff = np.mean(diffs) if diffs else 0
    
    # Classify
    is_video = (hash_unique >= 10 and mean_diff > 2.0)  # Video: many unique frames, real motion
    is_screenshot_dup = (hash_unique <= 3)  # Screenshots: few unique images repeated
    
    status = 'VIDEO' if is_video else ('SCREENSHOT-DUP' if is_screenshot_dup else 'UNCLEAR')
    if status != 'VIDEO':
        problem_pids.append((pid, f'{status}: {hash_unique} unique hashes, mean_diff={mean_diff:.1f}'))
    
    all_results.append({'pid': pid, 'n_frames': len(frames), 'size_unique': size_unique,
                         'hash_unique': hash_unique, 'mean_diff': mean_diff, 'status': status})
    if status != 'VIDEO':
        print(f'  WARN {pid[:35]:37s} [{status:14s}] {hash_unique} uniq hashes, diff={mean_diff:.1f}, sizes={sizes[:3]}...')

# Summary
print(f'\n{"="*70}')
print(f'  VERIFICATION SUMMARY (checked {len(all_results)} patients)')
print(f'{"="*70}')
status_counts = Counter(r['status'] for r in all_results)
for s, c in status_counts.most_common():
    print(f'  {s:15s}: {c} patients')

print(f'\n  Hash uniqueness: mean={np.mean([r["hash_unique"] for r in all_results]):.1f}/12')
print(f'  Inter-frame diff: mean={np.mean([r["mean_diff"] for r in all_results]):.2f} (video >2, screenshot ~0)')

if problem_pids:
    print(f'\n  WARNING: PROBLEM PATIENTS ({len(problem_pids)}):')
    for pid, reason in problem_pids[:10]:
        print(f'    {pid[:40]:42s} {reason}')
else:
    print(f'\n  [OK] ALL {len(all_results)} PATIENTS HAVE REAL VIDEO FRAMES')
