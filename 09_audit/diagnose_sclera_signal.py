# -*- coding: utf-8 -*-
"""
Deep diagnosis: sclera color signal + raw vs SAM comparison + per-patient variance.
Tests whether the jaundice signal exists in sclera images or was destroyed by processing.
"""
import os, cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'sam_processed_v3')
RAW_VF = os.path.join(BASE, 'data', 'video_frames')
FIG_DIR = os.path.join(BASE, 'results', 'figures', 'diagnosis')
os.makedirs(FIG_DIR, exist_ok=True)


def read_img(path):
    fb = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def sclera_stats(img_rgb):
    """Extract sclera (white-eye) region and compute yellow metrics."""
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # Detect sclera: high V (brightness), low S (saturation)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    sclera_mask = (v > 100) & (s < 60)

    if sclera_mask.sum() < 50:
        # Fallback: use whole image
        sclera_mask = np.ones((h, w), dtype=bool)

    # Stats on sclera region
    sclera_pixels = img_rgb[sclera_mask]
    sclera_lab = lab[sclera_mask]

    return {
        'sclera_pct': float(sclera_mask.mean()),
        'sclera_R': float(sclera_pixels[:, 0].mean()),
        'sclera_G': float(sclera_pixels[:, 1].mean()),
        'sclera_B': float(sclera_pixels[:, 2].mean()),
        'sclera_Lab_b': float(sclera_lab[:, 2].mean()),
        'sclera_Lab_a': float(sclera_lab[:, 1].mean()),
        'sclera_yellow_ratio': float((sclera_lab[:, 2] > 145).mean()),
    }


# ── 1. Analyze sclera images from sam_processed_v3 ───────────
print('[1] Analyzing SCLERA images (sam_processed_v3)...')
sclera_records = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    for pid in sorted(os.listdir(cat_dir))[:30]:
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        scleras = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_sclera' in f and f.endswith('.jpg')])
        for sp in scleras[:2]:
            img = read_img(sp)
            if img is None:
                continue
            stats = sclera_stats(img)
            stats['category'] = cat
            stats['patient_id'] = pid
            stats['source'] = 'sclera_img'
            sclera_records.append(stats)

scdf = pd.DataFrame(sclera_records)
print(f'  Sclera images analyzed: {len(scdf)}')

# ── 2. Analyze face images (eye region only) ─────────────────
print('\n[2] Analyzing FACE images — eye region only...')
face_eye_records = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    for pid in sorted(os.listdir(cat_dir))[:30]:
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f and f.endswith('.jpg')])
        for fp in faces[:2]:
            img = read_img(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            # Extract eye region (top 20-45% of face)
            eye_region = img[int(h*0.2):int(h*0.45), :]
            stats = sclera_stats(eye_region)
            stats['category'] = cat
            stats['patient_id'] = pid
            stats['source'] = 'face_eye_region'
            face_eye_records.append(stats)

fedf = pd.DataFrame(face_eye_records)
print(f'  Face eye regions analyzed: {len(fedf)}')

# ── 3. Compare categories ────────────────────────────────────
print('\n[3] SCLERA IMAGE color comparison:')
print(f'{"Feature":<20}', end='')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    print(f'  {cat:>10}', end='')
print()
print('-' * 70)
for feat in ['sclera_R', 'sclera_G', 'sclera_B', 'sclera_Lab_b', 'sclera_yellow_ratio']:
    print(f'{feat:<20}', end='')
    for cat in ['normal', 'mild', 'moderate', 'severe']:
        m = scdf[scdf['category'] == cat][feat].mean()
        print(f'  {m:>10.2f}', end='')
    print()

# Cohen's d for sclera
print('\n  Normal vs Jaundice (sclera images):')
scdf['is_jaundice'] = scdf['category'] != 'normal'
for feat in ['sclera_Lab_b', 'sclera_yellow_ratio', 'sclera_R', 'sclera_G', 'sclera_B']:
    n_vals = scdf[~scdf['is_jaundice']][feat]
    j_vals = scdf[scdf['is_jaundice']][feat]
    pooled_std = np.sqrt((n_vals.std()**2 + j_vals.std()**2) / 2)
    d = (j_vals.mean() - n_vals.mean()) / (pooled_std + 1e-6)
    print(f'    {feat:<22}: normal={n_vals.mean():7.2f}  jaundice={j_vals.mean():7.2f}  Cohen_d={d:+.2f}')

print('\n[3b] FACE EYE REGION color comparison:')
fedf['is_jaundice'] = fedf['category'] != 'normal'
for feat in ['sclera_Lab_b', 'sclera_yellow_ratio', 'sclera_R']:
    n_vals = fedf[~fedf['is_jaundice']][feat]
    j_vals = fedf[fedf['is_jaundice']][feat]
    pooled_std = np.sqrt((n_vals.std()**2 + j_vals.std()**2) / 2)
    d = (j_vals.mean() - n_vals.mean()) / (pooled_std + 1e-6)
    print(f'    {feat:<22}: normal={n_vals.mean():7.2f}  jaundice={j_vals.mean():7.2f}  Cohen_d={d:+.2f}')

# ── 4. RAW vs SAM: is signal lost in processing? ─────────────
print('\n[4] RAW frames vs SAM processed — signal preservation:')
raw_records, sam_records = [], []
for cat in ['normal', 'severe']:
    sam_dir = os.path.join(DATA, cat)
    vf_dir = os.path.join(RAW_VF, cat)
    if not os.path.exists(vf_dir):
        continue
    for pid in sorted(os.listdir(sam_dir))[:20]:
        # SAM sclera
        sam_p = os.path.join(sam_dir, pid)
        if not os.path.isdir(sam_p):
            continue
        scleras = [os.path.join(sam_p, f) for f in os.listdir(sam_p) if '_sclera' in f]
        if scleras:
            img = read_img(scleras[0])
            if img is not None:
                s = sclera_stats(img); s['category'] = cat; s['source'] = 'SAM'
                sam_records.append(s)
        # Raw frame
        vf_p = os.path.join(vf_dir, pid)
        if os.path.isdir(vf_p):
            raws = [f for f in os.listdir(vf_p) if f.endswith('.jpg')]
            if raws:
                img2 = read_img(os.path.join(vf_p, raws[0]))
                if img2 is not None:
                    h2, w2 = img2.shape[:2]
                    eye_r = img2[int(h2*0.2):int(h2*0.45), int(w2*0.15):int(w2*0.85)]
                    s2 = sclera_stats(eye_r); s2['category'] = cat; s2['source'] = 'Raw'
                    raw_records.append(s2)

raw_df = pd.DataFrame(raw_records)
sam_df = pd.DataFrame(sam_records)
if len(raw_df) > 0 and len(sam_df) > 0:
    print(f'\n  {"":<22} {"Raw-Normal":>12} {"Raw-Severe":>12} {"Diff":>8} | {"SAM-Normal":>12} {"SAM-Severe":>12} {"Diff":>8}')
    print('  ' + '-' * 95)
    for feat in ['sclera_Lab_b', 'sclera_yellow_ratio', 'sclera_R', 'sclera_G', 'sclera_B']:
        rn = raw_df[raw_df['category']=='normal'][feat].mean()
        rs = raw_df[raw_df['category']=='severe'][feat].mean()
        sn = sam_df[sam_df['category']=='normal'][feat].mean()
        ss = sam_df[sam_df['category']=='severe'][feat].mean()
        print(f'  {feat:<22} {rn:>12.2f} {rs:>12.2f} {rs-rn:>+8.2f} | {sn:>12.2f} {ss:>12.2f} {ss-sn:>+8.2f}')

# ── 5. Per-patient variance analysis ─────────────────────────
print('\n[5] Per-patient color variance (is lighting the dominant factor?)...')
import pandas as pd
face_records = []
for cat in ['normal', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    for pid in sorted(os.listdir(cat_dir))[:20]:
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f])
        for fp in faces[:6]:
            img = read_img(fp)
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            face_records.append({
                'category': cat, 'patient_id': pid,
                'Lab_b': lab[:, :, 2].mean(),
                'HSV_V': hsv[:, :, 2].mean(),
                'HSV_S': hsv[:, :, 1].mean(),
            })
pdf = pd.DataFrame(face_records)
if len(pdf) > 0:
    # Between-patient vs within-patient variance
    for feat in ['Lab_b', 'HSV_V', 'HSV_S']:
        within = pdf.groupby(['category', 'patient_id'])[feat].std().mean()
        between = pdf.groupby('patient_id')[feat].mean().std()
        total = pdf[feat].std()
        print(f'  {feat:<10}: within-patient std={within:.2f}  between-patient std={between:.2f}  total std={total:.2f}')
        print(f'             → Lighting/patient variance is {between/within:.1f}x the within-patient variance')

# ── 6. Distribution plots ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
# Sclera Lab_b by category
for ax, (df_in, feat, title) in zip(axes.ravel(), [
    (scdf, 'sclera_Lab_b', 'Sclera Lab b* (Yellow)'),
    (scdf, 'sclera_yellow_ratio', 'Sclera Yellow Ratio'),
    (scdf, 'sclera_R', 'Sclera Red Channel'),
    (fedf, 'sclera_Lab_b', 'Face-Eye Lab b*'),
    (fedf, 'sclera_yellow_ratio', 'Face-Eye Yellow Ratio'),
    (fedf, 'sclera_R', 'Face-Eye Red Channel'),
]):
    if len(df_in) == 0:
        continue
    for cat, color in [('normal', '#2A9D8F'), ('mild', '#F4D03F'), ('moderate', '#E67E22'), ('severe', '#E74C3C')]:
        vals = df_in[df_in['category'] == cat][feat].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=20, alpha=0.5, label=cat, color=color)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
plt.suptitle('Sclera & Eye Region Color Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'sclera_distributions.png'))
plt.close(fig)

print(f'\n[6] Figure saved: {FIG_DIR}/sclera_distributions.png')
