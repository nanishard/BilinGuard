# -*- coding: utf-8 -*-
"""
Deep SAM + Lighting Confound Analysis
Answers: 
1. Does SAM processing change/destroy the color signal?
2. Are normal vs jaundice images photographed in DIFFERENT environments?
3. What's the background color difference? (confound proxy)
4. Is CLAHE introducing or amplifying the confound?
"""
import os, cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\research\人脸识别营养\传染科'
SAM = os.path.join(BASE, 'data', 'sam_processed_v3')
RAW = os.path.join(BASE, 'data', 'video_frames')
FIG = os.path.join(BASE, 'results', 'figures', 'diagnosis')
os.makedirs(FIG, exist_ok=True)


def read_img(path):
    fb = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def full_color_profile(img_rgb, label=''):
    """Compute comprehensive color features."""
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Split into regions
    # Face center (assuming face is roughly centered)
    cy1, cy2 = int(h*0.2), int(h*0.8)
    cx1, cx2 = int(w*0.2), int(w*0.8)
    center = img_rgb[cy1:cy2, cx1:cx2]
    bg = np.ones((h, w), dtype=bool)
    bg[cy1:cy2, cx1:cx2] = False
    bg_pixels = img_rgb[bg]
    
    return {
        f'{label}_mean_R': img_rgb[:,:,0].mean(),
        f'{label}_mean_G': img_rgb[:,:,1].mean(),
        f'{label}_mean_B': img_rgb[:,:,2].mean(),
        f'{label}_bg_R': bg_pixels[:,0].mean() if len(bg_pixels)>0 else 0,
        f'{label}_bg_G': bg_pixels[:,1].mean() if len(bg_pixels)>0 else 0,
        f'{label}_bg_B': bg_pixels[:,2].mean() if len(bg_pixels)>0 else 0,
        f'{label}_Lab_L': lab[:,:,0].mean(),
        f'{label}_Lab_a': lab[:,:,1].mean(),
        f'{label}_Lab_b': lab[:,:,2].mean(),
        f'{label}_HSV_H': hsv[:,:,0].mean(),
        f'{label}_HSV_S': hsv[:,:,1].mean(),
        f'{label}_HSV_V': hsv[:,:,2].mean(),
        f'{label}_color_temp_R/B': img_rgb[:,:,0].mean() / (img_rgb[:,:,2].mean() + 1e-6),
    }


# ── 1. Match raw and SAM images for same patients ────────────
print('[1] Matching RAW and SAM images for paired comparison...')
raw_data = []
sam_data = []
paired = []

for cat in ['normal', 'mild', 'moderate', 'severe']:
    sam_cat = os.path.join(SAM, cat)
    raw_cat = os.path.join(RAW, cat)
    if not os.path.exists(raw_cat):
        continue
    
    for pid in sorted(os.listdir(sam_cat))[:25]:
        sam_p = os.path.join(sam_cat, pid)
        raw_p = os.path.join(raw_cat, pid)
        if not os.path.isdir(sam_p) or not os.path.isdir(raw_p):
            continue
        
        # Get SAM face images
        sam_faces = sorted([f for f in os.listdir(sam_p) if '_face' in f])
        raw_frames = sorted([f for f in os.listdir(raw_p) if f.endswith('.jpg')])
        
        if not sam_faces or not raw_frames:
            continue
        
        # Match by frame name
        for sf in sam_faces[:2]:
            frame_base = sf.replace('_face.jpg', '.jpg')
            if frame_base in raw_frames:
                raw_img = read_img(os.path.join(raw_p, frame_base))
                sam_img = read_img(os.path.join(sam_p, sf))
                if raw_img is not None and sam_img is not None:
                    raw_stats = full_color_profile(raw_img, 'raw')
                    sam_stats = full_color_profile(sam_img, 'sam')
                    raw_stats['category'] = cat
                    sam_stats['category'] = cat
                    raw_stats['patient_id'] = pid
                    sam_stats['patient_id'] = pid
                    raw_data.append(raw_stats)
                    sam_data.append(sam_stats)
                    paired.append((cat, pid, frame_base, raw_img, sam_img))

raw_df = pd.DataFrame(raw_data)
sam_df = pd.DataFrame(sam_data)
print(f'  Paired images: {len(paired)} ({raw_df["category"].value_counts().to_dict()})')

# ── 2. RAW vs SAM: Color change by category ──────────────────
print('\n[2] Does SAM processing change the color signal?')
print('    (Positive Δ = SAM INCREASED this feature)')

raw_df['is_jaundice'] = raw_df['category'] != 'normal'
sam_df['is_jaundice'] = sam_df['category'] != 'normal'

print(f'\n  {"Feature":<20} {"Raw-Norm":>9} {"Raw-Jaun":>9} {"Δsignal":>8} | {"SAM-Norm":>9} {"SAM-Jaun":>9} {"Δsignal":>8} | {"SAM preserves?":>15}')
print('  ' + '-' * 100)

for feat_short in ['mean_R', 'mean_G', 'mean_B', 'Lab_b', 'Lab_a', 'HSV_H', 'HSV_V', 'bg_R', 'bg_G', 'bg_B', 'color_temp_R/B']:
    raw_f = f'raw_{feat_short}'
    sam_f = f'sam_{feat_short}'
    if raw_f not in raw_df.columns or sam_f not in sam_df.columns:
        continue
    
    rn = raw_df[~raw_df['is_jaundice']][raw_f].mean()
    rj = raw_df[raw_df['is_jaundice']][raw_f].mean()
    sn = sam_df[~sam_df['is_jaundice']][sam_f].mean()
    sj = sam_df[sam_df['is_jaundice']][sam_f].mean()
    
    raw_signal = rj - rn
    sam_signal = sj - sn
    
    # Did SAM preserve the direction and magnitude of the signal?
    if abs(raw_signal) < 0.5:
        preserve = 'no signal'
    elif np.sign(raw_signal) == np.sign(sam_signal):
        ratio = abs(sam_signal) / (abs(raw_signal) + 1e-6)
        preserve = f'YES ({ratio:.1f}x)' if ratio > 0.5 else f'WEAKENED ({ratio:.1f}x)'
    else:
        preserve = 'INVERTED!'
    
    print(f'  {feat_short:<20} {rn:>9.2f} {rj:>9.2f} {raw_signal:>+8.2f} | {sn:>9.2f} {sj:>9.2f} {sam_signal:>+8.2f} | {preserve:>15}')

# ── 3. Background color: the confound proxy ──────────────────
print('\n[3] BACKGROUND COLOR comparison (lighting environment proxy):')
print('    If background differs between normal and jaundice,')
print('    the model learns ENVIRONMENT, not jaundice!')
print()

for source_label, df in [('RAW', raw_df), ('SAM', sam_df)]:
    print(f'  [{source_label}] Background color by category:')
    for feat in ['bg_R', 'bg_G', 'bg_B']:
        print(f'    {feat}:', end='')
        for cat in ['normal', 'mild', 'moderate', 'severe']:
            m = df[df['category'] == cat][f'{source_label.lower()}_{feat}'].mean()
            print(f'  {cat}={m:.1f}', end='')
        print()
    
    # Cohen's d for background
    for feat in ['bg_R', 'bg_G', 'bg_B']:
        n_vals = df[~df['is_jaundice']][f'{source_label.lower()}_{feat}']
        j_vals = df[df['is_jaundice']][f'{source_label.lower()}_{feat}']
        pooled = np.sqrt((n_vals.std()**2 + j_vals.std()**2) / 2)
        d = (j_vals.mean() - n_vals.mean()) / (pooled + 1e-6)
        print(f'    {feat} Cohen_d (normal vs jaundice): {d:+.2f}')
    print()

# ── 4. The CRITICAL test: signal vs confound ─────────────────
print('[4] CRITICAL: Signal-to-Confound Ratio')
print('    Face Lab_b difference (potential signal) vs Background difference (confound)')
print()

for source_label, df in [('RAW', raw_df), ('SAM', sam_df)]:
    prefix = source_label.lower()
    # Signal: face Lab_b difference
    face_n = df[~df['is_jaundice']][f'{prefix}_Lab_b']
    face_j = df[df['is_jaundice']][f'{prefix}_Lab_b']
    face_signal = abs(face_j.mean() - face_n.mean())
    
    # Confound: background mean color difference
    bg_n = df[~df['is_jaundice']][[f'{prefix}_bg_R', f'{prefix}_bg_G', f'{prefix}_bg_B']].mean(axis=1)
    bg_j = df[df['is_jaundice']][[f'{prefix}_bg_R', f'{prefix}_bg_G', f'{prefix}_bg_B']].mean(axis=1)
    bg_confound = abs(bg_j.mean() - bg_n.mean())
    
    ratio = face_signal / (bg_confound + 1e-6)
    print(f'  [{source_label}]')
    print(f'    Face Lab_b signal:     {face_signal:.2f}')
    print(f'    Background confound:   {bg_confound:.2f}')
    print(f'    Signal/Confound ratio: {ratio:.3f}  {"→ CONFOUND DOMINATES" if ratio < 1 else "→ signal detectable"}')
    print()

# ── 5. Color temperature analysis ────────────────────────────
print('[5] COLOR TEMPERATURE (R/B ratio) — lighting warmth indicator:')
print('    Higher = warmer light (incandescent), Lower = cooler light (fluorescent/LED)')
print()
for source_label, df in [('RAW', raw_df), ('SAM', sam_df)]:
    prefix = source_label.lower()
    feat = f'{prefix}_color_temp_R/B'
    print(f'  [{source_label}] R/B ratio by category:')
    for cat in ['normal', 'mild', 'moderate', 'severe']:
        m = df[df['category'] == cat][feat].mean()
        s = df[df['category'] == cat][feat].std()
        print(f'    {cat:12s}: {m:.3f} ± {s:.3f}')
    
    n_vals = df[~df['is_jaundice']][feat]
    j_vals = df[df['is_jaundice']][feat]
    pooled = np.sqrt((n_vals.std()**2 + j_vals.std()**2) / 2)
    d = (j_vals.mean() - n_vals.mean()) / (pooled + 1e-6)
    print(f'    Cohen_d (normal vs jaundice): {d:+.2f}')
    print(f'    → {"DIFFERENT lighting environments!" if abs(d) > 0.5 else "Similar lighting"}')
    print()

# ── 6. Visualization ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Background color scatter
ax = axes[0, 0]
for cat, color in [('normal', '#2A9D8F'), ('mild', '#F4D03F'), ('moderate', '#E67E22'), ('severe', '#E74C3C')]:
    sub = raw_df[raw_df['category'] == cat]
    ax.scatter(sub['raw_bg_R'], sub['raw_bg_G'], alpha=0.6, label=cat, c=color, s=30)
ax.set_xlabel('Background R'); ax.set_ylabel('Background G')
ax.set_title('RAW: Background Color by Category\n(Different clusters = different environments)', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# Color temperature distribution
ax = axes[0, 1]
for cat, color in [('normal', '#2A9D8F'), ('mild', '#F4D03F'), ('moderate', '#E67E22'), ('severe', '#E74C3C')]:
    vals = raw_df[raw_df['category'] == cat]['raw_color_temp_R/B'].dropna()
    ax.hist(vals, bins=15, alpha=0.5, label=cat, color=color)
ax.set_xlabel('R/B ratio (color temperature)'); ax.set_ylabel('Count')
ax.set_title('RAW: Color Temperature Distribution', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# Lab_b: Raw vs SAM
ax = axes[0, 2]
for cat, marker in [('normal', 'o'), ('mild', '^'), ('moderate', 's'), ('severe', 'D')]:
    sub_raw = raw_df[raw_df['category'] == cat]
    sub_sam = sam_df[sam_df['category'] == cat]
    merged = sub_raw.merge(sub_sam, on=['patient_id', 'category'])
    ax.scatter(merged['raw_Lab_b'], merged['sam_Lab_b'], marker=marker, label=cat, s=30, alpha=0.6)
ax.plot([125, 145], [125, 145], 'k--', lw=0.5)
ax.set_xlabel('Raw Lab b*'); ax.set_ylabel('SAM Lab b*')
ax.set_title('Lab b*: Raw vs SAM (paired)\n(Off diagonal = SAM changed color)', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# HSV_V: Raw vs SAM (brightness)
ax = axes[1, 0]
for cat, marker in [('normal', 'o'), ('mild', '^'), ('moderate', 's'), ('severe', 'D')]:
    sub_raw = raw_df[raw_df['category'] == cat]
    sub_sam = sam_df[sam_df['category'] == cat]
    merged = sub_raw.merge(sub_sam, on=['patient_id', 'category'])
    ax.scatter(merged['raw_HSV_V'], merged['sam_HSV_V'], marker=marker, label=cat, s=30, alpha=0.6)
ax.plot([100, 200], [100, 200], 'k--', lw=0.5)
ax.set_xlabel('Raw HSV V'); ax.set_ylabel('SAM HSV V')
ax.set_title('Brightness: Raw vs SAM (CLAHE effect)', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# Background brightness by category
ax = axes[1, 1]
for source_label, df, c in [('RAW', raw_df, '#457B9D'), ('SAM', sam_df, '#E76F51')]:
    prefix = source_label.lower()
    bg_mean = df[[f'{prefix}_bg_R', f'{prefix}_bg_G', f'{prefix}_bg_B']].mean(axis=1)
    cats = ['normal', 'mild', 'moderate', 'severe']
    means = [bg_mean[df['category'] == c].mean() for c in cats]
    ax.bar([f'{c}\n{source_label}' for c in cats] if source_label == 'SAM' else cats, means, alpha=0.7, color=c, label=source_label)
ax.set_ylabel('Background brightness (mean RGB)')
ax.set_title('Background Brightness by Category\n(Higher = brighter room)', fontsize=9, fontweight='bold')
ax.legend(fontsize=7)

# Face Lab_b by category: Raw vs SAM
ax = axes[1, 2]
cats = ['normal', 'mild', 'moderate', 'severe']
x = np.arange(len(cats)); width = 0.35
raw_means = [raw_df[raw_df['category'] == c]['raw_Lab_b'].mean() for c in cats]
sam_means = [sam_df[sam_df['category'] == c]['sam_Lab_b'].mean() for c in cats]
raw_stds = [raw_df[raw_df['category'] == c]['raw_Lab_b'].std() for c in cats]
sam_stds = [sam_df[sam_df['category'] == c]['sam_Lab_b'].std() for c in cats]
ax.bar(x - width/2, raw_means, width, yerr=raw_stds, label='Raw', color='#457B9D', capsize=3)
ax.bar(x + width/2, sam_means, width, yerr=sam_stds, label='SAM', color='#E76F51', capsize=3)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel('Lab b* (yellow-blue axis)')
ax.set_title('Face Lab b*: Raw vs SAM by Category', fontsize=9, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG, 'sam_lighting_confound_analysis.png'), dpi=200)
plt.close(fig)
print(f'\n[6] Figure saved: {FIG}/sam_lighting_confound_analysis.png')

# ── 7. CONCLUSION ────────────────────────────────────────────
print('\n' + '=' * 70)
print('  ANALYSIS CONCLUSION')
print('=' * 70)
