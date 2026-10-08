# -*- coding: utf-8 -*-
"""
Image Quality Diagnosis for Binary Classification
Checks if normal vs jaundice face images are visually distinguishable.
Computes color statistics, checks for label/processing issues.
"""
import os, cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'sam_processed_v3')
RAW_VF = os.path.join(BASE, 'data', 'video_frames')
FIG_DIR = os.path.join(BASE, 'results', 'figures', 'diagnosis')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'figure.dpi': 150})


def read_img(path):
    fb = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def color_stats(img_rgb):
    """Compute color features relevant to jaundice detection."""
    h, w = img_rgb.shape[:2]
    # Full image
    mean_r, mean_g, mean_b = img_rgb[:, :, 0].mean(), img_rgb[:, :, 1].mean(), img_rgb[:, :, 2].mean()
    # Center region (avoid background)
    center = img_rgb[h//4:3*h//4, w//4:3*w//4]
    # HSV
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    # Lab
    lab = cv2.cvtColor(center, cv2.COLOR_RGB2LAB)
    # Yellow metrics
    lab_b = lab[:, :, 2].mean()
    yellow_ratio = (lab[:, :, 2] > 150).mean()
    # R/G ratio (elevated in jaundice)
    rg_ratio = mean_r / (mean_g + 1e-6)
    # Green-red difference
    return {
        'mean_R': mean_r, 'mean_G': mean_g, 'mean_B': mean_b,
        'R_G': mean_r - mean_g,
        'RG_ratio': rg_ratio,
        'HSV_H': hsv[:, :, 0].mean(),
        'HSV_S': hsv[:, :, 1].mean(),
        'HSV_V': hsv[:, :, 2].mean(),
        'Lab_L': lab[:, :, 0].mean(),
        'Lab_a': lab[:, :, 1].mean(),
        'Lab_b': lab_b,
        'yellow_ratio': yellow_ratio,
        'center_R': center[:, :, 0].mean(),
        'center_G': center[:, :, 1].mean(),
        'center_B': center[:, :, 2].mean(),
    }


# ── Collect stats per category ───────────────────────────────
print('[1] Computing color statistics for all categories...')
records = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    pids = sorted(os.listdir(cat_dir))
    for pid in pids:
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f and f.endswith('.jpg')])
        for fp in faces[:4]:  # Up to 4 per patient
            img = read_img(fp)
            if img is None:
                continue
            stats = color_stats(img)
            stats['category'] = cat
            stats['patient_id'] = pid
            stats['path'] = os.path.basename(fp)
            records.append(stats)

df = pd.DataFrame(records)
df.to_csv(os.path.join(BASE, 'results', 'color_stats_diagnosis.csv'), index=False, encoding='utf-8-sig')
print(f'  Total images: {len(df)}')
print(f'  Per category: {df["category"].value_counts().to_dict()}')

# ── Statistical comparison ───────────────────────────────────
print('\n[2] Color statistics by category:')
print(f'{"Feature":<16}', end='')
for cat in ['normal', 'mild', 'moderate', 'severe']:
    print(f'  {cat:>10}', end='')
print('    normal_vs_severe')
print('-' * 80)

for feat in ['mean_R', 'mean_G', 'mean_B', 'R_G', 'Lab_b', 'yellow_ratio', 'HSV_H', 'HSV_S', 'HSV_V']:
    print(f'{feat:<16}', end='')
    means = {}
    for cat in ['normal', 'mild', 'moderate', 'severe']:
        m = df[df['category'] == cat][feat].mean()
        means[cat] = m
        print(f'  {m:>10.2f}', end='')
    diff = means.get('severe', 0) - means.get('normal', 0)
    print(f'    {diff:>+8.2f}')

# ── Key question: Can we distinguish normal vs jaundice? ─────
print('\n[3] Normal vs Jaundice (all grades combined):')
df['is_jaundice'] = df['category'] != 'normal'
for feat in ['mean_R', 'mean_G', 'mean_B', 'R_G', 'Lab_b', 'yellow_ratio', 'HSV_H']:
    n_mean = df[~df['is_jaundice']][feat].mean()
    j_mean = df[df['is_jaundice']][feat].mean()
    n_std = df[~df['is_jaundice']][feat].std()
    j_std = df[df['is_jaundice']][feat].std()
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((n_std**2 + j_std**2) / 2)
    cohens_d = (j_mean - n_mean) / (pooled_std + 1e-6)
    sep = '***' if abs(cohens_d) > 0.8 else ('**' if abs(cohens_d) > 0.5 else ('*' if abs(cohens_d) > 0.3 else ''))
    print(f'  {feat:<16}: normal={n_mean:7.2f}±{n_std:.2f}  jaundice={j_mean:7.2f}±{j_std:.2f}  Cohen_d={cohens_d:+.2f} {sep}')

# ── Visualize sample images ──────────────────────────────────
print('\n[4] Saving sample image grids...')
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
for ri, cat in enumerate(['normal', 'mild', 'moderate', 'severe']):
    cat_df = df[df['category'] == cat].sample(min(8, len(df[df['category']==cat])), random_state=42)
    for ci, (_, row) in enumerate(cat_df.iterrows()):
        ax = axes[ri, ci]
        img = read_img(os.path.join(DATA, cat, row['patient_id'], row['path']))
        if img is not None:
            ax.imshow(img)
            ax.set_title(f'b*={row["Lab_b"]:.0f} yr={row["yellow_ratio"]:.2f}', fontsize=7)
        ax.axis('off')
    axes[ri, 0].set_ylabel(cat, fontsize=12, fontweight='bold')
plt.suptitle('Face Images: Normal vs Jaundice (SAM v3 processed)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'sample_images_grid.png'))
plt.close(fig)

# ── Distribution plots ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, feat in zip(axes.ravel(), ['Lab_b', 'yellow_ratio', 'R_G', 'HSV_H', 'HSV_S', 'mean_R']):
    for cat, color in [('normal', '#2A9D8F'), ('mild', '#F4D03F'), ('moderate', '#E67E22'), ('severe', '#E74C3C')]:
        vals = df[df['category'] == cat][feat].dropna()
        ax.hist(vals, bins=30, alpha=0.5, label=cat, color=color)
    ax.set_title(feat, fontsize=11)
    ax.legend(fontsize=7)
plt.suptitle('Color Feature Distributions by Category', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'color_distributions.png'))
plt.close(fig)

# ── Compare SAM processed vs original frames ─────────────────
print('\n[5] Comparing SAM processed vs original video frames...')
raw_stats, sam_stats = [], []
for cat in ['normal', 'severe']:
    sam_dir = os.path.join(DATA, cat)
    vf_dir = os.path.join(RAW_VF, cat)
    if not os.path.exists(vf_dir):
        continue
    for pid in sorted(os.listdir(sam_dir))[:15]:
        # SAM processed
        sam_faces = [os.path.join(sam_dir, pid, f) for f in os.listdir(os.path.join(sam_dir, pid)) if '_face' in f]
        if not sam_faces:
            continue
        img = read_img(sam_faces[0])
        if img is not None:
            s = color_stats(img); s['source'] = 'SAM'; s['category'] = cat
            sam_stats.append(s)
        # Original frame
        vf_files = [f for f in os.listdir(os.path.join(vf_dir, pid)) if f.endswith('.jpg')] if os.path.exists(os.path.join(vf_dir, pid)) else []
        if vf_files:
            img2 = read_img(os.path.join(vf_dir, pid, vf_files[0]))
            if img2 is not None:
                s2 = color_stats(img2); s2['source'] = 'Raw'; s2['category'] = cat
                raw_stats.append(s2)

cmp_df = pd.DataFrame(raw_stats + sam_stats)
if len(cmp_df) > 0:
    print(f'\n  {"Feature":<14} {"Raw-Normal":>11} {"SAM-Normal":>11} {"Raw-Severe":>11} {"SAM-Severe":>11}')
    print('  ' + '-' * 60)
    for feat in ['Lab_b', 'yellow_ratio', 'HSV_H', 'HSV_V']:
        vals = {}
        for src in ['Raw', 'SAM']:
            for cat in ['normal', 'severe']:
                subset = cmp_df[(cmp_df['source'] == src) & (cmp_df['category'] == cat)]
                vals[f'{src}-{cat}'] = subset[feat].mean() if len(subset) > 0 else 0
        print(f'  {feat:<14} {vals["Raw-normal"]:>11.2f} {vals["SAM-normal"]:>11.2f} {vals["Raw-severe"]:>11.2f} {vals["SAM-severe"]:>11.2f}')

print(f'\n[6] Figures saved to: {FIG_DIR}')
print(f'    Color stats CSV: {BASE}\\results\\color_stats_diagnosis.csv')
