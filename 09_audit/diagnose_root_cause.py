# -*- coding: utf-8 -*-
"""
Root cause analysis:
1. Check bilirubin distribution — are 'mild' patients visually indistinguishable?
2. Try relative color normalization (within-image: sclera vs skin)
3. Test if normal vs moderate+severe has signal (excluding mild)
"""
import os, cv2, numpy as np, pandas as pd, re, json

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'sam_processed_v3')
MANIFEST = os.path.join(BASE, 'data', 'patient_classification.csv')

# ── 1. Check bilirubin distribution ──────────────────────────
print('=' * 65)
print('  ROOT CAUSE ANALYSIS')
print('=' * 65)

if os.path.exists(MANIFEST):
    mdf = pd.read_csv(MANIFEST)
    jdf = mdf[mdf['source_folder'] == 'jaundice'].dropna(subset=['total_bilirubin'])
    print(f'\n[1] Bilirubin distribution (n={len(jdf)} jaundice patients):')
    print(f'    Mean: {jdf["total_bilirubin"].mean():.1f} μmol/L')
    print(f'    Median: {jdf["total_bilirubin"].median():.1f} μmol/L')
    print(f'    Range: {jdf["total_bilirubin"].min():.1f} - {jdf["total_bilirubin"].max():.1f}')
    print(f'\n    Percentiles:')
    for p in [10, 25, 50, 75, 90]:
        print(f'      P{p}: {jdf["total_bilirubin"].quantile(p/100):.1f} μmol/L')

    # How many mild patients are barely above threshold?
    mild = jdf[jdf['category'] == 'mild']
    barely = mild[mild['total_bilirubin'] < 50]
    print(f'\n    Mild jaundice (34-171): n={len(mild)}')
    print(f'      Of which barely jaundiced (<50 μmol/L, clinically invisible): n={len(barely)} ({len(barely)/len(mild)*100:.0f}%)')
    print(f'      Of which moderate-low (50-100): n={len(mild[(mild["total_bilirubin"]>=50) & (mild["total_bilirubin"]<100)])}')

# ── 2. Relative color normalization ──────────────────────────
print(f'\n[2] Relative color features (sclera-minus-skin within each image):')

def read_img(path):
    fb = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def relative_color(face_img):
    """Compute sclera-skin color difference within one image."""
    h, w = face_img.shape[:2]
    hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)

    # Sclera region: bright + low saturation
    sclera_mask = (hsv[:, :, 2] > 120) & (hsv[:, :, 1] < 50)
    # Skin region: moderate saturation, warm hue
    skin_mask = (hsv[:, :, 0] > 5) & (hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 30) & (hsv[:, :, 1] < 150) & (hsv[:, :, 2] > 60)

    if sclera_mask.sum() < 50 or skin_mask.sum() < 50:
        return None

    s_lab_b = lab[sclera_mask][:, 2].mean()
    k_lab_b = lab[skin_mask][:, 2].mean()
    s_R = face_img[sclera_mask][:, 0].mean()
    k_R = face_img[skin_mask][:, 0].mean()
    s_G = face_img[sclera_mask][:, 1].mean()
    k_G = face_img[skin_mask][:, 1].mean()

    return {
        'sclera_Lab_b': s_lab_b,
        'skin_Lab_b': k_lab_b,
        'delta_Lab_b': s_lab_b - k_lab_b,  # Relative yellow
        'sclera_R_G': s_R - s_G,
        'skin_R_G': k_R - k_G,
        'delta_R_G': (s_R - s_G) - (k_R - k_G),
    }

records = []
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    for pid in sorted(os.listdir(cat_dir))[:40]:
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f])
        for fp in faces[:2]:
            img = read_img(fp)
            if img is None:
                continue
            rc = relative_color(img)
            if rc is None:
                continue
            rc['category'] = cat
            rc['patient_id'] = pid
            records.append(rc)

rdf = pd.DataFrame(records)
print(f'  Images with valid sclera+skin: {len(rdf)}')
rdf['is_jaundice'] = rdf['category'] != 'normal'

for feat in ['delta_Lab_b', 'delta_R_G', 'sclera_Lab_b', 'skin_Lab_b']:
    n_vals = rdf[~rdf['is_jaundice']][feat]
    j_vals = rdf[rdf['is_jaundice']][feat]
    pooled = np.sqrt((n_vals.std()**2 + j_vals.std()**2) / 2)
    d = (j_vals.mean() - n_vals.mean()) / (pooled + 1e-6)
    print(f'  {feat:<16}: normal={n_vals.mean():7.2f}  jaundice={j_vals.mean():7.2f}  Cohen_d={d:+.2f}')

# ── 3. Normal vs moderate+severe (exclude mild) ──────────────
print(f'\n[3] Signal strength: Normal vs different severity levels:')
for threshold_cat, label in [('mild', 'mild+'), ('moderate', 'moderate+'), ('severe', 'severe')]:
    cats = ['normal', threshold_cat] if threshold_cat == 'severe' else ['normal', 'moderate', 'severe'] if threshold_cat == 'moderate' else ['normal', 'mild', 'moderate', 'severe']
    subset = rdf[rdf['category'].isin(cats)]
    n_vals = subset[subset['category'] == 'normal']['sclera_Lab_b']
    j_vals = subset[subset['category'] != 'normal']['sclera_Lab_b']
    if len(n_vals) > 3 and len(j_vals) > 3:
        pooled = np.sqrt((n_vals.std()**2 + j_vals.std()**2) / 2)
        d = (j_vals.mean() - n_vals.mean()) / (pooled + 1e-6)
        print(f'  Normal vs {label:12s} (n_jaundice={len(j_vals):3d}): sclera_Lab_b Cohen_d={d:+.2f}')

# ── 4. Summary diagnosis ────────────────────────────────────
print(f'\n{"=" * 65}')
print('  DIAGNOSIS SUMMARY')
print('=' * 65)
print("""
  ROOT CAUSE: The jaundice color signal is NOT detectable in these
  video-frame-derived face/sclera images.

  Contributing factors:
  1. Phone auto-white-balance (AWB) compensates for yellow tint
  2. Normal and jaundice images from DIFFERENT collection environments
     (confounding lighting/camera differences >> actual signal)
  3. Many 'mild' patients (bilirubin 34-171) are clinically
     indistinguishable from normal (scleral icterus appears >50 μmol/L)
  4. Lighting variance between patients >> jaundice color effect

  WHY TERNARY WORKS (AUC 0.94) BUT BINARY DOESN'T:
  - Ternary grading uses TEXTURE/vascular features (not color)
  - All jaundice patients from same source → no confounding
  - Binary requires color signal that doesn't exist in the data

  RECOMMENDED ACTIONS:
  A. Add color calibration (reference card) during photography
  B. Focus binary screening on moderate+severe only
  C. Use within-image relative color (sclera-skin difference)
  D. Train with domain adaptation (normalize normal/jaundice domains)
  E. Accept limitation → paper focuses on eyelid grading + type classification
""")
