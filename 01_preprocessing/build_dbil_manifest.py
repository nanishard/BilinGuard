# -*- coding: utf-8 -*-
"""
DBIL-based classification: Build patient manifest + check color signal.
DBIL grading:
  Grade 0 (normal):  DBIL ≤ 10 μmol/L (normal 0-6.8 + mild 6.8-10 merged)
  Grade 1 (moderate): 10-68 μmol/L
  Grade 2 (severe): > 68 μmol/L
"""
import os, re, json, cv2, numpy as np, pandas as pd

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'sam_processed_v3')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')

# ── Load Excel DBIL ──────────────────────────────────────────
df_ex = pd.read_excel(EXCEL)
dbil_col = [c for c in df_ex.columns if '直接胆红素' in str(c)][0]
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)
df_ex['dbil'] = pd.to_numeric(df_ex[dbil_col], errors='coerce')


def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None

def strip_zeros(s):
    return s.lstrip('0') if s else s

def get_dbil(patient_id):
    hid = extract_hid(patient_id)
    if not hid: return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    if len(rows) > 0:
        return float(rows.iloc[0]['dbil'])
    return None

def dbil_grade(dbil):
    if dbil is None or pd.isna(dbil): return None
    if dbil <= 10: return 0   # normal + mild merged
    elif dbil <= 68: return 1  # moderate
    else: return 2             # severe


# ── Build patient manifest with DBIL grades ──────────────────
print('[1] Building DBIL-based patient classification...')

# Collect all patients from sam_processed_v3
patients = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cat_dir = os.path.join(DATA, cat)
    if not os.path.exists(cat_dir):
        continue
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p):
            continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            old_cat = cat
            # Determine DBIL grade
            if old_cat == 'normal':
                dbil_val = None  # Normal folder patients assumed normal
                grade = 0
            else:
                dbil_val = get_dbil(pid)
                grade = dbil_grade(dbil_val)
            patients[pid] = {
                'faces': faces,
                'old_category': old_cat,
                'dbil': dbil_val,
                'dbil_grade': grade,
            }

# Count grades
grades = [p['dbil_grade'] for p in patients.values() if p['dbil_grade'] is not None]
from collections import Counter
print(f'  Total patients: {len(patients)}')
print(f'  DBIL grade distribution: {Counter(grades)}')

# Also get eyelid images from manifest
mdf = pd.read_csv(MANIFEST)
for _, r in mdf.iterrows():
    pid = r['patient_id']
    if pid in patients and r['n_eyelid'] > 0:
        patients[pid]['eyelids'] = json.loads(r['eyelid_images'])

n_with_eyelid = sum(1 for p in patients.values() if 'eyelids' in p and p['eyelids'])
print(f'  Patients with eyelid photos: {n_with_eyelid}')

# ── Check color signal for DBIL grades ───────────────────────
print('\n[2] Color signal check for DBIL grades...')

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None

# Face color stats by DBIL grade
face_recs = []
eyelid_recs = []
for pid, pdata in patients.items():
    grade = pdata['dbil_grade']
    if grade is None:
        continue
    # Face
    for fp in pdata['faces'][:2]:
        img = read_img(fp)
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        face_recs.append({
            'grade': grade, 'pid': pid,
            'Lab_b': lab[:,:,2].mean(), 'Lab_a': lab[:,:,1].mean(),
            'HSV_H': hsv[:,:,0].mean(), 'HSV_S': hsv[:,:,1].mean(), 'HSV_V': hsv[:,:,2].mean(),
            'R': img[:,:,0].mean(), 'G': img[:,:,1].mean(), 'B': img[:,:,2].mean(),
            'R_G': img[:,:,0].mean() - img[:,:,1].mean(),
        })
    # Eyelid
    for ep in pdata.get('eyelids', [])[:1]:
        img = read_img(ep)
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        eyelid_recs.append({
            'grade': grade, 'pid': pid,
            'Lab_b': lab[:,:,2].mean(), 'Lab_a': lab[:,:,1].mean(),
            'HSV_H': hsv[:,:,0].mean(), 'HSV_S': hsv[:,:,1].mean(), 'HSV_V': hsv[:,:,2].mean(),
            'R': img[:,:,0].mean(), 'G': img[:,:,1].mean(), 'B': img[:,:,2].mean(),
            'R_G': img[:,:,0].mean() - img[:,:,1].mean(),
        })

fdf = pd.DataFrame(face_recs)
edf = pd.DataFrame(eyelid_recs)

print(f'\n  FACE images: {len(fdf)}  | EYELID images: {len(edf)}')

print(f'\n  FACE color stats by DBIL grade:')
print(f'  {"Feature":<10}', end='')
for g in [0, 1, 2]:
    print(f'  Grade{g}(n={len(fdf[fdf["grade"]==g])})', end='')
print('  Cohen_d(0vs2)')
for feat in ['Lab_b', 'HSV_H', 'HSV_S', 'HSV_V', 'R', 'R_G']:
    print(f'  {feat:<10}', end='')
    means = {}
    for g in [0, 1, 2]:
        m = fdf[fdf['grade']==g][feat].mean()
        means[g] = m
        print(f'  {m:>16.2f}', end='')
    n0 = fdf[fdf['grade']==0][feat]; n2 = fdf[fdf['grade']==2][feat]
    pooled = np.sqrt((n0.std()**2 + n2.std()**2)/2)
    d = (means[2] - means[0]) / (pooled + 1e-6)
    print(f'  {d:>+8.2f}')

print(f'\n  EYELID color stats by DBIL grade:')
print(f'  {"Feature":<10}', end='')
for g in [0, 1, 2]:
    print(f'  Grade{g}(n={len(edf[edf["grade"]==g])})', end='')
print('  Cohen_d(0vs2)')
for feat in ['Lab_b', 'HSV_H', 'HSV_S', 'HSV_V', 'R', 'R_G']:
    print(f'  {feat:<10}', end='')
    means = {}
    for g in [0, 1, 2]:
        m = edf[edf['grade']==g][feat].mean()
        means[g] = m
        print(f'  {m:>16.2f}', end='')
    n0 = edf[edf['grade']==0][feat]; n2 = edf[edf['grade']==2][feat]
    if len(n0)>1 and len(n2)>1:
        pooled = np.sqrt((n0.std()**2 + n2.std()**2)/2)
        d = (means[2] - means[0]) / (pooled + 1e-6)
        print(f'  {d:>+8.2f}')
    else:
        print('  N/A')

# ── Save DBIL manifest ───────────────────────────────────────
manifest_out = []
for pid, pdata in patients.items():
    if pdata['dbil_grade'] is not None:
        manifest_out.append({
            'patient_id': pid,
            'dbil_grade': pdata['dbil_grade'],
            'dbil_value': pdata['dbil'],
            'old_category': pdata['old_category'],
            'n_face': len(pdata['faces']),
            'n_eyelid': len(pdata.get('eyelids', [])),
            'face_images': json.dumps(pdata['faces'][:12]),
            'eyelid_images': json.dumps(pdata.get('eyelids', [])[:3]),
        })
out_df = pd.DataFrame(manifest_out)
out_df.to_csv(os.path.join(BASE, 'data', 'dbil_manifest.csv'), index=False, encoding='utf-8-sig')
print(f'\n[3] Saved DBIL manifest: {len(out_df)} patients → data/dbil_manifest.csv')
print(f'  Grade 0 (DBIL≤10): {sum(out_df["dbil_grade"]==0)} patients')
print(f'  Grade 1 (DBIL 10-68): {sum(out_df["dbil_grade"]==1)} patients')
print(f'  Grade 2 (DBIL >68): {sum(out_df["dbil_grade"]==2)} patients')
