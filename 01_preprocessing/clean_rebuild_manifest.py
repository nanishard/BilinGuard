# -*- coding: utf-8 -*-
"""
Clean contaminated NORMAL data + rebuild manifest.

After rebuild_normals.py extracted frames for the corrected 正常 folder,
video_frames/normal still holds STALE contaminated pids from the old run
(424 dirs vs 285 corrected). This script:
  1. Removes stale normal pids from data/video_frames/normal
  2. Clears data/sam_processed_v3/normal entirely (force SAM reprocess)
  3. Rebuilds clean_dataset_manifest.csv: 285 corrected normals + jaundice rows

Jaundice data (mild/moderate/severe) is NOT touched.
"""
import os, json, shutil
import pandas as pd

BASE = r'D:\research\人脸识别营养\传染科'
VF_NORMAL = os.path.join(BASE, 'data', 'video_frames', 'normal')
SAM_NORMAL = os.path.join(BASE, 'data', 'sam_processed_v3', 'normal')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
CORRECTED_JSON = os.path.join(BASE, 'data', 'normal_pids_corrected.json')

# ── Load corrected normal pids ───────────────────────────────
with open(CORRECTED_JSON, encoding='utf-8') as f:
    corrected = set(json.load(f))
print(f'[1] Corrected normal pids: {len(corrected)}')

# ── 1. Clean video_frames/normal ─────────────────────────────
print('\n[2] Cleaning video_frames/normal (remove stale contaminated pids)...')
removed_vf = 0
if os.path.exists(VF_NORMAL):
    for d in os.listdir(VF_NORMAL):
        if d not in corrected:
            shutil.rmtree(os.path.join(VF_NORMAL, d), ignore_errors=True)
            removed_vf += 1
remaining = len(os.listdir(VF_NORMAL)) if os.path.exists(VF_NORMAL) else 0
print(f'    Removed {removed_vf} stale dirs, {remaining} corrected normals remain')

# ── 2. Clear sam_processed_v3/normal ─────────────────────────
print('\n[3] Clearing sam_processed_v3/normal (force reprocess)...')
if os.path.exists(SAM_NORMAL):
    shutil.rmtree(SAM_NORMAL, ignore_errors=True)
    print('    Cleared.')
else:
    print('    (did not exist)')

# ── 3. Rebuild manifest ──────────────────────────────────────
print('\n[4] Rebuilding clean_dataset_manifest.csv...')
df = pd.read_csv(MANIFEST)

# keep all non-normal (jaundice) rows unchanged
jaundice = df[df['category'] != 'normal'].copy()
print(f'    Jaundice rows kept: {len(jaundice)} '
      f'(mild/mod/sev = '
      f'{(jaundice.category=="mild").sum()}/'
      f'{(jaundice.category=="moderate").sum()}/'
      f'{(jaundice.category=="severe").sum()})')

# build new normal rows
normal_rows = []
for pid in sorted(corrected):
    normal_rows.append({
        'patient_id': pid,
        'category': 'normal',
        'face_images': '[]',
        'n_face': 0,
        'eyelid_images': '[]',
        'n_eyelid': 0,
        'has_face': True,   # frames exist in video_frames/normal/{pid}
        'has_eyelid': False,
    })
normal_df = pd.DataFrame(normal_rows)
print(f'    New normal rows: {len(normal_df)}')

new_manifest = pd.concat([normal_df, jaundice], ignore_index=True)
new_manifest.to_csv(MANIFEST, index=False, encoding='utf-8-sig')
print(f'    Total manifest rows: {len(new_manifest)}')
print(f'    Saved -> {MANIFEST}')

# ── Summary ──────────────────────────────────────────────────
print('\n' + '=' * 50)
print('  CLEANUP + MANIFEST REBUILD DONE')
print('=' * 50)
print(f'  video_frames/normal      : {remaining} patients (was 424)')
print(f'  sam_processed_v3/normal  : cleared (will reprocess)')
print(f'  manifest normal rows     : {len(normal_df)} (was 138)')
