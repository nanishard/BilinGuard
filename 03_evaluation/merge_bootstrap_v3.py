# -*- coding: utf-8 -*-
"""
Merge REAL Bootstrap CIs into bootstrap_full_metrics_v3.csv.

Strategy:
  - Start from CURRENT CSV (preserves previously-merged real results)
  - For each task in the new bootstrap CSV:
      * Remove existing rows with that task
      * Append the new real-CI rows
  - Deduplicate
  - Save backup of pre-merge state
"""
import os, json, pandas as pd, shutil

BASE = r'D:\research\人脸识别营养\传染科'
TBL_V3 = os.path.join(BASE, 'results', 'tables_v3')

ORIG_CSV = os.path.join(TBL_V3, 'bootstrap_full_metrics_v3.csv')
NEW_CSV = os.path.join(TBL_V3, 'bootstrap_cp_meld_external.csv')
BACKUP_CSV = os.path.join(TBL_V3, 'bootstrap_full_metrics_v3.backup.csv')

# Load
orig = pd.read_csv(ORIG_CSV)
new = pd.read_csv(NEW_CSV)

print(f'Current CSV: {len(orig)} rows')
print(f'New CSV:     {len(new)} rows')
print(f'\nCurrent task counts:')
print(orig['task'].value_counts())
print(f'\nNew CSV task counts:')
print(new['task'].value_counts())

# Identify which tasks the new CSV covers
new_tasks = set(new['task'].unique())
print(f'\nTasks to replace: {sorted(new_tasks)}')

# Remove rows from current CSV for tasks present in new CSV
orig_filtered = orig[~orig['task'].isin(new_tasks)].copy()
print(f'\nAfter removing replaced tasks: {len(orig_filtered)} rows')

# Deduplicate (by name + task)
before = len(orig_filtered)
orig_filtered = orig_filtered.drop_duplicates(subset=['name', 'task'], keep='first').copy()
after = len(orig_filtered)
if before != after:
    print(f'Removed {before - after} additional duplicate row(s)')

# Append new real-CI rows
merged = pd.concat([orig_filtered, new], ignore_index=True)
print(f'\nMerged CSV: {len(merged)} rows')
print(f'\nMerged task counts:')
print(merged['task'].value_counts())

# Backup current state BEFORE overwriting
shutil.copy2(ORIG_CSV, BACKUP_CSV)
print(f'\nBackup saved: {BACKUP_CSV}')

# Save merged
merged.to_csv(ORIG_CSV, index=False, encoding='utf-8-sig')
print(f'Updated:  {ORIG_CSV}')

# Summary
print('\n' + '='*70)
print('  CHANGES SUMMARY')
print('='*70)
print(f'  Replaced tasks: {sorted(new_tasks)}')
print(f'  Removed old rows: {len(orig) - len(orig_filtered)}')
print(f'  Added new rows:   {len(new)}')
print(f'  Net change:       {len(merged) - len(orig):+d}')
print(f'  Total models now: {len(merged)}')

# Per-task AUC summary
print('\n  Per-task best AUC:')
for task in sorted(merged['task'].unique()):
    sub = merged[merged['task'] == task]
    best = sub.loc[sub['auc'].idxmax()]
    print(f'    {task:25s}: n={len(sub):2d} models, '
          f'best AUC={best["auc"]:.3f} [{best["auc_lo"]:.3f}-{best["auc_hi"]:.3f}] '
          f'({best["name"]}, N={best["n_val"]})')

# Save log
log = {
    'backup_file': BACKUP_CSV,
    'replaced_tasks': sorted(new_tasks),
    'removed_old_rows': int(len(orig) - len(orig_filtered)),
    'added_new_rows': int(len(new)),
    'final_total_models': int(len(merged)),
}
with open(os.path.join(TBL_V3, 'merge_log.json'), 'w', encoding='utf-8') as f:
    json.dump(log, f, indent=2)
print(f'\n  Log: {TBL_V3}/merge_log.json')
