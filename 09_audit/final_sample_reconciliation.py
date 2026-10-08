"""Final comprehensive sample size reconciliation."""
import os, pandas as pd

BASE = r'D:\research\人脸识别营养\传染科'

print('=' * 70)
print('  FINAL SAMPLE SIZE RECONCILIATION')
print('=' * 70)

# 1. Internal SAM data
int_data = os.path.join(BASE, 'data', 'face_v4')
int_counts = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(int_data, cat)
    if os.path.exists(cd):
        int_counts[cat] = len([d for d in os.listdir(cd) if os.path.isdir(os.path.join(cd, d))])
int_total = sum(int_counts.values())
int_normal = int_counts.get('normal', 0)
int_jaun = int_total - int_normal
print(f'\n[1] Internal (face_v4):')
for k, v in int_counts.items():
    print(f'    {k}: {v}')
print(f'    TOTAL: {int_total}  (normal={int_normal}, jaundice={int_jaun})')

# 2. Internal manifest (authoritative)
mdf = pd.read_csv(os.path.join(BASE, 'data', 'clean_dataset_manifest.csv'))
print(f'\n[2] clean_dataset_manifest.csv: {len(mdf)} rows')
print(f'    {mdf["category"].value_counts().to_dict()}')

# 3. External data (filter empty)
ext_data = os.path.join(BASE, 'data', 'external_processed_v3')
ext_valid = {'normal': 0, 'jaundice': 0}
ext_empty = []
for cat in ['normal', 'jaundice']:
    cd = os.path.join(ext_data, cat)
    if not os.path.exists(cd):
        continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p):
            continue
        faces = [f for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')]
        if faces:
            ext_valid[cat] += 1
        else:
            ext_empty.append(f'{cat}/{pid}')
ext_total = sum(ext_valid.values())
print(f'\n[3] External (external_processed_v3):')
print(f'    normal (valid): {ext_valid["normal"]}')
print(f'    jaundice (valid): {ext_valid["jaundice"]}  (excluded {len(ext_empty)} empty: {ext_empty})')
print(f'    TOTAL VALID: {ext_total}')

# 4. Baseline questionnaire
df = pd.read_excel(os.path.join(BASE, 'baseline6.17.xlsx'))
g = df['group'].value_counts().to_dict()
print(f'\n[4] baseline6.17.xlsx: {len(df)} rows')
print(f'    Group 1 (non-jaundice): {g.get(1, 0)}')
print(f'    Group 2 (jaundice):     {g.get(2, 0)}')
print(f'    TOTAL: {len(df)}')

# 5. Bootstrap CSV n_val per task
print(f'\n[5] Bootstrap validation set sizes (from v3 CSV):')
bs = pd.read_csv(os.path.join(BASE, 'results', 'tables_v3', 'bootstrap_full_metrics_v3.csv'))
for task in sorted(bs['task'].unique()):
    sub = bs[bs['task'] == task]
    n_vals = sub['n_val'].unique()
    print(f'    {task:30s}: n_val = {sorted(n_vals)} ({len(sub)} models)')

# 6. Summary table
print(f'\n{"=" * 70}')
print(f'  SUMMARY TABLE (for manuscript)')
print(f'{"=" * 70}')
print(f'  Enrolled (questionnaire, internal):   939  (baseline6.17.xlsx: {len(df)})')
print(f'    - Non-jaundice controls:            {g.get(1, 0)}')
print(f'    - Jaundice (internal):              {g.get(2, 0)}')
print(f'  With usable facial images (internal): {int_total}')
print(f'  External validation cohort:           {ext_total}')
print(f'    - Normal:                           {ext_valid["normal"]}')
print(f'    - Jaundice:                         {ext_valid["jaundice"]}')
print(f'  Domain-adapted held-out (20%% ext):   19  (8 normal + 11 jaundice)')
print(f'\n  Manuscript Table 1 current: 625 + 314 + 57 = 996')
print(f'  Recommended: 625 + 313 + 58 = 996 (fix 314->313, 57->58)')
print(f'  Or: clarify 939 internal + 58 external = 997 ≈ 996')
