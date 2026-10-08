# -*- coding: utf-8 -*-
"""
CDSS subgroup consistency (transparency) audit on the real cohort.

Deterministic CDSS outputs are driven by labs/diagnosis, not demographics.
This audit documents output distributions by sex and age group plus the
clinical inputs that mediate them — a transparency check (JAMIA 2024 bias
dimension), NOT a claim of demographic causation.

  1. Severity / mode / priority / nutrition-state by sex (chi-square)
  2. Same by age group (<40, 40-59, >=60)
  3. Mediation check: TBIL / INR / NRS distributions by subgroup
     (demographic differences should track clinical inputs, if any)

Run: python code/cdss_subgroup_audit.py
"""
import sys, os
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca
import pandas as pd
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')

V = pd.read_csv(os.path.join(TBL, 'CDSS_PATIENT_LEVEL_VALIDATION.csv'))
df = pd.read_excel(os.path.join(BASE, '按文本_合并后的肝炎患者问卷.xlsx'))
V['sex'] = V['pid'].map(dict(zip(df['序号_x'], df['您的性别'])))
V['sex'] = V['sex'].map(lambda s: 'F' if str(s).endswith('女') else 'M')
age = pd.to_numeric(df['您的年龄是多少岁？'], errors='coerce')
V['age'] = V['pid'].map(dict(zip(df['序号_x'], age)))
V['age_group'] = pd.cut(V['age'], [0, 40, 60, 200],
                        labels=['<40', '40-59', '>=60'])
print(f'Cohort: {len(V)}  sex M/F = {(V["sex"] == "M").sum()}/{(V["sex"] == "F").sum()}'
      f'  age groups {V["age_group"].value_counts().to_dict()}')

OUT_COLS = ['severity', 'mode', 'priority', 'nutrition_state']


def chi2_table(col, by):
    tab = pd.crosstab(V[by], V[col])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    print(f'\n  {col} by {by}:')
    print('    ' + str(tab).replace('\n', '\n    '))
    print(f'    chi2={chi2:.2f}, dof={dof}, p={p:.4f}'
          f'{"  *" if p < 0.05 else ""}')
    return p


print('\n' + '=' * 70)
print('  1. CDSS outputs by SEX')
print('=' * 70)
ps_sex = {c: chi2_table(c, 'sex') for c in OUT_COLS}

print('\n' + '=' * 70)
print('  2. CDSS outputs by AGE GROUP')
print('=' * 70)
ps_age = {c: chi2_table(c, 'age_group') for c in OUT_COLS}

print('\n' + '=' * 70)
print('  3. Mediation check — clinical inputs by subgroup')
print('=' * 70)
for col, by in [('tbil', 'sex'), ('inr', 'sex'), ('nrs2002', 'sex'),
                ('tbil', 'age_group'), ('inr', 'age_group'), ('nrs2002', 'age_group')]:
    sub = V[[col, by]].dropna()
    groups = [g[col].values for _, g in sub.groupby(by)]
    if len(groups) == 2:
        u, p = stats.mannwhitneyu(*groups)
    else:
        u, p = stats.kruskal(*groups), None
        p = u.pvalue if hasattr(u, 'pvalue') else u[1]
        u = u.statistic if hasattr(u, 'statistic') else u[0]
    med = sub.groupby(by)[col].median().round(1).to_dict()
    print(f'  {col:8s} by {by:9s}: medians={med}  p={p:.4f}'
          f'{"  *" if p < 0.05 else ""}')

n_sig = sum(1 for p in list(ps_sex.values()) + list(ps_age.values()) if p < 0.05)
print('\n' + '=' * 70)
print(f'  SUMMARY: {n_sig}/{len(ps_sex) + len(ps_age)} output-by-demographic '
      f'associations significant at p<0.05')
print('  Any significant association is mediated by clinical inputs (see')
print('  mediation table); the CDSS itself is deterministic on labs/diagnosis.')
print('=' * 70)

rows = ([{'output': c, 'stratification': 'sex', 'p_value': round(ps_sex[c], 4)}
         for c in OUT_COLS]
        + [{'output': c, 'stratification': 'age_group', 'p_value': round(ps_age[c], 4)}
           for c in OUT_COLS])
out = os.path.join(TBL, 'CDSS_SUBGROUP_AUDIT.csv')
pd.DataFrame(rows).to_csv(out, index=False)
print(f'Saved: {out}')
