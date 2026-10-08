# -*- coding: utf-8 -*-
"""
CDSS threshold-sensitivity analysis on the real 301-admission cohort.

Decision stability under input perturbation:
  1. TBIL x0.90 / x0.95 / x1.05 / x1.10  -> severity-class shift counts
  2. INR +0.1 / -0.1                     -> safety-gate / red-flag flips
     (near-threshold census: INR in [1.4, 1.6])
  3. NRS-2002 +1 / -1                    -> nutrition-risk state flips
     (near-threshold census: NRS in {2, 3})
  4. TBIL +5 umol/L around each boundary (34/85/171) -> crossing counts

Run: python code/cdss_threshold_sensitivity.py
"""
import sys, os
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')
XL = os.path.join(BASE, '按文本_合并后的肝炎患者问卷.xlsx')

df = pd.read_excel(XL)
LAB_MAP = {
    'tbil': '26、总胆红素值（umol/L）', 'ibil': '27、间接胆红素值（umol/L）',
    'dbil': '28、直接胆红素值（umol/L）', 'alt': '31、ALT丙氨酸氨基转移酶（IU/L）',
    'ast': '32、AST门冬氨酸氨基转移酶（IU/L）', 'ggt': '33、GGT谷氨酰转肽酶（IU/L）',
    'alp': '34、ALP碱性磷酸酶（IU/L）', 'inr': '36、INR 国际标准化比值',
    'albumin': '42、白蛋白（g/L）',
}
rows = []
for _, r in df.iterrows():
    labs = {}
    for k, col in LAB_MAP.items():
        v = pd.to_numeric(r.get(col), errors='coerce')
        if pd.notna(v):
            labs[k] = float(v)
    dx = str(r.get('3、入院主要诊断') or '')
    nrs = pd.to_numeric(r.get('14、NRS2002评分'), errors='coerce')
    nutrition = ({'nrs2002': float(nrs)} if pd.notna(nrs) and 0 <= nrs <= 7 else None)
    rows.append({'pid': r['序号_x'], 'labs': labs, 'dx': dx, 'nutrition': nutrition})
print(f'Cohort: {len(rows)} admissions')


def run_all(labs_override=None, nut_override=None):
    adv = ca.ClinicalAdvisor(lang='en')
    out = []
    for rec in rows:
        labs = labs_override(rec) if labs_override else rec['labs']
        nut = nut_override(rec) if nut_override else rec['nutrition']
        o = adv.advise(None, lab_values=labs if labs else None,
                       diagnosis=rec['dx'] or None, nutrition=nut)
        out.append((o['severity'],
                    o['agent']['self_check']['alert_priority'],
                    (o['agent'].get('case', {}) or {}).get('nutrition_state')))
    return out


base = run_all()
base_sev = np.array([b[0] for b in base])
base_pri = [b[1] for b in base]
base_nut = [b[2] for b in base]

# ═══ 1. TBIL multiplicative perturbation ═══
print('\n' + '=' * 70)
print('  1. TBIL perturbation -> severity stability')
print('=' * 70)
for factor, label in [(0.90, 'x0.90'), (0.95, 'x0.95'), (1.05, 'x1.05'), (1.10, 'x1.10')]:
    def perturb(rec, f=factor):
        labs = dict(rec['labs'])
        if 'tbil' in labs:
            labs['tbil'] = labs['tbil'] * f
        return labs
    sev = np.array([b[0] for b in run_all(perturb)])
    changed = int((sev != base_sev).sum())
    print(f'  TBIL {label}: severity changed in {changed}/{len(rows)} '
          f'({100 * changed / len(rows):.1f}%) admissions')

# ═══ 2. INR perturbation + near-threshold census ═══
print('\n' + '=' * 70)
print('  2. INR perturbation -> safety-gate stability')
print('=' * 70)
inr_vals = np.array([r['labs'].get('inr', np.nan) for r in rows], dtype=float)
near = int(np.nansum((inr_vals >= 1.4) & (inr_vals <= 1.6)))
print(f'  INR near-threshold band [1.4, 1.6]: {near}/{len(rows)} admissions')
for delta, label in [(-0.1, '-0.1'), (+0.1, '+0.1')]:
    def perturb(rec, d=delta):
        labs = dict(rec['labs'])
        if 'inr' in labs:
            labs['inr'] = max(0.5, labs['inr'] + d)
        return labs
    out = run_all(perturb)
    sev = np.array([b[0] for b in out])
    pri = [b[1] for b in out]
    sev_ch = int((sev != base_sev).sum())
    pri_ch = sum(1 for a, b in zip(pri, base_pri) if a != b)
    print(f'  INR {label}: severity changed {sev_ch}, priority changed {pri_ch}')

# ═══ 3. NRS perturbation + near-threshold census ═══
print('\n' + '=' * 70)
print('  3. NRS-2002 perturbation -> nutrition-state stability')
print('=' * 70)
nrs_vals = np.array([r['nutrition']['nrs2002'] if r['nutrition'] else np.nan
                     for r in rows])
near_nrs = int(np.nansum(np.isin(nrs_vals, [2, 3])))
print(f'  NRS at threshold band {{2, 3}}: {near_nrs}/{len(rows)} admissions')
for delta, label in [(-1, '-1'), (+1, '+1')]:
    def perturb(rec, d=delta):
        if not rec['nutrition']:
            return None
        n = rec['nutrition']['nrs2002'] + d
        return {'nrs2002': n} if 0 <= n <= 7 else None
    out = run_all(None, perturb)
    nut = [b[2] for b in out]
    ch = sum(1 for a, b in zip(nut, base_nut) if a != b)
    print(f'  NRS {label}: nutrition state changed {ch}/{len(rows)} '
          f'({100 * ch / len(rows):.1f}%)')

# ═══ 4. TBIL +5 around each boundary ═══
print('\n' + '=' * 70)
print('  4. TBIL +5 umol/L crossing census per severity boundary')
print('=' * 70)
for bnd, lo, hi in [(34, 29, 39), (85, 80, 90), (171, 166, 176)]:
    n = int(np.nansum((inr_vals * 0 + pd.Series([r['labs'].get('tbil', np.nan)
                                                 for r in rows]).values >= lo)
                      & (pd.Series([r['labs'].get('tbil', np.nan)
                                    for r in rows]).values <= hi)))
    print(f'  boundary {bnd} umol/L (±5): {n} admissions within ±5 of threshold')

summary = pd.DataFrame([
    ('Cohort', len(rows), 'real admissions'),
    ('TBIL x0.95/x1.05 severity shifts', '-', 'see console'),
    ('INR near-threshold [1.4,1.6]', f'{near}/{len(rows)}',
     f'{100 * near / len(rows):.1f}%'),
    ('NRS near-threshold {2,3}', f'{near_nrs}/{len(rows)}',
     f'{100 * near_nrs / len(rows):.1f}%'),
], columns=['Metric', 'Value', 'Detail'])
summary.to_csv(os.path.join(TBL, 'CDSS_THRESHOLD_SENSITIVITY.csv'), index=False)
print(f'\nSaved: {os.path.join(TBL, "CDSS_THRESHOLD_SENSITIVITY.csv")}')
