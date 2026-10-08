# -*- coding: utf-8 -*-
"""
CDSS v3 patient-level retrospective validation on the real 301-admission cohort.

Inputs : 按文本_合并后的肝炎患者问卷.xlsx (admission diagnosis, full labs, Child-Pugh class)
Engine : deployment/clinical_advisor.py (v3 agent pipeline)
Mode   : clinical-data path (labs + diagnosis + Child-Pugh; no AI imaging outputs)

Analyses
  A. Disease-inference reproduction on real free-text diagnoses vs the
     documented distribution (CLINICAL_ADVISOR_GUIDELINES.md §3)
  A2. Nutrition risk screening integration (NRS-2002 -> v3.1 nutrition module)
  B. CDSS severity vs Child-Pugh class agreement (external reference, n=67)
  C. Red-flag pathway trigger rates for diseases with an expected pathway
  D. Cohort safety profile: mode / alert priority / confidence / alert burden
  E. Data-completeness profile (critical labs)
  F. Stratified 30-case clinician review workbook (appropriateness scoring)
"""
import sys, os, json, re
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')
XLSX = os.path.join(BASE, '按文本_合并后的肝炎患者问卷.xlsx')

# ── documented expected distribution (guidelines doc §3, 301 real admissions) ──
EXPECTED_DIST = {
    'aclf': 87, 'decomp_cirrhosis': 86, 'hcc': 23, 'unexplained': 20,
    'chronic_viral': 19, 'ald': 16, 'dili': 13, 'salf': 9, 'alf': 6,
    'aih': 5, 'pbc': 5, 'choledocholithiasis': 4, 'hemolysis': 3,
    'cholangitis': 3, 'hev': 2,
}
# diseases whose profile carries an expected red-flag pathway
EXPECTED_PATHWAY = {
    'aclf': 'aclf', 'salf': 'acute_liver_failure', 'alf': 'acute_liver_failure',
    'dili': 'dili_urgent', 'hemolysis': 'hemolytic_crisis',
    'cholangitis': 'acute_cholangitis',
    'choledocholithiasis': 'obstructive_jaundice',
}

df = pd.read_excel(XLSX)
advisor = ca.ClinicalAdvisor(lang='cn')

LAB_MAP = {
    'tbil': '26、总胆红素值（umol/L）',
    'ibil': '27、间接胆红素值（umol/L）',
    'dbil': '28、直接胆红素值（umol/L）',
    'alt': '31、ALT丙氨酸氨基转移酶（IU/L）',
    'ast': '32、AST门冬氨酸氨基转移酶（IU/L）',
    'ggt': '33、GGT谷氨酰转肽酶（IU/L）',
    'alp': '34、ALP碱性磷酸酶（IU/L）',
    'inr': '36、INR 国际标准化比值',
    'albumin': '42、白蛋白（g/L）',
}
CP_MAP = {'A': 0, 'B': 1, 'C': 2}

rows = []
for _, r in df.iterrows():
    pid = r['序号_x']
    dx = str(r.get('3、入院主要诊断') or '')
    labs, labs_raw = {}, {}
    for k, col in LAB_MAP.items():
        v = pd.to_numeric(r.get(col), errors='coerce')
        if pd.notna(v):
            labs[k] = float(v)
            labs_raw[k] = float(v)
    results = {}
    cp_class = r.get('Child-pugh分级')
    if pd.notna(cp_class) and str(cp_class).strip().upper() in CP_MAP:
        results['cp'] = {'grading': {
            'grade_index': CP_MAP[str(cp_class).strip().upper()],
            'prediction': f'Child-Pugh {str(cp_class).strip().upper()}'}}
    nrs_v = pd.to_numeric(r.get('14、NRS2002评分'), errors='coerce')
    nutrition = None
    if pd.notna(nrs_v) and 0 <= nrs_v <= 7:
        nutrition = {'nrs2002': float(nrs_v)}
    try:
        out = advisor.advise(results if results else None,
                             lab_values=labs if labs else None,
                             diagnosis=dx if dx.strip() else None,
                             nutrition=nutrition)
        ag = out.get('agent', {})
        rf = [x['rule_id'] for x in ag.get('triggered_rules', [])
              if x['domain'] == 'red_flag']
        nut_rule = next((x['rule_id'] for x in ag.get('triggered_rules', [])
                         if x['domain'] == 'nutrition'), None)
        pathway = rf[0].replace('REDFLAG_', '') if rf else None
        rows.append({
            'pid': pid, 'diagnosis': dx[:60],
            'tbil': labs.get('tbil'), 'inr': labs.get('inr'),
            'cp_class': (str(cp_class).strip().upper()
                         if pd.notna(cp_class) else None),
            'inferred_disease': out.get('disease'),
            'severity': out.get('severity'),
            'mode': ag.get('mode'),
            'priority': ag.get('self_check', {}).get('alert_priority'),
            'confidence': ag.get('self_check', {}).get('confidence'),
            'pathway': pathway,
            'n_rules': len(ag.get('triggered_rules', [])),
            'nutrition_state': ag.get('case', {}).get('nutrition_state'),
            'nutrition_rule': nut_rule,
            'n_critical_labs_missing': len(ag.get('case', {})
                                           .get('labs_missing_critical', [])),
        })
    except Exception as e:
        rows.append({'pid': pid, 'diagnosis': dx[:60], 'error': str(e)[:100]})

V = pd.DataFrame(rows)
n_err = int(V['error'].notna().sum()) if 'error' in V.columns else 0
V = V[V['error'].isna()].drop(columns=['error'], errors='ignore') if 'error' in V.columns else V
print(f'Cohort: {len(V)} admissions processed, {n_err} runtime errors')

# ═══ A. disease inference reproduction ═══
print('\n' + '=' * 70)
print('  A. Disease inference on real free-text diagnoses')
print('=' * 70)
dist = V['inferred_disease'].value_counts()
mismatch = []
for did, exp_n in EXPECTED_DIST.items():
    got = int(dist.get(did, 0))
    flag = 'OK' if got == exp_n else 'MISMATCH'
    if got != exp_n:
        mismatch.append((did, exp_n, got))
    print(f'  {did:22s} expected={exp_n:3d} observed={got:3d}  {flag}')
extra = {d: int(n) for d, n in dist.items() if d not in EXPECTED_DIST}
if extra:
    print('  unexpected categories:', extra)
exact_match = int(sum(1 for d, e in EXPECTED_DIST.items() if int(dist.get(d, 0)) == e))
print(f'  -> {exact_match}/{len(EXPECTED_DIST)} categories reproduce exactly')

# ═══ A2. nutrition risk screening integration (v3.1) ═══
print('\n' + '=' * 70)
print('  A2. Nutrition risk screening (NRS-2002 -> CDSS nutrition module)')
print('=' * 70)
nrs = pd.to_numeric(df['14、NRS2002评分'], errors='coerce')
nrs_valid = nrs[(nrs >= 0) & (nrs <= 7)]
V['nrs2002'] = V['pid'].map(dict(zip(df.loc[nrs_valid.index, '序号_x'], nrs_valid)))
n_scored = int(V['nrs2002'].notna().sum())
n_at_risk = int((V['nrs2002'] >= 3).sum())
n_invalid = int(nrs.notna().sum()) - len(nrs_valid)
nut_rules = V['nutrition_rule'].value_counts().to_dict()
print(f'  NRS-2002 available: {n_scored}/{len(V)} (out-of-range: '
      f'{int(nrs.notna().sum()) - len(nrs_valid)})')
print(f'  at risk (NRS >= 3): {n_at_risk}/{n_scored} '
      f'({100 * n_at_risk / max(n_scored, 1):.1f}%)')
print(f'  nutrition rule distribution: {nut_rules}')
agree_nut = int(((V['nrs2002'] >= 3) &
                 (V['nutrition_rule'] == 'NUTRITION_RISK')).sum())
no_risk_ok = int(((V['nrs2002'] < 3) &
                  (V['nutrition_rule'] == 'NUTRITION_RESCREEN')).sum())
print(f'  NUTRITION_RISK fires for at-risk: {agree_nut}/{n_at_risk}')
print(f'  NUTRITION_RESCREEN for below-threshold: {no_risk_ok}/'
      f'{n_scored - n_at_risk}')
unscored_screen = int(((V['nrs2002'].isna()) &
                       (V['nutrition_rule'] == 'NUTRITION_SCREEN')).sum())
print(f'  NUTRITION_SCREEN prompts for unscreened: {unscored_screen}/'
      f'{int(V["nrs2002"].isna().sum())}')

# ═══ B. severity vs Child-Pugh agreement ═══
print('\n' + '=' * 70)
print('  B. CDSS severity vs Child-Pugh class (external reference)')
print('=' * 70)
cpv = V[V['cp_class'].notna()].copy()
cpv['cp_floor'] = cpv['cp_class'].map({'A': 0, 'B': 2, 'C': 3})
agree = (cpv['severity'] >= cpv['cp_floor']).sum()
print(f'  n with Child-Pugh class: {len(cpv)}')
print(f'  CDSS severity >= CP-implied floor: {agree}/{len(cpv)} '
      f'({100 * agree / len(cpv):.1f}%)')
for c in ['A', 'B', 'C']:
    sub = cpv[cpv['cp_class'] == c]
    if len(sub):
        print(f'    CP-{c}: n={len(sub)}, CDSS severity median={int(sub["severity"].median())}, '
              f'distribution={sub["severity"].value_counts().sort_index().to_dict()}')
b_detail = cpv[cpv['cp_class'] == 'B']
under_b = b_detail[b_detail['severity'] < 2]
print(f'  CP-B under-escalated (severity<2): {len(under_b)} cases'
      + (f' pids={list(under_b["pid"].head(5))}' if len(under_b) else ''))

# ═══ C. red-flag pathway trigger rates ═══
print('\n' + '=' * 70)
print('  C. Red-flag pathway trigger for diseases with expected pathway')
print('=' * 70)
for did, exp_pw in EXPECTED_PATHWAY.items():
    sub = V[V['inferred_disease'] == did]
    if not len(sub):
        continue
    hit = (sub['pathway'] == exp_pw).sum()
    alt = sub['pathway'].notna().sum()
    print(f'  {did:22s} n={len(sub):3d}  expected pathway fired: {hit:3d} '
          f'({100 * hit / len(sub):5.1f}%)   any red-flag: {alt:3d}')

# ═══ D. cohort safety profile ═══
print('\n' + '=' * 70)
print('  D. Cohort safety profile (mode / priority / confidence / burden)')
print('=' * 70)
for col in ['mode', 'priority', 'confidence', 'severity']:
    print(f'  {col:11s}: {V[col].value_counts().to_dict()}')
print(f'  alert burden: rules/case mean={V["n_rules"].mean():.1f} '
      f'median={V["n_rules"].median():.0f} max={V["n_rules"].max()}')
rf_any = V['pathway'].notna().sum()
print(f'  red-flag firing: {rf_any}/{len(V)} ({100 * rf_any / len(V):.1f}%) admissions')
high = (V['priority'] == 'HIGH').sum()
print(f'  HIGH priority : {high}/{len(V)} ({100 * high / len(V):.1f}%)')

# ═══ E. data completeness ═══
print('\n' + '=' * 70)
print('  E. Critical-lab completeness')
print('=' * 70)
print(f'  0 missing : {(V["n_critical_labs_missing"] == 0).sum()}/{len(V)}')
print(f'  >=1 missing: {(V["n_critical_labs_missing"] >= 1).sum()}/{len(V)}')

# ═══ F. stratified clinician review sample ═══
print('\n' + '=' * 70)
print('  F. Stratified clinician review sample (30 cases)')
print('=' * 70)
V['stratum'] = V['mode'].fillna('?') + '_sev' + V['severity'].astype(str)
rng = np.random.default_rng(42)
picks = []
for (m, s), grp in V.groupby(['mode', 'severity']):
    take = min(len(grp), max(2, round(30 * len(grp) / len(V))))
    picks += list(rng.choice(grp.index.values, size=take, replace=False))
sample = V.loc[picks].head(30).copy()

review_rows = []
for _, r in sample.iterrows():
    src = df[df['序号_x'] == r['pid']].iloc[0]
    labs = {}
    for k, col in LAB_MAP.items():
        v = pd.to_numeric(src.get(col), errors='coerce')
        if pd.notna(v):
            labs[k] = float(v)
    results = {}
    if pd.notna(src.get('Child-pugh分级')) and str(src['Child-pugh分级']).strip().upper() in CP_MAP:
        results['cp'] = {'grading': {
            'grade_index': CP_MAP[str(src['Child-pugh分级']).strip().upper()],
            'prediction': f'Child-Pugh {str(src["Child-pugh分级"]).strip().upper()}'}}
    nrs_v = pd.to_numeric(src.get('14、NRS2002评分'), errors='coerce')
    nutrition = None
    if pd.notna(nrs_v) and 0 <= nrs_v <= 7:
        nutrition = {'nrs2002': float(nrs_v)}
    out = advisor.advise(results if results else None,
                         lab_values=labs if labs else None,
                         diagnosis=str(src.get('3、入院主要诊断') or ''),
                         nutrition=nutrition)
    section_txt = []
    for sec in out['sections']:
        c = sec['content']
        txt = json.dumps(c, ensure_ascii=False, default=str) if isinstance(c, (dict, list)) else str(c)
        section_txt.append(f"[{sec['title']}] {txt[:300]}")
    rules = [f"{x['rule_id']}({x['grade']})" for x in out['agent']['triggered_rules']]
    review_rows.append({
        'pid': r['pid'], '入院诊断': str(src.get('3、入院主要诊断') or '')[:60],
        'TBIL': r['tbil'], 'INR': r['inr'], 'Child-Pugh': r['cp_class'],
        'NRS2002': (r['nrs2002'] if pd.notna(r.get('nrs2002')) else None),
        'CDSS_mode': r['mode'], 'CDSS_severity': r['severity'],
        '营养状态': r.get('nutrition_state'),
        '触发规则': '; '.join(rules),
        '分诊结论': out['summary'],
        '临床建议全文': ' || '.join(section_txt)[:1200],
        '合并诊断一致性(1-5)': None, '建议适当性(1-5)': None,
        '安全覆盖(1-5)': None, '营养建议适当性(1-5)': None,
        '审阅者': None, '审评意见': None,
    })
R = pd.DataFrame(review_rows)
review_path = os.path.join(TBL, 'CDSS_CLINICIAN_REVIEW_SAMPLE.xlsx')
R.to_excel(review_path, index=False)
print(f'  saved {len(R)} stratified cases -> {review_path}')
print(f'  strata: {dict(V.loc[R.index, "stratum"].value_counts())}')

# ═══ persist validation table + summary ═══
V.drop(columns=['stratum'], errors='ignore').to_csv(
    os.path.join(TBL, 'CDSS_PATIENT_LEVEL_VALIDATION.csv'), index=False)

summary = pd.DataFrame([
    ('Cohort processed', len(V), f'{n_err} runtime errors'),
    ('Disease categories reproducing documented distribution', f'{exact_match}/{len(EXPECTED_DIST)}',
     'keyword inference on real free text'),
    ('NRS-2002 screened', f'{n_scored}/{len(V)}',
     f'{n_at_risk} at risk (>=3); NUTRITION_RISK fired {agree_nut}/{n_at_risk}'),
    ('Severity >= Child-Pugh floor (n with CP class)', f'{agree}/{len(cpv)}',
     f'{100 * agree / max(len(cpv), 1):.1f}% external agreement'),
    ('Red-flag pathway overall firing', f'{rf_any}/{len(V)}',
     f'{100 * rf_any / len(V):.1f}%'),
    ('HIGH alert priority', f'{high}/{len(V)}', f'{100 * high / len(V):.1f}%'),
    ('Alert burden (rules per case)', f'{V["n_rules"].mean():.1f}',
     f'median {V["n_rules"].median():.0f}, max {V["n_rules"].max()}'),
    ('Critical labs complete', f'{(V["n_critical_labs_missing"] == 0).sum()}/{len(V)}',
     'tbil/dbil/inr/albumin'),
    ('Clinician review sample', len(R), 'stratified by mode x severity'),
], columns=['Metric', 'Value', 'Detail'])
summary.to_csv(os.path.join(TBL, 'CDSS_PATIENT_VALIDATION_SUMMARY.csv'), index=False)
print(f'\nSaved: CDSS_PATIENT_LEVEL_VALIDATION.csv, CDSS_PATIENT_VALIDATION_SUMMARY.csv')
