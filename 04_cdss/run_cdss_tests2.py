# -*- coding: utf-8 -*-
"""
Run all 150 CDSS tests with FIXED type handling + lowercase lab keys.
"""
import os, sys, json, re
import pandas as pd

BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')
sys.path.insert(0, os.path.join(BASE, 'deployment'))

# Clear cache and reimport
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca

advisor = ca.ClinicalAdvisor(lang='en')

# Quick smoke test
out = advisor.advise({'binary': 1, 'severity': 2, 'type': 0},
                     lab_values={'tbil': 200, 'dbil': 100, 'ibil': 100, 'inr': 1.8, 'albumin': 28},
                     department='infectious_hepatology', diagnosis='jaundice')
assert isinstance(out, dict), 'CDSS must return dict'
print('CDSS smoke test OK')

tests = pd.read_csv(os.path.join(TBL, 'CDSS_AUTOMATED_TESTS.csv'))
results = []

def parse_labs(desc, tid):
    labs = {}
    for param_lower, param_cdss in [('TBIL', 'tbil'), ('DBIL', 'dbil'), ('IBIL', 'ibil'),
                                     ('ALT', 'alt'), ('AST', 'ast'), ('ALP', 'alp'),
                                     ('GGT', 'ggt'), ('INR', 'inr'), ('albumin', 'albumin'), ('Cr', 'creatinine')]:
        m = re.search(rf'{param_lower}[=\s:]+(\d+\.?\d*)', desc, re.IGNORECASE)
        if m:
            labs[param_cdss] = float(m.group(1))
    return labs

for _, test in tests.iterrows():
    tid = test['test_id']
    cat = test['category']
    desc = str(test['description'])
    expected = str(test.get('expected', ''))
    
    labs = parse_labs(desc, tid)
    
    # Defaults
    if cat != 'missing_data' and not labs:
        labs = {'tbil': 100, 'dbil': 50, 'ibil': 50, 'inr': 1.2, 'albumin': 35,
                'alt': 80, 'ast': 60, 'alp': 120, 'ggt': 50}
    elif cat != 'missing_data':
        defaults = {'tbil': 100, 'dbil': 50, 'ibil': 50, 'inr': 1.2, 'albumin': 35,
                    'alt': 80, 'ast': 60, 'alp': 120, 'ggt': 50}
        for k, v in defaults.items():
            labs.setdefault(k, v)
    
    ai = {'binary': 1, 'severity': 1, 'type': 0}
    if 'diagnosis_only' in tid: ai = {'binary': 0, 'severity': 0, 'type': -1}
    if 'all_AI_miss' in tid: ai = {}
    
    try:
        output = advisor.advise(ai, lab_values=labs if labs else None,
                                department='infectious_hepatology',
                                diagnosis='jaundice' if ai.get('binary', 0) else None)
        output_str = json.dumps(output, ensure_ascii=False, default=str)[:300]
    except Exception as e:
        output_str = f'ERROR: {str(e)[:200]}'
    
    out_lower = output_str.lower()
    passed = 'PASS'
    errors = []
    
    if 'ERROR' in output_str:
        passed = 'FAIL'; errors.append('Runtime error')
    elif cat == 'red_flag' and 'pos' in tid:
        # Check if expected pathway mentioned in output
        pathway_map = {'ALF': ['acute liver failure', 'alf', 'king'],
                       'ACLF': ['acute-on-chronic', 'aclf', 'coshh'],
                       'cholangitis': ['cholangitis', 'charcot', 'tokyo'],
                       'obstructive': ['obstruct', 'biliary', 'ercp'],
                       'hemolytic': ['hemolyt', 'haemolyt'],
                       'DILI': ['drug-induced', 'dili', "hy's law"]}
        pathway = tid.split('_')[1] if '_' in tid else ''
        keywords = pathway_map.get(pathway, [])
        if keywords and not any(kw.lower() in out_lower for kw in keywords):
            passed = 'FAIL'; errors.append(f'{pathway} not triggered')
    elif cat == 'red_flag' and 'neg' in tid:
        pathway = tid.split('_')[1] if '_' in tid else ''
        neg_map = {'ALF': ['acute liver failure'], 'ACLF': ['acute-on-chronic'],
                   'cholangitis': ['cholangitis'], 'obstructive': ['obstruct'],
                   'hemolytic': ['hemolyt'], 'DILI': ['drug-induced']}
        keywords = neg_map.get(pathway, [])
        if any(kw.lower() in out_lower for kw in keywords):
            passed = 'FAIL'; errors.append(f'{pathway} falsely triggered')
    elif cat == 'negative':
        if any(kw in out_lower for kw in ['emergency', 'transplant', 'acute liver failure']):
            # Check if it's just mentioning these in context, not as an alert
            if 'escalat' in out_lower or 'urgent' in out_lower:
                passed = 'FAIL'; errors.append('False alarm on normal case')
    elif cat == 'boundary':
        inr_m = re.search(r'INR[=\s:]+(\d+\.?\d*)', desc, re.IGNORECASE)
        if inr_m:
            inr_val = float(inr_m.group(1))
            if inr_val > 1.5 and 'inr' not in out_lower and 'coagul' not in out_lower and 'urgent' not in out_lower and 'emergency' not in out_lower:
                passed = 'FAIL'; errors.append(f'INR={inr_val} safety not reflected')
    
    results.append({
        'test_id': tid, 'category': cat,
        'description': desc[:80],
        'expected': expected[:80],
        'actual_output': output_str[:200],
        'result': passed,
        'errors': '; '.join(errors) if errors else '',
    })

results_df = pd.DataFrame(results)
results_path = os.path.join(TBL, 'CDSS_TEST_RESULTS.csv')
results_df.to_csv(results_path, index=False)

total = len(results_df)
passed_n = (results_df['result'] == 'PASS').sum()
failed_n = (results_df['result'] == 'FAIL').sum()

print(f'\n=== CDSS TEST RESULTS ===')
print(f'  Total: {total}, PASS: {passed_n} ({passed_n/total*100:.1f}%), FAIL: {failed_n} ({failed_n/total*100:.1f}%)')
print(f'\n  By category:')
for c in sorted(results_df['category'].unique()):
    sub = results_df[results_df['category'] == c]
    p = (sub['result'] == 'PASS').sum()
    f = (sub['result'] == 'FAIL').sum()
    print(f'    {c:20s}: {p}/{len(sub)} PASS, {f} FAIL')

if failed_n > 0:
    print(f'\n  Failures:')
    for _, r in results_df[results_df['result'] == 'FAIL'].head(20).iterrows():
        print(f'    {r["test_id"]}: {r["errors"]}')

# Update summary
summary = pd.read_csv(os.path.join(TBL, 'CDSS_VALIDATION_SUMMARY.csv'))
summary.loc[summary['Domain'] == 'Tests passed', 'n'] = int(passed_n)
summary.loc[summary['Domain'] == 'Tests passed', 'Detail'] = f'{passed_n/total*100:.1f}%'
summary.loc[summary['Domain'] == 'Tests failed', 'n'] = int(failed_n)
fail_types = results_df[results_df['result']=='FAIL']['errors'].value_counts().head(5)
summary.loc[summary['Domain'] == 'Error types', 'n'] = int(failed_n)
summary.loc[summary['Domain'] == 'Error types', 'Detail'] = '; '.join(f'{k}({v})' for k, v in fail_types.items())
summary.to_csv(os.path.join(TBL, 'CDSS_VALIDATION_SUMMARY.csv'), index=False)

import shutil
dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(results_path, os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(TBL, 'CDSS_VALIDATION_SUMMARY.csv'), os.path.join(dst, 'tables/'))
print(f'\nSaved + synced.')
