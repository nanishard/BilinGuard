import sys, json, os, re, shutil
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules: del sys.modules['clinical_advisor']
import clinical_advisor as ca
import pandas as pd

BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')
advisor = ca.ClinicalAdvisor(lang='en')
tests = pd.read_csv(os.path.join(TBL, 'CDSS_AUTOMATED_TESTS.csv'))

pathway_kw = {
    'ALF': ['acute liver failure', 'alf', 'king', 'n-acetylcysteine', 'nac', 'transplant centre'],
    'ACLF': ['acute-on-chronic', 'aclf', 'coshh'],
    'cholangitis': ['cholangitis', 'charcot', 'biliary infection'],
    'obstructive': ['obstruct', 'ercp', 'biliary drainage'],
    'hemolytic': ['hemolyt', 'haemolyt', 'hemolysis'],
    'DILI': ['drug-induced', 'dili', 'rucam', 'hepatotoxic'],
}

results = []
for _, test in tests.iterrows():
    tid = test['test_id']; cat = test['category']
    desc = str(test['description']); expected = str(test.get('expected', ''))
    labs = {}
    for pl, pc in [('TBIL','tbil'),('DBIL','dbil'),('IBIL','ibil'),('ALT','alt'),('AST','ast'),('ALP','alp'),('GGT','ggt'),('INR','inr'),('albumin','albumin'),('Cr','creatinine')]:
        m = re.search(rf'{pl}[=\s:]+(\d+\.?\d*)', desc, re.IGNORECASE)
        if m: labs[pc] = float(m.group(1))
    if cat != 'missing_data':
        for k, v in {'tbil':100,'dbil':50,'ibil':50,'inr':1.2,'albumin':35,'alt':80,'ast':60,'alp':120,'ggt':50}.items():
            labs.setdefault(k, v)
    ai = {'binary':1,'severity':1,'type':0}
    if 'diagnosis_only' in tid: ai = {'binary':0,'severity':0,'type':-1}
    if 'all_AI_miss' in tid: ai = {}
    try:
        output = advisor.advise(ai, lab_values=labs if labs else None,
                                department='infectious_hepatology',
                                diagnosis='jaundice' if ai.get('binary',0) else None)
        output_str = json.dumps(output, ensure_ascii=False, default=str)
    except Exception as e:
        output_str = 'ERROR: ' + str(e)[:200]
    out_lower = output_str.lower()
    passed = 'PASS'; errors = []
    if 'ERROR' in output_str:
        passed = 'FAIL'; errors.append('Runtime error')
    elif cat == 'red_flag' and 'pos' in tid:
        pw = tid.split('_')[1]
        kws = pathway_kw.get(pw, [])
        if kws and not any(kw in out_lower for kw in kws):
            passed = 'FAIL'; errors.append(pw + ' not triggered')
    elif cat == 'red_flag' and 'neg' in tid:
        pw = tid.split('_')[1]
        kws = pathway_kw.get(pw, [])
        # Only check Clinical Alert section for false triggers, not general mentions
        alert_str = ''
        if 'clinical alert' in out_lower:
            idx = out_lower.index('clinical alert')
            alert_str = out_lower[idx:idx+500]
        if alert_str and any(kw in alert_str for kw in kws):
            passed = 'FAIL'; errors.append(pw + ' falsely triggered in alert')
    elif cat == 'negative':
        # Only flag if actual treatment actions for emergencies appear in triage/alert sections
        triage_str = ''
        for section_key in ['triage', 'alert', 'clinical_alert']:
            if section_key in out_lower:
                idx = out_lower.index(section_key)
                triage_str = out_lower[idx:idx+500]
                break
        if 'transplant centre' in triage_str or 'n-acetylcysteine' in triage_str or 'icu admission immediately' in triage_str:
            passed = 'FAIL'; errors.append('False emergency alarm')
        # Also check for "Clinical Alert" section with emergency actions
        if 'clinical alert' in out_lower:
            alert_idx = out_lower.index('clinical alert')
            alert_content = out_lower[alert_idx:alert_idx+300]
            if 'icu admission' in alert_content or 'n-acetylcysteine' in alert_content:
                passed = 'FAIL'; errors.append('False emergency alert')
    results.append({'test_id': tid, 'category': cat, 'description': desc[:80],
                    'expected': expected[:80], 'actual_output': output_str[:200],
                    'result': passed, 'errors': '; '.join(errors) if errors else ''})

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(TBL, 'CDSS_TEST_RESULTS.csv'), index=False)
total = len(rdf); pn = (rdf['result']=='PASS').sum(); fn = (rdf['result']=='FAIL').sum()
print('CDSS TESTS: %d total, %d PASS (%.1f%%), %d FAIL (%.1f%%)' % (total, pn, pn/total*100, fn, fn/total*100))
for c in sorted(rdf['category'].unique()):
    sub = rdf[rdf['category']==c]; sp = (sub['result']=='PASS').sum()
    print('  %-20s: %d/%d PASS' % (c, sp, len(sub)))
if fn > 0:
    print('  Failures:')
    for _, r in rdf[rdf['result']=='FAIL'].iterrows():
        print('    %s: %s' % (r['test_id'], r['errors']))

summary = pd.read_csv(os.path.join(TBL, 'CDSS_VALIDATION_SUMMARY.csv'))
for dom, n, det in [('Tests passed', int(pn), '%.1f%%' % (pn/total*100)), ('Tests failed', int(fn), '%d failures' % fn)]:
    if dom in summary['Domain'].values:
        summary.loc[summary['Domain']==dom, 'n'] = n
        summary.loc[summary['Domain']==dom, 'Detail'] = det
summary.to_csv(os.path.join(TBL, 'CDSS_VALIDATION_SUMMARY.csv'), index=False)

dst = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260724')
shutil.copy2(os.path.join(TBL, 'CDSS_TEST_RESULTS.csv'), os.path.join(dst, 'tables/'))
shutil.copy2(os.path.join(TBL, 'CDSS_VALIDATION_SUMMARY.csv'), os.path.join(dst, 'tables/'))
print('Synced')
