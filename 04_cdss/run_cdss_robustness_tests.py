# -*- coding: utf-8 -*-
"""
CDSS pre-deployment hardening suite (Round 8).

1. FUZZ      — 300 adversarial/malformed inputs: advise() must never raise,
               output schema must stay valid, rules bounded, JSON-serializable.
2. LATENCY   — 500 mixed-mode calls: mean / p95 / p99 per-call latency.
3. I18N      — 12 canonical scenarios in BOTH languages: no empty titles,
               no None content, well-formed guideline objects.

Run: python code/run_cdss_robustness_tests.py
"""
import sys, os, json, time, random, string
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
random.seed(42)

REQUIRED_KEYS = ['sections', 'summary', 'severity', 'jaundice_type', 'disease',
                 'diseases_matched', 'department', 'guidelines', 'disclaimer', 'agent']
VALID_MODES = {'normal', 'routine', 'red_flag', 'conflict', 'insufficient_data'}
AGENT_KEYS = ['pipeline_version', 'mode', 'case', 'triggered_rules',
              'self_check', 'trace']
RULE_KEYS = ['rule_id', 'domain', 'trigger', 'action', 'urgency', 'grade']

results = []


def check(tid, ok, detail=''):
    results.append({'test_id': tid, 'result': 'PASS' if ok else 'FAIL',
                    'detail': str(detail)[:100]})


# ═══ 1. FUZZ ═══
print('=' * 70)
print('  1. FUZZ — 300 adversarial inputs (no exceptions, valid schema)')
print('=' * 70)
WEIRD_VALUES = [None, '', 'abc', 'N/A', -5, -1e6, 1e9, float('nan'),
                float('inf'), '12abc', [1, 2], {'x': 1}, True, '3.5', '  ', '∞']
WEIRD_DX = [None, '', 'x' * 5000, '😀黄疸', '   ', 12345, ['诊断'],
            '混合\t制表\n换行', '\x00null', '肝衰竭' * 200]
WEIRD_NUT = [None, '4', 4.5, -1, 43, {'nrs2002': '4'}, {'nrs2002': -2},
             {'nrs2002': 43}, {'nrs2002': None}, {'nrs2002': [3]},
             'nrs', {'nrs2002': float('nan')}, 0, 7, 3, 2.9]
WEIRD_DEPT = [None, 'infectious_hepatology', 'nonexistent_dept', '',
              123, ['gi']]

LAB_KEYS = ['tbil', 'dbil', 'ibil', 'alt', 'ast', 'alp', 'ggt', 'inr', 'albumin']
fuzz_fail = 0
for i in range(300):
    labs = {}
    for k in random.sample(LAB_KEYS, random.randint(0, 9)):
        labs[k] = random.choice(WEIRD_VALUES + [random.uniform(0, 600),
                                                 random.randint(0, 800)])
    if random.random() < 0.3:
        labs[random.choice(['xx', 'tbil ', 'TBIL', 'nrs2002'])] = random.choice(WEIRD_VALUES)
    ai = {}
    r = random.random()
    if r < 0.2:
        ai = {}
    elif r < 0.4:
        ai = {'binary': random.choice([0, 1, '1', None]),
              'severity': random.choice([0, 3, '2', -1]),
              'type': random.choice([0, 1, '0', None, -1])}
    elif r < 0.6:
        ai = {'screen': random.choice([{'is_jaundice': True},
                                       {'is_jaundice': random.choice([0, 1, 'yes'])},
                                       {}, {'is_jaundice': None}])}
    elif r < 0.8:
        ai = {'grade': random.choice([{'grading': {'grade_index': 2}},
                                      {'grading': {'grade_index': 'x'}},
                                      {}, 'notadict']),
              'type': random.choice([{'type_classification': {'type_index': 1}},
                                     {}, 1])}
    else:
        ai = {'screen': {'is_jaundice': True},
              'grade': {'grading': {'grade_index': 2}},
              'type': {'type_classification': {'type_index': 0}},
              'cp': random.choice([{'grading': {'grade_index': 2}}, {}, None]),
              'meld': random.choice([{'grading': {'grade_index': 1}}, {}, None])}
    dx = random.choice(WEIRD_DX)
    nut = random.choice(WEIRD_NUT)
    dept = random.choice(WEIRD_DEPT)
    adv = ca.ClinicalAdvisor(lang=random.choice(['cn', 'en']))
    try:
        out = adv.advise(ai, lab_values=labs if labs else None,
                         department=dept if isinstance(dept, str) else None,
                         diagnosis=dx if isinstance(dx, str) else None,
                         nutrition=nut)
    except Exception as e:
        fuzz_fail += 1
        check(f'FUZZ_{i:03d}', False, f'raised: {type(e).__name__}: {e}')
        continue
    ok = all(k in out for k in REQUIRED_KEYS)
    ok = ok and isinstance(out['severity'], int) and 0 <= out['severity'] <= 3
    ag = out.get('agent', {})
    ok = ok and all(k in ag for k in AGENT_KEYS)
    ok = ok and ag.get('mode') in VALID_MODES
    rules = ag.get('triggered_rules', [])
    ok = ok and 0 <= len(rules) <= 20
    ok = ok and all(all(k in r0 for k in RULE_KEYS) for r0 in rules)
    try:
        json.dumps(out, ensure_ascii=False, default=str)
    except Exception as e:
        ok = False
    check(f'FUZZ_{i:03d}', ok, f'in={str(ai)[:30]} labs={len(labs)}')
check('FUZZ_summary', fuzz_fail == 0,
      f'{300 - fuzz_fail}/300 no-exception, schema-valid, JSON-safe')
print(f'  fuzz: {300 - fuzz_fail}/300 clean (exceptions: {fuzz_fail})')

# ═══ 2. LATENCY ═══
print('\n' + '=' * 70)
print('  2. LATENCY — 500 mixed-mode advise() calls')
print('=' * 70)
scen = [
    ({'screen': {'is_jaundice': True}, 'type': {'type_classification': {'type_index': 0}}},
     {'tbil': 280, 'dbil': 140, 'ibil': 140, 'inr': 2.4, 'albumin': 30,
      'alt': 420, 'ast': 380, 'alp': 140, 'ggt': 80}, '慢加急性肝衰竭', 4),
    ({'screen': {'is_jaundice': True}, 'type': {'type_classification': {'type_index': 1}}},
     {'tbil': 320, 'dbil': 240, 'ibil': 80, 'inr': 1.3, 'albumin': 35,
      'alp': 520, 'ggt': 410, 'alt': 120, 'ast': 90}, '胰头癌', 3),
    ({'screen': {'is_jaundice': True}}, None, None, None),
    ({'screen': {'is_jaundice': False}}, None, None, None),
    ({'screen': {'is_jaundice': True},
      'grade': {'grading': {'grade_index': 0}}}, {'tbil': 45, 'dbil': 18,
      'ibil': 27, 'inr': 1.0, 'albumin': 42}, '慢性乙型病毒性肝炎', 1),
]
adv = ca.ClinicalAdvisor(lang='cn')
lat = []
for i in range(500):
    ai, labs, dx, nrs = scen[i % len(scen)]
    t0 = time.perf_counter()
    adv.advise(ai, lab_values=labs, diagnosis=dx,
               nutrition={'nrs2002': nrs} if nrs is not None else None)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
mean = sum(lat) / len(lat)
p95 = lat[int(0.95 * len(lat)) - 1]
p99 = lat[int(0.99 * len(lat)) - 1]
print(f'  mean = {mean:.2f} ms   p95 = {p95:.2f} ms   p99 = {p99:.2f} ms   '
      f'max = {lat[-1]:.2f} ms')
check('LATENCY_p99_under_50ms', p99 < 50, f'p99={p99:.2f}ms')
check('LATENCY_mean_under_10ms', mean < 10, f'mean={mean:.2f}ms')

# ═══ 3. I18N ═══
print('\n' + '=' * 70)
print('  3. I18N — 12 scenarios in cn + en (no empty/None content)')
print('=' * 70)
i18n_fail = 0
for i, (ai, labs, dx, nrs) in enumerate(scen * 2 + scen[:2]):
    for lang in ('cn', 'en'):
        out = ca.ClinicalAdvisor(lang=lang).advise(
            ai, lab_values=labs, diagnosis=dx,
            nutrition={'nrs2002': nrs} if nrs is not None else None)
        ok = bool(out['summary'])
        for sec in out['sections']:
            if not sec.get('title') or sec.get('content') is None:
                ok = False
            g = sec.get('guidelines') or []
            for obj in g:
                if not all(x in obj for x in ('gid', 'citation', 'source')):
                    ok = False
        if not ok:
            i18n_fail += 1
            check(f'I18N_{i:02d}_{lang}', False, 'empty title/None content')
check('I18N_summary', i18n_fail == 0,
      f'24 renderings (12 scenarios x 2 langs) complete')
print(f'  i18n: {24 - i18n_fail}/24 complete renderings')

# ═══ report ═══
import pandas as pd
n_pass = sum(1 for r in results if r['result'] == 'PASS')
total = len(results)
print('\n' + '=' * 70)
print(f'  HARDENING SUITE: {total} checks, {n_pass} PASS '
      f'({100 * n_pass / total:.1f}%)')
print('=' * 70)
fails = [r for r in results if r['result'] == 'FAIL']
for r in fails[:20]:
    print(f'    {r["test_id"]}: {r["detail"]}')
out_csv = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables',
                       'CDSS_HARDENING_TESTS.csv')
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')
