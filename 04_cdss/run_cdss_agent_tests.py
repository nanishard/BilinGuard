# -*- coding: utf-8 -*-
"""
CDSS v3 agent-layer rule-level validation.

Layer 1 (A): structural invariants of the agent payload across all 150
             legacy scenarios (schema, GRADE validity, trace completeness,
             priority/severity consistency, no false red-flag firing).
Layer 2 (B): true rule-id-level pathway verification with purpose-built
             positive/negative inputs for all 6 red-flag pathways.
             (The legacy keyword-based runner verified pathway presence via
             substring matching, which can pass incidentally; this layer
             asserts on agent.triggered_rules rule_ids.)
Layer 3 (C): document residual weakness of the legacy keyword runner.
"""
import sys, json, os, re
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')
advisor = ca.ClinicalAdvisor(lang='en')
tests = pd.read_csv(os.path.join(TBL, 'CDSS_AUTOMATED_TESTS.csv'))

VALID_GRADE = {'1A', '1B', '2A', '2B', '2C'}
URGENCY_ORDER = {'emergency': 0, 'urgent': 1, 'routine': 2, 'none': 3}
results = []


def parse_labs(desc):
    labs = {}
    for pl, pc in [('TBIL', 'tbil'), ('DBIL', 'dbil'), ('IBIL', 'ibil'),
                   ('ALT', 'alt'), ('AST', 'ast'), ('ALP', 'alp'),
                   ('GGT', 'ggt'), ('INR', 'inr'), ('albumin', 'albumin'),
                   ('Cr', 'creatinine')]:
        m = re.search(rf'{pl}[=\s:]+(\d+\.?\d*)', desc, re.IGNORECASE)
        if m:
            labs[pc] = float(m.group(1))
    return labs


def run_advise(ai, labs, diagnosis='jaundice'):
    return advisor.advise(ai, lab_values=labs if labs else None,
                          department='infectious_hepatology', diagnosis=diagnosis)


def check(tid, cat, desc, ok, detail=''):
    results.append({'test_id': tid, 'category': cat, 'description': desc[:80],
                    'assertion': detail[:100],
                    'result': 'PASS' if ok else 'FAIL'})


# ═══ PART A: structural invariants on all 150 legacy scenarios ═══
print('=' * 70)
print('  PART A: agent payload structural invariants (150 scenarios)')
print('=' * 70)

for _, test in tests.iterrows():
    tid = test['test_id']
    cat = test['category']
    desc = str(test['description'])
    labs = parse_labs(desc)
    if cat != 'missing_data':
        for k, v in {'tbil': 100, 'dbil': 50, 'ibil': 50, 'inr': 1.2,
                     'albumin': 35, 'alt': 80, 'ast': 60, 'alp': 120,
                     'ggt': 50}.items():
            labs.setdefault(k, v)
    ai = {'binary': 1, 'severity': 1, 'type': 0}
    if 'diagnosis_only' in tid:
        ai = {'binary': 0, 'severity': 0, 'type': -1}
    if 'all_AI_miss' in tid:
        ai = {}
    out = run_advise(ai, labs)
    ag = out.get('agent', {})

    ok_schema = all(k in ag for k in ('pipeline_version', 'mode', 'case',
                                      'triggered_rules', 'self_check', 'trace'))
    check(tid + ':schema', cat, desc, ok_schema,
          'agent payload has all 6 top-level keys')

    rules_ok = True
    for r in ag.get('triggered_rules', []):
        if not all(k in r for k in ('rule_id', 'domain', 'trigger', 'action',
                                    'urgency', 'grade')):
            rules_ok = False
        if r.get('grade') not in VALID_GRADE:
            rules_ok = False
    check(tid + ':rules', cat, desc, rules_ok,
          'every triggered rule is complete and GRADE-valid')

    stages = [t['stage'] for t in ag.get('trace', [])]
    trace_ok = stages[:5] == ['perceive', 'route', 'reason', 'critique',
                              'explain'][:len(stages)]
    check(tid + ':trace', cat, desc, trace_ok,
          f'trace stages ordered: {stages}')

    sev_rule = any(r['rule_id'] == f"SEVERITY_{out['severity']}"
                   for r in ag.get('triggered_rules', []))
    check(tid + ':sevrule', cat, desc, sev_rule,
          f"SEVERITY_{out['severity']} rule present and matches output severity")

    rf_rules = [r for r in ag.get('triggered_rules', []) if r['domain'] == 'red_flag']
    prio = ag.get('self_check', {}).get('alert_priority')
    prio_ok = ((prio == 'HIGH') if (rf_rules or out['severity'] >= 3)
               else (prio != 'HIGH'))
    check(tid + ':prio', cat, desc, prio_ok,
          f'alert_priority={prio} consistent with red-flag/severity state')

    if cat == 'negative':
        no_rf = len(rf_rules) == 0 and ag.get('mode') != 'red_flag'
        check(tid + ':norf', cat, desc, no_rf,
              'negative scenario fires no red-flag rule')

    if cat == 'missing_data':
        case = ag.get('case', {})
        crit_miss = case.get('labs_missing_critical', [])
        lims = ag.get('self_check', {}).get('limitations', [])
        miss_ok = len(crit_miss) > 0 and any('missing' in str(l).lower() for l in lims)
        check(tid + ':missflag', cat, desc, miss_ok,
              f'missing critical labs surfaced: {crit_miss}')

    if cat == 'boundary' and labs.get('inr') and labs['inr'] > 1.5:
        gate = next((c for c in ag.get('self_check', {}).get('checks', [])
                     if c['check'] == 'coagulopathy_safety_gate'), None)
        check(tid + ':gate', cat, desc,
              gate is not None and gate['status'] == 'FLAG',
              f'INR={labs["inr"]} flags coagulopathy safety gate')

# ═══ PART B: rule-id-level pathway verification (purpose-built inputs) ═══
print('\n' + '=' * 70)
print('  PART B: rule-id-level red-flag pathway verification (12 cases)')
print('=' * 70)

default_ai = {'binary': 1, 'severity': 1, 'type': 0}
full_labs = {'tbil': 100, 'dbil': 50, 'ibil': 50, 'inr': 1.2, 'albumin': 35,
             'alt': 80, 'ast': 60, 'alp': 120, 'ggt': 50}

PATHWAY_CASES = [
    # (tid, pathway_rule_id, ai, labs, diagnosis, expect_fired)
    ('AGT_ALF_pos', 'REDFLAG_acute_liver_failure', default_ai,
     {**full_labs, 'inr': 2.0, 'tbil': 200}, 'jaundice', True),
    ('AGT_ALF_neg', 'REDFLAG_acute_liver_failure', default_ai,
     {**full_labs, 'inr': 1.2}, 'jaundice', False),
    ('AGT_ACLF_pos', 'REDFLAG_aclf', default_ai,
     {**full_labs, 'tbil': 200, 'inr': 1.6}, '慢加急性肝衰竭 慢性乙型病毒性肝炎', True),
    ('AGT_ACLF_neg', 'REDFLAG_aclf', default_ai,
     full_labs, '慢性乙型病毒性肝炎', False),
    ('AGT_cholangitis_pos', 'REDFLAG_acute_cholangitis', default_ai,
     {**full_labs, 'tbil': 150, 'dbil': 90}, '急性胆管炎 发热 腹痛', True),
    ('AGT_cholangitis_neg', 'REDFLAG_acute_cholangitis', default_ai,
     full_labs, '慢性乙型病毒性肝炎', False),
    ('AGT_obstructive_pos', 'REDFLAG_obstructive_jaundice', default_ai,
     {**full_labs, 'alp': 350, 'dbil': 80}, 'jaundice', True),
    ('AGT_obstructive_neg', 'REDFLAG_obstructive_jaundice', default_ai,
     {**full_labs, 'alp': 120, 'dbil': 50}, 'jaundice', False),
    ('AGT_hemolytic_pos', 'REDFLAG_hemolytic_crisis', default_ai,
     {**full_labs, 'ibil': 120, 'dbil': 15}, '溶血性贫血 黄疸', True),
    ('AGT_hemolytic_neg', 'REDFLAG_hemolytic_crisis', default_ai,
     full_labs, '慢性乙型病毒性肝炎', False),
    ('AGT_DILI_pos', 'REDFLAG_dili_urgent', default_ai,
     {**full_labs, 'alt': 500}, '药物性肝损伤', True),
    ('AGT_DILI_neg', 'REDFLAG_dili_urgent', default_ai,
     full_labs, '慢性乙型病毒性肝炎', False),
]

for tid, rule_id, ai, labs, dx, expect in PATHWAY_CASES:
    out = run_advise(ai, labs, diagnosis=dx)
    ag = out.get('agent', {})
    fired_ids = [r['rule_id'] for r in ag.get('triggered_rules', [])]
    fired = rule_id in fired_ids
    check(tid, 'agent_pathway',
          f'{rule_id} {"expected" if expect else "must not fire"}',
          fired == expect,
          f'fired={fired}, expected={expect}, mode={ag.get("mode")}')

# ═══ PART C: legacy keyword-runner weakness documentation ═══
print('\n' + '=' * 70)
print('  PART C: legacy keyword-runner cross-check')
print('=' * 70)
out = run_advise(default_ai, full_labs)  # RF_ALF_pos legacy scenario verbatim
ag = out.get('agent', {})
fired_ids = [r['rule_id'] for r in ag.get('triggered_rules', [])]
legacy_verdict = ('The legacy RF_ALF_pos scenario (default labs, INR=1.2) '
                  'does NOT fire the ALF pathway rule; it passed the legacy '
                  'keyword runner only via incidental "alf" substring text.')
check('AGT_legacy_RF_ALF_pos_audit', 'audit', legacy_verdict,
      'REDFLAG_acute_liver_failure' not in fired_ids,
      f'fired rules: {fired_ids}')

# ═══ PART D: nutrition module (v3.1) rule-level verification ═══
print('\n' + '=' * 70)
print('  PART D: nutrition support module (v3.1) verification')
print('=' * 70)
NUTRITION_CASES = [
    # (tid, nutrition input, expected rule, expected urgency)
    ('AGT_NUT_risk_high', {'nrs2002': 4}, 'NUTRITION_RISK', 'urgent'),
    ('AGT_NUT_risk_boundary', 3, 'NUTRITION_RISK', 'urgent'),
    ('AGT_NUT_risk_top', 7, 'NUTRITION_RISK', 'urgent'),
    ('AGT_NUT_norisk', 2, 'NUTRITION_RESCREEN', 'routine'),
    ('AGT_NUT_norisk_zero', 0, 'NUTRITION_RESCREEN', 'routine'),
    ('AGT_NUT_unscreened', None, 'NUTRITION_SCREEN', 'routine'),
    ('AGT_NUT_invalid_high', {'nrs2002': 43}, 'NUTRITION_SCREEN', 'routine'),
    ('AGT_NUT_invalid_neg', {'nrs2002': -1}, 'NUTRITION_SCREEN', 'routine'),
]
for tid, nut_in, exp_rule, exp_urg in NUTRITION_CASES:
    out = advisor.advise(default_ai, lab_values=full_labs, nutrition=nut_in)
    nut_rules = [r for r in out['agent']['triggered_rules']
                 if r['domain'] == 'nutrition']
    ok = (len(nut_rules) == 1 and nut_rules[0]['rule_id'] == exp_rule
          and nut_rules[0]['urgency'] == exp_urg
          and nut_rules[0]['grade'] == '1B')
    # at-risk cases must carry ESPEN guideline anchoring + nutrition section
    if exp_rule == 'NUTRITION_RISK':
        ok = ok and 'ESPEN_LIVER_2019' in nut_rules[0]['guideline_ids']
        ok = ok and any(s['title'] in ('Nutrition Support', '营养支持')
                        for s in out['sections'])
    check(tid, 'nutrition', f'{exp_rule} expected for input {nut_in}', ok,
          f'fired={[r["rule_id"] for r in nut_rules]}')
# cholestatic type addendum must be present when at risk + cholestatic
out_chol = advisor.advise({'binary': 1, 'type': 1}, lab_values=full_labs,
                          nutrition=5)
chol_sec = next((s for s in out_chol['sections']
                 if s['title'] in ('Nutrition Support', '营养支持')), None)
ok_chol = bool(chol_sec) and 'MCT' in json.dumps(
    chol_sec['content'], ensure_ascii=False)
check('AGT_NUT_cholestatic_addendum', 'nutrition',
      'cholestatic MCT + fat-soluble vitamin addendum at risk', ok_chol,
      'type_note present' if ok_chol else 'missing addendum')

# ═══ PART E: blind-review feedback improvements (v3.2) ═══
print('\n' + '=' * 70)
print('  PART E: blind-review feedback improvements')
print('=' * 70)
# E1a: AI type without fractionation labs -> verification qualifier
out_e1a = advisor.advise({'screen': {'is_jaundice': True},
                          'type': {'type_classification': {'type_index': 1}}},
                         lab_values={'inr': 1.1, 'albumin': 40})
t_sec = next((s for s in out_e1a['sections'] if 'Jaundice Type' in s['title']), None)
ok_e1a = bool(t_sec) and 'verification' in (t_sec['content'] or {})
check('AGT_E1_type_verification_missing_labs', 'blind_review_fix',
      'AI-derived type carries lab-verification qualifier', ok_e1a,
      str((t_sec['content'] or {}).get('verification'))[:60])
# E1b: AI type concordant with labs -> no qualifier
out_e1b = advisor.advise({'screen': {'is_jaundice': True},
                          'type': {'type_classification': {'type_index': 1}}},
                         lab_values={'tbil': 120, 'dbil': 90, 'ibil': 30,
                                     'inr': 1.1, 'albumin': 40})
t_sec_b = next((s for s in out_e1b['sections'] if 'Jaundice Type' in s['title']), None)
ok_e1b = bool(t_sec_b) and 'verification' not in (t_sec_b['content'] or {})
check('AGT_E1_concordant_no_qualifier', 'blind_review_fix',
      'lab-confirmed type carries no qualifier', ok_e1b, '')
# E1c: AI type discordant with labs -> mismatch note
out_e1c = advisor.advise({'screen': {'is_jaundice': True},
                          'type': {'type_classification': {'type_index': 1}}},
                         lab_values={'tbil': 120, 'dbil': 20, 'ibil': 100,
                                     'inr': 1.1, 'albumin': 40})
t_sec_c = next((s for s in out_e1c['sections'] if 'Jaundice Type' in s['title']), None)
blob_c = json.dumps(t_sec_c['content'], ensure_ascii=False)
ok_e1c = 'verification' in (t_sec_c['content'] or {}) and 'disagree' in blob_c
check('AGT_E1_discordant_mismatch_note', 'blind_review_fix',
      'AI-lab type mismatch surfaces verification warning', ok_e1c, '')
# E2: red-flag pathway + mild labs -> triage no longer "no intervention"
out_e2 = advisor.advise({'binary': 1, 'type': 1},
                        lab_values={'tbil': 20, 'dbil': 10, 'ibil': 10,
                                    'alp': 380, 'ggt': 300, 'inr': 1.0,
                                    'albumin': 42, 'alt': 40, 'ast': 35},
                        diagnosis='胆总管结石')
tri_sec = next((s for s in out_e2['sections'] if 'Triage' in s['title']), None)
tri_txt = json.dumps(tri_sec['content'], ensure_ascii=False) if tri_sec else ''
ok_e2 = bool(tri_sec) and '24-48' in tri_txt and 'No specific' not in tri_txt
check('AGT_E2_triage_escalation_with_pathway', 'blind_review_fix',
      'red-flag + mild labs -> prompt-evaluation triage (not "none")',
      ok_e2, tri_txt[:80])
tri_rule = next(r for r in out_e2['agent']['triggered_rules']
                if r['rule_id'] == 'TRIAGE')
check('AGT_E2_triage_rule_urgency', 'blind_review_fix',
      'TRIAGE rule urgency escalated with pathway', tri_rule['urgency'] == 'urgent',
      f"urgency={tri_rule['urgency']}")
# E3: obstructive pathway actions begin with imaging confirmation
pw = ca.CLINICAL_PATHWAYS['obstructive_jaundice']
ok_e3 = pw['en_actions'][0].lower().startswith('confirm') and \
    '超声' in pw['cn_actions'][0]
check('AGT_E3_obstructive_imaging_first', 'blind_review_fix',
      'obstructive pathway: imaging confirmation is first action', ok_e3,
      pw['en_actions'][0][:60])
# E4: nutrition weight-individualised targets
out_e4 = advisor.advise({'binary': 1, 'type': 1}, lab_values=full_labs,
                        nutrition={'nrs2002': 4, 'weight': 62})
n_sec = next((s for s in out_e4['sections'] if s['title'] in ('Nutrition Support', '营养支持')), None)
n_blob = json.dumps(n_sec['content'], ensure_ascii=False) if n_sec else ''
ok_e4 = bool(n_sec) and 'calculated_targets' in (n_sec['content'] or {}) \
    and '62' in n_blob
check('AGT_E4_nutrition_weight_targets', 'blind_review_fix',
      'weight-individualised kcal/protein targets rendered', ok_e4,
      str((n_sec['content'] or {}).get('calculated_targets'))[:70])

# ═══ report ═══
rdf = pd.DataFrame(results)
out_path = os.path.join(TBL, 'CDSS_AGENT_TEST_RESULTS.csv')
rdf.to_csv(out_path, index=False)
total = len(rdf)
pn = int((rdf['result'] == 'PASS').sum())
fn = int((rdf['result'] == 'FAIL').sum())
print('\n' + '=' * 70)
print(f'  AGENT-LAYER TESTS: {total} total, {pn} PASS ({pn / total * 100:.1f}%), '
      f'{fn} FAIL')
print('=' * 70)
for c in sorted(rdf['category'].unique()):
    sub = rdf[rdf['category'] == c]
    print('  %-16s: %d/%d PASS' % (c, int((sub['result'] == 'PASS').sum()), len(sub)))
if fn > 0:
    print('  Failures:')
    for _, r in rdf[rdf['result'] == 'FAIL'].iterrows():
        print(f'    {r["test_id"]}: {r["assertion"]}')

summary_rows = [
    ('Agent payload schema', total, 'perceive/route/reason/critique/explain'),
    ('Rule-level pathway verification', len(PATHWAY_CASES), '6 pathways pos/neg'),
    ('Nutrition module verification', len(NUTRITION_CASES) + 1,
     'v3.1 NRS-2002 states + cholestatic addendum'),
    ('Blind-review feedback fixes', 7,
     'v3.2 type verification / triage escalation / imaging-first / weight targets'),
    ('GRADE validity', total, '1A/1B/2A/2B/2C'),
    ('Legacy keyword weakness documented', 1, 'RF_ALF_pos audit'),
]
pd.DataFrame(summary_rows, columns=['Domain', 'n', 'Detail']).to_csv(
    os.path.join(TBL, 'CDSS_AGENT_VALIDATION_SUMMARY.csv'), index=False)
print(f'\nSaved: {out_path}')
print('Saved: CDSS_AGENT_VALIDATION_SUMMARY.csv')
