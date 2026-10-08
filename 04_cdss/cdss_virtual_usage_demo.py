# -*- coding: utf-8 -*-
"""
CDSS virtual usage session (dry-run of the deployed system).

Simulates 8 realistic consultations across departments and decision modes,
each exercising the full closed loop:
input -> advice sections -> agent self-audit -> PHI-safe audit record
-> post-session telemetry (JAMIA-style monitoring view).

The audit log for this session is written to a dedicated file (not the
production log). Run: python code/cdss_virtual_usage_demo.py
"""
import sys, os, json
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
AUDIT = os.path.join(BASE, 'results', 'cdss_virtual_usage_log.jsonl')
if os.path.exists(AUDIT):
    os.remove(AUDIT)

W = 78


def hr(c='─'):
    print(c * W)


def show_case(idx, title, dept, results, labs, dx, nrs):
    print()
    hr('═')
    print(f'  虚拟会诊 {idx}  |  {title}')
    hr('═')
    print(f'  科室: {dept}    入院诊断: {dx or "—"}')
    lab_str = '  '.join(f'{k}={v}' for k, v in (labs or {}).items())
    print(f'  化验: {lab_str or "（未提供）"}')
    print(f'  NRS-2002: {nrs if nrs is not None else "（未筛查）"}')
    nutrition = {'nrs2002': float(nrs)} if nrs is not None else None
    advisor = ca.ClinicalAdvisor(lang='cn', department=dept, audit_log=AUDIT)
    out = advisor.advise(results, lab_values=labs, diagnosis=dx, nutrition=nutrition)
    ag = out['agent']

    print()
    print(f'  ▸ 分诊结论: {out["summary"]}')
    print(f'  ▸ 智能体: 模式={ag["mode"]}  优先级={ag["self_check"]["alert_priority"]}'
          f'  置信度={ag["self_check"]["confidence"]}'
          f'  规则={ag["self_check"]["alerts_total"]} 条')

    key_titles = ['临床警示', '严重程度评估', '营养支持', '分诊与转诊',
                  'Child-Pugh / MELD 风险分层']
    for sec in out['sections']:
        if sec['title'].split(':')[0] in key_titles or '临床警示' in sec['title']:
            print(f'\n  ◆ {sec["title"]} {sec.get("icon", "")}')
            c = sec['content']
            if isinstance(c, dict):
                for k, v in list(c.items())[:6]:
                    if isinstance(v, list):
                        for it in v[:4]:
                            print(f'     · {it}')
                    else:
                        print(f'     {k}: {v}')
            elif isinstance(c, list):
                for it in c[:5]:
                    print(f'     · {it}')
            else:
                print(f'     {c}')

    print(f'\n  ▸ 触发规则 (GRADE):')
    for r in ag['triggered_rules']:
        print(f'     [{r["grade"]}] {r["rule_id"]} ({r["urgency"]})')

    flags = [c for c in ag['self_check']['checks'] if c['status'] != 'PASS']
    if flags:
        print(f'\n  ▸ 自审警示:')
        for c in flags:
            print(f'     [{c["status"]}] {c["check"]}: {c["detail"][:70]}')
    return out


# ═══ The 8 virtual consultations ═══
cases = [
    (1, 'ICU · 慢加急性肝衰竭 + 乙肝（红色警示 + 营养风险）', 'icu',
     {'screen': {'is_jaundice': True},
      'grade': {'grading': {'grade_index': 2, 'prediction': 'Severe'}},
      'type': {'type_classification': {'type_index': 0, 'prediction': 'Hepatocellular'}}},
     {'tbil': 280, 'dbil': 140, 'ibil': 140, 'alt': 420, 'ast': 380, 'alp': 140,
      'ggt': 80, 'inr': 2.4, 'albumin': 30},
     '慢加急性肝衰竭，慢性乙型病毒性肝炎', 4),

    (2, '肝胆外科 · 胰头癌恶性梗阻（胆汁郁积 + 营养风险）', 'hpb_surgery',
     {'screen': {'is_jaundice': True},
      'type': {'type_classification': {'type_index': 1, 'prediction': 'Cholestatic'}}},
     {'tbil': 320, 'dbil': 240, 'ibil': 80, 'alt': 120, 'ast': 90, 'alp': 520,
      'ggt': 410, 'inr': 1.3, 'albumin': 35},
     '胰头癌', 3),

    (3, '血液科 · AIHA 溶血性黄疸', 'hematology',
     {'screen': {'is_jaundice': True},
      'type': {'type_classification': {'type_index': 0, 'prediction': 'Hepatocellular'}}},
     {'tbil': 95, 'dbil': 12, 'ibil': 83, 'alt': 25, 'ast': 20, 'alp': 90,
      'ggt': 30, 'inr': 1.0, 'albumin': 44},
     '自身免疫性溶血性贫血', 1),

    (4, '感染科 · 药物性肝损伤（Hy 定律）', 'infectious_hepatology',
     {'screen': {'is_jaundice': True},
      'type': {'type_classification': {'type_index': 0, 'prediction': 'Hepatocellular'}}},
     {'tbil': 110, 'dbil': 60, 'ibil': 50, 'alt': 480, 'ast': 360, 'alp': 140,
      'ggt': 90, 'inr': 1.1, 'albumin': 39},
     '肝功能异常原因待查:药物性？其他？', None),

    (5, '感染科门诊 · 慢乙肝轻度（常规低优先级）', 'infectious_hepatology',
     {'screen': {'is_jaundice': True},
      'grade': {'grading': {'grade_index': 0, 'prediction': 'Mild'}},
      'type': {'type_classification': {'type_index': 0, 'prediction': 'Hepatocellular'}}},
     {'tbil': 45, 'dbil': 18, 'ibil': 27, 'alt': 120, 'ast': 80, 'alp': 110,
      'ggt': 60, 'inr': 1.0, 'albumin': 42},
     '慢性乙型病毒性肝炎', 1),

    (6, '眼科转诊 · 筛查阴性（无需处理）', 'ophthalmology',
     {'screen': {'is_jaundice': False}}, None, None, None),

    (7, '急诊 · AI-化验冲突（AI 判重度，化验正常）', 'emergency',
     {'screen': {'is_jaundice': True},
      'grade': {'grading': {'grade_index': 2, 'prediction': 'Severe'}}},
     {'tbil': 15, 'dbil': 3, 'ibil': 12, 'alt': 25, 'ast': 20, 'alp': 90,
      'ggt': 30, 'inr': 1.0, 'albumin': 44},
     '黄疸待查', None),

    (8, '消化内科 · 黄疸筛查阳性但关键化验缺失', 'gi',
     {'screen': {'is_jaundice': True}},
     {'tbil': 120, 'dbil': 70},
     '黄疸原因待查', None),
]

print('╔' + '═' * W)
print('║  BilinGuard CDSS v3.1 — 虚拟使用演练（8 例会诊）')
print('║  每例完整走: 输入 → 建议分节 → 智能体自审 → PHI 安全审计日志')
print('╚' + '═' * W)

outputs = []
for idx, title, dept, results, labs, dx, nrs in cases:
    outputs.append(show_case(idx, title, dept, results, labs, dx, nrs))

# ═══ Audit records written this session ═══
print()
hr('═')
print('  会话审计日志（PHI 安全，无诊断原文/化验值/患者标识）')
hr('═')
with open(AUDIT, encoding='utf-8') as f:
    recs = [json.loads(x) for x in f if x.strip()]
for r in recs:
    print(f'  {r["timestamp"]}  mode={r["mode"]:<15s} prio={r["alert_priority"]:<7s}'
          f' sev={r["severity"]}  rules={r["n_rules"]}  nut={r.get("nutrition_state")}')

# ═══ Post-session telemetry ═══
print()
hr('═')
print('  会话后遥测（部署监测视图）')
hr('═')
sys.path.insert(0, os.path.join(BASE, 'code'))
from cdss_audit_log_analytics import load_records, analyse
recs = load_records(AUDIT)
analyse(recs)

# persist a compact session report
rows = []
for (idx, title, dept, _, _, dx, nrs), out in zip(cases, outputs):
    ag = out['agent']
    rows.append({
        'case': idx, 'scenario': title, 'department': dept,
        'mode': ag['mode'], 'priority': ag['self_check']['alert_priority'],
        'confidence': ag['self_check']['confidence'],
        'severity': out['severity'], 'nutrition_state': ag['case'].get('nutrition_state'),
        'rules_fired': ';'.join(r['rule_id'] for r in ag['triggered_rules']),
        'summary': out['summary'],
    })
import pandas as pd
out_csv = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables',
                       'CDSS_VIRTUAL_USAGE_SESSION.csv')
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f'\nSaved session report: {out_csv}')
print(f'Saved audit log     : {AUDIT}')
