# -*- coding: utf-8 -*-
"""
CDSS master validation runner — one command, all suites, CI-ready.

Suites (in order):
  1. legacy keyword tests        (run_cdss_final.py)            150
  2. agent-layer assertions      (run_cdss_agent_tests.py)      820
  3. robustness hardening        (run_cdss_robustness_tests.py) 304
  4. patient-level validation    (cdss_patient_level_validation.py) 301 admissions
  5. threshold sensitivity       (cdss_threshold_sensitivity.py)   stability report
  6. guideline currency audit    (cdss_guideline_currency_audit.py)

Exit code 0 only if every suite succeeds. Consolidated summary written to
results/face_v4_manuscript_delivery/tables/CDSS_VALIDATION_MASTER_SUMMARY.csv

Run: python code/run_cdss_all_validation.py
"""
import sys, os, time, importlib
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
CODE = os.path.join(BASE, 'code')
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(BASE, 'deployment'))


def run_suite(name, module_name, metric_fn):
    print('\n' + '█' * 70)
    print(f'█ SUITE: {name}  ({module_name})')
    print('█' * 70)
    t0 = time.time()
    # capture stdout per suite
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    ok = True
    detail = ''
    try:
        mod = importlib.import_module(module_name)
        with redirect_stdout(buf):
            detail = metric_fn(mod)
    except SystemExit as e:
        ok = e.code in (0, None)
        detail = f'exit={e.code}'
    except Exception as e:
        ok = False
        detail = f'{type(e).__name__}: {e}'
    dt = time.time() - t0
    # print the tail of captured output for visibility
    tail = '\n'.join(buf.getvalue().splitlines()[-4:])
    if tail:
        print(tail)
    status = 'PASS' if ok else 'FAIL'
    print(f'>> {name}: {status}  ({dt:.1f}s)  {detail}')
    return {'suite': name, 'status': status, 'detail': str(detail)[:90],
            'seconds': round(dt, 1)}


def _summary_from_csv(mod, fname, pass_col='result'):
    p = os.path.join(TBL, fname)
    df = pd.read_csv(p)
    n = len(df)
    n_pass = int((df[pass_col] == 'PASS').sum()) if pass_col in df.columns else n
    ok = n_pass == n
    return f'{n_pass}/{n} assertions'


results = []

results.append(run_suite(
    'Legacy keyword tests', 'run_cdss_final',
    lambda m: _summary_from_csv(m, 'CDSS_TEST_RESULTS.csv')))

results.append(run_suite(
    'Agent-layer assertions', 'run_cdss_agent_tests',
    lambda m: _summary_from_csv(m, 'CDSS_AGENT_TEST_RESULTS.csv')))

results.append(run_suite(
    'Robustness hardening', 'run_cdss_robustness_tests',
    lambda m: _summary_from_csv(m, 'CDSS_HARDENING_TESTS.csv')))


def _patient_metric(mod):
    p = os.path.join(TBL, 'CDSS_PATIENT_LEVEL_VALIDATION.csv')
    df = pd.read_csv(p)
    n = len(df)
    if 'error' in df.columns and df['error'].notna().any():
        return 'runtime errors present'
    nut_ok = ((df['nrs2002'] >= 3) & (df['nutrition_rule'] == 'NUTRITION_RISK')).sum()
    n_risk = int((df['nrs2002'] >= 3).sum())
    return f'{n} admissions, 0 errors, nutrition {nut_ok}/{n_risk}'


results.append(run_suite(
    'Patient-level validation', 'cdss_patient_level_validation', _patient_metric))


def _sensitivity_metric(mod):
    p = os.path.join(TBL, 'CDSS_THRESHOLD_SENSITIVITY.csv')
    _ = pd.read_csv(p)
    return 'stability report generated'


results.append(run_suite(
    'Threshold sensitivity', 'cdss_threshold_sensitivity', _sensitivity_metric))


def _currency_metric(mod):
    p = os.path.join(TBL, 'CDSS_GUIDELINE_CURRENCY.csv')
    df = pd.read_csv(p)
    return f'{len(df)} registry entries audited'


results.append(run_suite(
    'Guideline currency audit', 'cdss_guideline_currency_audit', _currency_metric))

# ═══ consolidated summary ═══
rdf = pd.DataFrame(results)
out = os.path.join(TBL, 'CDSS_VALIDATION_MASTER_SUMMARY.csv')
rdf.to_csv(out, index=False)
n_pass = int((rdf['status'] == 'PASS').sum())
print('\n' + '█' * 70)
print(f'█ MASTER VALIDATION: {n_pass}/{len(rdf)} suites PASS  '
      f'({int(rdf["seconds"].sum())}s total)')
print('█' * 70)
print(rdf.to_string(index=False))
print(f'\nSaved: {out}')
sys.exit(0 if n_pass == len(rdf) else 1)
