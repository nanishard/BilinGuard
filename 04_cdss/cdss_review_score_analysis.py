# -*- coding: utf-8 -*-
"""
Analyse the completed CDSS clinician review workbook (closed-loop step).

Input : results/face_v4_manuscript_delivery/tables/CDSS_CLINICIAN_REVIEW_SAMPLE.xlsx
        with >=2 reviewers filling the four 1-5 appropriateness scales:
        合并诊断一致性 / 建议适当性 / 安全覆盖 / 营养建议适当性

Computes
  1. Per-domain appropriateness: mean (SD), % >= 4 (appropriateness rate)
  2. Inter-rater reliability: ICC(2,k) + ICC(2,1) per domain; pairwise
     quadratic-weighted kappa (Cohen) between reviewers
  3. Stratified results by CDSS mode and severity
  4. Low-score case listing (any domain <= 2) for qualitative review

Run: python code/cdss_review_score_analysis.py
(if scores are not yet filled in, the script reports completion state)
"""
import sys, os, json
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
XL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables',
                  'CDSS_CLINICIAN_REVIEW_SAMPLE.xlsx')

DOMAINS = ['合并诊断一致性(1-5)', '建议适当性(1-5)',
           '安全覆盖(1-5)', '营养建议适当性(1-5)']


def icc_2k(scores_matrix):
    """ICC(2,k) and ICC(2,1) via two-way ANOVA, matrix = raters x subjects."""
    X = np.asarray(scores_matrix, dtype=float)
    n, k = X.shape  # subjects, raters
    if n < 2 or k < 2:
        return None, None
    grand = X.mean()
    ss_total = ((X - grand) ** 2).sum()
    ss_rows = k * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((X.mean(axis=0) - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / max((n - 1) * (k - 1), 1)
    denom_2k = ms_rows + (ms_cols - ms_err) / n
    denom_21 = ms_rows + ms_cols - ms_err + (ms_cols - ms_err) / n
    icc_k = (ms_rows - (ms_cols + (k - 1) * ms_err) / k) / (k * denom_2k) * k \
        if denom_2k > 0 else None
    icc_k = None if icc_k is None else icc_k
    # standard closed forms
    icc_k = (ms_rows - ms_err) / (ms_rows + (ms_cols - ms_err) / n) \
        if (ms_rows + (ms_cols - ms_err) / n) > 0 else None
    icc_1 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + (k / n) * (ms_cols - ms_err)) \
        if (ms_rows + (k - 1) * ms_err + (k / n) * (ms_cols - ms_err)) > 0 else None
    return icc_k, icc_1


def weighted_kappa(r1, r2, min_r=1, max_r=5):
    """Quadratic-weighted Cohen's kappa for ordinal 1-5 ratings."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    m = max_r - min_r + 1
    w = np.array([[(abs(i - j) / (m - 1)) ** 2 for j in range(m)] for i in range(m)])
    def hist(x):
        h = np.zeros(m)
        for v in x:
            idx = int(round(v)) - min_r
            if 0 <= idx < m:
                h[idx] += 1
        return h / max(len(x), 1)
    p_o_mat = np.outer(hist(r1), hist(r2))
    w_sum = (w * p_o_mat).sum()
    pe = (w * p_o_mat).sum()  # placeholder, compute properly below
    # observed agreement distribution
    n = len(r1)
    agree = np.zeros((m, m))
    for a, b in zip(r1, r2):
        i, j = int(round(a)) - min_r, int(round(b)) - min_r
        if 0 <= i < m and 0 <= j < m:
            agree[i, j] += 1
    p_obs = agree / max(n, 1)
    h1, h2 = hist(r1), hist(r2)
    p_exp = np.outer(h1, h2)
    kappa = 1 - (w * p_obs).sum() / max((w * p_exp).sum(), 1e-9)
    return kappa


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else XL
    if not os.path.exists(path):
        print(f'Workbook not found: {path}')
        return
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    print(f'Review workbook: {path}')
    print(f'  {len(df)} ratings from {df["审阅者"].dropna().nunique()} reviewer(s)')

    reviewers = sorted(df['审阅者'].dropna().astype(str).unique())
    scored = df[df[DOMAINS].notna().all(axis=1)]
    n_complete = len(scored)
    print(f'\nReviewers detected: {reviewers if reviewers else "NONE YET"}')
    print(f'Cases with all 4 domains scored: {n_complete}/{len(df)}')
    if n_complete < 2 or len(reviewers) < 2:
        print('\n>>> Not enough completed reviews to compute reliability.')
        print('    Fill 审阅者 + four 1-5 scores for >=2 reviewers, then re-run.')
        return

    print('\n' + '=' * 70)
    print('  1. Per-domain appropriateness (pooled across reviewers)')
    print('=' * 70)
    rows = []
    for d in DOMAINS:
        vals = df[d].dropna().astype(float)
        if len(vals):
            rows.append({
                'Domain': d, 'n': len(vals),
                'mean': vals.mean().round(2),
                'sd': vals.std().round(2),
                'rate_ge4': f'{100 * (vals >= 4).mean():.1f}%',
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n' + '=' * 70)
    print('  2. Inter-rater reliability')
    print('=' * 70)
    rel_rows = []
    for d in DOMAINS:
        wide = scored.pivot_table(index='pid', columns='审阅者', values=d)
        wide = wide.dropna()
        if wide.shape[1] >= 2 and len(wide) >= 2:
            icc_k, icc_1 = icc_2k(wide.values)
            raters = list(wide.columns)
            kappas = [weighted_kappa(wide[raters[i]], wide[raters[j]])
                      for i in range(len(raters)) for j in range(i + 1, len(raters))]
            pair_names = [f'{raters[i]}-{raters[j]}'
                          for i in range(len(raters)) for j in range(i + 1, len(raters))]
            rel_rows.append({
                'Domain': d, 'n_cases': len(wide), 'raters': len(raters),
                'ICC(2,k)': round(icc_k, 3) if icc_k is not None else None,
                'ICC(2,1)': round(icc_1, 3) if icc_1 is not None else None,
                'weighted_kappa_pairs': '; '.join(
                    f'{n}={round(k, 3)}' for n, k in zip(pair_names, kappas)),
            })
    print(pd.DataFrame(rel_rows).to_string(index=False))

    # ── multi-rater consensus & arbitration (>=3 reviewers) ──
    reviewers_all = sorted(df['审阅者'].dropna().unique())
    if len(reviewers_all) >= 3:
        print('\n' + '=' * 70)
        print(f'  2b. Three-rater consensus (median) & arbitration ({reviewers_all})')
        print('=' * 70)
        cons_rows = []
        consensus_records = []
        for d in DOMAINS:
            wide = scored.pivot_table(index='pid', columns='审阅者', values=d)
            wide = wide.dropna()
            med = wide.median(axis=1)
            cons_rows.append({
                'Domain': d, 'n': len(med),
                'consensus_mean': round(med.mean(), 2),
                'consensus_sd': round(med.std(), 2),
                'consensus_ge4': f'{100 * (med >= 4).mean():.1f}%',
            })
            for pid, v in med.items():
                consensus_records.append({'pid': pid, 'domain': d,
                                          'consensus': float(v)})
        print(pd.DataFrame(cons_rows).to_string(index=False))
        cons_df = pd.DataFrame(consensus_records)
        cons_wide = cons_df.pivot(index='pid', columns='domain', values='consensus')
        cons_wide.to_csv(os.path.join(BASE, 'results', 'face_v4_manuscript_delivery',
                                      'tables', 'CDSS_REVIEW_CONSENSUS.csv'),
                         encoding='utf-8-sig')
        print('  consensus scores saved -> CDSS_REVIEW_CONSENSUS.csv')

        # arbitration subset: |A-B| >= 2 — where does C fall?
        if '审阅者A' in reviewers_all and '审阅者B' in reviewers_all:
            arb_rows = []
            for pid, g in df.groupby('pid'):
                ga = g[g['审阅者'] == '审阅者A'].iloc[0]
                gb = g[g['审阅者'] == '审阅者B'].iloc[0]
                gc = g[g['审阅者'] != '审阅者A']
                gc = gc[gc['审阅者'] != '审阅者B']
                if not len(gc):
                    continue
                gc = gc.iloc[0]
                for d in DOMAINS:
                    if pd.notna(ga[d]) and pd.notna(gb[d]) and pd.notna(gc[d]):
                        if abs(ga[d] - gb[d]) >= 2:
                            lo, hi = sorted([ga[d], gb[d]])
                            pos = ('C=A侧' if gc[d] == lo else
                                   'C=B侧' if gc[d] == hi else
                                   'C居中' if lo < gc[d] < hi else
                                   'C超出区间' + ('下' if gc[d] < lo else '上'))
                            arb_rows.append({
                                'pid': pid, 'domain': d[:8],
                                'A': int(ga[d]), 'B': int(gb[d]), 'C': int(gc[d]),
                                'consensus': int(pd.Series([ga[d], gb[d], gc[d]]).median()),
                                '定位': pos,
                            })
            if arb_rows:
                ARB = pd.DataFrame(arb_rows)
                print(f'\n  仲裁子集（|A-B|>=2）: {len(ARB)} 病例×维度组合')
                print(ARB['定位'].value_counts().to_string())
                ARB.to_csv(os.path.join(BASE, 'results', 'face_v4_manuscript_delivery',
                                        'tables', 'CDSS_ARBITRATION_SUBSET.csv'),
                           index=False, encoding='utf-8-sig')
                print('  saved -> CDSS_ARBITRATION_SUBSET.csv')

    print('\n' + '=' * 70)
    print('  3. Stratified appropriateness (mean of 建议适当性 by mode/severity)')
    print('=' * 70)
    if '建议适当性(1-5)' in df.columns:
        strat = df.groupby(['CDSS_mode', 'CDSS_severity'])[
            '建议适当性(1-5)'].agg(['mean', 'count']).round(2)
        print(strat.to_string())

    print('\n' + '=' * 70)
    print('  4. Low-score cases (any domain <= 2) — qualitative review list')
    print('=' * 70)
    low = df[(df[DOMAINS] <= 2).any(axis=1)]
    if len(low):
        for _, r in low.iterrows():
            print(f"  pid={r['pid']} mode={r['CDSS_mode']} sev={r['CDSS_severity']} "
                  f"scores={[r[d] for d in DOMAINS]} comment={str(r.get('审评意见', ''))[:60]}")
    else:
        print('  none')

    out = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables',
                       'CDSS_REVIEW_SCORE_ANALYSIS.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
