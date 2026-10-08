# -*- coding: utf-8 -*-
"""
CDSS guideline currency audit (v3).

1. Classify all registry guidelines: current (<=5y) / classical instrument /
   review-recommended (>5y, non-classical).
2. Check whether a newer counterpart already exists in the registry.
3. Live PubMed verification for the top stale clinical-practice guidelines.
4. Output CDSS_GUIDELINE_CURRENCY.csv + console report.
"""
import sys, os, time
sys.path.insert(0, r'D:\research\人脸识别营养\传染科\deployment')
if 'clinical_advisor' in sys.modules:
    del sys.modules['clinical_advisor']
import clinical_advisor as ca
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables')

NOW = 2026
CLASSICAL = set(ca._CLASSICAL_GIDS)
NEWER_IN_REGISTRY = {
    'EASL_CHB_2017': 'AASLD_CHB_2024 (AASLD 2024 HBV guidance already cited)',
    'CIRRHOSIS_2019': 'EASL_DECOMP_2018 (EASL decompensated cirrhosis, 2024 rev, already cited)',
    'EASL_PBC_2017': 'PBC_2021 (CSH 2021 consensus, already cited)',
    'APASL_ACLF_2019': 'none in registry',
    'LF_2018': 'none in registry (COSSH-ACLF still standard)',
    'ALD_2018': 'none in registry',
    'ACG_ALD_2018': 'none in registry',
    'TOKYO_TG18': 'none in registry (TG18 remains current standard)',
    'AIH_2021': 'none in registry',
    'PBC_2021': 'none in registry',
    'MELD_3_0_2021': 'MELD_3_0 (OPTN 2023, already cited)',
}

rows = []
for gid, e in ca.GUIDELINES.items():
    try:
        year = int(e[4])
    except (TypeError, ValueError, IndexError):
        year = None
    if gid in CLASSICAL:
        status = 'classical-instrument'
    elif year is None:
        status = 'unknown-year'
    elif year >= NOW - 5:
        status = 'current'
    else:
        status = 'review-recommended'
    rows.append({
        'guideline_id': gid, 'title': str(e[0])[:80],
        'organisation': str(e[3]), 'year': year,
        'age_years': (NOW - year) if year else None,
        'status': status,
        'newer_counterpart_in_registry': NEWER_IN_REGISTRY.get(gid, '—'),
    })
G = pd.DataFrame(rows)
print('=' * 78)
print(f'  GUIDELINE CURRENCY AUDIT — {len(G)} entries, reference year {NOW}')
print('=' * 78)
print(G['status'].value_counts().to_string())
print('\n  Review-recommended (>5y, non-classical):')
stale = G[G['status'] == 'review-recommended'].sort_values('year')
for _, r in stale.iterrows():
    newer = r['newer_counterpart_in_registry']
    print(f"    {r['guideline_id']:22s} ({int(r['year'])}) {str(r['title'])[:52]:52s} | {newer}")

# ── live PubMed verification for top stale CPGs ──
print('\n' + '=' * 78)
print('  PubMed live check for successor guidelines (retrieved', NOW, ')')
print('=' * 78)
try:
    from Bio import Entrez
    Entrez.email = 'medclaw@freedomai.com'
    QUERIES = [
        ('EASL_CHB_2017', '"EASL"[TIAB] AND "hepatitis B"[TIAB] AND ("2024"[DP] OR "2025"[DP])'),
        ('CIRRHOSIS_2019', 'cirrhosis[TI] AND (guideline[TI] OR guidance[TI]) AND ("2023"[DP] OR "2024"[DP])'),
        ('TOKYO_TG18', '("Tokyo Guidelines"[TI] OR "acute cholangitis"[TI]) AND ("2023"[DP] OR "2024"[DP] OR "2025"[DP])'),
        ('LF_2018', '("liver failure"[TI]) AND (guideline[TI] OR consensus[TI]) AND ("2023"[DP] OR "2024"[DP])'),
        ('ALD_2018', '("alcoholic liver disease"[TI] OR "alcohol-related liver disease"[TI]) AND (guideline[TI] OR guidance[TI]) AND ("2022"[DP] OR "2023"[DP])'),
        ('AIH_2021', '("autoimmune hepatitis"[TI]) AND (guideline[TI] OR guidance[TI] OR consensus[TI]) AND ("2023"[DP] OR "2024"[DP])'),
    ]
    pubmed_notes = {}
    for gid, q in QUERIES:
        try:
            h = Entrez.esearch(db='pubmed', term=q, retmax=3, sort='date')
            rec = Entrez.read(h); h.close()
            ids = rec['IdList']
            note = f"{len(rec['Count'])} hits"
            if ids:
                h2 = Entrez.efetch(db='pubmed', id=','.join(ids), rettype='xml')
                recs = Entrez.read(h2); h2.close()
                titles = [str(a['MedlineCitation']['Article'].get('ArticleTitle', ''))[:90]
                          for a in recs['PubmedArticle']]
                note += '; latest: ' + titles[0]
            pubmed_notes[gid] = note
            print(f'  {gid:22s} -> {note[:100]}')
            time.sleep(0.4)
        except Exception as ex:
            pubmed_notes[gid] = f'search error: {ex}'
            print(f'  {gid:22s} -> search error: {str(ex)[:60]}')
    G['pubmed_update_check'] = G['guideline_id'].map(
        lambda g: pubmed_notes.get(g, ''))
except ImportError:
    print('  (Bio.Entrez unavailable — skipping live check)')

out = os.path.join(TBL, 'CDSS_GUIDELINE_CURRENCY.csv')
G.to_csv(out, index=False)
print(f'\nSaved: {out}')

# also expose stale count actually cited by the agent layer on the real cohort
pv = os.path.join(TBL, 'CDSS_PATIENT_LEVEL_VALIDATION.csv')
if os.path.exists(pv):
    print('\nStale-guideline exposure on the real cohort is surfaced per-case in '
          'agent.self_check.guidelines_stale_cited (v3).')
