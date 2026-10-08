# -*- coding: utf-8 -*-
"""
AUDIT: Jaundice Type task — verify the near-perfect AUCs are legitimate.
Checks:
  A1: train/val patient disjoint (type2 face split)
  A2: name-stem collision across train/val (same person, different pid strings)
  A3: label mapping correctness (spot-check Excel rows for val patients)
  A4: score distribution per class (is separation real or artifact?)
  A5: permutation test — shuffle val labels, AUC must collapse to ~0.5
  A6: eyelid legacy val consistency (its own split, no overlap with its training)
  A7: fusion cell recomputation independently (n, labels, merged probs)
  A8: checkpoint sanity (num_classes=2 head)
Output: results/type_audit.json
"""
import os, re, json, random, math, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import torch, timm

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
SEED = 42
random.seed(SEED); np.random.seed(SEED)

report = {}

# ── rebuild the exact type2 split (same code as train_type_binary_v2.py) ──
df_ex = pd.read_excel(EXCEL)
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)
JT_COL = [c for df_c in [df_ex.columns] for c in df_c if '黄疸类型' in str(c)][0]

def strip_zeros(s): return s.lstrip('0') if s else s
def type_label(pid):
    nums = re.findall(r'\d+', pid.replace('.zip', ''))
    hid = nums[-1] if nums else None
    if not hid: return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    if len(rows) == 0: return None
    jt = pd.to_numeric(rows.iloc[0][JT_COL], errors='coerce')
    if pd.isna(jt): return None
    if jt == 2: return 0
    if jt == 3: return 1
    return None

pats = {}
for cat in ['mild', 'moderate', 'severe']:
    cd = os.path.join(DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if os.path.isdir(p):
            pats[pid] = cat

labels = {pid: type_label(pid) for pid in pats}
labels = {pid: l for pid, l in labels.items() if l is not None}
by_grade = {}
for pid, g in labels.items(): by_grade.setdefault(g, []).append(pid)
tr_p, va_p = set(), set()
for g, pids in by_grade.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_p.update(pids[:n]); tr_p.update(pids[n:])

# A1: disjoint
inter = tr_p & va_p
report['A1_train_val_disjoint'] = {'pass': len(inter) == 0, 'overlap': sorted(inter)}

# A2: name-stem collision (strip digits → Chinese name part)
def stem(pid):
    s = re.sub(r'\d+', '', pid).replace('.zip', '')
    return s[-3:] if len(s) >= 3 else s
tr_stems = {}
for p in tr_p: tr_stems.setdefault(stem(p), []).append(p)
collisions = [(p, tr_stems[stem(p)]) for p in va_p if stem(p) in tr_stems]
report['A2_name_stem_collision'] = {'pass': len(collisions) == 0,
                                    'n_collisions': len(collisions),
                                    'examples': [(v, t) for v, t in collisions[:10]]}

# A3: label spot-check — reload Excel rows for all val patients, verify consistency
a3 = []
for pid in sorted(va_p):
    nums = re.findall(r'\d+', pid.replace('.zip', ''))
    hid = nums[-1] if nums else None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    how = 'exact' if len(rows) else 'fallback'
    if not len(rows):
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    jt = pd.to_numeric(rows.iloc[0][JT_COL], errors='coerce') if len(rows) else None
    tbil_col = list(df_ex.columns)[list(df_ex.columns).index(JT_COL) + 1]
    tbil = pd.to_numeric(rows.iloc[0][tbil_col], errors='coerce') if len(rows) else None
    a3.append({'pid': pid, 'label': labels[pid], 'excel_jt': None if pd.isna(jt) else int(jt),
               'match': how, 'tbil': None if pd.isna(tbil) else float(tbil)})
mismatch = [r for r in a3 if (r['excel_jt'] == 2) != (r['label'] == 0)]
report['A3_label_consistency'] = {'pass': len(mismatch) == 0, 'n_val': len(a3),
                                  'mismatches': mismatch,
                                  'fallback_matches': sum(1 for r in a3 if r['match'] == 'fallback'),
                                  'tbil_by_label': {str(l): float(np.nanmean([r['tbil'] for r in a3 if r['label'] == l])) for l in (0, 1)}}

# A4: score distribution (type2 face probs)
t2 = json.load(open(os.path.join(RES, 'type2_face_probs.json'), encoding='utf-8'))
a4 = {}
for arch, d in t2['face'].items():
    P1 = {pid: v[0][1] for pid, v in d.items()}
    pos = sorted(v for pid, v in P1.items() if d[pid][1] == 1)
    neg = sorted(v for pid, v in P1.items() if d[pid][1] == 0)
    a4[arch] = {'n_pos': len(pos), 'n_neg': len(neg),
                'pos_min': float(min(pos)), 'pos_max': float(max(pos)),
                'neg_max': float(max(neg)), 'neg_mean': float(np.mean(neg)),
                'separation_margin': float(min(pos) - max(neg))}
report['A4_score_distribution'] = a4

# A5: permutation test (ensemble probs, 200 shuffles)
dicts = [{pid: (np.array(v[0]), v[1]) for pid, v in t2['face'][a].items()} for a in t2['face']]
common = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
ens = {pid: (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1]) for pid in common}
true = np.array([ens[p][1] for p in common])
P1 = np.array([ens[p][0][1] for p in common])
real_auc = roc_auc_score(true, P1)
rng = np.random.RandomState(0)
perm = [roc_auc_score(rng.permutation(true), P1) for _ in range(200)]
report['A5_permutation_test'] = {'real_auc': float(real_auc),
                                 'perm_mean': float(np.mean(perm)), 'perm_max': float(np.max(perm)),
                                 'pass': real_auc > float(np.max(perm))}

# A6: eyelid legacy val — check it was a proper held-out split (rebuild from train_eyelid_meld_type logic)
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
mdf = pd.read_csv(MANIFEST)
eyelid_patients = {}
for _, r in mdf.iterrows():
    if r['n_eyelid'] > 0:
        imgs = json.loads(r['eyelid_images'])
        valid = [p for p in imgs if os.path.exists(p)]
        if valid: eyelid_patients[r['patient_id']] = valid
type_items = []
for pid, e_paths in eyelid_patients.items():
    nums = re.findall(r'\d{6,}', pid)
    hid = nums[-1] if nums else None
    if not hid: continue
    rows = df_ex[df_ex['hid_str'].astype(str).str.strip().str.lstrip('0') == hid.lstrip('0')]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].astype(str).str.strip().str.lstrip('0').str.endswith(hid[-6:])]
    if len(rows) == 0: continue
    jt = pd.to_numeric(rows.iloc[0][JT_COL], errors='coerce')
    if pd.notna(jt) and jt in (2, 3):
        type_items.append((e_paths[0], 0 if jt == 2 else 1, pid))
random.seed(SEED)
by_l = {}
for it in type_items: by_l.setdefault(it[1], []).append(it)
va_e, tr_e = [], []
for label, its in by_l.items():
    random.shuffle(its)
    n = max(1, int(len(its) * 0.2))
    va_e.extend(its[:n]); tr_e.extend(its[n:])
va_e_pids = set(p for _, _, p in va_e); tr_e_pids = set(p for _, _, p in tr_e)
report['A6_eyelid_val'] = {'n_val': len(va_e_pids), 'n_train': len(tr_e_pids),
                           'val_label_dist': {str(l): sum(1 for _, ll, _ in va_e if ll == l) for l in (0, 1)},
                           'train_val_disjoint': len(va_e_pids & tr_e_pids) == 0}

# A7: independent fusion recomputation
cache = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
eye_ens = cache['Jaundice Type']['eyelid']['__ensemble__']
fa_ens = {pid: ens[pid] for pid in common}
both = sorted(set(fa_ens) & set(eye_ens))
fus_true = np.array([fa_ens[p][1] for p in both])
fus_p = np.array([(fa_ens[p][0] + np.array(eye_ens[p][0])) / 2 for p in both])
fus_auc = roc_auc_score(fus_true, fus_p[:, 1]) if len(np.unique(fus_true)) > 1 else None
label_agree = all(fa_ens[p][1] == eye_ens[p][1] for p in both)
report['A7_fusion_check'] = {'n': len(both), 'label_agreement': label_agree,
                             'label_dist': {str(l): int((fus_true == l).sum()) for l in (0, 1)},
                             'recomputed_auc': None if fus_auc is None else float(fus_auc)}

# A8: checkpoint sanity
for f, bb in [('type2_face_convnext.pt', 'convnext_tiny'), ('type2_face_efficientnet.pt', 'efficientnet_b0')]:
    sd = torch.load(os.path.join(MODEL, f), map_location='cpu')
    head_keys = [k for k in sd if 'head' in k or 'classifier' in k or 'fc' in k]
    out_dim = int(sd[head_keys[-1]].shape[0]) if head_keys else None
    report.setdefault('A8_checkpoints', {})[f] = {'head_out_dim': out_dim, 'keys': len(sd)}

with open(os.path.join(RES, 'type_audit.json'), 'w', encoding='utf-8') as fp:
    json.dump(report, fp, indent=2, ensure_ascii=False, default=str)
print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
