# -*- coding: utf-8 -*-
"""
A1: Restricted CP/MELD analysis — jaundiced-only patients with fully computed scores.
  (a) Evaluate existing models on jaundiced patients only (exclude controls assigned grade A/low)
  (b) TBIL-only baseline: logistic regression on TBIL predicting CP BvC / MELD category
Output: results/restricted_cp_meld_analysis.json
"""
import os, re, json, math, random
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))

SEED = 42; random.seed(SEED); np.random.seed(SEED)

# ── 1. Identify val patients with real (non-default) CP/MELD ──
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
df_ex = pd.read_excel(EXCEL)
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)

def strip_zeros(s): return s.lstrip('0') if s else s
def get_clinical(pid):
    nums = re.findall(r'\d+', pid.replace('.zip', ''))
    hid = nums[-1] if nums else None
    if not hid: return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    if len(rows) == 0: return None
    r = rows.iloc[0]
    cp_grade = str(r.iloc[15]).strip().upper() if pd.notna(r.iloc[15]) else None
    tbil = pd.to_numeric(r.iloc[51], errors='coerce')
    inr = pd.to_numeric(r.iloc[61], errors='coerce')
    alb = pd.to_numeric(r.iloc[67], errors='coerce')
    meld = None
    if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
        meld = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr, 1)) + 9.57 * math.log(1.0) + 6.43
        meld = round(max(meld, 6))
    return {'cp_grade': cp_grade, 'meld': meld, 'tbil': tbil}

# from canonical val cache, get all val pids with their CP/MELD labels
cp_val = list(V2['Child-Pugh']['face']['convnext'].keys())
meld_val = list(V2['MELD']['face']['convnext'].keys())

# map pid -> default or real?
clinical = {}
for pid in cp_val:
    c = get_clinical(pid)
    if c:
        clinical[pid] = c
    else:
        clinical[pid] = {'cp_grade': 'A', 'meld': 8, 'tbil': 15, 'is_default': True}  # controls

jaundiced_cp = {pid: c for pid, c in clinical.items() if not c.get('is_default') and c['cp_grade'] in ('A', 'B', 'C')}
jaundiced_meld = {pid: c for pid, c in clinical.items() if not c.get('is_default') and c['meld'] is not None}

print(f'CP val total: {len(cp_val)}, jaundiced with real score: {len(jaundiced_cp)}')
print(f'MELD val total: {len(meld_val)}, jaundiced with real score: {len(jaundiced_meld)}')

# check class distribution
from collections import Counter
cp_real_dist = Counter(c['cp_grade'] for c in jaundiced_cp.values())
print('CP (jaundiced):', dict(cp_real_dist))
meld_real = [c['meld'] for c in jaundiced_meld.values()]
print('MELD (jaundiced): median', np.median(meld_real))

# ── 2. Evaluate existing models on jaundiced-only subset ──
def eval_on_subset(task, scope, arch, pids):
    d = V2[task][scope][arch]
    sub = {p: d[p] for p in pids if p in d}
    if not sub: return None
    t = np.array([sub[p][1] for p in sub])
    P = np.array([sub[p][0] for p in sub])
    if len(np.unique(t)) < 2: return None
    nc = len(np.unique(t))
    try:
        auc = roc_auc_score(t, P[:, 1]) if nc == 2 else roc_auc_score(t, P, multi_class='ovr', labels=sorted(set(t)))
    except: auc = float('nan')
    pred = P.argmax(1)
    f1 = f1_score(t, pred, average='macro', labels=sorted(set(t)), zero_division=0)
    acc = accuracy_score(t, pred)
    return {'auc': round(auc, 4), 'f1': round(f1, 4), 'acc': round(acc, 4), 'n': len(t), 'dist': dict(Counter(t))}

# CP: subset with real A/B/C → B vs C and A vs (B,C)
cp_pids = sorted(jaundiced_cp.keys())
cp_labels = {pid: jaundiced_cp[pid]['cp_grade'] for pid in cp_pids}
# original models expect labels 0(A),1(B),2(C). Filter to jaundiced only
results_cp = {}
for scope in ['face', 'eyelid']:
    for arch in ['convnext', 'vit', 'swin', 'efficientnet']:
        if arch not in V2['Child-Pugh'][scope]: continue
        d = V2['Child-Pugh'][scope][arch]
        sub = {p: d[p] for p in cp_pids if p in d}
        if not sub: continue
        t = np.array([sub[p][1] for p in sub])
        P = np.array([sub[p][0] for p in sub])
        if len(np.unique(t)) < 2: continue
        # B vs C binary
        mask = (t >= 1)
        if mask.sum() > 0 and (1 - mask.sum()) > 0:
            y = t[mask] - 1  # B=0, C=1
            Pb = P[mask][:, 1].copy()
            Pb_2 = P[mask][:, 2].copy()
            Pbc = np.column_stack([Pb, Pb_2])
            Pbc_norm = Pbc / Pbc.sum(1, keepdims=True)
            auc_bc = roc_auc_score(y, Pbc_norm[:, 1]) if len(np.unique(y)) > 1 else float('nan')
        else:
            auc_bc = float('nan')
        # original multi-class AUC on subset
        labels_s = sorted(set(int(x) for x in t))
        try:
            auc_macro = roc_auc_score(t, P, multi_class='ovr', labels=labels_s)
        except: auc_macro = float('nan')
        results_cp[f'{scope}_{arch}'] = {'n': int(len(t)), 'dist': {int(k): int(v) for k, v in Counter(t).items()},
                                          'auc_ovr_original': round(auc_macro, 4),
                                          'auc_BvC': round(auc_bc, 4)}

# MELD: subset with real MELD
meld_pids = sorted(jaundiced_meld.keys())
# categorize: Low(<=20), Mid(21-30), High(>30)
meld_labels = {}
for pid in meld_pids:
    m = jaundiced_meld[pid]['meld']
    meld_labels[pid] = 0 if m <= 20 else (1 if m <= 30 else 2)
print('MELD category dist:', dict(Counter(meld_labels.values())))

results_meld = {}
for scope in ['face', 'eyelid']:
    for arch in ['convnext', 'vit', 'swin', 'efficientnet']:
        if arch not in V2['MELD'][scope]: continue
        d = V2['MELD'][scope][arch]
        sub = {p: d[p] for p in meld_pids if p in d}
        if not sub: continue
        t = np.array([sub[p][1] for p in sub])
        P = np.array([sub[p][0] for p in sub])
        if len(np.unique(t)) < 2: continue
        # Low vs (Mid+High)
        y_bin = (t >= 1).astype(int)
        if len(np.unique(y_bin)) > 1:
            auc_low_vs_higher = roc_auc_score(y_bin, P[:, 1:].sum(1))
        else:
            auc_low_vs_higher = float('nan')
        labels_s = sorted(set(int(x) for x in t))
        try:
            auc_macro = roc_auc_score(t, P, multi_class='ovr', labels=labels_s)
        except: auc_macro = float('nan')
        results_meld[f'{scope}_{arch}'] = {'n': int(len(t)), 'dist': {int(k): int(v) for k, v in Counter(t).items()},
                                            'auc_ovr_original': round(auc_macro, 4),
                                            'auc_LowVsHigher': round(auc_low_vs_higher, 4)}

# ── 3. TBIL-only baseline ──
def tbil_baseline(pids, labels_dict, task_name):
    X = []; y = []
    for pid in pids:
        tbil = jaundiced_cp.get(pid, jaundiced_meld.get(pid, {})).get('tbil', None)
        if tbil is None or pd.isna(tbil): continue
        if pid in labels_dict:
            X.append([math.log(max(tbil, 1))])
            y.append(labels_dict[pid])
    if len(np.unique(y)) < 2: return None
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler(); Xt = scaler.fit_transform(X)
    lr = LogisticRegression(random_state=SEED, max_iter=1000).fit(Xt, y)
    yp = lr.predict_proba(Xt)
    nc = len(np.unique(y))
    try:
        auc = roc_auc_score(y, yp[:, 1]) if nc == 2 else roc_auc_score(y, yp, multi_class='ovr', labels=sorted(set(y)))
    except: auc = float('nan')
    return {'n': len(y), 'auc': round(auc, 4), 'f1': round(f1_score(y, lr.predict(Xt), average='macro', zero_division=0), 4)}

cp_binary_labels = {pid: 0 if g == 'B' else 1 for pid, g in cp_labels.items() if g in ('B', 'C')}
meld_binary_labels = {pid: (0 if meld_labels[pid] == 0 else 1) for pid in meld_pids}
tbil_cp = tbil_baseline(cp_pids, cp_binary_labels, 'CP BvC')
tbil_meld = tbil_baseline(meld_pids, meld_binary_labels, 'MELD low vs higher')

out = {
    'cp_clinical_dist': {str(k): v for k, v in cp_real_dist.items()},
    'cp_jaundiced_val_n': len(cp_pids),
    'cp_models_jaundiced': results_cp,
    'meld_clinical_dist': {str(k): v for k, v in Counter(meld_labels.values()).items()},
    'meld_jaundiced_val_n': len(meld_pids),
    'meld_models_jaundiced': results_meld,
    'tbil_baseline': {'CP_BvC': tbil_cp, 'MELD_LowVsHigher': tbil_meld},
}
print('\n=== RESTRICTED CP ANALYSIS (jaundiced only) ===')
print(f'  N={len(cp_pids)}, dist={dict(cp_real_dist)}')
for k, v in results_cp.items():
    print(f'  {k:<20} OVR={v["auc_ovr_original"]:.3f}  BvC={v["auc_BvC"]:.3f}  dist={v["dist"]}')
print(f'  TBIL baseline CP BvC: auc={tbil_cp["auc"]:.3f} (n={tbil_cp["n"]})' if tbil_cp else '  TBIL baseline CP BvC: N/A')
print()
print('=== RESTRICTED MELD ANALYSIS (jaundiced only) ===')
print(f'  N={len(meld_pids)}, dist={dict(Counter(meld_labels.values()))}')
for k, v in results_meld.items():
    print(f'  {k:<20} OVR={v["auc_ovr_original"]:.3f}  LowVsHigh={v["auc_LowVsHigher"]:.3f}  dist={v["dist"]}')
print(f'  TBIL baseline MELD low vs higher: auc={tbil_meld["auc"]:.3f} (n={tbil_meld["n"]})' if tbil_meld else '  TBIL baseline MELD: N/A')

with open(os.path.join(RES, 'restricted_cp_meld_analysis.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print('\n=== RESTRICTED CP ANALYSIS (jaundiced only) ===')
print(f'  N={len(cp_pids)}, dist={dict(cp_real_dist)}')
for k, v in results_cp.items():
    print(f'  {k:<20} OVR={v["auc_ovr_original"]:.3f}  BvC={v["auc_BvC"]:.3f}  dist={v["dist"]}')
print(f'  TBIL baseline CP BvC: auc={tbil_cp["auc"]:.3f} (n={tbil_cp["n"]})')
print()
print('=== RESTRICTED MELD ANALYSIS (jaundiced only) ===')
print(f'  N={len(meld_pids)}, dist={dict(Counter(meld_labels.values()))}')
for k, v in results_meld.items():
    print(f'  {k:<20} OVR={v["auc_ovr_original"]:.3f}  LowVsHigh={v["auc_LowVsHigher"]:.3f}  dist={v["dist"]}')
print(f'  TBIL baseline MELD low vs higher: auc={tbil_meld["auc"]:.3f} (n={tbil_meld["n"]})')
