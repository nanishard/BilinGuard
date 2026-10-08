# -*- coding: utf-8 -*-
"""
P5-corrected: TBIL baseline for CP/MELD restricted analysis.
Properly: fit logistic on TRAIN set only, evaluate on VAL set once.
No validation-set fitting.
"""
import os, re, json, math, random
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
SEED = 42; random.seed(SEED); np.random.seed(SEED)

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
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hid[-6:])]
    if len(rows) == 0: return None
    r = rows.iloc[0]
    cp_grade = str(r.iloc[15]).strip().upper() if pd.notna(r.iloc[15]) else None
    tbil = pd.to_numeric(r.iloc[51], errors='coerce')
    inr = pd.to_numeric(r.iloc[61], errors='coerce')
    meld = None
    if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
        meld = 3.78 * math.log(max(tbil,1)) + 11.2 * math.log(max(inr,1)) + 9.57 + 6.43
        meld = round(max(meld, 6))
    return {'cp_grade': cp_grade, 'meld': meld, 'tbil': tbil}

# Get all val pids for CP/MELD
cp_val_pids = list(V2['Child-Pugh']['face']['convnext'].keys())
meld_val_pids = list(V2['MELD']['face']['convnext'].keys())

# Also need TRAIN pids - rebuild from sam_pats split
sam_pats = {}
for cat in ['normal','mild','moderate','severe']:
    cd = os.path.join(BASE, 'data', 'face_v4', cat)
    if not os.path.isdir(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = [f for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')]
        if faces: sam_pats[pid] = {'cat': cat}

# CP labels
cp_labels = {}
for pid, p in sam_pats.items():
    if p['cat'] == 'normal':
        cp_labels[pid] = 0  # A
    else:
        c = get_clinical(pid)
        if c and c['cp_grade'] in ('A','B','C'):
            cp_labels[pid] = {'A':0,'B':1,'C':2}[c['cp_grade']]
# Split (same as training)
by_grade = {}
for pid, g in cp_labels.items(): by_grade.setdefault(g, []).append(pid)
tr_pids, va_pids = set(), set()
for g, pids in by_grade.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_pids.update(pids[:n]); tr_pids.update(pids[n:])

# MELD labels
meld_labels = {}
for pid, p in sam_pats.items():
    c = get_clinical(pid)
    if c and c['meld'] is not None:
        m = c['meld']
        meld_labels[pid] = 0 if m <= 20 else (1 if m <= 30 else 2)
    elif p['cat'] == 'normal':
        meld_labels[pid] = 0
# Split
by_g = {}
for pid, g in meld_labels.items(): by_g.setdefault(g, []).append(pid)
tr_m, va_m = set(), set()
for g, pids in by_g.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_m.update(pids[:n]); tr_m.update(pids[n:])

# Restrict to jaundiced only
def get_tbil(pid):
    c = get_clinical(pid)
    return c['tbil'] if c and pd.notna(c.get('tbil')) else None

# CP restricted: jaundiced train/val with TBIL
cp_jaundiced_tr = {pid: cp_labels[pid] for pid in tr_pids if pid in cp_labels and cp_labels[pid] >= 0 and get_clinical(pid) and get_clinical(pid)['cp_grade'] in ('A','B','C')}
cp_jaundiced_va = {pid: cp_labels[pid] for pid in va_pids if pid in cp_labels and cp_labels[pid] >= 0 and get_clinical(pid) and get_clinical(pid)['cp_grade'] in ('A','B','C')}

# MELD restricted
meld_jaundiced_tr = {pid: meld_labels[pid] for pid in tr_m if pid in meld_labels and get_clinical(pid) and get_clinical(pid)['meld'] is not None}
meld_jaundiced_va = {pid: meld_labels[pid] for pid in va_m if pid in meld_labels and get_clinical(pid) and get_clinical(pid)['meld'] is not None}

print(f'CP restricted: train={len(cp_jaundiced_tr)}, val={len(cp_jaundiced_va)}')
print(f'  Val dist: {dict(pd.Series(list(cp_jaundiced_va.values())).value_counts().sort_index())}')
print(f'MELD restricted: train={len(meld_jaundiced_tr)}, val={len(meld_jaundiced_va)}')
print(f'  Val dist: {dict(pd.Series(list(meld_jaundiced_va.values())).value_counts().sort_index())}')

def tbil_baseline(train_labels, val_labels, task_name):
    """Fit TBIL logistic on TRAIN, evaluate on VAL."""
    X_tr, y_tr = [], []
    for pid, lab in train_labels.items():
        tbil = get_tbil(pid)
        if tbil is None or pd.isna(tbil): continue
        X_tr.append([math.log(max(tbil, 1))])
        y_tr.append(lab)
    if len(set(y_tr)) < 2:
        print(f'  {task_name}: only 1 class in train, skip'); return None
    X_tr = np.array(X_tr); y_tr = np.array(y_tr)
    scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(random_state=SEED, max_iter=1000, C=1e9).fit(X_tr_s, y_tr)

    X_va, y_va = [], []
    for pid, lab in val_labels.items():
        tbil = get_tbil(pid)
        if tbil is None or pd.isna(tbil): continue
        X_va.append([math.log(max(tbil, 1))])
        y_va.append(lab)
    if len(set(y_va)) < 2:
        print(f'  {task_name}: only 1 class in val, skip'); return None
    X_va = np.array(X_va); y_va = np.array(y_va)
    X_va_s = scaler.transform(X_va)
    yp = lr.predict_proba(X_va_s)
    nc = len(set(y_va))
    try:
        auc = roc_auc_score(y_va, yp[:, 1]) if nc == 2 else roc_auc_score(y_va, yp, multi_class='ovr', labels=sorted(set(y_va)))
    except: auc = float('nan')
    pred = lr.predict(X_va_s)
    f1 = f1_score(y_va, pred, average='macro', zero_division=0)
    acc = accuracy_score(y_va, pred)
    return {'auc': round(auc, 4), 'f1': round(f1, 4), 'acc': round(acc, 4),
            'n_train': len(y_tr), 'n_val': len(y_va), 'train_dist': dict(pd.Series(y_tr).value_counts().sort_index()),
            'val_dist': dict(pd.Series(y_va).value_counts().sort_index())}

print('\n=== CORRECTED TBIL BASELINE (fit on train, eval on val) ===')
cp_baseline = tbil_baseline(cp_jaundiced_tr, cp_jaundiced_va, 'CP')
print(f'  CP BvC (TBIL): {cp_baseline}')
meld_baseline = tbil_baseline(meld_jaundiced_tr, meld_jaundiced_va, 'MELD')
print(f'  MELD (TBIL): {meld_baseline}')

# Also compute TBIL+age+sex baseline
def tbil_age_sex(train_labels, val_labels, task_name):
    """TBIL + age + sex baseline."""
    X_tr, y_tr = [], []
    for pid, lab in train_labels.items():
        c = get_clinical(pid)
        tbil = c['tbil'] if c else None
        if tbil is None: continue
        row = df_ex[df_ex['hid_str'].apply(strip_zeros) == strip_zeros(re.findall(r'\d+', pid)[-1])]
        if len(row) == 0:
            row = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(re.findall(r'\d+', pid)[-1][-6:])]
        age = pd.to_numeric(row.iloc[0].iloc[5], errors='coerce') if len(row) else 50
        sex = pd.to_numeric(row.iloc[0].iloc[4], errors='coerce') if len(row) else 1
        X_tr.append([math.log(max(tbil,1)), float(age) if pd.notna(age) else 50, float(sex) if pd.notna(sex) else 1])
        y_tr.append(lab)
    if len(set(y_tr)) < 2: return None
    X_tr = np.array(X_tr); y_tr = np.array(y_tr)
    scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(random_state=SEED, max_iter=1000, C=1e9).fit(X_tr_s, y_tr)
    X_va, y_va = [], []
    for pid, lab in val_labels.items():
        c = get_clinical(pid)
        tbil = c['tbil'] if c else None
        if tbil is None: continue
        row = df_ex[df_ex['hid_str'].apply(strip_zeros) == strip_zeros(re.findall(r'\d+', pid)[-1])]
        if len(row) == 0:
            row = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(re.findall(r'\d+', pid)[-1][-6:])]
        age = pd.to_numeric(row.iloc[0].iloc[5], errors='coerce') if len(row) else 50
        sex = pd.to_numeric(row.iloc[0].iloc[4], errors='coerce') if len(row) else 1
        X_va.append([math.log(max(tbil,1)), float(age) if pd.notna(age) else 50, float(sex) if pd.notna(sex) else 1])
        y_va.append(lab)
    if len(set(y_va)) < 2: return None
    X_va_s = scaler.transform(np.array(X_va))
    yp = lr.predict_proba(X_va_s)
    nc = len(set(y_va))
    try: auc = roc_auc_score(y_va, yp[:, 1]) if nc == 2 else roc_auc_score(y_va, yp, multi_class='ovr', labels=sorted(set(y_va)))
    except: auc = float('nan')
    return {'auc': round(auc, 4), 'n_val': len(y_va)}

print('\n=== TBIL + AGE + SEX BASELINE (fit on train, eval on val) ===')
cp_3var = tbil_age_sex(cp_jaundiced_tr, cp_jaundiced_va, 'CP')
print(f'  CP: {cp_3var}')
meld_3var = tbil_age_sex(meld_jaundiced_tr, meld_jaundiced_va, 'MELD')
print(f'  MELD: {meld_3var}')

# Save corrected results
result = {
    'cp_restricted': {'n_train': len(cp_jaundiced_tr), 'n_val': len(cp_jaundiced_va),
                      'val_dist': dict(pd.Series(list(cp_jaundiced_va.values())).value_counts().sort_index()),
                      'face_swin_auc': 0.827,
                      'tbil_only_auc': cp_baseline['auc'] if cp_baseline else None,
                      'tbil_age_sex_auc': cp_3var['auc'] if cp_3var else None},
    'meld_restricted': {'n_train': len(meld_jaundiced_tr), 'n_val': len(meld_jaundiced_va),
                        'val_dist': dict(pd.Series(list(meld_jaundiced_va.values())).value_counts().sort_index()),
                        'face_swin_auc': 0.840,
                        'tbil_only_auc': meld_baseline['auc'] if meld_baseline else None,
                        'tbil_age_sex_auc': meld_3var['auc'] if meld_3var else None},
}
with open(os.path.join(RES, 'restricted_cp_meld_corrected.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, default=str)
print('\nSaved: restricted_cp_meld_corrected.json')
