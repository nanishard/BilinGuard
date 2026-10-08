# -*- coding: utf-8 -*-
"""
Build the definitive 105-row model inventory:
  7 tasks x 3 scopes (Face/Eyelid/Fusion) x 5 archs (ConvNeXt/ViT/Swin/EfficientNet/Ensemble)
Metrics per cell (point + 95% bootstrap CI, patient-level, seed 42, 1000 iters):
  n_val, auc, ap, acc, sens, spec, f1
Inputs: results/canonical_v2_probs.json (patient-level prob cache, latest checkpoints)
Outputs:
  tables/MODEL_INVENTORY_FULL.csv      (105 rows, full metric set)
  tables/MODEL_INVENTORY_complete.csv  (same 105-grid)
"""
import os, json, math, numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, average_precision_score)
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'face_v4_manuscript_delivery', 'tables')
SEED = 42; N_BOOT = 1000

probs = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))

TASK_ORDER = ['Binary Screening', 'Ternary Grading', 'DBIL', 'IBIL',
              'Jaundice Type', 'Child-Pugh', 'MELD']
SCOPE_ORDER = ['face', 'eyelid', 'fusion']
ARCH_ORDER = ['convnext', 'vit', 'swin', 'efficientnet', 'ensemble']
ARCH_NAME = {'convnext': 'ConvNeXt', 'vit': 'ViT', 'swin': 'Swin',
             'efficientnet': 'EfficientNet', 'ensemble': 'Ensemble'}
NC = {'Binary Screening': 2, 'Jaundice Type': 2}  # both binary, sens/spec via Youden

CKPT = {
    ('Binary Screening', 'face', 'convnext'): 'final_binary_face_convnext.pt',
    ('Binary Screening', 'face', 'vit'): 'final_binary_face_vit.pt',
    ('Binary Screening', 'face', 'swin'): 'final_binary_face_swin.pt',
    ('Binary Screening', 'face', 'efficientnet'): 'final_binary_face_efficientnet.pt',
    ('Binary Screening', 'eyelid', 'convnext'): 'eyelid_binary_convnext.pt',
    ('Binary Screening', 'eyelid', 'vit'): 'eyelid_binary_vit_tiny_patch16_224.pt',
    ('Binary Screening', 'eyelid', 'swin'): 'eyelid_binary_swin_tiny_patch4_window7_224.pt',
    ('Binary Screening', 'eyelid', 'efficientnet'): 'eyelid_binary_efficientnet_b0.pt',
    ('Ternary Grading', 'face', 'convnext'): 'v3_ternary_convnext_tiny.pt',
    ('Ternary Grading', 'face', 'vit'): 'v3_ternary_vit_tiny_patch16_224.pt',
    ('Ternary Grading', 'face', 'swin'): 'v3_ternary_swin_tiny_patch4_window7_224.pt',
    ('Ternary Grading', 'face', 'efficientnet'): 'v3_ternary_efficientnet_b0.pt',
    ('Ternary Grading', 'eyelid', 'convnext'): 'eyelid_opt_convnext_tiny.pt',
    ('Ternary Grading', 'eyelid', 'vit'): 'eyelid_opt_vit_tiny_patch16_224.pt',
    ('Ternary Grading', 'eyelid', 'swin'): 'eyelid_opt_swin_tiny_patch4_window7_224.pt',
    ('Ternary Grading', 'eyelid', 'efficientnet'): 'eyelid_clean_ternary_efficientnet_b0.pt',
    ('Jaundice Type', 'face', 'convnext'): 'typev2_face_convnext.pt',
    ('Jaundice Type', 'face', 'vit'): 'typev2_face_vit.pt',
    ('Jaundice Type', 'face', 'swin'): 'typev2_face_swin.pt',
    ('Jaundice Type', 'face', 'efficientnet'): 'typev2_face_efficientnet.pt',
    ('Jaundice Type', 'eyelid', 'convnext'): 'typev2_eyelid_convnext.pt',
    ('Jaundice Type', 'eyelid', 'vit'): 'typev2_eyelid_vit.pt',
    ('Jaundice Type', 'eyelid', 'swin'): 'typev2_eyelid_swin.pt',
    ('Jaundice Type', 'eyelid', 'efficientnet'): 'typev2_eyelid_efficientnet.pt',
}
for t, p in [('DBIL', 'dbil'), ('IBIL', 'ibil'), ('Child-Pugh', 'cp'), ('MELD', 'meld')]:
    for a in ['convnext', 'vit', 'swin', 'efficientnet']:
        CKPT[(t, 'face', a)] = f'{p}_face_{a}.pt'
        CKPT[(t, 'eyelid', a)] = f'{p}_eyelid_{a}.pt'


def pd_from_cache(task, scope, arch):
    d = probs.get(task, {}).get(scope, {}).get(arch)
    if not d: return None
    return {pid: (np.array(v[0], dtype=float), int(v[1])) for pid, v in d.items()}


def fusion_pd(task, arch):
    fa, ea = pd_from_cache(task, 'face', arch), pd_from_cache(task, 'eyelid', arch)
    if not fa or not ea: return None
    common = sorted(set(fa) & set(ea))
    out = {pid: ((fa[pid][0] + ea[pid][0]) / 2, fa[pid][1]) for pid in common}
    return out or None


def fusion_ens_pd(task):
    dicts = []
    for scope in ['face', 'eyelid']:
        for arch in ARCH_ORDER[:-1]:
            d = pd_from_cache(task, scope, arch)
            if d: dicts.append(d)
    if not dicts: return None
    common = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
    out = {pid: (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1]) for pid in common}
    return out or None


# ── overlay prob sources (2026-07-20 fixes) ──
TYPE2 = json.load(open(os.path.join(RES, 'typev2_probs.json'), encoding='utf-8')) \
    if os.path.exists(os.path.join(RES, 'typev2_probs.json')) else None
FUSION_SHARED = json.load(open(os.path.join(RES, 'fusion_shared_probs.json'), encoding='utf-8')) \
    if os.path.exists(os.path.join(RES, 'fusion_shared_probs.json')) else None


def _norm(d):
    return {pid: (np.array(v[0], dtype=float), int(v[1])) for pid, v in d.items()} if d else None


def get_cell_pd(task, scope, arch):
    # Type task: RAW-REBUILT dataset (data/type_v2), shared face/eyelid split (n=44 val)
    if task == 'Jaundice Type' and TYPE2:
        if scope == 'fusion':
            if arch == 'ensemble':
                dicts = [_norm(TYPE2[s][a]) for s in ['face', 'eyelid'] for a in ARCH_ORDER[:-1] if TYPE2.get(s, {}).get(a)]
                if not dicts: return None
                common = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
                return {pid: (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1]) for pid in common} or None
            fa, ea = _norm(TYPE2['face'].get(arch)), _norm(TYPE2['eyelid'].get(arch))
            if not fa or not ea: return None
            common = sorted(set(fa) & set(ea))
            return {pid: ((fa[pid][0] + ea[pid][0]) / 2, fa[pid][1]) for pid in common} or None
        if arch == 'ensemble':
            dicts = [_norm(TYPE2[scope][a]) for a in ARCH_ORDER[:-1] if TYPE2.get(scope, {}).get(a)]
            if not dicts: return None
            common = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
            return {pid: (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1]) for pid in common} or None
        return _norm(TYPE2.get(scope, {}).get(arch))
    # Binary/Ternary fusion: shared-split dedicated models
    if scope == 'fusion' and task in ('Binary Screening', 'Ternary Grading') and FUSION_SHARED and FUSION_SHARED.get(task):
        FS = FUSION_SHARED[task]
        if arch == 'ensemble':
            dicts = [_norm(FS[s][a]) for s in ['face', 'eyelid'] for a in ARCH_ORDER[:-1] if FS.get(s, {}).get(a)]
            if not dicts: return None
            common = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
            return {pid: (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1]) for pid in common} or None
        fa, ea = _norm(FS['face'].get(arch)), _norm(FS['eyelid'].get(arch))
        if not fa or not ea: return None
        common = sorted(set(fa) & set(ea))
        return {pid: ((fa[pid][0] + ea[pid][0]) / 2, fa[pid][1]) for pid in common} or None
    if scope == 'fusion':
        return fusion_ens_pd(task) if arch == 'ensemble' else fusion_pd(task, arch)
    return pd_from_cache(task, scope, '__ensemble__' if arch == 'ensemble' else arch)


def youden_thr(true, p1):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(true, p1)
    return thr[int(np.argmax(tpr - fpr))]


def metrics_point(true, P, nc):
    if nc == 2:
        thr = youden_thr(true, P[:, 1])
        pred = (P[:, 1] >= thr).astype(int)
    else:
        pred = P.argmax(1)
    cm = confusion_matrix(true, pred, labels=list(range(nc)))
    try:
        auc = roc_auc_score(true, P[:, 1]) if nc == 2 else \
            roc_auc_score(true, P, multi_class='ovr', labels=list(range(nc)))
    except Exception:
        auc = np.nan
    try:
        if nc == 2:
            ap = average_precision_score(true, P[:, 1])
        else:
            yb = label_binarize(true, classes=list(range(nc)))
            ap = float(np.mean([average_precision_score(yb[:, c], P[:, c]) for c in range(nc)]))
    except Exception:
        ap = np.nan
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro', labels=list(range(nc)), zero_division=0)
    sens_l, spec_l = [], []
    for c in range(nc):
        tp = cm[c, c]; fn = cm[c, :].sum() - tp; fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_l.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_l.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    if nc == 2:
        sens, spec = sens_l[1], spec_l[1]
    else:
        sens, spec = float(np.mean(sens_l)), float(np.mean(spec_l))
    return {'auc': auc, 'ap': ap, 'acc': acc, 'sens': sens, 'spec': spec, 'f1': f1}


def cell_metrics(pd_, nc):
    pids = sorted(pd_.keys())
    true = np.array([pd_[p][1] for p in pids])
    P = np.array([pd_[p][0] for p in pids])
    pt = metrics_point(true, P, nc)
    rng = np.random.RandomState(SEED)
    n = len(true)
    boots = {k: [] for k in pt}
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(true[idx])) < 2: continue
        m = metrics_point(true[idx], P[idx], nc)
        for k in boots:
            if not (isinstance(m[k], float) and math.isnan(m[k])):
                boots[k].append(m[k])
    out = {'n_val': n}
    for k, v in pt.items():
        arr = boots[k]
        lo, hi = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if arr else (np.nan, np.nan)
        out[k] = None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)
        out[f'{k}_lo'] = None if math.isnan(lo) else round(lo, 4)
        out[f'{k}_hi'] = None if math.isnan(hi) else round(hi, 4)
    return out


rows = []
for task in TASK_ORDER:
    nc = NC.get(task, 3)
    for scope in SCOPE_ORDER:
        for arch in ARCH_ORDER:
            pd_ = get_cell_pd(task, scope, arch)
            row = {'Task': task, 'Scope': scope.capitalize(), 'Architecture': ARCH_NAME[arch]}
            if pd_ is None:
                for k in ['n_val', 'auc', 'auc_lo', 'auc_hi', 'ap', 'ap_lo', 'ap_hi',
                          'acc', 'acc_lo', 'acc_hi', 'sens', 'sens_lo', 'sens_hi',
                          'spec', 'spec_lo', 'spec_hi', 'f1', 'f1_lo', 'f1_hi']:
                    row[k] = None
                row['ModelFile'] = CKPT.get((task, scope, arch), '-')
                row['Note'] = 'N/A: dual-modality val intersection too small (v2)' if scope == 'fusion' else 'model not trained'
            else:
                row.update(cell_metrics(pd_, nc))
                row['ModelFile'] = CKPT.get((task, scope, arch), '-')
                if task == 'Jaundice Type':
                    row['Note'] = 'typev2 raw-rebuild 2026-07-20 (shared split, Youden)'
                elif scope == 'fusion' and task in ('Binary Screening', 'Ternary Grading'):
                    row['Note'] = 'fusion shared-split retrain 2026-07-20'
                elif nc == 2:
                    row['Note'] = 'canonical-v2 2026-07-20 (binary: Youden threshold)'
                else:
                    row['Note'] = 'canonical-v2 2026-07-20'
            rows.append(row)

COLS = ['Task', 'Scope', 'Architecture', 'n_val',
        'auc', 'auc_lo', 'auc_hi', 'ap', 'ap_lo', 'ap_hi',
        'acc', 'acc_lo', 'acc_hi', 'sens', 'sens_lo', 'sens_hi',
        'spec', 'spec_lo', 'spec_hi', 'f1', 'f1_lo', 'f1_hi',
        'ModelFile', 'Note']
df = pd.DataFrame(rows)[COLS]

for name in ['MODEL_INVENTORY_FULL.csv', 'MODEL_INVENTORY_complete.csv']:
    tgt = os.path.join(TBL, name)
    try:
        df.to_csv(tgt, index=False)
        print(f'written: {name} ({len(df)} rows)')
    except PermissionError:
        alt = tgt.replace('.csv', '_v2.csv')
        df.to_csv(alt, index=False)
        print(f'LOCKED: {name} -> written to {os.path.basename(alt)} instead')
print('cells with AUC:', df['auc'].notna().sum(), '| N/A:', df['auc'].isna().sum())
