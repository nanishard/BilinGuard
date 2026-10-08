# -*- coding: utf-8 -*-
"""
Individual Doctor Bootstrap CI Analysis
Loads all 题库-*.xlsx files, computes per-doctor metrics with 95% CI.
Metrics: F1, Sensitivity, Specificity, Accuracy, AUC, AP — for binary AND ternary.
Also: Wilcoxon signed-rank test (each doctor vs aggregate).
"""
import os, glob, json, numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, average_precision_score)
from sklearn.preprocessing import label_binarize
from scipy import stats

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'tables')
SEED = 42; N_BOOT = 1000
np.random.seed(SEED)

# ── Load ground truth from doctor files (Col3 = correct answer) ─
# 题库.xlsx is just a template; actual GT is in doctor file column 3
_ref_df = pd.read_excel(os.path.join(BASE, '题库-传染科1.xlsx'))
_ref_df = _ref_df.dropna(subset=[_ref_df.columns[0]])
_ref_df['case_id'] = _ref_df.iloc[:, 0].astype(int)
_ref_df['gt_answer'] = pd.to_numeric(_ref_df.iloc[:, 3], errors='coerce')  # Col3 = 答案 (correct answer)
_ref_df = _ref_df.dropna(subset=['gt_answer'])
_ref_df['gt_answer'] = _ref_df['gt_answer'].astype(int)
gt_map = dict(zip(_ref_df['case_id'], _ref_df['gt_answer']))

case_ids = np.array(sorted(gt_map.keys()))
gt = np.array([gt_map[c] for c in case_ids])
print(f'Ground truth: {len(gt)} cases, 1={sum(gt==1)}, 2={sum(gt==2)}, 3={sum(gt==3)}, 4={sum(gt==4)}')

# Convert to 0-indexed: 0=normal, 1=mild, 2=moderate, 3=severe
gt_4class = gt - 1  # 0,1,2,3
gt_binary = (gt > 1).astype(int)  # 0=normal, 1=jaundiced
gt_ternary = np.where(gt > 1, gt - 1, 0)  # 0=normal, 1=mild, 2=moderate, 3=severe → but for ternary: 0=normal, 1=mild, 2=mod+, skip severe separately
# Actually ternary: 0=mild, 1=moderate, 2=severe (for jaundice-only)
# For binary: 0=normal, 1=jaundiced
print(f'Binary: normal={sum(gt_binary==0)}, jaundiced={sum(gt_binary==1)}')

# ── Load all doctor files ────────────────────────────────────
doctor_files = {
    '传染科1': '题库-传染科1.xlsx',
    '公卫1': '题库-公卫1.xlsx',
    '公卫2': '题库-公卫2.xlsx',
    '眼科医生': '题库-眼科医生.xlsx',
    '口腔医生': '题库-口腔医生.xlsx',
    '肝外专培': '题库-肝外专培.xlsx',
}

doctors = {}
for name, fname in doctor_files.items():
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f'  SKIP {name}: file not found')
        continue
    df = pd.read_excel(fpath)
    df = df.dropna(subset=[df.columns[0]])  # Drop rows without case_id
    df['case_id'] = df.iloc[:, 0].astype(int)
    # Doctor's answer is Col2 (答题级别), GT is Col3 (答案)
    df['doc_answer'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    df = df.dropna(subset=['doc_answer'])
    df['doc_answer'] = df['doc_answer'].astype(int)
    times = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0).values
    # Match to ground truth by case_id
    case_map = dict(zip(df['case_id'].values, range(len(df))))
    matched_preds = []
    matched_times = []
    for cid in case_ids:
        if cid in case_map:
            idx = case_map[cid]
            matched_preds.append(df['doc_answer'].iloc[idx])
            matched_times.append(times[idx])
        else:
            matched_preds.append(-1)
            matched_times.append(np.nan)
    matched_preds = np.array(matched_preds)
    matched_times = np.array(matched_times)
    doctors[name] = {'preds_4class': matched_preds, 'times': matched_times}
    n_valid = sum(matched_preds > 0)
    valid_arr = matched_preds > 0
    acc = np.mean(matched_preds[valid_arr] == gt[valid_arr]) if n_valid > 0 else 0
    print(f'  {name}: {n_valid}/{len(gt)} valid, raw accuracy={acc:.3f}')

# ── Try to load additional doctors with different format ─────
for fname, name in [('题库-眼科医生2.xlsx', '眼科医生2'), ('题库-口腔医生2.xlsx', '口腔医生2'),
                     ('题库-xx科.xlsx', 'xx科'), ('题库-hao.xlsx', 'hao')]:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath): continue
    df = pd.read_excel(fpath)
    # These have different column structures, try to extract answer
    if df.shape[1] >= 3:
        # Try column 2 (answer_level or answer)
        for col_try in [2, 1, 3]:
            if col_try < df.shape[1]:
                vals = df.iloc[:, col_try].dropna().unique()
                if all(v in [1, 2, 3, 4] or v in [1.0, 2.0, 3.0, 4.0] for v in vals[:10]):
                    preds = df.iloc[:, col_try].values
                    case_ids_df = df.iloc[:, 0].values
                    case_map = dict(zip(case_ids_df, range(len(df))))
                    matched = []
                    for cid in case_ids:
                        matched.append(preds[case_map[cid]].astype(int) if cid in case_map else -1)
                    matched = np.array(matched)
                    if sum(matched > 0) > 50:
                        doctors[name] = {'preds_4class': matched, 'times': np.full(len(gt), np.nan)}
                        print(f'  {name}: {sum(matched>0)}/{len(gt)} valid (from col {col_try})')
                    break

print(f'\nTotal doctors loaded: {len(doctors)}')


# ── Metrics computation ──────────────────────────────────────
def compute_metrics_binary(true, pred_binary):
    """Binary metrics: 0=normal, 1=jaundiced."""
    cm = confusion_matrix(true, pred_binary, labels=[0, 1])
    tp, fn = cm[1, 1], cm[1, 0]
    fp, tn = cm[0, 1], cm[0, 0]
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(true, pred_binary)
    f1 = f1_score(true, pred_binary, zero_division=0)
    # For AUC with hard predictions, use accuracy as proxy if no probabilities
    try:
        auc = roc_auc_score(true, pred_binary)
    except:
        auc = 0.5
    ap = average_precision_score(true, pred_binary) if len(np.unique(true)) > 1 else 0
    return {'auc': auc, 'acc': acc, 'f1': f1, 'sens': sens, 'spec': spec, 'ap': ap}


def compute_metrics_ternary(true_3, pred_3):
    """Ternary metrics: 0=mild, 1=moderate, 2=severe (jaundice only)."""
    nc = 3
    cm = confusion_matrix(true_3, pred_3, labels=list(range(nc)))
    acc = accuracy_score(true_3, pred_3)
    f1 = f1_score(true_3, pred_3, average='macro', zero_division=0)
    sens_list, spec_list = [], []
    for c in range(nc):
        tp = cm[c, c]; fn = cm[c, :].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    sens = float(np.mean(sens_list))
    spec = float(np.mean(spec_list))
    # AUC for ternary with hard predictions
    probs = np.zeros((len(true_3), nc))
    for i in range(len(pred_3)):
        probs[i, pred_3[i]] = 1.0
    try:
        auc = roc_auc_score(true_3, probs, multi_class='ovr', labels=list(range(nc)))
    except:
        auc = 0.5
    try:
        yb = label_binarize(true_3, classes=list(range(nc)))
        ap = np.mean([average_precision_score(yb[:, c], probs[:, c]) for c in range(nc)])
    except:
        ap = 0
    return {'auc': auc, 'acc': acc, 'f1': f1, 'sens': sens, 'spec': spec, 'ap': ap}


def bootstrap_ci_binary(true, pred, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    n = len(true)
    res = {k: [] for k in ['auc', 'acc', 'f1', 'sens', 'spec', 'ap']}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = true[idx], pred[idx]
        if len(np.unique(yt)) < 2: continue
        m = compute_metrics_binary(yt, yp)
        for k in res:
            if not np.isnan(m[k]): res[k].append(m[k])
    def ci(arr): arr=[x for x in arr if not np.isnan(x)]; return (float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))) if arr else (0,0)
    return {f'{k}_lo': ci(res[k])[0] for k in res} | {f'{k}_hi': ci(res[k])[1] for k in res}


def bootstrap_ci_ternary(true, pred, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    n = len(true)
    res = {k: [] for k in ['auc', 'acc', 'f1', 'sens', 'spec', 'ap']}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = true[idx], pred[idx]
        if len(np.unique(yt)) < 2: continue
        m = compute_metrics_ternary(yt, yp)
        for k in res:
            if not np.isnan(m[k]): res[k].append(m[k])
    def ci(arr): arr=[x for x in arr if not np.isnan(x)]; return (float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))) if arr else (0,0)
    return {f'{k}_lo': ci(res[k])[0] for k in res} | {f'{k}_hi': ci(res[k])[1] for k in res}


def fmt(v, lo, hi): return f'{v:.3f} [{lo:.3f}-{hi:.3f}]'

# ── Compute per-doctor metrics ───────────────────────────────
print('\n' + '='*90)
print('  INDIVIDUAL DOCTOR RESULTS WITH BOOTSTRAP 95% CI')
print('='*90)

all_doctor_results = []

# Binary evaluation
print('\n--- BINARY (Normal vs Jaundiced) ---')
print(f'{"Doctor":<15} {"F1 [CI]":>24} {"Sens [CI]":>24} {"Spec [CI]":>24} {"Acc [CI]":>24} {"AUC [CI]":>24}')
print('-' * 140)

binary_correct_per_doctor = {}
for name, data in doctors.items():
    preds = data['preds_4class']
    valid = preds > 0
    if sum(valid) < 50: continue
    # Binary: answer 1 = normal(0), 2-4 = jaundiced(1)
    pred_bin = np.where(preds > 1, 1, 0)
    m = compute_metrics_binary(gt_binary[valid], pred_bin[valid])
    ci = bootstrap_ci_binary(gt_binary[valid], pred_bin[valid])
    binary_correct_per_doctor[name] = (pred_bin[valid] == gt_binary[valid]).astype(int)
    all_doctor_results.append({'doctor': name, 'task': 'Binary', **m, **ci, 'n': int(sum(valid))})
    print(f'{name:<15} {fmt(m["f1"],ci["f1_lo"],ci["f1_hi"]):>24} {fmt(m["sens"],ci["sens_lo"],ci["sens_hi"]):>24} {fmt(m["spec"],ci["spec_lo"],ci["spec_hi"]):>24} {fmt(m["acc"],ci["acc_lo"],ci["acc_hi"]):>24} {fmt(m["auc"],ci["auc_lo"],ci["auc_hi"]):>24}')

# Ternary evaluation (jaundice only: mild=0, moderate=1, severe=2)
print('\n--- TERNARY (Mild/Moderate/Severe, jaundice cases only) ---')
jaundice_mask = gt_binary == 1
gt_tern_jaun = gt_4class[jaundice_mask] - 1  # 0=mild, 1=moderate, 2=severe
print(f'  Jaundice cases: {sum(jaundice_mask)} (mild={sum(gt_tern_jaun==0)}, mod={sum(gt_tern_jaun==1)}, sev={sum(gt_tern_jaun==2)})')

print(f'{"Doctor":<15} {"F1 [CI]":>24} {"Sens [CI]":>24} {"Spec [CI]":>24} {"Acc [CI]":>24} {"AUC [CI]":>24}')
print('-' * 140)

ternary_correct_per_doctor = {}
for name, data in doctors.items():
    preds = data['preds_4class']
    valid_all = preds > 0
    # Only jaundice cases
    valid = valid_all & jaundice_mask
    if sum(valid) < 10: continue
    pred_tern = preds[valid] - 2  # 2→0(mild), 3→1(mod), 4→2(sev)
    pred_tern = np.clip(pred_tern, 0, 2)
    gt_t = gt_tern_jaun[valid[jaundice_mask]]
    m = compute_metrics_ternary(gt_t, pred_tern)
    ci = bootstrap_ci_ternary(gt_t, pred_tern)
    ternary_correct_per_doctor[name] = (pred_tern == gt_t).astype(int)
    all_doctor_results.append({'doctor': name, 'task': 'Ternary', **m, **ci, 'n': int(sum(valid))})
    print(f'{name:<15} {fmt(m["f1"],ci["f1_lo"],ci["f1_hi"]):>24} {fmt(m["sens"],ci["sens_lo"],ci["sens_hi"]):>24} {fmt(m["spec"],ci["spec_lo"],ci["spec_hi"]):>24} {fmt(m["acc"],ci["acc_lo"],ci["acc_hi"]):>24} {fmt(m["auc"],ci["auc_lo"],ci["auc_hi"]):>24}')

# ── Aggregate clinician metrics ──────────────────────────────
print('\n--- AGGREGATE CLINICIANS ---')
# Binary aggregate (majority vote)
all_bin_preds = []
for name, data in doctors.items():
    preds = data['preds_4class']
    pred_bin = np.where(preds > 1, 1, 0)
    all_bin_preds.append(pred_bin)
all_bin_preds = np.array(all_bin_preds)
# Majority vote
majority_bin = (all_bin_preds.mean(axis=0) > 0.5).astype(int)
valid_mask = np.all(all_bin_preds >= 0, axis=0)

m = compute_metrics_binary(gt_binary[valid_mask], majority_bin[valid_mask])
ci = bootstrap_ci_binary(gt_binary[valid_mask], majority_bin[valid_mask])
all_doctor_results.append({'doctor': 'Aggregate', 'task': 'Binary', **m, **ci, 'n': int(sum(valid_mask))})
print(f'{"Aggregate":<15} {fmt(m["f1"],ci["f1_lo"],ci["f1_hi"]):>24} {fmt(m["sens"],ci["sens_lo"],ci["sens_hi"]):>24} {fmt(m["spec"],ci["spec_lo"],ci["spec_hi"]):>24} {fmt(m["acc"],ci["acc_lo"],ci["acc_hi"]):>24} {fmt(m["auc"],ci["auc_lo"],ci["auc_hi"]):>24}')

# ── Wilcoxon tests ───────────────────────────────────────────
print('\n--- WILCOXON SIGNED-RANK TESTS (per-doctor correct vs aggregate correct) ---')
agg_correct = (majority_bin[valid_mask] == gt_binary[valid_mask]).astype(int)
for name in doctors:
    if name not in binary_correct_per_doctor: continue
    doc_correct = binary_correct_per_doctor[name]
    # Align lengths
    n = min(len(doc_correct), len(agg_correct))
    dc = doc_correct[:n]; ac = agg_correct[:n]
    if len(np.unique(dc)) > 1 and len(np.unique(ac)) > 1:
        stat, p = stats.wilcoxon(dc, ac)
    else:
        stat, p = 0, 1.0
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f'  {name:<15}: W={stat:.1f}, p={p:.4f} {sig}')

# ── Time efficiency ──────────────────────────────────────────
print('\n--- RESPONSE TIME (seconds) ---')
time_groups = []
time_names = []
for name, data in doctors.items():
    times = data['times']
    valid_times = times[~np.isnan(times)]
    if len(valid_times) > 0:
        print(f'  {name:<15}: median={np.median(valid_times):.1f}s, mean={np.mean(valid_times):.1f}s')
        time_groups.append(valid_times)
        time_names.append(name)
if len(time_groups) >= 2:
    H, p = stats.kruskal(*time_groups)
    print(f'\n  Kruskal-Wallis: H={H:.2f}, p={p:.4f}')

# ── Save ─────────────────────────────────────────────────────
rdf = pd.DataFrame(all_doctor_results)

# Manuscript TSV
with open(os.path.join(TBL, 'doctor_bootstrap_results.tsv'), 'w', encoding='utf-8') as f:
    f.write('Task\tDoctor\tF1 score\tSensitivity\tSpecificity\tAccuracy\tAUC ROC\tAverage Precision\tN\n')
    for _, r in rdf.iterrows():
        f.write('\t'.join([
            r['task'], r['doctor'],
            fmt(r['f1'], r['f1_lo'], r['f1_hi']),
            fmt(r['sens'], r['sens_lo'], r['sens_hi']),
            fmt(r['spec'], r['spec_lo'], r['spec_hi']),
            fmt(r['acc'], r['acc_lo'], r['acc_hi']),
            fmt(r['auc'], r['auc_lo'], r['auc_hi']),
            fmt(r['ap'], r['ap_lo'], r['ap_hi']),
            str(int(r['n'])),
        ]) + '\n')

rdf.to_csv(os.path.join(TBL, 'doctor_bootstrap_results.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(RES, 'doctor_bootstrap_results.json'), 'w') as f:
    json.dump(all_doctor_results, f, indent=2, default=str)

print(f'\n  TSV: {TBL}/doctor_bootstrap_results.tsv')
print(f'  CSV: {TBL}/doctor_bootstrap_results.csv')
