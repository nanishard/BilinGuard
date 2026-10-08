# -*- coding: utf-8 -*-
"""
Verify/normalize external-validation CIs for BOTH binary and ternary, using the
project's standard bootstrap protocol (RandomState(42), 1000 iters, 2.5/97.5
percentiles) so the two rows in Table2_layer1_trimmed.csv are methodologically
consistent.

Sources:
  binary  : results/external_validation_frozen.json  (patient_probs, n=117)
  ternary : external_validation_results/.../patient_level_results.xlsx (n=57,
            TBIL-derived Mild/Moderate/Severe + model probs)
"""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
SEED = 42; N_BOOT = 1000


def boot(true, prob_or_P, metric_fn, need_binary_classes=0):
    """Generic percentile bootstrap. metric_fn(true, P) -> scalar (nan-skipped)."""
    rng = np.random.RandomState(SEED); n = len(true); out = []
    true = np.asarray(true)
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        try:
            v = metric_fn(true[idx], prob_or_P[idx])
            if v is not None and not np.isnan(v):
                out.append(v)
        except Exception:
            continue
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ============ BINARY EXTERNAL (n=117) ============
EXT = json.load(open(os.path.join(RES, 'external_validation_frozen.json'), encoding='utf-8'))
pp = EXT['patient_probs']
pids = sorted(pp.keys())
b_true = np.array([pp[p]['label'] for p in pids])
b_prob = np.array([pp[p]['prob'] for p in pids])
thr = EXT['frozen_threshold']
b_pred = (b_prob >= thr).astype(int)

b_auc = roc_auc_score(b_true, b_prob)
b_ap = average_precision_score(b_true, b_prob)
b_acc = accuracy_score(b_true, b_pred)
b_f1 = f1_score(b_true, b_pred)
b_auc_lo, b_auc_hi = boot(b_true, b_prob, lambda t, p: roc_auc_score(t, p))
b_ap_lo, b_ap_hi = boot(b_true, b_prob, lambda t, p: average_precision_score(t, p))
b_acc_lo, b_acc_hi = boot(b_true, b_pred, lambda t, p: accuracy_score(t, p))
b_f1_lo, b_f1_hi = boot(b_true, b_pred, lambda t, p: f1_score(t, p))
cm_b = confusion_matrix(b_true, b_pred, labels=[0, 1])

print('=== BINARY EXTERNAL (Swin frozen, n=%d, thr=%.3f) ===' % (len(pids), thr))
print('  AUC : %.3f (%.3f-%.3f)' % (b_auc, b_auc_lo, b_auc_hi))
print('  AP  : %.3f (%.3f-%.3f)' % (b_ap, b_ap_lo, b_ap_hi))
print('  Acc : %.3f (%.3f-%.3f)' % (b_acc, b_acc_lo, b_acc_hi))
print('  F1  : %.3f (%.3f-%.3f)' % (b_f1, b_f1_lo, b_f1_hi))
print('  CM (Tn/Fp, Fn/Tp):', cm_b.tolist())

# ============ TERNARY EXTERNAL (n=57) ============
XLSX = os.path.join(BASE, 'external_validation_results', 'external_validation_results',
                    'patient_level_results.xlsx')
df = pd.read_excel(XLSX)
t_true = df['true_class'].to_numpy()
tP = df[['prob_mild', 'prob_moderate', 'prob_severe']].to_numpy()
t_pred = df['final_pred'].to_numpy()
yb = label_binarize(t_true, classes=[0, 1, 2])

t_auc = np.mean([roc_auc_score(yb[:, c], tP[:, c]) for c in range(3)])
t_ap = np.mean([average_precision_score(yb[:, c], tP[:, c]) for c in range(3)])
t_acc = accuracy_score(t_true, t_pred)
t_f1 = f1_score(t_true, t_pred, average='macro')

def macro_auc(t, P):
    yb_ = label_binarize(t, classes=[0, 1, 2])
    aucs = []
    for c in range(3):
        if len(np.unique(yb_[:, c])) < 2:
            continue  # class absent in this resample -> skip column
        aucs.append(roc_auc_score(yb_[:, c], P[:, c]))
    return np.mean(aucs) if aucs else np.nan

def macro_ap(t, P):
    yb_ = label_binarize(t, classes=[0, 1, 2])
    aps = []
    for c in range(3):
        if len(np.unique(yb_[:, c])) < 2:
            continue
        aps.append(average_precision_score(yb_[:, c], P[:, c]))
    return np.mean(aps) if aps else np.nan

t_auc_lo, t_auc_hi = boot(t_true, tP, macro_auc)
t_ap_lo, t_ap_hi = boot(t_true, tP, macro_ap)
t_acc_lo, t_acc_hi = boot(t_true, t_pred, lambda t, p: accuracy_score(t, p))
t_f1_lo, t_f1_hi = boot(t_true, t_pred, lambda t, p: f1_score(t, p, average='macro'))
cm_t = confusion_matrix(t_true, t_pred, labels=[0, 1, 2])

print('\n=== TERNARY EXTERNAL (Swin, n=%d) ===' % len(df))
print('  class counts (Mild/Mod/Sev):', [int((t_true == c).sum()) for c in range(3)])
print('  Macro AUC : %.3f (%.3f-%.3f)' % (t_auc, t_auc_lo, t_auc_hi))
print('  Macro AP  : %.3f (%.3f-%.3f)' % (t_ap, t_ap_lo, t_ap_hi))
print('  Accuracy  : %.3f (%.3f-%.3f)' % (t_acc, t_acc_lo, t_acc_hi))
print('  F1 (macro): %.3f (%.3f-%.3f)' % (t_f1, t_f1_lo, t_f1_hi))
print('  CM (rows=Mild/Mod/Sev):', cm_t.tolist())

# save normalized ternary result
out = {
    'model': 'canonical-v2 Swin ternary (external validation)',
    'n_total': int(len(df)),
    'class_counts': {'Mild': int((t_true == 0).sum()), 'Moderate': int((t_true == 1).sum()),
                     'Severe': int((t_true == 2).sum())},
    'auc_macro': float(t_auc), 'auc_lo': float(t_auc_lo), 'auc_hi': float(t_auc_hi),
    'ap_macro': float(t_ap), 'ap_lo': float(t_ap_lo), 'ap_hi': float(t_ap_hi),
    'accuracy': float(t_acc), 'acc_lo': float(t_acc_lo), 'acc_hi': float(t_acc_hi),
    'f1_macro': float(t_f1), 'f1_lo': float(t_f1_lo), 'f1_hi': float(t_f1_hi),
    'confusion_matrix': cm_t.tolist(),
}
with open(os.path.join(RES, 'external_ternary_validation_frozen.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('\nSaved ternary result -> results/external_ternary_validation_frozen.json')
