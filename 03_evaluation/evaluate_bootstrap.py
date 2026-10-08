"""
BilinGuard: Comprehensive Bootstrap Evaluation
Computes all performance metrics with 95% CIs via patient-level bootstrap.
Outputs: results JSON + per-model CSV + formatted table.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              accuracy_score, confusion_matrix, precision_recall_curve,
                              roc_curve)
sys.path.insert(0, os.path.dirname(__file__))
from config import *

N_BOOTSTRAP_DEFAULT = 1000

def bootstrap_metric(y_true, y_pred, y_prob, metric_fn, n_boot=N_BOOTSTRAP_DEFAULT, seed=SEED):
    """Bootstrap a single metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = np.array(y_true)[idx]
        yp = np.array(y_pred)[idx]
        ypr = np.array(y_prob)[idx] if y_prob is not None else None
        try:
            v = metric_fn(yt, yp, ypr)
            if v is not None and not np.isnan(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return None, None, None
    mean = np.mean(vals)
    lower = np.percentile(vals, 2.5)
    upper = np.percentile(vals, 97.5)
    return mean, lower, upper


def evaluate_binary(y_true, y_pred, y_prob, n_boot=N_BOOTSTRAP_DEFAULT):
    """Full binary evaluation with bootstrap CIs."""
    results = {}
    # Point estimates
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results['point'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0,
        'specificity': tn / (tn + fp) if (tn + fp) else 0,
        'ppv': tp / (tp + fp) if (tp + fp) else 0,
        'npv': tn / (tn + fn) if (tn + fn) else 0,
        'auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        'ap': average_precision_score(y_true, y_prob) if y_prob is not None else None,
    }
    # Bootstrap CIs
    def m_acc(yt, yp, ypr): return accuracy_score(yt, yp)
    def m_f1(yt, yp, ypr): return f1_score(yt, yp)
    def m_auc(yt, yp, ypr): return roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else None
    def m_ap(yt, yp, ypr): return average_precision_score(yt, ypr) if len(np.unique(yt)) > 1 else None
    def m_sens(yt, yp, ypr):
        cm = confusion_matrix(yt, yp)
        if cm.shape == (2, 2):
            tn2, fp2, fn2, tp2 = cm.ravel()
            return tp2 / (tp2 + fn2) if (tp2 + fn2) else 0
        return None
    def m_spec(yt, yp, ypr):
        cm = confusion_matrix(yt, yp)
        if cm.shape == (2, 2):
            tn2, fp2, fn2, tp2 = cm.ravel()
            return tn2 / (tn2 + fp2) if (tn2 + fp2) else 0
        return None

    for name, fn in [('accuracy', m_acc), ('f1', m_f1), ('sensitivity', m_sens),
                       ('specificity', m_spec), ('auc', m_auc), ('ap', m_ap)]:
        mean, lo, hi = bootstrap_metric(y_true, y_pred, y_prob, fn, n_boot)
        if mean is not None:
            results[f'{name}_ci'] = {'mean': mean, 'lower': lo, 'upper': hi}
    return results


def evaluate_multiclass(y_true, y_pred, y_prob, n_boot=N_BOOTSTRAP_DEFAULT):
    """Full multiclass evaluation with bootstrap CIs."""
    results = {}
    n_classes = len(np.unique(y_true))
    # Point estimates
    results['point'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    if y_prob is not None and n_classes > 1:
        try:
            results['point']['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception:
            pass
        try:
            results['point']['ap_micro'] = average_precision_score(y_true, y_prob, average='micro')
        except Exception:
            pass
    # Per-class
    cm = confusion_matrix(y_true, y_pred)
    for i in range(min(n_classes, cm.shape[0])):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        results[f'class_{i}'] = {
            'sensitivity': tp / (tp + fn) if (tp + fn) else 0,
            'specificity': tn / (tn + fp) if (tn + fp) else 0,
            'ppv': tp / (tp + fp) if (tp + fp) else 0,
        }
    # Bootstrap
    def m_acc(yt, yp, ypr): return accuracy_score(yt, yp)
    def m_f1(yt, yp, ypr): return f1_score(yt, yp, average='macro')
    def m_auc(yt, yp, ypr):
        if ypr is not None and len(np.unique(yt)) > 1:
            return roc_auc_score(yt, ypr, multi_class='ovr')
        return None
    for name, fn in [('accuracy', m_acc), ('f1_macro', m_f1), ('auc_ovr', m_auc)]:
        mean, lo, hi = bootstrap_metric(y_true, y_pred, y_prob, fn, n_boot)
        if mean is not None:
            results[f'{name}_ci'] = {'mean': mean, 'lower': lo, 'upper': hi}
    return results


def format_ci(mean, lower, upper):
    """Format as '0.798 (0.756\u20130.838)'."""
    return f'{mean:.3f} ({lower:.3f}\u2013{upper:.3f})'


def generate_results_table(all_results, task='binary'):
    """Generate formatted results table from evaluation results."""
    rows = []
    for model_name, res in all_results.items():
        point = res.get('point', {})
        row = {'Model': model_name}
        for metric in ['accuracy', 'f1', 'f1_macro', 'sensitivity', 'specificity', 'auc', 'auc_ovr', 'ap']:
            pv = point.get(metric)
            ci = res.get(f'{metric}_ci')
            if ci:
                row[metric] = format_ci(ci['mean'], ci['lower'], ci['upper'])
            elif pv is not None:
                row[metric] = f'{pv:.3f}'
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('=== Bootstrap Evaluation (Demo with Simulated Data) ===\n')
    np.random.seed(42)
    # Simulated binary
    y_true_b = np.random.binomial(1, 0.4, 200)
    y_prob_b = np.clip(y_true_b * 0.9 + np.random.normal(0.2, 0.15, 200), 0, 1)
    y_pred_b = (y_prob_b > 0.5).astype(int)
    print('Binary evaluation:')
    res_b = evaluate_binary(y_true_b, y_pred_b, y_prob_b, n_boot=500)
    for k, v in res_b.items():
        if 'ci' in k:
            print(f'  {k}: {format_ci(v["mean"], v["lower"], v["upper"])}')
        elif isinstance(v, dict) and 'accuracy' in v:
            for mk, mv in v.items():
                print(f'  {mk}: {mv:.3f}' if isinstance(mv, float) else f'  {mk}: {mv}')

    # Simulated multiclass
    y_true_m = np.random.randint(0, 3, 200)
    y_prob_m = np.random.dirichlet(np.ones(3) * 2, 200)
    y_pred_m = y_prob_m.argmax(axis=1)
    print('\nMulticlass evaluation:')
    res_m = evaluate_multiclass(y_true_m, y_pred_m, y_prob_m, n_boot=500)
    for k, v in res_m.items():
        if 'ci' in k:
            print(f'  {k}: {format_ci(v["mean"], v["lower"], v["upper"])}')

    # Save demo
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, 'bootstrap_evaluation_demo.json')
    with open(out, 'w') as f:
        json.dump({'binary': res_b, 'multiclass': res_m}, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f'\nSaved: {out}')
    print('\nNote: Run with real model predictions for publication results.')
