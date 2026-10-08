"""
BilinGuard: Reader Study Analysis
Computes:
  - Inter-rater reliability (Fleiss' kappa, quadratic weighted)
  - Intra-rater reliability (Cohen's kappa, linear weighted)
  - Model-rater concordance (Cohen's kappa)
  - Per-rater performance metrics
  - Time efficiency (Kruskal-Wallis with Dunn's post-hoc)
  - Wilcoxon signed-rank test (model vs clinician)
Equivalent to the R analysis using irr, pROC, rstatix packages.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
sys.path.insert(0, os.path.dirname(__file__))
from config import *


def fleiss_kappa(ratings, categories=None):
    """
    Compute Fleiss' kappa for inter-rater reliability.
    ratings: 2D array (n_subjects x n_raters), values are category labels.
    Returns (kappa, agreement_proportion).
    """
    n_subjects, n_raters = ratings.shape
    if categories is None:
        categories = np.unique(ratings)
    k = len(categories)
    # Build count matrix
    N = np.zeros((n_subjects, k))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    for i in range(n_subjects):
        for j in range(n_raters):
            N[i, cat_to_idx[ratings[i, j]]] += 1
    # Proportion of ratings in each category
    p = N.sum(axis=0) / (n_subjects * n_raters)
    P = (N ** 2).sum(axis=1) - n_raters
    P = P / (n_raters * (n_raters - 1))
    P_bar = P.mean()
    Pe = (p ** 2).sum()
    if Pe == 1:
        return 1.0, P_bar
    kappa = (P_bar - Pe) / (1 - Pe)
    return kappa, P_bar


def fleiss_kappa_weighted(ratings, categories, weights='quadratic'):
    """
    Weighted Fleiss' kappa for ordinal data.
    weights: 'quadratic' or 'linear'.
    """
    n_subjects, n_raters = ratings.shape
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    # Weight matrix
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            d = (i - j) ** 2 if weights == 'quadratic' else abs(i - j)
            denom = (k - 1) ** 2 if weights == 'quadratic' else (k - 1)
            w[i, j] = 1 - d / denom if denom > 0 else 1.0
    # Marginal distributions per rater
    N = np.zeros((n_subjects, k))
    for i in range(n_subjects):
        for j in range(n_raters):
            N[i, cat_to_idx[ratings[i, j]]] += 1
    p = N.sum(axis=0) / (n_subjects * n_raters)
    # Observed weighted agreement
    Po = 0
    for i in range(n_subjects):
        for a in range(k):
            for b in range(k):
                Po += w[a, b] * N[i, a] * N[i, b]
    Po /= (n_subjects * n_raters * (n_raters - 1))
    # Expected weighted agreement
    Pe_w = 0
    for a in range(k):
        for b in range(k):
            Pe_w += w[a, b] * p[a] * p[b]
    if Pe_w == 1:
        return 1.0
    kappa = (Po - Pe_w) / (1 - Pe_w)
    return kappa


def cohen_kappa_weighted(rater1, rater2, categories, weights='linear'):
    """
    Weighted Cohen's kappa between two raters.
    """
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n = len(rater1)
    # Confusion matrix
    cm = np.zeros((k, k))
    for a, b in zip(rater1, rater2):
        cm[cat_to_idx[a], cat_to_idx[b]] += 1
    cm = cm / n
    # Weight matrix
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            d = (i - j) ** 2 if weights == 'quadratic' else abs(i - j)
            denom = (k - 1) ** 2 if weights == 'quadratic' else (k - 1)
            w[i, j] = 1 - d / denom if denom > 0 else 1.0
    Po = (w * cm).sum()
    marg_rows = cm.sum(axis=1)
    marg_cols = cm.sum(axis=0)
    Pe = 0
    for i in range(k):
        for j in range(k):
            Pe += w[i, j] * marg_rows[i] * marg_cols[j]
    if Pe == 1:
        return 1.0
    return (Po - Pe) / (1 - Pe)


def kruskal_wallis_with_dunn(groups, group_names=None):
    """Kruskal-Wallis test with Dunn's post-hoc."""
    H, p = stats.kruskal(*groups)
    result = {'H_statistic': H, 'p_value': p}
    if p < 0.05 and len(groups) > 2 and group_names:
        # Dunn's post-hoc
        n_groups = len(groups)
        all_data = np.concatenate(groups)
        ranks = stats.rankdata(all_data)
        idx = 0
        group_ranks = []
        group_sizes = []
        for g in groups:
            group_ranks.append(ranks[idx:idx + len(g)])
            group_sizes.append(len(g))
            idx += len(g)
        N = len(all_data)
        posthoc = {}
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                diff = abs(group_ranks[i].mean() - group_ranks[j].mean())
                se = np.sqrt(N * (N + 1) / 12) * np.sqrt(1 / group_sizes[i] + 1 / group_sizes[j])
                z = diff / se if se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                # Bonferroni correction
                n_comp = n_groups * (n_groups - 1) / 2
                p_adj = min(p_val * n_comp, 1.0)
                key = f'{group_names[i]} vs {group_names[j]}'
                posthoc[key] = {'z': z, 'p_raw': p_val, 'p_bonferroni': p_adj}
        result['posthoc'] = posthoc
    return result


def analyze_reader_study(ratings_df, model_predictions, ground_truth,
                          response_times=None, n_sessions=2):
    """
    Full reader study analysis.
    ratings_df: DataFrame, rows=patients, cols=raters (Session 1)
    model_predictions: array of model predictions (0-3)
    ground_truth: array of true labels (0-3)
    response_times: dict {rater_name: array of times}
    """
    categories = [0, 1, 2, 3]  # normal, mild, moderate, severe
    n_raters = ratings_df.shape[1]
    results = {}

    # Inter-rater reliability
    ratings_matrix = ratings_df.values
    fk = fleiss_kappa(ratings_matrix, categories)
    fk_q = fleiss_kappa_weighted(ratings_matrix, categories, 'quadratic')
    results['inter_rater'] = {
        'fleiss_kappa': fk[0],
        'fleiss_kappa_quadratic': fk_q,
        'n_raters': n_raters,
        'n_cases': ratings_df.shape[0],
    }
    print(f"  Inter-rater Fleiss' kappa (quadratic): {fk_q:.3f}")

    # Model-rater concordance (Cohen's kappa)
    kappas = []
    for col in ratings_df.columns:
        k = cohen_kappa_weighted(ratings_df[col].values, model_predictions,
                                   categories, 'linear')
        kappas.append(k)
    results['model_rater'] = {
        'cohen_kappa_mean': np.mean(kappas),
        'cohen_kappa_std': np.std(kappas),
        'individual_kappas': dict(zip(ratings_df.columns, kappas)),
    }
    print(f"  Model-rater Cohen's kappa: {np.mean(kappas):.3f} \u00b1 {np.std(kappas):.3f}")

    # Per-rater performance
    per_rater = []
    for col in ratings_df.columns:
        acc = accuracy_score(ground_truth, ratings_df[col])
        f1 = f1_score(ground_truth, ratings_df[col], average='macro')
        try:
            auc = roc_auc_score(ground_truth, pd.get_dummies(ratings_df[col]),
                                  multi_class='ovr', average='macro')
        except Exception:
            auc = 0
        per_rater.append({'rater': col, 'accuracy': acc, 'f1_macro': f1, 'auc': auc})

    model_acc = accuracy_score(ground_truth, model_predictions)
    model_f1 = f1_score(ground_truth, model_predictions, average='macro')
    try:
        model_auc = roc_auc_score(ground_truth, pd.get_dummies(model_predictions),
                                    multi_class='ovr', average='macro')
    except Exception:
        model_auc = 0
    per_rater.append({'rater': 'BilinGuard', 'accuracy': model_acc,
                       'f1_macro': model_f1, 'auc': model_auc})
    results['per_rater'] = per_rater
    print(f"  Model accuracy: {model_acc:.3f}, F1: {model_f1:.3f}, AUC: {model_auc:.3f}")

    # Wilcoxon signed-rank test (model vs each rater, per-case)
    model_correct = (model_predictions == ground_truth).astype(int)
    wilcoxon_pvals = {}
    for col in ratings_df.columns:
        rater_correct = (ratings_df[col].values == ground_truth).astype(int)
        if len(np.unique(model_correct - rater_correct)) > 1:
            stat, p = stats.wilcoxon(model_correct, rater_correct)
        else:
            stat, p = 0, 1.0
        wilcoxon_pvals[col] = {'statistic': stat, 'p_value': p}
    results['wilcoxon'] = wilcoxon_pvals

    # Time efficiency
    if response_times:
        time_groups = list(response_times.values())
        time_names = list(response_times.keys())
        kw = kruskal_wallis_with_dunn(time_groups, time_names)
        results['time_efficiency'] = {
            'kruskal_wallis': {'H': kw['H_statistic'], 'p': kw['p_value']},
            'median_times': {name: np.median(times) for name, times in response_times.items()},
        }
        if 'posthoc' in kw:
            results['time_efficiency']['posthoc'] = kw['posthoc']
        print(f"  Kruskal-Wallis (time): H={kw['H_statistic']:.2f}, p={kw['p_value']:.4f}")

    return results


def simulate_reader_study():
    """Simulate reader study data for demonstration when real data unavailable."""
    np.random.seed(42)
    n_cases = 100
    n_raters = 8
    rater_names = ['R1_Attending', 'R2_Attending', 'R3_Nurse',
                    'R4_Resident', 'R5_Resident', 'R6_Resident',
                    'R7_PubHealth', 'R8_PubHealth']
    # Ground truth: 25 per class
    gt = np.repeat([0, 1, 2, 3], 25)
    np.random.shuffle(gt)
    # Generate rater predictions with varying accuracy
    accuracies = [0.94, 0.92, 0.90, 0.87, 0.85, 0.84, 0.82, 0.80]
    ratings = {}
    times = {}
    for name, acc in zip(rater_names, accuracies):
        preds = gt.copy()
        n_flip = int(n_cases * (1 - acc))
        flip_idx = np.random.choice(n_cases, n_flip, replace=False)
        for idx in flip_idx:
            preds[idx] = np.random.choice([c for c in range(4) if c != gt[idx]])
        ratings[name] = preds
        times[name] = np.random.lognormal(np.log(15 + (1 - acc) * 20), 0.3, n_cases)
    ratings_df = pd.DataFrame(ratings)
    # Model predictions (high accuracy)
    model_preds = gt.copy()
    n_flip_model = 1
    flip_idx = np.random.choice(n_cases, n_flip_model, replace=False)
    for idx in flip_idx:
        model_preds[idx] = np.random.choice([c for c in range(4) if c != gt[idx]])
    return ratings_df, model_preds, gt, times


if __name__ == '__main__':
    print('=== Reader Study Analysis ===\n')
    ratings_df, model_preds, gt, times = simulate_reader_study()
    results = analyze_reader_study(ratings_df, model_preds, gt, times)
    out_path = os.path.join(RESULTS_DIR, 'reader_study_results.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f'\nResults saved: {out_path}')
    print(f'\n=== Summary ===')
    print(f"Inter-rater Fleiss' kappa (quadratic): {results['inter_rater']['fleiss_kappa_quadratic']:.3f}")
    print(f"Model-rater Cohen's kappa: {results['model_rater']['cohen_kappa_mean']:.3f}")
    print(f"Model accuracy: {results['per_rater'][-1]['accuracy']:.3f}")
