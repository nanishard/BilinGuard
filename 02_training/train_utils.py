"""
BilinGuard: Training Utilities
"""
import os, json, torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from config import *

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return total_loss / len(loader), all_labels, all_preds, all_probs

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        labels = batch['label'].to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return total_loss / len(loader), all_labels, all_preds, all_probs

def bootstrap_metrics(y_true, y_prob, n_bootstrap=1000, ci=95):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    metrics_list = []
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 1
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_t = np.array(y_true)[idx]
        if n_classes > 1:
            y_p = y_prob[idx]
            y_pred = y_p.argmax(axis=1)
        else:
            y_p = y_prob[idx]
            y_pred = (y_p > 0.5).astype(int)
        if len(np.unique(y_t)) < 2:
            continue
        try:
            if n_classes > 1:
                auc = roc_auc_score(y_t, y_p, multi_class='ovr')
            else:
                auc = roc_auc_score(y_t, y_p)
            ap = average_precision_score(y_t, y_p) if n_classes == 2 else 0
            f1 = f1_score(y_t, y_pred, average='macro' if n_classes > 2 else 'binary')
            acc = accuracy_score(y_t, y_pred)
            metrics_list.append({'auc': auc, 'ap': ap, 'f1': f1, 'acc': acc})
        except Exception:
            continue
    if not metrics_list:
        return {}
    df = {k: [m[k] for m in metrics_list] for k in metrics_list[0]}
    result = {}
    for k, v in df.items():
        result[k] = float(np.mean(v))
        result[f'{k}_lower'] = float(np.percentile(v, (100 - ci) / 2))
        result[f'{k}_upper'] = float(np.percentile(v, 100 - (100 - ci) / 2))
    return result

def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = None, None, None, None
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    n_classes = len(np.unique(y_true))
    sens, spec = 0, 0
    if n_classes == 2 and cm.shape == (2, 2):
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'sensitivity': sens,
        'specificity': spec,
    }
    if y_prob is not None:
        if n_classes > 1 and y_prob.shape[1] > 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception:
                metrics['auc'] = 0
        elif n_classes == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except Exception:
                metrics['auc'] = 0
    return metrics
