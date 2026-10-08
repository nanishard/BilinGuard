"""
BilinGuard: Train Ternary Grading Models
Trains all vision backbones + YellowFeatures + DynamicFusion ensemble.
"""
import os, sys, json, torch
import torch.nn as nn
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from data_loader import get_dataloaders
from models import create_model, BACKBONE_NAMES
from train_utils import train_epoch, eval_epoch, bootstrap_metrics, compute_metrics


def train_backbone(name, num_classes=3, fold=0):
    print(f'[Grading] Training {name} (fold {fold})...')
    device = torch.device(DEVICE)
    train_loader, val_loader = get_dataloaders(task='grading', fold=fold)
    model = create_model(name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_pred, y_prob = eval_epoch(model, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, np.array(y_prob))
        scheduler.step()
        auc = metrics.get('auc', 0)
        if auc > best_auc:
            best_auc = auc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'{name}_fold{fold}.pt'))
        if epoch % 10 == 0:
            print(f'  Epoch {epoch}: val_loss={val_loss:.4f}, auc={auc:.4f}')
    return best_auc


def train_dynamicfusion(fold=0):
    print(f'[Grading] Training DynamicFusion ensemble (fold {fold})...')
    device = torch.device(DEVICE)
    train_loader, val_loader = get_dataloaders(task='grading', fold=fold)

    backbones = []
    for name in BACKBONE_NAMES:
        model = create_model(name, num_classes=3).to(device)
        ckpt = os.path.join(MODEL_DIR, f'{name}_fold{fold}.pt')
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f'  Loaded pretrained {name}')
        backbones.append(model)

    ensemble = create_model('dynamicfusion', num_classes=3).to(device)
    # Copy weights from backbone classifiers
    with torch.no_grad():
        for i, m in enumerate(backbones):
            ensemble.models[i].load_state_dict(m.state_dict(), strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': ensemble.weights, 'lr': 1e-3},
    ] + [{'params': m.parameters(), 'lr': LR * 0.1} for m in ensemble.models],
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS // 2)

    best_auc = 0
    for epoch in range(1, EPOCHS // 2 + 1):
        train_loss, _, _, _ = train_epoch(ensemble, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_pred, y_prob = eval_epoch(ensemble, val_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred, np.array(y_prob))
        scheduler.step()
        auc = metrics.get('auc', 0)
        if auc > best_auc:
            best_auc = auc
            torch.save(ensemble.state_dict(), os.path.join(MODEL_DIR, f'DynamicFusion_fold{fold}.pt'))
        if epoch % 5 == 0:
            print(f'  Epoch {epoch}: val_loss={val_loss:.4f}, auc={auc:.4f}, weights={torch.softmax(ensemble.weights, dim=0).data.cpu().numpy()}')
    return best_auc


def evaluate_models(fold=0):
    print(f'[Grading] Evaluating all models (fold {fold})...')
    device = torch.device(DEVICE)
    _, val_loader = get_dataloaders(task='grading', fold=fold)
    results = {}

    all_models = BACKBONE_NAMES + ['YellowFeatures', 'DynamicFusion']
    for name in all_models:
        model = create_model(name, num_classes=3).to(device)
        ckpt = os.path.join(MODEL_DIR, f'{name}_fold{fold}.pt')
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        _, y_true, y_pred, y_prob = eval_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
        metrics = compute_metrics(y_true, y_pred, np.array(y_prob))
        boot = bootstrap_metrics(y_true, np.array(y_prob))
        results[name] = {**metrics, **boot}
        print(f'  {name}: AUC={metrics.get("auc",0):.4f}, F1={metrics["f1_macro"]:.4f}')

    os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'grading_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).T.to_csv(os.path.join(RESULTS_DIR, 'tables', 'grading_metrics.csv'))
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not args.skip_train:
        for name in BACKBONE_NAMES + ['YellowFeatures']:
            train_backbone(name, num_classes=3, fold=args.fold)
        train_dynamicfusion(fold=args.fold)

    results = evaluate_models(fold=args.fold)
