"""
BilinGuard: Train Binary Classifiers
Includes:
  - YOLO-based image classifier
  - Questionnaire-based ML models (XGBoost, RF, SVM)
"""
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from data_loader import get_dataloaders
from models import create_model
from train_utils import train_epoch, eval_epoch, bootstrap_metrics, compute_metrics


def train_yolo(fold=0):
    print(f'[Binary] Training YOLO (fold {fold})...')
    device = torch.device(DEVICE)
    train_loader, val_loader = get_dataloaders(task='binary', fold=fold)
    model = create_model('yolo', num_classes=2).to(device)
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
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'yolo_binary_fold{fold}.pt'))
        if epoch % 10 == 0:
            print(f'  Epoch {epoch}: val_loss={val_loss:.4f}, auc={auc:.4f}')
    return best_auc


def extract_questionnaire_features():
    """Load questionnaire data and prepare for ML models"""
    path = os.path.join(DATA_PROCESSED, 'questionnaire_clean.csv')
    if not os.path.exists(path):
        print('[Binary] Questionnaire data not found, skipping ML models')
        return None, None
    df = pd.read_csv(path)
    exclude_cols = ['序号_x', '提交答卷时间_x', '所用时间_x', '来源_x', '来源详情_x', '来自IP_x',
                    '序号_y', '提交答卷时间_y', '所用时间_y', '来源_y', '来源详情_y', '来自IP_y',
                    '1、患者姓名', '2、患者住院号', '姓名']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    X = df[feature_cols].fillna(df[feature_cols].median()).values
    y_cols = [c for c in df.columns if '黄疸' in c or 'jaundice' in c.lower()]
    if not y_cols:
        y_cols = ['25、黄疸类型']
    if y_cols[0] in df.columns:
        y = (df[y_cols[0]].notna() & (df[y_cols[0]] != '')).astype(int).values
    else:
        y = np.zeros(len(df))
    return X, y, feature_cols


def train_questionnaire_models():
    print(f'[Binary] Training questionnaire-based ML models...')
    X, y, _ = extract_questionnaire_features()
    if X is None:
        return
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models_dict = {
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED),
        'SVM': SVC(kernel='rbf', probability=True, random_state=SEED),
    }
    results = {}
    for name, clf in models_dict.items():
        clf.fit(X_train_s, y_train)
        y_prob = clf.predict_proba(X_test_s)
        y_pred = clf.predict(X_test_s)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        boot = bootstrap_metrics(y_test, y_prob)
        results[name] = {**metrics, **boot}
        print(f'  {name}: AUC={metrics.get("auc",0):.4f}, F1={metrics["f1_macro"]:.4f}')
        # Save model
        joblib.dump(clf, os.path.join(MODEL_DIR, f'questionnaire_{name.lower()}.pkl'))
    with open(os.path.join(RESULTS_DIR, 'questionnaire_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-yolo', action='store_true')
    parser.add_argument('--skip-questionnaire', action='store_true')
    args = parser.parse_args()

    if not args.skip_yolo:
        aucs = []
        for fold in range(N_FOLDS):
            auc = train_yolo(fold)
            aucs.append(auc)
        print(f'Binary YOLO: mean AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}')

    if not args.skip_questionnaire:
        results = train_questionnaire_models()
