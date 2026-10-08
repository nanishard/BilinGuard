# -*- coding: utf-8 -*-
"""
Questionnaire ML Models (XGBoost, RF, SVM) for Binary Screening + External Validation Bootstrap.
Table 2 requires: XGBoost, RF, SVM with F1/Sens/Spec/Acc/AUC/AP + 95% CI.
External validation: Bootstrap CI for all external models.
"""
import os, json, random, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
TBL = os.path.join(RES, 'tables')
FIG = os.path.join(RES, 'figures', 'bootstrap')
os.makedirs(FIG, exist_ok=True)
SEED = 42; N_BOOT = 1000
random.seed(SEED); np.random.seed(SEED)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('WARNING: xgboost not installed, using GradientBoosting instead')
    from sklearn.ensemble import GradientBoostingClassifier as XGBFallback


def compute_all_metrics(true, probs, nc=2):
    pred = probs.argmax(1)
    cm = confusion_matrix(true, pred, labels=list(range(nc)))
    try: auc = roc_auc_score(true, probs[:, 1]) if nc == 2 else roc_auc_score(true, probs, multi_class='ovr', labels=list(range(nc)))
    except: auc = 0
    try: ap = average_precision_score(true, probs[:, 1]) if nc == 2 else np.mean([average_precision_score((true==c).astype(int), probs[:,c]) for c in range(nc)])
    except: ap = 0
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro', labels=list(range(nc)), zero_division=0)
    tp = cm[1,1] if nc==2 else np.trace(cm)
    fn = cm[1,:].sum()-cm[1,1] if nc==2 else (cm.sum()-np.trace(cm))
    fp = cm[:,1].sum()-cm[1,1] if nc==2 else 0
    tn = cm[0,0] if nc==2 else 0
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    if nc > 2:
        sens_list, spec_list = [], []
        for c in range(nc):
            tp_c = cm[c,c]; fn_c = cm[c,:].sum()-tp_c; fp_c = cm[:,c].sum()-tp_c; tn_c = cm.sum()-tp_c-fn_c-fp_c
            sens_list.append(tp_c/(tp_c+fn_c) if (tp_c+fn_c)>0 else 0)
            spec_list.append(tn_c/(tn_c+fp_c) if (tn_c+fp_c)>0 else 0)
        sens = float(np.mean(sens_list)); spec = float(np.mean(spec_list))
    return {'auc': float(auc), 'ap': float(ap), 'acc': float(acc), 'f1': float(f1),
            'sens': float(sens), 'spec': float(spec)}


def bootstrap_ci(true, probs, nc=2, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    n = len(true)
    res = {k: [] for k in ['auc','ap','acc','f1','sens','spec']}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = true[idx], probs[idx]
        if len(np.unique(yt)) < 2: continue
        m = compute_all_metrics(yt, yp, nc)
        for k in res:
            if not np.isnan(m[k]): res[k].append(m[k])
    def ci(arr): arr=[x for x in arr if not np.isnan(x)]; return (float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))) if arr else (0,0)
    out = {}
    for k in res: lo, hi = ci(res[k]); out[f'{k}_lo'] = lo; out[f'{k}_hi'] = hi
    return out


def fmt(val, ci_lo, ci_hi):
    return f'{val:.3f} [{ci_lo:.3f}-{ci_hi:.3f}]'


# ═════════════════════════════════════════════════════════════
# PART 1: Questionnaire ML Models
# ═════════════════════════════════════════════════════════════
print('='*70)
print('  PART 1: Questionnaire ML Models (XGBoost, RF, SVM)')
print('='*70)

df = pd.read_excel(os.path.join(BASE, 'baseline6.17.xlsx'))
df.columns = ['name','hid','diagnosis','group','gender','age','height','weight','bmi','heart_rate','waist','hip']

# Features
df['whr'] = df['waist'] / df['hip']  # Waist-to-hip ratio
feature_cols = ['gender','age','height','weight','bmi','heart_rate','waist','hip','whr']
X = df[feature_cols].values.astype(float)
y = (df['group'] == 2).astype(int).values  # 0=normal, 1=jaundiced

print(f'  Dataset: {len(y)} patients (normal={sum(y==0)}, jaundiced={sum(y==1)})')
print(f'  Features: {feature_cols}')

# Train/test split (80/20, stratified)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f'  Train: {len(y_tr)} (normal={sum(y_tr==0)}, jaundiced={sum(y_tr==1)})')
print(f'  Val:   {len(y_va)} (normal={sum(y_va==0)}, jaundiced={sum(y_va==1)})')

# Standardize
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

questionnaire_results = []

# XGBoost
if HAS_XGB:
    model_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                   random_state=SEED, eval_metric='logloss', use_label_encoder=False)
    model_xgb.fit(X_tr, y_tr)
    probs_xgb = model_xgb.predict_proba(X_va)
    m = compute_all_metrics(y_va, probs_xgb)
    ci = bootstrap_ci(y_va, probs_xgb)
    questionnaire_results.append({'name': 'XGBoost', **m, **ci, 'n_val': len(y_va)})
    print(f'  XGBoost: AUC={fmt(m["auc"],ci["auc_lo"],ci["auc_hi"])} Sens={fmt(m["sens"],ci["sens_lo"],ci["sens_hi"])} Spec={fmt(m["spec"],ci["spec_lo"],ci["spec_hi"])}')

# Random Forest
model_rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED)
model_rf.fit(X_tr, y_tr)
probs_rf = model_rf.predict_proba(X_va)
m = compute_all_metrics(y_va, probs_rf)
ci = bootstrap_ci(y_va, probs_rf)
questionnaire_results.append({'name': 'Random Forest', **m, **ci, 'n_val': len(y_va)})
print(f'  RF:      AUC={fmt(m["auc"],ci["auc_lo"],ci["auc_hi"])} Sens={fmt(m["sens"],ci["sens_lo"],ci["sens_hi"])} Spec={fmt(m["spec"],ci["spec_lo"],ci["spec_hi"])}')

# SVM
model_svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=SEED)
model_svm.fit(X_tr_s, y_tr)
probs_svm = model_svm.predict_proba(X_va_s)
m = compute_all_metrics(y_va, probs_svm)
ci = bootstrap_ci(y_va, probs_svm)
questionnaire_results.append({'name': 'SVM', **m, **ci, 'n_val': len(y_va)})
print(f'  SVM:     AUC={fmt(m["auc"],ci["auc_lo"],ci["auc_hi"])} Sens={fmt(m["sens"],ci["sens_lo"],ci["sens_hi"])} Spec={fmt(m["spec"],ci["spec_lo"],ci["spec_hi"])}')

# SHAP for XGBoost (Figure 2A)
if HAS_XGB:
    try:
        import shap
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(X_va)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_va, feature_names=feature_cols, show=False, plot_type='bar')
        plt.title('SHAP Feature Importance (XGBoost)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig_path = os.path.join(RES, 'figures', 'Fig2A_SHAP.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  SHAP figure saved: {fig_path}')
    except Exception as e:
        print(f'  SHAP failed: {e}')

# ROC curves for questionnaire models
fig, ax = plt.subplots(figsize=(7, 6))
from sklearn.metrics import roc_curve
for name, probs, color in [('XGBoost', probs_xgb, '#E76F51'), ('Random Forest', probs_rf, '#2A9D8F'), ('SVM', probs_svm, '#457B9D')]:
    fpr, tpr, _ = roc_curve(y_va, probs[:, 1])
    auc_v = roc_auc_score(y_va, probs[:, 1])
    ax.plot(fpr, tpr, lw=1.5, color=color, label=f'{name} (AUC={auc_v:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Questionnaire ML Models: Binary Screening ROC', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig_path = os.path.join(RES, 'figures', 'Fig2B_questionnaire_roc.png')
fig.savefig(fig_path, dpi=300); fig.savefig(fig_path.replace('.png', '.svg'))
plt.close()
print(f'  ROC figure saved: {fig_path}')

# PR curves
fig, ax = plt.subplots(figsize=(7, 6))
from sklearn.metrics import precision_recall_curve
for name, probs, color in [('XGBoost', probs_xgb, '#E76F51'), ('Random Forest', probs_rf, '#2A9D8F'), ('SVM', probs_svm, '#457B9D')]:
    precision, recall, _ = precision_recall_curve(y_va, probs[:, 1])
    ap_v = average_precision_score(y_va, probs[:, 1])
    ax.plot(recall, precision, lw=1.5, color=color, label=f'{name} (AP={ap_v:.3f})')
ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
ax.set_ylabel('Precision (PPV)', fontsize=11)
ax.set_title('Precision-Recall Curves: Questionnaire Models', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
fig_path = os.path.join(RES, 'figures', 'Fig2E_questionnaire_pr.png')
fig.savefig(fig_path, dpi=300); fig.savefig(fig_path.replace('.png', '.svg'))
plt.close()
print(f'  PR figure saved: {fig_path}')


# ═════════════════════════════════════════════════════════════
# PART 2: External Validation Bootstrap CI
# ═════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('  PART 2: External Validation Bootstrap CI')
print('='*70)

# Load external validation predictions (re-compute from models)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm, cv2
from PIL import Image

EXT = os.path.join(BASE, 'data', 'external_processed_v3')
MODEL_DIR = os.path.join(BASE, 'models')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

def predict_ext(model, patient_faces):
    valid = [f for f in patient_faces[:12] if os.path.exists(f)]
    if not valid: return np.array([0.5, 0.5])
    tensors = []
    for f in valid:
        img = read_img(f)
        if img is None: continue
        img = clahe(img)
        tensors.append(ev_tf(Image.fromarray(img).convert('RGB')).unsqueeze(0))
    if not tensors: return np.array([0.5, 0.5])
    t = torch.cat(tensors).to(DEV)
    with torch.no_grad():
        return F.softmax(model(t), dim=1).mean(0).cpu().numpy()

# Build external patient list
ext_patients = []
for label, lbl_int in [('normal', 0), ('jaundice', 1)]:
    cat_dir = os.path.join(EXT, label)
    if not os.path.exists(cat_dir): continue
    for pid in sorted(os.listdir(cat_dir)):
        p_dir = os.path.join(cat_dir, pid)
        if not os.path.isdir(p_dir): continue
        faces = sorted([os.path.join(p_dir, f) for f in os.listdir(p_dir) if '_face' in f and f.endswith('.jpg')])
        if faces:
            ext_patients.append({'faces': faces, 'label': lbl_int, 'pid': pid})

y_ext = np.array([p['label'] for p in ext_patients])
print(f'  External: {len(ext_patients)} patients (normal={sum(y_ext==0)}, jaundice={sum(y_ext==1)})')

EXT_MODELS = [
    ('Ext-TypeBin-ConvNeXt', 'convnext_tiny', 'type_binary_convnext_tiny.pt'),
    ('Ext-TypeBin-ViT', 'vit_tiny_patch16_224', 'type_binary_vit_tiny_patch16_224.pt'),
    ('Ext-TypeBin-EffNet', 'efficientnet_b0', 'type_binary_efficientnet_b0.pt'),
    ('Ext-TypeBin-Swin', 'swin_tiny_patch4_window7_224', 'type_binary_swin_tiny_patch4_window7_224.pt'),
    ('Ext-V3Bin-ConvNeXt', 'convnext_tiny', 'v3_binary_convnext_tiny.pt'),
    ('Ext-V3Bin-ViT', 'vit_tiny_patch16_224', 'v3_binary_vit_tiny_patch16_224.pt'),
    ('Ext-V3Bin-EffNet', 'efficientnet_b0', 'v3_binary_efficientnet_b0.pt'),
]

ext_results = []
ext_all_preds = {}
for name, bb, ckpt in EXT_MODELS:
    path = os.path.join(MODEL_DIR, ckpt)
    if not os.path.exists(path):
        print(f'    SKIP {name}: not found'); continue
    m = timm.create_model(bb, pretrained=False, num_classes=2).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True)); m.eval()
    probs = np.array([predict_ext(m, p['faces']) for p in ext_patients])
    del m; torch.cuda.empty_cache()
    ext_all_preds[name] = probs
    metrics = compute_all_metrics(y_ext, probs)
    ci = bootstrap_ci(y_ext, probs)
    ext_results.append({'name': name, 'task': 'External-Validation', 'nc': 2, 'n_val': len(y_ext), **metrics, **ci})
    print(f'  {name}: AUC={fmt(metrics["auc"],ci["auc_lo"],ci["auc_hi"])} Sens={fmt(metrics["sens"],ci["sens_lo"],ci["sens_hi"])} Spec={fmt(metrics["spec"],ci["spec_lo"],ci["spec_hi"])}')

# YOLO external
from ultralytics import YOLO
yolo_path = os.path.join(MODEL_DIR, 'v3_yolo_binary.pt')
if os.path.exists(yolo_path):
    ym = YOLO(yolo_path)
    probs = []
    for p in ext_patients:
        faces = [f for f in p['faces'][:12] if os.path.exists(f)]
        ps = []
        for fp in faces:
            r = ym.predict(fp, verbose=False)
            if r and len(r) > 0: ps.append(r[0].probs.data.cpu().numpy())
        probs.append(np.mean(ps, axis=0) if ps else [0.5, 0.5])
    probs = np.array(probs)
    ext_all_preds['YOLO'] = probs
    metrics = compute_all_metrics(y_ext, probs)
    ci = bootstrap_ci(y_ext, probs)
    ext_results.append({'name': 'Ext-YOLO-Binary', 'task': 'External-Validation', 'nc': 2, 'n_val': len(y_ext), **metrics, **ci})
    print(f'  Ext-YOLO-Binary: AUC={fmt(metrics["auc"],ci["auc_lo"],ci["auc_hi"])} Sens={fmt(metrics["sens"],ci["sens_lo"],ci["sens_hi"])} Spec={fmt(metrics["spec"],ci["spec_lo"],ci["spec_hi"])}')

# Ensemble
if len(ext_all_preds) >= 2:
    ens = np.mean(list(ext_all_preds.values()), axis=0)
    metrics = compute_all_metrics(y_ext, ens)
    ci = bootstrap_ci(y_ext, ens)
    ext_results.append({'name': 'Ext-Ensemble', 'task': 'External-Validation', 'nc': 2, 'n_val': len(y_ext), **metrics, **ci})
    print(f'  Ext-Ensemble: AUC={fmt(metrics["auc"],ci["auc_lo"],ci["auc_hi"])}')


# ═════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ═════════════════════════════════════════════════════════════
print('\n' + '='*90)
print('  MANUSCRIPT TABLE FORMAT')
print('='*90)

# Combine questionnaire + external
all_new = []
for r in questionnaire_results:
    all_new.append({**r, 'task': 'Questionnaire-ML', 'nc': 2})
for r in ext_results:
    all_new.append(r)

print('\nModel\tF1 score\tSensitivity\tSpecificity\tAccuracy\tAUC ROC\tAverage Precision')
print('-'*120)
for r in sorted(all_new, key=lambda x: (x.get('task',''), -x.get('auc',0))):
    row = '\t'.join([
        r['name'],
        fmt(r['f1'], r['f1_lo'], r['f1_hi']),
        fmt(r['sens'], r['sens_lo'], r['sens_hi']),
        fmt(r['spec'], r['spec_lo'], r['spec_hi']),
        fmt(r['acc'], r['acc_lo'], r['acc_hi']),
        fmt(r['auc'], r['auc_lo'], r['auc_hi']),
        fmt(r['ap'], r['ap_lo'], r['ap_hi']),
    ])
    print(row)

# TSV
with open(os.path.join(TBL, 'questionnaire_and_external_bootstrap.tsv'), 'w', encoding='utf-8') as f:
    f.write('Task\tModel\tF1 score\tSensitivity\tSpecificity\tAccuracy\tAUC ROC\tAverage Precision\n')
    for r in sorted(all_new, key=lambda x: (x.get('task',''), -x.get('auc',0))):
        f.write('\t'.join([
            r.get('task',''), r['name'],
            fmt(r['f1'], r['f1_lo'], r['f1_hi']),
            fmt(r['sens'], r['sens_lo'], r['sens_hi']),
            fmt(r['spec'], r['spec_lo'], r['spec_hi']),
            fmt(r['acc'], r['acc_lo'], r['acc_hi']),
            fmt(r['auc'], r['auc_lo'], r['auc_hi']),
            fmt(r['ap'], r['ap_lo'], r['ap_hi']),
        ]) + '\n')

# JSON
with open(os.path.join(RES, 'questionnaire_and_external_bootstrap.json'), 'w') as f:
    json.dump(all_new, f, indent=2, default=str)

# External ROC figure
fig, ax = plt.subplots(figsize=(8, 7))
for name, probs in ext_all_preds.items():
    fpr, tpr, _ = roc_curve(y_ext, probs[:, 1])
    auc_v = roc_auc_score(y_ext, probs[:, 1])
    short = name.replace('Ext-TypeBin-','T-').replace('Ext-V3Bin-','V3-').replace('Ext-','')
    ax.plot(fpr, tpr, lw=1.3, label=f'{short} ({auc_v:.3f})')
ax.plot([0,1],[0,1],'k--',lw=0.5,alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('External Validation: Binary Screening ROC', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
fig_path = os.path.join(RES, 'figures', 'Fig_external_roc_with_ci.png')
fig.savefig(fig_path, dpi=300); fig.savefig(fig_path.replace('.png','.svg'))
plt.close()

print(f'\n  TSV:    {TBL}/questionnaire_and_external_bootstrap.tsv')
print(f'  JSON:   {RES}/questionnaire_and_external_bootstrap.json')
print(f'  SHAP:   {RES}/figures/Fig2A_SHAP.png')
print(f'  ROC:    {RES}/figures/Fig2B_questionnaire_roc.png')
print(f'  PR:     {RES}/figures/Fig2E_questionnaire_pr.png')
print(f'  ExtROC: {fig_path}')
