# -*- coding: utf-8 -*-
"""
COMPREHENSIVE Bootstrap Evaluation — ALL models × ALL metrics.
Metrics: AUC, Accuracy, Sensitivity, Specificity, F1 — all with 95% CI.
1000 bootstrap iterations, patient-level resampling.

Model groups:
  G1: Binary screening (Normal vs Jaundice) — face images
  G2: TBIL eyelid ternary (mild/moderate/severe) — opt + clean + original
  G3: Face ternary v3 (mild/moderate/severe) — SAM v3 face images
  G4: YOLO v3 binary + ternary
  G5: DBIL ternary (re-run with full metrics)
  G6: IBIL ternary (re-run with full metrics)
  G7: Jaundice type (binary hepato vs chol)
  G8: External validation (binary)
"""
import os, re, json, random, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, recall_score, precision_score,
                              average_precision_score)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
FIG = os.path.join(RES, 'figures', 'bootstrap')
TBL = os.path.join(RES, 'tables')
os.makedirs(FIG, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMG_SIZE = 224; N_BOOT = 1000
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})


def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None

def clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def predict_patient(model, paths, n_max=12):
    valid = [p for p in paths[:n_max] if os.path.exists(p)]
    if not valid: return None
    tensors = []
    for p in valid:
        img = read_img(p)
        if img is None: continue
        img = clahe(img)
        tensors.append(ev_tf(Image.fromarray(img).convert('RGB')).unsqueeze(0))
    if not tensors: return None
    t = torch.cat(tensors).to(DEV)
    with torch.no_grad():
        return F.softmax(model(t), dim=1).mean(0).cpu().numpy()


def compute_all_metrics(true, probs, nc):
    """Compute point estimates for all metrics."""
    pred = probs.argmax(1)
    cm = confusion_matrix(true, pred, labels=list(range(nc)))

    # AUC
    try:
        auc = roc_auc_score(true, probs[:, 1]) if nc == 2 else roc_auc_score(true, probs, multi_class='ovr', labels=list(range(nc)))
    except Exception:
        auc = 0

    # Average Precision (AP)
    try:
        if nc == 2:
            ap = average_precision_score(true, probs[:, 1])
        else:
            # For multiclass: label_binarize then average
            from sklearn.preprocessing import label_binarize
            yb = label_binarize(true, classes=list(range(nc)))
            ap = np.mean([average_precision_score(yb[:, c], probs[:, c]) for c in range(nc)])
    except Exception:
        ap = 0

    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro', labels=list(range(nc)), zero_division=0)

    # Sensitivity (recall) and Specificity per class
    sens_list, spec_list = [], []
    for c in range(nc):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    if nc == 2:
        sens = sens_list[1]  # Sensitivity for positive class (jaundice)
        spec = spec_list[1]  # Specificity for positive class
    else:
        sens = float(np.mean(sens_list))  # Macro-averaged
        spec = float(np.mean(spec_list))

    return {'auc': auc, 'ap': ap, 'acc': acc, 'f1': f1, 'sens': sens, 'spec': spec,
            'sens_list': sens_list, 'spec_list': spec_list}


def bootstrap_all(true, probs, nc, n_boot=N_BOOT):
    """Bootstrap all metrics."""
    rng = np.random.RandomState(SEED)
    n = len(true)
    results = {k: [] for k in ['auc', 'ap', 'acc', 'f1', 'sens', 'spec']}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = true[idx]; yp = probs[idx]
        if len(np.unique(yt)) < 2: continue
        m = compute_all_metrics(yt, yp, nc)
        for k in results:
            if not np.isnan(m[k]):
                results[k].append(m[k])

    def ci(key):
        arr = results[key]
        if not arr: return (0, 0)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {f'{k}_lo': ci(k)[0] for k in results} | {f'{k}_hi': ci(k)[1] for k in results}


def eval_model(name, backbone_id, ckpt, nc, val_items, img_key='faces'):
    """Full evaluation: load, predict, compute metrics + bootstrap."""
    if not val_items:
        print(f'    SKIP {name}: empty val set'); return None
    path = os.path.join(MODEL, ckpt)
    if not os.path.exists(path):
        print(f'    SKIP {name}: not found'); return None
    try:
        m = timm.create_model(backbone_id, pretrained=False, num_classes=nc).to(DEV)
        m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True)); m.eval()
    except Exception as e:
        print(f'    SKIP {name}: {e}'); return None

    probs, true = [], []
    for item in tqdm(val_items, desc=name, leave=False):
        imgs = item.get(img_key, item.get('faces', []))
        pr = predict_patient(m, imgs)
        if pr is None: pr = np.ones(nc) / nc
        probs.append(pr); true.append(item['label'])
    probs = np.array(probs); true = np.array(true)
    del m; torch.cuda.empty_cache()

    point = compute_all_metrics(true, probs, nc)
    bs = bootstrap_all(true, probs, nc)

    result = {
        'name': name, 'nc': nc, 'n_val': len(true),
        'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
        'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
        'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
        'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
        'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
        'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi'],
    }
    print(f'    {name}: AUC={point["auc"]:.3f}[{bs["auc_lo"]:.3f}-{bs["auc_hi"]:.3f}] '
          f'Sens={point["sens"]:.3f}[{bs["sens_lo"]:.3f}-{bs["sens_hi"]:.3f}] '
          f'Spec={point["spec"]:.3f}[{bs["spec_lo"]:.3f}-{bs["spec_hi"]:.3f}] '
          f'F1={point["f1"]:.3f} AP={point["ap"]:.3f} N={len(true)}')
    return result, probs, true


def patient_level(probs_list, true_list, pid_list):
    """Aggregate image-level to patient-level."""
    d = {}
    for p, l, pid in zip(probs_list, true_list, pid_list):
        d.setdefault(pid, ([], l)); d[pid][0].append(p)
    pids = list(d.keys())
    avg = np.array([np.mean(d[p][0], 0) for p in pids])
    true = np.array([d[p][1] for p in pids])
    return avg, true, pids


# ── Data builders ────────────────────────────────────────────
def load_excel():
    df_ex = pd.read_excel(EXCEL)
    df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)
    dbil_col = [c for c in df_ex.columns if '直接胆红素' in str(c)][0]
    ibil_col = [c for c in df_ex.columns if '间接胆红素' in str(c)][0]
    jt_col = [c for c in df_ex.columns if '黄疸类型' in str(c)][0]
    df_ex['dbil'] = pd.to_numeric(df_ex[dbil_col], errors='coerce')
    df_ex['ibil'] = pd.to_numeric(df_ex[ibil_col], errors='coerce')
    return df_ex, jt_col

df_ex, jt_col = load_excel()

def get_lab(pid, col):
    nums = re.findall(r'\d+', pid.replace('.zip', ''))
    hid = nums[-1] if nums else None
    if not hid: return None
    hn = hid.lstrip('0')
    rows = df_ex[df_ex['hid_str'].apply(lambda s: s.lstrip('0') if s else s) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(lambda s: s.lstrip('0') if s else s).str.endswith(hn[-6:])]
    return float(rows.iloc[0][col]) if len(rows) > 0 else None

def dbil_grade(v):
    if v is None or pd.isna(v): return None
    return 0 if v <= 10 else (1 if v <= 68 else 2)

def ibil_grade(v):
    if v is None or pd.isna(v): return None
    return 0 if v <= 20 else (1 if v <= 50 else 2)


def collect_sam_patients():
    """Collect all patients from face_v4 with faces + eyelids."""
    patients = {}
    for cat in ['normal', 'mild', 'moderate', 'severe']:
        cd = os.path.join(DATA, cat)
        if not os.path.exists(cd): continue
        for pid in os.listdir(cd):
            p = os.path.join(cd, pid)
            if not os.path.isdir(p): continue
            faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
            if faces:
                patients[pid] = {'faces': faces, 'old_cat': cat}
    mdf = pd.read_csv(MANIFEST)
    for _, r in mdf.iterrows():
        if r['patient_id'] in patients and r['n_eyelid'] > 0:
            patients[r['patient_id']]['eyelids'] = json.loads(r['eyelid_images'])
    return patients


def stratified_split(pid_to_grade, ratio=0.2):
    by_grade = {}
    for pid, g in pid_to_grade.items():
        by_grade.setdefault(g, []).append(pid)
    tr, va = set(), set()
    for g, pids in by_grade.items():
        random.seed(SEED); random.shuffle(pids)
        n = max(1, int(len(pids) * ratio))
        va.update(pids[:n]); tr.update(pids[n:])
    return tr, va


BACKBONES = [('convnext', 'convnext_tiny'), ('vit', 'vit_tiny_patch16_224'),
             ('efficientnet', 'efficientnet_b0'), ('swin', 'swin_tiny_patch4_window7_224')]

all_results = []
sam_pats = collect_sam_patients()
mdf = pd.read_csv(MANIFEST)

# ═════════════════════════════════════════════════════════════
# G1: BINARY SCREENING (Normal vs Jaundice) — Face
# ═════════════════════════════════════════════════════════════
print('\n' + '█'*60)
print('  G1: Binary Screening (Normal vs Jaundice) — Face')
print('█'*60)

bin_pids = {}
for pid, p in sam_pats.items():
    bin_pids[pid] = 0 if p['old_cat'] == 'normal' else 1
tr_b, va_b = stratified_split(bin_pids)
va_bin = [{'faces': sam_pats[pid]['faces'], 'label': bin_pids[pid], 'pid': pid}
          for pid in va_b if pid in sam_pats]
from collections import Counter
print(f'  Val: {len(va_bin)} {Counter(p["label"] for p in va_bin)}')

BINARY_FACE = [
    ('Bin-type-ConvNeXt', 'convnext_tiny', 'type_binary_convnext_tiny.pt'),
    ('Bin-type-ViT', 'vit_tiny_patch16_224', 'type_binary_vit_tiny_patch16_224.pt'),
    ('Bin-type-EffNet', 'efficientnet_b0', 'type_binary_efficientnet_b0.pt'),
    ('Bin-type-Swin', 'swin_tiny_patch4_window7_224', 'type_binary_swin_tiny_patch4_window7_224.pt'),
    ('Bin-v3-ConvNeXt', 'convnext_tiny', 'v3_binary_convnext_tiny.pt'),
    ('Bin-v3-ViT', 'vit_tiny_patch16_224', 'v3_binary_vit_tiny_patch16_224.pt'),
    ('Bin-v3-EffNet', 'efficientnet_b0', 'v3_binary_efficientnet_b0.pt'),
]
bin_preds = {}
for name, bb, ckpt in BINARY_FACE:
    r = eval_model(name, bb, ckpt, 2, va_bin, 'faces')
    if r:
        res, probs, true = r; res['task'] = 'Binary-Screening'; res['modality'] = 'Face'
        all_results.append(res); bin_preds[name] = (probs, true)

if len(bin_preds) >= 2:
    common_true = list(bin_preds.values())[0][1]
    ens = np.mean([p for p, _ in bin_preds.values()], axis=0)
    point = compute_all_metrics(common_true, ens, 2)
    bs = bootstrap_all(common_true, ens, 2)
    all_results.append({'name': 'Bin-Ensemble', 'task': 'Binary-Screening', 'modality': 'Face',
                         'nc': 2, 'n_val': len(common_true),
                         'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
                         'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
                         'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
                         'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
                         'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
                         'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi']})
    print(f'    Bin-Ensemble: AUC={point["auc"]:.3f}')

# ═════════════════════════════════════════════════════════════
# G2: TBIL EYELID TERNARY (mild/moderate/severe)
# ═════════════════════════════════════════════════════════════
print('\n' + '█'*60)
print('  G2: TBIL Eyelid Ternary (mild/moderate/severe)')
print('█'*60)

tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
j_df = mdf[mdf['category'].isin(['mild','moderate','severe']) & (mdf['n_eyelid']>0)]
tbil_val = []
for _, r in j_df.iterrows():
    imgs = json.loads(r['eyelid_images'])
    tbil_val.append({'eyelids': imgs, 'label': tmap[r['category']], 'pid': r['patient_id']})
random.seed(SEED)
by_l = {}
for p in tbil_val: by_l.setdefault(p['label'], []).append(p)
va_t = []
for l, g in by_l.items():
    random.shuffle(g); va_t.extend(g[:max(1, int(len(g)*0.2))])
print(f'  Val: {len(va_t)} {Counter(p["label"] for p in va_t)}')

EYE_MODELS = [
    ('TBIL-Opt-ConvNeXt', 'convnext_tiny', 'eyelid_opt_convnext_tiny.pt'),
    ('TBIL-Opt-ViT', 'vit_tiny_patch16_224', 'eyelid_opt_vit_tiny_patch16_224.pt'),
    ('TBIL-Opt-Swin', 'swin_tiny_patch4_window7_224', 'eyelid_opt_swin_tiny_patch4_window7_224.pt'),
    ('TBIL-Clean-ConvNeXt', 'convnext_tiny', 'eyelid_clean_ternary_convnext_tiny.pt'),
    ('TBIL-Clean-ViT', 'vit_tiny_patch16_224', 'eyelid_clean_ternary_vit_tiny_patch16_224.pt'),
    ('TBIL-Clean-EffNet', 'efficientnet_b0', 'eyelid_clean_ternary_efficientnet_b0.pt'),
]
eye_preds = {}
for name, bb, ckpt in EYE_MODELS:
    r = eval_model(name, bb, ckpt, 3, va_t, 'eyelids')
    if r:
        res, probs, true = r; res['task'] = 'TBIL-Eyelid'; res['modality'] = 'Eyelid'
        all_results.append(res); eye_preds[name] = (probs, true)

if len(eye_preds) >= 2:
    common_true = list(eye_preds.values())[0][1]
    ens = np.mean([p for p, _ in eye_preds.values()], axis=0)
    point = compute_all_metrics(common_true, ens, 3)
    bs = bootstrap_all(common_true, ens, 3)
    all_results.append({'name': 'TBIL-Eyelid-Ensemble', 'task': 'TBIL-Eyelid', 'modality': 'Eyelid',
                         'nc': 3, 'n_val': len(common_true),
                         'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
                         'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
                         'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
                         'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
                         'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
                         'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi']})
    print(f'    TBIL-Eyelid-Ensemble: AUC={point["auc"]:.3f}')

# ═════════════════════════════════════════════════════════════
# G3: FACE TERNARY v3 (mild/moderate/severe)
# ═════════════════════════════════════════════════════════════
print('\n' + '█'*60)
print('  G3: Face Ternary v3 (mild/moderate/severe)')
print('█'*60)

tern_pids = {pid: tmap[p['old_cat']] for pid, p in sam_pats.items() if p['old_cat'] != 'normal'}
tr_t3, va_t3 = stratified_split(tern_pids)
va_tern = [{'faces': sam_pats[pid]['faces'], 'label': tern_pids[pid], 'pid': pid}
           for pid in va_t3 if pid in sam_pats]
print(f'  Val: {len(va_tern)} {Counter(p["label"] for p in va_tern)}')

FACE_TERN = [
    ('FaceTern-v3-ConvNeXt', 'convnext_tiny', 'v3_ternary_convnext_tiny.pt'),
    ('FaceTern-v3-ViT', 'vit_tiny_patch16_224', 'v3_ternary_vit_tiny_patch16_224.pt'),
    ('FaceTern-v3-EffNet', 'efficientnet_b0', 'v3_ternary_efficientnet_b0.pt'),
    ('FaceTern-v3-Swin', 'swin_tiny_patch4_window7_224', 'v3_ternary_swin_tiny_patch4_window7_224.pt'),
]
tern_preds = {}
for name, bb, ckpt in FACE_TERN:
    r = eval_model(name, bb, ckpt, 3, va_tern, 'faces')
    if r:
        res, probs, true = r; res['task'] = 'Face-Ternary'; res['modality'] = 'Face'
        all_results.append(res); tern_preds[name] = (probs, true)

if len(tern_preds) >= 2:
    common_true = list(tern_preds.values())[0][1]
    ens = np.mean([p for p, _ in tern_preds.values()], axis=0)
    point = compute_all_metrics(common_true, ens, 3)
    bs = bootstrap_all(common_true, ens, 3)
    all_results.append({'name': 'FaceTern-v3-Ensemble', 'task': 'Face-Ternary', 'modality': 'Face',
                         'nc': 3, 'n_val': len(common_true),
                         'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
                         'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
                         'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
                         'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
                         'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
                         'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi']})
    print(f'    FaceTern-v3-Ensemble: AUC={point["auc"]:.3f}')

# ═════════════════════════════════════════════════════════════
# G4: YOLO v3
# ═════════════════════════════════════════════════════════════
print('\n' + '█'*60)
print('  G4: YOLO v3')
print('█'*60)
from ultralytics import YOLO

# YOLO binary
yolo_bin_path = os.path.join(MODEL, 'v3_yolo_binary.pt')
if os.path.exists(yolo_bin_path):
    ym = YOLO(yolo_bin_path)
    probs, true = [], []
    for item in tqdm(va_bin, desc='YOLO-Bin', leave=False):
        faces = [f for f in item['faces'][:12] if os.path.exists(f)]
        if not faces: probs.append([0.5,0.5]); true.append(item['label']); continue
        ps = []
        for fp in faces:
            r = ym.predict(fp, verbose=False)
            if r and len(r)>0: ps.append(r[0].probs.data.cpu().numpy())
        probs.append(np.mean(ps, axis=0) if ps else [0.5,0.5]); true.append(item['label'])
    probs = np.array(probs); true = np.array(true)
    point = compute_all_metrics(true, probs, 2)
    bs = bootstrap_all(true, probs, 2)
    all_results.append({'name': 'YOLO-Binary', 'task': 'Binary-Screening', 'modality': 'Face-YOLO',
                         'nc': 2, 'n_val': len(true),
                         'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
                         'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
                         'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
                         'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
                         'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
                         'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi']})
    print(f'    YOLO-Binary: AUC={point["auc"]:.3f}')

# YOLO ternary
yolo_tern_path = os.path.join(MODEL, 'v3_yolo_ternary.pt')
if os.path.exists(yolo_tern_path):
    ym = YOLO(yolo_tern_path)
    probs, true = [], []
    for item in tqdm(va_tern, desc='YOLO-Tern', leave=False):
        faces = [f for f in item['faces'][:12] if os.path.exists(f)]
        if not faces: probs.append([1/3,1/3,1/3]); true.append(item['label']); continue
        ps = []
        for fp in faces:
            r = ym.predict(fp, verbose=False)
            if r and len(r)>0: ps.append(r[0].probs.data.cpu().numpy())
        probs.append(np.mean(ps, axis=0) if ps else [1/3,1/3,1/3]); true.append(item['label'])
    probs = np.array(probs); true = np.array(true)
    point = compute_all_metrics(true, probs, 3)
    bs = bootstrap_all(true, probs, 3)
    all_results.append({'name': 'YOLO-Ternary', 'task': 'Face-Ternary', 'modality': 'Face-YOLO',
                         'nc': 3, 'n_val': len(true),
                         'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
                         'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
                         'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
                         'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
                         'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
                         'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi']})
    print(f'    YOLO-Ternary: AUC={point["auc"]:.3f}')

# ═════════════════════════════════════════════════════════════
# G5: DBIL — from manifest
# ═════════════════════════════════════════════════════════════
print('\n' + '█'*60)
print('  G5: DBIL Ternary')
print('█'*60)
dbil_mdf = pd.read_csv(os.path.join(BASE, 'data', 'dbil_manifest.csv'))
dbil_grades = {r['patient_id']: int(r['dbil_grade']) for _, r in dbil_mdf.iterrows()}
tr_d, va_d = stratified_split(dbil_grades)
# dbil_manifest.eyelid_images is empty -> pull eyelids from MAIN manifest (2026-07-20 fix)
_main_eye = {}
for _, r in mdf.iterrows():
    if r['n_eyelid'] > 0:
        _main_eye[r['patient_id']] = json.loads(r['eyelid_images'])
va_d_face = [{'faces': json.loads(dbil_mdf[dbil_mdf['patient_id']==pid].iloc[0]['face_images']),
              'label': dbil_grades[pid], 'pid': pid} for pid in va_d if pid in dbil_grades]
va_d_eye = [{'eyelids': _main_eye[pid],
             'label': dbil_grades[pid], 'pid': pid} for pid in va_d if pid in _main_eye]
print(f'  Val face: {len(va_d_face)}, eye: {len(va_d_eye)}')

for mod, key, items in [('Eyelid', 'eyelids', va_d_eye), ('Face', 'faces', va_d_face)]:
    for name, bb in BACKBONES:
        ckpt = f'dbil_{mod.lower()}_{name}.pt'
        r = eval_model(f'DBIL-{mod}-{name}', bb, ckpt, 3, items, key)
        if r:
            res, _, _ = r; res['task'] = 'DBIL'; res['modality'] = mod
            all_results.append(res)

# ═════════════════════════════════════════════════════════════
# G6: IBIL — from manifest
# ═════════════════════════════════════════════════════════════
print('\n' + '█'*60)
print('  G6: IBIL Ternary')
print('█'*60)
ibil_mdf = pd.read_csv(os.path.join(BASE, 'data', 'ibil_manifest.csv'))
ibil_grades = {r['patient_id']: int(r['ibil_grade']) for _, r in ibil_mdf.iterrows()}
tr_i, va_i = stratified_split(ibil_grades)
va_i_face = [{'faces': json.loads(ibil_mdf[ibil_mdf['patient_id']==pid].iloc[0]['face_images']),
              'label': ibil_grades[pid], 'pid': pid} for pid in va_i if pid in ibil_grades]
va_i_eye = [{'eyelids': _main_eye[pid],
             'label': ibil_grades[pid], 'pid': pid} for pid in va_i if pid in _main_eye]
print(f'  Val face: {len(va_i_face)}, eye: {len(va_i_eye)}')

for mod, key, items in [('Eyelid', 'eyelids', va_i_eye), ('Face', 'faces', va_i_face)]:
    for name, bb in BACKBONES:
        ckpt = f'ibil_{mod.lower()}_{name}.pt'
        r = eval_model(f'IBIL-{mod}-{name}', bb, ckpt, 3, items, key)
        if r:
            res, _, _ = r; res['task'] = 'IBIL'; res['modality'] = mod
            all_results.append(res)

# ═════════════════════════════════════════════════════════════
# G7: Jaundice Type (binary hepato vs chol)
# ═════════════════════════════════════════════════════════════
print('\n' + '█'*60)
print('  G7: Jaundice Type (Hepato vs Chol)')
print('█'*60)
type_pids = {}
for pid, p in sam_pats.items():
    if p['old_cat'] == 'normal': continue
    jt = get_lab(pid, jt_col)
    if jt in (1, 2): type_pids[pid] = 0
    elif jt == 3: type_pids[pid] = 1
tr_ty, va_ty = stratified_split(type_pids)
va_type = [{'faces': sam_pats[pid]['faces'], 'label': type_pids[pid], 'pid': pid}
           for pid in va_ty if pid in sam_pats]
print(f'  Val: {len(va_type)} {Counter(p["label"] for p in va_type)}')

TYPE_MODELS = [
    ('Type-EffNet', 'efficientnet_b0', 'type_class_efficientnet_b0.pt'),
    ('Type-Swin', 'swin_tiny_patch4_window7_224', 'type_class_swin_tiny_patch4_window7_224.pt'),
    ('Type-ViT', 'vit_tiny_patch16_224', 'type_class_vit_tiny_patch16_224.pt'),
    ('Type-ConvNeXt', 'convnext_tiny', 'type_class_convnext_tiny.pt'),
]
for name, bb, ckpt in TYPE_MODELS:
    r = eval_model(name, bb, ckpt, 2, va_type, 'faces')
    if r:
        res, _, _ = r; res['task'] = 'Jaundice-Type'; res['modality'] = 'Face'
        all_results.append(res)

# ═════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ═════════════════════════════════════════════════════════════
print('\n' + '='*90)
print(f'  COMPREHENSIVE BOOTSTRAP RESULTS — ALL METRICS ({N_BOOT} iterations)')
print('='*90)

# Manuscript-format table: one row per model, tab-separated
print('\n\n=== MANUSCRIPT TABLE FORMAT (tab-separated, paste into Excel) ===\n')
header = 'Model\tF1 score\tSensitivity\tSpecificity\tAccuracy\tAUC ROC\tAverage Precision'
print(header)
print('-'*120)
for r in sorted(all_results, key=lambda x: (x.get('task',''), -x.get('auc',0))):
    row = '\t'.join([
        r['name'],
        f'{r["f1"]:.3f} [{r["f1_lo"]:.3f}-{r["f1_hi"]:.3f}]',
        f'{r["sens"]:.3f} [{r["sens_lo"]:.3f}-{r["sens_hi"]:.3f}]',
        f'{r["spec"]:.3f} [{r["spec_lo"]:.3f}-{r["spec_hi"]:.3f}]',
        f'{r["acc"]:.3f} [{r["acc_lo"]:.3f}-{r["acc_hi"]:.3f}]',
        f'{r["auc"]:.3f} [{r["auc_lo"]:.3f}-{r["auc_hi"]:.3f}]',
        f'{r["ap"]:.3f} [{r["ap_lo"]:.3f}-{r["ap_hi"]:.3f}]',
    ])
    print(row)

# Also save tab-separated file
with open(os.path.join(TBL, 'bootstrap_manuscript_table.tsv'), 'w', encoding='utf-8') as f:
    f.write('Task\t' + header + '\n')
    for r in sorted(all_results, key=lambda x: (x.get('task',''), -x.get('auc',0))):
        row = '\t'.join([
            r.get('task',''),
            r['name'],
            f'{r["f1"]:.3f} [{r["f1_lo"]:.3f}-{r["f1_hi"]:.3f}]',
            f'{r["sens"]:.3f} [{r["sens_lo"]:.3f}-{r["sens_hi"]:.3f}]',
            f'{r["spec"]:.3f} [{r["spec_lo"]:.3f}-{r["spec_hi"]:.3f}]',
            f'{r["acc"]:.3f} [{r["acc_lo"]:.3f}-{r["acc_hi"]:.3f}]',
            f'{r["auc"]:.3f} [{r["auc_lo"]:.3f}-{r["auc_hi"]:.3f}]',
            f'{r["ap"]:.3f} [{r["ap_lo"]:.3f}-{r["ap_hi"]:.3f}]',
        ])
        f.write(row + '\n')

rdf = pd.DataFrame(all_results)
rdf.to_csv(os.path.join(TBL, 'bootstrap_full_metrics.csv'), index=False, encoding='utf-8-sig')
with open(os.path.join(RES, 'bootstrap_full_metrics.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# DOCX
doc = Document()
doc.add_heading('Comprehensive Bootstrap Results — All Models × All Metrics', level=2)
doc.add_paragraph(f'{N_BOOT} bootstrap iterations, patient-level resampling. '
                   f'95% CI = [2.5th, 97.5th] percentile. '
                   f'Sensitivity/Specificity: per-class macro-averaged for ternary, positive class for binary.')
for task in sorted(set(r.get('task','') for r in all_results)):
    sub = sorted([r for r in all_results if r.get('task') == task], key=lambda x: -x.get('auc',0))
    doc.add_heading(task, level=3)
    cols = ['Model', 'F1 [CI]', 'Sens [CI]', 'Spec [CI]', 'Acc [CI]', 'AUC [CI]', 'AP [CI]']
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = 'Light Grid Accent 1'
    for i, c in enumerate(cols):
        t.rows[0].cells[i].text = c
        for p in t.rows[0].cells[i].paragraphs:
            for run in p.runs: run.bold = True; run.font.size = Pt(7)
    for r in sub:
        cells = t.add_row().cells
        vals = [r['name'],
                f'{r["f1"]:.3f} [{r["f1_lo"]:.3f}-{r["f1_hi"]:.3f}]',
                f'{r["sens"]:.3f} [{r["sens_lo"]:.3f}-{r["sens_hi"]:.3f}]',
                f'{r["spec"]:.3f} [{r["spec_lo"]:.3f}-{r["spec_hi"]:.3f}]',
                f'{r["acc"]:.3f} [{r["acc_lo"]:.3f}-{r["acc_hi"]:.3f}]',
                f'{r["auc"]:.3f} [{r["auc_lo"]:.3f}-{r["auc_hi"]:.3f}]',
                f'{r["ap"]:.3f} [{r["ap_lo"]:.3f}-{r["ap_hi"]:.3f}]']
        for i, v in enumerate(vals):
            cells[i].text = str(v)
            for p in cells[i].paragraphs:
                for run in p.runs: run.font.size = Pt(7)
doc.save(os.path.join(TBL, 'Table_bootstrap_full_metrics.docx'))

print(f'\n  CSV:  {TBL}/bootstrap_full_metrics.csv')
print(f'  DOCX: {TBL}/Table_bootstrap_full_metrics.docx')
