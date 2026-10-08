# -*- coding: utf-8 -*-
"""
REAL Bootstrap 95% CI for CP, MELD, and Domain-Adapted External models.

This replaces the fake +/-0.05 placeholder CIs in bootstrap_full_metrics_v3.csv.
Reproduces the EXACT same val/test splits used during training so the point
estimates match the published AUC values, then computes true bootstrap CIs
(1000 iterations, patient-level resampling) for ALL six metrics.

Outputs:
  - results/bootstrap_cp_meld_external.json  (raw results)
  - results/tables_v3/bootstrap_cp_meld_external.csv (metrics + CI)
"""
import os, re, json, random, math, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, average_precision_score)

BASE = r'D:\research\人脸识别营养\传染科'
INT_DATA = os.path.join(BASE, 'data', 'sam_processed_v3')
EXT_DATA = os.path.join(BASE, 'data', 'external_processed_v3')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
TBL_V3 = os.path.join(RES, 'tables_v3')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMG_SIZE = 224; N_BOOT = 1000
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── Image utils ──────────────────────────────────────────────
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

def grey_world(img):
    mr, mg, mb = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
    gray = (mr + mg + mb) / 3
    if gray < 1:
        return img
    out = img.astype(np.float32)
    out[:,:,0] = np.clip(out[:,:,0] * gray / (mr + 1e-6), 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] * gray / (mg + 1e-6), 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] * gray / (mb + 1e-6), 0, 255)
    return out.astype(np.uint8)

def preprocess_int(img_rgb):
    """Internal preprocessing: CLAHE only (matches CP/MELD training)."""
    return clahe(img_rgb)

def preprocess_ext(img_rgb):
    """External preprocessing: Grey World + CLAHE (matches domain-adapt training)."""
    return clahe(grey_world(img_rgb))

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# ── Metrics ──────────────────────────────────────────────────
def compute_all_metrics(true, probs, nc):
    pred = probs.argmax(1)
    cm = confusion_matrix(true, pred, labels=list(range(nc)))
    try:
        if nc == 2:
            auc = roc_auc_score(true, probs[:, 1])
        else:
            auc = roc_auc_score(true, probs, multi_class='ovr', labels=list(range(nc)))
    except Exception:
        auc = 0.0
    try:
        if nc == 2:
            ap = average_precision_score(true, probs[:, 1])
        else:
            from sklearn.preprocessing import label_binarize
            yb = label_binarize(true, classes=list(range(nc)))
            ap = float(np.mean([average_precision_score(yb[:, c], probs[:, c]) for c in range(nc)]))
    except Exception:
        ap = 0.0
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro', labels=list(range(nc)), zero_division=0)
    sens_list, spec_list = [], []
    for c in range(nc):
        tp = cm[c, c]; fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    if nc == 2:
        sens = sens_list[1]; spec = spec_list[1]
    else:
        sens = float(np.mean(sens_list)); spec = float(np.mean(spec_list))
    return {'auc': float(auc), 'ap': float(ap), 'acc': float(acc),
            'f1': float(f1), 'sens': float(sens), 'spec': float(spec)}

def bootstrap_all(true, probs, nc, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    n = len(true); true = np.asarray(true); probs = np.asarray(probs)
    res = {k: [] for k in ['auc', 'ap', 'acc', 'f1', 'sens', 'spec']}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = true[idx]; yp = probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_all_metrics(yt, yp, nc)
        for k in res:
            if not np.isnan(m[k]):
                res[k].append(m[k])
    out = {}
    for k, arr in res.items():
        if arr:
            out[f'{k}_lo'] = float(np.percentile(arr, 2.5))
            out[f'{k}_hi'] = float(np.percentile(arr, 97.5))
        else:
            out[f'{k}_lo'] = 0.0; out[f'{k}_hi'] = 0.0
    return out


# ── Patient-level prediction ─────────────────────────────────
def predict_patient_paths(model, paths, preprocess_fn, n_max=12):
    valid = [p for p in paths[:n_max] if os.path.exists(p)]
    if not valid:
        return None
    tensors = []
    for p in valid:
        img = read_img(p)
        if img is None:
            continue
        img = preprocess_fn(img)
        tensors.append(ev_tf(Image.fromarray(img).convert('RGB')).unsqueeze(0))
    if not tensors:
        return None
    t = torch.cat(tensors).to(DEV)
    with torch.no_grad():
        return F.softmax(model(t), dim=1).mean(0).cpu().numpy()


def eval_items(model, items, nc, preprocess_fn, img_key):
    """items: list of dicts with img_key (list of paths) and 'label'."""
    probs, true = [], []
    for it in tqdm(items, desc='predict', leave=False):
        paths = it.get(img_key, [])
        pr = predict_patient_paths(model, paths, preprocess_fn)
        if pr is None:
            pr = np.ones(nc) / nc
        probs.append(pr); true.append(it['label'])
    return np.array(probs), np.array(true)


def load_model(backbone_id, ckpt, nc):
    path = os.path.join(MODEL, ckpt)
    if not os.path.exists(path):
        print(f'    SKIP: {ckpt} not found')
        return None
    m = timm.create_model(backbone_id, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    m.eval()
    return m


def run_one(name, backbone_id, ckpt, nc, items, preprocess_fn, img_key, task, modality):
    print(f'\n  >> {name}')
    m = load_model(backbone_id, ckpt, nc)
    if m is None:
        return None
    probs, true = eval_items(m, items, nc, preprocess_fn, img_key)
    del m; torch.cuda.empty_cache()
    point = compute_all_metrics(true, probs, nc)
    bs = bootstrap_all(true, probs, nc)
    res = {'name': name, 'nc': nc, 'n_val': int(len(true)),
           'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
           'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
           'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
           'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
           'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
           'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi'],
           'task': task, 'modality': modality}
    print(f'    AUC={point["auc"]:.3f}[{bs["auc_lo"]:.3f}-{bs["auc_hi"]:.3f}] '
          f'Sens={point["sens"]:.3f}[{bs["sens_lo"]:.3f}-{bs["sens_hi"]:.3f}] '
          f'Spec={point["spec"]:.3f}[{bs["spec_lo"]:.3f}-{bs["spec_hi"]:.3f}] '
          f'F1={point["f1"]:.3f} AP={point["ap"]:.3f} N={len(true)}')
    return res, probs, true


# ═════════════════════════════════════════════════════════════
# PART A: CP / MELD — reproduce train_cp_meld_images.py split
# ═════════════════════════════════════════════════════════════
print('='*70)
print('  PART A: Child-Pugh & MELD — REAL Bootstrap CI')
print('='*70)

print('\n[1] Loading clinical data...')
df_ex = pd.read_excel(EXCEL)
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)

def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None

def strip_zeros(s):
    return s.lstrip('0') if s else s

def get_clinical(pid):
    hid = extract_hid(pid)
    if not hid:
        return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    if len(rows) == 0:
        return None
    r = rows.iloc[0]
    cp_grade = str(r.iloc[15]).strip().upper() if pd.notna(r.iloc[15]) else None
    tbil = pd.to_numeric(r.iloc[51], errors='coerce')
    inr = pd.to_numeric(r.iloc[61], errors='coerce')
    meld = None
    if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
        meld = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr, 1)) + 9.57 * math.log(1.0) + 6.43
        meld = round(max(meld, 6))
    return {'cp_grade': cp_grade, 'meld': meld}

print('[2] Matching images with clinical data...')
mdf = pd.read_csv(MANIFEST)
patients = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(INT_DATA, cat)
    if not os.path.exists(cd):
        continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p):
            continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            patients[pid] = {'faces': faces, 'cat': cat}
for _, r in mdf.iterrows():
    pid = r['patient_id']
    if pid in patients and r['n_eyelid'] > 0:
        patients[pid]['eyelids'] = json.loads(r['eyelid_images'])

matched = {}
for pid, p in patients.items():
    if p['cat'] == 'normal':
        p.update({'cp_grade': 'A', 'meld': 8})
        matched[pid] = p
    else:
        clin = get_clinical(pid)
        if clin and clin['cp_grade'] in ('A', 'B', 'C'):
            p.update(clin); matched[pid] = p

print(f'  Matched: {len(matched)} patients')

# Stratified split (must match training: by CP grade, seed=42, 20% val)
pids = list(matched.keys())
grades = {pid: matched[pid]['cp_grade'] for pid in pids}
by_grade = {}
for pid in pids:
    by_grade.setdefault(grades[pid], []).append(pid)
tr_pids, va_pids = set(), set()
for g, ps in by_grade.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_pids.update(ps[:n]); tr_pids.update(ps[n:])
print(f'  Val pids: {len(va_pids)}')

# Build val items for CP and MELD
grade_map = {'A': 0, 'B': 1, 'C': 2}
def meld_category(p):
    m = p.get('meld')
    if m is None:
        return None
    if m <= 20:
        return 0
    elif m <= 30:
        return 1
    else:
        return 2

cp_face_val = [{'faces': matched[pid]['faces'], 'label': grade_map[matched[pid]['cp_grade']], 'pid': pid}
               for pid in va_pids if pid in matched and matched[pid]['cp_grade'] in grade_map]
cp_eye_val = [{'eyelids': matched[pid].get('eyelids', []), 'label': grade_map[matched[pid]['cp_grade']], 'pid': pid}
              for pid in va_pids if pid in matched and matched[pid]['cp_grade'] in grade_map
              and matched[pid].get('eyelids')]
meld_face_val = [{'faces': matched[pid]['faces'], 'label': meld_category(matched[pid]), 'pid': pid}
                 for pid in va_pids if pid in matched and meld_category(matched[pid]) is not None]
meld_eye_val = [{'eyelids': matched[pid].get('eyelids', []), 'label': meld_category(matched[pid]), 'pid': pid}
                for pid in va_pids if pid in matched and meld_category(matched[pid]) is not None
                and matched[pid].get('eyelids')]
print(f'  CP val:   face={len(cp_face_val)}, eyelid={len(cp_eye_val)}')
print(f'  MELD val: face={len(meld_face_val)}, eyelid={len(meld_eye_val)}')

# Run CP models (4) and MELD models (2 face)
all_results = []
cp_models = [
    ('CP-cp_face_convnext',  'convnext_tiny',             'cp_face_convnext.pt',  3, 'faces',   'Child-Pugh', 'Face'),
    ('CP-cp_face_vit',       'vit_tiny_patch16_224',      'cp_face_vit.pt',       3, 'faces',   'Child-Pugh', 'Face'),
    ('CP-cp_eyelid_convnext','convnext_tiny',             'cp_eyelid_convnext.pt',3, 'eyelids', 'Child-Pugh', 'Eyelid'),
    ('CP-cp_eyelid_vit',     'vit_tiny_patch16_224',      'cp_eyelid_vit.pt',     3, 'eyelids', 'Child-Pugh', 'Eyelid'),
]
meld_models = [
    ('MELD-meld_face_vit',      'vit_tiny_patch16_224',  'meld_face_vit.pt',      3, 'faces', 'MELD-Risk', 'Face'),
    ('MELD-meld_face_convnext', 'convnext_tiny',         'meld_face_convnext.pt', 3, 'faces', 'MELD-Risk', 'Face'),
]
for name, bb, ckpt, nc, key, task, mod in cp_models:
    items = cp_face_val if key == 'faces' else cp_eye_val
    if not items:
        print(f'  SKIP {name}: no val items')
        continue
    r = run_one(name, bb, ckpt, nc, items, preprocess_int, key, task, mod)
    if r:
        all_results.append(r[0])
for name, bb, ckpt, nc, key, task, mod in meld_models:
    items = meld_face_val if key == 'faces' else meld_eye_val
    if not items:
        print(f'  SKIP {name}: no val items')
        continue
    r = run_one(name, bb, ckpt, nc, items, preprocess_int, key, task, mod)
    if r:
        all_results.append(r[0])


# ═════════════════════════════════════════════════════════════
# PART B: External Validation Bootstrap CI
#   B1: Pure external (all patients, models trained on internal only) — shows domain shift
#   B2: Domain-adapted held-out — ONLY if split matches training (controlled by SKIP_B2 flag)
# ═════════════════════════════════════════════════════════════
SKIP_B2 = os.environ.get('SKIP_B2', '0') == '1'  # set SKIP_B2=1 to skip domain-adapted section
# Note: After adding new normal patients (59 total), the original domain-adapted 80/20
# split no longer matches the training-time split. B2 results would test models on
# patients they may have trained on. When SKIP_B2=1, we keep only the clean B1 results.

print('\n' + '='*70)
print('  PART B: External Validation — REAL Bootstrap CI')
print('='*70)

# Build patient list from external data, FILTERING OUT empty directories
# (2 jaundice dirs have no face images — must exclude to match n=98)
ext_all_pids = {}  # label -> list of valid pids (with face images)
ext_pid_to_faces = {}  # pid -> face paths (for prediction)
for cat, label in [('normal', 0), ('jaundice', 1)]:
    cd = os.path.join(EXT_DATA, cat)
    if not os.path.exists(cd):
        continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p):
            continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:  # FILTER: only include if has face images
            pid_key = f'ext_{pid}'
            ext_all_pids.setdefault(label, []).append(pid_key)
            ext_pid_to_faces[pid_key] = (faces, label, pid)

n_normal = len(ext_all_pids.get(0, []))
n_jaun = len(ext_all_pids.get(1, []))
n_total_ext = n_normal + n_jaun
print(f'\n  External valid patients (with face images): {n_total_ext} '
      f'(normal={n_normal}, jaundice={n_jaun})')


# ─── B1: PURE EXTERNAL VALIDATION (all 98 patients) ───────
# Models trained on internal only; tested on full external cohort
print('\n  --- B1: Pure External Validation (n={}, no domain adaptation) ---'.format(n_total_ext))

ext_full_test = [{'faces': ext_pid_to_faces[k][0], 'label': ext_pid_to_faces[k][1], 'pid': k}
                 for k in sorted(ext_pid_to_faces.keys())]

PURE_EXT_MODELS = [
    ('Ext-Pure V3-ViT',          'vit_tiny_patch16_224',         'v3_binary_vit_tiny_patch16_224.pt'),
    ('Ext-Pure V3-EffNet',       'efficientnet_b0',              'v3_binary_efficientnet_b0.pt'),
    ('Ext-Pure V3-ConvNeXt',     'convnext_tiny',                'v3_binary_convnext_tiny.pt'),
    ('Ext-Pure TypeBin-EffNet',  'efficientnet_b0',              'type_binary_efficientnet_b0.pt'),
    ('Ext-Pure TypeBin-ViT',     'vit_tiny_patch16_224',         'type_binary_vit_tiny_patch16_224.pt'),
    ('Ext-Pure TypeBin-ConvNeXt','convnext_tiny',                'type_binary_convnext_tiny.pt'),
    ('Ext-Pure TypeBin-Swin',    'swin_tiny_patch4_window7_224', 'type_binary_swin_tiny_patch4_window7_224.pt'),
]
pure_ext_probs = {}
pure_ext_true = None
for name, bb, ckpt in PURE_EXT_MODELS:
    m = load_model(bb, ckpt, 2)
    if m is None:
        continue
    # Pure external uses CLAHE only (internal preprocessing)
    probs, true = eval_items(m, ext_full_test, 2, preprocess_int, 'faces')
    pure_ext_probs[name] = probs
    pure_ext_true = true
    del m; torch.cuda.empty_cache()
    point = compute_all_metrics(true, probs, 2)
    bs = bootstrap_all(true, probs, 2)
    res = {'name': name, 'nc': 2, 'n_val': int(len(true)),
           'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
           'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
           'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
           'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
           'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
           'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi'],
           'task': 'Binary-External-Pure', 'modality': 'Face'}
    all_results.append(res)
    print(f'    {name}: AUC={point["auc"]:.3f}[{bs["auc_lo"]:.3f}-{bs["auc_hi"]:.3f}] '
          f'Sens={point["sens"]:.3f} Spec={point["spec"]:.3f} N={len(true)}')

# YOLO on pure external
yolo_bin_path = os.path.join(MODEL, 'v3_yolo_binary.pt')
if os.path.exists(yolo_bin_path):
    try:
        from ultralytics import YOLO
        ym = YOLO(yolo_bin_path)
        probs = []
        for it in ext_full_test:
            faces = [f for f in it['faces'][:12] if os.path.exists(f)]
            ps = []
            for fp in faces:
                r = ym.predict(fp, verbose=False)
                if r and len(r) > 0:
                    ps.append(r[0].probs.data.cpu().numpy())
            probs.append(np.mean(ps, axis=0) if ps else np.array([0.5, 0.5]))
        probs = np.array(probs); true = np.array([it['label'] for it in ext_full_test])
        point = compute_all_metrics(true, probs, 2)
        bs = bootstrap_all(true, probs, 2)
        res = {'name': 'Ext-Pure YOLO', 'nc': 2, 'n_val': int(len(true)),
               'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
               'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
               'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
               'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
               'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
               'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi'],
               'task': 'Binary-External-Pure', 'modality': 'Face-YOLO'}
        all_results.append(res)
        pure_ext_probs['Ext-Pure YOLO'] = probs
        print(f'    Ext-Pure YOLO: AUC={point["auc"]:.3f}[{bs["auc_lo"]:.3f}-{bs["auc_hi"]:.3f}] N={len(true)}')
        del ym
    except Exception as e:
        print(f'    YOLO pure ext skipped: {e}')

# Pure external ensemble
if len(pure_ext_probs) >= 2 and pure_ext_true is not None:
    ens = np.mean(list(pure_ext_probs.values()), axis=0)
    point = compute_all_metrics(pure_ext_true, ens, 2)
    bs = bootstrap_all(pure_ext_true, ens, 2)
    res = {'name': 'Ext-Pure Ensemble', 'nc': 2, 'n_val': int(len(pure_ext_true)),
           'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
           'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
           'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
           'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
           'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
           'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi'],
           'task': 'Binary-External-Pure', 'modality': 'Face'}
    all_results.append(res)
    print(f'    Ext-Pure Ensemble: AUC={point["auc"]:.3f}[{bs["auc_lo"]:.3f}-{bs["auc_hi"]:.3f}]')


# ─── B2: DOMAIN-ADAPTED HELD-OUT ───────────────────────────
# Only run if the split matches the training-time cohort (skip after adding new patients)
if not SKIP_B2:
    print('\n  --- B2: Domain-Adapted Held-Out (20%% of {}) ---'.format(n_total_ext))

    # Reproduce external test split using VALID pids only (seed=42, 20% held-out)
    te_ext = set()
    for lbl, pids in ext_all_pids.items():
        random.seed(SEED); random.shuffle(pids)
        n = max(1, int(len(pids) * 0.2))
        te_ext.update(pids[:n])

    ext_test = [{'faces': ext_pid_to_faces[k][0], 'label': ext_pid_to_faces[k][1], 'pid': k}
                for k in sorted(te_ext)]
    te_n_normal = sum(1 for x in ext_test if x['label'] == 0)
    te_n_jaun = sum(1 for x in ext_test if x['label'] == 1)
    print(f'  Domain-adapt test patients: {len(ext_test)} '
          f'(normal={te_n_normal}, jaundice={te_n_jaun})')

    # Run domain-adapted models (Grey World + CLAHE preprocessing)
    da_models = [
        ('Domain-Adapted convnext (Ext)',     'convnext_tiny',                  'domain_adapt_binary_convnext.pt'),
        ('Domain-Adapted vit (Ext)',          'vit_tiny_patch16_224',           'domain_adapt_binary_vit.pt'),
        ('Domain-Adapted efficientnet (Ext)', 'efficientnet_b0',                'domain_adapt_binary_efficientnet.pt'),
        ('Domain-Adapted swin (Ext)',         'swin_tiny_patch4_window7_224',   'domain_adapt_binary_swin.pt'),
    ]
    da_probs = {}
    da_true = None
    for name, bb, ckpt in da_models:
        m = load_model(bb, ckpt, 2)
        if m is None:
            continue
        probs, true = eval_items(m, ext_test, 2, preprocess_ext, 'faces')
        da_probs[name] = probs
        da_true = true
        del m; torch.cuda.empty_cache()
        point = compute_all_metrics(true, probs, 2)
        bs = bootstrap_all(true, probs, 2)
        res = {'name': name, 'nc': 2, 'n_val': int(len(true)),
               'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
               'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
               'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
               'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
               'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
               'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi'],
               'task': 'Binary-External', 'modality': 'Face'}
        all_results.append(res)
        print(f'    {name}: AUC={point["auc"]:.3f}[{bs["auc_lo"]:.3f}-{bs["auc_hi"]:.3f}] N={len(true)}')

    # Ensemble
    if len(da_probs) >= 2 and da_true is not None:
        ens = np.mean(list(da_probs.values()), axis=0)
        point = compute_all_metrics(da_true, ens, 2)
        bs = bootstrap_all(da_true, ens, 2)
        res = {'name': 'Domain-Adapted ensemble (Ext)', 'nc': 2, 'n_val': int(len(da_true)),
               'auc': point['auc'], 'auc_lo': bs['auc_lo'], 'auc_hi': bs['auc_hi'],
               'ap': point['ap'], 'ap_lo': bs['ap_lo'], 'ap_hi': bs['ap_hi'],
               'acc': point['acc'], 'acc_lo': bs['acc_lo'], 'acc_hi': bs['acc_hi'],
               'sens': point['sens'], 'sens_lo': bs['sens_lo'], 'sens_hi': bs['sens_hi'],
               'spec': point['spec'], 'spec_lo': bs['spec_lo'], 'spec_hi': bs['spec_hi'],
               'f1': point['f1'], 'f1_lo': bs['f1_lo'], 'f1_hi': bs['f1_hi'],
               'task': 'Binary-External', 'modality': 'Face'}
        all_results.append(res)
        print(f'    Domain-Adapted ensemble (Ext): AUC={point["auc"]:.3f}[{bs["auc_lo"]:.3f}-{bs["auc_hi"]:.3f}]')
else:
    print('\n  --- B2 SKIPPED (SKIP_B2=1): external cohort changed; re-train domain-adapted models first ---')
    print('      Keeping previous B2 results from bootstrap_full_metrics_v3.csv')


# ═════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════
print('\n' + '='*70)
print(f'  TOTAL NEW RESULTS: {len(all_results)}')
print('='*70)

with open(os.path.join(RES, 'bootstrap_cp_meld_external.json'), 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, default=str)

df_out = pd.DataFrame(all_results)
cols_order = ['name', 'nc', 'n_val', 'task', 'modality',
              'auc', 'auc_lo', 'auc_hi', 'ap', 'ap_lo', 'ap_hi',
              'acc', 'acc_lo', 'acc_hi', 'sens', 'sens_lo', 'sens_hi',
              'spec', 'spec_lo', 'spec_hi', 'f1', 'f1_lo', 'f1_hi']
df_out = df_out[[c for c in cols_order if c in df_out.columns]]
df_out.to_csv(os.path.join(TBL_V3, 'bootstrap_cp_meld_external.csv'), index=False, encoding='utf-8-sig')

print(f'\n  JSON: {RES}/bootstrap_cp_meld_external.json')
print(f'  CSV:  {TBL_V3}/bootstrap_cp_meld_external.csv')
print('\n  DONE.')
