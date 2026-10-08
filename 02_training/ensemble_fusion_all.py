# -*- coding: utf-8 -*-
"""
Unified Ensemble + Fusion evaluation engine for ALL 7 BilinGuard tasks.

For each task, rebuilds the CANONICAL val split (the one behind the reported AUCs),
evaluates every available architecture per modality (face / eyelid), then computes:
  - Ensemble (scope): unweighted mean of available arch patient-level probs
  - Fusion (arch):    0.5*face + 0.5*eyelid probs on patients present in BOTH val sets
  - Fusion Ensemble:  mean of all available modality-arch probs on the intersection

Every reported AUC is re-computed for verification (printed side by side).
Results: results/ensemble_fusion_results.json (+ per-model prob cache)
"""
import os, re, json, random, math, numpy as np, pandas as pd, cv2
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import Counter

BASE = r'D:\research\人脸识别营养\传染科'
DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG_SIZE = 224; BS = 32

BB = {'convnext': 'convnext_tiny', 'vit': 'vit_tiny_patch16_224',
      'efficientnet': 'efficientnet_b0', 'swin': 'swin_tiny_patch4_window7_224'}


# ── preprocessing ────────────────────────────────────────────
def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ImgDS(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid


@torch.no_grad()
def model_patient_probs(ckpt, backbone, nc, items):
    """Return {pid: (prob_vec, label)} patient-level mean-softmax."""
    path = os.path.join(MODEL, ckpt)
    if not os.path.exists(path):
        print(f'      MISSING {ckpt}')
        return None
    try:
        m = timm.create_model(backbone, pretrained=False, num_classes=nc).to(DEV)
        m.load_state_dict(torch.load(path, map_location=DEV))
        m.eval()
    except Exception as e:
        print(f'      LOAD-FAIL {ckpt}: {e}')
        return None
    dl = DataLoader(ImgDS(items), BS, shuffle=False, num_workers=0)
    d = {}
    for im, la, pi in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        for p, l, pid in zip(o, la.numpy(), pi):
            d.setdefault(pid, ([], int(l)))
            d[pid][0].append(p)
    del m; torch.cuda.empty_cache()
    return {pid: (np.mean(v[0], 0), v[1]) for pid, v in d.items()}


def auc_of(prob_dict, nc, pids=None):
    pids = pids or list(prob_dict.keys())
    avg = np.array([prob_dict[p][0] for p in pids])
    true = np.array([prob_dict[p][1] for p in pids])
    if len(np.unique(true)) < 2:
        return float('nan'), 0.0, len(pids)
    try:
        auc = roc_auc_score(true, avg[:, 1]) if nc == 2 else \
            roc_auc_score(true, avg, multi_class='ovr', labels=list(range(nc)))
    except Exception:
        auc = float('nan')
    acc = accuracy_score(true, avg.argmax(1))
    return auc, acc, len(pids)


def merge_probs(dicts, pids):
    """Mean of prob vectors across dicts for each pid present in ALL dicts."""
    out = {}
    for pid in pids:
        if all(pid in d for d in dicts):
            out[pid] = (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1])
    return out


# ── shared data ──────────────────────────────────────────────
print('[init] loading excel + manifest + sam patients...')
df_ex = pd.read_excel(EXCEL)
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)
JT_COL = [c for c in df_ex.columns if '黄疸类型' in str(c)][0]
mdf = pd.read_csv(MANIFEST)

def strip_zeros(s): return s.lstrip('0') if s else s
def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None

def excel_row(pid):
    hid = extract_hid(pid)
    if not hid: return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    return rows.iloc[0] if len(rows) else None

def collect_sam_patients():
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
    for _, r in mdf.iterrows():
        if r['patient_id'] in patients and r['n_eyelid'] > 0:
            patients[r['patient_id']]['eyelids'] = json.loads(r['eyelid_images'])
    return patients

sam_pats = collect_sam_patients()

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

def items_from(pids, patients, key, label_map, n_img):
    out = []
    for pid in pids:
        if pid not in patients: continue
        lab = label_map(pid) if callable(label_map) else label_map[pid]
        if lab is None: continue
        for ip in patients[pid].get(key, [])[:n_img]:
            if os.path.exists(ip):
                out.append((ip, lab, pid))
    return out


RESULTS = {}   # task -> scope/arch -> metrics
PROBS = {}     # task -> scope -> arch -> {pid:(prob,label)}


def run_task(task, nc, splits, models, reported):
    """splits: {'face': items, 'eyelid': items}; models: {'face': {arch: ckpt}, 'eyelid': {...}}"""
    print('\n' + '=' * 66)
    print(f'  TASK: {task} (nc={nc})')
    print('=' * 66)
    RESULTS[task] = {}
    PROBS[task] = {}
    for scope in ['face', 'eyelid']:
        items = splits.get(scope)
        if not items:
            continue
        n_pat = len(set(p for _, _, p in items))
        print(f'  [{scope}] val items={len(items)} patients={n_pat} {dict(Counter(l for _, l, _ in items))}')
        PROBS[task][scope] = {}
        for arch, ckpt in models.get(scope, {}).items():
            pd_ = model_patient_probs(ckpt, BB[arch], nc, items)
            if pd_ is None: continue
            PROBS[task][scope][arch] = pd_
            auc, acc, n = auc_of(pd_, nc)
            rep = reported.get(scope, {}).get(arch)
            flag = ''
            if rep is not None:
                diff = auc - rep
                flag = f' reported={rep:.4f} diff={diff:+.4f}' + (' OK' if abs(diff) < 0.02 else ' MISMATCH')
            print(f'    {scope:<7} {arch:<13} AUC={auc:.4f} ACC={acc:.4f} n={n}{flag}')
            RESULTS[task][f'{scope}_{arch}'] = {'auc': auc, 'acc': acc, 'n': n, 'reported': rep}
        # ensemble within scope
        archs = list(PROBS[task][scope].keys())
        if len(archs) >= 2:
            common = set.intersection(*[set(PROBS[task][scope][a].keys()) for a in archs])
            ens = merge_probs([PROBS[task][scope][a] for a in archs], sorted(common))
            auc, acc, n = auc_of(ens, nc)
            rep = reported.get(scope, {}).get('ensemble')
            flag = f' reported={rep:.4f}' if rep is not None else ''
            print(f'    {scope:<7} ENSEMBLE({len(archs)}) AUC={auc:.4f} ACC={acc:.4f} n={n}{flag}')
            RESULTS[task][f'{scope}_ensemble'] = {'auc': auc, 'acc': acc, 'n': n, 'archs': archs, 'reported': rep}
            PROBS[task][scope]['__ensemble__'] = ens

    # fusion
    if 'face' in PROBS[task] and 'eyelid' in PROBS[task]:
        fa = {a: d for a, d in PROBS[task]['face'].items() if not a.startswith('__')}
        ea = {a: d for a, d in PROBS[task]['eyelid'].items() if not a.startswith('__')}
        fpids = set().union(*[set(d.keys()) for d in fa.values()]) if fa else set()
        epids = set().union(*[set(d.keys()) for d in ea.values()]) if ea else set()
        both = sorted(fpids & epids)
        print(f'  [fusion] patients with both modalities in val: {len(both)}')
        for arch in sorted(set(fa) & set(ea)):
            fus = merge_probs([fa[arch], ea[arch]], both)
            if not fus: continue
            auc, acc, n = auc_of(fus, nc)
            print(f'    fusion  {arch:<13} AUC={auc:.4f} ACC={acc:.4f} n={n}')
            RESULTS[task][f'fusion_{arch}'] = {'auc': auc, 'acc': acc, 'n': n}
        dicts = [d for d in list(fa.values()) + list(ea.values())]
        fus_all = merge_probs(dicts, both)
        if fus_all:
            auc, acc, n = auc_of(fus_all, nc)
            rep = reported.get('fusion', {}).get('ensemble')
            flag = f' reported={rep:.4f}' if rep is not None else ''
            print(f'    fusion  ENSEMBLE({len(dicts)}) AUC={auc:.4f} ACC={acc:.4f} n={n}{flag}')
            RESULTS[task]['fusion_ensemble'] = {'auc': auc, 'acc': acc, 'n': n, 'reported': rep}


# ═════════════════════════════════════════════════════════════
# TASK 1: Binary Screening
# ═════════════════════════════════════════════════════════════
# Face val: train_binary_final.py split (face_v4 patients, per-label seed42)
bin_pids = {pid: (0 if p['old_cat'] == 'normal' else 1) for pid, p in sam_pats.items()}
by_label = {}
for pid, l in bin_pids.items(): by_label.setdefault(l, []).append(pid)
tr_b, va_b = set(), set()
for g, pids in by_label.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_b.update(pids[:n]); tr_b.update(pids[n:])
bin_face_items = items_from(va_b, sam_pats, 'faces', bin_pids, 4)

# Eyelid val: reconstructed split B (217 manifest + 156 extracted/normal scan)
def build_bin_eyelid():
    items = []
    df = mdf[mdf['n_eyelid'] > 0]
    for _, r in df.iterrows():
        imgs = json.loads(r['eyelid_images'])
        for pth in imgs[:1]:
            items.append((pth, 0 if r['category'] == 'normal' else 1, r['patient_id']))
    normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
    for pid in sorted(os.listdir(normal_ext)):
        pp = os.path.join(normal_ext, pid)
        if not os.path.isdir(pp): continue
        cand = None
        for root, dirs, files in os.walk(pp):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                    fp = os.path.join(root, f)
                    if os.path.getsize(fp) > 1000000:
                        cand = fp; break
            if cand: break
        if cand:
            items.append((cand, 0, pid))
    by_l = {}
    for it in items: by_l.setdefault(it[1], []).append(it)
    va = []
    random.seed(SEED)
    for label, its in by_l.items():
        its = list(its); random.shuffle(its)
        n = max(1, int(len(its) * 0.2))
        va.extend(its[:n])
    return va

bin_eye_items = build_bin_eyelid()

run_task('Binary Screening', 2,
         {'face': bin_face_items, 'eyelid': bin_eye_items},
         {'face': {'convnext': 'final_binary_face_convnext.pt', 'vit': 'final_binary_face_vit.pt',
                   'efficientnet': 'final_binary_face_efficientnet.pt', 'swin': 'final_binary_face_swin.pt'},
          'eyelid': {'convnext': 'eyelid_binary_convnext.pt', 'vit': 'eyelid_binary_vit_tiny_patch16_224.pt',
                     'efficientnet': 'eyelid_binary_efficientnet_b0.pt',
                     'swin': 'eyelid_binary_swin_tiny_patch4_window7_224.pt'}},
         {'face': {'convnext': 0.9523, 'vit': 0.9548, 'swin': 0.9618, 'efficientnet': 0.9452, 'ensemble': 0.9622},
          'eyelid': {'convnext': 0.5257, 'vit': 0.7301, 'swin': 0.7342, 'efficientnet': 0.7060},
          'fusion': {'ensemble': 0.9862}})

# ═════════════════════════════════════════════════════════════
# TASK 2: Ternary Grading
# ═════════════════════════════════════════════════════════════
tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
# Face: bootstrap G3 split
tern_pids = {pid: tmap[p['old_cat']] for pid, p in sam_pats.items() if p['old_cat'] != 'normal'}
_, va_t3 = stratified_split(tern_pids)
tern_face_items = items_from(va_t3, sam_pats, 'faces', tern_pids, 4)
# Eyelid: bootstrap G2 split
j_df = mdf[mdf['category'].isin(['mild', 'moderate', 'severe']) & (mdf['n_eyelid'] > 0)]
tbil_val = [{'eyelids': json.loads(r['eyelid_images']), 'label': tmap[r['category']], 'pid': r['patient_id']}
            for _, r in j_df.iterrows()]
random.seed(SEED)
by_l = {}
for p in tbil_val: by_l.setdefault(p['label'], []).append(p)
va_teye = []
for l, g in by_l.items():
    random.shuffle(g); va_teye.extend(g[:max(1, int(len(g) * 0.2))])
tern_eye_items = []
for p in va_teye:
    for ip in p['eyelids'][:12]:
        if os.path.exists(ip):
            tern_eye_items.append((ip, p['label'], p['pid']))

run_task('Ternary Grading', 3,
         {'face': tern_face_items, 'eyelid': tern_eye_items},
         {'face': {'convnext': 'v3_ternary_convnext_tiny.pt', 'vit': 'v3_ternary_vit_tiny_patch16_224.pt',
                   'efficientnet': 'v3_ternary_efficientnet_b0.pt', 'swin': 'v3_ternary_swin_tiny_patch4_window7_224.pt'},
          'eyelid': {'convnext': 'eyelid_opt_convnext_tiny.pt', 'vit': 'eyelid_opt_vit_tiny_patch16_224.pt',
                     'efficientnet': 'eyelid_clean_ternary_efficientnet_b0.pt',
                     'swin': 'eyelid_opt_swin_tiny_patch4_window7_224.pt'}},
         {'face': {'convnext': 0.7879, 'vit': 0.8529, 'swin': 0.8280, 'efficientnet': 0.8011, 'ensemble': 0.8587},
          'eyelid': {'convnext': 0.8578, 'vit': 0.8424, 'swin': 0.7069, 'efficientnet': 0.5739, 'ensemble': 0.7885}})

# ═════════════════════════════════════════════════════════════
# TASK 3/4: DBIL / IBIL (bootstrap G5/G6 canonical splits)
# ═════════════════════════════════════════════════════════════
for task_name, mfile, gcol, rep in [
    ('DBIL', 'dbil_manifest.csv', 'dbil_grade',
     {'face': {'convnext': 0.7997, 'vit': 0.8140, 'swin': 0.8240, 'efficientnet': 0.8302, 'ensemble': 0.8300},
      'eyelid': {'convnext': 0.8145, 'vit': 0.6919, 'swin': 0.7750, 'efficientnet': 0.5696}}),
    ('IBIL', 'ibil_manifest.csv', 'ibil_grade',
     {'face': {'convnext': 0.6013, 'vit': 0.6352, 'swin': 0.6469, 'efficientnet': 0.6981, 'ensemble': 0.6863},
      'eyelid': {'convnext': 0.6475, 'vit': 0.6449, 'swin': 0.5258, 'efficientnet': 0.4865}})]:
    sub = pd.read_csv(os.path.join(BASE, 'data', mfile))
    grades = {r['patient_id']: int(r[gcol]) for _, r in sub.iterrows()}
    _, va_g = stratified_split(grades)
    # faces from task manifest; eyelids from MAIN manifest (task manifests have empty/stale eyelid_images)
    mdf_e = {}
    for _, r in mdf.iterrows():
        if r['n_eyelid'] > 0:
            mdf_e[r['patient_id']] = json.loads(r['eyelid_images'])
    pats = {}
    for _, r in sub.iterrows():
        pid = r['patient_id']
        pats[pid] = {'faces': json.loads(r['face_images']),
                     'eyelids': mdf_e.get(pid, [])}
    face_items = items_from(va_g, pats, 'faces', grades, 4)
    eye_items = items_from(va_g, pats, 'eyelids', grades, 1)
    pref = task_name.lower()
    run_task(task_name, 3,
             {'face': face_items, 'eyelid': eye_items},
             {'face': {a: f'{pref}_face_{a}.pt' for a in BB},
              'eyelid': {a: f'{pref}_eyelid_{a}.pt' for a in BB}},
             rep)

# ═════════════════════════════════════════════════════════════
# TASK 5: Jaundice Type
# ═════════════════════════════════════════════════════════════
# Face: bootstrap G7 split (jt in (1,2)->0, 3->1)
type_pids = {}
for pid, p in sam_pats.items():
    if p['old_cat'] == 'normal': continue
    row = excel_row(pid)
    if row is None: continue
    jt = pd.to_numeric(row[JT_COL], errors='coerce')
    if pd.isna(jt): continue
    if jt in (1, 2): type_pids[pid] = 0
    elif jt == 3: type_pids[pid] = 1
_, va_ty = stratified_split(type_pids)
type_face_items = items_from(va_ty, sam_pats, 'faces', type_pids, 4)

# Eyelid: train_eyelid_meld_type.py Task A split (jt==2->0, jt==3->1); FIRST split call after seed
def build_type_eyelid_val():
    eyelid_patients = {}
    for _, r in mdf.iterrows():
        if r['n_eyelid'] > 0:
            imgs = json.loads(r['eyelid_images'])
            valid = [p for p in imgs if os.path.exists(p)]
            if valid:
                eyelid_patients[r['patient_id']] = valid
    type_items = []
    for pid, e_paths in eyelid_patients.items():
        nums = re.findall(r'\d{6,}', pid)
        hid = nums[-1] if nums else None
        if not hid: continue
        rows = df_ex[df_ex['hid_str'].astype(str).str.strip().str.lstrip('0') == hid.lstrip('0')]
        if len(rows) == 0:
            rows = df_ex[df_ex['hid_str'].astype(str).str.strip().str.lstrip('0').str.endswith(hid[-6:])]
        if len(rows) == 0: continue
        jt = pd.to_numeric(rows.iloc[0][JT_COL], errors='coerce')
        if pd.notna(jt) and jt in (2, 3):
            type_items.append((e_paths[0], 0 if jt == 2 else 1, pid))
    random.seed(SEED)   # replicate module-level seed state: Task A is the first split
    by_l2 = {}
    for it in type_items: by_l2.setdefault(it[1], []).append(it)
    va = []
    for label, its in by_l2.items():
        random.shuffle(its)
        n = max(1, int(len(its) * 0.2))
        va.extend(its[:n])
    return va

type_eye_items = build_type_eyelid_val()

run_task('Jaundice Type', 2,
         {'face': type_face_items, 'eyelid': type_eye_items},
         {'face': {'convnext': 'type_class_convnext_tiny.pt', 'vit': 'type_class_vit_tiny_patch16_224.pt',
                   'efficientnet': 'type_class_efficientnet_b0.pt', 'swin': 'type_class_swin_tiny_patch4_window7_224.pt'},
          'eyelid': {'convnext': 'type_eyelid_convnext_tiny.pt', 'vit': 'type_eyelid_vit_tiny_patch16_224.pt',
                     'efficientnet': 'type_eyelid_efficientnet_b0.pt', 'swin': 'type_eyelid_swin_tiny_patch4_window7_224.pt'}},
         {'face': {'convnext': 0.9653, 'vit': 0.9306, 'swin': 0.9653, 'efficientnet': 0.9444},
          'eyelid': {'convnext': 0.9730, 'vit': 0.7297, 'swin': 0.8514, 'efficientnet': 0.6824}})

# ═════════════════════════════════════════════════════════════
# TASK 6/7: Child-Pugh & MELD (train_cp_meld_images.py split)
# ═════════════════════════════════════════════════════════════
def get_clinical_cp(pid):
    row = excel_row(pid)
    if row is None: return None
    cp_grade = str(row.iloc[15]).strip().upper() if pd.notna(row.iloc[15]) else None
    tbil = pd.to_numeric(row.iloc[51], errors='coerce')
    inr = pd.to_numeric(row.iloc[61], errors='coerce')
    meld = None
    if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
        meld = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr, 1)) + 9.57 * math.log(1.0) + 6.43
        meld = round(max(meld, 6))
    return {'cp_grade': cp_grade, 'meld': meld}

matched = {}
for pid, p in sam_pats.items():
    if p['old_cat'] == 'normal':
        q = dict(p); q.update({'cp_grade': 'A', 'meld': 8})
        matched[pid] = q
    else:
        clin = get_clinical_cp(pid)
        if clin and clin['cp_grade'] in ('A', 'B', 'C'):
            q = dict(p); q.update(clin)
            matched[pid] = q

cp_pids = list(matched.keys())
grades_cp = {pid: matched[pid]['cp_grade'] for pid in cp_pids}
by_grade = {}
for pid in cp_pids: by_grade.setdefault(grades_cp[pid], []).append(pid)
tr_cp, va_cp = set(), set()
for g, ps in by_grade.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_cp.update(ps[:n]); tr_cp.update(ps[n:])

grade_map = {'A': 0, 'B': 1, 'C': 2}
cp_label = {pid: grade_map[matched[pid]['cp_grade']] for pid in cp_pids}
cp_face_items = items_from(va_cp, matched, 'faces', cp_label, 4)
cp_eye_items = items_from(va_cp, matched, 'eyelids', cp_label, 1)

run_task('Child-Pugh', 3,
         {'face': cp_face_items, 'eyelid': cp_eye_items},
         {'face': {a: f'cp_face_{a}.pt' for a in BB},
          'eyelid': {a: f'cp_eyelid_{a}.pt' for a in BB}},
         {'face': {'convnext': 0.8816, 'vit': 0.8738, 'swin': 0.8622, 'efficientnet': 0.7383},
          'eyelid': {'convnext': 0.7187, 'vit': 0.5311, 'swin': 0.8098, 'efficientnet': 0.6673}})

# MELD face: same split, meld_category labels
def meld_cat(m):
    if m is None: return None
    return 0 if m <= 20 else (1 if m <= 30 else 2)
meld_label = {pid: meld_cat(matched[pid]['meld']) for pid in cp_pids}
meld_face_items = items_from(va_cp, matched, 'faces', meld_label, 4)

# MELD eyelid: train_meld_eyelid_vit val (n=59) — rebuild exactly
def build_meld_eyelid_val():
    eyelid_patients = {}
    for _, r in mdf.iterrows():
        if r['n_eyelid'] > 0:
            imgs = json.loads(r['eyelid_images'])
            valid = [p for p in imgs if os.path.exists(p)]
            if valid:
                eyelid_patients[r['patient_id']] = valid
    type_items, meld_items = [], []
    for pid, e_paths in eyelid_patients.items():
        row = excel_row(pid)
        if row is None: continue
        jt = pd.to_numeric(row[JT_COL], errors='coerce')
        if pd.notna(jt) and jt in (2, 3):
            type_items.append((e_paths[0], 0 if jt == 2 else 1, pid))
        tbil = pd.to_numeric(row.iloc[51], errors='coerce')
        inr = pd.to_numeric(row.iloc[61], errors='coerce')
        if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
            ms = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr, 1)) + 9.57 * math.log(1.0) + 6.43
            ms = round(max(ms, 6))
            meld_items.append((e_paths[0], 0 if ms <= 20 else (1 if ms <= 30 else 2), pid))
    added = 0
    normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
    for pid in sorted(os.listdir(normal_ext)):
        pp = os.path.join(normal_ext, pid)
        if not os.path.isdir(pp): continue
        if any(it[2] == pid for it in meld_items): continue
        cands = []
        for root, dirs, files in os.walk(pp):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                    fp = os.path.join(root, f)
                    if os.path.getsize(fp) > 1000000:
                        cands.append(fp)
        if cands:
            meld_items.append((cands[0], 0, pid)); added += 1
            if added >= 150: break
    random.seed(SEED)
    def split(items, ratio=0.2):
        by_l3 = {}
        for it in items: by_l3.setdefault(it[1], []).append(it)
        tr, va = [], []
        for label, its in by_l3.items():
            random.shuffle(its)
            n = max(1, int(len(its) * ratio))
            va.extend(its[:n]); tr.extend(its[n:])
        random.shuffle(tr); random.shuffle(va)
        return tr, va
    split(type_items)   # consume RNG like Task A
    _, va_m = split(meld_items)
    return va_m

meld_eye_items = build_meld_eyelid_val()

run_task('MELD', 3,
         {'face': meld_face_items, 'eyelid': meld_eye_items},
         {'face': {a: f'meld_face_{a}.pt' for a in BB},
          'eyelid': {'convnext': 'meld_eyelid_convnext.pt', 'vit': 'meld_eyelid_vit.pt'}},
         {'face': {'convnext': 0.8919, 'vit': 0.9255, 'swin': 0.8426, 'efficientnet': 0.7259},
          'eyelid': {'convnext': 0.8678, 'vit': 0.7444}})

# ═════════════════════════════════════════════════════════════
# SAVE
# ═════════════════════════════════════════════════════════════
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(x) for x in o]
    if isinstance(o, (np.floating, float)): return None if (isinstance(o, float) and math.isnan(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o

with open(os.path.join(RES, 'ensemble_fusion_results.json'), 'w', encoding='utf-8') as f:
    json.dump(clean(RESULTS), f, indent=2, ensure_ascii=False)

# prob cache for future bootstrap
cache = {}
for task, scopes in PROBS.items():
    cache[task] = {}
    for scope, archs in scopes.items():
        cache[task][scope] = {a: {pid: [prob.tolist(), lab] for pid, (prob, lab) in d.items()}
                              for a, d in archs.items()}
with open(os.path.join(RES, 'predictions_cache_ensemble.json'), 'w', encoding='utf-8') as f:
    json.dump(clean(cache), f, ensure_ascii=False)

print('\n' + '=' * 66)
print('  SUMMARY')
print('=' * 66)
for task, res in RESULTS.items():
    print(f'\n  {task}')
    for k, v in res.items():
        if v['auc'] is None: continue
        rep = f" (reported {v['reported']:.4f})" if v.get('reported') else ''
        print(f'    {k:<22} AUC={v["auc"]:.4f} n={v["n"]}{rep}')
print(f'\n  Saved: results/ensemble_fusion_results.json')
print(f'  Saved: results/predictions_cache_ensemble.json')
