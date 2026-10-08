# -*- coding: utf-8 -*-
"""
Unified protocol v2: ALL tasks x scopes x archs + ensembles + fusions,
ONE evaluation protocol (bootstrap-style): <=12 images/patient, mean softmax,
patient-level; 1000-iteration patient bootstrap 95% CI for every AUC.

Val patient sets are IDENTICAL to ensemble_fusion_all.py (v1); only the
image-aggregation protocol changes (<=12 imgs everywhere) and CIs are added.

Output:
  results/canonical_v2_results.json     (every cell: auc, lo, hi, acc, n)
  results/canonical_v2_probs.json       (patient-level prob cache)
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
IMG_SIZE = 224; BS = 32; N_IMG = 12; N_BOOT = 1000

BB = {'convnext': 'convnext_tiny', 'vit': 'vit_tiny_patch16_224',
      'efficientnet': 'efficientnet_b0', 'swin': 'swin_tiny_patch4_window7_224'}


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
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid


@torch.no_grad()
def model_patient_probs(ckpt, backbone, nc, items):
    path = os.path.join(MODEL, ckpt)
    if not os.path.exists(path):
        print(f'      MISSING {ckpt}'); return None
    try:
        m = timm.create_model(backbone, pretrained=False, num_classes=nc).to(DEV)
        m.load_state_dict(torch.load(path, map_location=DEV)); m.eval()
    except Exception as e:
        print(f'      LOAD-FAIL {ckpt}: {e}'); return None
    dl = DataLoader(ImgDS(items), BS, shuffle=False, num_workers=0)
    d = {}
    for im, la, pi in dl:
        o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
        for p, l, pid in zip(o, la.numpy(), pi):
            d.setdefault(pid, ([], int(l))); d[pid][0].append(p)
    del m; torch.cuda.empty_cache()
    return {pid: (np.mean(v[0], 0), v[1]) for pid, v in d.items()}


def auc_val(true, probs, nc):
    if len(np.unique(true)) < 2: return float('nan')
    try:
        return roc_auc_score(true, probs[:, 1]) if nc == 2 else \
            roc_auc_score(true, probs, multi_class='ovr', labels=list(range(nc)))
    except Exception:
        return float('nan')


def eval_pd(pd_, nc, pids=None):
    pids = pids or sorted(pd_.keys())
    avg = np.array([pd_[p][0] for p in pids]); true = np.array([pd_[p][1] for p in pids])
    auc = auc_val(true, avg, nc)
    acc = accuracy_score(true, avg.argmax(1))
    rng = np.random.RandomState(SEED)
    boots = []
    n = len(true)
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        a = auc_val(true[idx], avg[idx], nc)
        if not math.isnan(a): boots.append(a)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float('nan'),) * 2
    return {'auc': float(auc), 'lo': lo, 'hi': hi, 'acc': float(acc), 'n': len(pids)}


def merge_probs(dicts, pids):
    out = {}
    for pid in pids:
        if all(pid in d for d in dicts):
            out[pid] = (np.mean([d[pid][0] for d in dicts], 0), dicts[0][pid][1])
    return out


# ── shared data ──
print('[init] loading data...')
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
main_eye = {}
for _, r in mdf.iterrows():
    if r['n_eyelid'] > 0:
        main_eye[r['patient_id']] = json.loads(r['eyelid_images'])

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

def items_from(pids, pats, key, label_map, n_img=N_IMG):
    out = []
    for pid in pids:
        if pid not in pats: continue
        lab = label_map(pid) if callable(label_map) else label_map.get(pid)
        if lab is None: continue
        for ip in pats[pid].get(key, [])[:n_img]:
            if os.path.exists(ip):
                out.append((ip, lab, pid))
    return out


RESULTS, PROBS = {}, {}

def run_task(task, nc, splits, models):
    print('\n' + '=' * 66)
    print(f'  TASK: {task} (nc={nc})')
    print('=' * 66)
    RESULTS[task] = {}
    PROBS[task] = {}
    for scope in ['face', 'eyelid']:
        items = splits.get(scope)
        if not items: continue
        print(f'  [{scope}] items={len(items)} patients={len(set(p for _, _, p in items))} {dict(Counter(l for _, l, _ in items))}')
        PROBS[task][scope] = {}
        for arch, ckpt in models.get(scope, {}).items():
            pd_ = model_patient_probs(ckpt, BB[arch], nc, items)
            if pd_ is None: continue
            PROBS[task][scope][arch] = pd_
            r = eval_pd(pd_, nc)
            print(f'    {scope:<7} {arch:<13} AUC={r["auc"]:.4f} [{r["lo"]:.3f}-{r["hi"]:.3f}] ACC={r["acc"]:.3f} n={r["n"]}')
            RESULTS[task][f'{scope}_{arch}'] = r
        archs = list(PROBS[task][scope].keys())
        if len(archs) >= 2:
            common = set.intersection(*[set(PROBS[task][scope][a].keys()) for a in archs])
            ens = merge_probs([PROBS[task][scope][a] for a in archs], sorted(common))
            r = eval_pd(ens, nc)
            print(f'    {scope:<7} ENSEMBLE({len(archs)}) AUC={r["auc"]:.4f} [{r["lo"]:.3f}-{r["hi"]:.3f}] n={r["n"]}')
            RESULTS[task][f'{scope}_ensemble'] = {**r, 'archs': archs}
            PROBS[task][scope]['__ensemble__'] = ens
    if 'face' in PROBS[task] and 'eyelid' in PROBS[task]:
        fa = {a: d for a, d in PROBS[task]['face'].items() if not a.startswith('__')}
        ea = {a: d for a, d in PROBS[task]['eyelid'].items() if not a.startswith('__')}
        fpids = set().union(*[set(d.keys()) for d in fa.values()]) if fa else set()
        epids = set().union(*[set(d.keys()) for d in ea.values()]) if ea else set()
        both = sorted(fpids & epids)
        print(f'  [fusion] intersection patients: {len(both)}')
        for arch in sorted(set(fa) & set(ea)):
            fus = merge_probs([fa[arch], ea[arch]], both)
            if not fus: continue
            r = eval_pd(fus, nc)
            print(f'    fusion  {arch:<13} AUC={r["auc"]:.4f} [{r["lo"]:.3f}-{r["hi"]:.3f}] n={r["n"]}')
            RESULTS[task][f'fusion_{arch}'] = r
        dicts = list(fa.values()) + list(ea.values())
        fus_all = merge_probs(dicts, both)
        if fus_all:
            r = eval_pd(fus_all, nc)
            print(f'    fusion  ENSEMBLE({len(dicts)}) AUC={r["auc"]:.4f} [{r["lo"]:.3f}-{r["hi"]:.3f}] n={r["n"]}')
            RESULTS[task]['fusion_ensemble'] = r


# ═══ TASK 1: Binary Screening ═══
bin_pids = {pid: (0 if p['old_cat'] == 'normal' else 1) for pid, p in sam_pats.items()}
by_label = {}
for pid, l in bin_pids.items(): by_label.setdefault(l, []).append(pid)
tr_b, va_b = set(), set()
for g, pids in by_label.items():
    random.seed(SEED); random.shuffle(pids)
    n = max(1, int(len(pids) * 0.2))
    va_b.update(pids[:n]); tr_b.update(pids[n:])
bin_face_items = items_from(va_b, sam_pats, 'faces', bin_pids)

def build_bin_eyelid():
    pats, labels = {}, {}
    df = mdf[mdf['n_eyelid'] > 0]
    for _, r in df.iterrows():
        pats[r['patient_id']] = {'eyelids': json.loads(r['eyelid_images'])}
        labels[r['patient_id']] = 0 if r['category'] == 'normal' else 1
    normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
    for pid in sorted(os.listdir(normal_ext)):
        pp = os.path.join(normal_ext, pid)
        if not os.path.isdir(pp): continue
        cands = []
        for root, dirs, files in os.walk(pp):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                    fp = os.path.join(root, f)
                    if os.path.getsize(fp) > 1000000:
                        cands.append(fp)
        if cands:
            pats[pid] = {'eyelids': cands}; labels[pid] = 0
    items = [(ip, labels[pid], pid) for pid in pats for ip in pats[pid]['eyelids'][:1]]
    by_l = {}
    for it in items: by_l.setdefault(it[1], []).append(it)
    va_p = set()
    random.seed(SEED)
    for label, its in by_l.items():
        its = list(its); random.shuffle(its)
        n = max(1, int(len(its) * 0.2))
        va_p.update(it[2] for it in its[:n])
    return items_from(va_p, pats, 'eyelids', labels)

bin_eye_items = build_bin_eyelid()

run_task('Binary Screening', 2,
         {'face': bin_face_items, 'eyelid': bin_eye_items},
         {'face': {'convnext': 'final_binary_face_convnext.pt', 'vit': 'final_binary_face_vit.pt',
                   'efficientnet': 'final_binary_face_efficientnet.pt', 'swin': 'final_binary_face_swin.pt'},
          'eyelid': {'convnext': 'eyelid_binary_convnext.pt', 'vit': 'eyelid_binary_vit_tiny_patch16_224.pt',
                     'efficientnet': 'eyelid_binary_efficientnet_b0.pt',
                     'swin': 'eyelid_binary_swin_tiny_patch4_window7_224.pt'}})

# ═══ TASK 2: Ternary Grading ═══
tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tern_pids = {pid: tmap[p['old_cat']] for pid, p in sam_pats.items() if p['old_cat'] != 'normal'}
_, va_t3 = stratified_split(tern_pids)
tern_face_items = items_from(va_t3, sam_pats, 'faces', tern_pids)

j_df = mdf[mdf['category'].isin(['mild', 'moderate', 'severe']) & (mdf['n_eyelid'] > 0)]
tbil_val = [{'label': tmap[r['category']], 'pid': r['patient_id']} for _, r in j_df.iterrows()]
random.seed(SEED)
by_l = {}
for p in tbil_val: by_l.setdefault(p['label'], []).append(p)
va_tp = []
for l, g in by_l.items():
    random.shuffle(g); va_tp.extend(g[:max(1, int(len(g) * 0.2))])
va_tp = [p['pid'] for p in va_tp]
tp_labels = {p['pid']: p['label'] for p in tbil_val}
eye_pats = {pid: {'eyelids': main_eye[pid]} for pid in va_tp if pid in main_eye}
tern_eye_items = items_from(va_tp, eye_pats, 'eyelids', tp_labels)

run_task('Ternary Grading', 3,
         {'face': tern_face_items, 'eyelid': tern_eye_items},
         {'face': {'convnext': 'v3_ternary_convnext_tiny.pt', 'vit': 'v3_ternary_vit_tiny_patch16_224.pt',
                   'efficientnet': 'v3_ternary_efficientnet_b0.pt', 'swin': 'v3_ternary_swin_tiny_patch4_window7_224.pt'},
          'eyelid': {'convnext': 'eyelid_opt_convnext_tiny.pt', 'vit': 'eyelid_opt_vit_tiny_patch16_224.pt',
                     'efficientnet': 'eyelid_clean_ternary_efficientnet_b0.pt',
                     'swin': 'eyelid_opt_swin_tiny_patch4_window7_224.pt'}})

# ═══ TASK 3/4: DBIL / IBIL ═══
for task_name, mfile, gcol in [('DBIL', 'dbil_manifest.csv', 'dbil_grade'),
                               ('IBIL', 'ibil_manifest.csv', 'ibil_grade')]:
    sub = pd.read_csv(os.path.join(BASE, 'data', mfile))
    grades = {r['patient_id']: int(r[gcol]) for _, r in sub.iterrows()}
    _, va_g = stratified_split(grades)
    pats = {}
    for _, r in sub.iterrows():
        pid = r['patient_id']
        pats[pid] = {'faces': json.loads(r['face_images']), 'eyelids': main_eye.get(pid, [])}
    pref = task_name.lower()
    run_task(task_name, 3,
             {'face': items_from(va_g, pats, 'faces', grades),
              'eyelid': items_from(va_g, pats, 'eyelids', grades)},
             {'face': {a: f'{pref}_face_{a}.pt' for a in BB},
              'eyelid': {a: f'{pref}_eyelid_{a}.pt' for a in BB}})

# ═══ TASK 5: Jaundice Type ═══
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
type_face_items = items_from(va_ty, sam_pats, 'faces', type_pids)

def build_type_eyelid_val():
    eyelid_patients = {}
    for _, r in mdf.iterrows():
        if r['n_eyelid'] > 0:
            imgs = json.loads(r['eyelid_images'])
            valid = [p for p in imgs if os.path.exists(p)]
            if valid: eyelid_patients[r['patient_id']] = valid
    type_items, labs = [], {}
    for pid, e_paths in eyelid_patients.items():
        row = excel_row(pid)
        if row is None: continue
        jt = pd.to_numeric(row[JT_COL], errors='coerce')
        if pd.notna(jt) and jt in (2, 3):
            type_items.append((e_paths[0], 0 if jt == 2 else 1, pid))
            labs[pid] = 0 if jt == 2 else 1
    random.seed(SEED)
    by_l2 = {}
    for it in type_items: by_l2.setdefault(it[1], []).append(it)
    va_p = []
    for label, its in by_l2.items():
        random.shuffle(its)
        n = max(1, int(len(its) * 0.2))
        va_p.extend(it[2] for it in its[:n])
    pats = {pid: {'eyelids': eyelid_patients[pid]} for pid in va_p}
    return items_from(va_p, pats, 'eyelids', labs)

type_eye_items = build_type_eyelid_val()

run_task('Jaundice Type', 2,
         {'face': type_face_items, 'eyelid': type_eye_items},
         {'face': {'convnext': 'type_class_convnext_tiny.pt', 'vit': 'type_class_vit_tiny_patch16_224.pt',
                   'efficientnet': 'type_class_efficientnet_b0.pt', 'swin': 'type_class_swin_tiny_patch4_window7_224.pt'},
          'eyelid': {'convnext': 'type_eyelid_convnext_tiny.pt', 'vit': 'type_eyelid_vit_tiny_patch16_224.pt',
                     'efficientnet': 'type_eyelid_efficientnet_b0.pt', 'swin': 'type_eyelid_swin_tiny_patch4_window7_224.pt'}})

# ═══ TASK 6/7: Child-Pugh & MELD ═══
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
        q = dict(p); q.update({'cp_grade': 'A', 'meld': 8}); matched[pid] = q
    else:
        clin = get_clinical_cp(pid)
        if clin and clin['cp_grade'] in ('A', 'B', 'C'):
            q = dict(p); q.update(clin); matched[pid] = q

cp_pids = list(matched.keys())
by_grade = {}
for pid in cp_pids: by_grade.setdefault(matched[pid]['cp_grade'], []).append(pid)
tr_cp, va_cp = set(), set()
for g, ps in by_grade.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_cp.update(ps[:n]); tr_cp.update(ps[n:])

grade_map = {'A': 0, 'B': 1, 'C': 2}
cp_label = {pid: grade_map[matched[pid]['cp_grade']] for pid in cp_pids}

run_task('Child-Pugh', 3,
         {'face': items_from(va_cp, matched, 'faces', cp_label),
          'eyelid': items_from(va_cp, matched, 'eyelids', cp_label)},
         {'face': {a: f'cp_face_{a}.pt' for a in BB},
          'eyelid': {a: f'cp_eyelid_{a}.pt' for a in BB}})

def meld_cat(m):
    if m is None: return None
    return 0 if m <= 20 else (1 if m <= 30 else 2)
meld_label = {pid: meld_cat(matched[pid]['meld']) for pid in cp_pids}
meld_face_items = items_from(va_cp, matched, 'faces', meld_label)

def build_meld_eyelid_val():
    eyelid_patients = {}
    for _, r in mdf.iterrows():
        if r['n_eyelid'] > 0:
            imgs = json.loads(r['eyelid_images'])
            valid = [p for p in imgs if os.path.exists(p)]
            if valid: eyelid_patients[r['patient_id']] = valid
    type_items, meld_items, labs, pats = [], [], {}, {}
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
            labs[pid] = meld_items[-1][1]; pats[pid] = {'eyelids': e_paths}
    added = 0
    normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
    for pid in sorted(os.listdir(normal_ext)):
        pp = os.path.join(normal_ext, pid)
        if not os.path.isdir(pp): continue
        if any(it[2] == pid for it in meld_items): continue
        cands = []
        for root, dirs, files in os.walk(pp):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                    fp = os.path.join(root, f)
                    if os.path.getsize(fp) > 1000000:
                        cands.append(fp)
        if cands:
            meld_items.append((cands[0], 0, pid)); labs[pid] = 0; pats[pid] = {'eyelids': cands}
            added += 1
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
    split(type_items)
    _, va_m = split(meld_items)
    va_p = [it[2] for it in va_m]
    return items_from(va_p, pats, 'eyelids', labs)

meld_eye_items = build_meld_eyelid_val()

run_task('MELD', 3,
         {'face': meld_face_items, 'eyelid': meld_eye_items},
         {'face': {a: f'meld_face_{a}.pt' for a in BB},
          'eyelid': {'convnext': 'meld_eyelid_convnext.pt', 'vit': 'meld_eyelid_vit.pt',
                     'swin': 'meld_eyelid_swin.pt', 'efficientnet': 'meld_eyelid_efficientnet.pt'}})

# ═══ SAVE ═══
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(x) for x in o]
    if isinstance(o, float) and math.isnan(o): return None
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o

with open(os.path.join(RES, 'canonical_v2_results.json'), 'w', encoding='utf-8') as f:
    json.dump(clean(RESULTS), f, indent=2, ensure_ascii=False)

cache = {}
for task, scopes in PROBS.items():
    cache[task] = {}
    for scope, archs in scopes.items():
        cache[task][scope] = {a: {pid: [prob.tolist(), lab] for pid, (prob, lab) in d.items()}
                              for a, d in archs.items()}
with open(os.path.join(RES, 'canonical_v2_probs.json'), 'w', encoding='utf-8') as f:
    json.dump(clean(cache), f, ensure_ascii=False)

print('\n' + '=' * 66)
print('  CANONICAL V2 SUMMARY (12-img protocol, 95% CI)')
print('=' * 66)
for task, res in RESULTS.items():
    print(f'\n  {task}')
    for k, v in res.items():
        if v.get('auc') is None: continue
        print(f'    {k:<22} AUC={v["auc"]:.4f} [{v["lo"]:.3f}-{v["hi"]:.3f}] n={v["n"]}')
print('\n  Saved: results/canonical_v2_results.json')
print('  Saved: results/canonical_v2_probs.json')
