# -*- coding: utf-8 -*-
"""
Generate and cache raw predictions for all figure-generation tasks.

This runs inference ONCE, saves (probs, true, pids) per task to disk,
so figure scripts can load without re-running models.

Cache: results/figure_predictions_cache.npz
"""
import os, re, json, random, math, pickle, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm

BASE = r'D:\research\人脸识别营养\传染科'
INT_DATA = os.path.join(BASE, 'data', 'sam_processed_v3')
EXT_DATA = os.path.join(BASE, 'data', 'external_processed_v3')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
CACHE = os.path.join(RES, 'figure_predictions_cache.npz')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMG_SIZE = 224
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
    if gray < 1: return img
    out = img.astype(np.float32)
    out[:,:,0] = np.clip(out[:,:,0] * gray / (mr + 1e-6), 0, 255)
    out[:,:,1] = np.clip(out[:,:,1] * gray / (mg + 1e-6), 0, 255)
    out[:,:,2] = np.clip(out[:,:,2] * gray / (mb + 1e-6), 0, 255)
    return out.astype(np.uint8)

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def predict_patient(model, paths, prep_fn, n_max=12):
    valid = [p for p in paths[:n_max] if os.path.exists(p)]
    if not valid: return None
    tensors = []
    for p in valid:
        img = read_img(p)
        if img is None: continue
        img = prep_fn(img)
        tensors.append(tf(Image.fromarray(img).convert('RGB')).unsqueeze(0))
    if not tensors: return None
    t = torch.cat(tensors).to(DEV)
    with torch.no_grad():
        return F.softmax(model(t), dim=1).mean(0).cpu().numpy()

def load_model(bb, ckpt, nc):
    path = os.path.join(MODEL, ckpt)
    if not os.path.exists(path): return None
    m = timm.create_model(bb, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    m.eval()
    return m

def run_inference(bb, ckpt, nc, items, prep_fn, img_key='faces'):
    m = load_model(bb, ckpt, nc)
    if m is None: return None, None
    probs, true = [], []
    for it in items:
        pr = predict_patient(m, it.get(img_key, []), prep_fn)
        if pr is None: pr = np.ones(nc) / nc
        probs.append(pr); true.append(it['label'])
    del m; torch.cuda.empty_cache()
    return np.array(probs), np.array(true)


# ── Data loading ─────────────────────────────────────────────
print('[1] Loading clinical data...')
df_ex = pd.read_excel(EXCEL)
df_ex['hid_str'] = df_ex['2、患者住院号'].astype(str)

def extract_hid(fn):
    nums = re.findall(r'\d+', fn.replace('.zip', ''))
    return nums[-1] if nums else None

def strip_zeros(s): return s.lstrip('0') if s else s

def get_clinical(pid):
    hid = extract_hid(pid)
    if not hid: return None
    hn = strip_zeros(hid)
    rows = df_ex[df_ex['hid_str'].apply(strip_zeros) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hn[-6:])]
    if len(rows) == 0: return None
    r = rows.iloc[0]
    cp_grade = str(r.iloc[15]).strip().upper() if pd.notna(r.iloc[15]) else None
    tbil = pd.to_numeric(r.iloc[51], errors='coerce')
    inr = pd.to_numeric(r.iloc[61], errors='coerce')
    meld = None
    if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
        meld = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr, 1)) + 9.57 * math.log(1.0) + 6.43
        meld = round(max(meld, 6))
    return {'cp_grade': cp_grade, 'meld': meld}

# Collect internal patients
mdf = pd.read_csv(MANIFEST)
patients = {}
for cat in ['normal', 'mild', 'moderate', 'severe']:
    cd = os.path.join(INT_DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces: patients[pid] = {'faces': faces, 'cat': cat}
for _, r in mdf.iterrows():
    pid = r['patient_id']
    if pid in patients and r['n_eyelid'] > 0:
        patients[pid]['eyelids'] = json.loads(r['eyelid_images'])

# CP/MELD cohort
matched = {}
for pid, p in patients.items():
    if p['cat'] == 'normal':
        p.update({'cp_grade': 'A', 'meld': 8}); matched[pid] = p
    else:
        clin = get_clinical(pid)
        if clin and clin['cp_grade'] in ('A', 'B', 'C'):
            p.update(clin); matched[pid] = p

# Stratified split by CP grade (matches training)
pids = list(matched.keys())
grades = {pid: matched[pid]['cp_grade'] for pid in pids}
by_grade = {}
for pid in pids: by_grade.setdefault(grades[pid], []).append(pid)
va_pids = set()
for g, ps in by_grade.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_pids.update(ps[:n])

# Binary split
bin_pids = {pid: (0 if p['cat'] == 'normal' else 1) for pid, p in patients.items()}
by_label = {}
for pid, lbl in bin_pids.items(): by_label.setdefault(lbl, []).append(pid)
va_bin_pids = set()
for lbl, ps in by_label.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_bin_pids.update(ps[:n])

# TBIL ternary split
tmap = {'mild': 0, 'moderate': 1, 'severe': 2}
tern_pids = {pid: tmap[p['cat']] for pid, p in patients.items() if p['cat'] != 'normal'}
by_t = {}
for pid, g in tern_pids.items(): by_t.setdefault(g, []).append(pid)
va_tern_pids = set()
for g, ps in by_t.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_tern_pids.update(ps[:n])

# Type split
jt_col = [c for c in df_ex.columns if '黄疸类型' in str(c)][0]
def get_lab_val(pid, col):
    nums = re.findall(r'\d+', pid.replace('.zip', ''))
    hid = nums[-1] if nums else None
    if not hid: return None
    hn = hid.lstrip('0')
    rows = df_ex[df_ex['hid_str'].apply(lambda s: s.lstrip('0') if s else s) == hn]
    if len(rows) == 0:
        rows = df_ex[df_ex['hid_str'].apply(lambda s: s.lstrip('0') if s else s).str.endswith(hn[-6:])]
    return float(rows.iloc[0][col]) if len(rows) > 0 else None

type_pids = {}
for pid, p in patients.items():
    if p['cat'] == 'normal': continue
    jt = get_lab_val(pid, jt_col)
    if jt in (1, 2): type_pids[pid] = 0
    elif jt == 3: type_pids[pid] = 1
by_ty = {}
for pid, g in type_pids.items(): by_ty.setdefault(g, []).append(pid)
va_type_pids = set()
for g, ps in by_ty.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_type_pids.update(ps[:n])

# Build external data (filtered)
ext_all = {}
for cat, label in [('normal', 0), ('jaundice', 1)]:
    cd = os.path.join(EXT_DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces: ext_all[pid] = {'faces': faces, 'label': label}

# External domain-adapted held-out (n=19)
ext_by_label = {}
for pid, p in ext_all.items(): ext_by_label.setdefault(p['label'], []).append(pid)
te_ext = set()
for lbl, ps in ext_by_label.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    te_ext.update(ps[:n])

# ── Run inference and cache ──────────────────────────────────
cache = {}

# 1. Binary internal
print('[2] Binary internal (n=110)...')
va_bin = [{'faces': patients[pid]['faces'], 'label': bin_pids[pid], 'pid': pid}
          for pid in va_bin_pids if pid in patients]
for name, bb, ckpt in [('Bin-v3-ViT', 'vit_tiny_patch16_224', 'v3_binary_vit_tiny_patch16_224.pt'),
                        ('Bin-v3-ConvNeXt', 'convnext_tiny', 'v3_binary_convnext_tiny.pt'),
                        ('Bin-v3-EffNet', 'efficientnet_b0', 'v3_binary_efficientnet_b0.pt')]:
    probs, true = run_inference(bb, ckpt, 2, va_bin, clahe)
    if probs is not None:
        cache[f'binary_int_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_bin]}

# 2. External pure (all models on n=117)
print('[3] External pure (n=117)...')
ext_items = [{'faces': ext_all[pid]['faces'], 'label': ext_all[pid]['label'], 'pid': pid}
             for pid in sorted(ext_all.keys())]
for name, bb, ckpt in [('V3-ViT', 'vit_tiny_patch16_224', 'v3_binary_vit_tiny_patch16_224.pt'),
                        ('V3-ConvNeXt', 'convnext_tiny', 'v3_binary_convnext_tiny.pt'),
                        ('V3-EffNet', 'efficientnet_b0', 'v3_binary_efficientnet_b0.pt')]:
    probs, true = run_inference(bb, ckpt, 2, ext_items, clahe)
    if probs is not None:
        cache[f'binary_ext_pure_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in ext_items]}

# 3. Domain-adapted (n=19)
print('[4] Domain-adapted (n=19)...')
da_items = [{'faces': ext_all[pid]['faces'], 'label': ext_all[pid]['label'], 'pid': pid}
            for pid in sorted(te_ext) if pid in ext_all]
for name, bb, ckpt in [('DA-ViT', 'vit_tiny_patch16_224', 'domain_adapt_binary_vit.pt'),
                        ('DA-ConvNeXt', 'convnext_tiny', 'domain_adapt_binary_convnext.pt'),
                        ('DA-EffNet', 'efficientnet_b0', 'domain_adapt_binary_efficientnet.pt'),
                        ('DA-Swin', 'swin_tiny_patch4_window7_224', 'domain_adapt_binary_swin.pt')]:
    probs, true = run_inference(bb, ckpt, 2, da_items, lambda x: clahe(grey_world(x)))
    if probs is not None:
        cache[f'binary_ext_da_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in da_items]}

# 4. TBIL facial ternary
print('[5] TBIL facial ternary (n=40)...')
va_tern = [{'faces': patients[pid]['faces'], 'label': tern_pids[pid], 'pid': pid}
           for pid in va_tern_pids if pid in patients]
for name, bb, ckpt in [('FaceTern-ConvNeXt', 'convnext_tiny', 'v3_ternary_convnext_tiny.pt'),
                        ('FaceTern-ViT', 'vit_tiny_patch16_224', 'v3_ternary_vit_tiny_patch16_224.pt'),
                        ('FaceTern-Swin', 'swin_tiny_patch4_window7_224', 'v3_ternary_swin_tiny_patch4_window7_224.pt')]:
    probs, true = run_inference(bb, ckpt, 3, va_tern, clahe)
    if probs is not None:
        cache[f'tbil_face_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_tern]}

# 5. TBIL eyelid ternary
print('[6] TBIL eyelid ternary (n=42)...')
j_df = mdf[mdf['category'].isin(['mild','moderate','severe']) & (mdf['n_eyelid']>0)]
va_eye_items = []
for _, r in j_df.iterrows():
    pid = r['patient_id']
    if pid in va_tern_pids:
        imgs = json.loads(r['eyelid_images'])
        va_eye_items.append({'eyelids': imgs, 'label': tmap[r['category']], 'pid': pid})
for name, bb, ckpt in [('EyeTBIL-ConvNeXt', 'convnext_tiny', 'eyelid_opt_convnext_tiny.pt'),
                        ('EyeTBIL-ViT', 'vit_tiny_patch16_224', 'eyelid_opt_vit_tiny_patch16_224.pt')]:
    probs, true = run_inference(bb, ckpt, 3, va_eye_items, clahe, 'eyelids')
    if probs is not None:
        cache[f'tbil_eye_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_eye_items]}

# 6. DBIL eyelid
print('[7] DBIL eyelid (n=52)...')
dbil_mdf = pd.read_csv(os.path.join(BASE, 'data', 'dbil_manifest.csv'))
dbil_grades = {r['patient_id']: int(r['dbil_grade']) for _, r in dbil_mdf.iterrows()}
by_d = {}
for pid, g in dbil_grades.items(): by_d.setdefault(g, []).append(pid)
va_d = set()
for g, ps in by_d.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_d.update(ps[:n])
va_d_items = [{'eyelids': json.loads(dbil_mdf[dbil_mdf['patient_id']==pid].iloc[0]['eyelid_images']),
               'label': dbil_grades[pid], 'pid': pid}
              for pid in va_d if pid in dbil_grades
              and json.loads(dbil_mdf[dbil_mdf['patient_id']==pid].iloc[0]['eyelid_images'])]
probs, true = run_inference('convnext_tiny', 'dbil_eyelid_convnext.pt', 3, va_d_items, clahe, 'eyelids')
if probs is not None:
    cache['dbil_eye_ConvNeXt'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_d_items]}

# 7. IBIL face
print('[8] IBIL face (n=55)...')
ibil_mdf = pd.read_csv(os.path.join(BASE, 'data', 'ibil_manifest.csv'))
ibil_grades = {r['patient_id']: int(r['ibil_grade']) for _, r in ibil_mdf.iterrows()}
by_i = {}
for pid, g in ibil_grades.items(): by_i.setdefault(g, []).append(pid)
va_i = set()
for g, ps in by_i.items():
    random.seed(SEED); random.shuffle(ps)
    n = max(1, int(len(ps) * 0.2))
    va_i.update(ps[:n])
va_i_items = [{'faces': json.loads(ibil_mdf[ibil_mdf['patient_id']==pid].iloc[0]['face_images']),
               'label': ibil_grades[pid], 'pid': pid}
              for pid in va_i if pid in ibil_grades]
probs, true = run_inference('swin_tiny_patch4_window7_224', 'ibil_face_swin.pt', 3, va_i_items, clahe, 'faces')
if probs is not None:
    cache['ibil_face_Swin'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_i_items]}

# 8. Jaundice type
print('[9] Jaundice type (n=39)...')
va_type = [{'faces': patients[pid]['faces'], 'label': type_pids[pid], 'pid': pid}
           for pid in va_type_pids if pid in patients]
for name, bb, ckpt in [('Type-EffNet', 'efficientnet_b0', 'type_class_efficientnet_b0.pt'),
                        ('Type-ConvNeXt', 'convnext_tiny', 'type_class_convnext_tiny.pt')]:
    probs, true = run_inference(bb, ckpt, 2, va_type, clahe)
    if probs is not None:
        cache[f'type_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_type]}

# 9. Child-Pugh
print('[10] Child-Pugh (n=110)...')
grade_map = {'A': 0, 'B': 1, 'C': 2}
va_cp = [{'faces': matched[pid]['faces'], 'label': grade_map[matched[pid]['cp_grade']], 'pid': pid}
         for pid in va_pids if pid in matched and matched[pid]['cp_grade'] in grade_map]
for name, bb, ckpt in [('CP-ConvNeXt', 'convnext_tiny', 'cp_face_convnext.pt'),
                        ('CP-ViT', 'vit_tiny_patch16_224', 'cp_face_vit.pt')]:
    probs, true = run_inference(bb, ckpt, 3, va_cp, clahe)
    if probs is not None:
        cache[f'cp_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_cp]}

# 10. MELD
print('[11] MELD (n=110)...')
def meld_cat(p):
    m = p.get('meld')
    if m is None: return None
    if m <= 20: return 0
    elif m <= 30: return 1
    else: return 2
va_meld = [{'faces': matched[pid]['faces'], 'label': meld_cat(matched[pid]), 'pid': pid}
           for pid in va_pids if pid in matched and meld_cat(matched[pid]) is not None]
for name, bb, ckpt in [('MELD-ViT', 'vit_tiny_patch16_224', 'meld_face_vit.pt'),
                        ('MELD-ConvNeXt', 'convnext_tiny', 'meld_face_convnext.pt')]:
    probs, true = run_inference(bb, ckpt, 3, va_meld, clahe)
    if probs is not None:
        cache[f'meld_{name}'] = {'probs': probs, 'true': true, 'pids': [it['pid'] for it in va_meld]}

# ── Save cache ───────────────────────────────────────────────
print(f'\n[SAVE] Caching {len(cache)} prediction sets to {CACHE}')
np.savez_compressed(CACHE, **{k: pickle.dumps(v) for k, v in cache.items()})
print('Done.')
for k in sorted(cache.keys()):
    print(f'  {k:40s}: n={len(cache[k]["true"])}')
