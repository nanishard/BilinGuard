# -*- coding: utf-8 -*-
"""
V4-lite external reprocessing: YuNet detect + padded bbox crop ONLY.
No SAM, no alignment, no CLAHE at save time (eval code applies CLAHE once).
Compare with v3-processed and full-v4 results.
"""
import os, json, shutil, numpy as np, cv2, pandas as pd
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = r'D:\research\人脸识别营养\传染科'
EXT_RAW = os.path.join(BASE, 'data', 'external_frames')
EXT_LITE = os.path.join(BASE, 'data', 'external_v4_lite')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
TEMP_DIR = r'C:\Users\o\AppData\Local\Temp\opencode'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224; BS = 32

# YuNet setup
yunet_tmp = os.path.join(TEMP_DIR, 'yunet.onnx')
if not os.path.exists(yunet_tmp):
    shutil.copy2(os.path.join(MODEL, 'face_detection_yunet_2023mar.onnx'), yunet_tmp)
detector = cv2.FaceDetectorYN.create(yunet_tmp, "", (640, 480),
                                     score_threshold=0.5, nms_threshold=0.3, top_k=5000)

def process_frame_lite(frame_bgr):
    """YuNet detect + padded bbox crop only. No SAM, no alignment, no CLAHE."""
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    if faces is None or len(faces) == 0: return None
    face = faces[np.argmax(faces[:, -1])]
    x, y, fw, fh = face[:4].astype(np.float32)
    score = float(face[14])
    if score < 0.5: return None
    # Pad 30% for context
    pad = 0.3
    x1 = max(0, int(x - fw * pad))
    y1 = max(0, int(y - fh * pad))
    x2 = min(w, int(x + fw + fw * pad))
    y2 = min(h, int(y + fh + fh * pad))
    crop = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)[y1:y2, x1:x2]
    if crop.size == 0: return None
    # Resize to 224x224
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

# Process external frames
print('V4-lite processing (YuNet bbox crop only)...')
for label, cat in [(0, 'normal'), (1, 'jaundice')]:
    cat_dir = os.path.join(EXT_RAW, cat)
    out_cat = os.path.join(EXT_LITE, cat)
    os.makedirs(out_cat, exist_ok=True)
    n_total, n_kept = 0, 0
    for pid in sorted(os.listdir(cat_dir)):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p): continue
        out_p = os.path.join(out_cat, pid)
        os.makedirs(out_p, exist_ok=True)
        existing = [f for f in os.listdir(out_p) if f.endswith('_face.jpg')]
        if len(existing) >= 3: n_kept += len(existing); continue
        for fname in sorted([f for f in os.listdir(p) if f.endswith('.jpg')])[:12]:
            n_total += 1
            fb = np.fromfile(os.path.join(p, fname), dtype=np.uint8)
            frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if frame is None: continue
            face = process_frame_lite(frame)
            if face is None: continue
            out_name = fname.replace('.jpg', '_face.jpg')
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(os.path.join(out_p, out_name))
                n_kept += 1
    print(f'  {cat}: kept {n_kept}/{n_total} frames')

# Evaluate
ev_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe_fn(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab); l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe_fn(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid

patients = []
for label, cat in [(0, 'normal'), (1, 'jaundice')]:
    cat_dir = os.path.join(EXT_LITE, cat)
    if not os.path.isdir(cat_dir): continue
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if f.endswith('_face.jpg')])
        if faces: patients.append({'pid': pid, 'label': label, 'faces': faces[:12]})

items = [(fp, p['label'], p['pid']) for p in patients for fp in p['faces']]
print(f'\nV4-lite external: {len(patients)} patients, {len(items)} images')

ARCHS = [('convnext', 'convnext_tiny', 'final_binary_face_convnext.pt'),
         ('vit', 'vit_tiny_patch16_224', 'final_binary_face_vit.pt'),
         ('swin', 'swin_tiny_patch4_window7_224', 'final_binary_face_swin.pt'),
         ('efficientnet', 'efficientnet_b0', 'final_binary_face_efficientnet.pt')]

all_probs = {}
for name, bb, ckpt in ARCHS:
    m = timm.create_model(bb, pretrained=False, num_classes=2).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, ckpt), map_location=DEV))
    m.eval()
    dl = DataLoader(DS(items), BS, shuffle=False, num_workers=0)
    d = {}
    with torch.no_grad():
        for im, la, pi in dl:
            o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
            for p, l, pid in zip(o, la.numpy(), pi):
                d.setdefault(pid, ([], int(l))); d[pid][0].append(p)
    del m; torch.cuda.empty_cache()
    all_probs[name] = {pid: (np.mean(v[0], 0), v[1]) for pid, v in d.items()}
    t_arr = np.array([v[1] for v in all_probs[name].values()])
    p_arr = np.array([float(v[0][1]) for v in all_probs[name].values()])
    auc = roc_auc_score(t_arr, p_arr) if len(np.unique(t_arr)) > 1 else 0
    print(f'  {name:<15} AUC = {auc:.4f}')

# Ensemble
ens = {}
for pid in all_probs['convnext']:
    ps = [all_probs[a][pid][0] for a in ['convnext','vit','swin','efficientnet'] if pid in all_probs[a]]
    if len(ps) == 4: ens[pid] = (np.mean(ps, 0), all_probs['convnext'][pid][1])

t = np.array([v[1] for v in ens.values()]); p = np.array([v[0][1] for v in ens.values()])
auc_ens = roc_auc_score(t, p)
ap_ens = average_precision_score(t, p)

rng = np.random.RandomState(42); auc_b = []
for _ in range(1000):
    idx = rng.randint(0, len(t), len(t))
    if len(np.unique(t[idx])) < 2: continue
    auc_b.append(roc_auc_score(t[idx], p[idx]))

print(f'\n=== V4-LITE EXTERNAL ENSEMBLE ===')
print(f'  N={len(ens)} AUC={auc_ens:.4f} [{np.percentile(auc_b, 2.5):.4f}-{np.percentile(auc_b, 97.5):.4f}] AP={ap_ens:.4f}')

# Also try Swin-only (best from previous experiment)
swin_probs = all_probs['swin']
t_s = np.array([v[1] for v in swin_probs.values()]); p_s = np.array([v[0][1] for v in swin_probs.values()])
auc_swin = roc_auc_score(t_s, p_s)
print(f'  Swin-only: AUC={auc_swin:.4f}')

# Top-2 (Swin+ViT)
top2 = {}
for pid in all_probs['swin']:
    if pid in all_probs['vit']:
        top2[pid] = ((all_probs['swin'][pid][0] + all_probs['vit'][pid][0]) / 2, all_probs['swin'][pid][1])
t2 = np.array([v[1] for v in top2.values()]); p2 = np.array([v[0][1] for v in top2.values()])
print(f'  Swin+ViT:  AUC={roc_auc_score(t2, p2):.4f}')

print(f'\n=== COMPARISON (all external evaluations) ===')
print(f'  v3 original + Ensemble: 0.6444')
print(f'  v3 original + Swin:     0.7385')
print(f'  v4 full + Ensemble:     0.5564')
print(f'  v4-lite + Ensemble:     {auc_ens:.4f}')
print(f'  v4-lite + Swin:         {auc_swin:.4f}')

result = {'pipeline': 'v4-lite (YuNet bbox crop, no SAM/alignment)', 'n': len(ens),
          'ensemble_auc': float(auc_ens), 'swin_auc': float(auc_swin)}
with open(os.path.join(RES, 'external_v4_lite.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
