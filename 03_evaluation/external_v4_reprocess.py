# -*- coding: utf-8 -*-
"""
Reprocess external validation frames with YuNet+SAM v4 pipeline,
then evaluate canonical-v2 models on the v4-processed external images.
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
EXT_V4 = os.path.join(BASE, 'data', 'external_v4_processed')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
TEMP_DIR = r'C:\Users\o\AppData\Local\Temp\opencode'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224; BS = 32
DET_SCORE_MIN = 0.50; SKIN_FRACTION_MIN = 0.35

# YuNet setup
yunet_tmp = os.path.join(TEMP_DIR, 'yunet.onnx')
if not os.path.exists(yunet_tmp):
    shutil.copy2(os.path.join(MODEL, 'face_detection_yunet_2023mar.onnx'), yunet_tmp)
detector = cv2.FaceDetectorYN.create(yunet_tmp, "", (640, 480),
                                     score_threshold=DET_SCORE_MIN, nms_threshold=0.3, top_k=5000)

# SAM setup
from transformers import SamModel, SamProcessor
print('Loading SAM...')
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEV)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model.eval()

def skin_fraction(img_rgb):
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    cb = ycbcr[:,:,2].astype(int); cr = ycbcr[:,:,1].astype(int)
    return ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).mean()

def apply_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def align_face(img_rgb, landmarks_5, target_size=224):
    scale = target_size / 112.0
    src = np.array([[30.2946,51.6963],[65.5318,51.5014],[48.0252,71.7366],
                    [33.5493,92.3655],[62.7299,92.2041]], dtype=np.float32) * scale
    dst = np.array(landmarks_5, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC)
    if M is None: return cv2.resize(img_rgb, (target_size, target_size))
    return cv2.warpAffine(img_rgb, M, (target_size, target_size),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    if faces is None or len(faces) == 0: return None
    face = faces[np.argmax(faces[:, -1])]
    x, y, fw, fh = face[:4].astype(np.float32)
    landmarks = face[4:14].reshape(5, 2).copy()
    score = float(face[14])
    if score < DET_SCORE_MIN: return None
    if fw * fh < 40*40: return None
    pw, ph = fw * 0.15, fh * 0.15
    x1, y1 = max(0, int(x - pw)), max(0, int(y - ph))
    x2, y2 = min(w, int(x + fw + pw)), min(h, int(y + fh + ph))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        inputs = sam_processor(Image.fromarray(frame_rgb), input_boxes=[[[x1, y1, x2, y2]]],
                               return_tensors='pt').to(DEV)
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu())
        sam_mask = masks[0][0][0].numpy()
    except Exception:
        sam_mask = np.zeros((h, w), dtype=bool); sam_mask[y1:y2, x1:x2] = True
    if sam_mask.any():
        rows = np.any(sam_mask, axis=1); cols = np.any(sam_mask, axis=0)
        ys = np.where(rows)[0]; xs = np.where(cols)[0]
        rmin, rmax = max(0, ys[0]-3), min(h, ys[-1]+3)
        cmin, cmax = max(0, xs[0]-3), min(w, xs[-1]+3)
    else:
        rmin, rmax, cmin, cmax = y1, y2, x1, x2
    crop = frame_rgb[rmin:rmax, cmin:cmax].copy()
    if crop.size == 0: return None
    if skin_fraction(crop) < SKIN_FRACTION_MIN: return None
    lm = landmarks.copy(); lm[:, 0] -= cmin; lm[:, 1] -= rmin
    return apply_clahe(align_face(crop, lm, IMG_SIZE))

# Process all external patients
print('Processing external frames with v4 pipeline...')
os.makedirs(EXT_V4, exist_ok=True)
stats = {'total': 0, 'kept': 0, 'no_face': 0, 'low_skin': 0}
for label, cat in [(0, 'normal'), (1, 'jaundice')]:
    cat_dir = os.path.join(EXT_RAW, cat)
    out_cat = os.path.join(EXT_V4, cat)
    os.makedirs(out_cat, exist_ok=True)
    for pid in sorted(os.listdir(cat_dir)):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p): continue
        out_p = os.path.join(out_cat, pid)
        os.makedirs(out_p, exist_ok=True)
        # Skip if already processed
        existing = [f for f in os.listdir(out_p) if f.endswith('_face.jpg')] if os.path.isdir(out_p) else []
        if len(existing) >= 3:
            stats['kept'] += len(existing); continue
        frames = sorted([f for f in os.listdir(p) if f.endswith('.jpg')])
        n_kept = 0
        for fname in frames[:12]:
            stats['total'] += 1
            fb = np.fromfile(os.path.join(p, fname), dtype=np.uint8)
            frame = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if frame is None: continue
            face = process_frame(frame)
            if face is None:
                stats['no_face'] += 1; continue
            out_name = fname.replace('.jpg', '_face.jpg')
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(os.path.join(out_p, out_name))
                n_kept += 1; stats['kept'] += 1
            if n_kept >= 12: break
    print(f'  {cat}: done')

print(f'V4 processing: kept {stats["kept"]}/{stats["total"]} frames (no_face={stats["no_face"]})')

# Now evaluate canonical-v2 models on v4-processed external images
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
    cat_dir = os.path.join(EXT_V4, cat)
    if not os.path.isdir(cat_dir): continue
    for pid in os.listdir(cat_dir):
        p = os.path.join(cat_dir, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if f.endswith('_face.jpg')])
        if faces: patients.append({'pid': pid, 'label': label, 'faces': faces[:12]})

items = [(fp, p['label'], p['pid']) for p in patients for fp in p['faces']]
print(f'\nV4-processed external: {len(patients)} patients, {len(items)} images')

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

# Individual AUCs
print('\n=== V4-PROCESSED EXTERNAL: Individual backbones ===')
for name in all_probs:
    t = np.array([v[1] for v in all_probs[name].values()])
    p = np.array([v[0][1] for v in all_probs[name].values()])
    auc = roc_auc_score(t, p) if len(np.unique(t)) > 1 else float('nan')
    print(f'  {name:<15} AUC = {auc:.4f}')

# Ensemble
ens = {}
for pid in all_probs['convnext']:
    ps = [all_probs[a][pid][0] for a in ['convnext','vit','swin','efficientnet'] if pid in all_probs[a]]
    if len(ps) == 4: ens[pid] = (np.mean(ps, 0), all_probs['convnext'][pid][1])

t = np.array([v[1] for v in ens.values()])
p = np.array([v[0][1] for v in ens.values()])
auc_ens = roc_auc_score(t, p)
ap_ens = average_precision_score(t, p)

# Bootstrap CI
rng = np.random.RandomState(42)
auc_b = []
for _ in range(1000):
    idx = rng.randint(0, len(t), len(t))
    if len(np.unique(t[idx])) < 2: continue
    auc_b.append(roc_auc_score(t[idx], p[idx]))

print(f'\n=== V4-PROCESSED EXTERNAL ENSEMBLE ===')
print(f'  N={len(ens)} (normal={int((t==0).sum())}, jaundice={int((t==1).sum())})')
print(f'  AUC = {auc_ens:.4f} [{np.percentile(auc_b, 2.5):.4f}-{np.percentile(auc_b, 97.5):.4f}]')
print(f'  AP  = {ap_ens:.4f}')

result = {
    'pipeline': 'v4 (YuNet+SAM) reprocessed external',
    'n': len(ens), 'auc': float(auc_ens),
    'auc_lo': float(np.percentile(auc_b, 2.5)), 'auc_hi': float(np.percentile(auc_b, 97.5)),
    'ap': float(ap_ens),
    'individual_aucs': {a: float(roc_auc_score(np.array([v[1] for v in all_probs[a].values()]),
                                                np.array([v[0][1] for v in all_probs[a].values()])))
                        for a in all_probs},
}
with open(os.path.join(RES, 'external_validation_v4_reprocessed.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f'\nSaved: results/external_validation_v4_reprocessed.json')
