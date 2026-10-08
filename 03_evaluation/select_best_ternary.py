# -*- coding: utf-8 -*-
"""
Select the best ternary model for the Severe class. Re-runs the candidate face
ternary backbones (ConvNeXt, ViT, Swin) on the external jaundice cohort
(n=57, TBIL-derived Mild/Moderate/Severe), testing CLAHE on/off to find the
matching preprocessing, and reports per-class metrics with Severe focus.
"""
import os, json
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, accuracy_score)
from sklearn.preprocessing import label_binarize

BASE = r'D:\research\人脸识别营养\传染科'
EXT_DIR = os.path.join(BASE, 'data', 'external_processed_v3', 'jaundice')
MODEL_DIR = os.path.join(BASE, 'models')
VIDEO_XLSX = os.path.join(BASE, 'external_validation_results',
                          'external_validation_results', 'video_level_results.xlsx')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG = 224; BS = 32
N = ['Mild', 'Moderate', 'Severe']

# ---- robust folder -> label map via video_level_results.xlsx ----
v = pd.read_excel(VIDEO_XLSX)
folder2info = {}
for _, r in v.iterrows():
    fn = str(r['folder_name']).strip()
    if fn and fn.lower() != 'nan':
        folder2info[fn] = (int(r['true_class']), float(r['bilirubin']))
print('folder->label map:', len(folder2info), 'folders from video_level_results')

# which folders physically exist
ext_folders = set(os.listdir(EXT_DIR))
matched = {f: info for f, info in folder2info.items() if f in ext_folders}
print('matched existing folders:', len(matched))
# also try substring match for any not exact
if len(matched) < len(folder2info):
    for f in list(folder2info):
        if f in matched:
            continue
        for ef in ext_folders:
            if f in ef or ef in f:
                matched[ef] = folder2info[f]; break
print('matched after substring:', len(matched))

# ---- image read ----
def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab); l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

class DS(Dataset):
    def __init__(self, items, use_clahe):
        self.items = items; self.use_clahe = use_clahe
        self.tf = transforms.Compose([
            transforms.Resize((IMG, IMG)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, folder = self.items[i]
        img = read_img(path)
        if img is None: img = np.full((IMG, IMG, 3), 128, dtype=np.uint8)
        if self.use_clahe: img = clahe(img)
        return self.tf(Image.fromarray(img).convert('RGB')), folder

# build items (up to 12 faces per matched folder)
items = []
for folder in matched:
    d = os.path.join(EXT_DIR, folder)
    if not os.path.isdir(d): continue
    faces = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('_face.jpg')])
    for fp in faces[:12]:
        items.append((fp, folder))
print('face-image items:', len(items), '\n')

def run(model_file, backbone, use_clahe):
    m = timm.create_model(backbone, pretrained=False, num_classes=3).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_file), map_location=DEV))
    m.eval()
    dl = DataLoader(DS(items, use_clahe), BS, shuffle=False, num_workers=0)
    agg = {}
    with torch.no_grad():
        for im, folder in dl:
            o = F.softmax(m(im.to(DEV)), 1).cpu().numpy()
            for p, f in zip(o, folder):
                agg.setdefault(f, []).append(p)
    del m
    if DEV == 'cuda': torch.cuda.empty_cache()
    # patient-level aggregate
    folders = sorted([f for f in matched if f in agg])
    true = np.array([matched[f][0] for f in folders])
    P = np.array([np.mean(agg[f], 0) for f in folders])
    return true, P

def metrics(true, P):
    yb = label_binarize(true, classes=[0, 1, 2])
    pred = P.argmax(1)
    aucs = [roc_auc_score(yb[:, c], P[:, c]) if len(np.unique(yb[:, c])) > 1 else np.nan for c in range(3)]
    out = {'macAUC': np.nanmean(aucs), 'acc': accuracy_score(true, pred),
           'macF1': f1_score(true, pred, average='macro')}
    for c in range(3):
        out['AUC_%s' % N[c]] = aucs[c]
        out['AP_%s' % N[c]] = average_precision_score(yb[:, c], P[:, c]) if len(np.unique(yb[:, c])) > 1 else np.nan
        out['F1_%s' % N[c]] = f1_score(true, pred, labels=[c], average='macro', zero_division=0)
        out['Rec_%s' % N[c]] = recall_score(true, pred, labels=[c], average='macro', zero_division=0)
    return out

cands = [
    ('face_ternary_convnext_tiny.pt', 'convnext_tiny'),
    ('face_ternary_vit_tiny_patch16_224.pt', 'vit_tiny_patch16_224'),
    ('face_ternary_swin_tiny_patch4_window7_224.pt', 'swin_tiny_patch4_window7_224'),
    ('face_ternary_sam_swin_tiny_patch4_window7_224.pt', 'swin_tiny_patch4_window7_224'),
    ('face_ternary_sam_convnext_tiny.pt', 'convnext_tiny'),
    ('face_ternary_sam_vit_tiny_patch16_224.pt', 'vit_tiny_patch16_224'),
]
print('%-40s %-7s %6s %6s | %6s %6s %6s %6s | %6s %6s' %
      ('model+prep', 'macAUC', 'acc', 'macF1', 'SevAUC', 'SevAP', 'SevF1', 'SevRec', 'ModAUC', 'MildAUC'))
results = {}
for mf, bb in cands:
    for cla in (False, True):
        true, P = run(mf, bb, cla)
        mt = metrics(true, P)
        tag = '%s [CLAHE=%s]' % (bb.replace('_tiny', '').replace('_patch16_224', '').replace('_patch4_window7_224', ''), cla)
        print('%-40s %-7s %6.3f %6.3f | %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f' %
              (tag, '%.3f' % mt['macAUC'], mt['acc'], mt['macF1'],
               mt['AUC_Severe'], mt['AP_Severe'], mt['F1_Severe'], mt['Rec_Severe'],
               mt['AUC_Moderate'], mt['AUC_Mild']))
        results[tag] = {'metrics': mt, 'true': true.tolist(), 'P': P.tolist()}

with open(os.path.join(BASE, 'results', 'ternary_model_selection_external.json'), 'w', encoding='utf-8') as f:
    json.dump({k: v['metrics'] for k, v in results.items()}, f, indent=2)
print('\nSaved: results/ternary_model_selection_external.json')
