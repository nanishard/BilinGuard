# -*- coding: utf-8 -*-
"""
Evaluate ALL eyelid binary models (convnext/vit/effnet/swin) on a reconstructed
val split, to fill in the missing ConvNeXt AUC and validate consistency with
the already-reported sibling AUCs (ViT 0.6525 / Swin 0.8700 / EffNet 0.4104).

Dataset reconstruction:
  - Jaundice: 217 patients from clean_dataset_manifest.csv (n_eyelid>0, first image)
  - Normal:   patients from data/extracted/normal with IMG*.jpg >1MB
              (same scan logic as train_eyelid_meld_type.py normal controls)
Split variants tested:
  A) whole-list shuffle, seed 42, first 20% = val   (train_binary_fix.py style)
  B) per-label stratified shuffle, seed 42, 20% val (train_eyelid_meld_type.py style)
"""
import os, json, random, numpy as np, pandas as pd
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import cv2

BASE = r'D:\research\人脸识别营养\传染科'
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
IMG_SIZE = 224; BS = 32


def load_img(path):
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


class DS(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label, pid = self.items[i]
        img = load_img(path)
        if img is None:
            img = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
        img = clahe(img)
        return ev_tf(Image.fromarray(img).convert('RGB')), label, pid


def build_items():
    items = []
    df = pd.read_csv(MANIFEST)
    df = df[df['n_eyelid'] > 0]
    for _, r in df.iterrows():
        imgs = json.loads(r['eyelid_images'])
        for p in imgs[:1]:
            items.append((p, 0 if r['category'] == 'normal' else 1, r['patient_id']))
    n_jau = sum(1 for _, l, _ in items if l == 1)

    normal_ext = os.path.join(BASE, 'data', 'extracted', 'normal')
    n_norm = 0
    for pid in sorted(os.listdir(normal_ext)):
        pp = os.path.join(normal_ext, pid)
        if not os.path.isdir(pp): continue
        cand = None
        for root, dirs, files in os.walk(pp):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg')) and 'feature' not in f.lower() and f.startswith('IMG'):
                    fp = os.path.join(root, f)
                    if os.path.getsize(fp) > 1000000:
                        cand = fp
                        break
            if cand: break
        if cand:
            items.append((cand, 0, pid))
            n_norm += 1
    print(f'  Dataset: jaundice={n_jau}, normal={n_norm}, total={len(items)}')
    return items


def split_whole(items, ratio=0.2):
    items = list(items)
    random.seed(SEED); random.shuffle(items)
    n = max(1, int(len(items) * ratio))
    return items[n:], items[:n]


def split_strat(items, ratio=0.2):
    by_label = {}
    for it in items: by_label.setdefault(it[1], []).append(it)
    tr, va = [], []
    random.seed(SEED)
    for label, its in by_label.items():
        its = list(its)
        random.shuffle(its)
        n = max(1, int(len(its) * ratio))
        va.extend(its[:n]); tr.extend(its[n:])
    return tr, va


@torch.no_grad()
def evaluate(name, backbone, va_items):
    m = timm.create_model(backbone, pretrained=False, num_classes=2).to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODEL, name), map_location=DEV))
    m.eval()
    vrl = DataLoader(DS(va_items), BS, shuffle=False, num_workers=0)
    ps, ls, pis = [], [], []
    for im, la, pi in vrl:
        o = m(im.to(DEV))
        ps.extend(F.softmax(o, 1).cpu().numpy()); ls.extend(la.numpy()); pis.extend(pi)
    d = {}
    for p, l, pi in zip(ps, ls, pis):
        d.setdefault(pi, ([], l)); d[pi][0].append(p)
    pids = list(d.keys())
    avg = np.array([np.mean(d[p][0], 0) for p in pids])
    true = np.array([d[p][1] for p in pids])
    if len(np.unique(true)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(true, avg[:, 1])
    acc = accuracy_score(true, avg.argmax(1))
    del m; torch.cuda.empty_cache()
    return auc, acc, len(pids), true, avg[:, 1], pids


MODELS = [
    ('eyelid_binary_convnext.pt', 'convnext_tiny', 'ConvNeXt', 0.0),     # AUC missing
    ('eyelid_binary_vit_tiny_patch16_224.pt', 'vit_tiny_patch16_224', 'ViT', 0.6525),
    ('eyelid_binary_efficientnet_b0.pt', 'efficientnet_b0', 'EfficientNet', 0.4104),
    ('eyelid_binary_swin_tiny_patch4_window7_224.pt', 'swin_tiny_patch4_window7_224', 'Swin', 0.8700),
]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    items = build_items()

    results = {}
    for split_name, split_fn in [('A_whole', split_whole), ('B_strat', split_strat)]:
        tr, va = split_fn(items)
        n0 = sum(1 for _, l, _ in va if l == 0); n1 = sum(1 for _, l, _ in va if l == 1)
        print(f'\n=== Split {split_name}: val={len(va)} (normal={n0}, jaundice={n1}) ===')
        for fname, bb, arch, reported in MODELS:
            auc, acc, np_, true, prob, pids = evaluate(fname, bb, va)
            print(f'  {arch:<14} AUC={auc:.4f}  ACC={acc:.4f}  (reported={reported})')
            results.setdefault(arch, {})[split_name] = {'auc': float(auc), 'acc': float(acc), 'n_val': np_}
            if split_name == 'B_strat':
                results[arch]['val_detail'] = {
                    'pids': pids, 'true': [int(x) for x in true], 'prob': [float(x) for x in prob]}

    with open(os.path.join(RES, 'eval_eyelid_binary_all.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved: results/eval_eyelid_binary_all.json')


if __name__ == '__main__':
    main()
