# -*- coding: utf-8 -*-
"""
Figure 4C v4 — rigorous CAM gallery.
Every eyelid panel must have a visible, eye-focused heatmap.
Multi-arch candidate pool: for each task, try multiple eyelid archs,
compute eye-focus for each candidate, pick best (focus>=1.5 AND correct)
OR best available if none satisfy. IBIL: convnext/vit/swin/effnet.
Rows without acceptable eyelid CAM show face only.
"""
import os, re, json, random, math, shutil, copy
import numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
MASK_C = (18 / 255.0, 18 / 255.0, 18 / 255.0)

# ── eyelid iris-bar geometry fixed by the USER (measured from their edited
#    artwork in 224px image coords: [x0, y0, x1, y1]). These override the
#    automatic dark-blob detection so the bars stay exactly where the user
#    placed them. Keys match TASKS['key'].
EYELID_BAR_OVERRIDES = {
    'Binary':  [80.3, 123.4, 195.7, 174.3],
    'Ternary': [95.1, 167.4, 201.5, 218.3],
    'DBIL':    [46.0, 121.9, 192.9, 186.2],
    'IBIL':    [71.6, 77.9, 191.4, 127.7],
    'Type':    [72.7, 118.3, 210.2, 168.9],
    'CP':      [52.9, 107.9, 207.3, 158.7],
    'MELD':    [68.0, 116.2, 203.3, 167.0],
}
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm

BASE = r'D:\research\人脸识别营养\传染科'
FACE_DIR = os.path.join(BASE, 'data', 'face_v4')
TYPE_DIR = os.path.join(BASE, 'data', 'type_v2')
MODEL_DIR = os.path.join(BASE, 'models')
MANIFEST = os.path.join(BASE, 'data', 'clean_dataset_manifest.csv')
RES = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
TEMP_DIR = r'C:\Users\o\AppData\Local\Temp\opencode'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMG_SIZE = 224
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.5,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                     'svg.fonttype': 'none'})
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ── utils ──
def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None
def clahe_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

_yunet = None
def get_yunet():
    global _yunet
    if _yunet is None:
        yunet_tmp = os.path.join(TEMP_DIR, 'yunet.onnx')
        if not os.path.exists(yunet_tmp):
            shutil.copy2(os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx'), yunet_tmp)
        _yunet = cv2.FaceDetectorYN.create(yunet_tmp, "", (640, 480),
                                           score_threshold=0.3, nms_threshold=0.3, top_k=5000)
    return _yunet

def eye_mask(img_rgb):
    h, w = img_rgb.shape[:2]
    det = get_yunet()
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is not None and len(faces):
        f = faces[np.argmax(faces[:, -1])]
        lm = f[4:14].reshape(5, 2); le, re = lm[0], lm[1]
        cx, cy = (le[0] + re[0]) / 2, (le[1] + re[1]) / 2
        r = max(abs(le[0] - re[0]) * 1.6, 40)
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
        if mask.sum() >= 200: return mask
    cx, cy = int(w * 0.4), int(h * 0.4)
    mask = np.zeros((h, w), dtype=bool); mask[cy:3*cy, cx:3*cx] = True
    return mask

class GradCAM:
    def __init__(self, model, bb):
        self.model = model; self.bb = bb; self.gradients = self.activations = None
        if bb == 'convnext_tiny': target = model.stages[3]
        elif bb == 'efficientnet_b0': target = model.blocks[6]
        elif bb == 'vit_tiny_patch16_224': target = model.blocks[11]
        elif bb == 'swin_tiny_patch4_window7_224': target = model.layers[3]
        else: target = None
        if target:
            target.register_forward_hook(self._fwd)
            target.register_full_backward_hook(self._bwd)
    def _fwd(self, m, i, o): self.activations = o
    def _bwd(self, m, gi, go): self.gradients = go[0]
    def _to4d(self, t):
        if t.dim() == 4: return t
        if t.dim() == 3:
            b, n, c = t.shape
            if self.bb == 'vit_tiny_patch16_224': t = t[:, 1:, :]
            s = int(math.isqrt(n))
            if s * s == n: return t.transpose(1, 2).reshape(b, c, s, s)
        return None
    def generate(self, tensor, tc=None):
        self.model.zero_grad(); self.activations = None; self.gradients = None
        out = self.model(tensor)
        if tc is None: tc = out.argmax(1).item()
        self.model.zero_grad(); out[0, tc].backward()
        if self.gradients is None or self.activations is None: return None
        a, g = self._to4d(self.activations), self._to4d(self.gradients)
        if a is None or g is None: return None
        w = g.mean((2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def load_model(bb, ck, nc):
    p = os.path.join(MODEL_DIR, ck)
    if not os.path.exists(p): return None, None
    m = timm.create_model(bb, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(p, map_location=DEV, weights_only=True)); m.eval()
    return m, GradCAM(m, bb)

def eval_cam(model, cam_g, img):
    img224 = cv2.resize(clahe_rgb(img), (IMG_SIZE, IMG_SIZE))
    tensor = tf(Image.fromarray(img224).convert('RGB')).unsqueeze(0).to(DEV)
    return img224, cam_g.generate(tensor) if cam_g else None

# ── de-identification: eye-bar mask geometry (YuNet landmarks) ──
# returns (x0, y0, x1, y1) in image pixel coords; the mask is drawn later as a
# separate VECTOR rectangle so it stays editable in the SVG.
def eye_bar_geom(img):
    h, w = img.shape[:2]
    det = get_yunet()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is not None and len(faces):
        f = faces[np.argmax(faces[:, -1])]
        lm = f[4:14].reshape(5, 2)
        le, re = lm[0], lm[1]
        d = float(np.hypot(le[0] - re[0], le[1] - re[1]))
        if d < 10: d = w * 0.35
        x0 = int(min(le[0], re[0]) - d * 0.55); x1 = int(max(le[0], re[0]) + d * 0.55)
        yc = (le[1] + re[1]) / 2
        y0 = int(yc - d * 0.38); y1 = int(yc + d * 0.38)
    else:
        x0, x1 = int(w * 0.08), int(w * 0.92)
        y0, y1 = int(h * 0.32), int(h * 0.56)
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))

# ── de-identification: iris bar geometry for eyelid close-ups (dark round blob) ──
# returns (x0, y0, x1, y1) in image pixel coords; the bar is drawn later as a
# separate VECTOR rectangle so it stays editable in the SVG.
def iris_bar_geom(img, verbose=False):
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, 5)
    yc = None
    thr = np.percentile(g, 10)
    m = (g <= thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    best, bs = None, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if bw == 0 or bh == 0 or area < 0.004 * h * w:
            continue
        aspect = bw / bh
        cx, cy = cent[i]
        if not (0.6 <= aspect <= 2.2):
            continue
        if not (0.15 * w <= cx <= 0.85 * w):
            continue
        if not (0.10 * h <= cy <= 0.90 * h):
            continue
        mean_b = float(g[lab == i].mean())
        # iris/pupil = the round dark blob nearest the crop centre
        score = -(abs(cy / h - 0.5) + 0.5 * abs(cx / w - 0.5)) + 0.5 * (area / (h * w))
        if verbose:
            print('    blob area=%.3f aspect=%.2f cx=%.2f cy=%.2f bright=%.0f score=%.3f' %
                  (area / (h * w), aspect, cx / w, cy / h, mean_b, score))
        if score > bs:
            bs, best = score, cy
    if best is not None:
        yc = float(best)
    else:
        prof = g.astype(np.float32).mean(axis=1)
        k = max(3, h // 15)
        prof_s = np.convolve(prof, np.ones(k, np.float32) / k, mode='same')
        lo, hi = int(h * 0.18), int(h * 0.82)
        yc = float(lo + int(np.argmin(prof_s[lo:hi])))
        if verbose:
            print('    fallback dark-row y=%.2f' % (yc / h))
    y0 = int(yc - h * 0.12); y1 = int(yc + h * 0.12)
    return (0, max(0, y0), w, min(h, y1))

mdf = pd.read_csv(MANIFEST)
EYE_MAP = {}
for _, r in mdf.iterrows():
    if r['n_eyelid'] > 0:
        EYE_MAP[r['patient_id']] = json.loads(r['eyelid_images'])
V2 = json.load(open(os.path.join(RES, 'canonical_v2_probs.json'), encoding='utf-8'))
T2 = json.load(open(os.path.join(RES, 'typev2_probs.json'), encoding='utf-8'))

def face_path_of(pid):
    for cat in ['normal', 'mild', 'moderate', 'severe']:
        d = os.path.join(FACE_DIR, cat, pid)
        if os.path.isdir(d):
            fs = sorted([f for f in os.listdir(d) if f.endswith('_face.jpg')])
            if fs: return os.path.join(d, fs[len(fs)//2])
    return None

def eyelid_path_of(pid):
    for p in EYE_MAP.get(pid, []):
        if os.path.exists(p): return p
    for cat in ['mild', 'moderate', 'severe', 'normal']:
        base = os.path.join(BASE, 'data', 'extracted', cat, pid)
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    if f.lower().endswith(('.jpg','.jpeg')) and 'feature' not in f.lower() and os.path.getsize(fp)>1000000:
                        return fp
    return None

# ── task definitions ──
ALL_ARCH = [('convnext_tiny', 'convnext'), ('vit_tiny_patch16_224', 'vit'),
            ('swin_tiny_patch4_window7_224', 'swin'), ('efficientnet_b0', 'efficientnet')]

def pool_v2_probs(task, scope, cls):
    d = V2[task][scope]
    best = {}
    for arch, pds in d.items():
        if arch.startswith('__'): continue
        for pid, (prob, lab) in pds.items():
            if lab == cls: best[pid] = max(best.get(pid, 0), float(prob[cls]))
    return sorted(best.items(), key=lambda x: -x[1])

TASKS = [
    dict(key='Binary', name='Binary\nScreening', cls=1,
         face_arch='convnext_tiny', face_ck='final_binary_face_convnext.pt', face_nc=2,
         eye_pool=[(pid, 1.0) for pid, v in json.load(open(os.path.join(RES, 'fusion_shared_probs.json'), encoding='utf-8'))['Binary Screening']['eyelid']['convnext'].items() if v[1] == 1],
         eye_archs=[('convnext_tiny', 'eyelid_binary_fusion_convnext.pt')]),
    dict(key='Ternary', name='Ternary\nGrading', cls=2,
         face_arch='convnext_tiny', face_ck='v3_ternary_convnext_tiny.pt', face_nc=3,
         eye_pool=pool_v2_probs('Ternary Grading', 'eyelid', 2),
         eye_archs=[('convnext_tiny', 'eyelid_opt_convnext_tiny.pt')]),
    dict(key='DBIL', name='DBIL', cls=2,
         face_arch='convnext_tiny', face_ck='dbil_face_convnext.pt', face_nc=3,
         eye_pool=pool_v2_probs('DBIL', 'eyelid', 2),
         eye_archs=[('convnext_tiny', 'dbil_eyelid_convnext.pt')]),
    dict(key='IBIL', name='IBIL', cls=2,
         face_arch='convnext_tiny', face_ck='ibil_face_convnext.pt', face_nc=3,
         eye_pool=pool_v2_probs('IBIL', 'eyelid', 2),
         eye_archs=[(bb, p)for bb, p in [('convnext_tiny','ibil_eyelid_convnext.pt'),('vit_tiny_patch16_224','ibil_eyelid_vit.pt'),('swin_tiny_patch4_window7_224','ibil_eyelid_swin.pt'),('efficientnet_b0','ibil_eyelid_efficientnet.pt')]]),
    dict(key='Type', name='Jaundice\nType', cls=1,
         face_arch='convnext_tiny', face_ck='typev2_face_convnext.pt', face_nc=2,
         eye_pool=[(pid, float(v[0][1])) for pid, v in T2['eyelid']['convnext'].items() if v[1]==1],
         eye_archs=[(bb, p)for bb, p in [('convnext_tiny','typev2_eyelid_convnext.pt'),('vit_tiny_patch16_224','typev2_eyelid_vit.pt'),('swin_tiny_patch4_window7_224','typev2_eyelid_swin.pt'),('efficientnet_b0','typev2_eyelid_efficientnet.pt')]]),
    dict(key='CP', name='Child-\nPugh', cls=2,
         face_arch='convnext_tiny', face_ck='cp_face_convnext.pt', face_nc=3,
         eye_pool=pool_v2_probs('Child-Pugh', 'eyelid', 2),
         eye_archs=[(bb, p)for bb, p in [('convnext_tiny','cp_eyelid_convnext.pt'),('vit_tiny_patch16_224','cp_eyelid_vit.pt'),('swin_tiny_patch4_window7_224','cp_eyelid_swin.pt'),('efficientnet_b0','cp_eyelid_efficientnet.pt')]]),
    dict(key='MELD', name='MELD', cls=2,
         face_arch='convnext_tiny', face_ck='meld_face_convnext.pt', face_nc=3,
         eye_pool=pool_v2_probs('MELD', 'eyelid', 2),
         eye_archs=[('convnext_tiny', 'meld_eyelid_convnext.pt')]),
]

used_pids = set()
for t in TASKS:
    t['face_model'], t['face_cam'] = load_model(t['face_arch'], t['face_ck'], t['face_nc'])
    # try each eye arch until we find a good combo
    t['eye_model'] = t['eye_cam'] = None; t['eye_ckpt'] = None; t['chosen'] = None
    for min_focus in [1.5, 1.0]:
        for ea_bb, ea_ck in t['eye_archs']:
            model, cam_g = load_model(ea_bb, ea_ck, max(2, t['face_nc']))
            if model is None: continue
            best_c = None
            for pid, conf in t['eye_pool'][:25]:
                if pid in used_pids: continue
                fp = face_path_of(pid)
                ep = eyelid_path_of(pid)
                if t['key']=='Type':
                    ep2 = os.path.join(TYPE_DIR, 'eyelids', pid, 'eyelid.jpg')
                    if os.path.exists(ep2): ep = ep2
                    d = os.path.join(TYPE_DIR, 'faces', pid)
                    if os.path.isdir(d):
                        fs = sorted([f for f in os.listdir(d) if f.endswith('_face.jpg')])
                        if fs: fp = os.path.join(d, fs[len(fs)//2])
                if not fp or not ep: continue
                img_e = read_img(ep)
                if img_e is None: continue
                img224_e, cam_e = eval_cam(model, cam_g, img_e)
                if cam_e is None: continue
                m = eye_mask(img224_e)
                fs = float(cam_e[m].mean() / (cam_e.mean() + 1e-8)) if m.sum()>=50 else 0.0
                with torch.no_grad():
                    pred = model(tf(Image.fromarray(img224_e).convert('RGB')).unsqueeze(0).to(DEV)).argmax(1).item()
                ok = pred == t['cls']
                score = (ok * 5 + min(fs, 5))
                if best_c is None or score > best_c['score']:
                    best_c = {'pid': pid, 'conf': conf, 'focus': fs, 'pred_ok': ok,
                              'fp': fp, 'ep': ep, 'img_e': img_e, 'cam_e': cam_e, 'img224_e': img224_e,
                              'score': score}
            if best_c and best_c['focus'] >= min_focus:
                t['eye_model'] = model; t['eye_cam'] = cam_g; t['eye_ckpt'] = ea_ck
                used_pids.add(best_c['pid']); t['chosen'] = best_c
                break  # break arch loop
        if t['chosen']: break  # break min_focus loop
    # face image
    if t['chosen'] is None and t['eye_pool']:
        for pid_, _ in t['eye_pool']:
            if pid_ not in used_pids:
                fp_ = face_path_of(pid_)
                if fp_:
                    t['chosen'] = {'pid': pid_, 'fp': fp_, 'ep': None, 'cam_e': None, 'focus': 0, 'img224_e': None}
                    used_pids.add(pid_); break
    if t['chosen']:
        fp = t['chosen'].get('fp')
        if fp:
            img_f = read_img(fp)
            t['img224_f'], t['cam_f'] = eval_cam(t['face_model'], t['face_cam'], img_f) if t['face_cam'] else (None, None)
        else:
            t['img224_f'] = t['cam_f'] = None
    else:
        t['img224_f'] = t['cam_f'] = None
    print(f"  {t['key']}: face_cam={'OK' if t['cam_f'] is not None else 'X'}   "
          f"eye_cam={'OK' if t['chosen'] and t['chosen'].get('cam_e') is not None else 'X'} ({t['eye_ckpt'] or 'none'})   "
          f"patient={t['chosen']['pid'][:26] if t['chosen'] else 'NONE'}   "
          + (f"focus={t['chosen']['focus']:.2f}" if t['chosen'] and t['chosen'].get('cam_e') is not None else f"no eye CAM"))

# ── render ──
rows = [t for t in TASKS if t.get('chosen')]
fig, axes = plt.subplots(len(rows), 5, figsize=(12, 2.1 * len(rows)))
for col, label in enumerate(['Task', 'Face Original', 'Face Grad-CAM', 'Eyelid Original', 'Eyelid Grad-CAM']):
    axes[0, col].set_title(label, fontsize=15, fontweight='bold', pad=5)
for row, t in enumerate(rows):
    ch = t['chosen']
    axes[row,0].text(0.5, 0.5, t['name'].replace('\n', ' '), ha='center', va='center',
                     fontsize=16, fontweight='bold', transform=axes[row,0].transAxes)
    axes[row,0].axis('off')
    if t['img224_f'] is not None:
        imgf = t['img224_f']                      # raw image; mask drawn as vector below
        x0, y0, x1, y1 = eye_bar_geom(imgf)
        axes[row,1].imshow(imgf, zorder=1)
        axes[row,1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=MASK_C,
                                        edgecolor='none', zorder=2))
        axes[row,1].axis('off')
        axes[row,2].imshow(imgf, zorder=1)
        axes[row,2].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=MASK_C,
                                        edgecolor='none', zorder=2))
        if t['cam_f'] is not None:
            axes[row,2].imshow(t['cam_f'], cmap='jet', alpha=0.4, zorder=3)
        axes[row,2].axis('off')
    else:
        axes[row,1].axis('off'); axes[row,2].text(0.5,0.5,'No face', ha='center',va='center', transform=axes[row,2].transAxes)
    if ch.get('cam_e') is not None:
        imge = ch['img224_e']                     # raw image; mask drawn as vector below
        _ov = EYELID_BAR_OVERRIDES.get(t['key'])
        ex0, ey0, ex1, ey1 = _ov if _ov else iris_bar_geom(imge)
        axes[row,3].imshow(imge, zorder=1)
        axes[row,3].add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, facecolor=MASK_C,
                                        edgecolor='none', zorder=2))
        axes[row,3].axis('off')
        axes[row,4].imshow(imge, zorder=1)
        axes[row,4].add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, facecolor=MASK_C,
                                        edgecolor='none', zorder=2))
        axes[row,4].imshow(ch['cam_e'], cmap='jet', alpha=0.4, zorder=3)
        axes[row,4].text(4, 218, f"focus={ch['focus']:.2f}", fontsize=5.5, color='white',
                         bbox=dict(fill=True, color='black', alpha=0.6, pad=1))
        axes[row,4].axis('off')
    else:
        axes[row,3].axis('off')
        axes[row,4].text(0.5, 0.5, 'No meaningful\nCAM for eyelid', ha='center', va='center',
                         transform=axes[row,4].transAxes, fontsize=6, color='gray'); axes[row,4].axis('off')

plt.tight_layout(rect=[0, 0, 1, 1])

def _save_svg_robust(fg, final_path, dpi):
    """Write the SVG to a temp file, verify it, then copy into place (retry loop),
    because large SVG saves occasionally get zero-filled on this machine."""
    import time, shutil, tempfile
    import xml.etree.ElementTree as ET
    tmpdir = tempfile.gettempdir()
    for k in range(6):
        tmp = os.path.join(tmpdir, 'fig4_gallery_tmp.svg')
        fg.savefig(tmp, dpi=dpi)
        raw = open(tmp, 'rb').read()
        ok = raw.count(b'\x00') == 0
        if ok:
            try:
                ET.fromstring(raw)
            except Exception:
                ok = False
        if ok:
            shutil.copy2(tmp, final_path)
            if open(final_path, 'rb').read() == raw:
                os.remove(tmp)
                return True
        time.sleep(0.4)
    return False

out_png = os.path.join(FIG_DIR, 'Figure4_gallery_crm.png')
out_svg = os.path.join(FIG_DIR, 'Figure4_gallery_crm.svg')
fig.savefig(out_png, dpi=300)
print('SVG robust save:', _save_svg_robust(fig, out_svg, 200))
plt.close(fig)
print('\nSaved to', FIG_DIR)
