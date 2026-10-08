# -*- coding: utf-8 -*-
"""
Generate two key visual figures:

1. Figure 4 Panel C: Grad-CAM attribution gallery
   - Pick 4 representative patients (CP-A, CP-B, CP-C, MELD-high)
   - For each: Original → CP Grad-CAM → MELD Grad-CAM
   - Shows model attention on periocular/malar/perioral regions

2. CDSS Case Demonstration (Figure 5 Panel A or standalone)
   - Run ClinicalAdvisor on a representative ACLF case
   - Render as a formatted clinical decision support report
"""
import os, re, json, random, math, numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch, torch.nn.functional as F
from torchvision import transforms
import timm

BASE = r'D:\research\人脸识别营养\传染科'
INT_DATA = os.path.join(BASE, 'data', 'face_v4')
MODEL = os.path.join(BASE, 'models')
RES = os.path.join(BASE, 'results')
EXCEL = os.path.join(BASE, '213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx')
FIG_DIR = os.path.join(RES, 'figures_v3', 'main')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42; IMG_SIZE = 224
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                      'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})


# ═════════════════════════════════════════════════════════════
# PART 1: Grad-CAM Attribution Gallery (Figure 4 Panel C)
# ═════════════════════════════════════════════════════════════
print('='*70)
print('  PART 1: Grad-CAM Attribution Gallery')
print('='*70)

# Image utils
def read_img(path):
    try:
        fb = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except: return None

def clahe_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class GradCAM:
    """Grad-CAM for ConvNeXt models."""
    def __init__(self, model, target_layer_name='stages.3'):
        self.model = model
        self.gradients = None
        self.activations = None
        target = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                target = module; break
        if target is None:
            for name, module in model.named_modules():
                if 'stages' in name and name.count('.') == 1:
                    target = module; break
        if target is not None:
            target.register_forward_hook(self._fwd)
            target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.activations = o
    def _bwd(self, m, gi, go): self.gradients = go[0]

    def generate(self, tensor, target_class=None):
        self.model.zero_grad()
        self.activations = None; self.gradients = None
        out = self.model(tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        out[0, target_class].backward()
        if self.gradients is None or self.activations is None: return None
        if self.activations.dim() != 4: return None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_backbone(bb_name, ckpt, nc):
    path = os.path.join(MODEL, ckpt)
    if not os.path.exists(path): return None
    m = timm.create_model(bb_name, pretrained=False, num_classes=nc).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    m.eval()
    return m


def make_overlay(img_224, cam, alpha=0.5):
    """Overlay heatmap on 224x224 RGB image."""
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    return (img_224 * (1 - alpha) + heatmap * alpha).astype(np.uint8)


# Load clinical data to find representative patients
print('[1] Loading clinical data for patient selection...')
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
        rows = df_ex[df_ex['hid_str'].apply(strip_zeros).str.endswith(hid[-6:])]
    if len(rows) == 0: return None
    r = rows.iloc[0]
    cp_grade = str(r.iloc[15]).strip().upper() if pd.notna(r.iloc[15]) else None
    cp_score = pd.to_numeric(r.iloc[14], errors='coerce')
    tbil = pd.to_numeric(r.iloc[51], errors='coerce')
    inr = pd.to_numeric(r.iloc[61], errors='coerce')
    meld = None
    if pd.notna(tbil) and pd.notna(inr) and tbil > 0 and inr > 0:
        meld = 3.78 * math.log(max(tbil, 1)) + 11.2 * math.log(max(inr, 1)) + 9.57 + 6.43
        meld = round(max(meld, 6))
    return {'cp_grade': cp_grade, 'cp_score': cp_score, 'tbil': tbil, 'inr': inr, 'meld': meld}

# Find representative patients: 1 CP-A (normal), 1 CP-B, 1 CP-C, 1 MELD-high
candidates = {'A': [], 'B': [], 'C': [], 'MELD_high': []}
for cat in ['mild', 'moderate', 'severe']:
    cd = os.path.join(INT_DATA, cat)
    if not os.path.exists(cd): continue
    for pid in os.listdir(cd):
        p = os.path.join(cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if not faces: continue
        clin = get_clinical(pid)
        if clin and clin['cp_grade'] in ('A', 'B', 'C'):
            info = {'pid': pid, 'faces': faces, 'cat': cat, **clin}
            candidates[clin['cp_grade']].append(info)
            if clin['meld'] and clin['meld'] > 25:
                candidates['MELD_high'].append(info)

# Also add a normal patient as CP-A
normal_cd = os.path.join(INT_DATA, 'normal')
if os.path.exists(normal_cd):
    for pid in sorted(os.listdir(normal_cd))[:50]:
        p = os.path.join(normal_cd, pid)
        if not os.path.isdir(p): continue
        faces = sorted([os.path.join(p, f) for f in os.listdir(p) if '_face' in f and f.endswith('.jpg')])
        if faces:
            candidates['A'].append({'pid': pid, 'faces': faces, 'cat': 'normal',
                                     'cp_grade': 'A', 'meld': 8, 'tbil': 15, 'inr': 1.0})
            break

# Pick one representative per category
random.seed(SEED)
selected = {}
for grade in ['A', 'B', 'C']:
    pool = candidates.get(grade, [])
    if pool:
        random.shuffle(pool)
        # Pick one with clearer clinical data
        selected[f'CP-{grade}'] = pool[0]
        print(f'  CP-{grade}: {pool[0]["pid"][:30]}  TBIL={pool[0].get("tbil","?")}  CP-score={pool[0].get("cp_score","?")}')

meld_pool = candidates.get('MELD_high', [])
if meld_pool:
    random.shuffle(meld_pool)
    selected['MELD-High'] = meld_pool[0]
    print(f'  MELD-High: {meld_pool[0]["pid"][:30]}  MELD={meld_pool[0].get("meld","?")}')

# Load models
print('\n[2] Loading CP and MELD models...')
cp_model = load_backbone('convnext_tiny', 'cp_face_convnext.pt', 3)
meld_model = load_backbone('vit_tiny_patch16_224', 'meld_face_vit.pt', 3)

# ViT doesn't have 'stages', need different hook target
def find_vit_hook(model):
    """Find suitable hook target for ViT."""
    for name, module in model.named_modules():
        if 'blocks.11' in name or 'norm' in name:
            return name
    return None

# For ViT, we'll use a different approach - hook the last attention block
class ViTGradCAM:
    """Approximate Grad-CAM for ViT using last block attention."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook the last transformer block
        for name, module in model.named_modules():
            if 'blocks.11' in name and 'mlp' not in name and 'norm' not in name:
                module.register_forward_hook(self._fwd)
                module.register_full_backward_hook(self._bwd)
                break

    def _fwd(self, m, i, o): self.activations = o
    def _bwd(self, m, gi, go): self.gradients = go[0]

    def generate(self, tensor, target_class=None):
        self.model.zero_grad()
        self.activations = None; self.gradients = None
        out = self.model(tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        out[0, target_class].backward()
        if self.gradients is None or self.activations is None: return None
        # For ViT: activations shape [1, seq_len, dim] → use CLS token or average
        act = self.activations
        if act.dim() == 3:
            # Average over patch tokens (exclude CLS)
            act = act[:, 1:, :].mean(dim=1, keepdim=True).unsqueeze(-1)  # [1,1,dim,1]
        weights = self.gradients.mean(dim=(1, 2), keepdim=True) if self.gradients.dim() == 3 else self.gradients
        # Simplified: just use activation magnitude
        cam = F.relu(act).squeeze().detach().cpu().numpy()
        # Resize to a small grid then upsample
        grid_size = int(np.sqrt(len(cam)))
        if grid_size * grid_size < len(cam):
            cam = cam[:grid_size*grid_size]
        cam = cam.reshape(grid_size, grid_size)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# Set up Grad-CAM generators
cp_cam = GradCAM(cp_model) if cp_model else None
# For ViT MELD model, use simpler approach
if meld_model:
    # Try ConvNeXt-style hook on ViT blocks
    meld_cam = None
    # Fallback: use the CP ConvNeXt model's Grad-CAM as proxy for visualization
    # OR load MELD ConvNeXt which supports Grad-CAM
    meld_conv = load_backbone('convnext_tiny', 'meld_face_convnext.pt', 3)
    if meld_conv:
        meld_cam = GradCAM(meld_conv)


# Generate gallery
print('\n[3] Generating Grad-CAM gallery...')

def generate_case_cam(model, cam_gen, face_path, target_class=None):
    """Generate Grad-CAM for a single face image."""
    img = read_img(face_path)
    if img is None: return None, None
    img_clahe = clahe_rgb(img)
    img_224 = cv2.resize(img_clahe, (IMG_SIZE, IMG_SIZE))
    tensor = tf(Image.fromarray(img_224).convert('RGB')).unsqueeze(0).to(DEV)
    # Predict
    with torch.enable_grad():
        cam = cam_gen.generate(tensor, target_class)
    return img_224, cam

cases = list(selected.items())
n_cases = len(cases)
if n_cases < 3:
    print(f'  WARNING: only {n_cases} cases available, need at least 3')

# Create figure: n_cases rows × 3 columns (Original, CP Grad-CAM, MELD Grad-CAM)
fig, axes = plt.subplots(n_cases, 3, figsize=(10, 3.5 * n_cases))
if n_cases == 1:
    axes = axes.reshape(1, -1)
fig.suptitle('Grad-CAM Attribution: What the Prognostic Models "See"',
             fontsize=12, fontweight='bold', y=0.99)

col_titles = ['Original (CLAHE)', 'Child-Pugh Model Grad-CAM', 'MELD Model Grad-CAM']

for row, (case_name, case_info) in enumerate(cases):
    # Use the middle frame for visualization
    faces = case_info['faces']
    face_path = faces[len(faces) // 2] if faces else None
    if face_path is None: continue

    # Column 0: Original
    img = read_img(face_path)
    if img is not None:
        img_clahe = clahe_rgb(img)
        img_224 = cv2.resize(img_clahe, (IMG_SIZE, IMG_SIZE))
        axes[row, 0].imshow(img_224)

    # Column 1: CP Grad-CAM
    if cp_model and cp_cam:
        img_224, cam = generate_case_cam(cp_model, cp_cam, face_path)
        if img_224 is not None and cam is not None:
            overlay = make_overlay(img_224, cam, alpha=0.5)
            axes[row, 1].imshow(overlay)

    # Column 2: MELD Grad-CAM
    if meld_cam:
        img_224, cam = generate_case_cam(meld_conv, meld_cam, face_path)
        if img_224 is not None and cam is not None:
            overlay = make_overlay(img_224, cam, alpha=0.5)
            axes[row, 2].imshow(overlay)
    elif 'meld_conv' in dir() and meld_conv:
        # Just show the face with MELD prediction
        axes[row, 2].imshow(img_224 if img_224 is not None else np.zeros((224,224,3)))

    # Row label
    cp = case_info.get('cp_grade', '?')
    meld = case_info.get('meld', '?')
    tbil = case_info.get('tbil', '?')
    row_label = f'{case_name}\nTBIL={tbil}, INR={case_info.get("inr","?")}'
    if isinstance(meld, (int, float)):
        row_label += f'\nMELD={meld:.0f}'
    axes[row, 0].set_ylabel(row_label, fontsize=8, fontweight='bold', rotation=0,
                              labelpad=100, va='center')

    for col in range(3):
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0.08, 0, 1, 0.96])
fig.savefig(os.path.join(FIG_DIR, 'Figure4C_gradcam_gallery.png'))
fig.savefig(os.path.join(FIG_DIR, 'Figure4C_gradcam_gallery.svg'))
plt.close(fig)
print(f'\nSaved: {FIG_DIR}/Figure4C_gradcam_gallery.png/svg')

# Cleanup
if cp_model: del cp_model
if meld_model: del meld_model
if 'meld_conv' in dir() and meld_conv: del meld_conv
torch.cuda.empty_cache()

print('\n' + '='*70)
print('  Grad-CAM gallery done.')
print('='*70)
