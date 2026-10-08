"""
BilinGuard Inference Engine v4 — with built-in interpretability.
  - Eyelid-only ternary grading (clean models)
  - Grad-CAM heatmap overlay
  - Color signal analysis (Lab b*, yellow ratio)
  - t-SNE positioning (patient location in feature space)
"""
import os, time, base64, io, cv2, numpy as np

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

try:
    import torch
    import torch.nn.functional as F
    import timm
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEVERITY_COLORS = {0: '#F4D03F', 1: '#E67E22', 2: '#E74C3C'}
SEVERITY_INFO = {
    0: {'label': 'Mild', 'bilirubin': '34.2\u2013171 \u03bcmol/L (2\u201310 mg/dL)',
        'recommendation': 'Routine monitoring recommended. Consider outpatient follow-up.'},
    1: {'label': 'Moderate', 'bilirubin': '171\u2013342 \u03bcmol/L (10\u201320 mg/dL)',
        'recommendation': 'Inpatient management advised. Monitor liver function closely.'},
    2: {'label': 'Severe', 'bilirubin': '> 342 \u03bcmol/L (> 20 mg/dL)',
        'recommendation': 'Urgent clinical evaluation required. Consider specialist consultation.'},
}

# ═══════════════════════════════════════════════════════════════════════
# Canonical model registry — MODEL_INVENTORY_105.csv (canonical-v2, 2026-07-20)
# 7 tasks x 2 scopes (Face / Eyelid) x 4 backbones = 56 file-backed models.
# The Fusion scope and the "Ensemble" architecture column are computed at
# inference time (no checkpoint file). num_classes: Binary/Type = 2,
# Ternary/DBIL/IBIL/Child-Pugh/MELD = 3.
# ═══════════════════════════════════════════════════════════════════════

# ---- Binary Screening (normal vs jaundiced, 2 classes) ----
FACE_BINARY_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'final_binary_face_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'final_binary_face_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'final_binary_face_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'final_binary_face_efficientnet.pt'},
]
EYELID_BINARY_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'eyelid_binary_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'eyelid_binary_vit_tiny_patch16_224.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'eyelid_binary_swin_tiny_patch4_window7_224.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'eyelid_binary_efficientnet_b0.pt'},
]

# ---- Ternary Grading (TBIL mild / moderate / severe, 3 classes) ----
FACE_TERNARY_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'v3_ternary_convnext_tiny.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'v3_ternary_vit_tiny_patch16_224.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'v3_ternary_swin_tiny_patch4_window7_224.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'v3_ternary_efficientnet_b0.pt'},
]
# Eyelid ternary (canonical-v2: 3 opt models + 1 clean EfficientNet)
EYELID_TERNARY_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'eyelid_opt_convnext_tiny.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'eyelid_opt_vit_tiny_patch16_224.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'eyelid_opt_swin_tiny_patch4_window7_224.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'eyelid_clean_ternary_efficientnet_b0.pt'},
]
# Legacy alias
CLEAN_MODEL_CONFIGS = EYELID_TERNARY_CONFIGS

# ---- Jaundice Type (hepatocellular vs cholestatic, 2 classes) ----
TYPE_FACE_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'typev2_face_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'typev2_face_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'typev2_face_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'typev2_face_efficientnet.pt'},
]
TYPE_EYELID_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'typev2_eyelid_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'typev2_eyelid_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'typev2_eyelid_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'typev2_eyelid_efficientnet.pt'},
]
# Legacy alias — single-image /api/type endpoint uses face typev2 models
TYPE_CLASS_CONFIGS = TYPE_FACE_CONFIGS

TYPE_INFO = {
    0: {'label': 'Hepatocellular', 'description': 'Liver cell damage pattern (viral hepatitis, cirrhosis, drug-induced). '
          'Typically elevated ALT/AST with mixed bilirubin.'},
    1: {'label': 'Cholestatic', 'description': 'Bile flow obstruction pattern (biliary obstruction, primary biliary cholangitis). '
          'Typically elevated ALP/GGT with conjugated bilirubin.'},
}

# DBIL ternary classification models (≤10 / 10-68 / >68 μmol/L)
DBIL_EYELID_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'dbil_eyelid_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'dbil_eyelid_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'dbil_eyelid_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'dbil_eyelid_efficientnet.pt'},
]
DBIL_FACE_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'dbil_face_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'dbil_face_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'dbil_face_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'dbil_face_efficientnet.pt'},
]

# IBIL ternary classification models (≤20 / 20-50 / >50 μmol/L)
IBIL_FACE_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'ibil_face_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'ibil_face_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'ibil_face_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'ibil_face_efficientnet.pt'},
]
IBIL_EYELID_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'ibil_eyelid_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'ibil_eyelid_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'ibil_eyelid_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'ibil_eyelid_efficientnet.pt'},
]

DBIL_INFO = {
    0: {'label': 'Normal / Borderline', 'range': 'DBIL ≤ 10 μmol/L',
        'rec': 'Direct bilirubin within normal range. No cholestasis indicated.'},
    1: {'label': 'Moderate Elevation', 'range': 'DBIL 10-68 μmol/L',
        'rec': 'Moderate direct bilirubin elevation. Hepatobiliary evaluation advised.'},
    2: {'label': 'Severe Elevation', 'range': 'DBIL > 68 μmol/L',
        'rec': 'Severe elevation suggesting biliary obstruction or severe hepatocellular injury. Urgent workup required.'},
}
IBIL_INFO = {
    0: {'label': 'Normal / Borderline', 'range': 'IBIL ≤ 20 μmol/L',
        'rec': 'Indirect bilirubin within or near normal range.'},
    1: {'label': 'Moderate Elevation', 'range': 'IBIL 20-50 μmol/L',
        'rec': 'Moderate indirect hyperbilirubinemia. Consider hemolysis or hepatic dysfunction.'},
    2: {'label': 'Severe Elevation', 'range': 'IBIL > 50 μmol/L',
        'rec': 'Severe indirect hyperbilirubinemia. Hemolytic workup and liver evaluation indicated.'},
}

# Child-Pugh grade prediction (A / B / C, 3 classes)
CP_FACE_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'cp_face_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'cp_face_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'cp_face_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'cp_face_efficientnet.pt'},
]
CP_EYELID_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'cp_eyelid_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'cp_eyelid_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'cp_eyelid_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'cp_eyelid_efficientnet.pt'},
]

# MELD risk category prediction (Low / Mid / High, 3 classes)
MELD_FACE_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'meld_face_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'meld_face_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'meld_face_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'meld_face_efficientnet.pt'},
]
MELD_EYELID_CONFIGS = [
    {'backbone': 'convnext_tiny', 'checkpoint': 'meld_eyelid_convnext.pt'},
    {'backbone': 'vit_tiny_patch16_224', 'checkpoint': 'meld_eyelid_vit.pt'},
    {'backbone': 'swin_tiny_patch4_window7_224', 'checkpoint': 'meld_eyelid_swin.pt'},
    {'backbone': 'efficientnet_b0', 'checkpoint': 'meld_eyelid_efficientnet.pt'},
]

CP_INFO = {
    0: {'label': 'Child-Pugh A', 'range': 'Score 5-6, 1-year survival ~95%',
        'rec': 'Well-compensated cirrhosis. Outpatient management. Treat underlying cause.'},
    1: {'label': 'Child-Pugh B', 'range': 'Score 7-9, 1-year survival ~80%',
        'rec': 'Significant functional compromise. Consider inpatient workup. Monitor for decompensation.'},
    2: {'label': 'Child-Pugh C', 'range': 'Score 10-15, 1-year survival ~45%',
        'rec': 'Decompensated cirrhosis. Inpatient management. Evaluate for liver transplant.'},
}

MELD_INFO = {
    0: {'label': 'Low Risk (MELD ≤20)', 'range': '3-month mortality <6%',
        'rec': 'Low short-term mortality risk. Routine monitoring and treatment.'},
    1: {'label': 'Medium Risk (MELD 21-30)', 'range': '3-month mortality ~20%',
        'rec': 'Moderate risk. Close monitoring. Consider transplant evaluation.'},
    2: {'label': 'High Risk (MELD >30)', 'range': '3-month mortality >50%',
        'rec': 'High mortality risk. Urgent transplant evaluation. ICU-level care.'},
}


class GradCAMGenerator:
    """Generate Grad-CAM heatmap for ConvNeXt models."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_registered = False
        self._register()

    def _register(self):
        target = None
        for name, module in self.model.named_modules():
            if name == 'stages.3':
                target = module; break
        if target is None:
            for name, module in self.model.named_modules():
                if 'stages' in name and name.count('.') == 1:
                    target = module; break
        if target is not None:
            target.register_forward_hook(self._forward_hook)
            target.register_full_backward_hook(self._backward_hook)
            self._hook_registered = True

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor, target_class=None):
        if not self._hook_registered:
            return None
        self.model.zero_grad()
        self.activations = None
        self.gradients = None
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            return None
        if self.activations.dim() != 4:
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def encode_image_base64(img_rgb):
    """Encode RGB numpy array to base64 JPEG."""
    buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[1]
    return base64.b64encode(buf).decode('utf-8')


def encode_overlay(img_rgb, cam, alpha=0.5):
    """Create Grad-CAM overlay and encode to base64."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (img_rgb * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return encode_image_base64(overlay)


def analyze_color(img_rgb):
    """Extract color features from eyelid image center region."""
    h, w = img_rgb.shape[:2]
    center = img_rgb[h//4:3*h//4, w//4:3*w//4]
    lab = cv2.cvtColor(center, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    b_channel = lab[:, :, 2]
    return {
        'lab_b': float(b_channel.mean()),
        'lab_a': float(lab[:, :, 1].mean()),
        'lab_l': float(lab[:, :, 0].mean()),
        'hsv_h': float(hsv[:, :, 0].mean()),
        'hsv_s': float(hsv[:, :, 1].mean()),
        'hsv_v': float(hsv[:, :, 2].mean()),
        'yellow_ratio': float((b_channel > 150).mean()),
        'mean_r': float(center[:, :, 0].mean()),
        'mean_g': float(center[:, :, 1].mean()),
        'mean_b': float(center[:, :, 2].mean()),
    }


class BilinGuardPredictor:
    """Eyelid-only predictor with interpretability."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if _HAS_TORCH else None
        self.models = []              # eyelid ternary (legacy flat list + Grad-CAM)
        self.model_names = []
        self.gradcam_gen = None
        self.face_binary_models = []
        self.eyelid_binary_models = []
        self.face_ternary_models = []
        self.dbil_face_models = []
        self.dbil_eyelid_models = []
        self.ibil_face_models = []
        self.ibil_eyelid_models = []
        self.type_face_models = []
        self.type_eyelid_models = []
        self.cp_face_models = []
        self.cp_eyelid_models = []
        self.meld_face_models = []
        self.meld_eyelid_models = []
        self.sam_model = None
        self.sam_processor = None
        self.face_cascade = None
        self._load_all_models()
        self._load_sam()  # Load SAM for face preprocessing

    def _load_model_list(self, configs, num_classes, tag):
        """Load a config list into (backbone_name, model) tuples (dedup by checkpoint)."""
        loaded, seen = [], set()
        if not _HAS_TORCH or not self.model_dir:
            return loaded
        for cfg in configs:
            ckpt = os.path.join(self.model_dir, cfg['checkpoint'])
            if ckpt in seen:
                continue
            seen.add(ckpt)
            if not os.path.exists(ckpt):
                print(f'  [{tag}] Missing: {cfg["checkpoint"]}')
                continue
            try:
                m = timm.create_model(cfg['backbone'], pretrained=False, num_classes=num_classes).to(self.device)
                m.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
                m.eval()
                loaded.append((cfg['backbone'], m))
                print(f'  [{tag}] Loaded: {cfg["checkpoint"]}')
            except Exception as e:
                print(f'  [{tag}] Failed ({cfg["checkpoint"]}): {e}')
        return loaded

    def _load_all_models(self):
        """Load the full canonical registry (56 file-backed models)."""
        # Eyelid ternary → legacy flat self.models / self.model_names + Grad-CAM
        for name, m in self._load_model_list(EYELID_TERNARY_CONFIGS, 3, 'Eyelid-Ternary'):
            self.models.append(m)
            self.model_names.append(name)
            if name == 'convnext_tiny' and self.gradcam_gen is None:
                self.gradcam_gen = GradCAMGenerator(m)
        self.face_binary_models = self._load_model_list(FACE_BINARY_CONFIGS, 2, 'Face-Binary')
        self.eyelid_binary_models = self._load_model_list(EYELID_BINARY_CONFIGS, 2, 'Eyelid-Binary')
        self.face_ternary_models = self._load_model_list(FACE_TERNARY_CONFIGS, 3, 'Face-Ternary')
        self.dbil_face_models = self._load_model_list(DBIL_FACE_CONFIGS, 3, 'DBIL-Face')
        self.dbil_eyelid_models = self._load_model_list(DBIL_EYELID_CONFIGS, 3, 'DBIL-Eyelid')
        self.ibil_face_models = self._load_model_list(IBIL_FACE_CONFIGS, 3, 'IBIL-Face')
        self.ibil_eyelid_models = self._load_model_list(IBIL_EYELID_CONFIGS, 3, 'IBIL-Eyelid')
        self.type_face_models = self._load_model_list(TYPE_FACE_CONFIGS, 2, 'Type-Face')
        self.type_eyelid_models = self._load_model_list(TYPE_EYELID_CONFIGS, 2, 'Type-Eyelid')
        self.cp_face_models = self._load_model_list(CP_FACE_CONFIGS, 3, 'CP-Face')
        self.cp_eyelid_models = self._load_model_list(CP_EYELID_CONFIGS, 3, 'CP-Eyelid')
        self.meld_face_models = self._load_model_list(MELD_FACE_CONFIGS, 3, 'MELD-Face')
        self.meld_eyelid_models = self._load_model_list(MELD_EYELID_CONFIGS, 3, 'MELD-Eyelid')

    @property
    def type_class_models(self):
        """Backward-compatible alias — single-image /api/type uses face typev2 models."""
        return self.type_face_models

    @property
    def is_trained(self):
        return len(self.models) > 0 or len(self.face_binary_models) > 0

    def _load_sam(self):
        """Load SAM model for face segmentation (matches training pipeline)."""
        if not _HAS_TORCH or not self.model_dir:
            return
        try:
            from transformers import SamModel, SamProcessor
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            self.sam_model.eval()
            # Load face cascade for bounding box detection
            cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self.face_cascade = None
            print('  [SAM] Loaded successfully')
        except Exception as e:
            self.sam_model = None
            print(f'  [SAM] Failed to load: {e}')

    def _preprocess_face_sam(self, img_rgb):
        """SAM face segmentation pipeline — matches training preprocessing.
        Steps: face detection → SAM segmentation → tight crop → CLAHE → resize."""
        h, w = img_rgb.shape[:2]
        # Step 1: Get face bounding box
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        bbox = None
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                areas = [fw * fh for _, _, fw, fh in faces]
                x, y, fw, fh = faces[np.argmax(areas)]
                bbox = [int(x), int(y), int(x + fw), int(y + fh)]
        if bbox is None:
            bbox = [int(w * 0.25), int(h * 0.1), int(w * 0.75), int(h * 0.85)]
        # Step 2: SAM segmentation
        face_crop = img_rgb  # fallback: use whole image
        if self.sam_model is not None:
            try:
                from PIL import Image as PILImage
                pil = PILImage.fromarray(img_rgb)
                inputs = self.sam_processor(pil, input_boxes=[[[list(bbox)]]], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.sam_model(**inputs)
                masks = self.sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu())
                mask = masks[0][0][0].numpy()
                if mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    ys = np.where(rows)[0]
                    xs = np.where(cols)[0]
                    rmin = max(0, ys[0] - 5)
                    rmax = min(h, ys[-1] + 5)
                    cmin = max(0, xs[0] - 5)
                    cmax = min(w, xs[-1] + 5)
                    face_crop = img_rgb[rmin:rmax, cmin:cmax].copy()
            except Exception:
                face_crop = img_rgb
        # Step 3: CLAHE
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        face_clahe = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        # Step 4: Resize
        face_224 = cv2.resize(face_clahe, (IMG_SIZE, IMG_SIZE))
        return face_224

    def _preprocess(self, img_rgb, use_sam=False, normalize_color=True):
        """Preprocess image for model input.
        use_sam=True: SAM face segmentation + CLAHE (for face models)
        use_sam=False: CLAHE only (for eyelid models)
        normalize_color=True: Grey World normalization to reduce domain shift
        Returns normalized tensor (C, H, W)."""
        if normalize_color:
            img_rgb = self._grey_world(img_rgb)
        if use_sam and self.sam_model is not None:
            img_224 = self._preprocess_face_sam(img_rgb)
        else:
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            img_224 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        norm = img_224.astype(np.float32) / 255.0
        norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
        return norm.transpose(2, 0, 1)

    @staticmethod
    def _grey_world(img_rgb):
        """Grey World color constancy: normalize so mean R=G=B.
        Reduces domain shift between internal and external data."""
        mean_r = img_rgb[:, :, 0].mean()
        mean_g = img_rgb[:, :, 1].mean()
        mean_b = img_rgb[:, :, 2].mean()
        gray = (mean_r + mean_g + mean_b) / 3.0
        if gray < 1: return img_rgb
        out = img_rgb.astype(np.float32)
        out[:, :, 0] *= gray / (mean_r + 1e-6)
        out[:, :, 1] *= gray / (mean_g + 1e-6)
        out[:, :, 2] *= gray / (mean_b + 1e-6)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _predict(self, tensor):
        """Run ensemble prediction."""
        all_probs = []
        with torch.no_grad():
            x = torch.FloatTensor(tensor).to(self.device)
            for m in self.models:
                out = m(x)
                all_probs.append(F.softmax(out, dim=1).mean(dim=0).cpu().numpy())
        weights = np.ones(len(all_probs)) / len(all_probs)
        result = np.zeros(3)
        for w, p in zip(weights, all_probs):
            result += w * p
        return result

    def analyze_eyelid(self, file_path):
        """Full analysis with Grad-CAM and color interpretation."""
        t0 = time.time()

        # Load image
        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except:
            img = None
        if img is None:
            img = cv2.imread(file_path)
        if img is None:
            return {'error': 'Could not read image file.'}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        # Preprocess
        tensor = np.stack([self._preprocess(img_rgb)] * 12)

        # Prediction
        grade_prob = self._predict(tensor)
        grade_idx = int(np.argmax(grade_prob))

        # Color analysis
        color_features = analyze_color(img_rgb)

        # Grad-CAM (if ConvNeXt available)
        gradcam_base64 = None
        gradcam_only_base64 = None
        if self.gradcam_gen is not None:
            t_input = torch.FloatTensor([self._preprocess(img_rgb)]).to(self.device)
            cam = self.gradcam_gen.generate(t_input, target_class=grade_idx)
            if cam is not None:
                gradcam_base64 = encode_overlay(img_resized, cam, alpha=0.5)
                gradcam_only_base64 = encode_overlay(img_resized, cam, alpha=0.9)

        elapsed = time.time() - t0
        info = SEVERITY_INFO[grade_idx]

        result = {
            'binary': {
                'prediction': 'Jaundiced',
                'probability_normal': 0.0,
                'probability_jaundice': 1.0,
            },
            'is_jaundice': True,
            'inference_time': round(elapsed, 2),
            'n_frames': 12,
            'model_mode': 'trained' if self.is_trained else 'demo',
            'model_details': [f'Ensemble: {len(self.models)} models ({", ".join(self.model_names)})'],
            'input_mode': 'eyelid',
            'thumbnails': [encode_image_base64(img_resized)],
            'grading': {
                'prediction': info['label'],
                'grade_index': grade_idx,
                'probabilities': {
                    'mild': float(grade_prob[0]),
                    'moderate': float(grade_prob[1]),
                    'severe': float(grade_prob[2]),
                },
                'bilirubin_range': info['bilirubin'],
                'recommendation': info['recommendation'],
                'color': SEVERITY_COLORS[grade_idx],
            },
            'interpretability': {
                'gradcam_overlay': gradcam_base64,
                'gradcam_only': gradcam_only_base64,
                'color_features': color_features,
                'has_gradcam': gradcam_base64 is not None,
            },
        }
        return result

    def analyze_face(self, file_path, is_video=True):
        """Face mode: delegate to eyelid if face models unavailable."""
        return self.analyze_eyelid(file_path)

    def analyze_combined(self, face_path, eyelid_path, is_video=True):
        """Combined mode: use eyelid result (face data too sparse)."""
        return self.analyze_eyelid(eyelid_path if eyelid_path else face_path)

    def screen_face(self, file_path):
        """Binary screening: normal vs jaundiced using face photo.
        Applies SAM face segmentation to match training preprocessing."""
        t0 = time.time()
        if not self.face_binary_models:
            return {'error': 'Face binary screening models not loaded.'}

        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            img = cv2.imread(file_path)
        if img is None:
            return {'error': 'Could not read image file.'}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        # SAM preprocessing for face (matches training: SAM crop + CLAHE)
        tensor = np.stack([self._preprocess(img_rgb, use_sam=True)] * 8)

        x = torch.FloatTensor(tensor).to(self.device)
        all_probs = []
        model_names = []
        with torch.no_grad():
            for name, m in self.face_binary_models:
                out = m(x)
                all_probs.append(F.softmax(out, dim=1).mean(dim=0).cpu().numpy())
                model_names.append(name)

        avg_prob = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))

        elapsed = time.time() - t0
        return {
            'binary': {
                'prediction': 'Jaundiced' if pred_idx == 1 else 'Normal',
                'probability_normal': float(avg_prob[0]),
                'probability_jaundice': float(avg_prob[1]),
            },
            'is_jaundice': pred_idx == 1,
            'inference_time': round(elapsed, 2),
            'model_mode': 'trained',
            'input_mode': 'face_screen',
            'model_details': [f'Ensemble: {len(self.face_binary_models)} models ({", ".join(model_names)})'],
            'thumbnails': [encode_image_base64(img_resized)],
        }

    def classify_type(self, file_path):
        """Jaundice type classification: hepatocellular vs cholestatic."""
        t0 = time.time()
        if not self.type_class_models:
            return {'error': 'Jaundice type classification models not loaded.'}

        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            img = cv2.imread(file_path)
        if img is None:
            return {'error': 'Could not read image file.'}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        tensor = np.stack([self._preprocess(img_rgb, use_sam=True)] * 8)

        x = torch.FloatTensor(tensor).to(self.device)
        all_probs = []
        model_names = []
        with torch.no_grad():
            for name, m in self.type_class_models:
                out = m(x)
                all_probs.append(F.softmax(out, dim=1).mean(dim=0).cpu().numpy())
                model_names.append(name)

        avg_prob = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))
        info = TYPE_INFO[pred_idx]

        elapsed = time.time() - t0
        return {
            'type_classification': {
                'prediction': info['label'],
                'type_index': pred_idx,
                'probability_hepatocellular': float(avg_prob[0]),
                'probability_cholestatic': float(avg_prob[1]),
                'description': info['description'],
            },
            'inference_time': round(elapsed, 2),
            'model_mode': 'trained',
            'input_mode': 'jaundice_type',
            'model_details': [f'Ensemble: {len(self.type_class_models)} models ({", ".join(model_names)})'],
            'thumbnails': [encode_image_base64(img_resized)],
        }

    def _classify_ternary(self, file_path, model_list, info_dict, task_name, nc=3):
        """Generic ternary classification (DBIL or IBIL).
        Uses SAM for face inputs, CLAHE only for eyelid inputs."""
        t0 = time.time()
        if not model_list:
            return {'error': f'{task_name} models not loaded.'}

        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            img = cv2.imread(file_path)
        if img is None:
            return {'error': 'Could not read image file.'}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        # Face models need SAM preprocessing; eyelid models use CLAHE only
        use_sam = 'face' in task_name and self.sam_model is not None
        tensor = np.stack([self._preprocess(img_rgb, use_sam=use_sam)] * 8)

        x = torch.FloatTensor(tensor).to(self.device)
        all_probs = []
        model_names = []
        with torch.no_grad():
            for name, m in model_list:
                out = m(x)
                all_probs.append(F.softmax(out, dim=1).mean(dim=0).cpu().numpy())
                model_names.append(name)

        avg_prob = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))
        info = info_dict[pred_idx]
        grade_labels = [info_dict[i]['label'] for i in range(nc)]
        elapsed = time.time() - t0

        return {
            'grading': {
                'prediction': info['label'],
                'grade_index': pred_idx,
                'probabilities': {grade_labels[i].lower().split('/')[0].strip(): float(avg_prob[i]) for i in range(nc)},
                'bilirubin_range': info['range'],
                'recommendation': info['rec'],
            },
            'inference_time': round(elapsed, 2),
            'model_mode': 'trained',
            'input_mode': task_name,
            'model_details': [f'Ensemble: {len(model_list)} models'],
            'thumbnails': [encode_image_base64(img_resized)],
        }

    def classify_dbil_eyelid(self, file_path):
        return self._classify_ternary(file_path, self.dbil_eyelid_models, DBIL_INFO, 'dbil_eyelid')

    def classify_dbil_face(self, file_path):
        return self._classify_ternary(file_path, self.dbil_face_models, DBIL_INFO, 'dbil_face')

    def classify_ibil_face(self, file_path):
        return self._classify_ternary(file_path, self.ibil_face_models, IBIL_INFO, 'ibil_face')

    def classify_ibil_eyelid(self, file_path):
        return self._classify_ternary(file_path, self.ibil_eyelid_models, IBIL_INFO, 'ibil_eyelid')

    def classify_child_pugh(self, file_path):
        """Child-Pugh grade prediction (A/B/C) from face photo."""
        return self._classify_ternary(file_path, self.cp_face_models, CP_INFO, 'cp_face')

    def classify_child_pugh_eyelid(self, file_path):
        """Child-Pugh grade prediction (A/B/C) from eyelid photo."""
        return self._classify_ternary(file_path, self.cp_eyelid_models, CP_INFO, 'cp_eyelid')

    def classify_meld(self, file_path):
        """MELD risk category (Low/Mid/High) from face photo."""
        return self._classify_ternary(file_path, self.meld_face_models, MELD_INFO, 'meld_face')

    def classify_meld_eyelid(self, file_path):
        """MELD risk category (Low/Mid/High) from eyelid photo."""
        return self._classify_ternary(file_path, self.meld_eyelid_models, MELD_INFO, 'meld_eyelid')

    @staticmethod
    def _read_rgb(file_path):
        """Read an image file as RGB numpy array (unicode-path safe)."""
        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            img = cv2.imread(file_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def classify_type_eyelid(self, file_path):
        """Jaundice type classification from eyelid photo (CLAHE-only, no SAM)."""
        t0 = time.time()
        if not self.type_eyelid_models:
            return {'error': 'Eyelid jaundice type models not loaded.'}
        img_rgb = self._read_rgb(file_path)
        if img_rgb is None:
            return {'error': 'Could not read image file.'}
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        tensor = np.stack([self._preprocess(img_rgb, use_sam=False)] * 8)
        x = torch.FloatTensor(tensor).to(self.device)
        all_probs, model_names = [], []
        with torch.no_grad():
            for name, m in self.type_eyelid_models:
                out = m(x)
                all_probs.append(F.softmax(out, dim=1).mean(dim=0).cpu().numpy())
                model_names.append(name)
        avg_prob = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))
        info = TYPE_INFO[pred_idx]
        elapsed = time.time() - t0
        return {
            'type_classification': {
                'prediction': info['label'],
                'type_index': pred_idx,
                'probability_hepatocellular': float(avg_prob[0]),
                'probability_cholestatic': float(avg_prob[1]),
                'description': info['description'],
            },
            'inference_time': round(elapsed, 2),
            'model_mode': 'trained',
            'input_mode': 'jaundice_type_eyelid',
            'model_details': [f'Ensemble: {len(self.type_eyelid_models)} models ({", ".join(model_names)})'],
            'thumbnails': [encode_image_base64(img_resized)],
        }

    def screen_eyelid(self, file_path):
        """Binary screening from eyelid photo (CLAHE-only, no SAM)."""
        t0 = time.time()
        if not self.eyelid_binary_models:
            return {'error': 'Eyelid binary screening models not loaded.'}
        img_rgb = self._read_rgb(file_path)
        if img_rgb is None:
            return {'error': 'Could not read image file.'}
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        tensor = np.stack([self._preprocess(img_rgb, use_sam=False)] * 8)
        x = torch.FloatTensor(tensor).to(self.device)
        all_probs, model_names = [], []
        with torch.no_grad():
            for name, m in self.eyelid_binary_models:
                out = m(x)
                all_probs.append(F.softmax(out, dim=1).mean(dim=0).cpu().numpy())
                model_names.append(name)
        avg_prob = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))
        elapsed = time.time() - t0
        return {
            'binary': {
                'prediction': 'Jaundiced' if pred_idx == 1 else 'Normal',
                'probability_normal': float(avg_prob[0]),
                'probability_jaundice': float(avg_prob[1]),
            },
            'is_jaundice': pred_idx == 1,
            'inference_time': round(elapsed, 2),
            'model_mode': 'trained',
            'input_mode': 'eyelid_screen',
            'model_details': [f'Ensemble: {len(self.eyelid_binary_models)} models ({", ".join(model_names)})'],
            'thumbnails': [encode_image_base64(img_resized)],
        }

    def classify_grade_face(self, file_path):
        """TBIL ternary grading from face photo."""
        if not self.face_ternary_models:
            return {'error': 'Face ternary models not loaded.'}
        t0 = time.time()
        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            img = cv2.imread(file_path)
        if img is None:
            return {'error': 'Could not read image file.'}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        tensor = np.stack([self._preprocess(img_rgb, use_sam=True)] * 8)
        x = torch.FloatTensor(tensor).to(self.device)
        all_probs = []
        model_names = []
        with torch.no_grad():
            for name, m in self.face_ternary_models:
                out = m(x)
                all_probs.append(F.softmax(out, dim=1).mean(dim=0).cpu().numpy())
                model_names.append(name)
        avg_prob = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))
        grade_labels = ['Mild', 'Moderate', 'Severe']
        grade_colors = {'Mild': '#F4D03F', 'Moderate': '#E67E22', 'Severe': '#E74C3C'}
        recs = ['Mild jaundice (TBIL 34.2-171). Outpatient follow-up suggested.',
                'Moderate jaundice (TBIL 171-342). Inpatient management advised.',
                'Severe jaundice (TBIL >342). Urgent evaluation required.']
        elapsed = time.time() - t0
        return {
            'grading': {
                'prediction': grade_labels[pred_idx],
                'grade_index': pred_idx,
                'probabilities': {grade_labels[i].lower(): float(avg_prob[i]) for i in range(3)},
                'bilirubin_range': ['TBIL 34.2-171 umol/L', 'TBIL 171-342 umol/L', 'TBIL >342 umol/L'][pred_idx],
                'recommendation': recs[pred_idx],
            },
            'inference_time': round(elapsed, 2),
            'model_mode': 'trained',
            'input_mode': 'grade_face',
            'model_details': [f'Ensemble: {len(self.face_ternary_models)} models'],
        }

    @staticmethod
    def extract_video_frames(video_path, max_frames=12):
        """Extract frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total > 0:
            indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        else:
            indices = range(max_frames)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def analyze_video(self, video_path, mode='screen'):
        """Analyze video: extract frames → SAM preprocess each → multi-frame average.
        Matches training pipeline: SAM face segmentation + CLAHE + patient-level aggregation."""
        t0 = time.time()
        frames = self.extract_video_frames(video_path)
        if not frames:
            return {'error': 'Could not extract frames from video.'}

        # Determine if this mode uses face models (need SAM) or eyelid models
        face_modes = {'screen', 'grade', 'dbil', 'ibil', 'type'}
        use_sam = mode in face_modes and self.sam_model is not None

        model_sets = {
            'screen': (self.face_binary_models, 2),
            'grade': (self.face_ternary_models or [(self.model_names[i], self.models[i]) for i in range(min(3, len(self.models)))], 3),
            'dbil': (self.dbil_face_models or self.dbil_eyelid_models, 3),
            'ibil': (self.ibil_face_models or self.ibil_eyelid_models, 3),
            'type': (self.type_class_models, 2),
            'cp': (self.cp_face_models, 3),
            'meld': (self.meld_face_models, 3),
        }
        if mode not in model_sets:
            return {'error': f'Unknown mode: {mode}'}
        model_list, nc = model_sets[mode]
        if not model_list:
            return {'error': f'No models for mode: {mode}'}

        all_probs = []
        for frame in frames:
            # Apply SAM preprocessing for face modes (matches training)
            tensor = np.stack([self._preprocess(frame, use_sam=use_sam)] * 4)
            x = torch.FloatTensor(tensor).to(self.device)
            fp = []
            with torch.no_grad():
                for name, m in model_list:
                    fp.append(F.softmax(m(x), dim=1).mean(dim=0).cpu().numpy())
            all_probs.append(np.mean(fp, axis=0))

        avg_prob = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_prob))
        elapsed = time.time() - t0

        if nc == 2:
            labels = ['Normal', 'Jaundiced'] if mode == 'screen' else ['Hepatocellular', 'Cholestatic']
            result_key = 'binary' if mode == 'screen' else 'type_classification'
            extra = {}
            if mode == 'screen':
                extra = {'is_jaundice': pred_idx == 1,
                         'probability_normal': float(avg_prob[0]),
                         'probability_jaundice': float(avg_prob[1])}
            elif mode == 'type':
                extra = {'type_index': pred_idx,
                         'probability_hepatocellular': float(avg_prob[0]),
                         'probability_cholestatic': float(avg_prob[1]),
                         'description': TYPE_INFO[pred_idx].get('description', '')}
            return {
                result_key: {
                    'prediction': labels[pred_idx],
                    'probabilities': {labels[i].lower(): float(avg_prob[i]) for i in range(2)},
                    **extra,
                },
                'inference_time': round(elapsed, 2), 'n_frames': len(frames),
                'model_mode': 'trained', 'input_mode': f'video_{mode}',
            }
        else:
            info_map = {'grade': (['Mild', 'Moderate', 'Severe'], None),
                        'dbil': (['Normal / Borderline', 'Moderate Elevation', 'Severe Elevation'], DBIL_INFO),
                        'ibil': (['Normal / Borderline', 'Moderate Elevation', 'Severe Elevation'], IBIL_INFO),
                        'cp': (['Child-Pugh A', 'Child-Pugh B', 'Child-Pugh C'], CP_INFO),
                        'meld': (['Low Risk', 'Medium Risk', 'High Risk'], MELD_INFO)}
            gl, info_dict = info_map.get(mode, (['G0', 'G1', 'G2'], None))
            extra = {}
            if info_dict:
                extra = {'bilirubin_range': info_dict[pred_idx]['range'],
                         'recommendation': info_dict[pred_idx]['rec']}
            return {
                'grading': {'prediction': gl[pred_idx], 'grade_index': pred_idx,
                            'probabilities': {gl[i].lower(): float(avg_prob[i]) for i in range(3)},
                            **extra},
                'inference_time': round(elapsed, 2), 'n_frames': len(frames),
                'model_mode': 'trained', 'input_mode': f'video_{mode}',
            }

    def analyze_combined(self, face_path, eyelid_path=None, is_video=False):
        """Combined face + eyelid comprehensive analysis."""
        t0 = time.time()
        results = {}
        if face_path:
            if is_video:
                results['screening'] = self.analyze_video(face_path, 'screen')
            elif self.face_binary_models:
                results['screening'] = self.screen_face(face_path)
        if eyelid_path and self.models:
            results['grading'] = self.analyze_eyelid(eyelid_path)
        if face_path and not is_video and self.type_class_models:
            sc = results.get('screening', {})
            if sc.get('is_jaundice', True):
                results['type'] = self.classify_type(face_path)
        results['inference_time'] = round(time.time() - t0, 2)
        results['input_mode'] = 'combined'
        return results
