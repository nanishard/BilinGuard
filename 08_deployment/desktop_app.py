# -*- coding: utf-8 -*-
"""
BilinGuard Desktop Application — v4.0
=====================================
Brand-new UI built from the ui-ux-pro-max design system:
  Style : Soft UI Evolution (enterprise / health) — WCAG AA+
  Color : Trust teal (#0F766E) + professional blue (#0369A1)
  Type  : unified clean sans (Segoe UI / Microsoft YaHei UI)
  Dials : variance 4, motion 2 (subtle), density 8 (dashboard)
  Icons : SVG only (no emoji)
  Modes : Light + Dark (full)

Backend (model loading, analysis pipeline, Grad-CAM, clinical advisor)
is preserved unchanged.
"""
import os, sys, time, cv2, numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'deployment'))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# PHI-safe CDSS audit trail (closed-loop logging; see clinical_advisor.py v3)
AUDIT_LOG = Path(__file__).parent / 'audit' / 'cdss_advice_log.jsonl'
try:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                              QFileDialog, QProgressBar, QFrame, QMessageBox,
                              QSlider, QGroupBox, QTextEdit, QScrollArea, QSizePolicy,
                              QComboBox, QLineEdit, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon, QPainter
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

try:
    from PyQt5.QtSvg import QSvgRenderer
    HAS_SVG = True
except Exception:
    HAS_SVG = False

# Clinical Decision Support backend
try:
    from clinical_advisor import ClinicalAdvisor, DEPARTMENTS
    HAS_ADVISOR = True
except Exception:
    ClinicalAdvisor = None
    DEPARTMENTS = {}
    HAS_ADVISOR = False


# ════════════════════════════════════════════════════════════════════
# DESIGN TOKENS  (from ui-ux-pro-max: Soft UI Evolution, trust teal)
# ════════════════════════════════════════════════════════════════════
_PALETTE_LIGHT = {
    'bg': '#F0FDFA', 'surface': '#FFFFFF', 'surface_alt': '#F5FBF9',
    'header': '#FFFFFF',
    'primary': '#0F766E', 'primary_hover': '#0B5E57', 'primary_active': '#094A45',
    'secondary': '#14B8A6', 'accent': '#0369A1', 'accent_hover': '#075E8A',
    'on_primary': '#FFFFFF',
    'text': '#0B3B38', 'text_secondary': '#4B6B66', 'text_muted': '#5C7873',
    'border': '#99F6E4', 'border_strong': '#5EEAD4',
    'muted': '#E8F0F3', 'soft': '#CCFBF1',
    'danger': '#DC2626', 'danger_soft': '#FEE2E2',
    'normal': '#16A34A', 'normal_soft': '#DCFCE7',
    'mild': '#D97706', 'mild_soft': '#FEF3C7',
    'moderate': '#EA580C', 'moderate_soft': '#FFEDD5',
    'severe': '#DC2626', 'severe_soft': '#FEE2E2', 'warning': '#D97706', 'warning_soft': '#FEF3C7',
    'shadow': 'rgba(15,118,110,0.10)',
}
_PALETTE_DARK = {
    'bg': '#06201E', 'surface': '#0E2A27', 'surface_alt': '#123330',
    'header': '#0B2321',
    'primary': '#2DD4BF', 'primary_hover': '#14B8A6', 'primary_active': '#0D9488',
    'secondary': '#5EEAD4', 'accent': '#38BDF8', 'accent_hover': '#0EA5E9',
    'on_primary': '#04201E',
    'text': '#E6F2F0', 'text_secondary': '#A7C2BD', 'text_muted': '#9CB8B4',
    'border': '#1F4742', 'border_strong': '#2E6059',
    'muted': '#112E2B', 'soft': '#133E3B',
    'danger': '#F87171', 'danger_soft': '#3A1A1A',
    'normal': '#22C55E', 'normal_soft': '#13291D',
    'mild': '#F59E0B', 'mild_soft': '#33270A',
    'moderate': '#FB923C', 'moderate_soft': '#33210B',
    'severe': '#F87171', 'severe_soft': '#3A1A1A', 'warning': '#F59E0B', 'warning_soft': '#33260A',
    'shadow': 'rgba(0,0,0,0.40)',
}
COLORS = dict(_PALETTE_LIGHT)


class Theme:
    mode = 'light'
    _listeners = []

    @classmethod
    def toggle(cls):
        cls.apply('dark' if cls.mode == 'light' else 'light')

    @classmethod
    def apply(cls, mode):
        cls.mode = mode
        COLORS.update(_PALETTE_DARK if mode == 'dark' else _PALETTE_LIGHT)
        for cb in cls._listeners:
            try:
                cb()
            except Exception as e:
                print('[Theme]', e, flush=True)

    @classmethod
    def register(cls, cb):
        if cb not in cls._listeners:
            cls._listeners.append(cb)


# 8dp spacing rhythm + unified type scale (single family: Segoe UI / YaHei UI)
class FontScale:
    BASE = 14
    MIN = 10
    MAX = 22
    current = 14
    app = None
    refresh_callback = None
    _T = {'display': 2.6, 'h1': 1.7, 'h2': 1.22, 'h3': 1.02,
          'body': 0.95, 'small': 0.86, 'caption': 0.78}

    @classmethod
    def set(cls, size):
        cls.current = max(cls.MIN, min(cls.MAX, size))
        if cls.app:
            cls.app.setFont(QFont(_FONT, cls.current))
        if cls.refresh_callback:
            cls.refresh_callback()

    @classmethod
    def ts(cls, key='body'):
        return max(8, int(round(cls.current * cls._T.get(key, 1.0))))


def _font_family():
    return 'Segoe UI, Microsoft YaHei UI' if os.name == 'nt' else 'Arial'

_FONT = _font_family()


# ── SVG icons (no emoji) ────────────────────────────────────────────
def _svg(path_d, color='#0F766E', size=20, fill=False, stroke_width=1.8, extra=''):
    fill_attr = f'fill="{color}" stroke="none"' if fill else f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"'
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" {extra}>'
         f'<path d="{path_d}" {fill_attr}/></svg>')
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    if HAS_SVG:
        try:
            r = QSvgRenderer(s.encode('utf-8'))
            p = QPainter(pix); r.render(p); p.end()
            return QIcon(pix)
        except Exception:
            pass
    return QIcon()


ICONS = {
    'logo': 'M12 2 C7 8 5 11 5 14 a7 7 0 0 0 14 0 c0-3-2-6-7-12 z',          # droplet (fill)
    'upload': 'M12 16 V4 M7 9 l5 -5 l5 5 M5 20 h14',                          # upload tray
    'eye': 'M2 12 s4 -7 10 -7 s10 7 10 7 s-4 7 -10 7 s-10 -7 -10 -7 z M12 9 a3 3 0 1 0 0 6 a3 3 0 0 0 0 -6',
    'analyze': 'M5 12 l4 4 l10 -10',                                          # check / run
    'trash': 'M4 7 h16 M9 7 V4 h6 V7 M6 7 l1 13 h10 l1 -13',
    'settings': 'M12 9 a3 3 0 1 0 0 6 a3 3 0 0 0 0 -6 z M12 2 v3 M12 19 v3 M4.2 4.2 l2.1 2.1 M17.7 17.7 l2.1 2.1 M2 12 h3 M19 12 h3 M4.2 19.8 l2.1 -2.1 M17.7 6.3 l2.1 -2.1',
    'globe': 'M12 3 a9 9 0 1 0 0 18 a9 9 0 0 0 0 -18 z M3 12 h18 M12 3 c3 3 3 15 0 18 c-3 -3 -3 -15 0 -18',
    'sun': 'M12 7 a5 5 0 1 0 0 10 a5 5 0 0 0 0 -10 z M12 2 v2 M12 20 v2 M4 12 H2 M22 12 h-2 M5 5 l1.5 1.5 M17.5 17.5 L19 19 M19 5 l-1.5 1.5 M6.5 17.5 L5 19',
    'moon': 'M21 13 A9 9 0 0 1 11 3 a7 7 0 1 0 10 10 z',
    'stethoscope': 'M6 3 v6 a4 4 0 0 0 8 0 V3 M6 3 H4 M14 3 h2 M10 17 v0 M10 17 a5 5 0 0 0 8 0 a3 3 0 0 0 -2 -2',
    'shield': 'M12 2 l8 3 v6 c0 5 -3.5 8 -8 11 c-4.5 -3 -8 -6 -8 -11 V5 z',
    'arrow_left': 'M19 12 H5 M12 6 l-6 6 l6 6',
}


def icon(name, color=None, size=20):
    color = color or COLORS['primary']
    fill = name in ('logo', 'shield')
    return _svg(ICONS.get(name, ''), color, size, fill=fill)


class ClickableLabel(QLabel):
    """A QLabel that emits clicked() on left mouse press (and lets clicks pass, not swallow)."""
    clicked = pyqtSignal()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
            e.accept()
        else:
            super().mousePressEvent(e)


# ── i18n ────────────────────────────────────────────────────────────
class I18n:
    LANG = 'cn'
    T = {
        'en': {
            'title': 'BilinGuard', 'subtitle': 'AI Jaundice Assessment · West China Hospital',
            'loading': 'Loading models…', 'models': 'Models ready',
            'capture': '1 · Image Capture', 'capture_sub': 'Upload face photo/video (required) and everted-eyelid photo (optional)',
            'ctx_title': 'Clinical Context (optional)',
            'ctx_hint': 'Department + admission diagnosis + recent labs give tailored, guideline-cited advice.',
            'ctx_dept': 'Dept', 'ctx_dept_auto': 'Auto (infer)',
            'ctx_dx': 'Admission diagnosis', 'ctx_dx_ph': 'e.g. ACLF, chronic hepatitis B, choledocholithiasis',
            'ctx_labs': 'Recent labs (optional)',
            'analyze': 'Analyze', 'analyzing': 'Analyzing…', 'clear': 'Clear',
            'upload_title': 'Facial photo / video', 'upload_hint': 'Drop here or click to browse  ·  MP4, AVI, MOV, JPG, PNG',
            'drop_or_click': 'Drop file, or click to browse',
            'upload_eyelid': 'Everted eyelid photo', 'upload_optional': 'Optional — improves grading accuracy',
            'section_screen': 'Screening', 'section_grade': 'Severity (TBIL)',
            'section_dbil': 'Direct Bilirubin', 'section_ibil': 'Indirect Bilirubin', 'section_type': 'Jaundice Type',
            'section_cp': 'Child-Pugh Grade', 'section_meld': 'MELD Risk Score',
            'normal': 'Normal', 'jaundiced': 'Jaundiced', 'mild': 'Mild', 'moderate': 'Moderate', 'severe': 'Severe',
            'hepatocellular': 'Hepatocellular', 'cholestatic': 'Cholestatic',
            'hepato_desc': 'Pattern suggests liver-cell damage.', 'chol_desc': 'Pattern suggests biliary obstruction.',
            'results': '2 · Assessment', 'awaiting': 'Awaiting analysis',
            'severity_hero': 'Severity', 'type_badge_none': '—',
            'advice_title': '3 · Clinical Guidance', 'advidence_empty': 'Run an analysis to see tailored, guideline-cited guidance.',
            'advice_summary': 'Triage summary', 'advice_evidence': 'Evidence',
            'interp_title': '4 · Model Interpretability',
            'face_analysis': 'Face', 'eyelid_analysis': 'Eyelid',
            'color_title': 'Conjunctival colour', 'agreement_title': 'Model agreement',
            'time': 'Inference time', 'settings': 'Settings', 'appearance': 'Appearance',
            'font_size': 'Font size', 'font_small': 'S', 'font_medium': 'M', 'font_large': 'L', 'font_xlarge': 'XL',
            'reset': 'Reset', 'theme_toggle': 'Toggle theme', 'no_data': 'Not available',
            'about': 'BilinGuard uses deep learning on facial / eyelid images for non-invasive bilirubin assessment. For clinical decision support only — does not replace serum bilirubin testing.',
        },
        'cn': {
            'title': 'BilinGuard', 'subtitle': 'AI 黄疸评估 · 四川大学华西医院',
            'loading': '正在加载模型…', 'models': '模型就绪',
            'capture': '1 · 影像采集', 'capture_sub': '上传面部照片/视频（必需）与翻上眼睑照片（可选）',
            'ctx_title': '临床背景（可选）',
            'ctx_hint': '选择科室 + 填写入院诊断 + 近期化验，可获得带指南引用的个性化建议。',
            'ctx_dept': '科室', 'ctx_dept_auto': '自动（推断）',
            'ctx_dx': '入院诊断', 'ctx_dx_ph': '如：慢加急性肝衰竭、慢性乙型病毒性肝炎、胆总管结石',
            'ctx_labs': '近期化验（可选）',
            'analyze': '开始分析', 'analyzing': '分析中…', 'clear': '清除',
            'upload_title': '面部照片 / 视频', 'upload_hint': '拖入此处或点击浏览  ·  MP4、AVI、MOV、JPG、PNG',
            'drop_or_click': '拖入文件，或点击选择',
            'upload_eyelid': '翻上眼睑照片', 'upload_optional': '可选 —— 可提升分级准确性',
            'section_screen': '筛查', 'section_grade': '严重度（TBIL）',
            'section_dbil': '直接胆红素', 'section_ibil': '间接胆红素', 'section_type': '黄疸类型',
            'section_cp': 'Child-Pugh 分级', 'section_meld': 'MELD 风险评分',
            'normal': '正常', 'jaundiced': '黄疸', 'mild': '轻度', 'moderate': '中度', 'severe': '重度',
            'hepatocellular': '肝细胞性', 'cholestatic': '胆汁郁积性',
            'hepato_desc': '模式提示肝细胞损伤。', 'chol_desc': '模式提示胆道梗阻。',
            'results': '2 · 评估结果', 'awaiting': '等待分析',
            'severity_hero': '严重程度', 'type_badge_none': '—',
            'advice_title': '3 · 临床指导', 'advidence_empty': '运行分析后显示带指南引用的个性化指导。',
            'advice_summary': '分诊小结', 'advice_evidence': '依据指南',
            'interp_title': '4 · 模型可解释性',
            'face_analysis': '面部', 'eyelid_analysis': '眼睑',
            'color_title': '结膜色彩', 'agreement_title': '模型一致性',
            'time': '推理耗时', 'settings': '设置', 'appearance': '外观',
            'font_size': '字体大小', 'font_small': '小', 'font_medium': '中', 'font_large': '大', 'font_xlarge': '特大',
            'reset': '恢复默认', 'theme_toggle': '切换主题', 'no_data': '不可用',
            'about': 'BilinGuard 用深度学习分析面部/眼睑图像，实现无创胆红素评估。仅供临床决策支持，不替代血清胆红素检测。',
        }
    }

    @classmethod
    def get(cls, k): return cls.T[cls.LANG].get(k, k)

    @classmethod
    def toggle(cls): cls.LANG = 'cn' if cls.LANG == 'en' else 'en'


# ════════════════════════════════════════════════════════════════════
# WORKER THREADS  (backend — unchanged)
# ════════════════════════════════════════════════════════════════════
class ModelLoader(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)

    def run(self):
        try:
            from inference import BilinGuardPredictor
            self.progress.emit(I18n.get('loading'))
            p = BilinGuardPredictor(model_dir=str(MODEL_DIR) if MODEL_DIR.exists() else None)
            self.done.emit(p)
        except Exception as e:
            self.progress.emit(f'Error: {e}')
            self.done.emit(None)


class GradcamWorker(QThread):
    """Compute Grad-CAM + colour analysis + per-model agreement off the UI thread,
    so the interface never freezes during interpretability."""
    done = pyqtSignal(dict)

    def __init__(self, predictor, face_path, eye_path, owner):
        super().__init__()
        self.predictor = predictor
        self.face = face_path
        self.eye = eye_path
        self.owner = owner

    def run(self):
        data = {'face': None, 'eye': None, 'color': None, 'individual': None, 'confidence': None}
        try:
            p = self.predictor
            if self.face and os.path.exists(self.face):
                data['face'] = self.owner._compute_gradcam(p, self.face, 'face')
            if self.eye and os.path.exists(self.eye):
                data['eye'] = self.owner._compute_gradcam(p, self.eye, 'eyelid')
            # colour analysis (numpy only — thread-safe)
            src = self.eye or self.face
            if src and os.path.exists(src):
                from inference import analyze_color
                fb = np.fromfile(src, dtype=np.uint8); im = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                if im is not None:
                    data['color'] = analyze_color(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            # per-binary-model agreement on the processed face (numpy/torch only)
            processed = (data['face'] or {}).get('original')
            models = getattr(p, 'face_binary_models', None)
            if processed is not None and models and HAS_TORCH:
                norm = processed.astype(np.float32) / 255.0
                norm = (norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                t = torch.FloatTensor(np.stack([norm.transpose(2, 0, 1)])).to(p.device)
                individual = []
                with torch.no_grad():
                    for name, m in models:
                        out = m(t)
                        prob = F.softmax(out, dim=1).mean(0).cpu().numpy()
                        pred = I18n.get('jaundiced') if prob[1] > 0.5 else I18n.get('normal')
                        individual.append((name, pred, float(prob[1])))
                if individual:
                    probs = np.array([x[2] for x in individual])
                    ens = float(np.mean(probs)); p0, p1 = 1 - ens, ens
                    entropy = float(-p0 * np.log2(p0 + 1e-10) - p1 * np.log2(p1 + 1e-10))
                    margin = abs(ens - 0.5) * 2
                    preds = [1 if x[2] > 0.5 else 0 for x in individual]
                    agreement = max(preds.count(0), preds.count(1)) / len(preds)
                    data['individual'] = individual
                    data['confidence'] = {'entropy': entropy, 'margin': float(margin), 'agreement': float(agreement)}
        except Exception as e:
            data['error'] = str(e); print('[interp worker]', e, flush=True)
        self.done.emit(data)


# ════════════════════════════════════════════════════════════════════
# WIDGETS
# ════════════════════════════════════════════════════════════════════
class UploadArea(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self, mode='face', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setAcceptDrops(True)
        self.setMinimumHeight(150 if mode == 'face' else 132)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName('upload')
        lay = QVBoxLayout(self); lay.setAlignment(Qt.AlignCenter); lay.setSpacing(8)
        self.icon_lbl = QLabel(); self.icon_lbl.setAlignment(Qt.AlignCenter)
        self.headline = QLabel(); self.headline.setAlignment(Qt.AlignCenter)
        self.hint = QLabel(); self.hint.setAlignment(Qt.AlignCenter); self.hint.setWordWrap(True)
        self.preview = QLabel(); self.preview.setAlignment(Qt.AlignCenter); self.preview.setVisible(False)
        self.filename_lbl = QLabel(); self.filename_lbl.setAlignment(Qt.AlignCenter); self.filename_lbl.setVisible(False)
        # children transparent to mouse so ANY click in the zone opens the file dialog
        for child in (self.icon_lbl, self.headline, self.hint, self.preview, self.filename_lbl):
            child.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay.addWidget(self.icon_lbl); lay.addWidget(self.headline)
        lay.addWidget(self.hint); lay.addWidget(self.preview); lay.addWidget(self.filename_lbl)
        self.update_text()

    def update_text(self):
        showing_preview = self.preview.isVisible()
        self.icon_lbl.setVisible(not showing_preview)
        self.headline.setVisible(not showing_preview)
        self.hint.setVisible(not showing_preview or True)
        if not showing_preview:
            self.icon_lbl.setPixmap(icon('upload', COLORS['primary'], 34).pixmap(34, 34))
            self.headline.setText(I18n.get('drop_or_click'))
            self.headline.setStyleSheet(
                f"color:{COLORS['primary']}; font-size:{FontScale.ts('h3')}px; font-weight:700;")
            self.hint.setText(I18n.get('upload_hint' if self.mode == 'face' else 'upload_optional'))
            self.hint.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:{FontScale.ts('caption')}px;")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv')):
                self.file_dropped.emit(p); break

    def mousePressEvent(self, e):
        p, _ = QFileDialog.getOpenFileName(self, I18n.get('capture'), '',
                                           'Images & Videos (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv)')
        if p:
            self.file_dropped.emit(p)

    def _show_filename(self, path, is_video=False):
        name = os.path.basename(path)
        if len(name) > 28: name = name[:25] + '…'
        self.filename_lbl.setText(('🎬 ' if is_video else '📄 ') + name)
        self.filename_lbl.setStyleSheet(
            f"color:{COLORS['primary_active']}; font-size:{FontScale.ts('small')}px; font-weight:600;")
        self.filename_lbl.setVisible(True)

    def show_preview(self, path):
        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read(); cap.release()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    qi = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                    self.preview.setPixmap(QPixmap.fromImage(qi).scaled(240, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.preview.setVisible(True)
                    self.icon_lbl.setVisible(False); self.headline.setVisible(False); self.hint.setVisible(False)
                    self._show_filename(path, is_video=True); return
        pix = QPixmap(path)
        if not pix.isNull():
            self.preview.setPixmap(pix.scaled(240, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.preview.setVisible(True)
            self.icon_lbl.setVisible(False); self.headline.setVisible(False); self.hint.setVisible(False)
            self._show_filename(path, is_video=False)

    def reset(self):
        self.preview.clear(); self.preview.setVisible(False)
        self.filename_lbl.clear(); self.filename_lbl.setVisible(False)
        self.icon_lbl.setVisible(True); self.headline.setVisible(True); self.hint.setVisible(True)
        self.update_text()


class ProbBar(QWidget):
    def __init__(self, label, value=0, color='#0F766E', parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        row = QHBoxLayout(self); row.setContentsMargins(0, 1, 0, 1); row.setSpacing(10)
        self.lbl = QLabel(label); self.lbl.setFixedWidth(108)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(int(value * 100))
        self.bar.setFormat(f'{value * 100:.1f}%'); self.bar.setFixedHeight(14)
        self._color = color
        row.addWidget(self.lbl); row.addWidget(self.bar, 1)

    def apply(self):
        self.lbl.setStyleSheet(f"color:{COLORS['text_secondary']}; font-size:{FontScale.ts('small')}px;")
        self.bar.setStyleSheet(
            f"QProgressBar {{ border:none; border-radius:8px; background:{COLORS['muted']}; "
            f"font-size:10px; color:{COLORS['text']}; text-align:center; }} "
            f"QProgressBar::chunk {{ background-color:{self._color}; border-radius:8px; }}")


class ResultSection(QFrame):
    def __init__(self, key, title, color, parent=None):
        super().__init__(parent)
        self.key = key; self.color = color
        self.setObjectName('result')
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 13, 16, 14); lay.setSpacing(7)
        self.title_lbl = QLabel(title)
        self.pred_lbl = QLabel('—'); self.pred_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.bars_layout = QVBoxLayout(); self.bars_layout.setSpacing(5)
        self.desc_lbl = QLabel(); self.desc_lbl.setWordWrap(True); self.desc_lbl.setVisible(False)
        lay.addWidget(self.title_lbl); lay.addWidget(self.pred_lbl); lay.addLayout(self.bars_layout); lay.addWidget(self.desc_lbl)

    def set_result(self, prediction, probs=None, desc=None):
        self.pred_lbl.setText(prediction)
        for i in reversed(range(self.bars_layout.count())):
            w = self.bars_layout.itemAt(i).widget()
            if w: w.setParent(None)
        if probs:
            for label, val, color in probs:
                pb = ProbBar(label, val, color); pb.apply(); self.bars_layout.addWidget(pb)
        if desc:
            self.desc_lbl.setText(desc); self.desc_lbl.setVisible(True)

    def set_na(self, reason=None):
        self.pred_lbl.setText(I18n.get('no_data'))
        if reason: self.desc_lbl.setText(reason); self.desc_lbl.setVisible(True)


class SeverityGauge(QWidget):
    """4-segment donut: Normal / Mild / Moderate / Severe."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.severity = -1; self.label = ''; self.value_text = ''

    def set_data(self, severity, label, value_text=''):
        self.severity = severity; self.label = label or ''; self.value_text = value_text or ''; self.update()

    def reset(self):
        self.severity = -1; self.label = ''; self.value_text = ''; self.update()

    def paintEvent(self, _ev):
        from PyQt5.QtGui import QPainter, QColor, QPen, QFont
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        R = min(w, h) / 2 - 12
        rect = QRectF(cx - R, cy - R, R * 2, R * 2)
        seg = [COLORS['normal'], COLORS['mild'], COLORS['moderate'], COLORS['severe']]
        thick = 14
        p.setPen(QPen(QColor(COLORS['border']), thick, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 135 * 16, -270 * 16)
        span = int(-270 * 16 / 4)
        for i in range(4):
            c = QColor(seg[i])
            if self.severity >= 0:
                if i == self.severity:
                    p.setPen(QPen(c, thick + 5, Qt.SolidLine, Qt.RoundCap))
                elif i > self.severity:
                    c2 = QColor(c); c2.setAlphaF(0.22); p.setPen(QPen(c2, thick, Qt.SolidLine, Qt.RoundCap))
                else:
                    p.setPen(QPen(c, thick, Qt.SolidLine, Qt.RoundCap))
            else:
                c2 = QColor(c); c2.setAlphaF(0.22); p.setPen(QPen(c2, thick, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(rect, 135 * 16 + i * span, span)
        if self.severity < 0:
            p.setPen(QColor(COLORS['text_muted']))
            p.setFont(QFont(_FONT, FontScale.ts('small')))
            p.drawText(rect, Qt.AlignCenter, I18n.get('awaiting')); return
        # centre: show the numeric value if present, otherwise the severity label — never both
        centre = self.value_text if self.value_text else self.label
        p.setPen(QColor(seg[self.severity]))
        p.setFont(QFont(_FONT, FontScale.ts('h1'), QFont.Bold))
        p.drawText(QRectF(0, cy - R * 0.5, w, R), Qt.AlignCenter, centre)
        # small line below only when the centre is a number (so the severity word isn't shown twice)
        if self.value_text:
            p.setPen(QColor(COLORS['text']))
            p.setFont(QFont(_FONT, FontScale.ts('small'), QFont.Bold))
            p.drawText(QRectF(0, cy + R * 0.18, w, 24), Qt.AlignCenter, self.label)


# ════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = [None]
        self.face_path = None
        self.eyelid_path = None
        self._last_results = None; self._last_advice = None
        self._models_failed = False
        self.init_ui()
        Theme.register(self._apply_styles)
        self._apply_styles()
        self.load_models()

    # ── layout ─────────────────────────────────────────────────────
    def init_ui(self):
        self.setWindowTitle('BilinGuard  v4.0')
        self.setMinimumSize(1120, 760)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # Header
        self.header = QFrame(); self.header.setObjectName('header'); self.header.setFixedHeight(60)
        hb = QHBoxLayout(self.header); hb.setContentsMargins(28, 0, 22, 0); hb.setSpacing(12)
        self.logo_lbl = ClickableLabel(); self.logo_lbl.setFixedSize(30, 30)
        self.logo_lbl.setCursor(Qt.PointingHandCursor)
        brand = QVBoxLayout(); brand.setSpacing(0); brand.setContentsMargins(0, 0, 0, 0)
        self.brand_name = ClickableLabel('BilinGuard'); self.brand_name.setCursor(Qt.PointingHandCursor)
        self.brand_sub = ClickableLabel(); self.brand_sub.setCursor(Qt.PointingHandCursor)
        for w in (self.brand_name, self.brand_sub): brand.addWidget(w)
        # clicking logo or brand always returns to the analysis page
        for cl in (self.logo_lbl, self.brand_name, self.brand_sub):
            cl.clicked.connect(lambda: self.switch_page('analyze'))
        hb.addWidget(self.logo_lbl); hb.addLayout(brand); hb.addSpacing(8)
        hb.addStretch()
        self.status = QLabel(); self.status.setMinimumWidth(120); self.status.setAlignment(Qt.AlignCenter)
        hb.addWidget(self.status); hb.addSpacing(6)
        self.lang_btn = self._icon_btn('globe'); self.lang_btn.clicked.connect(self.toggle_lang)
        self.theme_btn = self._icon_btn('moon'); self.theme_btn.clicked.connect(self._toggle_theme)
        self.settings_btn = self._icon_btn('settings'); self.settings_btn.clicked.connect(self._toggle_settings)
        for b in (self.lang_btn, self.theme_btn, self.settings_btn): hb.addWidget(b)
        root.addWidget(self.header)

        # Stack
        self.stack = QStackedWidget()
        self.pages = {}

        # ── Analyse page (single clean dashboard column) ──
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        page = QWidget(); L = QVBoxLayout(page)
        L.setContentsMargins(32, 24, 32, 36); L.setSpacing(18)

        # Section: Capture
        cap, cap_body = self._section(I18n.get('capture'), I18n.get('capture_sub'))
        uploads = QHBoxLayout(); uploads.setSpacing(14)
        self.face_box = QFrame(); self.face_box.setObjectName('uploadwrap')
        fb = QVBoxLayout(self.face_box); fb.setContentsMargins(12, 12, 12, 12); fb.setSpacing(6)
        self.face_title = QLabel(I18n.get('upload_title')); fb.addWidget(self.face_title)
        self.upload = UploadArea('face'); self.upload.file_dropped.connect(self._on_face); fb.addWidget(self.upload)
        uploads.addWidget(self.face_box)
        self.eye_box = QFrame(); self.eye_box.setObjectName('uploadwrap')
        eb = QVBoxLayout(self.eye_box); eb.setContentsMargins(12, 12, 12, 12); eb.setSpacing(6)
        self.eye_title = QLabel(I18n.get('upload_eyelid')); eb.addWidget(self.eye_title)
        self.upload_eye = UploadArea('eye'); self.upload_eye.file_dropped.connect(self._on_eyelid); eb.addWidget(self.upload_eye)
        uploads.addWidget(self.eye_box)
        cap_body.addLayout(uploads)
        # buttons + progress
        act = QHBoxLayout(); act.setSpacing(10)
        self.analyze_btn = QPushButton(); self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.setIcon(icon('analyze', COLORS['on_primary'], 18))
        self.analyze_btn.setEnabled(False); self.analyze_btn.clicked.connect(self.run_all)
        act.addWidget(self.analyze_btn, 1)
        self.clear_btn = QPushButton(); self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.setIcon(icon('trash', COLORS['text_secondary'], 17)); self.clear_btn.clicked.connect(self.clear_all)
        self.clear_btn.setFixedWidth(96); act.addWidget(self.clear_btn)
        cap_body.addLayout(act)
        self.progress = QProgressBar(); self.progress.setVisible(False); self.progress.setFixedHeight(6); cap_body.addWidget(self.progress)
        self.progress_label = QLabel(); self.progress_label.setVisible(False); cap_body.addWidget(self.progress_label)
        L.addWidget(cap)

        # Section: Clinical context
        ctx, cl = self._section(I18n.get('ctx_title'), I18n.get('ctx_hint'))
        r1 = QHBoxLayout(); r1.setSpacing(10)
        self.dept_label = QLabel(I18n.get('ctx_dept')); r1.addWidget(self.dept_label)
        self.dept_combo = QComboBox(); self.dept_combo.setMinimumWidth(180); self._populate_dept_combo(); r1.addWidget(self.dept_combo)
        r1.addSpacing(6)
        self.dx_label = QLabel(I18n.get('ctx_dx')); r1.addWidget(self.dx_label)
        self.dx_edit = QLineEdit(); self.dx_edit.setPlaceholderText(I18n.get('ctx_dx_ph')); r1.addWidget(self.dx_edit, 1)
        cl.addLayout(r1)
        self.labs_title_lbl = QLabel(I18n.get('ctx_labs')); cl.addWidget(self.labs_title_lbl)
        self.lab_fields = {}
        grid = QGridLayout(); grid.setSpacing(6); grid.setHorizontalSpacing(8)
        # (key, label, unit, reference_range, clinically_critical)
        self._lab_specs = [('tbil', 'TBIL', 'μmol/L', '3.4–17.1', '>85'),
                           ('dbil', 'DBIL', 'μmol/L', '0–6.8', '>68'),
                           ('ibil', 'IBIL', 'μmol/L', '1.7–10.2', '>50'),
                           ('alt', 'ALT', 'U/L', '7–40', '>400'),
                           ('ast', 'AST', 'U/L', '8–40', '>400'),
                           ('alp', 'ALP', 'U/L', '40–150', '>400'),
                           ('ggt', 'GGT', 'U/L', '7–45', '>200'),
                           ('inr', 'INR', '', '0.8–1.2', '>1.5'),
                           ('albumin', 'ALB', 'g/L', '35–55', '<30'),
                           ('nrs2002', 'NRS-2002', '', '0–7', '>7')]
        for i, (k, lab, unit, ref, crit) in enumerate(self._lab_specs):
            cell = QHBoxLayout(); cell.setSpacing(4)
            l = QLabel(lab); l.setObjectName('labtag'); cell.addWidget(l)
            ed = QLineEdit(); ed.setPlaceholderText(unit); ed.setMaximumWidth(64); ed.setObjectName('lab'); cell.addWidget(ed)
            # Tooltip with reference range and critical threshold (bilingual)
            tip_cn = f'{lab}（{unit}）\n参考范围: {ref}\n临床警戒: {crit}'
            tip_en = f'{lab} ({unit})\nReference: {ref}\nCritical: {crit}'
            ed.setToolTip(tip_cn if I18n.LANG == 'cn' else tip_en)
            # Live validation: highlight abnormal values
            ed.textChanged.connect(lambda _t, e=ed, k=k, r=ref, c=crit: self._validate_lab(e, k, r, c))
            self.lab_fields[k] = ed
            wrap = QWidget(); wrap.setLayout(cell); grid.addWidget(wrap, i // 5, i % 5)
        # Reference range hint row
        hint = QLabel('💡 ' + ('悬停查看参考范围 · 异常值将自动标红' if I18n.LANG == 'cn'
                               else 'Hover for reference ranges · abnormal values flagged'))
        hint.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:{FontScale.ts('small')}px; padding:2px 0;")
        cl.addWidget(hint)
        cl.addLayout(grid)
        L.addWidget(ctx)

        # Section: Assessment (hero + result sections)
        self.results_section, rb = self._section(I18n.get('results'), None)
        rb.setSpacing(14)
        hero = QHBoxLayout(); hero.setSpacing(18)
        self.gauge = SeverityGauge(); self.gauge.setFixedWidth(160); hero.addWidget(self.gauge, 0, Qt.AlignVCenter)
        rhs = QVBoxLayout(); rhs.setSpacing(8)
        self.severity_hero_title = QLabel(I18n.get('severity_hero'))
        self.type_badge = QLabel(I18n.get('type_badge_none')); self.type_badge.setObjectName('badge'); self.type_badge.setAlignment(Qt.AlignCenter)
        self.triage_summary = QLabel(I18n.get('awaiting')); self.triage_summary.setWordWrap(True)
        # CDSS v3 agent self-check meta (mode / priority / confidence / rules / nutrition)
        self.agent_meta = QLabel(''); self.agent_meta.setWordWrap(True)
        self.agent_meta.setTextFormat(Qt.RichText)
        self.agent_meta.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:{FontScale.ts('small')}px;")
        rhs.addWidget(self.severity_hero_title); rhs.addWidget(self.type_badge)
        rhs.addWidget(self.triage_summary); rhs.addWidget(self.agent_meta); rhs.addStretch()
        hero.addLayout(rhs, 1)
        rb.addLayout(hero)
        # result sections grid — clean 2-column layout, no empty cells.
        # 7 cards: screen/grade, dbil/ibil, type/cp, meld (spans full width)
        self.result_sections = {}
        secs = [('screen', COLORS['primary']), ('grade', '#D97706'),
                ('dbil', '#0369A1'), ('ibil', '#7C3AED'),
                ('type', '#0EA5E9'), ('cp', '#CA8A04'), ('meld', COLORS['danger'])]
        gridw = QWidget(); self.results_grid = QGridLayout(gridw)
        self.results_grid.setSpacing(10); self.results_grid.setContentsMargins(0, 0, 0, 0)
        self.results_grid.setColumnStretch(0, 1); self.results_grid.setColumnStretch(1, 1)
        for i, (key, color) in enumerate(secs):
            sec = ResultSection(key, I18n.get(f'section_{key}'), color)
            if i < len(secs) - 1:
                self.results_grid.addWidget(sec, i // 2, i % 2)        # 2 per row
            else:
                self.results_grid.addWidget(sec, i // 2, 0, 1, 2)      # last card spans both columns
            self.result_sections[key] = sec
        rb.addWidget(gridw)
        self.time_label = QLabel(); rb.addWidget(self.time_label)
        L.addWidget(self.results_section)

        # Section: Clinical Guidance
        self.advice_section, ab = self._section(I18n.get('advice_title'), None)
        self.advice_empty = QLabel(I18n.get('advidence_empty')); ab.addWidget(self.advice_empty)
        self.advice_container = QFrame(); self.advice_container.setStyleSheet('background:transparent;')
        self.advice_layout = QVBoxLayout(self.advice_container); self.advice_layout.setContentsMargins(0, 0, 0, 0); self.advice_layout.setSpacing(10)
        ab.addWidget(self.advice_container)
        L.addWidget(self.advice_section)

        # Section: Interpretability
        self.interp_section, ib = self._section(I18n.get('interp_title'), None)
        ib.setSpacing(12)
        self.face_interp_title = QLabel(I18n.get('face_analysis')); self.face_interp_title.setVisible(False); ib.addWidget(self.face_interp_title)
        self.face_images_row = QHBoxLayout(); self.face_images_row.setSpacing(10)
        self.img_face_original, self.img_face_overlay, self.img_face_heatmap = self._img_panel(), self._img_panel(), self._img_panel()
        for w in (self.img_face_original, self.img_face_overlay, self.img_face_heatmap):
            box = QFrame(); box.setObjectName('imgpanel'); bx = QVBoxLayout(box); bx.setContentsMargins(8, 8, 8, 8)
            bx.addWidget(w); self.face_images_row.addWidget(box)
        self.face_images_widget = QWidget(); self.face_images_widget.setLayout(self.face_images_row); self.face_images_widget.setVisible(False)
        ib.addWidget(self.face_images_widget)
        self.eye_interp_title = QLabel(I18n.get('eyelid_analysis')); self.eye_interp_title.setVisible(False); ib.addWidget(self.eye_interp_title)
        self.eye_images_row = QHBoxLayout(); self.eye_images_row.setSpacing(10)
        self.img_eye_original, self.img_eye_overlay, self.img_eye_heatmap = self._img_panel(), self._img_panel(), self._img_panel()
        for w in (self.img_eye_original, self.img_eye_overlay, self.img_eye_heatmap):
            box = QFrame(); box.setObjectName('imgpanel'); bx = QVBoxLayout(box); bx.setContentsMargins(8, 8, 8, 8)
            bx.addWidget(w); self.eye_images_row.addWidget(box)
        self.eye_images_widget = QWidget(); self.eye_images_widget.setLayout(self.eye_images_row); self.eye_images_widget.setVisible(False)
        ib.addWidget(self.eye_images_widget)
        bot = QHBoxLayout(); bot.setSpacing(12)
        cbox = QFrame(); cbox.setObjectName('imgpanel'); ccol = QVBoxLayout(cbox)
        self.color_title = QLabel(I18n.get('color_title')); ccol.addWidget(self.color_title)
        self.color_bars_layout = QVBoxLayout(); self.color_bars_layout.setSpacing(3); ccol.addLayout(self.color_bars_layout)
        gbox = QFrame(); gbox.setObjectName('imgpanel'); gcol = QVBoxLayout(gbox)
        self.agreement_title = QLabel(I18n.get('agreement_title')); gcol.addWidget(self.agreement_title)
        self.agreement_layout = QVBoxLayout(); self.agreement_layout.setSpacing(3); gcol.addLayout(self.agreement_layout)
        self.confidence_label = QLabel('—'); gcol.addWidget(self.confidence_label)
        bot.addWidget(cbox, 1); bot.addWidget(gbox, 1); ib.addLayout(bot)
        L.addWidget(self.interp_section)

        L.addStretch()
        scroll.setWidget(page)
        self.stack.addWidget(scroll); self.pages['analyze'] = scroll

        # Settings
        sp = self._make_settings(); self.stack.addWidget(sp); self.pages['settings'] = sp
        root.addWidget(self.stack, 1)
        self.switch_page('analyze')
        FontScale.refresh_callback = self._refresh_fonts

    def _icon_btn(self, name):
        b = QPushButton(); b.setFixedSize(38, 38); b.setCursor(Qt.PointingHandCursor)
        b.setObjectName('iconbtn'); b._icon_name = name; return b

    def _img_panel(self):
        lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter); lbl.setMinimumSize(150, 150); lbl.setText('—'); return lbl

    def _soft_shadow(self, widget, blur=26, y=4, alpha=38):
        """Apply a hue-tinted soft shadow (Soft-UI depth). Qt QSS has no box-shadow."""
        eff = QGraphicsDropShadowEffect(widget)
        eff.setBlurRadius(blur); eff.setOffset(0, y)
        # tint the shadow toward the primary teal
        from PyQt5.QtGui import QColor
        base = QColor(COLORS['primary']); base.setAlpha(alpha)
        eff.setColor(base)
        widget.setGraphicsEffect(eff)

    def _section(self, title, subtitle):
        """Returns (frame, body_layout). The frame holds the section header (title/subtitle)
        plus a styled card body; add section content to body_layout."""
        f = QFrame(); f.setObjectName('section')
        outer = QVBoxLayout(f); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(6)
        t = QLabel(title); t.setObjectName('sectiontitle'); outer.addWidget(t)
        if subtitle:
            s = QLabel(subtitle); s.setObjectName('sectionsub'); s.setWordWrap(True); outer.addWidget(s)
        body = QFrame(); body.setObjectName('card')
        self._soft_shadow(body)
        body_lay = QVBoxLayout(body); body_lay.setContentsMargins(18, 14, 18, 16); body_lay.setSpacing(10)
        outer.addWidget(body)
        f.body = body
        return f, body_lay

    # ── styling (unified, theme-aware) ─────────────────────────────
    def _apply_styles(self):
        c = COLORS
        if FontScale.app is not None:
            FontScale.app.setStyleSheet(self._global_qss())
        self.setStyleSheet(f"QMainWindow {{ background-color:{c['bg']}; }}")
        # header
        self.header.setStyleSheet(
            f"QFrame#header {{ background-color:{c['header']}; border-bottom:1px solid {c['border']}; }}")
        self.logo_lbl.setPixmap(icon('logo', c['primary'], 30).pixmap(30, 30))
        self.brand_name.setStyleSheet(f"color:{c['text']}; font-size:{FontScale.ts('h2')}px; font-weight:700; letter-spacing:0.3px;")
        self.brand_sub.setText(I18n.get('subtitle'))
        self.brand_sub.setStyleSheet(f"color:{c['text_muted']}; font-size:{FontScale.ts('caption')}px;")
        self.status.setStyleSheet(
            f"QLabel {{ background-color:{c['soft']}; color:{c['primary_active']}; border-radius:12px; "
            f"padding:0 12px; font-size:{FontScale.ts('caption')}px; font-weight:600; }}")
        for b in (self.lang_btn, self.theme_btn, self.settings_btn):
            ic = b._icon_name
            if ic == 'moon': ic = 'sun' if Theme.mode == 'dark' else 'moon'
            col = c['text_secondary']
            b.setIcon(icon(ic, col, 18)); b.setIconSize(__import__('PyQt5').QtCore.QSize(18, 18))
            b.setStyleSheet(
                f"QPushButton#iconbtn {{ background:transparent; border:none; border-radius:8px; }} "
                f"QPushButton#iconbtn:hover {{ background-color:{c['muted']}; }}")
        # global QSS for structural widgets
        q = f"""
        QFrame#section {{ background:transparent; }}
        QLabel#sectiontitle {{ color:{c['text']}; font-size:{FontScale.ts('h1')}px; font-weight:700; letter-spacing:-0.2px; }}
        QLabel#sectionsub {{ color:{c['text_secondary']}; font-size:{FontScale.ts('small')}px; }}
        QFrame#uploadwrap {{ background-color:{c['surface']}; border:1px solid {c['border']}; border-radius:12px; }}
        QFrame#upload {{ background-color:{c['soft']}; border:2px dashed {c['primary']}; border-radius:12px; }}
        QFrame#upload:hover {{ border-style:solid; background-color:{c['primary']}; }}
        QFrame#upload:hover > QLabel {{ }}
        QLabel#result, QLabel#imgpanel {{ }} 
        ResultSection {{ background-color:{c['surface']}; border:1px solid {c['border']}; border-radius:12px; border-left:4px solid {c['primary']}; }}
        QFrame#imgpanel {{ background-color:{c['surface_alt']}; border:1px solid {c['border']}; border-radius:8px; }}
        QFrame#badge-as QLabel#badge {{ }}
        """
        self.centralWidget().setStyleSheet(self.centralWidget().styleSheet().split('/*GEN*/')[0] + '/*GEN*/' + q)
        # text widgets
        self.face_title.setStyleSheet(f"color:{c['primary']}; font-size:{FontScale.ts('small')}px; font-weight:600;")
        self.eye_title.setStyleSheet(f"color:{c['accent']}; font-size:{FontScale.ts('small')}px; font-weight:600;")
        for lbl, key in ((self.dept_label, 'ctx_dept'), (self.dx_label, 'ctx_dx'), (self.labs_title_lbl, 'ctx_labs')):
            lbl.setText(I18n.get(key))
            lbl.setStyleSheet(f"color:{c['text_secondary']}; font-size:{FontScale.ts('small')}px; font-weight:600;")
        self.analyze_btn.setText('  ' + I18n.get('analyze'))
        self.analyze_btn.setStyleSheet(
            f"QPushButton {{ background-color:{c['primary']}; color:{c['on_primary']}; border:none; border-radius:8px; "
            f"font-size:{FontScale.ts('h3')}px; font-weight:600; padding:0 18px; text-align:left; }} "
            f"QPushButton:hover {{ background-color:{c['primary_hover']}; }} "
            f"QPushButton:pressed {{ background-color:{c['primary_active']}; }} "
            f"QPushButton:disabled {{ background-color:{c['muted']}; color:{c['text_secondary']}; }}")
        self.clear_btn.setText('  ' + I18n.get('clear'))
        self.clear_btn.setStyleSheet(
            f"QPushButton {{ background-color:{c['surface']}; color:{c['text_secondary']}; border:1px solid {c['border_strong']}; "
            f"border-radius:8px; font-size:{FontScale.ts('small')}px; text-align:left; }} "
            f"QPushButton:hover {{ background-color:{c['muted']}; color:{c['text']}; }}")
        self.progress.setStyleSheet(f"QProgressBar {{ border:none; background:{c['border']}; border-radius:6px; }} QProgressBar::chunk {{ background:{c['primary']}; border-radius:6px; }}")
        self.progress_label.setStyleSheet(f"color:{c['text_secondary']}; font-size:{FontScale.ts('caption')}px;")
        self.time_label.setStyleSheet(f"color:{c['text_muted']}; font-size:{FontScale.ts('caption')}px;")
        inq = (f"QComboBox, QLineEdit {{ padding:6px 10px; border:1px solid {c['border_strong']}; border-radius:8px; "
               f"background:{c['surface']}; color:{c['text']}; font-size:{FontScale.ts('small')}px; }} "
               f"QLineEdit#lab {{ padding:4px 6px; border:1px solid {c['border']}; border-radius:6px; max-width:64px; }} "
               f"QLabel#labtag {{ color:{c['text_muted']}; font-size:10px; }} "
               f"QComboBox::drop-down {{ border:none; width:20px; }}")
        self.dept_combo.setStyleSheet(inq); self.dx_edit.setStyleSheet(inq)
        for ed in self.lab_fields.values(): ed.setStyleSheet(inq)
        # hero
        self.severity_hero_title.setStyleSheet(f"color:{c['text_muted']}; font-size:{FontScale.ts('caption')}px; font-weight:600; letter-spacing:0.5px;")
        self.type_badge.setStyleSheet(
            f"QLabel#badge {{ background:{c['soft']}; color:{c['primary_active']}; border-radius:8px; "
            f"padding:4px 10px; font-size:{FontScale.ts('small')}px; font-weight:700; max-width:120px; }}")
        self.triage_summary.setStyleSheet(f"color:{c['text']}; font-size:{FontScale.ts('h3')}px; font-weight:600;")
        # result sections
        for sec in self.result_sections.values():
            sec.setStyleSheet(
                f"ResultSection {{ background-color:{c['surface']}; border:1px solid {c['border']}; border-radius:12px; border-left:4px solid {sec.color}; }}")
            sec.title_lbl.setStyleSheet(f"color:{sec.color}; font-size:{FontScale.ts('small')}px; font-weight:700; letter-spacing:0.3px;")
            sec.pred_lbl.setStyleSheet(f"color:{c['text']}; font-size:{FontScale.ts('h1')}px; font-weight:700;")
            sec.desc_lbl.setStyleSheet(f"color:{c['text_secondary']}; font-size:{FontScale.ts('small')}px;")
        for lbl, col in ((self.face_interp_title, c['primary']), (self.eye_interp_title, c['accent']),
                         (self.color_title, c['accent']), (self.agreement_title, c['primary'])):
            lbl.setStyleSheet(f"color:{col}; font-size:{FontScale.ts('small')}px; font-weight:600;")
        self.confidence_label.setStyleSheet(f"color:{c['text_secondary']}; font-size:{FontScale.ts('small')}px;")
        # re-apply prob-bar styles (result sections + colour analysis) so they follow font/theme
        for holder in (list(self.result_sections.values()) + [None]):
            if holder is None:
                lay = getattr(self, 'color_bars_layout', None)
            else:
                lay = holder.bars_layout
            if lay is None: continue
            for i in range(lay.count()):
                pb = lay.itemAt(i).widget()
                if pb is not None and hasattr(pb, 'apply'): pb.apply()
        self.gauge.update()
        self.advice_empty.setStyleSheet(f"color:{c['text_secondary']}; font-size:{FontScale.ts('small')}px; padding:4px;")
        # re-render the clinical guidance so its cards follow font + theme changes
        if getattr(self, '_last_results', None) is not None:
            try: self._render_advice(self._last_results)
            except Exception: pass

    def _global_qss(self):
        c = COLORS
        return f"""
        QWidget {{ color:{c['text']}; font-family:'{_FONT}'; }}
        /* visible keyboard focus (replaces the removed default outline) */
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border:2px solid {c['primary']}; }}
        QPushButton:focus {{ border:2px solid {c['primary']}; }}
        QTextEdit:focus, QPlainTextEdit:focus {{ border:2px solid {c['primary']}; }}
        QScrollBar:vertical {{ background:transparent; width:12px; margin:2px; }}
        QScrollBar::handle:vertical {{ background:{c['border_strong']}; border-radius:6px; min-height:40px; }}
        QScrollBar::handle:vertical:hover {{ background:{c['text_muted']}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        QScrollBar:horizontal {{ background:transparent; height:12px; margin:2px; }}
        QScrollBar::handle:horizontal {{ background:{c['border_strong']}; border-radius:6px; min-width:40px; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
        QToolTip {{ background:{c['text']}; color:{c['surface']}; border:none; padding:6px 10px; border-radius:6px; }}
        QComboBox QAbstractItemView {{ background:{c['surface']}; color:{c['text']}; border:1px solid {c['border']}; selection-background-color:{c['soft']}; selection-color:{c['primary_active']}; }}
        """

    def _toggle_theme(self):
        Theme.toggle()

    # ── nav / lang / settings ──────────────────────────────────────
    def _toggle_settings(self):
        on_settings = self.stack.currentWidget() is self.pages.get('settings')
        self.switch_page('analyze' if on_settings else 'settings')

    def switch_page(self, key):
        if key in self.pages:
            self.stack.setCurrentWidget(self.pages[key])
        # settings button becomes a "back to analysis" arrow when on the settings page
        on_settings = self.stack.currentWidget() is self.pages.get('settings')
        if hasattr(self, 'settings_btn'):
            self.settings_btn._icon_name = 'arrow_left' if on_settings else 'settings'
            ic = self.settings_btn._icon_name
            self.settings_btn.setIcon(icon(ic, COLORS['text_secondary'], 18))
            self.settings_btn.setToolTip(I18n.get('analyze') if on_settings else I18n.get('settings'))

    def toggle_lang(self):
        I18n.toggle()
        self.setWindowTitle(I18n.get('title') + '  v4.0')
        self.lang_btn.setToolTip('中文/EN' if I18n.LANG == 'cn' else 'English / 中文')
        # keep the settings-button tooltip in sync if currently on the settings page
        on_settings = self.stack.currentWidget() is self.pages.get('settings')
        if hasattr(self, 'settings_btn'):
            self.settings_btn.setToolTip(I18n.get('analyze') if on_settings else I18n.get('settings'))
        # refresh every language-dependent label, then re-apply visual styles
        self._refresh_i18n()
        self._apply_styles()   # _apply_styles re-renders the guidance cards too

    def _refresh_i18n(self):
        """Re-apply every language-dependent string after I18n.LANG changes."""
        # section titles (影像采集 / 临床背景 / 评估结果 / 临床指导 / 模型可解释性 + 5 result sections)
        self._retitle_sections()
        # capture section: upload zones, box titles, placeholder, department combo
        self.upload.update_text(); self.upload_eye.update_text()
        self.face_title.setText(I18n.get('upload_title')); self.eye_title.setText(I18n.get('upload_eyelid'))
        self.dx_edit.setPlaceholderText(I18n.get('ctx_dx_ph'))
        self._populate_dept_combo()   # preserves the current selection
        # section subtitles (only capture & clinical-context sections carry one)
        page = self.pages['analyze'].widget()
        subs = [I18n.get('capture_sub'), I18n.get('ctx_hint')]
        si = 0
        for i in range(page.layout().count()):
            it = page.layout().itemAt(i)
            w = it.widget() if it else None
            if w is not None and w.objectName() == 'section' and si < len(subs):
                s = w.findChild(QLabel, 'sectionsub')
                if s is not None:
                    s.setText(subs[si]); si += 1
        # assessment hero
        self.severity_hero_title.setText(I18n.get('severity_hero'))
        if getattr(self, '_last_results', None) is None:
            self.triage_summary.setText(I18n.get('awaiting'))
            self.type_badge.setText(I18n.get('type_badge_none'))
            self.advice_empty.setText(I18n.get('advidence_empty'))
        # interpretability titles
        self.face_interp_title.setText(I18n.get('face_analysis'))
        self.eye_interp_title.setText(I18n.get('eyelid_analysis'))
        self.color_title.setText(I18n.get('color_title'))
        self.agreement_title.setText(I18n.get('agreement_title'))
        # settings page
        if hasattr(self, '_settings_title'):
            self._settings_title.setText(I18n.get('settings'))
            self._appearance_box.setTitle(I18n.get('appearance'))
            self.font_label.setText(I18n.get('font_size'))
            for b, key, _val in self.preset_btns:
                b.setText(I18n.get(key))
            self._reset_btn.setText(I18n.get('reset'))
            self.about_text.setPlainText(I18n.get('about'))
            self._model_box.setTitle('Model Information' if I18n.LANG == 'en' else '模型信息')
            self._update_model_info()
        # result sections + advice (skip Grad-CAM on a pure language refresh)
        if getattr(self, '_last_results', None) is not None:
            self._show_results(self._last_results, refresh_interp=False)
        # header status pill
        self._refresh_status()

    def _refresh_status(self):
        """Re-localise the header status pill after a language switch."""
        if getattr(self, '_model_counts', None):
            c = self._model_counts
            self.status.setText('✓ ' + I18n.get('models') + f' ({c["total"]}' + ('+SAM)' if c['sam'] else ')'))
        elif getattr(self, '_models_failed', False):
            self.status.setText('⚠ ' + ('Demo' if I18n.LANG == 'en' else '演示模式'))

    def _retitle_sections(self):
        # map: find section frames by walking the analyse page layout
        page = self.pages['analyze'].widget()
        titles = [I18n.get('capture'), I18n.get('ctx_title'), I18n.get('results'),
                  I18n.get('advice_title'), I18n.get('interp_title')]
        lay = page.layout()
        idx = 0
        for i in range(lay.count()):
            it = lay.itemAt(i)
            w = it.widget() if it else None
            if w is not None and w.objectName() == 'section' and idx < len(titles):
                t = w.findChild(QLabel, 'sectiontitle')
                if t: t.setText(titles[idx])
                idx += 1
        for key, sec in self.result_sections.items():
            sec.title_lbl.setText(I18n.get(f'section_{key}'))

    def _make_settings(self):
        page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(36, 28, 36, 28); lay.setSpacing(16)
        self._settings_title = QLabel(I18n.get('settings'))
        self._settings_title.setObjectName('sectiontitle')
        lay.addWidget(self._settings_title)
        self._appearance_box = box = QGroupBox(I18n.get('appearance'))
        bl = QVBoxLayout(box); bl.setSpacing(12)
        row = QHBoxLayout()
        self.font_label = QLabel(I18n.get('font_size')); row.addWidget(self.font_label)
        self.font_value = QLabel(f'{FontScale.current}pt'); row.addWidget(self.font_value); row.addStretch()
        bl.addLayout(row)
        self.font_slider = QSlider(Qt.Horizontal); self.font_slider.setRange(FontScale.MIN, FontScale.MAX)
        self.font_slider.setValue(FontScale.current); self.font_slider.valueChanged.connect(self._on_font_change)
        bl.addWidget(self.font_slider)
        prow = QHBoxLayout(); self.preset_btns = []
        for key, val in [('font_small', 11), ('font_medium', 13),
                         ('font_large', 16), ('font_xlarge', 19)]:
            b = QPushButton(I18n.get(key)); b.setFixedHeight(34)
            b.clicked.connect(lambda _, v=val: self.font_slider.setValue(v))
            prow.addWidget(b); self.preset_btns.append((b, key, val))
        bl.addLayout(prow)
        self._reset_btn = rb = QPushButton(I18n.get('reset')); rb.setFixedHeight(34); rb.setFixedWidth(140)
        rb.clicked.connect(lambda: self.font_slider.setValue(FontScale.BASE)); bl.addWidget(rb)
        lay.addWidget(box)
        about = QGroupBox(I18n.get('title')); self._about_box = about
        al = QVBoxLayout(about)
        self.about_text = QTextEdit(); self.about_text.setPlainText(I18n.get('about'))
        self.about_text.setReadOnly(True); self.about_text.setMaximumHeight(110); al.addWidget(self.about_text); lay.addWidget(about)
        m = QGroupBox('Model Information' if I18n.LANG == 'en' else '模型信息'); self._model_box = m
        ml = QVBoxLayout(m)
        self.model_info = QLabel(I18n.get('loading')); ml.addWidget(self.model_info); lay.addWidget(m)
        lay.addStretch()
        return page

    def _on_font_change(self, val):
        FontScale.set(val)
        self.font_value.setText(f'{val}pt')

    def _refresh_fonts(self):
        if hasattr(self, 'font_value'):
            self.font_value.setText(f'{FontScale.current}pt')
        self._apply_styles()

    # ── capture ────────────────────────────────────────────────────
    def _on_face(self, path):
        self.face_path = path; self.upload.show_preview(path); self._check_ready()

    def _on_eyelid(self, path):
        self.eyelid_path = path; self.upload_eye.show_preview(path); self._check_ready()

    def _check_ready(self):
        self.analyze_btn.setEnabled(self.face_path is not None and self.predictor[0] is not None)

    def clear_all(self):
        self.face_path = None; self.eyelid_path = None
        self.upload.reset(); self.upload_eye.reset()
        self.analyze_btn.setEnabled(False)
        for sec in self.result_sections.values(): sec.setVisible(True); sec.set_na()
        self._last_results = None; self._last_advice = None
        self.gauge.reset(); self.type_badge.setText(I18n.get('type_badge_none'))
        self.triage_summary.setText(I18n.get('awaiting'))
        self._clear_layout(self.advice_layout)
        self.advice_empty.setText(I18n.get('advidence_empty'))
        self.advice_layout.addWidget(self.advice_empty)

    # ── clinical-context helpers ───────────────────────────────────
    def _populate_dept_combo(self):
        prev = self.dept_combo.currentData() if hasattr(self, 'dept_combo') else None
        self.dept_combo.blockSignals(True); self.dept_combo.clear()
        self.dept_combo.addItem(I18n.get('ctx_dept_auto'), '__auto__')
        for did, info in DEPARTMENTS.items():
            self.dept_combo.addItem(info.get(f'name_{I18n.LANG}', info.get('name_en', did)), did)
        idx = 0
        if prev is not None:
            for i in range(self.dept_combo.count()):
                if self.dept_combo.itemData(i) == prev:
                    idx = i; break
        self.dept_combo.setCurrentIndex(idx)
        self.dept_combo.blockSignals(False)

    def _read_labs(self):
        labs = {}
        for spec in getattr(self, '_lab_specs', []):
            key = spec[0]
            ed = self.lab_fields.get(key)
            if ed is None: continue
            txt = ed.text().strip()
            if not txt: continue
            try: labs[key] = float(txt)
            except ValueError: pass
        return labs

    def _validate_lab(self, edit, key, ref_range, critical):
        """Live-validate a lab input: parse ref/critical thresholds and color the field."""
        txt = edit.text().strip()
        if not txt:
            edit.setStyleSheet('')  # default
            return
        try:
            val = float(txt)
        except ValueError:
            edit.setStyleSheet('')  # default
            return
        # Parse reference range "lo–hi" (en dash or hyphen)
        import re as _re
        nums = [float(x) for x in _re.split(r'[–\-]', ref_range) if x.strip()[:1].isdigit()]
        lo, hi = (nums[0], nums[1]) if len(nums) >= 2 else (None, None)
        # Parse critical threshold (e.g. '>85' or '<30')
        crit_op, crit_val = None, None
        m = _re.match(r'\s*([<>=≤≥])\s*(\d+(?:\.\d+)?)', critical)
        if m:
            crit_op = m.group(1); crit_val = float(m.group(2))
        # Decide color
        danger = COLORS.get('danger', '#DC2626')
        warn = COLORS.get('moderate', '#F59E0B')
        is_critical = False
        if crit_op in ('>', '≥') and crit_val is not None and val > crit_val:
            is_critical = True
        elif crit_op in ('<', '≤') and crit_val is not None and val < crit_val:
            is_critical = True
        if is_critical:
            edit.setStyleSheet(f"QLineEdit#lab {{ border:1.5px solid {danger}; background:{danger}11; }}")
        elif (lo is not None and val < lo) or (hi is not None and val > hi):
            edit.setStyleSheet(f"QLineEdit#lab {{ border:1.5px solid {warn}; background:{warn}11; }}")
        else:
            edit.setStyleSheet('')  # normal

    def _read_dept(self):
        if not HAS_ADVISOR: return None
        data = self.dept_combo.currentData()
        return None if (data is None or data == '__auto__') else data

    def _clear_layout(self, layout):
        if layout is None: return
        for i in reversed(range(layout.count())):
            w = layout.itemAt(i).widget()
            if w is not None: w.setParent(None)

    def _format_advice_content(self, content):
        lab = {'mechanism': '机制', 'pattern': '胆红素模式', 'lab_pattern': '化验模式',
               'department': '科室', 'principle': '处理原则', 'focus': '重点', 'specialty': '专科',
               'referral': '转诊', 'causes': '常见病因', 'workup': '检查', 'on_jaundice': '处置',
                'jaundice_type': '黄疸类型', 'suggested_department': '建议科室', 'type_decision': '按黄疸类型决策',
                'label': '级别', 'tbil': '胆红素', 'management': '处理', 'follow_up': '随访', 'lifestyle': '生活',
                'child_pugh': 'Child-Pugh', 'meld': 'MELD', 'implication': '临床含义'}
        if isinstance(content, str): return content.replace('\n', '<br>')
        if isinstance(content, list):
            return '<ul style="margin:4px 0;">' + ''.join(f'<li>{it}</li>' for it in content) + '</ul>' if content else ''
        if isinstance(content, dict):
            parts = []
            for k, v in content.items():
                if isinstance(v, list):
                    if not v: continue
                    parts.append(f'<b>{lab.get(k,k)}</b><ul style="margin:2px 0;">' + ''.join(f'<li>{it}</li>' for it in v) + '</ul>')
                else:
                    parts.append(f'<b>{lab.get(k,k)}:</b> {v}')
            return '<br>'.join(parts)
        return str(content)

    # ── models ─────────────────────────────────────────────────────
    def load_models(self):
        self.loader = ModelLoader()
        self.loader.progress.connect(lambda m: self.status.setText('⏳ ' + m))
        self.loader.done.connect(self._on_models)
        self.loader.start()

    def _on_models(self, predictor):
        self.predictor[0] = predictor
        if predictor and (predictor.is_trained or getattr(predictor, 'face_binary_models', None)):
            n_eye = len(predictor.models)
            n_bin = len(getattr(predictor, 'face_binary_models', []))
            n_ftern = len(getattr(predictor, 'face_ternary_models', []))
            n_type = len(getattr(predictor, 'type_class_models', []))
            n_dbil_e = len(getattr(predictor, 'dbil_eyelid_models', []))
            n_dbil_f = len(getattr(predictor, 'dbil_face_models', []))
            n_ibil_e = len(getattr(predictor, 'ibil_eyelid_models', []))
            n_ibil_f = len(getattr(predictor, 'ibil_face_models', []))
            n_cp = len(getattr(predictor, 'cp_face_models', []))
            n_meld = len(getattr(predictor, 'meld_face_models', []))
            total = n_eye + n_bin + n_ftern + n_type + n_dbil_e + n_dbil_f + n_ibil_e + n_ibil_f + n_cp + n_meld
            sam = getattr(predictor, 'sam_model', None) is not None
            self.status.setText('✓ ' + I18n.get('models') + f' ({total}' + ('+SAM)' if sam else ')'))
            dev = 'CUDA' if HAS_TORCH and torch.cuda.is_available() else 'CPU'
            self._model_counts = dict(dev=dev, eye=n_eye, bin=n_bin, ftern=n_ftern, type=n_type,
                                      dbil_e=n_dbil_e, dbil_f=n_dbil_f, ibil_e=n_ibil_e, ibil_f=n_ibil_f,
                                      cp=n_cp, meld=n_meld, sam=sam, total=total)
            self._update_model_info(); self._check_ready()
        else:
            self._models_failed = True
            self.status.setText('⚠ ' + ('Demo' if I18n.LANG == 'en' else '演示模式'))

    def _update_model_info(self):
        if not hasattr(self, 'model_info') or not hasattr(self, '_model_counts'): return
        c = self._model_counts
        if I18n.LANG == 'cn':
            self.model_info.setText(f'设备：{c["dev"]}\nSAM 面部分割：{"是" if c["sam"] else "否"}\n'
                                    f'眼睑三分类(TBIL)：{c["eye"]}  面部二分类：{c["bin"]}\n'
                                    f'面部三分类：{c["ftern"]}  黄疸类型：{c["type"]}\n'
                                    f'DBIL 眼睑{c["dbil_e"]}/面部{c["dbil_f"]}  IBIL 眼睑{c["ibil_e"]}/面部{c["ibil_f"]}\n'
                                    f'Child-Pugh：{c["cp"]}  MELD：{c["meld"]}\n'
                                    f'模型总计：{c["total"]}')
        else:
            self.model_info.setText(f'Device: {c["dev"]}\nSAM: {"Y" if c["sam"] else "N"}\n'
                                    f'Eyelid TBIL: {c["eye"]}  Face Binary: {c["bin"]}\n'
                                    f'Face TBIL: {c["ftern"]}  Type: {c["type"]}\n'
                                    f'DBIL E{c["dbil_e"]}/F{c["dbil_f"]}  IBIL E{c["ibil_e"]}/F{c["ibil_f"]}\n'
                                    f'Child-Pugh: {c["cp"]}  MELD: {c["meld"]}\n'
                                    f'Total: {c["total"]}')

    # ── analysis pipeline (unchanged logic) ────────────────────────
    def run_all(self):
        if not self.predictor[0] or not self.face_path: return
        self.progress.setVisible(True); self.progress.setRange(0, 0)
        self.progress_label.setVisible(True); self.progress_label.setText(I18n.get('analyzing'))
        self.analyze_btn.setEnabled(False)
        QApplication.processEvents()
        import threading
        self._analysis_result = None; self._analysis_done = False
        predictor = self.predictor[0]; fp = self.face_path; ep = self.eyelid_path

        def _worker():
            p = predictor
            is_video = fp and fp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
            results = {}

            def run_task(video_fn, image_fn, eyelid_fn=None):
                try:
                    if is_video and fp: return video_fn()
                    elif ep and eyelid_fn: return eyelid_fn()
                    elif fp and image_fn: return image_fn()
                    else: return {'error': 'No input'}
                except Exception as e: return {'error': str(e)}

            results['screen'] = run_task(lambda: p.analyze_video(fp, 'screen'), lambda: p.screen_face(fp))
            results['grade'] = run_task(lambda: p.analyze_video(fp, 'grade'),
                                        lambda: p.classify_grade_face(fp), lambda: p.analyze_eyelid(ep))
            results['dbil'] = run_task(lambda: p.analyze_video(fp, 'dbil'),
                                       lambda: p.classify_dbil_face(fp), lambda: p.classify_dbil_eyelid(ep))
            results['ibil'] = run_task(lambda: p.analyze_video(fp, 'ibil'),
                                       lambda: p.classify_ibil_face(fp), lambda: p.classify_ibil_eyelid(ep))
            results['type'] = run_task(lambda: p.analyze_video(fp, 'type'), lambda: p.classify_type(fp))
            results['cp'] = run_task(lambda: p.analyze_video(fp, 'cp'), lambda: p.classify_child_pugh(fp))
            results['meld'] = run_task(lambda: p.analyze_video(fp, 'meld'), lambda: p.classify_meld(fp))
            frame_path = fp
            if is_video:
                try:
                    frames = p.extract_video_frames(fp, max_frames=12)
                    if frames:
                        import tempfile
                        mid = len(frames) // 2
                        frame_path = os.path.join(tempfile.gettempdir(), 'bilinguard_frame.jpg')
                        cv2.imwrite(frame_path, cv2.cvtColor(frames[mid], cv2.COLOR_RGB2BGR))
                except Exception: pass
            self._analysis_result = results; self._analysis_frame_path = frame_path; self._analysis_done = True

        self._thread = threading.Thread(target=_worker, daemon=True); self._thread.start()
        self._poll_timer = QTimer(); self._poll_timer.timeout.connect(self._poll_analysis); self._poll_timer.start(200)

    def _poll_analysis(self):
        QApplication.processEvents()
        if self._analysis_done:
            self._poll_timer.stop()
            self.progress.setVisible(False); self.progress_label.setVisible(False)
            self.analyze_btn.setEnabled(True)
            self._last_frame_path = getattr(self, '_analysis_frame_path', self.face_path)
            self._analysis_done = False
            self._show_results(self._analysis_result)

    def _show_results(self, results, refresh_interp=True):
        # Screen
        sec = self.result_sections['screen']
        if 'screen' in results:
            r = results['screen']
            if 'error' in r: sec.set_na(r['error'])
            elif 'binary' in r:
                b = r['binary']; pj = b.get('probability_jaundice', 0)
                sec.set_result(b.get('prediction', '?'),
                               [(I18n.get('normal'), b.get('probability_normal', 0), COLORS['normal']),
                                (I18n.get('jaundiced'), pj, COLORS['danger'])])
                sec.pred_lbl.setStyleSheet(f"color:{COLORS['danger'] if pj > 0.5 else COLORS['normal']}; font-size:{FontScale.ts('h1')}px; font-weight:700;")
        else: sec.set_na()
        # Grade
        sec = self.result_sections['grade']
        if 'grade' in results:
            r = results['grade']
            if 'error' in r: sec.set_na(r['error'])
            elif 'grading' in r:
                gr = r['grading']; gi = gr.get('grade_index', 0)
                cols = [COLORS['mild'], COLORS['moderate'], COLORS['severe']]
                probs = [(k, v, cols[i] if i < 3 else COLORS['text']) for i, (k, v) in enumerate(gr.get('probabilities', {}).items())]
                sec.set_result(gr.get('prediction', '?'), probs, gr.get('recommendation', ''))
                sec.pred_lbl.setStyleSheet(f"color:{cols[gi] if gi < 3 else COLORS['text']}; font-size:{FontScale.ts('h1')}px; font-weight:700;")
        else: sec.set_na()
        # DBIL
        sec = self.result_sections['dbil']
        if 'dbil' in results:
            r = results['dbil']
            if 'error' in r: sec.set_na(r['error'])
            elif 'grading' in r:
                gr = r['grading']
                probs = [(k, v, [COLORS['normal'], COLORS['moderate'], COLORS['severe']][i] if i < 3 else COLORS['text'])
                         for i, (k, v) in enumerate(gr.get('probabilities', {}).items())]
                sec.set_result(gr.get('prediction', '?'), probs, gr.get('bilirubin_range', ''))
        else: sec.set_na()
        # IBIL
        sec = self.result_sections['ibil']
        if 'ibil' in results:
            r = results['ibil']
            if 'error' in r: sec.set_na(r['error'])
            elif 'grading' in r:
                gr = r['grading']
                probs = [(k, v, [COLORS['normal'], COLORS['moderate'], COLORS['severe']][i] if i < 3 else COLORS['text'])
                         for i, (k, v) in enumerate(gr.get('probabilities', {}).items())]
                sec.set_result(gr.get('prediction', '?'), probs, gr.get('bilirubin_range', ''))
        else: sec.set_na()
        # Type
        sec = self.result_sections['type']
        if 'type' in results:
            r = results['type']
            if 'error' in r: sec.set_na(r['error'])
            elif 'type_classification' in r:
                tc = r['type_classification']; ti = tc.get('type_index', 0)
                sec.set_result(tc.get('prediction', '?'),
                               [(I18n.get('hepatocellular'), tc.get('probability_hepatocellular', 0), COLORS['primary']),
                                (I18n.get('cholestatic'), tc.get('probability_cholestatic', 0), COLORS['accent'])],
                               I18n.get('hepato_desc') if ti == 0 else I18n.get('chol_desc'))
                sec.pred_lbl.setStyleSheet(f"color:{COLORS['primary'] if ti == 0 else COLORS['accent']}; font-size:{FontScale.ts('h1')}px; font-weight:700;")
        else: sec.set_na()
        # Child-Pugh
        sec = self.result_sections.get('cp')
        if sec:
            if 'cp' in results:
                r = results['cp']
                if 'error' in r: sec.set_na(r['error'])
                elif 'grading' in r:
                    gr = r['grading']
                    probs = [(k, v, [COLORS['normal'], COLORS['warning'], COLORS['danger']][i] if i < 3 else COLORS['text'])
                             for i, (k, v) in enumerate(gr.get('probabilities', {}).items())]
                    sec.set_result(gr.get('prediction', '?'), probs, gr.get('bilirubin_range', ''))
            else: sec.set_na()
        # MELD
        sec = self.result_sections.get('meld')
        if sec:
            if 'meld' in results:
                r = results['meld']
                if 'error' in r: sec.set_na(r['error'])
                elif 'grading' in r:
                    gr = r['grading']
                    probs = [(k, v, [COLORS['normal'], COLORS['warning'], COLORS['danger']][i] if i < 3 else COLORS['text'])
                             for i, (k, v) in enumerate(gr.get('probabilities', {}).items())]
                    sec.set_result(gr.get('prediction', '?'), probs, gr.get('bilirubin_range', ''))
            else: sec.set_na()
        for s in self.result_sections.values():
            for i in range(s.bars_layout.count()):
                pb = s.bars_layout.itemAt(i).widget()
                if pb and hasattr(pb, 'apply'): pb.apply()
        total_time = sum(r.get('inference_time', 0) for r in results.values() if isinstance(r, dict))
        self.time_label.setText(f'{I18n.get("time")}: {total_time:.2f}s')
        self._render_advice(results)
        if not refresh_interp:
            return
        # interpretability — computed in a background thread so the UI never freezes
        _pred = self.predictor[0]
        _face = getattr(self, '_last_frame_path', None)
        _eye = self.eyelid_path
        if (not _face or not os.path.exists(_face)):
            _face = self.face_path if (self.face_path and not self.face_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))) else None
        if ((_face and os.path.exists(_face)) or (_eye and os.path.exists(_eye))) and _pred:
            self._run_interpretability(_pred, _face, _eye)
        else:
            self.face_interp_title.setVisible(False); self.face_images_widget.setVisible(False)
            self.eye_interp_title.setVisible(False); self.eye_images_widget.setVisible(False)

    def _compute_gradcam(self, predictor, image_path, source_type='face'):
        import matplotlib.cm as cm
        fb = np.fromfile(image_path, dtype=np.uint8); img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        if img is None: return {'error': 'Cannot read'}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        use_sam = source_type == 'face' and getattr(predictor, 'sam_model', None) is not None
        if use_sam:
            processed = predictor._preprocess_face_sam(img_rgb)
        else:
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB); l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
            processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            processed = cv2.resize(processed, (224, 224))
        disp = processed.copy()
        cam_model = None
        if source_type == 'face':
            for name, m in getattr(predictor, 'face_binary_models', []):
                if 'convnext' in name.lower(): cam_model = m; break
        if cam_model is None:
            for i in range(min(len(predictor.model_names), len(predictor.models))):
                if 'convnext' in predictor.model_names[i].lower(): cam_model = predictor.models[i]; break
        overlay = heatmap = None
        if cam_model is not None and HAS_TORCH:
            try:
                from inference import GradCAMGenerator
                gen = GradCAMGenerator(cam_model)
                norm = processed.astype(np.float32) / 255.0
                norm = (norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                t = torch.FloatTensor(np.stack([norm.transpose(2, 0, 1)])).to(predictor.device)
                cam = gen.generate(t)
                if cam is not None:
                    hm = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
                    overlay = (disp * 0.5 + hm * 0.5).astype(np.uint8); heatmap = hm
            except Exception as e:
                print('[Grad-CAM]', e, flush=True)
        return {'original': disp, 'overlay': overlay, 'heatmap': heatmap}

    def _display_gradcam(self, result, lo, lov, lhm):
        def _set(lbl, arr):
            if arr is not None:
                h, w = arr.shape[:2]
                qi = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
                lbl.setPixmap(QPixmap.fromImage(qi).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        _set(lo, result.get('original')); _set(lov, result.get('overlay')); _set(lhm, result.get('heatmap'))

    def _run_interpretability(self, predictor, face_path, eye_path):
        """Show placeholders, then compute Grad-CAM + colour + agreement off the UI thread."""
        # placeholders while computing
        if face_path and os.path.exists(face_path):
            self.face_interp_title.setVisible(True); self.face_images_widget.setVisible(True)
            for lbl in (self.img_face_original, self.img_face_overlay, self.img_face_heatmap):
                lbl.setText('…'); lbl.setPixmap(QPixmap())
        else:
            self.face_interp_title.setVisible(False); self.face_images_widget.setVisible(False)
        if eye_path and os.path.exists(eye_path):
            self.eye_interp_title.setVisible(True); self.eye_images_widget.setVisible(True)
            for lbl in (self.img_eye_original, self.img_eye_overlay, self.img_eye_heatmap):
                lbl.setText('…'); lbl.setPixmap(QPixmap())
        else:
            self.eye_interp_title.setVisible(False); self.eye_images_widget.setVisible(False)
        QApplication.processEvents()
        self._interp_worker = GradcamWorker(predictor, face_path, eye_path, self)
        self._interp_worker.done.connect(self._on_interp_done)
        self._interp_worker.start()

    def _on_interp_done(self, data):
        """Receive background-computed interpretability and render it (main thread)."""
        face = data.get('face'); eye = data.get('eye')
        if face is not None:
            self.face_interp_title.setVisible(True); self.face_images_widget.setVisible(True)
            self._display_gradcam(face, self.img_face_original, self.img_face_overlay, self.img_face_heatmap)
        else:
            self.face_interp_title.setVisible(False); self.face_images_widget.setVisible(False)
        if eye is not None:
            self.eye_interp_title.setVisible(True); self.eye_images_widget.setVisible(True)
            self._display_gradcam(eye, self.img_eye_original, self.img_eye_overlay, self.img_eye_heatmap)
        else:
            self.eye_interp_title.setVisible(False); self.eye_images_widget.setVisible(False)
        # colour bars
        cd = data.get('color') or {}
        for i in reversed(range(self.color_bars_layout.count())):
            w = self.color_bars_layout.itemAt(i).widget()
            if w: w.setParent(None)
        for label, val, lo, hi, color in [('Lab b*', cd.get('lab_b', 0), 128, 145, COLORS['mild']),
                                          ('Yellow ratio', cd.get('yellow_ratio', 0), 0, 0.3, COLORS['danger']),
                                          ('Saturation', cd.get('hsv_s', 0), 0, 150, COLORS['accent']),
                                          ('Hue', cd.get('hsv_h', 0), 0, 60, COLORS['primary'])]:
            pct = max(0, min(1, (val - lo) / (hi - lo))) if hi > lo else 0
            bar = ProbBar(label, pct, color); bar.apply()
            bar.bar.setFormat(f'{val:.1f}' if val >= 1 else f'{val*100:.1f}%')
            self.color_bars_layout.addWidget(bar)
        # agreement
        self._render_agreement(data.get('individual'), data.get('confidence'))

    def _render_agreement(self, individual, confidence):
        """Render the Model-Agreement panel from precomputed per-model votes."""
        for i in reversed(range(self.agreement_layout.count())):
            w = self.agreement_layout.itemAt(i).widget()
            if w: w.setParent(None)
        if not individual:
            self.confidence_label.setText('—'); return
        for name, pred, prob in individual:
            short = (name.replace('_tiny_patch16_224', '').replace('_tiny_patch4_window7_224', '')
                     .replace('_b0', '').replace('.pt', ''))
            col = COLORS['danger'] if pred in (I18n.get('jaundiced'), 'Jaundiced') else COLORS['normal']
            row = QLabel(f'{short}  ·  {pred}  ({prob:.0%})')
            row.setStyleSheet(f"color:{col}; font-size:{FontScale.ts('small')}px;")
            self.agreement_layout.addWidget(row)
        if confidence:
            lab = 'Entropy' if I18n.LANG == 'en' else '熵'
            mlab = 'Margin' if I18n.LANG == 'en' else '裕度'
            alab = 'Agreement' if I18n.LANG == 'en' else '一致性'
            self.confidence_label.setText(
                f'{lab} {confidence.get("entropy",0):.2f}  ·  {mlab} {confidence.get("margin",0):.0%}  ·  {alab} {confidence.get("agreement",0):.0%}')
            self.confidence_label.setStyleSheet(f"color:{COLORS['text_secondary']}; font-size:{FontScale.ts('small')}px;")
            self.confidence_label.setText('—')

    # ── clinical guidance rendering ────────────────────────────────
    def _render_advice(self, results):
        self._last_results = results
        self._clear_layout(self.advice_layout)
        self.advice_empty.setParent(None)
        if not HAS_ADVISOR:
            self.advice_empty.setText(I18n.get('advidence_empty')); self.advice_layout.addWidget(self.advice_empty); return
        try:
            _labs = self._read_labs()
            _nrs = _labs.pop('nrs2002', None)
            _nutrition = {'nrs2002': _nrs} if _nrs is not None else None
            advice = ClinicalAdvisor(lang=I18n.LANG, audit_log=str(AUDIT_LOG)).advise(
                results, _labs, department=self._read_dept(),
                diagnosis=(self.dx_edit.text().strip() or None),
                nutrition=_nutrition)
        except Exception:
            import traceback; traceback.print_exc(); return
        self._last_advice = advice
        # hero
        sev = advice.get('severity', 0)
        sev_cn = ['正常/临界', '轻度黄疸', '中度黄疸', '重度黄疸']; sev_en = ['Normal', 'Mild', 'Moderate', 'Severe']
        lbl = (sev_en if I18n.LANG == 'en' else sev_cn)[min(sev, 3)]
        tbil = self._read_labs().get('tbil')
        self.gauge.set_data(sev, lbl, value_text=(f'{tbil:g} μmol/L' if tbil is not None else ''))
        jt = advice.get('jaundice_type')
        jtn = {'hepatocellular': ('肝细胞性', 'Hepatocellular'), 'cholestatic': ('胆汁郁积性', 'Cholestatic'),
               'hemolytic': ('溶血性', 'Haemolytic')}.get(jt)
        self.type_badge.setText(jtn[0 if I18n.LANG == 'cn' else 1] if jtn else I18n.get('type_badge_none'))
        bc = COLORS.get(jt, COLORS['primary']) if jt else COLORS['primary']
        self.type_badge.setStyleSheet(
            f"QLabel#badge {{ background:{bc}22; color:{bc}; border-radius:8px; padding:5px 12px; "
            f"font-size:{FontScale.ts('body')}px; font-weight:700; max-width:150px; }}")
        # triage summary: drop the leading severity token so the severity word is shown only on the gauge
        summary = advice.get('summary', '')
        sep = '｜' if '｜' in summary else (' | ' if ' | ' in summary else None)
        if sep and summary.count(sep) >= 1:
            summary = sep.join(summary.split(sep)[1:]).lstrip()
        sev_color = [COLORS['normal'], COLORS['mild'], COLORS['moderate'], COLORS['severe']][min(sev, 3)]
        self.triage_summary.setText(summary or lbl)
        self.triage_summary.setStyleSheet(
            f"color:{sev_color}; font-size:{FontScale.ts('h3')}px; font-weight:600;")
        # CDSS v3 agent self-check meta line
        ag = advice.get('agent') or {}
        sc = ag.get('self_check') or {}
        if sc:
            cn = I18n.LANG == 'cn'
            mode_txt = {'red_flag': ('红色警示', 'Red flag'), 'routine': ('常规', 'Routine'),
                        'normal': ('正常', 'Normal'), 'conflict': ('数据冲突', 'Data conflict'),
                        'insufficient_data': ('数据不足', 'Insufficient data')}.get(
                            ag.get('mode'), (str(ag.get('mode')),) * 2)
            pr = sc.get('alert_priority') or '-'
            pr_color = {'HIGH': COLORS['severe'], 'MEDIUM': COLORS['moderate'],
                        'LOW': COLORS['mild'], 'NONE': COLORS['normal']}.get(pr, COLORS['text_muted'])
            conf = {'high': ('高', 'High'), 'moderate': ('中', 'Moderate'),
                    'low': ('低', 'Low')}.get(sc.get('confidence'),
                                               (str(sc.get('confidence')),) * 2)
            nut = (ag.get('case', {}) or {}).get('nutrition_state')
            nut_txt = {'at_risk': ('营养风险', 'Nutrition risk'), 'no_risk': ('无营养风险', 'No nutrition risk'),
                       'unscreened': ('营养未筛查', 'Nutrition unscreened')}.get(nut)
            L = 0 if cn else 1
            html = (f'{("决策模式", "Mode")[L]} <b>{mode_txt[L]}</b> · '
                    f'{("优先级", "Priority")[L]} <b style="color:{pr_color}">{pr}</b> · '
                    f'{("置信度", "Confidence")[L]} <b>{conf[L]}</b> · '
                    f'{("规则", "Rules")[L]} <b>{sc.get("alerts_total", 0)}</b>')
            if nut_txt:
                html += f' · <b>{nut_txt[L]}</b>'
            self.agent_meta.setText(html)
        else:
            self.agent_meta.setText('')
        # section cards
        for sec in advice.get('sections', []):
            self.advice_layout.addWidget(self._advice_card(sec))
        disc = advice.get('disclaimer', '')
        if disc:
            d = QLabel(disc); d.setWordWrap(True)
            d.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:{FontScale.ts('small')}px; padding:4px 2px;")
            self.advice_layout.addWidget(d)

    def _advice_card(self, sec):
        ic = sec.get('icon', ''); title = sec.get('title', '')
        if '🚨' in ic: accent, soft = COLORS['danger'], COLORS['danger_soft']
        elif '⚠' in ic: accent, soft = COLORS['moderate'], COLORS['moderate_soft']
        else: accent, soft = COLORS['primary'], COLORS['soft']
        card = QFrame(); card.setStyleSheet(
            f"QFrame {{ background:{COLORS['surface']}; border:1px solid {COLORS['border']}; border-radius:8px; border-left:4px solid {accent}; }}")
        cl = QVBoxLayout(card); cl.setContentsMargins(16, 12, 16, 13); cl.setSpacing(6)
        h = QLabel(f'<b>{title}</b>'); h.setTextFormat(Qt.RichText)
        h.setStyleSheet(f"color:{accent}; font-size:{FontScale.ts('h3')}px; font-weight:700;")
        cl.addWidget(h)
        body = QLabel(self._format_advice_content(sec.get('content', '')))
        body.setTextFormat(Qt.RichText); body.setWordWrap(True)
        body.setStyleSheet(f"color:{COLORS['text']}; font-size:{FontScale.ts('body')}px; line-height:140%;")
        cl.addWidget(body)
        refs = sec.get('guidelines') or []
        if refs:
            eh = QLabel(f'<b>{I18n.get("advice_evidence")}</b>'); eh.setTextFormat(Qt.RichText)
            eh.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:{FontScale.ts('small')}px; padding-top:3px;")
            cl.addWidget(eh)
            for g in refs:
                line = (f'<b style="color:{COLORS["primary_active"]};">[{g.get("gid","")}]</b> '
                        f'<span style="color:{COLORS["text_secondary"]};">{g.get("citation","")}</span> '
                        f'<span style="color:{COLORS["text_muted"]};">({g.get("source","")} {g.get("year","")})</span>')
                rl = QLabel(line); rl.setTextFormat(Qt.RichText); rl.setWordWrap(True)
                rl.setStyleSheet(f"font-size:{FontScale.ts('small')}px; color:{COLORS['text_secondary']}; padding-left:14px;")
                cl.addWidget(rl)
        return card


def main():
    print('=' * 50, flush=True); print('  BilinGuard Desktop v4.0', flush=True); print('=' * 50, flush=True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    FontScale.app = app
    app.setFont(QFont(_FONT, FontScale.current))
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
