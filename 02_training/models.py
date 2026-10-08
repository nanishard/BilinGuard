"""
BilinGuard: Model Definitions
Architectures:
  1. YOLO11-based binary classifier
  2. Vision backbones (ConvNeXt, Swin, EfficientNet, ViT)
  3. YellowFeatures (sclera-focused classifier)
  4. DynamicFusion ensemble (BilinGuard)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import *

# ---------------------------------------------------------------------------
# YOLO-based binary classifier
# ---------------------------------------------------------------------------
class YOLOClassifier(nn.Module):
    def __init__(self, num_classes=2, backbone='yolo11n'):
        super().__init__()
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(f'{backbone}.pt')
            self.yolo.model.classifier = nn.Identity()
            in_features = self.yolo.model.model[-1].in_channels if hasattr(self.yolo.model.model[-1], 'in_channels') else 512
        except ImportError:
            self.yolo = None
            in_features = 512
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, F, C, H, W) where F = n_frames -> (B*F, C, H, W)
        B, F, C, H, W = x.shape
        if self.yolo is not None:
            feats = []
            for i in range(B):
                yolo_pred = self.yolo(x[i], verbose=False)
                feat = yolo_pred[0].boxes.data if yolo_pred[0].boxes is not None else torch.zeros((F, 5), device=x.device)
                feats.append(feat.mean(dim=0))
            feat = torch.stack(feats)
        else:
            feat = self.global_pool(x.view(B * F, C, H, W)).view(B, F, -1).mean(dim=1)
        return self.classifier(feat)

# ---------------------------------------------------------------------------
# Vision backbone extractor
# ---------------------------------------------------------------------------
BACKBONE_BUILDERS = {
    'convnext_tiny': lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    'swin_tiny': lambda: models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1),
    'efficientnet_b0': lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    'vit_tiny_patch16_224': lambda: models.vit_tiny_patch16_224(weights=models.ViT_Tiny_Patch16_224_Weights.IMAGENET1K_V1),
}

class VisionBackbone(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', num_classes=3, dropout=0.3):
        super().__init__()
        builder = BACKBONE_BUILDERS.get(backbone_name)
        if builder is None:
            raise ValueError(f'Unknown backbone: {backbone_name}')
        base = builder()
        if hasattr(base, 'classifier'):
            in_features = base.classifier[-1].in_features
            base.classifier = nn.Identity()
        elif hasattr(base, 'head'):
            in_features = base.head.in_features
            base.head = nn.Identity()
        elif hasattr(base, 'fc'):
            in_features = base.fc.in_features
            base.fc = nn.Identity()
        else:
            in_features = 512
        self.backbone = base
        self.pool = nn.AdaptiveAvgPool2d(1) if backbone_name not in ['vit_tiny_patch16_224'] else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        feats = self.backbone(x)
        if isinstance(self.pool, nn.AdaptiveAvgPool2d):
            feats = self.pool(feats).flatten(1)
        feats = feats.view(B, F, -1).mean(dim=1)
        return self.classifier(feats)

# ---------------------------------------------------------------------------
# YellowFeatures: Sclera-focused module
# ---------------------------------------------------------------------------
class YellowFeatures(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Sclera color analysis branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).flatten(1)
        x = x.view(B, F, -1).mean(dim=1)
        return self.classifier(x)

# ---------------------------------------------------------------------------
# Weighted Voting Ensemble (DynamicFusion)
# ---------------------------------------------------------------------------
class DynamicFusion(nn.Module):
    def __init__(self, backbone_models, num_classes=3):
        super().__init__()
        self.models = nn.ModuleList(backbone_models)
        self.weights = nn.Parameter(torch.ones(len(backbone_models)) / len(backbone_models))

    def forward(self, x, extra_input=None):
        outputs = []
        for m in self.models:
            out = m(x)
            outputs.append(out)
        stacked = torch.stack(outputs)
        w = F.softmax(self.weights, dim=0).unsqueeze(-1).unsqueeze(-1)
        weighted = (stacked * w).sum(dim=0)
        return weighted

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_model(model_name, num_classes=3):
    name = model_name.lower()
    if name == 'yolo':
        return YOLOClassifier(num_classes=2)
    elif name == 'yellowfeatures':
        return YellowFeatures(num_classes=num_classes)
    elif name in BACKBONE_BUILDERS:
        return VisionBackbone(name, num_classes=num_classes)
    elif name == 'dynamicfusion':
        backbones = [VisionBackbone(bn, num_classes=num_classes) for bn in BACKBONE_NAMES]
        return DynamicFusion(backbones, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model: {model_name}')
