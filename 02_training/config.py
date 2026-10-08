"""
BilinGuard: Configuration
"""
import os
import torch

BASE_DIR = r'D:\research\人脸识别营养\传染科'
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data
NORMAL_DIR = os.path.join(BASE_DIR, '正常')
JAUNDICE_DIR = os.path.join(BASE_DIR, '人脸肝炎', '人脸肝炎')
N_FRAMES = 12
VIDEO_LENGTH_SEC = 15
IMG_SIZE = 224

# Labels
BILIRUBIN_THRESHOLDS = {
    'normal': (0, 2.0),
    'mild': (2.0, 5.0),
    'moderate': (5.0, 10.0),
    'severe': (10.0, 100.0),
}

# Training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
N_BOOTSTRAP = 1000
N_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5

# YOLO
YOLO_VERSION = 'yolo11n'

# Vision backbones
BACKBONE_NAMES = ['convnext_tiny', 'swin_tiny', 'efficientnet_b0', 'vit_tiny_patch16_224']
ENSEMBLE_NAME = 'DynamicFusion'
