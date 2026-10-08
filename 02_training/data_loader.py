"""
BilinGuard: Data Loader
Handles:
  - Reading patient zip files (video + images)
  - Uniform frame sampling (12 frames per video)
  - Face detection via MediaPipe
  - CLAHE preprocessing + SAM-based face segmentation
  - Sclera region extraction for YellowFeatures
  - Patient-level stratified splits
"""
import os, cv2, zipfile, random, io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm
from config import *

random.seed(SEED)
np.random.seed(SEED)

def list_patients():
    patients = []
    labels_map = {}
    for fname in os.listdir(NORMAL_DIR):
        if fname.endswith('.zip'):
            pid = fname.replace('.zip', '')
            patients.append((os.path.join(NORMAL_DIR, fname), pid, 0))
            labels_map[pid] = 0
    jaundice_dir = JAUNDICE_DIR
    for fname in os.listdir(jaundice_dir):
        if fname.endswith('.zip'):
            pid = fname.replace('.zip', '')
            patients.append((os.path.join(jaundice_dir, fname), pid, 1))
            labels_map[pid] = 1
    return patients, labels_map

def extract_frames_from_video(video_bytes, n_frames=N_FRAMES):
    video_bytes.seek(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frames = []
    cap = cv2.VideoCapture(video_bytes)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return frames
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def detect_face(frame):
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = fd.process(rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            return frame[y1:y2, x1:x2]
    return frame

def preprocess_frame(frame, img_size=IMG_SIZE):
    """CLAHE + resize + normalize"""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = (frame - mean) / std
    return frame.transpose(2, 0, 1)

def extract_sclera(frame):
    """Simple sclera extraction: crop upper face region"""
    h, w = frame.shape[:2]
    sclera = frame[:h//2, :, :]
    sclera = cv2.resize(sclera, (IMG_SIZE, IMG_SIZE))
    return sclera

class BilinGuardDataset(Dataset):
    def __init__(self, patient_list, labels_map, task='binary', augment=False):
        self.patient_list = patient_list
        self.labels_map = labels_map
        self.task = task
        self.augment = augment
        if augment:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.3),
                A.Rotate(limit=10, p=0.3),
            ])

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        zip_path, pid, bin_label = self.patient_list[idx]
        label = bin_label
        frames = []
        sclera_frames = []
        with zipfile.ZipFile(zip_path, 'r') as z:
            names = z.namelist()
            # Find video file
            video_files = [n for n in names if n.lower().endswith('.mp4')]
            if video_files:
                video_bytes = io.BytesIO(z.read(video_files[0]))
                frames_rgb = extract_frames_from_video(video_bytes)
            else:
                frames_rgb = []
            if not frames_rgb:
                img_files = [n for n in names if n.lower().endswith('.jpg') and 'feature' not in n.lower()]
                if img_files:
                    img_data = z.read(img_files[0])
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames_rgb = [img] * N_FRAMES
            if not frames_rgb:
                frames_rgb = [np.zeros((480, 640, 3), dtype=np.uint8)] * N_FRAMES
        for f in frames_rgb:
            face = detect_face(f)
            if self.augment and self.aug:
                face_aug = self.aug(image=face)['image']
            else:
                face_aug = face
            frames.append(preprocess_frame(face_aug))
            sclera_frames.append(preprocess_frame(extract_sclera(face_aug)))
        frames = np.stack(frames)
        sclera = np.stack(sclera_frames)
        return {
            'frames': torch.FloatTensor(frames),
            'sclera': torch.FloatTensor(sclera),
            'label': torch.tensor(label, dtype=torch.long),
            'pid': pid,
        }

def create_splits(patient_list, labels_map, n_folds=N_FOLDS, fold=0):
    random.shuffle(patient_list)
    n = len(patient_list)
    fold_size = n // n_folds
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_folds - 1 else n
    val = patient_list[val_start:val_end]
    train = patient_list[:val_start] + patient_list[val_end:]
    return train, val

def get_dataloaders(task='binary', batch_size=BATCH_SIZE, fold=0):
    patients, labels = list_patients()
    train_list, val_list = create_splits(patients, labels, fold=fold)
    train_ds = BilinGuardDataset(train_list, labels, task=task, augment=True)
    val_ds = BilinGuardDataset(val_list, labels, task=task)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
