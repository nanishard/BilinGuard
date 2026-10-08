[README.md](https://github.com/user-attachments/files/33184224/README.md)

# BilinGuard — Source Code for Cell Reports Medicine Submission

## Overview

This directory contains the source code for BilinGuard, a deep-learning system
for adult jaundice screening and severity classification from facial video.

## Directory structure

```
09_Code/
├── requirements.txt              Python dependencies
├── 01_preprocessing/            Video frame extraction, face detection (YuNet),
│                                SAM segmentation, CLAHE, quality control
├── 02_training/                 Model architectures, training scripts for all
│                                7 tasks × 3 input configurations × 4 backbones
├── 03_evaluation/               Bootstrap CI, external validation, fusion
│                                paired comparison, restricted analyses
├── 04_cdss/                     Rule-based clinical decision support system:
│                                audit, validation (150 synthetic scenarios),
│                                threshold sensitivity, blind review
├── 05_reader_study/             Eight-reader blinded study analysis
├── 06_figures_tables/           Manuscript figure and table generation
├── 07_supplementary/            Supplemental figure and table generation
├── 08_deployment/               Desktop application (PyQt5), CDSS engine,
│                                model inference
├── 09_audit/                    Data audits, shortcut-learning detection,
│                                quality diagnostics
└── utils/                       Shared utilities, prediction, clinical models
```

## Key files

### Preprocessing
- `face_preprocess_v4.py` — Canonical-v2 facial preprocessing pipeline
- `sam_preprocess.py` — SAM-based face segmentation
- `extract_video_frames.py` — Uniform temporal sampling (12 frames)

### Training
- `models.py` — Model architectures (ConvNeXt-Tiny, ViT-Tiny/16, Swin-Tiny, EfficientNet-B0)
- `train_binary.py` — Binary jaundice screening training
- `train_ternary.py` — Three-class TBIL grading
- `train_type_classification.py` — Jaundice-type classification
- `train_fusion_shared_split.py` — Dual-view fusion with shared held-out split

### Evaluation
- `bootstrap_ci_all.py` — 1000-replicate patient-level bootstrap for all metrics
- `fusion_paired_comparison.py` — Paired ΔAUC with CIs
- `external_validation_final.py` — Frozen external validation (v4-lite preprocessing)

### CDSS
- `clinical_advisor.py` — CDSS engine (64 guidelines, 51 rules, 6 red-flag pathways)
- `run_cdss_all_validation.py` — 150 prespecified synthetic safety scenarios

### Deployment
- `desktop_app.py` — PyQt5 desktop application with seven-task dashboard
- `inference.py` — Model inference engine

## Software requirements

See `requirements.txt`. Key dependencies:
- Python 3.11
- PyTorch >= 2.0
- OpenCV (YuNet face detection)
- Segment Anything Model (SAM)
- scikit-learn, pandas, numpy

## Data availability

The deidentified facial-video data, paired laboratory reference values, and
trained model weights are available for non-commercial research on reasonable
request to the lead contact (zhanghaohuaxi@163.com), subject to institutional
data-governance approval and a data-use agreement.

## License

This code is provided for peer review and reproducibility verification.
