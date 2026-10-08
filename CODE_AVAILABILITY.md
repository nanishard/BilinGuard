# Code Availability Statement (for Cell Reports Medicine submission)

## Original code

The complete source code for BilinGuard (preprocessing, model training,
evaluation, clinical decision support system, reader study analysis, and
desktop application) is provided in the accompanying 09_Code/ directory.

**Code location for reviewers**: Included as supplemental files in this
submission package. Will be deposited in a general-purpose repository
(e.g., Zenodo or GitHub) upon acceptance and assigned a DOI.

**Access**: Free, anonymous access for reviewers. No tokens or registration
required — all code files are provided directly.

**License**: Will be determined upon acceptance (recommended: MIT or Apache 2.0).

## Software dependencies

- Python 3.11
- PyTorch >= 2.0 (deep learning framework)
- OpenCV >= 4.8 (YuNet face detection, image processing)
- Segment Anything Model (SAM) for face segmentation
- scikit-learn >= 1.3 (evaluation metrics)
- PyQt5 (desktop application)
- Full list in `requirements.txt`

## Hardware requirements

- Training: 1× GPU (≥16 GB VRAM, e.g., NVIDIA RTX 4060 Ti)
- Inference: CPU sufficient (~0.3 s per patient for 12 frames)
- Storage: ~50 GB for full dataset
