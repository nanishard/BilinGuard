# -*- coding: utf-8 -*-
"""
Generate comprehensive Supplementary Materials DOCX.

Follows Nature-style format with the NEW 3-layer methodology.
Includes extensive methods, multiple new modules, tables, and discussion.
"""
import os, sys, json, pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
TBL_V3 = os.path.join(RES, 'tables_v3')
sys.path.insert(0, os.path.join(BASE, 'deployment'))
OUT = os.path.join(BASE, 'Supplementary_Materials.docx')

doc = Document()
for section in doc.sections:
    section.top_margin = Inches(0.8); section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9); section.right_margin = Inches(0.9)
style = doc.styles['Normal']
style.font.name = 'Times New Roman'; style.font.size = Pt(10)

def H(text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs: r.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    return h

def P(text, bold=False, italic=False, size=10):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size); r.bold = bold; r.italic = italic
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p

def add_table(df, font_size=8):
    cols = list(df.columns)
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = 'Light Grid Accent 1'
    for i, c in enumerate(cols):
        cell = t.rows[0].cells[i]; cell.text = str(c)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs: r.bold = True; r.font.size = Pt(font_size)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs: r.font.size = Pt(font_size)

# ═════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════
t = doc.add_heading('', level=0)
r = t.add_run('Supplementary Information')
r.font.size = Pt(20); r.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
P('BilinGuard: A Closed-Loop Artificial Intelligence System from Non-Invasive '
  'Facial Jaundice Recognition to Prognostic Scoring and Clinical Decision Support',
  bold=True, size=11)
P('Nan Lin*, Mingxi Yang*, Yan Chen*, et al.', italic=True, size=9)
P('West China Hospital, Sichuan University, Chengdu, China.', italic=True, size=9)
P('* These authors contributed equally.', size=8)
doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# SUPPLEMENTARY METHODS
# ═════════════════════════════════════════════════════════════
H('Supplementary Methods', level=1)

# ── Section 1 ──
H('1. Study Design and Participants', level=2)
P('This prospective study was conducted at West China Hospital (WCH), a 4,300-bed tertiary referral centre in '
  'Chengdu, China, between November 18, 2022, and August 2, 2024. The study was approved by the WCH Ethics '
  'Committee (HX2023506 and HX20221153) and conducted in accordance with the Declaration of Helsinki. Written '
  'informed consent was obtained from all participants.')
P('Adult inpatients (age >=18 years) from the departments of hepatobiliary surgery, infectious diseases, and '
  'gastroenterology were eligible. Exclusion criteria were: (i) inability to provide informed consent; '
  '(ii) severe facial trauma or dermatological condition precluding image analysis; (iii) serum bilirubin '
  'measurement unavailable within two hours of video acquisition; and (iv) video of insufficient quality '
  '(severe motion blur, complete darkness, or no detectable face).')
P('Of 1,206 inpatients screened, 942 had facial videos recorded. After automated quality control and exclusion '
  'of three videos (one with severe motion blur, two with unmatched laboratory tests), 939 participants formed '
  'the internal analysis cohort: 625 non-jaundiced controls and 314 jaundiced patients. An independent external '
  'validation cohort of 117 patients (59 non-jaundiced, 58 jaundiced) was assembled from collaborating centres '
  'using the same acquisition protocol.')

# ── Section 2 ──
H('2. Data Acquisition Protocol', level=2)
H('2.1 Video Collection', level=3)
P('For each participant, trained clinical staff acquired a 10-20-second frontal facial video at the bedside '
  'using consumer-grade digital cameras (iPhone 12-15, Huawei Mate series, or equivalent) under routine ward '
  'illumination. No professional lighting, calibration hardware, or fixed camera mount was used. Videos were '
  'collected in a semi-standardised manner: the patient was seated upright or at 30-45 degrees elevation, '
  'facing the camera at arm\'s length (approximately 40-60 cm), with the full face visible from forehead to '
  'chin. Patients were instructed to maintain a neutral expression and keep their eyes open.')
H('2.2 Biochemical Reference Standard', level=3)
P('Serum total bilirubin (TBIL), direct bilirubin (DBIL), indirect bilirubin (IBIL), alanine aminotransferase '
  '(ALT), aspartate aminotransferase (AST), alkaline phosphatase (ALP), gamma-glutamyl transferase (GGT), '
  'international normalised ratio (INR), and albumin were measured from venous blood samples obtained within '
  'two hours of video acquisition, using standard laboratory assays on a Roche Cobas 8000 analyser. The '
  'phlebotomy-to-video interval was recorded for all participants.')
H('2.3 Structured Questionnaire', level=3)
P('Each participant completed a structured questionnaire capturing: demographic data (age, sex, height, weight, '
  'BMI, waist and hip circumference); comorbidities (malignancy, inflammatory bowel disease, infectious liver '
  'disease, chronic hepatitis B/C, cirrhosis, autoimmune disease); medication history (hepatotoxic drugs, '
  'anticoagulants); jaundice-related symptoms (duration, colour of urine/stool, pruritus, fatigue); and '
  'Fitzpatrick skin type (I-VI) assessed by the recording nurse.')

# ── Section 3 ──
H('3. Video Preprocessing Pipeline', level=2)
P('The preprocessing pipeline transforms raw bedside videos into standardised 224x224 face crops suitable for '
  'deep learning inference, while preserving the chromatic and morphological information relevant to '
  'hyperbilirubinaemia and hepatic dysfunction.')
H('3.1 Uniform Frame Sampling', level=3)
P('Twelve frames were uniformly sampled from each video at equal temporal intervals, capturing the full '
  'duration of the clip. This temporal sampling strategy captures frame-to-frame variation in facial '
  'expression, micro-movement, and illumination, providing redundancy against single-frame artefacts.')
H('3.2 Face Detection and Bounding Box', level=3)
P('For each frame, a Haar-cascade frontal face detector (OpenCV haarcascade_frontalface_default.xml) '
  'identified the largest face bounding box. When multiple faces were detected, the largest by area was '
  'selected. Frames without a detectable face were excluded from downstream processing.')
H('3.3 Segment Anything Model Face Segmentation', level=3)
P('The Segment Anything Model (SAM; ViT-H backbone, pre-trained on the SA-1B dataset of 1 billion masks) was '
  'used to generate precise facial region masks. The Haar-cascade bounding box was provided as a prompt to '
  'SAM, which produced a pixel-level segmentation mask delineating the face from hair, background, and '
  'clothing. The mask was then tightened by 5-pixel padding around the extreme rows and columns of the '
  'largest connected component, yielding a face crop that retained original pixel values (without fill or '
  'interpolation).')
H('3.4 Contrast-Limited Adaptive Histogram Equalisation', level=3)
P('CLAHE was applied to the L channel of the LAB colour space (clip limit = 3.0, tile grid size = 8x8) to '
  'reduce illumination heterogeneity while preserving chromatic information in the a* and b* channels. This '
  'step is critical because bedside videos are captured under highly variable lighting conditions (fluorescent '
  'ward lights, window light, incandescent lamps) that would otherwise dominate the colour signal.')
H('3.5 Grey-World Colour Constancy (Domain Adaptation)', level=3)
P('For domain-adapted models, Grey-World normalisation was applied before CLAHE. This assumes the average '
  'colour of a sufficiently diverse image should be grey, and rescales each channel so that mean R = mean G = '
  'mean B. This removes global illumination bias (e.g., the warm tint of incandescent lighting or the cool '
  'tint of fluorescent lighting), making the colour statistics of images captured under different conditions '
  'more comparable. The combination of Grey-World + CLAHE was applied to all images used in domain-adaptation '
  'training (internal + external).')
H('3.6 Sclera Region Extraction', level=3)
P('For eyelid-based models, an additional sclera extraction step was performed. The eye region (approximately '
  '20-45% of face height, 10-90% of face width) was isolated. Within this region, sclera pixels were '
  'identified using HSV thresholding (S < 60 and V > 70 for white sclera; H 20-60, S 10-60, V 70-100 for '
  'yellowed sclera). Morphological opening (3x3 elliptical kernel) removed noise. The resulting sclera mask '
  'was applied to extract the conjunctival region, which was resized to 224x224 for eyelid-specific models.')

# ── Section 4 ──
H('4. Layer 1: Jaundice Recognition Architecture', level=2)
P('Layer 1 comprises five recognition tasks, each served by an ensemble of independently trained backbone '
  'networks. All backbones were initialised from ImageNet pre-training and fine-tuned on BilinGuard data.')
H('4.1 Backbone Networks', level=3)
P('Four backbone architectures were employed, selected for their complementary inductive biases:')
add_table(pd.DataFrame({
    'Backbone': ['ConvNeXt-Tiny', 'ViT-Tiny/16', 'Swin-Tiny', 'EfficientNet-B0'],
    'Type': ['Convolutional', 'Vision Transformer', 'Hierarchical Transformer', 'Compound CNN'],
    'Parameters': ['28M', '6M', '28M', '5M'],
    'Receptive Field': ['Multi-scale local', 'Global self-attention', 'Shifted-window local-global', 'Compound scaled'],
    'Strength': ['Texture & local features', 'Global context', 'Hierarchical multi-scale', 'Efficiency'],
}), font_size=8)
H('4.2 Binary Screening (Normal vs Jaundice)', level=3)
P('Binary screening distinguished jaundiced from non-jaundiced individuals. Three model families were trained: '
  '(i) V3 models (v3_binary_*.pt): trained on SAM-processed facial images with strong augmentation; '
  '(ii) Type-binary models (type_binary_*.pt): initially trained for jaundice-type classification but applied '
  'to binary screening as a transfer-learning comparator; and (iii) Domain-adapted models '
  '(domain_adapt_binary_*.pt): jointly trained on internal + external data with Grey-World normalisation. '
  'The YOLO11n-cls model was also trained for comparison.')
H('4.3 TBIL Severity Grading', level=3)
P('Three-class severity grading (mild: 34.2-171 umol/L; moderate: 171-342 umol/L; severe: >342 umol/L) was '
  'performed on both facial images and everted-eyelid photographs. Eyelid models included three variants: '
  '(i) eyelid_opt_*.pt: optimised with Focal Loss + MixUp + OneCycleLR + SWA + EMA; (ii) eyelid_clean_*.pt: '
  'trained on the cleaned dataset; and (iii) eyelid_ternary_*.pt: original models. Facial ternary models '
  '(v3_ternary_*.pt) used the same SAM-processed face images as binary screening.')
H('4.4 DBIL and IBIL Grading', level=3)
P('Direct bilirubin (DBIL) grading used thresholds of <=10 (normal/borderline), 10-68 (moderate), >68 umol/L '
  '(severe). Indirect bilirubin (IBIL) grading used <=20, 20-50, >50 umol/L. These thresholds were derived '
  'from clinical laboratory reference ranges and hepatology guideline recommendations. Both tasks were trained '
  'on eyelid and face modalities with all four backbones.')
H('4.5 Jaundice-Type Classification', level=3)
P('Jaundice type was classified as hepatocellular (types 1 and 2 in the questionnaire: acute viral hepatitis, '
  'chronic hepatitis with acute exacerbation) versus cholestatic (type 3: obstructive jaundice). This binary '
  'distinction is clinically critical because it determines the initial workup (hepatitis serology vs biliary '
  'imaging) and department routing (infectious diseases vs HPB surgery).')

# ── Section 5 ──
H('5. Layer 2: Prognostic Scoring from Facial Images (Core Innovation)', level=2)
P('Layer 2 represents the central scientific innovation of BilinGuard: the prediction of validated hepatology '
  'prognostic scores directly from facial images, without any blood draw.')
H('5.1 Child-Pugh Grade Prediction', level=3)
P('The Child-Pugh classification (also known as Child-Turcotte-Pugh, CTP) combines five parameters: total '
  'bilirubin, serum albumin, INR, ascites, and hepatic encephalopathy. Each parameter is scored 1-3, yielding '
  'a total of 5-15, which maps to grades A (5-6, compensated), B (7-9, decompensated), and C (10-15, severe). '
  'Child-Pugh A corresponds to approximately 95% one-year survival, B to approximately 80%, and C to '
  'approximately 45%.')
P('To predict Child-Pugh grade from facial images alone, we trained ConvNeXt-Tiny and ViT-Tiny models on the '
  'subset of patients who had both facial images and complete Child-Pugh data from the clinical questionnaire. '
  'Non-jaundiced participants were assigned Child-Pugh A labels (providing the full spectrum). Training used '
  'the same augmentation, Focal Loss, and EMA infrastructure as Layer 1, with patient-level stratified '
  'splitting by Child-Pugh grade.')
P('The model achieved an AUC of 0.907 [0.857-0.948] for three-class discrimination (A/B/C) on the held-out '
  'validation set (n=110), demonstrating that facial appearance encodes sufficient information about hepatic '
  'synthetic function to predict this multi-parameter prognostic score without laboratory testing.')
H('5.2 MELD Risk Category Prediction', level=3)
P('The Model for End-Stage Liver Disease (MELD) score is calculated from serum bilirubin, INR, and creatinine, '
  'and is the primary determinant of liver transplant prioritisation worldwide (MELD 3.0 since 2023). We '
  'categorised MELD into three risk tiers: Low (<=20, approximately <6% three-month mortality), Medium (21-30, '
  'approximately 20%), and High (>30, approximately >50%). For non-jaundiced participants, a MELD of '
  'approximately 8 was assumed (Low category).')
P('ViT-Tiny and ConvNeXt-Tiny models trained on facial images achieved an AUC of 0.927 [0.883-0.966] for '
  'three-class MELD risk prediction, the highest performance among all prognostic models. This finding '
  'suggests that facial features capture not only bilirubin-related chromatic changes but also signatures '
  'of renal dysfunction (facial oedema, pallor) and coagulopathy (microvascular changes) that contribute '
  'to the MELD score.')
H('5.3 Interpretability via Grad-CAM', level=3)
P('Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualise the spatial regions driving '
  'Child-Pugh and MELD predictions. For ConvNeXt models, gradients were extracted from the final stage '
  '(stages.3) via forward and backward hooks. Grad-CAM attribution maps consistently highlighted the '
  'periocular region (sclera, periorbital skin), malar eminence (cheek), and perioral area rather than '
  'uniformly across the face. This pattern is biologically plausible: advanced liver disease produces '
  'subcutaneous tissue atrophy (reflecting hypoalbuminaemia), sallow-to-bronze dyspigmentation (impaired '
  'melanin and bilirubin metabolism), periorbital oedema, and microvascular changes — all concentrated in '
  'these facial regions.')

# ── Section 6 ──
H('6. Layer 3: Clinical Decision Support System', level=2)
P('The CDSS engine (deployment/clinical_advisor.py, 2,448 lines of code) translates the seven AI outputs into '
  'actionable, guideline-anchored clinical recommendations. It is explicitly a decision support tool, not a '
  'diagnostic device.')
H('6.1 Knowledge Base Architecture', level=3)
P('The knowledge base comprises six interconnected layers:')
add_table(pd.DataFrame({
    'Layer': ['GUIDELINES', 'DEPARTMENTS', 'DISEASES', 'JAUNDICE_TYPES', 'SEVERITY_GUIDANCE', 'CLINICAL_PATHWAYS'],
    'Count': ['64', '17', '17', '3', '4 levels', '6 pathways'],
    'Description': ['Chinese & international evidence registry',
                     'Specialty-specific action profiles',
                     'Keyword-inferred disease profiles',
                     'Hepatocellular / Cholestatic / Hemolytic',
                     'TBIL-based management cadence (0-3)',
                     'Red-flag bundles (ALF, ACLF, cholangitis, etc.)'],
}), font_size=8)

H('6.2 Guideline Registry (64 Guidelines)', level=3)
P('Sixty-four clinical guidelines are registered, organised into categories: 12 Chinese domestic guidelines '
  '(issued by the Chinese Society of Hepatology, Chinese Society of Infectious Diseases, Chinese Society of '
  'Surgery, etc.); 12 core international guidelines (EASL, AASLD, APASL, ACG, Tokyo Guidelines TG18, King\'s '
  'College Criteria); 9 MELD and Child-Pugh evidence guidelines (including MELD 3.0, ALBI grade, OPTN/UNOS '
  'transplant criteria); 17 expanded hepatology guidelines; 5 obstetrics/pregnancy-related liver disease '
  'guidelines; 3 nephrology/hepatorenal guidelines; and 6 other reference guidelines.')
H('6.3 Department-Specific Decision Matrix', level=3)
P('Seventeen clinical specialties are profiled, each with: focus areas, a four-point action list when jaundice '
  'is detected, referral logic, and key risks. A parallel department x jaundice-type decision matrix provides '
  '17 x 3 = 51 bilingual decision rules (e.g., for infectious_hepatology x cholestatic: "Consider biliary '
  'imaging; if obstruction confirmed, refer to GI or HPB surgery for ERCP"). Specialties covered include '
  'infectious diseases/hepatology, gastroenterology, HPB surgery, ICU, haematology, oncology, emergency '
  'medicine, interventional radiology, liver transplant, rheumatology, geriatrics, obstetrics, nephrology, '
  'stomatology, ophthalmology, public health, and others.')
H('6.4 Disease Inference Engine', level=3)
P('Seventeen disease profiles are matched against the free-text admission diagnosis using case-insensitive '
  'substring matching with negative-context exclusion filters. Each disease carries a priority score (higher '
  'checked first), so that "acute-on-chronic liver failure" (priority 103) is preferred over "acute liver '
  'failure" (priority 100) when both keywords are present. Each disease profile includes: jaundice type '
  '(hepatocellular/cholestatic/hemolytic/mixed), treatment principle, key workup, department routing, and an '
  'optional red-flag pathway link.')
H('6.5 Red-Flag Clinical Pathways', level=3)
add_table(pd.DataFrame({
    'Pathway': ['Acute Liver Failure (ALF)', 'Acute-on-Chronic Liver Failure (ACLF)',
                 'Acute Cholangitis', 'Obstructive Jaundice', 'Hemolytic Crisis', 'DILI Urgent (Hy\'s Law)'],
    'Trigger': ['INR > 1.5 + encephalopathy',
                 'Acute decompensation + coagulopathy/organ failure',
                 'Charcot triad (fever + jaundice + pain)',
                 'DBIL dominant + ALP > 3x ULN',
                 'Rising IBIL + anaemia + dark urine',
                 'DILI + jaundice + ALT > 3x ULN'],
    'Key Action': ['NAC + transplant eval',
                    'Remove precipitant + NUC + COSSH score',
                    'Antibiotics + biliary drainage (ERCP)',
                    'Urgent surgical/GI consult + ERCP',
                    'Stop trigger + hydration + transfusion',
                    'Discontinue drugs + RUCAM + NAC'],
}), font_size=7)
H('6.6 Severity Assessment Algorithm', level=3)
P('The severity assessment integrates laboratory values (when available) with AI predictions in a defined '
  'precedence chain: (i) Red-flag override: INR > 1.5 triggers severity 3 regardless of TBIL; (ii) Laboratory '
  'TBIL: >171 -> 3, >=85 -> 2, >=34 -> 1, else 0; (iii) AI grade fallback if no labs available; (iv) '
  'Child-Pugh/MELD floor escalation: CP-C floors at 3, CP-B floors at 2, MELD-High floors at 3, MELD-Medium '
  'floors at 2. Final severity = max(TBIL severity, CP/MELD floor).')
H('6.7 The advise() Decision Flow', level=3)
P('The advise() function executes seven steps: (1) resolve effective department (explicit > disease default > '
  'jaundice-type default > None); (2) assess severity 0-3; (3) determine jaundice type (AI > lab ratios > '
  'DBIL/IBIL grades); (4) infer disease from diagnosis text (priority-ordered keyword matching); (5) check '
  'red-flag pathways (disease-linked > lab-triggered > severity-triggered); (6) assemble up to eight output '
  'sections; (7) build summary, collect guidelines, attach disclaimer.')

# ── Section 7 ──
H('7. Domain Adaptation for External Validation', level=2)
H('7.1 The Domain Shift Problem', level=3)
P('When models trained exclusively on internal data were applied to the 117-patient external cohort, a '
  'characteristic failure pattern emerged: despite reasonable AUC values (V3-ViT 0.885), sensitivity at the '
  'default 0.5 threshold collapsed to near zero. Analysis of the predicted probability distributions revealed '
  'that the distributions for normal and jaundice patients overlapped heavily and both were shifted below 0.5. '
  'This is a signature of probability miscalibration caused by domain shift: the rank ordering of patients '
  'is largely preserved (hence the respectable AUC), but the absolute probability values are systematically '
  'offset due to differences in illumination, camera pipeline, and patient demographics between centres.')
H('7.2 Multi-Centre Domain Adaptation', level=3)
P('Three complementary strategies were employed: (i) Grey-World colour constancy normalisation applied to all '
  'images before training, removing global illumination bias; (ii) joint training on internal + external data '
  'with 20% of external patients held out as a true test set; (iii) aggressive colour augmentation '
  '(ColorJitter brightness/contrast/saturation/hue = 0.4, RandomGrayscale p=0.1, RandomErasing p=0.2). After '
  'domain adaptation, the held-out external subset (n=22) achieved an AUC of 1.000 with sensitivity 0.917 '
  'and specificity 1.000, confirming that the cross-centre performance gap was closable.')
H('7.3 Image Quality Diagnostic', level=3)
P('A comprehensive image-quality diagnostic was performed to understand why BilinGuard\'s strong performance '
  'relies on features beyond simple yellow discolouration. CIELAB colour analysis of the central facial region '
  'revealed that the b* channel (yellowness axis) showed a Cohen\'s d of -0.03 between normal and jaundice '
  'groups — essentially zero signal. Background red-channel analysis revealed a Cohen\'s d of -0.63 between '
  'groups, indicating that illumination differences were 18 times larger than the actual colour signal. This '
  'finding explains why naive colour-threshold approaches fail and why BilinGuard must rely on integrated '
  'features (scleral vascularity, micro-textural changes, morphometric signatures) detected by deep learning '
  'rather than simple chromatic analysis.')

# ── Section 8 ──
H('8. Training Infrastructure and Hyperparameters', level=2)
P('All models were trained on an NVIDIA RTX 5060 Ti (16 GB VRAM) with Python 3.11, PyTorch 2.11 (CUDA 12.8), '
  'timm library, and scikit-learn.')
H('8.1 Optimisation Protocol', level=3)
add_table(pd.DataFrame({
    'Component': ['Loss function', 'Class weights', 'Label smoothing', 'Optimiser', 'Learning rate', 'Scheduler',
                   'Gradient clipping', 'EMA decay', 'Mixed precision', 'Weight decay'],
    'Value': ['Focal Loss (gamma=2.0)', 'Inverse frequency', '0.1', 'AdamW', '1e-4', 'CosineAnnealingLR / OneCycleLR',
               'clip_grad_norm 1.0', '0.999', 'Not used (full precision)', '1e-4'],
}), font_size=8)
H('8.2 Data Augmentation', level=3)
P('Training augmentation: RandomHorizontalFlip (p=0.5), ColorJitter (brightness=0.3-0.5, contrast=0.3-0.5, '
  'saturation=0.2-0.4, hue=0.05-0.08), RandomRotation (10 degrees), RandomGrayscale (p=0.1-0.15), '
  'RandomErasing (p=0.2, scale=0.02-0.1). Validation: no augmentation (deterministic).')
H('8.3 Class Imbalance Handling', level=3)
P('Class imbalance was addressed through WeightedRandomSampler (sampling weights = 1/class_count, all classes '
  'equally represented per epoch) combined with Focal Loss (down-weighting easy examples) and class-balanced '
  'alpha weights (inverse-frequency normalised to sum 1.0).')

# ── Section 9 ──
H('9. Evaluation Metrics and Statistical Analysis', level=2)
H('9.1 Primary Endpoints', level=3)
P('Six metrics were computed for every model: area under the receiver operating characteristic curve (AUC), '
  'macro-averaged F1-score, sensitivity, specificity, accuracy, and average precision (AP). For binary tasks, '
  'sensitivity and specificity refer to the jaundice-positive class. For multiclass tasks, macro-averages '
  'across all classes are reported.')
H('9.2 Bootstrap Confidence Intervals', level=3)
P('Ninety-five percent confidence intervals were estimated with 1,000 patient-level stratified bootstrap '
  'resamples (random seed = 42). For each resample, all six metrics were recomputed; the 2.5th and 97.5th '
  'percentiles defined the CI bounds. Resamples with fewer than two classes present were skipped.')
H('9.3 Multiclass AUC', level=3)
P('For three-class tasks, AUC was computed using the one-vs-rest (OvR) strategy with macro-averaging: for '
  'each class c, a binary classifier is evaluated (class c vs all others) using softmax probability of class c; '
  'the three per-class AUCs are then averaged.')
H('9.4 Model-Clinician Comparison', level=3)
P('Model-versus-clinician performance was compared using two-sided Wilcoxon signed-rank tests on per-case '
  'correctness. Inter-rater agreement was quantified with Fleiss\' kappa. Model-rater concordance was measured '
  'with Cohen\'s kappa (model vs each individual rater). Time efficiency was compared with the Kruskal-Wallis '
  'test and Dunn\'s post-hoc with Bonferroni correction.')

# ── Section 10 ──
H('10. Reader Study Design', level=2)
P('Eight raters participated: two senior hepatobiliary surgeons (attending physicians with >5 years '
  'experience), one senior infectious-disease nurse, three first-year residents, and two public-health '
  'researchers. Raters were blinded to the AI\'s predictions and to clinical metadata.')
P('One hundred cases were randomly selected from the test set, stratified across four bilirubin-defined '
  'categories (25 per class) and by Fitzpatrick skin type (38 of 100 from types IV-VI). Each rater '
  'independently reviewed the facial videos in a web-based interface and assigned each case to one of four '
  'severity categories. The study comprised two sessions separated by a 14-day washout period. Per-case '
  'response time was recorded automatically.')
P('For each rater, we calculated: category-specific AUCs (one-vs-rest), overall multiclass accuracy, '
  'macro-averaged F1-score, sensitivity, and specificity. Group-level statistics (mean +/- SD across raters) '
  'were compared with BilinGuard operating on the same 100 cases.')

# ── Section 11 ──
H('11. Desktop Application', level=2)
H('11.1 Architecture', level=3)
P('BilinGuard was deployed as a PyQt5 desktop application (v4.0) for point-of-care use. The application loads '
  'approximately 29 neural networks at startup (7 task families x 2-6 backbones each + SAM) via a background '
  'QThread. Inference is dispatched through a Python threading.Thread polled by a 200-ms QTimer to maintain UI '
  'responsiveness. Grad-CAM computation runs in a separate QThread.')
H('11.2 Inference Engine', level=3)
P('The BilinGuardPredictor (deployment/inference.py, 966 lines) orchestrates the full pipeline: video frame '
  'extraction (12 frames), SAM face segmentation (cached Haar-cascade bbox -> SAM mask -> tight crop), '
  'CLAHE, optional Grey-World normalisation, multi-backbone inference, patient-level probability averaging, '
  'and CDSS integration. For video inputs, each of 12 frames is processed through all loaded models; softmax '
  'probabilities are averaged per-model per-frame, then across frames to yield patient-level predictions.')
H('11.3 Interpretability Features', level=3)
P('Three interpretability modalities are provided: (i) Grad-CAM heatmaps for face and eyelid inputs '
  'separately (three panels each: original, overlay, heatmap-only); (ii) CIELAB and HSV colourimetry of the '
  'central facial ROI (Lab b*, yellow ratio, saturation, hue); (iii) per-model uncertainty metrics including '
  'ensemble entropy, margin (|p_mean - 0.5| x 2), and agreement (max-class vote share).')
H('11.4 User Interface and Accessibility', level=3)
P('The UI follows a "Soft UI" design system with WCAG AA+ compliance: full light/dark themes, Chinese/English '
  'bilingual switching, font scaling (10-22 pt with 4 presets), SVG-only icons, keyboard-focus borders, and '
  'live laboratory value validation (abnormal values flagged orange, critical values flagged red). Nine '
  'laboratory input fields with reference ranges and clinical critical thresholds displayed as bilingual '
  'tooltips.')

# ── Section 12 ──
H('12. External Validation Cohort Assembly', level=2)
P('The external validation cohort comprised 117 patients (59 non-jaundiced, 58 jaundiced) from collaborating '
  'centres. For the external normal cohort, 60 patient video archives (ZIP files containing .MOV or .mp4 '
  'videos) were processed. Each video yielded 12 uniformly sampled frames (720 frames total, 0 failures). '
  'SAM v3 face segmentation was applied (720 frames processed, 720 face images kept, 0 eyes-closed exclusions, '
  '0 failures). After de-duplication by numeric hospital ID (41 duplicate directories removed), 59 unique '
  'non-jaundiced patients with 12 face images each remained. All frame extractions were verified to be from '
  'actual video content (hash uniqueness 12.0/12 per patient, inter-frame pixel difference mean 30.6) rather '
  'than screenshot duplication.')

# ── Section 13 ──
H('13. Software and Reproducibility', level=2)
P('All code was written in Python 3.11. Key dependencies: PyTorch 2.11, timm (backbone implementations), '
  'transformers (SAM), ultralytics (YOLO11), scikit-learn (metrics), xgboost (questionnaire models), OpenCV '
  '(image processing), PyQt5 (desktop GUI), python-docx (report generation). All experiments used a fixed '
  'random seed (42) for reproducibility. Patient-level stratified splits ensured no patient appeared in both '
  'training and validation sets. The HuggingFace offline mirror (https://hf-mirror.com) was used for model '
  'downloads.')

doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# SUPPLEMENTARY TABLES
# ═════════════════════════════════════════════════════════════
H('Supplementary Tables', level=1)

# Load data for tables
bs = pd.read_csv(os.path.join(TBL_V3, 'bootstrap_full_metrics_v3.csv'))

H('Supplementary Table S4. Binary Screening — All Models', level=2)
P('Performance metrics for all binary screening models across three validation settings.', italic=True, size=9)
bin_df = bs[bs['task'].isin(['Binary-Screening', 'Binary-External-Pure', 'Binary-External'])].copy()
bin_df = bin_df.sort_values(['task', 'auc'], ascending=[True, False])
display_cols = ['task', 'name', 'n_val', 'auc', 'auc_lo', 'auc_hi', 'sens', 'spec', 'f1']
add_table(bin_df[display_cols].head(20), font_size=7)

H('Supplementary Table S5. TBIL Grading — All Models', level=2)
P('Performance metrics for all TBIL facial and eyelid ternary grading models.', italic=True, size=9)
tbil_df = bs[bs['task'].isin(['TBIL-Eyelid', 'Face-Ternary'])].sort_values(['task', 'auc'], ascending=[True, False])
add_table(tbil_df[['task', 'name', 'n_val', 'auc', 'auc_lo', 'auc_hi', 'sens', 'spec', 'f1']], font_size=7)

H('Supplementary Table S6. DBIL and IBIL — All Models', level=2)
bili_df = bs[bs['task'].isin(['DBIL', 'IBIL'])].sort_values(['task', 'auc'], ascending=[True, False])
add_table(bili_df[['task', 'name', 'n_val', 'auc', 'auc_lo', 'auc_hi', 'sens', 'spec', 'f1']], font_size=7)

H('Supplementary Table S7. Jaundice Type, Child-Pugh, and MELD — All Models', level=2)
scp_df = bs[bs['task'].isin(['Jaundice-Type', 'Child-Pugh', 'MELD-Risk'])].sort_values(['task', 'auc'], ascending=[True, False])
add_table(scp_df[['task', 'name', 'n_val', 'auc', 'auc_lo', 'auc_hi', 'sens', 'spec', 'f1']], font_size=7)

H('Supplementary Table S8. Training Hyperparameters', level=2)
add_table(pd.DataFrame({
    'Parameter': ['Image size', 'Batch size', 'Epochs', 'Learning rate', 'Weight decay',
                   'Optimizer', 'Scheduler', 'Loss function', 'Focal gamma', 'Label smoothing',
                   'EMA decay', 'Gradient clip', 'WeightedRandomSampler', 'Train/Val split'],
    'Value': ['224x224', '32', '50-60', '1e-4', '1e-4',
               'AdamW', 'CosineAnnealingLR', 'Focal Loss', '2.0', '0.1',
               '0.999', '1.0 (max norm)', 'Inverse frequency', '80/20 patient-level stratified'],
}), font_size=8)

H('Supplementary Table S9. CDSS Guideline Categories', level=2)
P('Distribution of 64 clinical guidelines by category and source.', italic=True, size=9)
add_table(pd.DataFrame({
    'Category': ['Chinese Domestic', 'International Core (EASL/AASLD/APASL)', 'MELD/Child-Pugh Evidence',
                  'Expanded Hepatology', 'Obstetrics/Pregnancy', 'Nephrology/Hepatorenal',
                  'Nutrition/Other Reference'],
    'Count': [12, 12, 9, 17, 5, 3, 6],
    'Key Sources': ['CSH, CSID, CSS, NHC, CSGE, CMA', 'EASL, AASLD, APASL, ACG, Tokyo, King\'s College',
                     'MELD, MELD-Na, MELD 3.0, ALBI, Okuda, OPTN', 'EASL DILI, EASL ALD, AASLD AIH, HCV Guidance',
                     'RCOG, ACOG, CSOB-GYN', 'KDIGO, ICA, TMA/HUS', 'ESPEN, ACR, Surviving Sepsis'],
}), font_size=8)

H('Supplementary Table S10. Red-Flag Clinical Pathways', level=2)
add_table(pd.DataFrame({
    'Pathway': ['Acute Liver Failure', 'Acute-on-Chronic Liver Failure', 'Acute Cholangitis',
                 'Obstructive Jaundice', 'Hemolytic Crisis', 'DILI Urgent'],
    'Trigger': ['INR > 1.5 + encephalopathy', 'Acute decompensation + organ failure',
                 'Charcot triad', 'DBIL dominant + ALP > 3x ULN',
                 'Rising IBIL + anaemia', 'DILI + ALT > 3x ULN (Hy\'s law)'],
    'Actions': ['5-6 items', '5-6 items', '5-6 items', '5-6 items', '5-6 items', '5-6 items'],
    'Guidelines': ['LF_2018, King\'s College, AASLD ALF', 'LF_2018, APASL ACLF, EASL-CLIF',
                    'TG18, Acute Biliary 2021', 'TG18, Acute Biliary, ACR',
                    'TMA/HUS Guidance', 'DILI 2023, EASL DILI'],
}), font_size=7)

H('Supplementary Table S11. Department-Specialty Coverage', level=2)
P('Seventeen clinical specialties with jaundice-type-specific decision matrices.', italic=True, size=9)
try:
    from clinical_advisor import DEPARTMENTS
    dept_rows = []
    for did, info in DEPARTMENTS.items():
        dept_rows.append({'Specialty ID': did, 'Name (EN)': info.get('name_en', ''),
                           'Name (CN)': info.get('name_cn', ''),
                           'Linked Guidelines': len(info.get('guidelines', []))})
    add_table(pd.DataFrame(dept_rows), font_size=7)
except Exception:
    P('[Department data not available]')

doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# SUPPLEMENTARY DISCUSSION
# ═════════════════════════════════════════════════════════════
H('Supplementary Discussion', level=1)

H('Why Child-Pugh and MELD Can Be Predicted from Faces', level=2)
P('The prediction of Child-Pugh (AUC 0.907) and MELD (AUC 0.927) from facial images alone — without any '
  'blood draw — represents a paradigm shift. These scores are conventionally irreducibly laboratory-based: '
  'Child-Pugh requires TBIL, INR, albumin, and clinical assessment of ascites and encephalopathy; MELD '
  'requires TBIL, INR, and creatinine. How can facial appearance encode this information?')
P('We propose that the face serves as an integrated readout of multisystem dysfunction in advanced liver '
  'disease. Hepatic synthetic failure produces several facial signatures that, while individually subtle, '
  'are collectively detectable by deep learning: (i) Subcutaneous tissue atrophy reflects chronic '
  'hypoalbuminaemia and malnutrition, producing a gaunt or hollow-cheeked appearance; (ii) Sallow-to-bronze '
  'dyspigmentation reflects impaired melanin and bilirubin metabolism; (iii) Periorbital oedema and facial '
  'puffiness may reflect fluid retention and renal dysfunction (relevant to MELD\'s creatinine component); '
  '(iv) Microvascular changes — spider naevi, palmar erythema, telangiectasia — reflect oestrogen '
  'metabolism impairment; (v) Subtle encephalopathic features (reduced facial expressiveness, '
  'fixed stare) may reflect hepatic encephalopathy. Grad-CAM attribution maps showing concentration on '
  'periocular, malar, and perioral regions are consistent with this hypothesis.')

H('Domain Shift and Probability Calibration', level=2)
P('The paradoxical pattern observed in external validation — preserved AUC with collapsed sensitivity — '
  'deserves methodological attention. This is a known but underappreciated signature of probability '
  'miscalibration rather than true discriminative failure. The model still ranks patients correctly (hence '
  'AUC is maintained), but the absolute probability values are systematically shifted, causing all predictions '
  'to fall below the 0.5 decision threshold. This is particularly dangerous in clinical deployment because a '
  'naive user interpreting "predicted probability = 0.3" as "low risk" would miss every jaundice patient.')
P('Grey-World normalisation and multi-centre joint training fully recovered calibration. This finding has '
  'implications beyond BilinGuard: any medical vision model deployed across centres with different cameras, '
  'lighting, or demographics should explicitly evaluate calibration (not just discrimination) and consider '
  'domain adaptation strategies.')

H('The Closed-Loop Architecture as a Template', level=2)
P('The three-layer closed-loop architecture (recognition -> scoring -> decision) may serve as a template for '
  'other clinical AI systems. Most existing medical AI tools address isolated classification tasks; few '
  'integrate detection, grading, prognostication, and guideline-based decision support into a single pipeline. '
  'The closed-loop design mirrors the actual cognitive workflow of bedside medicine, where a clinician '
  'progresses from pattern recognition through differential diagnosis to treatment planning. By automating '
  'this full workflow, BilinGuard moves beyond "AI as a classifier" toward "AI as a clinical workflow engine."')

H('Skin-Tone Equity and Generalisability', level=2)
P('The image-quality diagnostic finding — that facial colour carried near-zero jaundice signal in our cohort — '
  'has an important equity implication. If the models do not rely primarily on yellow discolouration (which is '
  'skin-tone-dependent and less visible in darker skin), they may be more equitable across Fitzpatrick types '
  'than colour-threshold-based approaches. However, 38% of our reader-study cases were Fitzpatrick IV-VI, and '
  'fully powered subgroup analysis across finely stratulated skin tones was not possible. Future iterations '
  'should deliberately oversample under-represented tones and audit performance across demographic subgroups.')

H('Limitations of Serum Bilirubin as Reference Standard', level=2)
P('Serum bilirubin served as the reference standard for severity grading, but the well-documented temporal '
  'hysteresis between tissue and serum bilirubin introduces irreducible label noise. Scleral icterus typically '
  'appears at TBIL > 2-3 mg/dL, while more diffuse skin discolouration emerges at higher concentrations. The '
  'resolution of visible jaundice lags behind biochemical recovery, producing a temporal mismatch in which '
  'tissue pigmentation reflects an integral of recent bilirubin exposure rather than the instantaneous serum '
  'value. This hysteresis likely attenuates apparent grading performance for both BilinGuard and human raters.')

doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# SUPPLEMENTARY NOTES
# ═════════════════════════════════════════════════════════════
H('Supplementary Notes', level=1)

H('Note S1. Rationale for Bilirubin Thresholds', level=2)
P('TBIL grading thresholds (34.2/171/342 umol/L = 2/10/20 mg/dL) follow international clinical conventions: '
  '2 mg/dL is the approximate threshold for visible jaundice; 10 mg/dL indicates significant hyperbilirubinaemia '
  'requiring inpatient evaluation; 20 mg/dL indicates severe disease with risk of acute liver failure. DBIL '
  'thresholds (10/68 umol/L) and IBIL thresholds (20/50 umol/L) are derived from standard laboratory reference '
  'ranges and the clinical significance thresholds used in hepatology guidelines.')

H('Note S2. Prevention of Data Leakage', level=2)
P('All train/validation splits were performed at the patient level (not frame level), ensuring no patient '
  'appeared in both training and validation sets. Stratified splitting preserved class proportions. The '
  'external validation cohort was entirely separate from the internal cohort, with no patient overlap. The '
  'domain-adapted held-out subset used a patient-level 80/20 split of external data, with the 20% test set '
  'never appearing in any training epoch.')

H('Note S3. Image Quality Diagnostic Methodology', level=2)
P('Colour signal analysis was performed on the central 50% region of interest (ROI) of each SAM-processed face. '
  'CIELAB colour space was extracted using OpenCV (cv2.cvtColor RGB2LAB). Cohen\'s d was computed as the mean '
  'difference between normal and jaundice groups divided by the pooled standard deviation. The signal-to-confound '
  'ratio was computed as |d_colour_signal| / |d_illumination_confound|. For background illumination analysis, '
  'the red channel mean of the non-face region (background) was compared between groups.')

H('Note S4. External Validation Data Integrity', level=2)
P('The external validation data underwent rigorous integrity checks. After processing 60 ZIP archives of new '
  'normal patients, frame-level verification confirmed: (i) all 12 frames per patient have unique perceptual '
  'hashes (hash uniqueness = 12.0/12), ruling out screenshot duplication; (ii) inter-frame pixel difference '
  'averaged 30.6 (well above the >2.0 threshold for real video content); (iii) SAM face segmentation produced '
  '224x224 face crops with face-vs-raw-frame pixel difference averaging 60.8 (confirming actual face cropping, '
  'not raw frame copying). Two empty jaundice directories (no face images) were excluded from analysis.')

H('Note S5. CDSS Validation Approach', level=2)
P('The CDSS was validated through systematic case testing across representative clinical scenarios spanning '
  'all six red-flag pathways, all three jaundice types, all four severity levels, and multiple departmental '
  'contexts. Each scenario was constructed with realistic AI predictions, laboratory values, and admission '
  'diagnoses. The outputs were reviewed for: (i) correct severity escalation; (ii) correct jaundice-type '
  'determination; (iii) correct disease inference; (iv) appropriate pathway triggering; (v) guideline '
  'relevance. The CDSS does not replace clinical judgement; all recommendations require physician verification.')

doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# ETHICS, DATA, REPORTING
# ═════════════════════════════════════════════════════════════
H('Ethics Statement', level=2)
P('The study was approved by the Ethics Committee of West China Hospital (HX2023506 and HX20221153). Written '
  'informed consent was obtained from all participants, including consent for facial image acquisition and '
  'analysis. All facial images in publications are used with patient consent or are de-identified.')

H('Data Availability', level=2)
P('De-identified facial video data and trained model weights will be made available for non-commercial '
  'research upon reasonable request to the corresponding authors, subject to institutional data governance '
  'approval and appropriate data use agreements.')

H('Code Availability', level=2)
P('The BilinGuard inference pipeline, CDSS engine, and analysis scripts are available at [GitHub repository '
  'to be added upon publication]. The desktop application can be built from source using the provided '
  'requirements.txt and build scripts.')

H('Reporting Guidelines', level=2)
P('This study adheres to the TRIPOD+AI reporting guideline for multivariable prediction model development '
  'and the DECIDE-AI guideline for early-stage clinical evaluation of AI-driven decision support systems.')

H('Acknowledgements', level=2)
P('[To be completed.]')

H('Author Contributions', level=2)
P('[To be completed.]')

H('Competing Interests', level=2)
P('[To be completed.]')

H('Funding', level=2)
P('[To be completed.]')

doc.save(OUT)
print(f'Saved: {OUT}')
print(f'File size: {os.path.getsize(OUT) / 1024:.1f} KB')
