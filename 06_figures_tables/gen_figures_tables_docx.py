# -*- coding: utf-8 -*-
"""
Generate a single DOCX containing ALL figures and tables with legends.

Includes:
  - 5 Main Figures (Figure 1-5) + Grad-CAM gallery + CDSS cases
  - 6 Supplementary Figures (SuppFig 1-6)
  - 5 Main Tables (Table 1-5)
  - 3 Supplementary Tables (S1-S3)
  - Complete legends/captions for every figure and table
"""
import os, pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = r'D:\research\人脸识别营养\传染科'
RES = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(RES, 'figures_v3', 'main')
TBL_V3 = os.path.join(RES, 'tables_v3')
OUT = os.path.join(BASE, 'Figures_and_Tables_Collection.docx')

doc = Document()

# Page setup
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

# Default font
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(10)

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
    return h

def add_para(text, bold=False, italic=False, size=10):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p

def add_figure(png_path, width=6.5, caption=None):
    if os.path.exists(png_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(png_path, width=Inches(width))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = cap.add_run(caption)
        run.font.size = Pt(9)
        run.italic = True

def add_table_from_csv(csv_path, max_rows=None, font_size=7):
    if not os.path.exists(csv_path):
        add_para(f'[Table not found: {csv_path}]')
        return
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)
    cols = list(df.columns)
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for i, c in enumerate(cols):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True; r.font.size = Pt(font_size)
    # Rows
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_size)

# ═════════════════════════════════════════════════════════════
# TITLE PAGE
# ═════════════════════════════════════════════════════════════
title = doc.add_heading('', level=0)
run = title.add_run('Figures and Tables Collection')
run.font.size = Pt(22); run.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
add_para('BilinGuard: A Closed-Loop AI System from Non-Invasive Facial Jaundice Recognition '
         'to Prognostic Scoring and Clinical Decision Support', bold=True, size=12)
add_para('West China Hospital, Sichuan University, Chengdu, China', italic=True, size=10)
add_para('')
add_para('This document contains all main figures (Figure 1-5), supplementary figures '
         '(Supplementary Figure S1-S6), main tables (Table 1-5), and supplementary tables '
         '(Supplementary Table S1-S3) with complete legends.', size=9)
doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# PART I: MAIN FIGURES
# ═════════════════════════════════════════════════════════════
add_heading('Part I. Main Figures', level=1)

# Figure 1
add_heading('Figure 1. System Overview', level=2)
add_figure(os.path.join(FIG_DIR, 'Figure1_system_overview.png'), width=6.5,
            caption='Figure 1. BilinGuard closed-loop AI system architecture, study cohort, and inference pipeline. '
            '(A) Three-layer architecture: facial video input drives Layer 1 (jaundice recognition: binary screening, '
            'TBIL/DBIL/IBIL grading, jaundice-type classification), Layer 2 (prognostic scoring: Child-Pugh and MELD '
            'prediction without blood draws), and Layer 3 (clinical decision support: 64 guidelines, 17 specialties, '
            '6 red-flag pathways). (B) CONSORT-style cohort flow: 1,206 screened, 939 enrolled (625 non-jaundice + '
            '313 jaundice internal), 117 external validation patients, 579 entered model development after SAM '
            'processing. (C) Deep learning pipeline: 12-frame sampling, SAM face segmentation, CLAHE + Grey-World '
            'normalisation, multi-backbone ensemble (ConvNeXt/ViT/Swin/EfficientNet), patient-level aggregation, '
            'CDSS integration. (D) Performance summary: 10 tasks, 67 models, best AUC per task.')
doc.add_page_break()

# Figure 2
add_heading('Figure 2. Binary Screening and Domain Adaptation', level=2)
add_figure(os.path.join(FIG_DIR, 'Figure2_binary_screening.png'), width=6.5,
            caption='Figure 2. Layer 1 — Binary jaundice screening performance. (A) Internal validation ROC with '
            '95% CI band (n=110, ViT AUC=0.993 [0.979-1.000]). (B) External validation without domain adaptation '
            '(n=117, V3-ViT AUC=0.885 [0.826-0.937]). (C) Domain-adapted held-out subset (n=22, all models '
            'AUC greater than or equal to 0.99). (D) Confusion matrix comparison internal vs external. '
            '(E) Six-metric radar chart (AUC/F1/Sens/Spec/Acc/AP). (F) Calibration curves with Brier scores.')
doc.add_page_break()

# Figure 3
add_heading('Figure 3. Multiclass Grading and Jaundice Type', level=2)
add_figure(os.path.join(FIG_DIR, 'Figure3_grading_v2.png'), width=6.5,
            caption='Figure 3. Layer 1 — Multiclass severity grading and jaundice-type classification. '
            '(A) TBIL facial ternary grading ROC (ConvNeXt AUC=0.955 [0.885-0.998], n=40). '
            '(B) TBIL eyelid ternary grading ROC (ConvNeXt AUC=0.858 [0.733-0.953], n=39). '
            '(C) Precision-recall curves across all grading tasks. '
            '(D) TBIL facial per-class one-vs-rest ROC with 95% CI bands (Mild/Moderate/Severe). '
            '(E) DBIL eyelid (AUC=0.815, n=52) and IBIL facial (AUC=0.720, n=55) ternary grading. '
            '(F) Jaundice-type classification (hepatocellular vs cholestatic, EfficientNet AUC=0.993, n=39) '
            'with confusion matrix.')
doc.add_page_break()

# Figure 4
add_heading('Figure 4. Non-Invasive Prognostic Scoring (Core Innovation)', level=2)
add_figure(os.path.join(FIG_DIR, 'Figure4_prognostic_scoring.png'), width=6.5,
            caption='Figure 4. Layer 2 — Non-invasive prediction of Child-Pugh and MELD from facial images. '
            '(A) Child-Pugh grade (A/B/C) prediction ROC with 95% CI band (ConvNeXt AUC=0.907 [0.857-0.948], '
            'n=110). (B) MELD risk category (Low/Medium/High) ROC (ViT AUC=0.927 [0.883-0.966], n=110). '
            'Both scores conventionally require serum chemistry but are here predicted from facial images alone. '
            '(C) Child-Pugh per-class one-vs-rest ROC. (D) MELD per-class one-vs-rest ROC. '
            '(E) Calibration curves for both prognostic models. (F) Confusion matrices with row-normalised '
            'percentages for Child-Pugh and MELD.')
doc.add_page_break()

# Figure 4C
add_heading('Figure 4 (continued). Grad-CAM Attribution Gallery', level=2)
add_figure(os.path.join(FIG_DIR, 'Figure4C_gradcam_gallery.png'), width=5.5,
            caption='Figure 4 (continued). Grad-CAM attribution maps for four representative patients, '
            'showing what the prognostic models attend to. Rows: Child-Pugh A (TBIL=49.6), Child-Pugh B '
            '(TBIL=174.1), Child-Pugh C (TBIL=409.5), and MELD High (MELD=43). Columns: Original CLAHE-processed '
            'face, Child-Pugh model Grad-CAM overlay, MELD model Grad-CAM overlay. Attention concentrates on '
            'the periocular region, malar eminence, and perioral area rather than uniformly across the face, '
            'consistent with detection of subcutaneous atrophy, dyspigmentation, and microvascular signatures '
            'of hepatic dysfunction.')
doc.add_page_break()

# Figure 5
add_heading('Figure 5. Reader Study and Clinical Decision Support', level=2)
add_figure(os.path.join(FIG_DIR, 'Figure5_reader_study.png'), width=6.5,
            caption='Figure 5. Reader study — BilinGuard versus eight clinicians. (A) Binary screening ROC '
            'with eight clinician operating points overlaid (BilinGuard AUC=0.993 vs clinician range 0.76-0.99). '
            '(B) Three-class grading ROC with clinician points (BilinGuard macro-AUC=0.798 vs clinician 0.59-0.73). '
            '(C) Per-rater AUC/Accuracy/F1 bar chart, BilinGuard highlighted, coloured by role '
            '(Attending/Nurse/Resident/Public Health). (D) Sensitivity-specificity scatter: eight clinicians '
            '(role-coded markers) and BilinGuard (red star). (E) Time efficiency boxplot (clinicians 16.5-19.0 s/case '
            'vs BilinGuard approximately 0.5 s/case, Kruskal-Wallis p=0.004). (F) Wilcoxon signed-rank test results '
            '(-log10 p), with Fleiss kappa and Cohen kappa annotations.')
doc.add_page_break()

# CDSS Cases
add_heading('Figure 5 (continued). CDSS Case Demonstrations', level=2)
add_figure(os.path.join(FIG_DIR, 'Figure5_CDSS_case1.png'), width=6.0,
            caption='Figure 5 (continued), Case 1. Acute-on-chronic liver failure (ACLF). The CDSS escalates '
            'severity to Level 3 (INR > 1.5), classifies jaundice as hepatocellular, infers ACLF from the '
            'admission diagnosis, triggers the ACLF red-flag pathway (immediate NUC therapy, COSSH-ACLF scoring, '
            'transplant evaluation), and cites five guidelines.')
add_figure(os.path.join(FIG_DIR, 'Figure5_CDSS_case2.png'), width=6.0,
            caption='Case 2. Obstructive jaundice due to choledocholithiasis. The CDSS classifies jaundice as '
            'cholestatic (DBIL-dominant, ALP/GGT elevated), infers biliary obstruction, triggers the obstructive '
            'jaundice pathway (urgent ERCP, coagulopathy correction), and routes to HPB surgery.')
add_figure(os.path.join(FIG_DIR, 'Figure5_CDSS_case3.png'), width=6.0,
            caption='Case 3. Mild viral hepatitis (HEV). The CDSS assigns Severity Level 1, classifies as '
            'hepatocellular, recommends outpatient monitoring with weekly LFTs, and cites chronic hepatitis '
            'guidelines.')
doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# PART II: MAIN TABLES
# ═════════════════════════════════════════════════════════════
add_heading('Part II. Main Tables', level=1)

add_heading('Table 1. Baseline Characteristics', level=2)
add_para('Table 1. Baseline characteristics of included participants. Data are n (%) or mean +/- SD.', italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'Table1_baseline.csv'))
doc.add_paragraph()

add_heading('Table 2. Layer 1 Recognition Performance', level=2)
add_para('Table 2. Performance metrics for all jaundice recognition models (Layer 1). Values are point '
         'estimates with 95% bootstrap confidence intervals. Binary-Screening: internal n=110; Binary-External-Pure: '
         'n=117; Binary-External (domain-adapted): n=22; TBIL-Eyelid: n=39-42; Face-Ternary: n=40; DBIL: n=52; '
         'IBIL: n=55; Jaundice-Type: n=39.', italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'Table2_layer1_recognition.csv'), font_size=6)
doc.add_page_break()

add_heading('Table 3. Layer 2 Prognostic Scoring', level=2)
add_para('Table 3. Non-invasive prediction of Child-Pugh grade and MELD risk from facial images (Layer 2). '
         'Both scores are conventionally laboratory-based (requiring TBIL, INR, albumin, creatinine, ascites, '
         'encephalopathy grading) but are here predicted without any blood draw. N=110 for all models.',
         italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'Table3_layer2_scoring.csv'))
doc.add_paragraph()

add_heading('Table 4. Reader Study Results', level=2)
add_para('Table 4. Reader study — BilinGuard versus eight clinicians for binary screening and three-class grading.',
         italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'Table4_reader_study.csv'))
add_para('Inter-rater Fleiss kappa = 0.69 (binary). Model-rater Cohen kappa = 0.83 +/- 0.06. '
         'Wilcoxon signed-rank: all raters p < 0.06, six of eight p < 0.01. '
         'Time efficiency: Kruskal-Wallis H=20.6, p=0.004.', italic=True, size=8)
doc.add_page_break()

add_heading('Table 5. CDSS Knowledge Base', level=2)
add_para('Table 5. Clinical decision support system knowledge base summary.', italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'Table5_cdss_knowledge.csv'))
doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# PART III: SUPPLEMENTARY FIGURES
# ═════════════════════════════════════════════════════════════
add_heading('Part III. Supplementary Figures', level=1)

supp_figs = [
    ('SuppFig1_forest_all_models.png',
     'Supplementary Figure S1. Forest plot of all 67 models across 10 tasks. '
     'Each point represents AUC-ROC with 95% bootstrap confidence interval (1000 iterations, '
     'patient-level resampling). Layer 1 (recognition) models shown in blue; Layer 2 (prognostic scoring) '
     'in orange. Horizontal dashed line indicates AUC=0.5 (random).'),
    ('SuppFig2_summary_3layer.png',
     'Supplementary Figure S2. Best AUC per task across the three-layer closed loop. '
     'Eight Layer 1 recognition tasks (blue) and two Layer 2 prognostic scoring tasks (orange). '
     'Error bars show 95% CI. Model name and validation set size annotated for each bar.'),
    ('SuppFig3_domain_adaptation.png',
     'Supplementary Figure S3. Domain adaptation restores external performance. '
     '(A) Before adaptation: predicted probability distributions for normal (teal) and jaundice (coral) '
     'patients on the 117-patient external cohort (V3-ViT). Probabilities collapse below the 0.5 threshold, '
     'causing zero sensitivity despite AUC=0.885. (B) After adaptation (Grey-World + multi-centre training): '
     'probabilities are well-separated, restoring sensitivity to 0.917. (C) AUC comparison across three '
     'validation approaches.'),
    ('SuppFig4_confusion_matrices.png',
     'Supplementary Figure S4. Confusion matrices for all 10 tasks (best model per task). '
     'Row-normalised percentages shown. Two-by-two matrices for binary tasks; three-by-three for ternary. '
     'Tasks: Binary Internal/External/Domain-Adapted, TBIL Face/Eyelid, DBIL, IBIL, Jaundice Type, '
     'Child-Pugh, MELD.'),
    ('SuppFig5_binary_models.png',
     'Supplementary Figure S5. All binary screening models compared. '
     '(A) AUC-ROC with 95% CI for all 28 binary models across three validation settings '
     '(internal n=110, external pure n=117, domain-adapted n=22). '
     '(B) Sensitivity vs specificity scatter, colour-coded by validation approach.'),
    ('SuppFig6_grading_models.png',
     'Supplementary Figure S6. All grading and scoring models compared. '
     'AUC-ROC with 95% CI for all 39 multiclass models across seven grading/scoring tasks. '
     'Layer 1 (recognition) in blue, Layer 2 (prognostic scoring) in orange. '
     'Tasks: TBIL-Eyelid, Face-Ternary, DBIL, IBIL, Jaundice-Type, Child-Pugh, MELD-Risk.'),
]

for fname, legend in supp_figs:
    add_heading(fname.replace('.png', '').replace('_', ' ').replace('SuppFig', 'Supplementary Figure'), level=2)
    add_figure(os.path.join(FIG_DIR, fname), width=6.5, caption=legend)
    doc.add_page_break()

# ═════════════════════════════════════════════════════════════
# PART IV: SUPPLEMENTARY TABLES
# ═════════════════════════════════════════════════════════════
add_heading('Part IV. Supplementary Tables', level=1)

add_heading('Supplementary Table S1. All 67 Models — Complete Metrics', level=2)
add_para('Supplementary Table S1. Complete performance metrics for all 67 models across 10 tasks. '
         'Six metrics (AUC, F1, Sensitivity, Specificity, Accuracy, Average Precision) with 95% bootstrap '
         'confidence intervals. Sorted by task then AUC descending.', italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'SuppTable_S1_all_67_models.csv'), font_size=6)
doc.add_page_break()

add_heading('Supplementary Table S2. 64 Clinical Guidelines', level=2)
add_para('Supplementary Table S2. Complete registry of 64 clinical guidelines integrated into the CDSS '
         'knowledge base, sorted by year (most recent first). Includes 12 Chinese domestic guidelines '
         '(CSH, CSID, CSS, NHC, CSGE) and 52 international guidelines (EASL, AASLD, APASL, ACG, Tokyo, '
         'King\'s College, KDIGO, OPTN, etc.).', italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'SuppTable_S2_64_guidelines.csv'), font_size=7)
doc.add_page_break()

add_heading('Supplementary Table S3. 17 Disease Profiles', level=2)
add_para('Supplementary Table S3. Disease profiles integrated into the CDSS inference engine, sorted by '
         'matching priority (highest first). Each disease carries a jaundice type, default department, '
         'and optional red-flag pathway.', italic=True, size=9)
add_table_from_csv(os.path.join(TBL_V3, 'SuppTable_S3_17_diseases.csv'), font_size=8)

doc.save(OUT)
print(f'Saved: {OUT}')
print(f'File size: {os.path.getsize(OUT) / 1024 / 1024:.1f} MB')
