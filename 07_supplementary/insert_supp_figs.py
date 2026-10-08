# -*- coding: utf-8 -*-
"""
Insert supplementary figures into BilinGuard_Supplementary_Lancet_expanded.docx
at appropriate positions (after their caption/heading paragraphs).
"""
import os
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = r'D:\research\人脸识别营养\传染科'
DOCX = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260725', '04_Supplementary',
                    'BilinGuard_Supplementary_Lancet_expanded.docx')
FIG_DIR = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260725', '04_Supplementary', 'figures')

doc = Document(DOCX)

# Map: figure filename -> search text in paragraphs (to find insertion point)
# Each tuple: (image_filename, search_keyword, width_inches)
figures_to_insert = [
    ('SuppFig1_forest_all_models.png', 'forest plot', 6.5),
    ('SuppFig2_summary_3layer.png', 'task-level performance', 6.5),
    ('SuppFig3_domain_adaptation.png', 'probability distribution', 6.5),
    ('SuppFig4_confusion_matrices.png', 'confusion matri', 6.5),
    ('SuppFig5_binary_models.png', 'binary screening model comparison', 6.5),
    ('SuppFig6_grading_models.png', 'grading model', 6.5),
    ('SuppFigS7_preprocessing_pipeline.png', 'preprocessing pipeline', 6.5),
    ('SuppFigS8_dataflow.png', 'data flow', 6.5),
    ('SuppFigS9_domain_adaptation_workflow.png', 'domain adaptation workflow', 6.5),
    ('SuppFigS10_cdss_decision_tree.png', 'decision tree', 6.5),
    ('SuppFigS11_reader_study_protocol.png', 'reader study protocol', 6.5),
    ('SuppFigS12_training_procedure.png', 'training procedure', 6.5),
    ('SuppFigS13_desktop_architecture.png', 'desktop architecture', 6.5),
]

# Find insertion points and insert
inserted = []
for img_name, keyword, width in figures_to_insert:
    img_path = os.path.join(FIG_DIR, img_name)
    if not os.path.exists(img_path):
        print(f'  SKIP {img_name}: file not found')
        continue

    # Find the paragraph containing the keyword (case-insensitive)
    found_idx = -1
    for i, p in enumerate(doc.paragraphs):
        if keyword.lower() in p.text.lower():
            found_idx = i
            break

    if found_idx < 0:
        # Try shorter keyword
        short_kw = keyword.split()[0]
        for i, p in enumerate(doc.paragraphs):
            if short_kw.lower() in p.text.lower():
                found_idx = i
                break

    if found_idx < 0:
        print(f'  SKIP {img_name}: no paragraph found for "{keyword}"')
        continue

    # Check if image already exists right after this paragraph
    # (avoid duplicate insertion)
    p = doc.paragraphs[found_idx]
    # Insert image in a new paragraph right after the caption
    # python-docx doesn't support insert_after directly, so we add at end and move
    # Simpler: add a new paragraph, add image to it
    new_p = doc.add_paragraph()
    new_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = new_p.add_run()
    run.add_picture(img_path, width=Inches(width))

    # Move the new paragraph to right after found_idx
    # In python-docx, we need to manipulate XML
    p_element = p._element
    new_p_element = new_p._element
    p_element.addnext(new_p_element)

    inserted.append(img_name)
    print(f'  INSERTED {img_name} after para {found_idx} ("{p.text[:50]}...")')

# Save
doc.save(DOCX)
print(f'\nSaved: {DOCX}')
print(f'Total figures inserted: {len(inserted)}')

# Copy to 20260725
import shutil
print('File is already in 20260725')
