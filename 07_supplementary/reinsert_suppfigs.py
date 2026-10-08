# -*- coding: utf-8 -*-
"""Re-insert regenerated S7-S13 into the supplementary docx (replace old versions)."""
import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

DOCX = r'D:\research\人脸识别营养\传染科\results\BilinGuard_Lancet_v2_20260725\04_Supplementary\BilinGuard_Supplementary_Lancet_expanded.docx'
FIG_DIR = r'D:\research\人脸识别营养\传染科\results\BilinGuard_Lancet_v2_20260725\04_Supplementary\figures'

doc = Document(DOCX)

# Find all figure caption paragraphs and their indices
fig_captions = {}
for i, p in enumerate(doc.paragraphs):
    t = p.text.strip()
    if t.startswith('Supplementary Figure S'):
        # Extract figure number
        for snum in ['S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13']:
            if f'Figure {snum}' in t or f'Figure {snum}.' in t:
                fig_captions[snum] = i
                break

print(f'Found figure captions: {sorted(fig_captions.keys())}')

# Map figure numbers to filenames
fig_files = {
    'S7': 'SuppFigS7_preprocessing_pipeline.png',
    'S8': 'SuppFigS8_dataflow.png',
    'S9': 'SuppFigS9_domain_adaptation_workflow.png',
    'S10': 'SuppFigS10_cdss_decision_tree.png',
    'S11': 'SuppFigS11_reader_study_protocol.png',
    'S12': 'SuppFigS12_training_procedure.png',
    'S13': 'SuppFigS13_desktop_architecture.png',
}

# Insert images after their captions (in reverse order to preserve indices)
inserted = []
for snum in sorted(fig_captions.keys(), reverse=True):
    if snum not in fig_files:
        continue
    img_path = os.path.join(FIG_DIR, fig_files[snum])
    if not os.path.exists(img_path):
        print(f'  SKIP {snum}: {fig_files[snum]} not found')
        continue
    
    cap_idx = fig_captions[snum]
    cap_para = doc.paragraphs[cap_idx]
    
    # Create new paragraph with image
    new_p = doc.add_paragraph()
    new_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = new_p.add_run()
    run.add_picture(img_path, width=Inches(6.0))
    
    # Move to right after caption
    cap_para._element.addnext(new_p._element)
    inserted.append(snum)
    print(f'  INSERTED {snum} ({fig_files[snum]}) after para {cap_idx}')

doc.save(DOCX)
print(f'\nSaved: {len(inserted)} figures inserted')

# Verify image count
img_count = sum(1 for rel in doc.part.rels.values() if 'image' in rel.reltype)
print(f'Total images in document: {img_count}')

import os
fsize = os.path.getsize(DOCX) / 1024 / 1024
print(f'File size: {fsize:.1f} MB')
