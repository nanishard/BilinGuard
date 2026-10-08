# -*- coding: utf-8 -*-
"""
Rebuild Table 1-5 as a single DOCX in Lancet three-line-table style
(top rule 1.5pt / header rule 0.75pt / bottom rule 1.5pt, no vertical rules),
with middle-dot decimals (0.956 -> 0·956) and en-dash CI separators.
Source CSVs: results/BilinGuard_Lancet_v2_20260725/03_Tables/*.csv
"""
import os, sys, re
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.stdout.reconfigure(encoding='utf-8')

BASE = r'D:\research\人脸识别营养\传染科'
TBL = os.path.join(BASE, 'results', 'BilinGuard_Lancet_v2_20260725', '03_Tables')
OUT = os.path.join(BASE, '20260911_CRM图表重设计', 'figures')
os.makedirs(OUT, exist_ok=True)

DARK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x55, 0x55, 0x55)


def mid_dot(s):
    """0.956 -> 0·956 ; hyphen between decimals inside CI -> en dash."""
    def _d(m): return m.group(1) + '·' + m.group(2)
    s = re.sub(r'(\d)\.(\d{3})', _d, str(s))
    s = re.sub(r'(?<=·\d{3})-(?=0·)', '–', s)
    return s


def set_border(cell, edge, sz=8, color='1A1A1A'):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn('w:tcBorders'))
    if borders is None:
        borders = OxmlElement('w:tcBorders'); tc_pr.append(borders)
    el = borders.find(qn(f'w:{edge}'))
    if el is None:
        el = OxmlElement(f'w:{edge}'); borders.append(el)
    el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), str(sz))
    el.set(qn('w:space'), '0'); el.set(qn('w:color'), color)


def cell_text(cell, txt, size=8, bold=False, italic=False, align='left', color=DARK):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = {'left': WD_ALIGN_PARAGRAPH.LEFT, 'center': WD_ALIGN_PARAGRAPH.CENTER,
                   'right': WD_ALIGN_PARAGRAPH.RIGHT}[align]
    p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
    r = p.add_run(str(txt))
    r.font.name = 'Arial'; r.font.size = Pt(size); r.bold = bold; r.italic = italic
    r.font.color.rgb = color
    r.element.rPr.rFonts.set(qn('w:eastAsia'), 'Arial')


def three_line_table(doc, df: pd.DataFrame, first_col_wide=True, fs_head=8, fs_body=7.5,
                     block_col=None, bold_rows=()):
    n_cols = len(df.columns)
    t = doc.add_table(rows=0, cols=n_cols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # header
    hr = t.add_row()
    for j, col in enumerate(df.columns):
        cell_text(hr.cells[j], col, size=fs_head, bold=True,
                  align='left' if j == 0 else 'center')
        set_border(hr.cells[j], 'top', sz=16)
        set_border(hr.cells[j], 'bottom', sz=8)
    # body
    prev_block = None
    for i, (_, row) in enumerate(df.iterrows()):
        tr = t.add_row()
        vals = list(row.values)
        new_block = block_col is not None and vals[block_col] != prev_block
        if block_col is not None and vals[block_col] == prev_block:
            vals[block_col] = ''
        if block_col is not None:
            prev_block = row.values[block_col]
        if block_col is not None and new_block and i > 0:
            for c in tr.cells: set_border(c, 'top', sz=4, color='999999')
        for j, v in enumerate(vals):
            bold = i in bold_rows
            cell_text(tr.cells[j], mid_dot(v), size=fs_body, bold=bold,
                      align='left' if j == 0 else 'center',
                      italic=(j == 0 and block_col is None and bold))
    last = t.rows[-1]
    for c in last.cells: set_border(c, 'bottom', sz=16)
    return t


def add_title(doc, label, caption):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run(f'{label}.  '); r1.bold = True
    r2 = p.add_run(caption); r2.bold = False
    for r in (r1, r2):
        r.font.name = 'Arial'; r.font.size = Pt(9); r.font.color.rgb = DARK
        r.element.rPr.rFonts.set(qn('w:eastAsia'), 'Arial')


def add_note(doc, txt):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(txt); r.italic = True
    r.font.name = 'Arial'; r.font.size = Pt(7); r.font.color.rgb = GREY
    r.element.rPr.rFonts.set(qn('w:eastAsia'), 'Arial')


doc = Document()
sec = doc.sections[0]
sec.left_margin, sec.right_margin = Cm(1.8), Cm(1.8)

# ── Table 1 ──────────────────────────────────────────────────────────────────
df1 = pd.read_csv(os.path.join(TBL, 'Table1_baseline.csv'))
df1.columns = [c.replace('n=625', 'n=625').replace('(', '\n(') for c in df1.columns]
add_title(doc, 'Table 1', 'Baseline characteristics of the internal and exploratory external cohorts')
three_line_table(doc, df1)
add_note(doc, 'Data are mean (SD) or n (%). BMI=body-mass index; TBIL=total bilirubin. '
              'Percentages are calculated within each cohort.')

# ── Table 2 ──────────────────────────────────────────────────────────────────
df2 = pd.read_csv(os.path.join(TBL, 'Table2_layer1_trimmed.csv'))
bold2 = [i for i, r in df2.iterrows() if 'External' in str(r['Task'])]
add_title(doc, 'Table 2', 'Jaundice-recognition performance under the patient-level analysis protocol')
three_line_table(doc, df2, block_col=0, bold_rows=bold2)
add_note(doc, 'Values are point estimates (95% CI) from 1,000 patient-level bootstrap iterations. '
              'AP=average precision. External rows report the exploratory frozen-threshold cohort.')

# ── Table 3 ──────────────────────────────────────────────────────────────────
df3 = pd.read_csv(os.path.join(TBL, 'Table3_layer2_hepatic_reserve.csv'))
df3.columns = ['Task', 'Scope', 'Model', 'n', 'AUC', 'AP', 'Acc', 'F1']
add_title(doc, 'Table 3', 'Exploratory classification of study-defined hepatic-reserve categories')
three_line_table(doc, df3, block_col=0)
add_note(doc, '*MELD fusion estimate is exploratory (n=9). Categories are study-defined and the '
              'control group was assigned the lowest tier by construction; TBIL enters the MELD '
              'formula, so TBIL-related comparators are part of the definition rather than '
              'independent validation.')

# ── Table 4 ──────────────────────────────────────────────────────────────────
df4 = pd.read_csv(os.path.join(TBL, 'Table4_reader_study.csv'))
add_title(doc, 'Table 4', 'BilinGuard compared with eight human readers')
bold4 = [i for i, r in df4.iterrows() if 'BilinGuard' in str(r['Rater'])]
three_line_table(doc, df4, bold_rows=bold4)
add_note(doc, 'κ=Cohen’s kappa against the reference standard (single-rater tasks). '
              'Reader ordering follows clinical role; the BilinGuard row is highlighted in bold.')

# ── Table 5 ──────────────────────────────────────────────────────────────────
df5 = pd.read_csv(os.path.join(TBL, 'Table5_cdss_validation.csv'))
add_title(doc, 'Table 5', 'Automated technical validation of the guideline-referenced CDSS '
                          'in 150 prespecified synthetic scenarios')
three_line_table(doc, df5)
add_note(doc, 'Synthetic technical tests verify rule execution and safety handling; they do not '
              'constitute patient-level clinical validation.')

out = os.path.join(OUT, 'Tables_1-5_redesign.docx')
doc.save(out)
print('saved', out)
