# -*- coding: utf-8 -*-
"""
Generate CDSS_临床建议审阅.docx — a consolidated, review-ready knowledge base
for the BilinGuard Clinical Decision Support System.

Pulled directly from deployment/clinical_advisor.py (single source of truth), so the
document matches exactly what the desktop app renders.

Structure
---------
Part 0  封面 / 说明 / 免责
Part 1  知识库总览 (静态参考)
   1.1 7 个科室 Profile
   1.2 16 个病因 Profile
   1.3 3 类黄疸
   1.4 4 档严重程度
   1.5 6 条临床警示路径
Part 2  科室 × 情况 建议矩阵 (worked scenarios, fully rendered + inline guidelines)
Part 3  附录：全部指南引用清单
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

DEPLOY = r'D:\research\人脸识别营养\传染科\deployment'
sys.path.insert(0, DEPLOY)
import clinical_advisor as ca
from clinical_advisor import (
    ClinicalAdvisor, DEPARTMENTS, DISEASES, JAUNDICE_TYPES,
    SEVERITY_GUIDANCE, CLINICAL_PATHWAYS, GUIDELINES,
)

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r'D:\research\人脸识别营养\传染科\CDSS_临床建议审阅.docx'

# ── colors ──────────────────────────────────────────────────
NAVY = RGBColor(0x1D, 0x35, 0x57)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)
ACCENT = RGBColor(0xE7, 0x6F, 0x51)
DANGER = RGBColor(0xE7, 0x4C, 0x3C)
GRAY = RGBColor(0x6C, 0x75, 0x7D)
DARK = RGBColor(0x1D, 0x35, 0x57)
MOD = RGBColor(0xE6, 0x7E, 0x22)

doc = Document()

# ── base styles ─────────────────────────────────────────────
def _set_cn_font(run, size=None, bold=None, color=None):
    run.font.name = 'Microsoft YaHei'
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts'); rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    if size is not None: run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    if color is not None: run.font.color.rgb = color

normal = doc.styles['Normal']
normal.font.name = 'Microsoft YaHei'
normal.element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
normal.font.size = Pt(10.5)

for i in range(1, 4):
    st = doc.styles[f'Heading {i}']
    st.font.name = 'Microsoft YaHei'
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    st.font.color.rgb = NAVY
doc.styles['Heading 1'].font.size = Pt(18)
doc.styles['Heading 2'].font.size = Pt(14)
doc.styles['Heading 3'].font.size = Pt(12)


def para(text='', size=10.5, bold=False, color=None, align=None, space_after=4, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None: p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        _set_cn_font(r, size, bold, color)
    return p


def bullet(text, size=10.5, color=None, level=0):
    p = doc.add_paragraph(style='List Bullet' if level == 0 else 'List Bullet 2')
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    _set_cn_font(r, size, False, color)
    return p


def guideline_block(gids, gid_objs=None):
    """Render a '依据指南' block. Accepts list of gids (resolved) or pre-built objects."""
    if gid_objs:
        objs = gid_objs
    else:
        objs = []
        seen = set()
        for g in gids:
            if g in GUIDELINES and g not in seen:
                seen.add(g); objs.append({'gid': g, 'short': GUIDELINES[g][0],
                                          'citation': GUIDELINES[g][2], 'source': GUIDELINES[g][3],
                                          'year': GUIDELINES[g][4]})
    if not objs:
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.left_indent = Pt(12)
    r = p.add_run('📚 依据指南 / Evidence')
    _set_cn_font(r, 9.5, True, GRAY)
    for g in objs:
        gp = doc.add_paragraph()
        gp.paragraph_format.space_after = Pt(1)
        gp.paragraph_format.left_indent = Pt(24)
        r1 = gp.add_run(f'[{g["gid"]}] ')
        _set_cn_font(r1, 9, True, TEAL)
        r2 = gp.add_run(f'{g["citation"]} ')
        _set_cn_font(r2, 9, False, GRAY)
        r3 = gp.add_run(f'({g["source"]} {g["year"]})')
        _set_cn_font(r3, 8.5, False, GRAY)


def hrule():
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement('w:pBdr'); bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1'); bottom.set(qn('w:color'), 'CCCCCC')
    pbdr.append(bottom); pPr.append(pbdr)
    p.paragraph_format.space_after = Pt(4)


def shade_cell(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)


def set_cell_text(cell, text, bold=False, color=None, size=9.5, align_left=True):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    if not align_left: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, line in enumerate(str(text).split('\n')):
        if i > 0:
            p = cell.add_paragraph(); p.paragraph_format.space_after = Pt(1)
        r = p.add_run(line)
        _set_cn_font(r, size, bold, color)


# ════════════════════════════════════════════════════════════
# PART 0 — COVER
# ════════════════════════════════════════════════════════════
para('BilinGuard 临床决策支持（CDSS）', 26, True, NAVY, WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para('科室 × 病因 × 黄疸状态 —— 临床建议与指南依据审阅文档', 13, False, TEAL, WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
para('本文件由 deployment/clinical_advisor.py 知识库自动生成，内容与桌面应用输出完全一致，供临床与审阅使用。',
     10.5, False, GRAY, WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para('四川大学华西医院 · BilinGuard 项目   |   生成日期：2026-07-16', 10, False, GRAY,
     WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
hrule()
para('免责声明：本文档为 AI 辅助筛查提示，BilinGuard 不作出诊断，不能替代临床判断、化验确认或主治医师决策。'
     '所有结果须以血清胆红素及完整检查为准。', 9.5, False, DANGER, space_after=10)

# manual contents
para('目  录', 14, True, NAVY, space_after=4)
for line in [
    '第一部分  知识库总览（静态参考）',
    '   1.1  使用科室（17 个）   ·   1.2  病因（17 种）   ·   1.3  黄疸类型（3 类）',
    '   1.4  严重程度（4 档）   ·   1.5  临床警示路径（6 条）   ·   1.6  营养支持知识库（v3.1）',
    '第二部分  科室 × 情况  建议矩阵（worked scenarios，每条建议附指南引用 + 智能体自审溯源）',
    '   2.1 感染/肝病科  ·  2.2 消化内科  ·  2.3 肝胆外科  ·  2.4 ICU  ·  2.5 血液科',
    '   2.6 肿瘤科  ·  2.7 急诊科  ·  2.8 介入科  ·  2.9 肝移植科  ·  2.10 风湿免疫科',
    '   2.11 老年科  ·  2.12 口腔科  ·  2.13 眼科  ·  2.14 公卫/体检',
    '   2.15 产科  ·  2.16 肾内科  ·  2.17 其他科室',
    '第三部分  Child-Pugh 与 MELD 评分参考（方法学 + 临床应用 + 科室决策映射）',
    '第四部分  附录：全部指南引用清单（中国 + 国际）',
]:
    para(line, 10.5, False, DARK, space_after=1)
doc.add_page_break()


# ════════════════════════════════════════════════════════════
# PART 1 — KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════
doc.add_heading('第一部分  知识库总览', level=1)

# 1.1 Departments
doc.add_heading('1.1  使用科室（17 个）', level=2)
dept_rows = [('科室', '重点 Focus', '发现黄疸时处置', '转诊逻辑', '依据指南')]
for did, info in DEPARTMENTS.items():
    dept_rows.append((
        info['name_cn'],
        info['focus_cn'],
        '；'.join(info['on_jaundice_cn']),
        info['referral_cn'],
        '、'.join(GUIDELINES[g][0].split('(')[0].strip() for g in info.get('guidelines', []) if g in GUIDELINES) or '—',
    ))
t = doc.add_table(rows=len(dept_rows), cols=5)
t.style = 'Light Grid Accent 1'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
widths = [1.3, 1.7, 2.4, 1.8, 1.4]
for ri, row in enumerate(dept_rows):
    for ci, val in enumerate(row):
        c = t.rows[ri].cells[ci]
        set_cell_text(c, val, bold=(ri == 0), color=(NAVY if ri == 0 else DARK), size=9)
        if ri == 0: shade_cell(c, '1D3557'); 
        for p in c.paragraphs: 
            for run in p.runs:
                if ri == 0: _set_cn_font(run, 9, True, RGBColor(0xFF,0xFF,0xFF))
        c.width = Inches(widths[ci])
doc.add_paragraph()

# 1.2 Diseases
doc.add_heading('1.2  病因（17 种，按入院诊断关键词推断）', level=2)
dis_rows = [('病因', '匹配关键词', '黄疸类型', '默认科室', '处理原则', '依据指南')]
# stable display order by priority desc
order = sorted(DISEASES.keys(), key=lambda d: -DISEASES[d].get('priority', 0))
for did in order:
    d = DISEASES[did]
    jtn = JAUNDICE_TYPES.get(d.get('jaundice_type'), {}).get('cn_name', d.get('jaundice_type', ''))
    dept = DEPARTMENTS.get(d.get('department'), {}).get('name_cn', '')
    kws = '、'.join(d.get('keywords', []))
    gids = '、'.join(GUIDELINES[g][0].split('(')[0].strip() for g in d.get('guidelines', []) if g in GUIDELINES) or '—'
    dis_rows.append((d['name_cn'], kws, jtn, dept, d.get('principle_cn', ''), gids))
t = doc.add_table(rows=len(dis_rows), cols=6)
t.style = 'Light Grid Accent 1'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
widths = [1.3, 1.5, 1.0, 1.3, 2.3, 1.6]
for ri, row in enumerate(dis_rows):
    for ci, val in enumerate(row):
        c = t.rows[ri].cells[ci]; set_cell_text(c, val, bold=(ri == 0), color=(NAVY if ri == 0 else DARK), size=8.5)
        if ri == 0:
            for p in c.paragraphs:
                for run in p.runs: _set_cn_font(run, 9, True, RGBColor(0xFF,0xFF,0xFF))
        c.width = Inches(widths[ci])
doc.add_paragraph()

# 1.3 Jaundice types
doc.add_heading('1.3  黄疸类型（3 类）', level=2)
for jt, info in JAUNDICE_TYPES.items():
    doc.add_heading(info['cn_name'] + f'  ({info["en_name"]})', level=3)
    para(f'机制：{info["mechanism"]}', 10.5, space_after=2)
    para(f'胆红素模式：{info["bilirubin_pattern"]}', 10.5, space_after=2)
    para(f'化验模式：{info["lab_pattern"]}', 10.5, space_after=2)
    para('常见病因：', 10.5, True, space_after=1)
    for c in info['common_causes_cn']: bullet(c, 10)
    para('推荐检查：', 10.5, True, space_after=1)
    for c in info['recommended_workup_cn']: bullet(c, 10)
    from clinical_advisor import TYPE_GUIDELINES
    guideline_block([], gid_objs=[{'gid': g, 'short': GUIDELINES[g][0],
                                   'citation': GUIDELINES[g][2], 'source': GUIDELINES[g][3],
                                   'year': GUIDELINES[g][4]} for g in TYPE_GUIDELINES.get(jt, []) if g in GUIDELINES])
    hrule()

# 1.4 Severity
doc.add_heading('1.4  严重程度（4 档，TBIL μmol/L）', level=2)
sev_rows = [('级别', 'TBIL 范围', '处理 Management', '随访 Follow-up', '生活 Lifestyle', '依据指南')]
for s in [0, 1, 2, 3]:
    g = SEVERITY_GUIDANCE[s]['cn']
    gids = ['CHILD_PUGH', 'MELD_3_0'] + (['LF_2018', 'AASLD_ALF_2023'] if s >= 3 else (['CIRRHOSIS_2019'] if s >= 2 else []))
    sev_rows.append((g['label'], g['tbil'], g['management'], g['follow_up'], g['lifestyle'],
                     '、'.join(GUIDELINES[x][0].split('(')[0].strip() for x in gids if x in GUIDELINES)))
t = doc.add_table(rows=len(sev_rows), cols=6)
t.style = 'Light Grid Accent 1'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
widths = [1.3, 1.0, 2.0, 2.0, 1.6, 1.5]
for ri, row in enumerate(sev_rows):
    for ci, val in enumerate(row):
        c = t.rows[ri].cells[ci]; set_cell_text(c, val, bold=(ri == 0), color=(NAVY if ri == 0 else DARK), size=8.5)
        if ri == 0:
            for p in c.paragraphs:
                for run in p.runs: _set_cn_font(run, 9, True, RGBColor(0xFF,0xFF,0xFF))
        c.width = Inches(widths[ci])
doc.add_paragraph()

# 1.5 Pathways
doc.add_heading('1.5  临床警示路径（6 条 red flags）', level=2)
for pid, pw in CLINICAL_PATHWAYS.items():
    doc.add_heading(f'{pid}', level=3)
    para(f'触发条件：{pw["cn_trigger"]}', 10.5, color=DANGER, space_after=2)
    para('关键动作：', 10.5, True, space_after=1)
    for a in pw['cn_actions']: bullet(a, 10)
    guideline_block(pw.get('guidelines', []))
    hrule()

# 1.6 Nutrition support (v3.1)
doc.add_heading('1.6  营养支持知识库（v3.1，NRS-2002 × ESPEN）', level=2)
para('营养风险筛查采用 NRS-2002（Kondrup 2003）：≥3 分为营养风险，触发 ESPEN 肝病营养指南干预路径；'
     '<3 分住院期间每周复评；未筛查或分值越界（>7）时提示完成筛查。'
     '慢性肝病即使 BMI 正常也需警惕肌少症。', 10.5, space_after=4)
para('营养风险干预包（GRADE 1B）：', 10.5, True, space_after=1)
for a in ca.NUTRITION_GUIDANCE['at_risk']['cn']['actions']:
    bullet(a, 10)
para('严重度附加（重度/ACLF）：' + ca.NUTRITION_SEVERITY_ADDENDA[3]['cn'], 10, False, MOD, space_after=1)
para('严重度附加（中度）：' + ca.NUTRITION_SEVERITY_ADDENDA[2]['cn'], 10, False, MOD, space_after=1)
para('类型附加 —— 胆汁郁积/梗阻：' + ca.NUTRITION_TYPE_ADDENDA['cholestatic']['cn'], 10, False, MOD, space_after=1)
para('类型附加 —— 肝细胞型：' + ca.NUTRITION_TYPE_ADDENDA['hepatocellular']['cn'], 10, False, MOD, space_after=1)
para('类型附加 —— 溶血：' + ca.NUTRITION_TYPE_ADDENDA['hemolytic']['cn'], 10, False, MOD, space_after=1)
guideline_block(ca.NUTRITION_GUIDELINE_IDS)
hrule()

doc.add_page_break()


# ════════════════════════════════════════════════════════════
# PART 2 — DEPARTMENT × SITUATION SCENARIO MATRIX
# ════════════════════════════════════════════════════════════
doc.add_heading('第二部分  科室 × 情况  建议矩阵', level=1)
para('以下每个场景按「科室 + 入院诊断 + 黄疸类型 + 严重度（由化验驱动）」组合，完整渲染 ClinicalAdvisor 输出，'
     '每条建议均附指南引用。', 10, False, GRAY, space_after=8)

# scenarios: (dept_code, dept_label, [ (title, type_index, diagnosis, labs) , ... ])
TYPE_PRED = {0: ('Hepatocellular', '肝细胞性'), 1: ('Cholestatic', '胆汁郁积性')}
SCENARIOS = [
    ('infectious_hepatology', '2.1  感染科 / 肝病科', [
        ('慢加急性肝衰竭 + 慢性乙肝（重度）', 0, '慢加急性肝衰竭，慢性乙型病毒性肝炎',
         {'tbil': 280, 'dbil': 140, 'ibil': 140, 'alt': 420, 'ast': 380, 'alp': 140, 'ggt': 80, 'inr': 2.4, 'albumin': 30}, 4),
        ('肝炎后肝硬化失代偿（中度）', 0, '肝炎后肝硬化失代偿期',
         {'tbil': 130, 'dbil': 70, 'ibil': 60, 'alt': 90, 'ast': 110, 'alp': 160, 'ggt': 120, 'inr': 1.4, 'albumin': 31}, 3),
        ('慢性乙型病毒性肝炎（轻度）', 0, '慢性乙型病毒性肝炎',
         {'tbil': 45, 'dbil': 18, 'ibil': 27, 'alt': 120, 'ast': 80, 'alp': 110, 'ggt': 60, 'inr': 1.0, 'albumin': 42}, 1),
        ('药物性肝损伤 Hy 定律（中度）', 0, '肝功能异常原因待查:药物性？其他？',
         {'tbil': 110, 'dbil': 60, 'ibil': 50, 'alt': 480, 'ast': 360, 'alp': 140, 'ggt': 90, 'inr': 1.1, 'albumin': 39}),
        ('自身免疫性肝炎（中度）', 0, '自身免疫性肝病',
         {'tbil': 95, 'dbil': 50, 'ibil': 45, 'alt': 260, 'ast': 210, 'alp': 180, 'ggt': 150, 'inr': 1.2, 'albumin': 35}),
        ('酒精性肝病（中度）', 0, '酒精性肝病',
         {'tbil': 120, 'dbil': 60, 'ibil': 60, 'alt': 150, 'ast': 220, 'alp': 200, 'ggt': 380, 'inr': 1.3, 'albumin': 30}),
    ]),
    ('gi', '2.2  消化内科', [
        ('原发性胆汁性胆管炎 PBC（中度·胆汁郁积）', 1, '原发性胆汁性胆管炎',
         {'tbil': 90, 'dbil': 55, 'ibil': 35, 'alt': 80, 'ast': 70, 'alp': 420, 'ggt': 360, 'inr': 1.1, 'albumin': 38}, 2),
        ('原发性硬化性胆管炎 PSC（中度）', 1, '原发性硬化性胆管炎',
         {'tbil': 100, 'dbil': 60, 'ibil': 40, 'alt': 90, 'ast': 80, 'alp': 480, 'ggt': 300, 'inr': 1.1, 'albumin': 37}),
        ('肝硬化失代偿 + 静脉曲张风险（中度）', 0, '肝硬化失代偿期',
         {'tbil': 110, 'dbil': 55, 'ibil': 55, 'alt': 70, 'ast': 90, 'alp': 150, 'ggt': 100, 'inr': 1.5, 'albumin': 29}),
        ('胆汁淤积原因待查（中度）', 1, '胆汁淤积原因待查:基因性？其他？',
         {'tbil': 140, 'dbil': 95, 'ibil': 45, 'alt': 60, 'ast': 50, 'alp': 360, 'ggt': 280, 'inr': 1.1, 'albumin': 40}),
    ]),
    ('hpb_surgery', '2.3  肝胆外科 / 普外科（肝胆胰）', [
        ('胆总管结石（中度·胆汁郁积）', 1, '胆总管结石',
         {'tbil': 130, 'dbil': 95, 'ibil': 35, 'alt': 90, 'ast': 70, 'alp': 380, 'ggt': 320, 'inr': 1.1, 'albumin': 40}),
        ('急性胆管炎（重度·化脓）', 1, '胆总管结石伴胆管炎',
         {'tbil': 190, 'dbil': 140, 'ibil': 50, 'alt': 130, 'ast': 100, 'alp': 450, 'ggt': 350, 'inr': 1.6, 'albumin': 33}),
        ('胆管癌 恶性梗阻（重度）', 1, '胆管癌',
         {'tbil': 300, 'dbil': 230, 'ibil': 70, 'alt': 120, 'ast': 90, 'alp': 520, 'ggt': 410, 'inr': 1.3, 'albumin': 35}),
    ]),
    ('icu', '2.4  ICU / 重症医学科', [
        ('急性肝衰竭 ALF（重度·INR 2.0）', 0, '急性肝衰竭',
         {'tbil': 250, 'dbil': 130, 'ibil': 120, 'alt': 1200, 'ast': 900, 'alp': 160, 'ggt': 90, 'inr': 2.0, 'albumin': 32}),
        ('慢加急性肝衰竭 器官衰竭（重度）', 0, '慢加急性肝衰竭',
         {'tbil': 310, 'dbil': 160, 'ibil': 150, 'alt': 380, 'ast': 340, 'alp': 150, 'ggt': 80, 'inr': 2.6, 'albumin': 28}, 5),
        ('急性化脓性胆管炎 脓毒症（重度）', 1, '急性胆管炎',
         {'tbil': 220, 'dbil': 160, 'ibil': 60, 'alt': 140, 'ast': 110, 'alp': 470, 'ggt': 360, 'inr': 1.8, 'albumin': 30}),
    ]),
    ('hematology', '2.5  血液科', [
        ('自身免疫性溶血性贫血 AIHA（中度·溶血）', 0, '自身免疫性溶血性贫血',
         {'tbil': 95, 'dbil': 12, 'ibil': 83, 'alt': 25, 'ast': 20, 'alp': 90, 'ggt': 30, 'inr': 1.0, 'albumin': 44}, 1),
        ('G6PD 缺乏症 溶血（中度）', 0, 'G6PD缺乏症',
         {'tbil': 100, 'dbil': 15, 'ibil': 85, 'alt': 30, 'ast': 25, 'alp': 95, 'ggt': 35, 'inr': 1.0, 'albumin': 43}),
        ('遗传性球形红细胞增多症（轻度）', 0, '遗传性球形红细胞增多症',
         {'tbil': 55, 'dbil': 8, 'ibil': 47, 'alt': 22, 'ast': 18, 'alp': 88, 'ggt': 28, 'inr': 1.0, 'albumin': 45}),
    ]),
    ('oncology', '2.6  肿瘤科', [
        ('胰头癌 恶性梗阻（重度·胆汁郁积）', 1, '胰头癌',
         {'tbil': 320, 'dbil': 240, 'ibil': 80, 'alt': 120, 'ast': 90, 'alp': 520, 'ggt': 410, 'inr': 1.3, 'albumin': 35}, 3),
        ('胆管癌 肝门部（重度）', 1, '肝门部胆管癌',
         {'tbil': 280, 'dbil': 210, 'ibil': 70, 'alt': 110, 'ast': 85, 'alp': 480, 'ggt': 390, 'inr': 1.2, 'albumin': 36}),
        ('肝细胞癌 HCC（中度·混合）', 0, '肝右叶占位：肿瘤？',
         {'tbil': 90, 'dbil': 50, 'ibil': 40, 'alt': 80, 'ast': 100, 'alp': 300, 'ggt': 250, 'inr': 1.2, 'albumin': 34}),
    ]),
    ('emergency', '2.7  急诊科', [
        ('急性重度黄疸 病因待查（重度）', 0, '肝功能异常待查',
         {'tbil': 260, 'dbil': 130, 'ibil': 130, 'alt': 300, 'ast': 280, 'alp': 150, 'ggt': 90, 'inr': 1.9, 'albumin': 33}),
        ('胆总管结石伴胆管炎 急诊（重度）', 1, '胆总管结石伴胆管炎',
         {'tbil': 200, 'dbil': 150, 'ibil': 50, 'alt': 140, 'ast': 110, 'alp': 460, 'ggt': 350, 'inr': 1.5, 'albumin': 34}),
        ('溶血危象 急诊（中度·溶血）', 0, '溶血性贫血',
         {'tbil': 110, 'dbil': 14, 'ibil': 96, 'alt': 28, 'ast': 22, 'alp': 92, 'ggt': 32, 'inr': 1.1, 'albumin': 40}),
    ]),
    ('interventional', '2.8  介入科', [
        ('胆管癌 恶性梗阻 PTCD 减压（重度·胆汁郁积）', 1, '胆管癌',
         {'tbil': 310, 'dbil': 240, 'ibil': 70, 'alt': 120, 'ast': 90, 'alp': 520, 'ggt': 410, 'inr': 1.4, 'albumin': 35}),
        ('肝细胞癌 TACE 术前评估（中度·混合）', 0, '肝细胞癌',
         {'tbil': 80, 'dbil': 45, 'ibil': 35, 'alt': 70, 'ast': 90, 'alp': 280, 'ggt': 230, 'inr': 1.2, 'albumin': 34}),
    ]),
    ('liver_transplant', '2.9  肝移植科', [
        ('急性肝衰竭 移植评估（重度·INR 2.2）', 0, '急性肝衰竭',
         {'tbil': 240, 'dbil': 125, 'ibil': 115, 'alt': 1100, 'ast': 850, 'alp': 160, 'ggt': 90, 'inr': 2.2, 'albumin': 31}),
        ('终末期肝硬化失代偿 移植评估（重度）', 0, '肝硬化失代偿期',
         {'tbil': 180, 'dbil': 95, 'ibil': 85, 'alt': 60, 'ast': 80, 'alp': 150, 'ggt': 100, 'inr': 1.8, 'albumin': 27}),
    ]),
    ('rheumatology', '2.10  风湿免疫科', [
        ('自身免疫性肝炎 AIH（中度·肝细胞）', 0, '自身免疫性肝炎',
         {'tbil': 95, 'dbil': 50, 'ibil': 45, 'alt': 260, 'ast': 210, 'alp': 180, 'ggt': 150, 'inr': 1.2, 'albumin': 35}),
        ('原发性胆汁性胆管炎 PBC 重叠（中度·胆汁郁积）', 1, '原发性胆汁性胆管炎',
         {'tbil': 90, 'dbil': 55, 'ibil': 35, 'alt': 80, 'ast': 70, 'alp': 420, 'ggt': 360, 'inr': 1.1, 'albumin': 38}),
    ]),
    ('geriatrics', '2.11  老年科', [
        ('老年药物性肝损伤 多重用药（中度）', 0, '药物性肝损伤',
         {'tbil': 115, 'dbil': 62, 'ibil': 53, 'alt': 460, 'ast': 350, 'alp': 150, 'ggt': 95, 'inr': 1.2, 'albumin': 33}),
        ('老年肝硬化失代偿（中度）', 0, '肝硬化失代偿期',
         {'tbil': 120, 'dbil': 60, 'ibil': 60, 'alt': 65, 'ast': 85, 'alp': 155, 'ggt': 105, 'inr': 1.4, 'albumin': 30}),
    ]),
    ('stomatology', '2.12  口腔科（筛查前哨）', [
        ('口腔黏膜黄染 偶然发现（轻度）', 0, '肝功能异常待查',
         {'tbil': 40, 'dbil': 16, 'ibil': 24, 'alt': 35, 'ast': 30, 'alp': 110, 'ggt': 45, 'inr': 1.0, 'albumin': 42}),
    ]),
    ('ophthalmology', '2.13  眼科（筛查前哨）', [
        ('巩膜黄染 偶然发现（轻度）', 0, '慢性乙型病毒性肝炎',
         {'tbil': 48, 'dbil': 19, 'ibil': 29, 'alt': 110, 'ast': 75, 'alp': 105, 'ggt': 58, 'inr': 1.0, 'albumin': 41}),
    ]),
    ('public_health', '2.14  公卫 / 体检 / 预防保健（筛查前哨）', [
        ('体检发现皮肤巩膜黄染（轻度）', 0, '慢性乙型病毒性肝炎',
         {'tbil': 52, 'dbil': 20, 'ibil': 32, 'alt': 120, 'ast': 80, 'alp': 108, 'ggt': 60, 'inr': 1.0, 'albumin': 42}),
    ]),
    ('obstetrics', '2.15  产科 / 母胎医学（妊娠相关肝病·类型决定处理）', [
        ('妊娠期肝内胆汁淤积症 ICP（胆汁郁积型）', 1, '妊娠期肝内胆汁淤积症',
         {'tbil': 40, 'dbil': 25, 'ibil': 15, 'alt': 60, 'ast': 50, 'alp': 350, 'ggt': 200, 'inr': 1.0, 'albumin': 38}),
        ('HELLP综合征（肝细胞型·重度）', 0, 'HELLP综合征',
         {'tbil': 90, 'dbil': 50, 'ibil': 40, 'alt': 320, 'ast': 280, 'alp': 200, 'ggt': 90, 'inr': 1.6, 'albumin': 30}),
        ('妊娠期急性脂肪肝 AFLP（肝细胞型·重度）', 0, '妊娠期急性脂肪肝',
         {'tbil': 130, 'dbil': 70, 'ibil': 60, 'alt': 250, 'ast': 220, 'alp': 180, 'ggt': 80, 'inr': 1.8, 'albumin': 28}),
    ]),
    ('nephrology', '2.16  肾内科（黄疸+肾损伤·类型决定处理）', [
        ('肝肾综合征 HRS-AKI（肝细胞型·中度）', 0, '肝硬化失代偿期',
         {'tbil': 120, 'dbil': 60, 'ibil': 60, 'alt': 60, 'ast': 80, 'alp': 150, 'ggt': 100, 'inr': 1.4, 'albumin': 28}),
        ('溶血尿毒综合征 HUS/TMA（溶血型·中度）', 0, '溶血性贫血',
         {'tbil': 80, 'dbil': 10, 'ibil': 70, 'alt': 25, 'ast': 20, 'alp': 90, 'ggt': 30, 'inr': 1.0, 'albumin': 38}),
    ]),
    ('other', '2.17  其他科室（少见黄疸科室统称）', [
        ('偶见黄疸 转诊查因（轻度）', 0, '肝功能异常待查',
         {'tbil': 55, 'dbil': 22, 'ibil': 33, 'alt': 45, 'ast': 38, 'alp': 120, 'ggt': 50, 'inr': 1.0, 'albumin': 43}),
        ('本科用药相关 药物性肝损伤（中度）', 0, '药物性肝损伤',
         {'tbil': 105, 'dbil': 55, 'ibil': 50, 'alt': 420, 'ast': 320, 'alp': 145, 'ggt': 90, 'inr': 1.1, 'albumin': 38}),
    ]),
]

adv = ClinicalAdvisor(lang='cn')


def render_advice_to_doc(advice):
    """Render one advisor output fully into the doc."""
    sev = advice.get('severity', 0)
    sev_cols = [TEAL, RGBColor(0xF4, 0xC4, 0x30), MOD, DANGER]
    accent = sev_cols[min(sev, 3)]
    # summary banner
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run('⚑ 分诊小结：')
    _set_cn_font(r, 11, True, accent)
    r2 = p.add_run(advice.get('summary', ''))
    _set_cn_font(r2, 11, True, DARK)
    meta = (f"严重度={sev}  |  黄疸类型={advice.get('jaundice_type') or '—'}  |  "
            f"主病因={advice.get('disease') or '—'}  |  生效科室={advice.get('department') or '—'}")
    para(meta, 9, False, GRAY, space_after=4)
    for sec in advice.get('sections', []):
        icon = sec.get('icon', '')
        col = DANGER if '🚨' in icon else (ACCENT if '⚠' in icon else TEAL)
        ph = doc.add_paragraph()
        ph.paragraph_format.space_after = Pt(1)
        r = ph.add_run(f'{icon}  {sec.get("title", "")}')
        _set_cn_font(r, 10.5, True, col)
        content = sec.get('content', '')
        if isinstance(content, str):
            para(content, 10, space_after=2)
        elif isinstance(content, list):
            for it in content: bullet(str(it), 10)
        elif isinstance(content, dict):
            labels = {'mechanism': '机制', 'pattern': '胆红素模式', 'lab_pattern': '化验模式',
                      'department': '科室', 'principle': '处理原则', 'focus': '重点',
                      'specialty': '专科', 'referral': '转诊', 'causes': '常见病因',
                      'workup': '推荐检查', 'on_jaundice': '发现黄疸时', 'jaundice_type': '黄疸类型',
                      'suggested_department': '建议科室', 'label': '级别', 'tbil': '胆红素范围',
                      'management': '处理', 'follow_up': '随访', 'lifestyle': '生活',
                      'type_decision': '按黄疸类型的临床决策',
                      'child_pugh': 'Child-Pugh', 'meld': 'MELD', 'implication': '临床含义'}
            for k, v in content.items():
                lab = labels.get(k, k)
                if isinstance(v, list):
                    pp = doc.add_paragraph(); pp.paragraph_format.space_after = Pt(1)
                    rr = pp.add_run(f'{lab}：'); _set_cn_font(rr, 10, True, DARK)
                    for it in v: bullet(str(it), 9.5)
                else:
                    pp = doc.add_paragraph(); pp.paragraph_format.space_after = Pt(1)
                    rr = pp.add_run(f'{lab}：'); _set_cn_font(rr, 10, True, DARK)
                    rr2 = pp.add_run(str(v)); _set_cn_font(rr2, 10, False, DARK)
        guideline_block([], gid_objs=sec.get('guidelines') or [])
    hrule()


for dept_code, dept_label, scens in SCENARIOS:
    doc.add_heading(dept_label, level=2)
    info = DEPARTMENTS.get(dept_code, {})
    if info:
        para(f'科室定位：{info["focus_cn"]}', 10, False, GRAY, space_after=6)
    for i, sc in enumerate(scens, 1):
        title, type_idx, dx, labs = sc[0], sc[1], sc[2], sc[3]
        nrs = sc[4] if len(sc) > 4 else None
        doc.add_heading(f'场景 {i}：{title}', level=3)
        para(f'入院诊断：{dx}', 9.5, False, DARK, space_after=1)
        lab_str = '，'.join(f'{k}={v}' for k, v in labs.items())
        nrs_str = f'    NRS-2002：{nrs}' if nrs is not None else ''
        para(f'BilinGuard 黄疸类型：{TYPE_PRED[type_idx][1]}    化验：{lab_str}{nrs_str}',
             9.5, False, GRAY, space_after=4)
        results = {'screen': {'is_jaundice': True},
                   'type': {'type_classification': {'type_index': type_idx, 'prediction': TYPE_PRED[type_idx][0]}}}
        # Inject Child-Pugh / MELD grading for chronic-liver-disease scenarios so the
        # advisor emits the CP/MELD risk-stratification section in this review document.
        _dx = dx or ''
        if any(k in _dx for k in ['肝硬化', '失代偿', '肝衰竭', '肝癌', '肝细胞癌', '肝占位',
                                   'cirrhosis', 'failure', 'hcc', '移植']):
            _tbil = labs.get('tbil', 0) or 0
            _inr = labs.get('inr', 1.0) or 1.0
            _cp = 2 if (_inr >= 1.5 or _tbil > 171) else (1 if _tbil >= 85 else 0)
            _meld = 2 if (_tbil > 171 and _inr >= 1.5) else (1 if _tbil >= 85 else 0)
            results['cp'] = {'grading': {'prediction': ['Child-Pugh A', 'Child-Pugh B', 'Child-Pugh C'][_cp],
                                         'grade_index': _cp, 'probabilities': {'A': 0.2, 'B': 0.3, 'C': 0.5}}}
            results['meld'] = {'grading': {'prediction': ['Low Risk', 'Medium Risk', 'High Risk'][_meld],
                                           'grade_index': _meld, 'probabilities': {'Low': 0.3, 'Medium': 0.4, 'High': 0.3}}}
        nutrition = {'nrs2002': float(nrs)} if nrs is not None else None
        advice = adv.advise(results, labs, department=dept_code, diagnosis=dx,
                            nutrition=nutrition)
        render_advice_to_doc(advice)
        # v3 agent self-audit trace (rule-level provenance for reviewers)
        ag = advice.get('agent') or {}
        sc_agent = ag.get('self_check') or {}
        if sc_agent:
            para(f'智能体自审（{ag.get("pipeline_version", "")}）：模式 {ag.get("mode")}'
                 f' · 警示优先级 {sc_agent.get("alert_priority")}'
                 f' · 数据置信度 {sc_agent.get("confidence")}'
                 f' · 触发规则 {sc_agent.get("alerts_total")} 条',
                 9.5, True, TEAL, space_after=2)
            for r in ag.get('triggered_rules', []):
                para(f"[{r['grade']}] {r['rule_id']}（{r['domain']}，{r['urgency']}）：{r['trigger']}",
                     9, False, GRAY, space_after=1)
    doc.add_page_break()


# ════════════════════════════════════════════════════════════
# PART 3 — CHILD-PUGH & MELD SCORING REFERENCE
# ════════════════════════════════════════════════════════════
def _cp_table(headers, rows, widths, fill='1D3557'):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers)); t.style = 'Light Grid Accent 1'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ci, h in enumerate(headers):
        c = t.rows[0].cells[ci]; set_cell_text(c, h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), size=9, align_left=False); shade_cell(c, fill)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = t.rows[ri + 1].cells[ci]; set_cell_text(c, val, size=9, align_left=(ci != 0))
    return t


doc.add_heading('第三部分  Child-Pugh 与 MELD 评分参考', level=1)
para('BilinGuard 的 Child-Pugh / MELD 由面部图像深度学习模型预测（inference.py: classify_child_pugh / classify_meld），'
     '用于辅助肝硬化 / 肝衰竭 / 肝移植风险分层。本部分汇总评分方法学、临床应用与指南依据。', 10, False, GRAY, space_after=8)

doc.add_heading('3.1  Child-Pugh（CTP）评分', level=2)
para('5 项参数（胆红素、白蛋白、INR/PT、腹水、肝性脑病）计 1/2/3 分，合计 5–15 分。', 10.5, space_after=4)
_cp_table(['参数', '1 分', '2 分', '3 分'],
          [['胆红素 (μmol/L)', '< 34', '34–51', '> 51'],
           ['白蛋白 (g/L)', '> 35', '28–35', '< 28'],
           ['PT 延长 (s) / INR', '< 4 / <1.7', '4–6 / 1.7–2.3', '> 6 / >2.3'],
           ['腹水', '无', '少量（可控）', '中–大量/顽固'],
           ['肝性脑病', '无', 'I–II 级', 'III–IV 级']],
          [2.1, 1.6, 1.7, 1.7])
para('')
_cp_table(['分级', '分数', '1 年生存', '2 年生存', '临床含义'],
          [['A', '5–6', '~95–100%', '~85%', '代偿期；可耐受肝切除/TACE'],
           ['B', '7–9', '~80%', '~60%', '失代偿；手术风险升高；TACE 需谨慎'],
           ['C', '10–15', '~45%', '~35%', '严重失代偿；禁忌大手术；考虑移植']],
          [0.7, 0.9, 1.4, 1.4, 2.5])
para('')
para('注：胆汁淤积性肝病（PBC）胆红素阈值不同（1分<68、2分68–170、3分>170 μmol/L）。Child-Pugh 含主观项（腹水/脑病），'
     'ALBI 分级（3.3）作为客观替代用于 HCC。', 9, False, GRAY)

doc.add_heading('3.2  MELD / MELD-Na / MELD 3.0', level=2)
para('MELD = 3.78×ln(胆红素) + 11.2×ln(INR) + 9.57×ln(肌酐) + 6.43；范围 6–40，预测 3 个月死亡，全球肝移植分配核心。', 10.5, space_after=3)
para('MELD-Na 纳入血钠（Biggins 2006）；MELD 3.0（Kim 2021，OPTN 自 2023 采用）新增白蛋白与性别。', 10.5, space_after=5)
_cp_table(['MELD 区间', '3 个月死亡风险', '临床决策含义'],
          [['≤ 9', '< 2%', '低风险；通常不符移植优先'],
           ['10–19', '~6%', '中等；评估移植指征'],
           ['20–29', '~20%', '高；优先移植评估'],
           ['30–39', '~53%', '很高；紧急移植'],
           ['≥ 40', '> 71%', '极高；1 年生存差']],
          [1.2, 1.6, 3.2])
para('')
para('BilinGuard MELD 模型输出三分类：Low (≤20) / Medium (21–30) / High (>30)，对应 3 个月死亡 <6% / ~20% / >50%。', 9, False, GRAY)

doc.add_heading('3.3  ALBI 分级（Child-Pugh 客观替代，用于 HCC）', level=2)
para('仅含白蛋白与胆红素，无主观项；HCC 肝储备评估一致性高于 Child-Pugh（Johnson 2015）。', 10.5, space_after=4)
_cp_table(['ALBI', '界值 (log-odds)', '适用'],
          [['1', '< −2.60', '肝储备良好（多数 CP-A）'],
           ['2', '−2.60 ~ −1.39', '中度（CP-A/B 之间）'],
           ['3', '≥ −1.39', '差（对应 CP-C）；预后最差']],
          [1.0, 1.9, 3.1])

doc.add_heading('3.4  Child-Pugh / MELD 临床应用', level=2)
for _t in [
    '肝移植分配：MELD/MELD-Na/MELD 3.0 是欧美（OPTN/UNOS）与中国（CLTR）供肝分配核心；分越高优先级越高。',
    'HCC 治疗决策：BCLC/CNLC 以 Child-Pugh（或 ALBI）评估肝储备——CP-A 可切除/消融；CP-B/C 倾向姑息或移植。',
    'TACE 安全性：通常要求 Child-Pugh ≤ B7；CP-C 禁忌。',
    '外科/肝切除风险：CP-A 可耐受；CP-B 风险升高；CP-C 禁忌大手术。',
    'ACLF / 失代偿期肝硬化：Child-Pugh + MELD 共同用于预后与 ICU/移植分层。',
    '门脉高压出血：Child-Pugh 分层用于曲张静脉出血防治强度。',
]:
    bullet(_t, 10)

doc.add_heading('3.5  BilinGuard CP/MELD 输出 → 各科室决策', level=2)
_cp_table(['使用科室', 'Child-Pugh 输出决策', 'MELD 输出决策'],
          [('感染科/肝病科', 'CP-C→失代偿/移植评估；CP-B→病因治疗+监测', 'MELD>15→移植评估；>25→紧急'),
           ('肝胆外科', 'CP-A 可切除；CP-C 禁忌手术', 'MELD 高→围术期死亡风险高'),
           ('肿瘤科', 'CP-A 可根治；ALBI 补充评估', 'MELD>20→系统治疗风险高'),
           ('ICU', 'CP-C + 器官衰竭 → ACLF', 'MELD>30→紧急移植/高死亡'),
           ('肝移植科', 'CP-C 评估移植', 'MELD 3.0 = 分配优先级核心'),
           ('介入科', 'CP≤B7 可 TACE；CP-C 禁忌', 'MELD 高→介入后肝衰风险')],
          [1.4, 2.9, 2.7])
doc.add_page_break()


# ════════════════════════════════════════════════════════════
# PART 4 — APPENDIX: full guideline registry
# ════════════════════════════════════════════════════════════
doc.add_heading('第四部分  附录：全部指南引用清单', level=1)

doc.add_heading('中国指南', level=2)
cn_ids = ['CHB_2022', 'CHC_2022', 'LF_2018', 'CIRRHOSIS_2019', 'DILI_2023', 'ALD_2018',
          'AIH_2021', 'PBC_2021', 'PSC_2023', 'ACUTE_BILIARY_2021', 'HCC_2024', 'VARICEAL_2023',
          'ICP_CHINA_2024', 'CHINA_HDP_2020',
          'CHINA_LIVER_TRANSPLANT_2018', 'CHINA_CLTR_MELD', 'CHINA_PORTAL_HYPERTENSION_2023']
gl_rows = [('编号 gid', '简称', '完整引用', '机构', '年份')]
for gid in cn_ids:
    if gid in GUIDELINES:
        g = GUIDELINES[gid]; gl_rows.append((gid, g[0], g[2], g[3], str(g[4])))
t = doc.add_table(rows=len(gl_rows), cols=5); t.style = 'Light Grid Accent 1'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, row in enumerate(gl_rows):
    for ci, val in enumerate(row):
        c = t.rows[ri].cells[ci]; set_cell_text(c, val, bold=(ri == 0), color=(NAVY if ri == 0 else DARK), size=8.5)
        if ri == 0:
            for p in c.paragraphs:
                for run in p.runs: _set_cn_font(run, 9, True, RGBColor(0xFF,0xFF,0xFF))

doc.add_heading('国际指南 International', level=2)
intl_ids = [g for g in GUIDELINES.keys() if g not in cn_ids]
gl_rows = [('gid', 'short title', 'full citation', 'body', 'year')]
for gid in intl_ids:
    g = GUIDELINES[gid]; gl_rows.append((gid, g[0], g[1], g[3], str(g[4])))
t = doc.add_table(rows=len(gl_rows), cols=5); t.style = 'Light Grid Accent 1'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, row in enumerate(gl_rows):
    for ci, val in enumerate(row):
        c = t.rows[ri].cells[ci]; set_cell_text(c, val, bold=(ri == 0), color=(NAVY if ri == 0 else DARK), size=8.5)
        if ri == 0:
            for p in c.paragraphs:
                for run in p.runs: _set_cn_font(run, 9, True, RGBColor(0xFF,0xFF,0xFF))

para('')
hrule()
para('— 文档结束。本文件由 deployment/clinical_advisor.py 自动生成；如需修改建议内容，请编辑该知识库后重新运行生成脚本。', 9, False, GRAY, WD_ALIGN_PARAGRAPH.CENTER)

def _save(doc, path):
    try:
        doc.save(path); return path
    except PermissionError:
        import time
        alt = path.replace('.docx', f'_v{time.strftime("%m%d")}.docx')
        doc.save(alt); return alt

saved = _save(doc, OUT)
print('Saved:', saved)
print('Size: %.1f KB' % (os.path.getsize(saved) / 1024))
