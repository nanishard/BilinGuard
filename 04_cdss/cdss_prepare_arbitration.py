# -*- coding: utf-8 -*-
"""
Prepare the THIRD-REVIEWER (arbitration) workbook with a CALIBRATED,
behaviourally-anchored rubric.

Root cause from round 1: reviewer A scored 'verifiability against visible
data' (1-4 anchor), reviewer B scored 'directional appropriateness' (3-5
anchor) -> systematic offset 1.03, kappa 0.01-0.18, 26/30 cases >=2 apart on
some domain. Fix: anchored 5-point definitions per domain + explicit
calibration instruction + worked examples.

Reviewer C scores ALL 30 cases blindly (seed 303, new order).
Coordinator-only artifact: _discrepant_cases.csv (26 cases, never shown to C).

Run: python code/cdss_prepare_arbitration.py
"""
import sys, os
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
SRC = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables',
                   'CDSS_CLINICIAN_REVIEW_SAMPLE.xlsx')
OUT_DIR = os.path.join(BASE, 'review_package')
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 303

NAVY = '1D3557'
thin = Side(style='thin', color='CCCCCC')
border = Border(left=thin, right=thin, top=thin, bottom=thin)

RUBRIC = [
    ('BilinGuard CDSS 临床建议盲评 —— 复评（校准版）', 15, True),
    ('（第三位审阅者 · 独立盲评 · 预计 45-60 分钟）', 11, False),
    ('', 10, False),
    ('【校准要点——请先读】', 12, True),
    ('评分对象是"输出的临床方向适当性"，而不是"以当前可见资料能否完全核验"。', 11, True),
    ('证据不足时：输出已恰当提示"待验证/待影像确认"→ 按方向适当打分（通常 4 分）；', 11, False),
    ('只有当输出在证据不足时仍"武断下结论"或漏掉关键风险时，才降至 2-3 分。', 11, False),
    ('每个维度独立打 1-5 整数，锚点定义如下（对照打分，不要凭整体印象）。', 11, False),
    ('', 10, False),
    ('【维度 1 · 合并诊断一致性】病因推断 / 严重度 / 黄疸类型与病例信息的吻合度', 12, True),
    ('5 = 方向完全一致，且分型/分级有可见数据支持', 10, False),
    ('4 = 方向正确；个别细节提示需补充化验/影像（输出已含验证限定语）', 10, False),
    ('3 = 主方向正确，但一处分型/严重度依据不足且输出未提示验证', 10, False),
    ('2 = 一处明确不一致或无依据的分型判断', 10, False),
    ('1 = 推断与病例信息方向性冲突', 10, False),
    ('', 10, False),
    ('【维度 2 · 建议适当性】处置场所 / 紧急度 / 转诊与具体建议', 12, True),
    ('5 = 与临床情境完全匹配', 10, False),
    ('4 = 总体匹配，个别环节可优化', 10, False),
    ('3 = 可接受但偏保守或偏激进（不影响安全）', 10, False),
    ('2 = 处置强度与病情明显不匹配', 10, False),
    ('1 = 可能延误诊治或不必要的侵入性处置', 10, False),
    ('', 10, False),
    ('【维度 3 · 安全覆盖】关键风险识别（肝衰竭/胆管炎/凝血障碍/溶血/DILI 等）', 12, True),
    ('5 = 关键风险全部识别，无过度警示', 10, False),
    ('4 = 全部识别，个别冗余提示', 10, False),
    ('3 = 主要风险覆盖，一处非关键遗漏', 10, False),
    ('2 = 一处关键风险遗漏或明显误报', 10, False),
    ('1 = 多处关键风险遗漏', 10, False),
    ('', 10, False),
    ('【维度 4 · 营养建议适当性】营养风险判断与 ESPEN 导向建议', 12, True),
    ('5 = 完全适当（含"提示筛查"在未筛查病例中的合理性）', 10, False),
    ('4 = 适当，个别表述可更具体（如未按体重给出目标值属可接受）', 10, False),
    ('3 = 基本适当但泛化', 10, False),
    ('2 = 一处与营养原则相悖', 10, False),
    ('1 = 建议可能造成营养风险', 10, False),
    ('', 10, False),
    ('【校准练习（不计入评分）】', 12, True),
    ('例 1：病例仅 TBIL 升高、AI 判胆汁郁积但无直/间胆分型；输出含"建议结合直/间胆验证"', 10, False),
    ('      → 维度 1 应打 4（方向+恰当限定），不宜因"无法核验"打 2。', 10, False),
    ('例 2：轻度黄疸病例输出"门诊周复查"，但该病例合并肝占位 → 维度 2 打 3（偏保守，不影响安全）。', 10, False),
    ('例 3：ACLF 病例输出未提 INR 监测 → 维度 3 打 2（关键遗漏）。', 10, False),
    ('', 10, False),
    ('【填写】四列各填 1-5 整数；≤2 分必须写"审评意见"说明。请独立完成，勿与他人讨论。', 11, True),
]

df = pd.read_excel(SRC)
rng = np.random.default_rng(SEED)
order = rng.permutation(len(df))
d = df.iloc[order].reset_index(drop=True)
d['审阅者'] = '审阅者C'

wb = Workbook()
ws = wb.active
ws.title = '校准评分说明'
ws.sheet_view.showGridLines = False
for i, (text, size, bold) in enumerate(RUBRIC, start=1):
    c = ws.cell(row=i, column=1, value=text)
    c.font = Font(name='Microsoft YaHei', size=size, bold=bold,
                  color=NAVY if bold else '333333')
    c.alignment = Alignment(vertical='center', wrap_text=True)
ws.column_dimensions['A'].width = 110

ws2 = wb.create_sheet('盲评病例')
ws2.sheet_view.showGridLines = False
header_fill = PatternFill('solid', fgColor=NAVY)
score_fill = PatternFill('solid', fgColor='FFF7E6')
cols = list(d.columns)
ws2.append(cols)
for j, col in enumerate(cols, start=1):
    c = ws2.cell(row=1, column=j)
    c.font = Font(name='Microsoft YaHei', size=10, bold=True, color='FFFFFF')
    c.fill = header_fill
    c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    c.border = border
    score_col = any(k in col for k in ('(1-5)', '审评意见'))
    ws2.column_dimensions[ws2.cell(row=1, column=j).column_letter].width = (
        30 if score_col else (45 if col in ('临床建议全文', '触发规则', '入院诊断') else 11))
for r_i, (_, row) in enumerate(d.iterrows(), start=2):
    for j, col in enumerate(cols, start=1):
        v = row[col]
        v = '' if pd.isna(v) else v
        c = ws2.cell(row=r_i, column=j, value=v)
        c.font = Font(name='Microsoft YaHei', size=10)
        c.alignment = Alignment(vertical='top', wrap_text=(col in (
            '临床建议全文', '触发规则', '入院诊断', '分诊结论', '审评意见')))
        c.border = border
        if any(k in col for k in ('(1-5)', '审评意见')):
            c.fill = score_fill
    ws2.row_dimensions[r_i].height = 60
ws2.freeze_panes = 'B2'

out = os.path.join(OUT_DIR, 'CDSS盲评_审阅者C_校准版.xlsx')
wb.save(out)
print(f'written: {out}  (30 cases, seed={SEED}, first pid={d["pid"].iloc[0]})')
print('coordinator-only: review_package/_discrepant_cases.csv (26 cases, do NOT show to C)')
