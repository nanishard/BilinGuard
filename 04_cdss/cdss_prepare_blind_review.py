# -*- coding: utf-8 -*-
"""
Prepare the blinded clinician-review package for distribution.

Input : results/face_v4_manuscript_delivery/tables/CDSS_CLINICIAN_REVIEW_SAMPLE.xlsx
Output: review_package/CDSS盲评_审阅者A.xlsx / 审阅者B.xlsx (each with
        a scoring-instruction sheet + 30 cases in REVIEWER-SPECIFIC RANDOM
        ORDER, reviewer name pre-filled, four 1-5 scales + comment blank)

Randomisation: seeded per reviewer (A=101, B=202) — order effects controlled,
reconciliation via pid after return.

Also emits: review_package/发放说明.md (coordinator checklist)

Run: python code/cdss_prepare_blind_review.py
"""
import sys, os
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'D:\research\人脸识别营养\传染科'
SRC = os.path.join(BASE, 'results', 'face_v4_manuscript_delivery', 'tables',
                   'CDSS_CLINICIAN_REVIEW_SAMPLE.xlsx')
OUT_DIR = os.path.join(BASE, 'review_package')
os.makedirs(OUT_DIR, exist_ok=True)

REVIEWERS = [('A', 101), ('B', 202)]   # (label, random seed)

INSTRUCTION_LINES = [
    ('BilinGuard CDSS 临床建议盲评', 16, True),
    ('（请独立评分，勿与其他审阅者讨论；预计 45-60 分钟）', 11, False),
    ('', 10, False),
    ('一、任务', 12, True),
    ('以下 30 例来自真实住院队列。每例已给出：入院诊断、关键化验、Child-Pugh、', 11, False),
    ('NRS-2002 营养评分，以及 BilinGuard CDSS 的完整输出（触发规则、分诊结论、临床建议全文）。', 11, False),
    ('请以感染/肝病专科医生视角，对 CDSS 输出做四维 1-5 分评分。', 11, False),
    ('', 10, False),
    ('二、评分维度（1=完全不适当 … 3=部分适当 … 5=完全适当）', 12, True),
    ('1. 合并诊断一致性：CDSS 的病因推断、严重程度、黄疸类型与病例临床信息的吻合程度。', 11, False),
    ('2. 建议适当性：处置场所、紧急程度、转诊去向与具体建议符合该病例临床情境的程度。', 11, False),
    ('3. 安全覆盖：关键风险（肝衰竭/胆管炎/凝血障碍/溶血/DILI 等）是否被充分识别，', 11, False),
    ('   有无漏报与误报。', 11, False),
    ('4. 营养建议适当性：营养风险判断与 ESPEN 建议对该病例的适当性；', 11, False),
    ('   未筛查病例评其"提示筛查"的合理性。', 11, False),
    ('', 10, False),
    ('三、填写方式', 12, True),
    ('在对应列填 1-5 整数；如有低分（≤2）请在"审评意见"写明原因（漏了什么/错在哪）。', 11, False),
    ('您的姓名已预填（审阅者列）；请勿修改病例顺序与任何临床内容列。', 11, False),
    ('', 10, False),
    ('四、声明', 12, True),
    ('本评评为研究用途；病例信息已做最小化处理（无姓名/住院号）。', 11, False),
]

df = pd.read_excel(SRC)
print(f'source: {len(df)} cases, columns OK: '
      f'{all(c in df.columns for c in ["pid", "临床建议全文", "审阅者"])}')

NAVY = '1D3557'
TEAL = '0F766E'
thin = Side(style='thin', color='CCCCCC')
border = Border(left=thin, right=thin, top=thin, bottom=thin)

for label, seed in REVIEWERS:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    d = df.iloc[order].reset_index(drop=True)
    d['审阅者'] = f'审阅者{label}'

    wb = Workbook()
    # ── sheet 1: instructions ──
    ws = wb.active
    ws.title = '评分说明'
    ws.sheet_view.showGridLines = False
    for i, (text, size, bold) in enumerate(INSTRUCTION_LINES, start=1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name='Microsoft YaHei', size=size, bold=bold,
                      color=NAVY if bold else '333333')
        c.alignment = Alignment(vertical='center')
    ws.column_dimensions['A'].width = 100

    # ── sheet 2: cases ──
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
        ws2.row_dimensions[r_i].height = 90 if r_i % 30 == 1 else 60
    ws2.freeze_panes = 'B2'

    out = os.path.join(OUT_DIR, f'CDSS盲评_审阅者{label}.xlsx')
    wb.save(out)
    print(f'  written: {out}  (case order seed={seed}, first pid={d["pid"].iloc[0]})')

# ── coordinator checklist ──
readme = f'''# CDSS 盲评发放说明（协调者清单）

生成时间：2026-09-16 · 源：CDSS_CLINICIAN_REVIEW_SAMPLE.xlsx（30 例真实队列分层抽样）

## 发放
1. 将 `review_package/CDSS盲评_审阅者A.xlsx`、`CDSS盲评_审阅者B.xlsx` 分别发给 ≥2 名感染/肝病专科医生
   （两人版本病例顺序不同——seed 101/202——回收后按 pid 对齐；请独立评分、勿交流）。
2. 提醒：评分说明在工作簿第一页；四个 1-5 维度；低分（≤2）需写"审评意见"。

## 回收后（一条命令）
```bash
# 1. 合并两位审阅者的回收文件（修改下方两个路径为实际回收文件）
python code/cdss_merge_review_returns.py "review_package/CDSS盲评_审阅者A_回收.xlsx" "review_package/CDSS盲评_审阅者B_回收.xlsx"

# 2. 统计分析（ICC(2,k)/加权kappa/分层适当率/低分清单）
python code/cdss_review_score_analysis.py "results/face_v4_manuscript_delivery/tables/CDSS_REVIEW_RETURNED_MERGED.xlsx"
```

## 产物链
- 发放包：`review_package/CDSS盲评_审阅者A.xlsx` / `_B.xlsx`（内置评分说明页）
- 合并输出：`results/.../CDSS_REVIEW_RETURNED_MERGED.xlsx`
- 分析输出：`results/.../CDSS_REVIEW_SCORE_ANALYSIS.csv`（论文 Table 素材）

## 质量要求
- 每位审阅者至少完成 28/30 例（缺失 >2 例需说明）
- 若两位审阅者某域加权 kappa < 0.4，需第三位仲裁
'''
with open(os.path.join(OUT_DIR, '发放说明.md'), 'w', encoding='utf-8') as f:
    f.write(readme)
print(f'  written: {os.path.join(OUT_DIR, "发放说明.md")}')
print('\nPackage ready.')
