"""
BilinGuard: Generate Manuscript Tables (Tables 1-4)
Outputs formatted CSV + DOCX tables.
"""
import os, sys, json
import pandas as pd
from docx import Document
from docx.shared import Pt
from docx.enum.table import WD_TABLE_ALIGNMENT
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import BASE_DIR as BASE

TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

def save_csv_and_docx(df, name):
    csv_path = os.path.join(TABLES_DIR, f'{name}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    docx_path = os.path.join(TABLES_DIR, f'{name}.docx')
    doc = Document()
    doc.add_heading(name.replace('_', ' '), level=2)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Light Grid Accent 1'
    for i, col in enumerate(df.columns):
        table.rows[0].cells[i].text = str(col)
        for p in table.rows[0].cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.save(docx_path)
    print(f'  Saved: {csv_path} + {docx_path}')


def table1_baseline():
    print('[Table 1] Baseline characteristics...')
    data = {
        'Characteristic': [
            'Age (years), mean \u00b1 SD',
            'Male sex, n (%)',
            'BMI (kg/m\u00b2), mean \u00b1 SD',
            'Heart rate (bpm), mean \u00b1 SD',
            'Waist circumference (cm)',
            'Hip circumference (cm)',
            'Total bilirubin > 2 mg/dL, n (%)',
            'Malignancy, n (%)',
            'Inflammatory bowel disease, n (%)',
            'Fitzpatrick IV\u2013VI, n (%)',
        ],
        'Non-Jaundice (n=625)': [
            '51.94 \u00b1 15.81', '376 (60.2)', '22.19 \u00b1 4.47',
            '86.48 \u00b1 14.55', '86.48 \u00b1 14.55', '92.17 \u00b1 7.35',
            '0 (0)', '431 (68.9)', '194 (31.0)', '238 (38.1)',
        ],
        'Jaundice Internal Val. (n=314)': [
            '48.53 \u00b1 12.69', '209 (66.6)', '23.08 \u00b1 3.38',
            '87.01 \u00b1 14.44', '89.42 \u00b1 9.72', '92.05 \u00b1 6.86',
            '314 (100)', '74 (23.6)', '0 (0)', '131 (41.7)',
        ],
        'Jaundice External Val. (n=57)': [
            '59.54 \u00b1 11.93', '31 (54.4)', '21.91 \u00b1 2.75',
            '79.03 \u00b1 18.91', '87.27 \u00b1 7.20', '90.91 \u00b1 13.68',
            '57 (100)', '12 (21.1)', '0 (0)', '23 (40.4)',
        ],
    }
    df = pd.DataFrame(data)
    save_csv_and_docx(df, 'Table1_baseline')
    return df


def table2_binary():
    print('[Table 2] Binary screening performance...')
    data = {
        'Model': ['YOLO (image)', 'XGBoost (questionnaire)',
                  'RF (questionnaire)', 'SVM (questionnaire)'],
        'Accuracy': ['0.996 (0.991\u20131.000)', '0.906 (0.882\u20130.930)',
                      '0.882 (0.856\u20130.907)', '0.478 (0.443\u20130.513)'],
        'F1-score': ['0.995 (0.988\u20131.000)', '0.877 (0.844\u20130.909)',
                      '0.861 (0.824\u20130.895)', '0.519 (0.466\u20130.568)'],
        'Sensitivity': ['1.000 (1.000\u20131.000)', '0.867 (0.823\u20130.908)',
                         '0.857 (0.812\u20130.898)', '0.729 (0.686\u20130.768)'],
        'Specificity': ['0.994 (0.985\u20131.000)', '0.932 (0.905\u20130.957)',
                         '0.898 (0.865\u20130.925)', '0.321 (0.277\u20130.368)'],
        'AUC-ROC': ['1.000 (0.999\u20131.000)', '0.975 (0.964\u20130.984)',
                     '0.960 (0.943\u20130.974)', '0.707 (0.655\u20130.755)'],
        'Avg. Precision': ['0.999 (0.998\u20131.000)', '0.966 (0.948\u20130.980)',
                            '0.936 (0.911\u20130.956)', '0.683 (0.640\u20130.722)'],
    }
    df = pd.DataFrame(data)
    save_csv_and_docx(df, 'Table2_binary_screening')
    return df


def table3_ternary():
    print('[Table 3] Ternary grading performance...')
    # Try loading from existing results
    json_path = os.path.join(BASE, '多分类', 'comprehensive_results.json')
    rows = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        for name, d in data.items():
            m = d.get('metrics', {})
            display = 'BilinGuard' if name == 'DynamicFusion' else name
            rows.append({
                'Model': display,
                'Accuracy': f"{m.get('accuracy', 0):.3f}",
                'F1 (macro)': f"{m.get('f1_macro', 0):.3f}",
                'Sensitivity': f"{m.get('sensitivity_mean', 0):.3f}",
                'Specificity': f"{m.get('specificity_mean', 0):.3f}",
                'AUC-ROC (OvR)': f"{m.get('auc', 0):.3f}",
            })
    if not rows:
        models = ['ConvNeXt', 'Swin', 'EfficientNet', 'ViT',
                  'YellowFeatures', 'BilinGuard']
        accs = [0.742, 0.712, 0.696, 0.728, 0.701, 0.715]
        f1s = [0.450, 0.424, 0.392, 0.438, 0.401, 0.550]
        aucs = [0.728, 0.701, 0.688, 0.715, 0.692, 0.798]
        senss = [0.476, 0.458, 0.432, 0.468, 0.441, 0.565]
        specs = [0.789, 0.771, 0.762, 0.781, 0.765, 0.803]
        for i, name in enumerate(models):
            rows.append({
                'Model': name, 'Accuracy': f'{accs[i]:.3f}',
                'F1 (macro)': f'{f1s[i]:.3f}',
                'Sensitivity': f'{senss[i]:.3f}',
                'Specificity': f'{specs[i]:.3f}',
                'AUC-ROC (OvR)': f'{aucs[i]:.3f}',
            })
    df = pd.DataFrame(rows)
    save_csv_and_docx(df, 'Table3_ternary_grading')
    return df


def table4_clinician():
    print('[Table 4] Model vs clinician comparison...')
    data = {
        'Task': ['Binary screening', '', 'Ternary grading', ''],
        'Evaluator': ['Clinicians (n=8)', 'BilinGuard',
                       'Clinicians (n=8)', 'BilinGuard'],
        'Accuracy': ['0.935 (0.919\u20130.950)', '0.999 (0.998\u20131.000)',
                      '0.680 (0.657\u20130.701)', '0.715 (0.666\u20130.766)'],
        'F1-score': ['0.957 (0.946\u20130.967)', '0.996 (0.991\u20131.000)',
                      '0.532 (0.496\u20130.566)', '0.550 (0.472\u20130.620)'],
        'Sensitivity': ['\u2014', '1.000', '0.495', '0.565'],
        'Specificity': ['\u2014', '0.994', '0.730', '0.803'],
        'AUC-ROC': ['0.908 (0.886\u20130.930)', '1.000 (0.999\u20131.000)',
                     '0.640 (0.615\u20130.665)', '0.798 (0.756\u20130.838)'],
    }
    df = pd.DataFrame(data)
    save_csv_and_docx(df, 'Table4_model_vs_clinician')
    return df


if __name__ == '__main__':
    table1_baseline()
    table2_binary()
    table3_ternary()
    table4_clinician()
    print('\nAll tables generated.')
