# -*- coding: utf-8 -*-
"""Print clean summary of all bootstrap metrics with CIs."""
import pandas as pd, json

df = pd.read_csv(r'D:\research\人脸识别营养\传染科\results\tables\bootstrap_full_metrics.csv')

print('=' * 120)
print('  COMPLETE BOOTSTRAP RESULTS — ALL METRICS WITH 95% CI (1000 iterations)')
print('=' * 120)

for task in sorted(df['task'].unique()):
    sub = df[df['task'] == task].sort_values('auc', ascending=False)
    nc = int(sub.iloc[0]['nc'])
    print(f'\n{"─"*120}')
    print(f'  {task} ({"binary" if nc==2 else "ternary"}, n_val={int(sub.iloc[0]["n_val"])})')
    print(f'{"─"*120}')
    print(f'  {"Model":<28} {"AUC [95% CI]":>26} {"Sens [95% CI]":>26} {"Spec [95% CI]":>26} {"F1 [95% CI]":>26}')
    print(f'  {"─"*116}')
    for _, r in sub.iterrows():
        auc = f'{r["auc"]:.3f} [{r["auc_lo"]:.3f}-{r["auc_hi"]:.3f}]'
        sens = f'{r["sens"]:.3f} [{r["sens_lo"]:.3f}-{r["sens_hi"]:.3f}]'
        spec = f'{r["spec"]:.3f} [{r["spec_lo"]:.3f}-{r["spec_hi"]:.3f}]'
        f1 = f'{r["f1"]:.3f} [{r["f1_lo"]:.3f}-{r["f1_hi"]:.3f}]'
        print(f'  {r["name"]:<28} {auc:>26} {sens:>26} {spec:>26} {f1:>26}')

print(f'\n{"="*120}')
print('  Files: results/tables/bootstrap_full_metrics.csv + Table_bootstrap_full_metrics.docx')
print('='*120)
