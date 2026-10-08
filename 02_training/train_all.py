"""
BilinGuard: Complete Training Orchestration
Runs the full reproducible pipeline:
  1. Data cleaning & merging
  2. Binary classifier training (YOLO + questionnaire ML)
  3. Ternary grading training (all backbones + YellowFeatures + ensemble)
  4. Evaluation with bootstrap CIs
"""
import os, sys, subprocess

BASE = os.path.dirname(__file__)
SCRIPTS = [
    'pipeline_data_cleaning.py',
    'train_binary.py',
    'train_grading.py',
]

def run(cmd):
    print(f'\n{"="*60}')
    print(f'Running: {cmd}')
    print(f'{"="*60}')
    result = subprocess.run(cmd, shell=True, cwd=BASE)
    if result.returncode != 0:
        print(f'ERROR: command failed with code {result.returncode}')
        sys.exit(1)
    print(f'Completed: {cmd}')

if __name__ == '__main__':
    # Step 1: Data cleaning
    run(f'{sys.executable} {os.path.join(BASE, "pipeline_data_cleaning.py")}')

    # Step 2: Binary training
    run(f'{sys.executable} {os.path.join(BASE, "train_binary.py")}')

    # Step 3: Ternary grading training (5-fold cross-validation)
    for fold in range(5):
        run(f'{sys.executable} {os.path.join(BASE, "train_grading.py")} --fold {fold}')

    # Step 4: Final evaluation
    run(f'{sys.executable} {os.path.join(BASE, "train_grading.py")} --skip-train --fold 0')

    print('\n' + '='*60)
    print('BilinGuard training pipeline complete!')
    print('='*60)
