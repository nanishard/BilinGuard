"""
BilinGuard: Master Evaluation Script
Runs the complete evaluation pipeline:
  1. Bootstrap evaluation
  2. Table generation (Tables 1-4)
  3. Figure generation (Figures 2-4, Supplementary S3-S10)
  4. Reader study analysis
  5. Summary report
"""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from config import *

CODE_DIR = os.path.dirname(__file__)

def run_script(name):
    path = os.path.join(CODE_DIR, name)
    print(f'\n{"=" * 60}')
    print(f'Running {name}...')
    print(f'{"=" * 60}')
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f'WARNING: {name} exited with code {result.returncode}')
    return result.returncode == 0

if __name__ == '__main__':
    scripts = [
        ('evaluate_bootstrap.py', 'Bootstrap evaluation'),
        ('generate_tables.py', 'Manuscript tables'),
        ('generate_figures.py', 'Manuscript figures'),
        ('reader_study_analysis.py', 'Reader study analysis'),
    ]
    print('=' * 60)
    print('BilinGuard Master Evaluation Pipeline')
    print('=' * 60)

    results = {}
    for script, desc in scripts:
        ok = run_script(script)
        results[desc] = 'PASS' if ok else 'FAIL'

    print('\n' + '=' * 60)
    print('Pipeline Summary')
    print('=' * 60)
    for desc, status in results.items():
        marker = '[OK]' if status == 'PASS' else '[FAIL]'
        print(f'  {marker} {desc}')

    # Check output files
    print(f'\nOutput files in {RESULTS_DIR}:')
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), RESULTS_DIR)
            size = os.path.getsize(os.path.join(root, f))
            print(f'  {rel} ({size / 1024:.1f} KB)')

    print('\nDone.')
