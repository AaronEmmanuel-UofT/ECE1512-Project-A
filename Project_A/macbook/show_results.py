"""Print the report's primary baseline results without running the models."""
import csv
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent/'Project_A_Starter_Code'


def main():
    print('ABMIL BASELINE TEST RESULTS (seed 42, 30 training epochs)\n')
    for dataset in ('C16', 'C17', 'BRACS'):
        result = json.loads((ROOT/f'results/{dataset}_baseline_windows/heldout_test/metrics.json').read_text())
        metrics = result['metrics']
        print(f"{dataset:5} ABMIL best epoch {result['best_epoch']:2} | "
              f"test accuracy {metrics['acc']:6.2f}% | macro AUC {metrics['auc']:.4f} | "
              f"macro F1 {metrics['f1']:.4f} | loss {metrics['loss']:.4f}")
    print('\nThese are the primary baselines in the group report, trained on Windows.')
    print('C17 train/validation share 27 patients; one seed per configuration.')
    print('The earlier Mac C17 baseline (epoch 19, accuracy 73.00%) is a separate environment result.')
    full = ROOT/'results/full_comparison_20261102/comparison.csv'
    if full.exists():
        print('\nFull pooling comparison:')
        print(full.read_text())
    else:
        status = ROOT/'results/full_comparison_20261102/status.json'
        if status.exists():
            print('\nCurrent comparison:', json.loads(status.read_text())['stage'])
    print('Separate interactive/debug runs (not the final baseline records):')
    for folder in sorted((ROOT/'results'/'interactive').glob('*')):
        history = folder/'history.json'
        summary = folder/'summary.json'
        if history.exists():
            rows = json.loads(history.read_text())
            print(f'{folder.name}: {len(rows)} epochs saved')
        if summary.exists():
            print('  '+summary.read_text().replace('\n', ' '))


if __name__ == '__main__':
    main()
