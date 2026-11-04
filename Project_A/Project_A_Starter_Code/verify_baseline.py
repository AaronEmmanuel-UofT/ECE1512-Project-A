"""Exercise the full CLI on synthetic C16/C17/BRACS-shaped fixtures.

This verifies software behavior, NOT pathology performance. Uses temporary HDF5
files and saves only a verification report. Run with the PyCharm interpreter.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from preflight import audit
from train import CONFIGS
from utils import classification_metrics, load_config

ROOT = Path(__file__).resolve().parent


def make_fixture(dataset, path):
    if dataset == 'C16':
        rows = [(f'train_{i:03}', i % 2) for i in range(80)]
        rows += [(f'test_{i:03}', i % 2) for i in range(20)]
    else:
        table = pd.read_csv(ROOT/'dataset_csv'/(CONFIGS[dataset]+'.csv'))
        if dataset == 'C17':
            table['fixture_split'] = np.where(table.center < 3, 'trainval', 'test')
            chosen = table.groupby(['fixture_split', 'label'], sort=True).head(12)
        else:
            chosen = table.groupby(['split_info', 'label'], sort=True).head(4)
        rows = [(str(row.slide_id), int(row.label)) for row in chosen.itertuples()]
    rng = np.random.default_rng(1512)
    with h5py.File(path, 'w') as h5:
        for index, (name, label) in enumerate(rows):
            group = h5.create_group(name)
            n = 3 + index % 6
            x = rng.normal(0, 0.1, (n, 8)).astype('float32')
            x[:, label] += 2
            group.create_dataset('feat', data=x)
            group.attrs['label'] = label


def run(command, expected_success=True):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=180)
    if (proc.returncode == 0) != expected_success:
        raise AssertionError(f'Unexpected exit {proc.returncode}: {proc.stdout}\n{proc.stderr}')
    return proc


def metric_crosscheck():
    rng = np.random.default_rng(42)
    for classes in (2, 4, 7):
        labels = np.tile(np.arange(classes), 9)
        p = rng.uniform(0.01, 1, (len(labels), classes))
        p /= p.sum(axis=1, keepdims=True)
        got = classification_metrics(torch.tensor(p), torch.tensor(labels), classes)
        # Average class-vs-rest AUC explicitly, including the binary case.
        auc = np.mean([roc_auc_score(labels == c, p[:, c]) for c in range(classes)])
        expected = dict(auc=auc, f1=f1_score(labels, p.argmax(1), average='macro'),
                        acc=100*accuracy_score(labels, p.argmax(1)))
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'windows_verification.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f'Report already exists: {output}')
    torch.set_num_threads(1)
    os.chdir(ROOT)
    metric_crosscheck()
    report = dict(kind='synthetic software verification; not benchmark performance',
                  python=sys.version, executable=sys.executable,
                  metric_crosscheck='PASS: sklearn agrees for 2, 4 and 7 classes', datasets={})
    with tempfile.TemporaryDirectory(prefix='ece1512_verify_') as tmp:
        tmp = Path(tmp)
        for dataset in CONFIGS:
            path = tmp/(dataset+'.h5')
            make_fixture(dataset, path)
            conf = load_config(ROOT/'config'/(CONFIGS[dataset]+'_medical_ssl_config.yml'))
            checked = audit(path, conf, scan_features=True)
            assert checked['passed'], checked['errors']
            # Reject corrupted data before training.
            with h5py.File(path, 'r+') as h5:
                name = sorted(h5)[0]
                original = h5[name]['feat'][0, 0]
                h5[name]['feat'][0, 0] = np.nan
            assert not audit(path, conf, scan_features=True)['passed']
            with h5py.File(path, 'r+') as h5:
                h5[name]['feat'][0, 0] = original
            dest = tmp/(dataset+'_run')
            command = [sys.executable, str(ROOT/'train.py'), '--dataset', dataset,
                       '--data', str(path), '--epochs', '2', '--threads', '1', '--output', str(dest)]
            # Check opt-in test path on C16; other fixtures must not evaluate test.
            if dataset == 'C16':
                command.append('--test')
            completed = run(command)
            required = ('config.json', 'environment.json', 'splits.json', 'history.json',
                        'best_state.pt', 'summary.json', 'loss_curves.png')
            assert all((dest/name).is_file() for name in required)
            history = json.loads((dest/'history.json').read_text())
            summary = json.loads((dest/'summary.json').read_text())
            assert len(history) == 2
            assert np.isfinite([[v for v in row.values()] for row in history]).all()
            scores = [row['val_auc']+row['val_f1'] for row in history]
            assert summary['best_epoch'] == int(np.argmax(scores))+1
            np.testing.assert_allclose(summary['best_validation_score'], max(scores))
            assert ('test' in summary) == (dataset == 'C16')
            # Existing output protection must preserve all prior run artifacts.
            before = {name: (dest/name).read_bytes() for name in required}
            run(command, expected_success=False)
            assert all((dest/name).read_bytes() == value for name, value in before.items())
            extension_dest = tmp/(dataset+'_attention_mean')
            extension_command = [sys.executable, str(ROOT/'train.py'), '--dataset', dataset,
                '--data', str(path), '--epochs', '2', '--threads', '1',
                '--method', 'attention_mean', '--mean-weight', '0.5',
                '--output', str(extension_dest)]
            run(extension_command)
            assert all((extension_dest/name).is_file() for name in required)
            assert (dest/'splits.json').read_bytes() == (extension_dest/'splits.json').read_bytes()
            extension_history = json.loads((extension_dest/'history.json').read_text())
            assert len(extension_history) == 2
            assert np.isfinite([[v for v in row.values()] for row in extension_history]).all()
            extension_config = json.loads((extension_dest/'config.json').read_text())
            assert extension_config['method'] == 'attention_mean'
            assert extension_config['hybrid_mean_weight'] == 0.5
            assert 'test' not in json.loads((extension_dest/'summary.json').read_text())
            report['datasets'][dataset] = dict(status='PASS',
                split_sizes={key: value['slides'] for key, value in checked['splits'].items()
                             if key in ('train', 'val', 'test')},
                checks=['finite feature audit', 'NaN rejection', '2 CLI epochs',
                        'all artifacts', 'validation checkpoint selection',
                        'test opt-in', 'overwrite protection',
                        '2 hybrid CLI epochs with identical split IDs'],
                warnings=checked['warnings'])
            print(dataset, 'PASS', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print('PASS: full synthetic runner verification. Report:', output)


if __name__ == '__main__':
    main()
