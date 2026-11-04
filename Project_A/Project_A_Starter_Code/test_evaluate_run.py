"""Synthetic-only checks for the held-out evaluator; never uses real test slides."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from dataloader import build_HDF5_feat_dataset
from evaluate_run import ROOT, digest, evaluate_run
from model import ABMIL
from verify_baseline import make_fixture


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='ece1512_eval_test_')
        self.base = Path(self.temp.name)
        self.run = self.base/'run'
        self.run.mkdir()
        self.output = self.base/'evaluation'
        self.data = self.base/'C16.h5'
        make_fixture('C16', self.data)
        conf = SimpleNamespace(dataset='CAMELYON16', data_dir=str(self.data),
            D_feat=8, D_inner=16, D_attn=8, n_class=2, droprate=0.0, B=2,
            seed=42, val_fraction=0.1, n_shot=0, method='abmil')
        datasets = build_HDF5_feat_dataset(str(self.data), conf)
        sources = {name: digest(ROOT/name) for name in
                   ('train.py', 'model.py', 'model_hybrid.py', 'engine.py', 'dataloader.py', 'utils.py')}
        objects = {'config.json': vars(conf),
            'environment.json': {'epochs': 2, 'source_sha256': sources},
            'history.json': [{'epoch': i, 'val_auc': 0.5, 'val_f1': 0.4} for i in (1, 2)],
            'summary.json': {'best_epoch': 1, 'best_validation_score': 0.9},
            'splits.json': {key: ds.data_names for key, ds in zip(('train', 'val', 'test'), datasets)}}
        for name, obj in objects.items():
            (self.run/name).write_text(json.dumps(obj))
        torch.manual_seed(42)
        torch.save(ABMIL(conf).state_dict(), self.run/'best_state.pt')

    def tearDown(self):
        self.temp.cleanup()

    def edit(self, name, transform):
        path = self.run/name
        content = json.loads(path.read_text())
        path.write_text(json.dumps(transform(content)))

    def test_outputs_probabilities_and_unchanged_checkpoint(self):
        before = digest(self.run/'best_state.pt')
        result = evaluate_run(self.run, self.output, threads=1, expected_epochs=2)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['benchmark'])
        frame = pd.read_csv(self.output/'predictions.csv')
        self.assertEqual(len(frame), 20)
        np.testing.assert_allclose(frame.filter(like='probability_class_').sum(axis=1), 1, atol=1e-6)
        self.assertEqual(digest(self.run/'best_state.pt'), before)
        self.assertTrue((self.output/'confusion_matrix.png').is_file())
        self.assertEqual(int(pd.read_csv(self.output/'confusion_matrix.csv', index_col=0).to_numpy().sum()), 20)
        with self.assertRaises(FileExistsError):
            evaluate_run(self.run, self.output, threads=1, expected_epochs=2)
        with self.assertRaisesRegex(ValueError, 'already has test'):
            evaluate_run(self.run, self.base/'different_output', threads=1, expected_epochs=2)

    def test_incomplete_training_rejected(self):
        self.edit('history.json', lambda content: content[:1])
        with self.assertRaisesRegex(ValueError, 'completed'):
            evaluate_run(self.run, self.output, threads=1, expected_epochs=2)

    def test_source_change_rejected(self):
        def change(content):
            content['source_sha256']['model.py'] = '0'*64
            return content
        self.edit('environment.json', change)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            evaluate_run(self.run, self.output, threads=1, expected_epochs=2)

    def test_overlap_rejected(self):
        def change(content):
            content['test'][0] = content['train'][0]
            return content
        self.edit('splits.json', change)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            evaluate_run(self.run, self.output, threads=1, expected_epochs=2)

    def test_debug_and_previously_tested_runs_rejected(self):
        with self.assertRaisesRegex(ValueError, 'completed 30'):
            evaluate_run(self.run, self.output, threads=1)
        def change(content):
            content['test'] = {'auc': 0.5}
            return content
        self.edit('summary.json', change)
        with self.assertRaisesRegex(ValueError, 'already has test'):
            evaluate_run(self.run, self.output, threads=1, expected_epochs=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
