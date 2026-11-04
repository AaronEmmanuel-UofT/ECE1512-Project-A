"""Evaluate a completed research run once on its saved held-out split.

Adapted from evaluate_run.py; the original baseline evaluator is unchanged.

Does not train, tune, regenerate checkpoints or overwrite earlier evaluations.
Saves slide probabilities so metrics and error analysis can be reproduced without
running the model on the held-out slides again.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from research.train_extension import build_model
from utils import Struct, classification_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate_run(run_dir, output, data_path=None, threads=2, expected_epochs=30):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Evaluation output already exists: {output}')
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    config = read_json(run_dir/'config.json')
    environment = read_json(run_dir/'environment.json')
    history = read_json(run_dir/'history.json')
    summary = read_json(run_dir/'summary.json')
    saved_splits = read_json(run_dir/'splits.json')
    if int(environment['epochs']) != expected_epochs or len(history) != expected_epochs:
        raise ValueError(f'Expected a completed {expected_epochs}-epoch run')
    if [row['epoch'] for row in history] != list(range(1, expected_epochs+1)):
        raise ValueError('History epochs are not consecutive')
    if 'test' in summary:
        raise ValueError('This run already has test metrics; reuse them rather than reevaluate')
    scores = [row['val_auc']+row['val_f1'] for row in history]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError('Invalid validation scores')
    selected = int(np.argmax(scores))+1
    if summary['best_epoch'] != selected or not math.isclose(
            summary['best_validation_score'], scores[selected-1], abs_tol=1e-7):
        raise ValueError('Summary does not match the original validation selection rule')
    for name, expected in environment['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Training source changed since run: {name}')
    for name, expected in environment['source_sha256'].items():
        if digest(run_dir/'source_snapshot'/name) != expected:
            raise ValueError(f'Archived training source changed: {name}')
    for key in ('train', 'val', 'test'):
        if not saved_splits[key] or len(set(saved_splits[key])) != len(saved_splits[key]):
            raise ValueError(f'Empty or duplicate {key} slide IDs')
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(saved_splits[left]) & set(saved_splits[right]):
            raise ValueError(f'Slide overlap between {left} and {right}')
    conf = Struct(**config)
    path = Path(data_path).resolve() if data_path else Path(conf.data_dir)
    if not path.is_absolute():
        path = ROOT/path
    conf.data_dir = str(path)
    # Resolve metadata against the same root used by train.py.
    for key in ('dataset_csv_dir', 'camelyon17_csv', 'bracs_csv'):
        if hasattr(conf, key) and not Path(getattr(conf, key)).is_absolute():
            setattr(conf, key, str(ROOT/getattr(conf, key)))
    datasets = build_HDF5_feat_dataset(str(path), conf)
    for key, ds in zip(('train', 'val', 'test'), datasets):
        if saved_splits[key] != ds.data_names:
            raise ValueError(f'{key} split no longer matches saved run')
    test_data = datasets[2]
    if test_data.feature_dim != int(conf.D_feat):
        raise ValueError('Feature dimension differs from trained model')
    method = getattr(conf, 'method', 'abmil')
    if method not in ('abmil', 'attention_mean', 'mean', 'fragility'):
        raise ValueError(f'Unknown method: {method}')
    model = build_model(conf).eval()
    checkpoint = run_dir/'best_state.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    loader = DataLoader(test_data, batch_size=int(conf.B), shuffle=False,
                        num_workers=0, collate_fn=mil_collate_fn)
    provenance = dict(run=str(run_dir), dataset=conf.dataset, method=method,
        seed=int(conf.seed), completed_epochs=len(history), best_epoch=selected,
        selection_rule='maximum validation macro AUROC + macro F1; earliest tie',
        checkpoint_sha256=digest(checkpoint), splits_sha256=digest(run_dir/'splits.json'),
        training_source_sha256=environment['source_sha256'],
        evaluator_sha256=digest(__file__), data_file=str(path), data_bytes=path.stat().st_size,
        test_slides=len(test_data), class_count=int(conf.n_class),
        device='cpu', cpu_threads=threads, python=platform.python_version(),
        torch=torch.__version__, created_utc=datetime.now(timezone.utc).isoformat(),
        benchmark=expected_epochs == 30)
    output.mkdir(parents=True, exist_ok=False)
    (output/'evaluator_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'evaluation_plan.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    labels_all, probabilities_all, slide_ids = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    gate_rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            if method == 'fragility':
                logits, diagnostics = model(batch['input'].float(), mask=batch['mask'], return_diagnostics=True)
                for i, slide in enumerate(batch['slide_id']):
                    row = dict(slide_id=slide, alpha=float(diagnostics['alpha'][i]),
                        valid_patches=int(diagnostics['valid_patches'][i]),
                        removed_attention_mass=float(diagnostics['removed_attention_mass'][i]),
                        removed_patch_equivalents=float(diagnostics['removed_patch_equivalents'][i]))
                    for c in range(conf.n_class):
                        row[f'attention_probability_class_{c}'] = float(diagnostics['attention_probabilities'][i,c])
                        row[f'probe_probability_class_{c}'] = float(diagnostics['probe_probabilities'][i,c])
                    gate_rows.append(row)
            else:
                logits = model(batch['input'].float(), mask=batch['mask'])
            labels = batch['label'].long()
            loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError('Nonfinite held-out predictions or loss')
            probabilities_all.append(logits.softmax(dim=1).cpu())
            labels_all.append(labels.cpu())
            slide_ids.extend(batch['slide_id'])
            total_loss += float(loss)
            if index % 50 == 0:
                print(f'{conf.dataset}: evaluated {len(slide_ids)}/{len(test_data)} test slides', flush=True)
    probabilities = torch.cat(probabilities_all)
    labels = torch.cat(labels_all)
    if labels.unique().numel() != int(conf.n_class):
        raise ValueError('Held-out split lacks a class; macro AUROC is undefined')
    metrics = classification_metrics(probabilities, labels, conf.n_class)
    metrics['loss'] = total_loss/len(test_data)
    y, p = labels.numpy(), probabilities.numpy()
    predicted = p.argmax(axis=1)
    per_auc = [roc_auc_score(y == c, p[:, c]) for c in range(conf.n_class)]
    independent = dict(auc=float(np.mean(per_auc)),
                       acc=100*accuracy_score(y, predicted),
                       f1=f1_score(y, predicted, average='macro'))
    for key, expected in independent.items():
        np.testing.assert_allclose(metrics[key], expected, atol=1e-5, rtol=0)
    frame = pd.DataFrame(dict(slide_id=slide_ids, label=y, predicted=predicted))
    for c in range(conf.n_class):
        frame[f'probability_class_{c}'] = p[:, c]
    assert slide_ids == saved_splits['test']
    frame.to_csv(output/'predictions.csv', index=False, float_format='%.10g')
    if gate_rows:
        assert [row['slide_id'] for row in gate_rows] == slide_ids
        pd.DataFrame(gate_rows).to_csv(output/'gate_diagnostics.csv', index=False, float_format='%.10g')
    matrix = confusion_matrix(y, predicted, labels=list(range(conf.n_class)))
    pd.DataFrame(matrix, index=[f'true_{c}' for c in range(conf.n_class)],
                 columns=[f'predicted_{c}' for c in range(conf.n_class)]).to_csv(output/'confusion_matrix.csv')
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=list(range(conf.n_class)), zero_division=0)
    pd.DataFrame(dict(class_id=range(conf.n_class), precision=precision, recall=recall,
        f1=f1, support=support, auc_one_vs_rest=per_auc)).to_csv(output/'per_class_metrics.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    shown = ax.imshow(matrix, cmap='Blues', vmin=0)
    fig.colorbar(shown, ax=ax, label='Number of slides')
    for i in range(conf.n_class):
        for j in range(conf.n_class):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black')
    ax.set(xticks=range(conf.n_class), yticks=range(conf.n_class), xlabel='Predicted class ID',
           ylabel='True class ID', title=f'{conf.dataset}: held-out test confusion matrix')
    fig.tight_layout()
    fig.savefig(output/'confusion_matrix.png', dpi=180)
    plt.close(fig)
    result = dict(**provenance, metrics=metrics, metric_crosscheck='passed against scikit-learn',
                  validation_at_selected_epoch=history[selected-1], status='complete')
    (output/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    # Record completion in the run as well as the evaluation folder. Preserve
    # the original validation-only summary and prevent later fresh-path repeats.
    (run_dir/'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    summary.update(test=metrics, test_evaluation=str(output),
                   test_checkpoint_sha256=provenance['checkpoint_sha256'])
    temporary = run_dir/'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    temporary.replace(run_dir/'summary.json')
    print(json.dumps(dict(dataset=conf.dataset, best_epoch=selected, test=metrics), indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New evaluation directory')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    evaluate_run(args.run, args.output, args.data, args.threads)


if __name__ == '__main__':
    main()
