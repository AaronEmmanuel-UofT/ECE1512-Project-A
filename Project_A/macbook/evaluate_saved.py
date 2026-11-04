"""Evaluate one selected checkpoint without retraining or changing its run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent.parent / 'Project_A_Starter_Code'
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from engine import evaluate
from model import ABMIL
from model_hybrid import AttentionMeanMIL
from utils import Struct, infer_feature_dim


def evaluate_run(run, data=None):
    run = Path(run).resolve()
    data = Path(data).resolve() if data else None
    os.chdir(ROOT)
    conf = Struct(**json.loads((run / 'config.json').read_text()))
    if data:
        conf.data_dir = str(data)
    datasets = build_HDF5_feat_dataset(conf.data_dir, conf)
    actual = dict(zip(('train', 'val', 'test'), (ds.data_names for ds in datasets)))
    saved = json.loads((run / 'splits.json').read_text())
    if actual != saved:
        raise ValueError('Saved split IDs/order differ from reconstructed data; refusing evaluation')
    if infer_feature_dim(datasets[0]) != int(conf.D_feat):
        raise ValueError('Feature dimension differs from saved configuration')
    summary = json.loads((run / 'summary.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    selected = max(history, key=lambda row: row['val_auc'] + row['val_f1'])
    if selected['epoch'] != summary['best_epoch']:
        raise ValueError('Saved summary does not match validation-selected epoch')
    if not math.isclose(selected['val_auc'] + selected['val_f1'], summary['best_validation_score']):
        raise ValueError('Saved selection score does not match history')
    method = getattr(conf, 'method', 'abmil')
    if method not in ('abmil', 'attention_mean'):
        raise ValueError(f'Unknown saved method: {method}')
    model = (ABMIL if method == 'abmil' else AttentionMeanMIL)(conf).to('cpu')
    checkpoint = run / 'best_state.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    loader = DataLoader(datasets[2], batch_size=int(conf.B), shuffle=False,
                        num_workers=0, collate_fn=mil_collate_fn)
    metrics = evaluate(model, torch.nn.CrossEntropyLoss(), loader, torch.device('cpu'), conf)
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError('Nonfinite held-out metric')
    return dict(kind='held-out evaluation of validation-selected checkpoint', run=str(run),
                dataset=conf.dataset, method=method, seed=conf.seed,
                mean_weight=getattr(conf, 'hybrid_mean_weight', 0.0),
                best_epoch=summary['best_epoch'], test_slides=len(datasets[2]), test=metrics,
                test_slide_ids=actual['test'], checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                python=platform.python_version(), torch=torch.__version__, device='cpu',
                cpu_threads=torch.get_num_threads(), source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                    for name in ('model.py', 'model_hybrid.py', 'engine.py', 'dataloader.py', 'utils.py')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Evaluation output already exists; refusing overwrite/repeated testing')
    torch.set_num_threads(2)
    result = evaluate_run(args.run, args.data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
