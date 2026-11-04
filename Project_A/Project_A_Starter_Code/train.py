"""Run from PyCharm or a terminal. No feature encoder training is required."""
import argparse
import json
import math
from pathlib import Path
import os
import platform
import time
import sys
import hashlib
import importlib.metadata
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from utils import load_config, seed_everything, infer_feature_dim
from model import ABMIL
from model_hybrid import AttentionMeanMIL
from engine import train_one_epoch, evaluate

CONFIGS = {'C16': 'camelyon16', 'C17': 'camelyon17', 'BRACS': 'bracs'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=CONFIGS, default='C16')
    parser.add_argument('--data', type=Path, help='Optional HDF5 path override')
    parser.add_argument('--epochs', type=int, help='Default: configured 30 epochs')
    parser.add_argument('--output', type=Path, required=True, help='New results directory')
    parser.add_argument('--test', action='store_true', help='Evaluate held-out test after validation selection')
    parser.add_argument('--threads', type=int, help='CPU thread count (small MIL bags can be faster with 1-4)')
    parser.add_argument('--method', choices=('abmil', 'attention_mean'), default='abmil')
    parser.add_argument('--mean-weight', type=float, default=0.5,
                        help='Mean contribution for attention_mean only, between 0 and 1')
    args = parser.parse_args()
    if not math.isfinite(args.mean_weight) or not 0 <= args.mean_weight <= 1:
        parser.error('--mean-weight must be between 0 and 1')
    if args.threads is not None:
        if args.threads < 1:
            parser.error('--threads must be positive')
        torch.set_num_threads(args.threads)
    # Resolve user paths before changing the working directory.
    output = args.output.resolve()
    data = args.data.resolve() if args.data else None
    root = Path(__file__).resolve().parent
    os.chdir(root)
    conf = load_config(root / 'config' / (CONFIGS[args.dataset] + '_medical_ssl_config.yml'))
    conf.method = args.method
    conf.hybrid_mean_weight = args.mean_weight if args.method == 'attention_mean' else 0.0
    if data:
        conf.data_dir = str(data)
    epochs = args.epochs if args.epochs is not None else int(conf.train_epoch)
    if epochs < 1:
        parser.error('--epochs must be positive')
    if not Path(conf.data_dir).is_file():
        parser.error(f'Features missing: {conf.data_dir}. See START_HERE.md.')
    seed_everything(int(conf.seed))
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)
    conf.D_feat = infer_feature_dim(train_data)
    for name, ds in zip(('train', 'validation', 'test'), (train_data, val_data, test_data)):
        if not len(ds):
            raise ValueError(f'Empty {name} set')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(batch_size=int(conf.B), num_workers=int(conf.n_worker),
                  pin_memory=bool(conf.pin_memory) and device.type == 'cuda', collate_fn=mil_collate_fn)
    train_loader = DataLoader(train_data, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **kwargs)
    # Fail rather than overwrite an earlier experiment.
    output.mkdir(parents=True, exist_ok=False)
    splits = {name: ds.data_names for name, ds in zip(('train','val','test'), (train_data,val_data,test_data))}
    (output/'splits.json').write_text(json.dumps(splits, indent=2))
    (output/'config.json').write_text(json.dumps(vars(conf), indent=2))
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
    source_hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                     for name in ('train.py', 'model.py', 'model_hybrid.py', 'engine.py', 'dataloader.py', 'utils.py')}
    (output/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        executable=sys.executable, torch=torch.__version__, device=str(device), epochs=epochs,
        cpu_threads=torch.get_num_threads(), command=sys.argv, packages=packages,
        source_sha256=source_hashes), indent=2))
    model_class = ABMIL if args.method == 'abmil' else AttentionMeanMIL
    model = model_class(conf).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))
    best, best_epoch, history = -math.inf, 0, []
    print('Device:', device, '| splits:', {k:len(v) for k,v in splits.items()}, flush=True)
    for epoch in range(1, epochs+1):
        start = time.time()
        tr = train_one_epoch(model, criterion, train_loader, optimizer, device)
        va = evaluate(model, criterion, val_loader, device, conf)
        score = va['auc'] + va['f1']
        if not math.isfinite(score):
            raise ValueError('Invalid validation selection score')
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), output/'best_state.pt')
        history.append(dict(epoch=epoch, **{'train_'+k:v for k,v in tr.items()},
                            **{'val_'+k:v for k,v in va.items()}))
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch}/{epochs}: train={tr} validation={va} best={best_epoch} seconds={time.time()-start:.1f}', flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('train_loss','val_loss'):
        plt.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    plt.xlabel('Epoch'); plt.ylabel('Cross entropy'); plt.legend(); plt.tight_layout()
    plt.savefig(output/'loss_curves.png', dpi=160); plt.close()
    summary = dict(best_epoch=best_epoch, best_validation_score=best)
    if args.test:
        model.load_state_dict(torch.load(output/'best_state.pt', map_location=device, weights_only=True))
        summary['test'] = evaluate(model, criterion, test_loader, device, conf)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)

if __name__ == '__main__':
    main()
