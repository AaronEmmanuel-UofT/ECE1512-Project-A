"""Read-only feature/split audit before an expensive training run.

No model fitting or prediction. --scan-features additionally reads every feature
value in bounded chunks to detect NaN/Inf (can take minutes on large files).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import os
import h5py
import numpy as np
import pandas as pd
from dataloader import build_HDF5_feat_dataset, _map_label
from utils import load_config

CONFIGS = {'C16': 'camelyon16', 'C17': 'camelyon17', 'BRACS': 'bracs'}


def audit(path, conf, scan_features=False):
    datasets = build_HDF5_feat_dataset(str(path), conf)
    split_names = ('train', 'val', 'test')
    splits = dict(zip(split_names, datasets))
    errors, warnings, report = [], [], {}
    ids = {key: set(ds.data_names) for key, ds in splits.items()}
    for i, left in enumerate(split_names):
        for right in split_names[i+1:]:
            overlap = ids[left] & ids[right]
            if overlap:
                errors.append(f'{left}/{right}: {len(overlap)} overlapping slide IDs')
    dimensions, patch_counts = set(), []
    with h5py.File(path, 'r') as h5:
        for key, ds in splits.items():
            counts = Counter()
            for name in ds.data_names:
                group = h5[name]
                try:
                    feat = group['feat']
                    if len(feat.shape) != 2 or min(feat.shape) == 0:
                        raise ValueError(f'invalid feature shape {feat.shape}')
                    if not np.issubdtype(feat.dtype, np.number):
                        raise ValueError(f'non-numeric features: {feat.dtype}')
                    dimensions.add(int(feat.shape[1]))
                    patch_counts.append(int(feat.shape[0]))
                    label = _map_label(int(group.attrs['label']), conf.dataset, int(conf.n_class))
                    if not 0 <= label < int(conf.n_class):
                        raise ValueError(f'label {label} out of range')
                    counts[label] += 1
                    if 'coords' in group and group['coords'].shape != (feat.shape[0], 2):
                        warnings.append(f'{name}: coordinates do not have shape [patches,2]')
                    if scan_features:
                        for start in range(0, feat.shape[0], 4096):
                            if not np.isfinite(feat[start:start+4096]).all():
                                raise ValueError('nonfinite features')
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f'{name}: {exc}')
            missing = sorted(set(range(int(conf.n_class))) - set(counts))
            if missing:
                errors.append(f'{key}: missing classes {missing}')
            report[key] = {'slides': len(ds), 'class_counts': dict(sorted(counts.items()))}
    if len(dimensions) != 1:
        errors.append(f'Expected one feature dimension, got {sorted(dimensions)}')
    if str(conf.dataset).upper() == 'CAMELYON17':
        csv_path = getattr(conf, 'camelyon17_csv',
                           str(Path(getattr(conf, 'dataset_csv_dir', 'dataset_csv'))/'camelyon17.csv'))
        metadata = pd.read_csv(csv_path).set_index('slide_id')
        if 'case_id' in metadata.columns:
            cases = {key: set(metadata.loc[ds.data_names, 'case_id'].astype(str))
                     for key, ds in splits.items()}
            report['patient_overlap'] = {}
            for i, left in enumerate(split_names):
                for right in split_names[i+1:]:
                    common = sorted(cases[left] & cases[right])
                    report['patient_overlap'][f'{left}/{right}'] = common
                    if common:
                        warnings.append(f'{left}/{right}: {len(common)} shared patients; '
                                        'do not claim patient-independent evaluation')
        else:
            warnings.append('No case_id column; patient overlap was not checked')
    return dict(dataset=conf.dataset, file=str(Path(path).resolve()),
                file_bytes=Path(path).stat().st_size, splits=report,
                feature_dimensions=sorted(dimensions),
                patches_min=min(patch_counts) if patch_counts else None,
                patches_max=max(patch_counts) if patch_counts else None,
                all_feature_values_scanned=scan_features,
                errors=errors, warnings=warnings, passed=not errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=CONFIGS, default='C16')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--scan-features', action='store_true')
    parser.add_argument('--output', type=Path, help='New JSON report; refuses overwrite')
    args = parser.parse_args()
    data = args.data.resolve() if args.data else None
    output = args.output.resolve() if args.output else None
    os.chdir(Path(__file__).resolve().parent)
    conf = load_config(Path('config')/(CONFIGS[args.dataset]+'_medical_ssl_config.yml'))
    path = data or Path(conf.data_dir)
    if not path.is_file():
        parser.error(f'Feature file missing: {path}. See START_HERE.md.')
    result = audit(path, conf, args.scan_features)
    content = json.dumps(result, indent=2)
    print(content)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(content+'\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
