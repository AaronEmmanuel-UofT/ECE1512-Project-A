"""Recreate the two C16 baseline attention figures from the selected checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import torch
from model import ABMIL
from utils import load_config
from wsi_visualization import visualize_wsi_attention

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Assumed level-0 patch footprint; not recorded in supplied HDF5')
    parser.add_argument('--show', action='store_true', help='Open the figures as well as saving them')
    args = parser.parse_args()
    torch.set_num_threads(2)
    conf = load_config(ROOT / 'config/camelyon16_medical_ssl_config.yml')
    conf.data_dir = str(ROOT / conf.data_dir)
    checkpoint = ROOT / 'results/C16_baseline_windows/best_state.pt'
    output = ROOT / 'results/C16_attention'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    records = visualize_wsi_attention(ABMIL(conf), conf, ROOT, state,
                                      patch_size_level0=args.patch_size,
                                      output_dir=output, show=args.show)
    metadata = {'checkpoint': 'results/C16_baseline_windows/best_state.pt',
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'selected_epoch': 3, 'patch_size_level0_assumed': args.patch_size,
                'coordinates': 'level-0 (x,y), assumed top-left patch corners',
                'color': 'within-slide attention percentile; not tumor probability',
                'slides': {}}
    for slide, record in records.items():
        metadata['slides'][slide] = {
            'patches': len(record['attention']), 'label': record['label'],
            'probabilities': record['probabilities'].tolist(),
            'attention_sum': float(record['attention'].sum()),
            'dimensions': record['dimensions'], 'annotation_contours': record['contours']}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Figures and raw attention saved in:', output)

if __name__ == '__main__':
    main()
