"""Download the fixed feature revision and verify complete SHA-256 checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['C16', 'C17', 'BRACS', 'all'], required=True)
    args = parser.parse_args()
    for dataset in (('C16', 'C17', 'BRACS') if args.dataset=='all' else (args.dataset,)):
        record = json.loads((ROOT/f'results/feature_downloads/{dataset}.json').read_text())
        target = ROOT/f'feature/{dataset}/patch_feats_pretrain_medical_ssl.h5'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != record['bytes'] or digest(target) != record['sha256']:
                raise ValueError(f'Existing file differs from the documented artifact: {target}')
            print(f'{dataset}: existing file verified', flush=True)
            continue
        temporary = target.with_suffix('.download')
        if temporary.exists():
            raise FileExistsError(f'Incomplete download already exists: {temporary}')
        print(f'{dataset}: downloading {record["bytes"] / 1e9:.2f} GB', flush=True)
        with urllib.request.urlopen(record['source'], timeout=120) as response, temporary.open('xb') as stream:
            while True:
                block = response.read(8*1024*1024)
                if not block:
                    break
                stream.write(block)
        if temporary.stat().st_size != record['bytes'] or digest(temporary) != record['sha256']:
            raise ValueError(f'Download failed checksum verification: {temporary}')
        temporary.rename(target)
        print(f'{dataset}: SHA-256 verified', flush=True)


if __name__=='__main__':
    main()
