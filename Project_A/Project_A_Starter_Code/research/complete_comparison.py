"""Run the frozen full extension comparison; keep existing results intact.

All training finishes before any held-out evaluation. Reopening this command
checks and reuses complete outputs; it never silently restarts a partial run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def verify_run(path):
    history = read(path / 'history.json')
    env = read(path / 'environment.json')
    summary = read(path / 'summary.json')
    assert env['epochs'] == 30 and len(history) == 30
    assert [row['epoch'] for row in history] == list(range(1, 31))
    selected = max(history, key=lambda row: row['val_auc'] + row['val_f1'])
    assert selected['epoch'] == summary['best_epoch']
    for name, expected in env['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256((path / 'source_snapshot' / name).read_bytes()).hexdigest() == expected, name
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        default=ROOT / 'results/full_comparison_20261102')
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'active.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', MPLCONFIGDIR=str(ROOT.parent / 'macbook/matplotlib'))
    completed = []

    def status(stage, **extra):
        record = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                      stage=stage, completed=completed.copy(), **extra)
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(out / 'status.json')
        print(f'\n{stage}', flush=True)

    def command(arguments, logfile):
        print(' '.join(map(str, arguments)), flush=True)
        started = time.monotonic()
        with logfile.open('a') as stream:
            process = subprocess.Popen(list(map(str, arguments)), cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait() != 0:
                    raise RuntimeError(f'Command failed; see {logfile}')
            except BaseException:
                process.terminate()
                process.wait()
                raise
        return time.monotonic() - started

    try:
        jobs = [(dataset, method) for method in ('fragility', 'mean')
                for dataset in ('C16', 'C17', 'BRACS')]
        for dataset, method in jobs:
            run = out / f'{dataset}_{method}'
            if not run.exists():
                status(f'Training {dataset} {method}: 30 epochs', current_run=str(run))
                seconds = command([sys.executable, '-u', ROOT / 'research/train_extension.py',
                                   '--dataset', dataset, '--method', method, '--epochs', '30',
                                   '--threads', '2', '--output', run], out / f'{dataset}_{method}.log')
                (run / 'wall_time.json').write_text(json.dumps(dict(training_seconds=seconds), indent=2))
            verify_run(run)
            reference = ROOT / f'results/{dataset}_baseline_windows/splits.json'
            assert read(run / 'splits.json') == read(reference)
            completed.append(f'{dataset} {method}: full training verified')
        status('All six training runs complete; settings fixed before held-out evaluation')
        for dataset, method in jobs:
            run = out / f'{dataset}_{method}'
            evaluation = run / 'heldout_test'
            if not evaluation.exists():
                status(f'Evaluating held-out {dataset} {method}', current_run=str(run))
                command([sys.executable, '-u', ROOT / 'research/evaluate_extension.py',
                         '--run', run, '--output', evaluation, '--threads', '2'],
                        out / f'{dataset}_{method}_evaluation.log')
            assert read(evaluation / 'metrics.json')['status'] == 'complete'
            verify_run(run)
            completed.append(f'{dataset} {method}: held-out evaluation saved')
        status('Full extension and mean-control experiments complete')
    except BaseException as exc:
        status('Stopped; inspect the recorded error before continuing', error=str(exc))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
