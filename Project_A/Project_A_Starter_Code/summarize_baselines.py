"""Build baseline tables and figures from saved test predictions, without inference."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
ORDER = ('CAMELYON16', 'CAMELYON17', 'BRACS')
SHORT = dict(zip(ORDER, ('C16', 'C17', 'BRACS')))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(runs, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Report already exists: {output}')
    loaded, rows = {}, []
    for location in runs:
        run = Path(location).resolve()
        conf, env = read(run/'config.json'), read(run/'environment.json')
        summary, history = read(run/'summary.json'), read(run/'history.json')
        splits = read(run/'splits.json')
        evaluation = read(run/'heldout_test'/'metrics.json')
        frame = pd.read_csv(run/'heldout_test'/'predictions.csv')
        name = conf['dataset']
        fixed = dict(D_feat=384, D_inner=256, D_attn=128, B=1, droprate=0.0,
                     seed=42, lr=0.0001, wd=0.00001, train_epoch=30)
        if any(conf.get(key) != value for key, value in fixed.items()):
            raise ValueError('Run differs from the fixed protocol described by this report')
        if name in loaded or name not in ORDER:
            raise ValueError(f'Duplicate or unexpected dataset: {name}')
        if conf.get('method', 'abmil') != 'abmil' or len(history) != 30 or env['epochs'] != 30:
            raise ValueError('Only completed 30-epoch ABMIL baselines belong in this table')
        if evaluation['status'] != 'complete' or not evaluation['benchmark']:
            raise ValueError('Incomplete or synthetic evaluation')
        if evaluation['best_epoch'] != summary['best_epoch']:
            raise ValueError('Evaluation did not use the validation-selected checkpoint')
        if hashlib.sha256((run/'best_state.pt').read_bytes()).hexdigest() != evaluation['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after held-out evaluation')
        if frame.slide_id.tolist() != splits['test']:
            raise ValueError('Saved predictions differ from saved test slide IDs')
        p = frame[[f'probability_class_{c}' for c in range(conf['n_class'])]].to_numpy()
        y, predicted = frame.label.to_numpy(), frame.predicted.to_numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(predicted, p.argmax(axis=1))
        recomputed = dict(acc=100*accuracy_score(y, predicted),
            f1=f1_score(y, predicted, average='macro'),
            auc=np.mean([roc_auc_score(y == c, p[:, c]) for c in range(conf['n_class'])]))
        for key, value in recomputed.items():
            np.testing.assert_allclose(value, evaluation['metrics'][key], rtol=0, atol=1e-5)
        best = history[summary['best_epoch']-1]
        row = dict(dataset=SHORT[name], method='ABMIL', seed=conf['seed'], epochs=len(history),
            training_slides=len(splits['train']), validation_slides=len(splits['val']),
            test_slides=len(splits['test']), classes=conf['n_class'], best_epoch=summary['best_epoch'],
            validation_accuracy_percent=best['val_acc'], validation_macro_auc=best['val_auc'],
            validation_macro_f1=best['val_f1'], validation_loss=best['val_loss'],
            test_accuracy_percent=evaluation['metrics']['acc'], test_macro_auc=evaluation['metrics']['auc'],
            test_macro_f1=evaluation['metrics']['f1'], test_loss=evaluation['metrics']['loss'],
            test_majority_class_accuracy_percent=float(100*frame.label.value_counts().max()/len(frame)),
            run_directory=str(run))
        rows.append(row)
        loaded[name] = dict(run=run, conf=conf, env=env, history=history,
                            splits=splits, evaluation=evaluation, predictions=frame, row=row,
                            per_class=pd.read_csv(run/'heldout_test'/'per_class_metrics.csv'))
    if set(loaded) != set(ORDER):
        raise ValueError('All three datasets are required for the baseline comparison')
    rows.sort(key=lambda row: ('C16', 'C17', 'BRACS').index(row['dataset']))
    c17 = loaded['CAMELYON17']
    metadata = pd.read_csv(ROOT/'dataset_csv'/'camelyon17.csv').set_index('slide_id')
    patients = {key: set(metadata.loc[ids, 'case_id']) for key, ids in c17['splits'].items()}
    overlap = {f'{left}/{right}': len(patients[left] & patients[right])
               for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output/'baseline_comparison.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True)
    for ax, name in zip(axes, ORDER):
        info = loaded[name]
        h = info['history']
        epochs = [row['epoch'] for row in h]
        ax.plot(epochs, [row['train_loss'] for row in h], color='#2166ac', label='Training loss')
        ax.plot(epochs, [row['val_loss'] for row in h], color='#b35806', label='Validation loss')
        ax.axvline(info['row']['best_epoch'], color='#555555', linestyle='--', linewidth=1,
                   label='Selected epoch')
        ax.set(title=f"{SHORT[name]} | selected epoch {info['row']['best_epoch']}",
               xlabel='Epoch', ylabel='Cross-entropy loss', xlim=(1, 30))
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output/'learning_curves.png', dpi=240)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    lines = ['# ABMIL baseline evaluation', '',
        'Generated '+datetime.now(ZoneInfo('America/Toronto')).strftime('%B %d, %Y, %H:%M %Z')+'.', '',
        'All three baselines completed 30 epochs. Each held-out evaluation used the checkpoint selected solely by validation macro AUROC + macro F1 (earliest epoch in a tie). Test predictions were saved; this report recomputes metrics from those predictions without rerunning inference.', '',
        '| Dataset | Train / validation / test slides | Selected epoch | Test accuracy (%) | Test macro AUROC | Test macro F1 | Test loss |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['training_slides']} / {row['validation_slides']} / {row['test_slides']} | {row['best_epoch']} | {row['test_accuracy_percent']:.2f} | {row['test_macro_auc']:.4f} | {row['test_macro_f1']:.4f} | {row['test_loss']:.4f} |")
    lines += ['', '## Learning curves', '', '![Training and validation loss](learning_curves.png)', '',
        'Each panel has its own loss scale. The dashed line is the epoch selected by validation AUROC + F1, not necessarily minimum validation loss.', '',
        '## Model structure and component roles', '',
        'The frozen image encoder is represented here by the supplied patch feature vectors. It supplies visual descriptions but is not updated during MIL training. A learned linear projection with ReLU transforms each 384-number patch vector into a 256-number representation.', '',
        'Gated attention computes a score from the elementwise product of a tanh branch and a sigmoid branch, then normalises scores across valid patches with softmax. Padded patches receive zero attention. A weighted sum produces one slide representation; patch order does not affect this pooling operation.', '',
        'The linear classifier maps the pooled representation to 2, 4 or 7 class logits. Cross-entropy trains the projection, attention and classifier together from slide labels. Class softmax is used during evaluation. Attention weights are pooling contributions, not patch-level tumour probabilities, and this baseline does not explicitly model spatial relationships between patches.', '',
        '## Training procedure', '',
        '- Supplied, frozen 384-dimensional patch features; no image encoder was trained in these experiments.',
        '- Learned 384-to-256 projection with ReLU, gated attention with 128 hidden units, weighted pooling, and a linear classifier.',
        '- Slide cross-entropy loss; AdamW with learning rate 0.0001 and weight decay 0.00001; batch size 1; dropout 0; seed 42; 30 epochs.',
        '- CPU execution with two training threads. No scheduler or warmup was applied; the supplied warmup_epoch field is unused.',
        '- Original slide IDs, split logic and class mappings were preserved. Every split contains all configured classes.',
        '- Macro AUROC averages one-vs-rest class AUCs. Accuracy is a percentage; AUROC and F1 are on a 0-1 scale.', '',
        '## Selected validation results and test comparison', '',
        '| Dataset | Validation accuracy (%) | Validation AUROC | Validation macro F1 | Test majority-class accuracy (%) |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['validation_accuracy_percent']:.2f} | {row['validation_macro_auc']:.4f} | {row['validation_macro_f1']:.4f} | {row['test_majority_class_accuracy_percent']:.2f} |")
    lines += ['', 'The majority-class accuracy is a descriptive reference computed from the test labels; it was not used to select or train the model. Refer to macro F1 and the per-class confusion matrices as well as accuracy, especially for imbalanced datasets.', '',
        '## Observed behaviour', '']
    for name in ORDER:
        info = loaded[name]
        row = info['row']
        weak = info['per_class'].sort_values(['recall', 'class_id']).iloc[0]
        support = int(weak['support'])
        correct = int(round(float(weak['recall'])*support))
        last = info['history'][-1]
        lines += [f"- **{SHORT[name]}:** selected validation macro F1 was {row['validation_macro_f1']:.4f}, compared with {row['test_macro_f1']:.4f} on test. The lowest test recall was class {int(weak['class_id'])}: {100*weak['recall']:.2f}% ({correct}/{support} slides correctly classified)."]
        minimum_val = min(epoch['val_loss'] for epoch in info['history'])
        if last['val_loss'] > 1.2*minimum_val and last['train_loss'] < 0.5*info['history'][0]['train_loss']:
            lines += [f"  Later epochs show training/validation divergence: final training loss {last['train_loss']:.4f}, final validation loss {last['val_loss']:.4f}, and minimum validation loss {minimum_val:.4f}. This is consistent with overfitting in later training; the reported test result still uses the original validation-selected checkpoint, not the last epoch."]
    lines += ['', 'Class IDs follow the supplied feature labels and configured mappings. These observations identify errors and learning patterns; they do not establish their biological cause or localise lesions.', '',
        '## Limitations and split units', '',
        f"- C17 has {overlap['train/val']} patients shared between train and validation, {overlap['train/test']} between train and test, and {overlap['val/test']} between validation and test. Validation is not patient-independent. Test uses held-out centres under the supplied course split, rather than the original challenge's hidden test cohort.",
        '- Results are slide-level, with multiple slides potentially belonging to one patient. No patient-level staging performance is claimed.',
        '- One seed was run. No across-seed uncertainty, statistical significance or clinical generalisation is established.',
        '- C16 validation contains only 27 slides; perfect validation results should not be treated as proof of perfect generalisation.',
        '- The hybrid extension is excluded from this baseline table. Its real-data comparison remains a separate experiment with matching splits.', '',
        '## Reproducibility and error analysis', '']
    hardware_path = ROOT/'results'/'baseline_hardware_20261029.json'
    if hardware_path.is_file():
        hardware = json.loads(hardware_path.read_text(encoding='utf-8-sig'))
        lines += [f"Host CPU: {hardware['Name']} ({hardware['NumberOfCores']} physical cores, {hardware['NumberOfLogicalProcessors']} logical processors). Training used two CPU threads per process. All runs used Python {loaded[ORDER[0]]['env']['python']} and PyTorch {loaded[ORDER[0]]['env']['torch']}.", '']
    for name in ORDER:
        info = loaded[name]
        rel = Path(os.path.relpath(info['run'], output)).as_posix()
        lines += [f"- {SHORT[name]}: [{info['run'].name}]({rel}/), [test confusion matrix]({rel}/heldout_test/confusion_matrix.png), [per-class metrics]({rel}/heldout_test/per_class_metrics.csv), [slide probabilities]({rel}/heldout_test/predictions.csv)."]
    lines += ['', 'Each run includes its configuration, source fingerprints, dependency versions, split IDs, 30-epoch history and selected checkpoint. Each heldout_test folder records checkpoint/evaluator fingerprints, an evaluation plan, metrics, per-slide probabilities and per-class errors.', '',
        'Baseline reference: Ilse, Tomczak and Welling, [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html), ICML 2018. Implementation is based on the supplied course scaffold with documented assistance.', '']
    (output/'BASELINE_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report_manifest.json').write_text(json.dumps(dict(
        generated_local=datetime.now(ZoneInfo('America/Toronto')).isoformat(),
        runs=[str(loaded[name]['run']) for name in ORDER],
        c17_patient_overlap=overlap, metrics_recomputed_from_saved_predictions=True,
        rows=rows), indent=2), encoding='utf-8')
    print(output/'BASELINE_RESULTS.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.runs, args.output)


if __name__ == '__main__':
    main()
