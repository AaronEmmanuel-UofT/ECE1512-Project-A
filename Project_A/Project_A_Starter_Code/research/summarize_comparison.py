"""Audit saved predictions and draw the full comparison without model inference."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/full_comparison_20261102'
DATASETS = ('C16', 'C17', 'BRACS')
METHODS = ('ABMIL', 'Fixed 50/50', 'Mean', 'Fragility')
COLORS = dict(zip(METHODS, ('#444444', '#d58a16', '#8f6b9b', '#146a9c')))


def read(path):
    return json.loads(path.read_text())


def audit(run):
    record = read(run / 'heldout_test/metrics.json')
    frame = pd.read_csv(run / 'heldout_test/predictions.csv')
    splits = read(run / 'splits.json')
    history = read(run / 'history.json')
    selected = max(history, key=lambda x: x['val_auc'] + x['val_f1'])
    assert len(history) == 30 and selected['epoch'] == record['best_epoch']
    assert hashlib.sha256((run / 'best_state.pt').read_bytes()).hexdigest() == record['checkpoint_sha256']
    for name, expected in record['training_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    assert frame.slide_id.tolist() == splits['test'] and frame.slide_id.is_unique
    n = int(record['class_count'])
    probabilities = frame[[f'probability_class_{c}' for c in range(n)]].to_numpy()
    labels = frame.label.to_numpy()
    predicted = probabilities.argmax(1)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=2e-6)
    assert np.array_equal(predicted, frame.predicted)
    metrics = dict(acc=100*accuracy_score(labels, predicted),
                   auc=np.mean([roc_auc_score(labels == c, probabilities[:, c]) for c in range(n)]),
                   f1=f1_score(labels, predicted, average='macro', zero_division=0))
    for key, value in metrics.items():
        np.testing.assert_allclose(value, record['metrics'][key], atol=1e-5, rtol=0)
    # CSV rounding can underflow extremely small probabilities; reproduce CE
    # only where saved probabilities retain enough information.
    positive = probabilities[np.arange(len(labels)), labels]
    if np.all(positive > 0):
        ce = float(-np.log(positive).mean())
        np.testing.assert_allclose(ce, record['metrics']['loss'], atol=1e-4, rtol=0)
    return record, frame, confusion_matrix(labels, predicted, labels=range(n))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=OUT)
    OUT=parser.parse_args().output_root.resolve()
    assert read(OUT / 'status.json')['stage'] == 'Full extension and mean-control experiments complete'
    rows, per_class, gate_summary, audit_record = [], [], [], {}
    base, frag, matrices = {}, {}, {}
    plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False})
    for dataset in DATASETS:
        baseline = ROOT / f'results/{dataset}_baseline_windows'
        reference = read(baseline / 'splits.json')
        audit_record[dataset] = {}
        for method, run in [('ABMIL', baseline), ('Fragility', OUT / f'{dataset}_fragility'),
                            ('Mean', OUT / f'{dataset}_mean')]:
            result, frame, matrix = audit(run)
            assert read(run / 'splits.json') == reference
            rows.append(dict(dataset=dataset, method=method, best_epoch=result['best_epoch'],
                             test_slides=result['test_slides'], **result['metrics']))
            audit_record[dataset][method] = dict(checkpoint_sha256=result['checkpoint_sha256'],
                                                predictions_sha256=hashlib.sha256((run / 'heldout_test/predictions.csv').read_bytes()).hexdigest(),
                                                metrics='independently verified', exact_splits=True)
            if method == 'ABMIL':
                base[dataset] = result
            if method == 'Fragility':
                frag[dataset] = result
                matrices[dataset] = matrix
                gate = pd.read_csv(run / 'heldout_test/gate_diagnostics.csv')
                assert gate.slide_id.tolist() == frame.slide_id.tolist()
                assert np.all((gate.alpha >= 0) & (gate.alpha <= 1))
                ac = [f'attention_probability_class_{c}' for c in range(result['class_count'])]
                pc = [f'probe_probability_class_{c}' for c in range(result['class_count'])]
                recomputed = .5*np.abs(gate[ac].to_numpy()-gate[pc].to_numpy()).sum(1)
                # Torch sums float32 class differences; CSV recomputation uses
                # float64. Allow the class-count-scaled float32 rounding bound.
                tolerance=result['class_count']*np.finfo(np.float32).eps
                np.testing.assert_allclose(gate.alpha, recomputed, atol=tolerance, rtol=0)
                audit_record[dataset][method]['maximum_gate_recomputation_error']=float(np.max(np.abs(gate.alpha-recomputed)))
                merged = gate.merge(frame, on='slide_id', validate='one_to_one', sort=False)
                merged['correct'] = merged.label == merged.predicted
                merged['attention_predicted'] = merged[ac].to_numpy().argmax(1)
                merged.to_csv(run / 'heldout_test/diagnostics_joined.csv', index=False)
                incorrect = merged.loc[~merged.correct, 'alpha']
                correct = merged.loc[merged.correct, 'alpha']
                gate_summary.append(dict(dataset=dataset, test_slides=len(gate),
                    mean_alpha=float(gate.alpha.mean()), median_alpha=float(gate.alpha.median()),
                    minimum=float(gate.alpha.min()), maximum=float(gate.alpha.max()),
                    mean_suppressed_mass=float(gate.removed_attention_mass.mean()),
                    incorrect_slides=len(incorrect), correct_slides=len(correct),
                    mean_alpha_incorrect=float(incorrect.mean()), mean_alpha_correct=float(correct.mean()),
                    attention_to_mixture_flips=int((merged.attention_predicted != merged.predicted).sum()),
                    introduced_errors=int(((merged.attention_predicted==merged.label)&(~merged.correct)).sum()),
                    repaired_errors=int(((merged.attention_predicted!=merged.label)&merged.correct).sum()),
                    patch_alpha_spearman=float(gate.valid_patches.corr(gate.alpha, method='spearman'))))
        fixed_run = ROOT / f'results/MAC_{dataset}_attention_mean_seed42'
        fixed = read(ROOT.parent / f'macbook/held_out/{dataset}_attention_mean.json')
        assert read(fixed_run / 'splits.json') == reference
        assert hashlib.sha256((fixed_run / 'best_state.pt').read_bytes()).hexdigest() == fixed['checkpoint_sha256']
        assert len(read(fixed_run / 'history.json')) == 30
        assert read(fixed_run / 'config.json')['hybrid_mean_weight'] == .5
        assert fixed['test_slide_ids'] == reference['test']
        for name,expected in fixed['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
        rows.append(dict(dataset=dataset, method='Fixed 50/50', best_epoch=fixed['best_epoch'],
                         test_slides=fixed['test_slides'], **fixed['test']))
        audit_record[dataset]['Fixed 50/50'] = dict(checkpoint_sha256=fixed['checkpoint_sha256'],
            exact_splits=True, limitation='Original aggregate held-out record; per-slide probabilities not retained')
        b = pd.read_csv(baseline / 'heldout_test/per_class_metrics.csv')
        f = pd.read_csv(OUT / f'{dataset}_fragility/heldout_test/per_class_metrics.csv')
        assert b.support.tolist() == f.support.tolist()
        for _, r in b.merge(f, on='class_id', suffixes=('_base', '_frag'), validate='one_to_one').iterrows():
            per_class.append(dict(dataset=dataset, class_id=int(r.class_id), support=int(r.support_base),
                baseline_recall=r.recall_base, fragility_recall=r.recall_frag,
                baseline_f1=r.f1_base, fragility_f1=r.f1_frag))
    comparison = pd.DataFrame(rows)
    comparison['method'] = pd.Categorical(comparison.method, METHODS, ordered=True)
    comparison['dataset'] = pd.Categorical(comparison.dataset, DATASETS, ordered=True)
    comparison = comparison.sort_values(['dataset', 'method'])
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    pd.DataFrame(per_class).to_csv(OUT / 'per_class_comparison.csv', index=False)
    pd.DataFrame(gate_summary).to_csv(OUT / 'gate_summary.csv', index=False)
    (OUT / 'comparison_audit.json').write_text(json.dumps(audit_record, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, dataset in zip(axes, DATASETS):
        for method, run in [('ABMIL', ROOT/f'results/{dataset}_baseline_windows'),
                            ('Fragility', OUT/f'{dataset}_fragility')]:
            history = read(run/'history.json')
            for field, style in [('train_loss', '-'), ('val_loss', '--')]:
                ax.plot([r['epoch'] for r in history], [r[field] for r in history],
                        color=COLORS[method], ls=style, label=f'{method} {"train" if field=="train_loss" else "validation"}')
        ax.set(title=dataset, xlabel='Epoch', ylabel='Cross entropy', xlim=(1,30))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/'full_learning_curves.png', dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = comparison[comparison.dataset == dataset].set_index('method').loc[list(METHODS)]
        bars = ax.bar(range(4), subset.f1, color=[COLORS[m] for m in METHODS], edgecolor='#222222', linewidth=.5)
        for bar, value in zip(bars, subset.f1):
            ax.text(bar.get_x()+bar.get_width()/2, value+.018, f'{value:.3f}', ha='center', fontsize=12)
        ax.set(title=f'{dataset} (test n={int(subset.test_slides.iloc[0])})', ylim=(0,1.04),
               xticks=range(4), xticklabels=['ABMIL','Fixed\n50/50','Mean','Fragility'])
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Held-out macro F1')
    fig.tight_layout(); fig.savefig(OUT/'pooling_controls.png', dpi=220); plt.close(fig)

    fig=plt.figure(figsize=(8,7.2))
    grid=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    for ax, dataset in zip(axes, DATASETS):
        matrix = matrices[dataset]; normalized = matrix/matrix.sum(1, keepdims=True)
        ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=11,
                        color='white' if normalized[i,j]>.55 else '#222222')
        ax.set(title=dataset, xlabel='Predicted class ID', ylabel='True class ID',
               xticks=range(len(matrix)), yticks=range(len(matrix)))
    fig.tight_layout(); fig.savefig(OUT/'fragility_confusion.png', dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(13,3.7),sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        data=pd.read_csv(OUT/f'{dataset}_fragility/heldout_test/diagnostics_joined.csv')
        for correct, color, marker in [(False,'#b56f18','x'), (True,'#146a9c','o')]:
            sub=data[data.correct==correct]
            ax.scatter(sub.valid_patches,sub.alpha,label='Correct' if correct else 'Incorrect',
                       color=color,marker=marker,alpha=.75,s=25)
        ax.set(title=dataset,xscale='log',xlabel='Valid patches (log scale)',ylim=(0,1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Adaptive mean weight alpha');axes[0].legend(fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'heldout_gate_diagnostics.png',dpi=220);plt.close(fig)
    print(comparison.to_string(index=False))
    print(pd.DataFrame(gate_summary).to_string(index=False))


if __name__ == '__main__':
    main()
