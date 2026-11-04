"""Record the initial development checks, never a benchmark comparison."""
from pathlib import Path
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    datasets=['C16','C17','BRACS'];rows=[];checks={};groups=[]
    log=(ROOT/'results/fragility_software_checks_20261030.log').read_text(encoding='utf-8-sig')
    if 'Ran 7 tests' not in log or '\nOK' not in log:raise ValueError('No complete passing software test log')
    for d in datasets:
        base=ROOT/f'results/{d}_baseline_windows';run=ROOT/f'results/{d}_fragility_debug_20261030'
        baseline_env=read(base/'environment.json')
        for name,sha in baseline_env['source_sha256'].items():assert digest(ROOT/name)==sha
        assert read(base/'splits.json')==read(run/'splits.json')
        s=read(run/'summary.json');history=read(run/'history.json')
        assert len(history)==2 and s['test_evaluated'] is False and 'test' not in s
        assert not (run/'heldout_test').exists()
        selected=history[s['best_epoch']-1];diag=read(run/'validation_diagnostics/summary.json')
        assert diag['test_evaluated'] is False
        assert diag['checkpoint_sha256']==digest(run/'best_state.pt')
        for name,sha in read(run/'environment.json')['source_sha256'].items():
            assert digest(ROOT/name)==sha and digest(run/'source_snapshot'/name)==sha
        for key in ['auc','acc','f1','loss']:assert abs(diag['metrics'][key]-selected['val_'+key])<1e-6
        row=dict(dataset=d,epochs=2,selected_epoch=s['best_epoch'],validation_accuracy=diag['metrics']['acc'],
            validation_auc=diag['metrics']['auc'],validation_macro_f1=diag['metrics']['f1'],
            mean_alpha=diag['alpha_summary']['mean'],median_alpha=diag['alpha_summary']['50%'],
            max_alpha=diag['alpha_summary']['max'],training_loss_first=history[0]['train_loss'],training_loss_last=history[-1]['train_loss'])
        rows.append(row)
        p=run/'validation_diagnostics/predictions.csv'
        for r in csv.DictReader(p.open()):
            r['dataset']=d;groups.append(r)
        checks[d]={'original_core_sources_unchanged':True,'exact_baseline_split_ids':True,
            'two_debug_epochs_complete':True,'validation_metrics_reproduced':True,'no_real_test_evaluation':True,
            'source_snapshot_verified':True}
    data=dict(status='PASS',kind='research development pilot; not a benchmark',date_toronto='2026-10-30',
        tests=7,test_log_sha256=digest(ROOT/'results/fragility_software_checks_20261030.log'),datasets=checks,rows=rows)
    (ROOT/'results/fragility_development_20261030.json').write_text(json.dumps(data,indent=2))
    lines=['# Extension development status','', 'Updated October 30 2026. The selected direction is [fragility-guided adaptive pooling](PROTOCOL.md).',
        '', '## Completed', '', '- Implemented the parameter-matched model, symmetric top-fraction probe and detached slide-specific gate.',
        '- Added a separate training runner with ABMIL, fixed 50/50, pure mean and fragility controls.',
        '- Added validation diagnostics and a guarded full-run held-out evaluator, preserving the original baseline code.',
        '- Passed seven tests, including end-to-end synthetic training across all three dataset formats, evaluator artifacts, source archive checks and debug/duplicate evaluation rejection.',
        '- Completed real two-epoch debug training on C16, C17 and BRACS with exact matching baseline split IDs.',
        '- Exported validation-only per-slide gate diagnostics and independently checked accuracy, AUROC and macro F1 against scikit-learn.',
        '', '## Debug results only', '', 'These are validation metrics at the selected epoch within a two-epoch software pilot. They are not the final extension results, must not be compared to the full baseline held-out table, and do not establish improvement. No real extension test predictions were generated.',
        '', '| Dataset | Selected debug epoch | Validation accuracy % | Validation macro AUROC | Validation macro F1 | Mean alpha | Median alpha | Maximum alpha |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['dataset']} | {r['selected_epoch']} | {r['validation_accuracy']:.2f} | {r['validation_auc']:.4f} | {r['validation_macro_f1']:.4f} | {r['mean_alpha']:.4f} | {r['median_alpha']:.4f} | {r['max_alpha']:.4f} |")
    lines += ['', 'Alpha is the total-variation change between attention-only and suppressed-evidence class probabilities. It is the mean-pooling weight, not a calibrated uncertainty score. Its variation shows that the code adapts across slides, not that those choices are helpful.',
        '', 'C17 still has majority-dominated decisions at this early stage. On BRACS, the validation-selection score is higher at epoch 1 than epoch 2, even though training loss decreases. This pilot supports functioning code and finite optimization, not a performance claim. The gate can weaken true focal disease evidence, so clean class recall must remain central to the later comparison.',
        '', '## Diagnostic figures']
    for d in datasets:lines+=['',f'### {d}', '',f'![{d} validation gate diagnostics](../results/{d}_fragility_debug_20261030/validation_diagnostics/gate_diagnostics.png)']
    lines += ['', '## Remaining research work','',
        '1. Complete 30-epoch fragility runs on all three datasets with the frozen protocol.',
        '2. Run fixed 50/50 and pure-mean controls to distinguish adaptive allocation from uniform smoothing.',
        '3. Evaluate the selected full-run checkpoints once using the dedicated research evaluator and save comparative test tables.',
        '4. Implement common-subset random/top/low-attention perturbation analysis and interpret accuracy alongside stability.',
        '5. Add measured results and limitations to the final report; additional matched seeds if time allows.',
        '', 'Full research runs have not been launched in this development stage. The October 29 baseline PDF remains a preserved snapshot; this status and the protocol are the subsequent research record.',
        '', 'Evidence: `results/fragility_software_checks_20261030.log`, `results/fragility_development_20261030.json`, and each `results/{dataset}_fragility_debug_20261030/` directory. All original baseline core fingerprints still match their saved runs.']
    (ROOT/'research/STATUS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(data,indent=2))
if __name__=='__main__':main()
