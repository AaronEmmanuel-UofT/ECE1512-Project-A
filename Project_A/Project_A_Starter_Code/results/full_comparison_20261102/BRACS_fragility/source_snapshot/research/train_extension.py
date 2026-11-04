"""Train a research extension or control without modifying baseline source.

Validation only: no test inference is available through this training command.
"""
import argparse
from pathlib import Path
import sys, os, json, math, hashlib, platform, importlib.metadata, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import ABMIL
from model_hybrid import AttentionMeanMIL
from model_fragility import FragilityGuidedMIL
from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from engine import train_one_epoch, evaluate
from utils import load_config, seed_everything, infer_feature_dim

CONFIGS={'C16':'camelyon16','C17':'camelyon17','BRACS':'bracs'}
SOURCES=['model.py','model_hybrid.py','model_fragility.py','engine.py','dataloader.py','utils.py','research/train_extension.py']

def build_model(conf):
    return {'abmil':ABMIL,'attention_mean':AttentionMeanMIL,'mean':AttentionMeanMIL,
            'fragility':FragilityGuidedMIL}[conf.method](conf)

def run(dataset,output,method='fragility',epochs=30,data=None,threads=2,seed=42):
    if method not in ('abmil','attention_mean','mean','fragility'):
        raise ValueError('Unknown research method')
    if epochs<1 or threads<1:
        raise ValueError('epochs and threads must be positive')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    data=Path(data).resolve() if data else None
    os.chdir(ROOT);torch.set_num_threads(threads)
    conf=load_config(ROOT/'config'/f'{CONFIGS[dataset]}_medical_ssl_config.yml')
    conf.method=method;conf.seed=seed;conf.probe_fraction=0.1
    conf.hybrid_mean_weight=1.0 if method=='mean' else 0.5
    if data:conf.data_dir=str(data)
    seed_everything(seed)
    datasets=build_HDF5_feat_dataset(conf.data_dir,conf)
    splits={key:ds.data_names for key,ds in zip(('train','val','test'),datasets)}
    for key,names in splits.items():
        if not names or len(names)!=len(set(names)):raise ValueError(f'Empty/duplicate {key}')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(splits[a])&set(splits[b]):raise ValueError('Split overlap')
    original=ROOT/f'results/{dataset}_baseline_windows/splits.json'
    # Synthetic fixtures deliberately use different IDs; real default data must match.
    matched=None
    if data is None:
        matched=splits==json.loads(original.read_text())
        if not matched:raise ValueError('Research splits differ from the completed baseline')
    conf.D_feat=infer_feature_dim(datasets[0])
    loaders=[DataLoader(ds,batch_size=int(conf.B),shuffle=(i==0),num_workers=0,
                        collate_fn=mil_collate_fn) for i,ds in enumerate(datasets[:2])]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(conf).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(conf.lr),weight_decay=float(conf.wd))
    criterion=nn.CrossEntropyLoss()
    output.mkdir(parents=True,exist_ok=False)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    for name in SOURCES:
        dest=output/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes((ROOT/name).read_bytes())
    environment=dict(python=platform.python_version(),torch=torch.__version__,executable=sys.executable,
        device=str(device),cpu_threads=threads,epochs=epochs,command=sys.argv,
        source_sha256=hashes,packages={x.metadata['Name']:x.version for x in importlib.metadata.distributions()})
    for name,obj in [('config',vars(conf)),('splits',splits),('environment',environment)]:
        (output/f'{name}.json').write_text(json.dumps(obj,indent=2))
    best=-math.inf;best_epoch=0;history=[]
    print(f'Research {dataset} {method}: epochs={epochs}, matched_baseline_splits={matched}; no test inference',flush=True)
    for epoch in range(1,epochs+1):
        start=time.monotonic()
        tr=train_one_epoch(model,criterion,loaders[0],optimizer,device)
        va=evaluate(model,criterion,loaders[1],device,conf)
        score=va['auc']+va['f1']
        if not math.isfinite(score):raise ValueError('Nonfinite selection score')
        if score>best:
            best,best_epoch=score,epoch;torch.save(model.state_dict(),output/'best_state.pt')
        history.append(dict(epoch=epoch,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()}))
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(f'Epoch {epoch}/{epochs}: train={tr} validation={va} best={best_epoch} seconds={time.monotonic()-start:.1f}',flush=True)
    summary=dict(best_epoch=best_epoch,best_validation_score=best,kind='full research run' if epochs==30 else 'debug only; not benchmark',
        matched_baseline_splits=matched,test_evaluated=False,method=method)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for key in ['train_loss','val_loss']:ax.plot(range(1,epochs+1),[r[key] for r in history],label=key)
    ax.axvline(best_epoch,color='gray',ls='--',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='Cross entropy',title=f'{dataset}: {method}');ax.legend();fig.tight_layout()
    fig.savefig(output/'loss_curves.png',dpi=160);plt.close(fig)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=CONFIGS,required=True)
    p.add_argument('--method',choices=['abmil','attention_mean','mean','fragility'],default='fragility')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data',type=Path)
    a=p.parse_args();print(json.dumps(run(a.dataset,a.output,a.method,a.epochs,a.data,a.threads,a.seed),indent=2))
if __name__=='__main__':main()
