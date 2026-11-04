"""Export validation-only research diagnostics; never loads test feature bags."""
import argparse,json,hashlib,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from utils import Struct,classification_metrics
from dataloader import build_HDF5_feat_dataset,mil_collate_fn
from research.train_extension import build_model

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def analyze(run_dir,output,threads=2):
    run_dir=Path(run_dir).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if threads<1:raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    conf=Struct(**read(run_dir/'config.json'));env=read(run_dir/'environment.json')
    history=read(run_dir/'history.json');summary=read(run_dir/'summary.json');splits=read(run_dir/'splits.json')
    if len(history)!=env['epochs'] or [x['epoch'] for x in history]!=list(range(1,env['epochs']+1)):
        raise ValueError('Research training is incomplete')
    scores=[r['val_auc']+r['val_f1'] for r in history]
    if not all(math.isfinite(s) for s in scores):raise ValueError('Nonfinite validation score')
    best=int(np.argmax(scores))+1
    if best!=summary['best_epoch'] or not math.isclose(scores[best-1],summary['best_validation_score'],abs_tol=1e-7):
        raise ValueError('Checkpoint selection mismatch')
    for name,sha in env['source_sha256'].items():
        if digest(ROOT/name)!=sha or digest(run_dir/'source_snapshot'/name)!=sha:
            raise ValueError(f'Source or snapshot mismatch: {name}')
    for key,ids in splits.items():
        if not ids or len(ids)!=len(set(ids)):raise ValueError(f'Invalid {key} IDs')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(splits[a])&set(splits[b]):raise ValueError('Slide split overlap')
    for key in ['data_dir','dataset_csv_dir','camelyon17_csv','bracs_csv']:
        if hasattr(conf,key) and not Path(getattr(conf,key)).is_absolute():
            setattr(conf,key,str(ROOT/getattr(conf,key)))
    datasets=build_HDF5_feat_dataset(conf.data_dir,conf)
    for key,ds in zip(['train','val','test'],datasets):
        if ds.data_names!=splits[key]:raise ValueError('Saved splits changed')
    if datasets[1].feature_dim!=int(conf.D_feat):raise ValueError('Feature dimension changed')
    model=build_model(conf).eval();checkpoint=run_dir/'best_state.pt';sha=digest(checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    loader=DataLoader(datasets[1],batch_size=int(conf.B),shuffle=False,num_workers=0,collate_fn=mil_collate_fn)
    frames=[];ys=[];ps=[];total_loss=0.
    with torch.no_grad():
        for batch in loader:
            x=batch['input'].float();labels=batch['label'].long()
            diagnostics=None
            if conf.method=='fragility':logits,diagnostics=model(x,mask=batch['mask'],return_diagnostics=True)
            else:logits=model(x,mask=batch['mask'])
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
            probs=logits.softmax(1);ps.append(probs);ys.append(labels)
            total_loss+=float(torch.nn.functional.cross_entropy(logits,labels,reduction='sum'))
            for i,slide in enumerate(batch['slide_id']):
                row=dict(slide_id=slide,label=int(labels[i]),predicted=int(probs[i].argmax()),
                    valid_patches=int(batch['mask'][i].sum()))
                row['correct']=row['label']==row['predicted']
                for c in range(conf.n_class):row[f'probability_class_{c}']=float(probs[i,c])
                if diagnostics:
                    for key in ['alpha','removed_attention_mass','removed_patch_equivalents']:
                        row[key]=float(diagnostics[key][i])
                    row['attention_only_predicted']=int(diagnostics['attention_probabilities'][i].argmax())
                    row['probe_predicted']=int(diagnostics['probe_probabilities'][i].argmax())
                    for c in range(conf.n_class):
                        row[f'attention_probability_class_{c}']=float(diagnostics['attention_probabilities'][i,c])
                        row[f'probe_probability_class_{c}']=float(diagnostics['probe_probabilities'][i,c])
                frames.append(row)
    labels=torch.cat(ys);probs=torch.cat(ps)
    if labels.unique().numel()!=int(conf.n_class):raise ValueError('Validation lacks a configured class')
    metrics=classification_metrics(probs,labels,conf.n_class);metrics['loss']=total_loss/len(labels)
    independent=dict(acc=100*accuracy_score(labels.numpy(),probs.argmax(1).numpy()),
        f1=f1_score(labels.numpy(),probs.argmax(1).numpy(),average='macro'),
        auc=np.mean([roc_auc_score(labels.numpy()==c,probs[:,c].numpy()) for c in range(conf.n_class)]))
    for k,value in independent.items():np.testing.assert_allclose(metrics[k],value,atol=1e-5,rtol=0)
    frame=pd.DataFrame(frames)
    if frame.slide_id.tolist()!=splits['val']:raise ValueError('Prediction order mismatch')
    if digest(checkpoint)!=sha:raise ValueError('Checkpoint changed during diagnostic evaluation')
    result=dict(kind='validation diagnostics; not held-out performance',dataset=conf.dataset,method=conf.method,
        epochs=env['epochs'],best_epoch=best,benchmark=False,test_evaluated=False,metrics=metrics,
        checkpoint_sha256=sha,analyzer_sha256=digest(__file__),validation_slides=len(frame),
        source_sha256=env['source_sha256'],metric_crosscheck='passed against scikit-learn')
    if conf.method=='fragility':
        result['alpha_summary']={k:float(v) for k,v in frame.alpha.describe().to_dict().items()}
        result['attention_to_hybrid_prediction_flips']=int((frame.predicted!=frame.attention_only_predicted).sum())
        result['mean_alpha_by_correctness']={str(k):float(v) for k,v in frame.groupby('correct').alpha.mean().items()}
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    (output/'analyzer_source.py').write_bytes(Path(__file__).read_bytes())
    frame.to_csv(output/'predictions.csv',index=False,float_format='%.10g')
    if conf.method=='fragility':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        axs[0].hist(frame.alpha,bins=12,color='#156a9a');axs[0].set(xlabel='Adaptive mean weight alpha',ylabel='Validation slides')
        for correct,color in [(False,'#bd5e25'),(True,'#43815a')]:
            sub=frame[frame.correct==correct]
            axs[1].scatter(sub.valid_patches,sub.alpha,label='Correct' if correct else 'Incorrect',color=color,alpha=.75)
        axs[1].set(xlabel='Valid patches (log scale)',ylabel='Adaptive mean weight alpha',xscale='log');axs[1].legend()
        fig.suptitle(f'{conf.dataset}: {env["epochs"]}-epoch validation diagnostic, not a benchmark')
        fig.tight_layout();fig.savefig(output/'gate_diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();analyze(a.run,a.output,a.threads)
if __name__=='__main__':main()
