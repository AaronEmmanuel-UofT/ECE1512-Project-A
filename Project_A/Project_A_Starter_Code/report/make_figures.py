"""Create report figures from saved artifacts only; no model inference."""
from pathlib import Path
import json, csv, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).parent/'figures'; OUT.mkdir(exist_ok=True)
DS=['C16','C17','BRACS']; COLORS=['#156a9a','#bd5e25','#43815a']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight'); plt.close(fig)
def diagram(name,rows,title):
    fig,ax=plt.subplots(figsize=(11,max(2.3,1.32*len(rows))))
    ax.set(xlim=(0,11),ylim=(0,len(rows)*1.6+.45)); ax.axis('off')
    for j,row in enumerate(rows):
        y=(len(rows)-j-1)*1.6+.25
        w=10.5/len(row)
        for i,(head,body) in enumerate(row):
            x=.25+i*w
            ax.add_patch(FancyBboxPatch((x,y),w-.28,1.08,boxstyle='round,pad=0.08',facecolor=['#e9f2f8','#f6efe8','#edf3ed'][j%3],edgecolor='#8b9ba6'))
            ax.text(x+(w-.28)/2,y+.76,head,ha='center',va='center',weight='bold',fontsize=11)
            ax.text(x+(w-.28)/2,y+.33,body,ha='center',va='center',fontsize=9)
            if i<len(row)-1: ax.annotate('',xy=(x+w-.06,y+.53),xytext=(x+w-.25,y+.53),arrowprops={'arrowstyle':'->','color':'#374b59'})
        if j<len(rows)-1: ax.annotate('',xy=(5.5,y-.43),xytext=(5.5,y-.03),arrowprops={'arrowstyle':'->','color':'#374b59'})
    ax.set_title(title,loc='left',fontsize=14,weight='bold',pad=15);save(fig,name)
diagram('01_pipeline',[[('Whole slide','Upstream patch extraction'),('Frozen encoder','Supplied feature extraction'),('HDF5 feature bag','N patches x 384 numbers')],[('Learned projection','384 -> 256 with ReLU'),('Gated attention','Scores -> valid-patch softmax'),('Weighted pooling','One 256-number slide vector')],[('Linear classifier','256 -> C class logits'),('Class softmax','C class probabilities'),('Slide prediction','Largest probability')]],'From a slide to a prediction')
diagram('02_attention',[[('Projected patch h','256 numbers'),('Two branches','tanh(Vh+bV) and sigmoid(Uh+bU)'),('Gate','Elementwise product')],[('Score layer','128 -> 1 scalar per patch'),('Mask and softmax','Invalid score = minus infinity'),('Weighted sum','z = sum of attention x h')]],'How gated attention computes a slide representation')
diagram('03_protocol',[[('Training split','Updates learned parameters'),('Validation split','Measures each saved epoch'),('Checkpoint selection','Max validation AUROC + macro F1')],[('Selected checkpoint','Earliest epoch if tied'),('Held-out test','One evaluation per baseline'),('Saved evidence','Probabilities, errors and fingerprints')]],'Training and evaluation have different roles')
fig,ax=plt.subplots(figsize=(10,4));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
nodes={'input':(0.2,1.5,2.1,'Projected patches','Same valid features'), 'att':(3.1,2.5,2.5,'Attention pooling','z_attention = sum a_i h_i'), 'mean':(3.1,.5,2.5,'Mean pooling','z_mean = sum h_i / N'), 'mix':(6.3,1.5,3.2,'Fixed mixture and classifier','0.5 z_attention + 0.5 z_mean')}
for key,(x,y,w,head,body) in nodes.items():
    ax.add_patch(FancyBboxPatch((x,y),w,.8,boxstyle='round,pad=.08',facecolor='#e9f2f8',edgecolor='#8b9ba6'));ax.text(x+w/2,y+.55,head,ha='center',weight='bold',fontsize=11);ax.text(x+w/2,y+.22,body,ha='center',fontsize=9)
for start,end in [((2.4,1.95),(3,2.85)),((2.4,1.85),(3,.95)),((5.7,2.85),(6.2,1.95)),((5.7,.95),(6.2,1.85))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#374b59'})
ax.set_title('Parallel pooling branches with shared projected features',loc='left',fontsize=14,weight='bold');save(fig,'04_hybrid')
fig,ax=plt.subplots(figsize=(10,2.4)); data=np.array([[1,1,1,1,0,0],[1,1,0,0,0,0]])
ax.imshow(data,cmap=matplotlib.colors.ListedColormap(['#eceff1','#16709e']),vmin=0,vmax=1,aspect='auto')
for r in range(2):
    for c in range(6): ax.text(c,r,'valid' if data[r,c] else 'padding',ha='center',va='center',color='white' if data[r,c] else '#444')
ax.set(xticks=range(6),xticklabels=[f'Patch {i+1}' for i in range(6)],yticks=[0,1],yticklabels=['Slide A','Slide B'],title='Illustrative mask: padding must receive zero attention'); save(fig,'05_mask')
audits={d:json.loads((ROOT/f'results/{d}_preflight_windows.json').read_text()) for d in DS}
fig,axs=plt.subplots(1,3,figsize=(12,3.8))
for ax,d in zip(axs,DS):
    n=len(audits[d]['splits']['test']['class_counts']); x=np.arange(n)
    for i,(split,color) in enumerate(zip(['train','val','test'],COLORS)):
        vals=[audits[d]['splits'][split]['class_counts'].get(str(k),0) for k in x]
        ax.bar(x+(i-1)*.25,vals,width=.24,label=split,color=color)
    ax.set(title=d,xlabel='Source class ID',ylabel='Slides',xticks=x); ax.legend(fontsize=8)
fig.tight_layout();save(fig,'06_class_counts')
shutil.copyfile(ROOT/'results/baseline_evaluation_20261029/learning_curves.png',OUT/'07_learning_curves.png')
shutil.copyfile(ROOT/'results/baseline_evaluation_20261029/learning_curves.svg',OUT/'07_learning_curves.svg')
rows=list(csv.DictReader((ROOT/'results/baseline_evaluation_20261029/baseline_comparison.csv').open()))
fig,axs=plt.subplots(1,3,figsize=(11,3.7))
for ax,key,title,scale in zip(axs,['test_accuracy_percent','test_macro_auc','test_macro_f1'],['Test accuracy (%)','Test macro AUROC','Test macro F1'],[100,1,1]):
    vals=[float(r[key]) for r in rows]; ax.bar(DS,vals,color=COLORS); ax.set(ylim=(0,scale*1.13),title=title)
    for i,v in enumerate(vals): ax.text(i,v+scale*.025,f'{v:.2f}' if scale==100 else f'{v:.4f}',ha='center')
fig.tight_layout();save(fig,'08_test_metrics')
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
for ax,key,title in zip(axs,['macro_auc','macro_f1'],['Macro AUROC','Macro F1']):
    x=np.arange(3); v=[float(r['validation_'+key]) for r in rows]; t=[float(r['test_'+key]) for r in rows]
    ax.bar(x-.18,v,.36,label='Selected validation',color='#156a9a'); ax.bar(x+.18,t,.36,label='Held-out test',color='#bd5e25');ax.set(xticks=x,xticklabels=DS,ylim=(0,1.13),title=title);ax.legend(fontsize=8)
fig.tight_layout();save(fig,'09_validation_test')
fig,axs=plt.subplots(1,3,figsize=(12,4))
classrows={}
preds={}
for ax,d in zip(axs,DS):
    base=ROOT/f'results/{d}_baseline_windows/heldout_test'
    cm=np.loadtxt(base/'confusion_matrix.csv',delimiter=',',skiprows=1,usecols=tuple(range(1,len(audits[d]['splits']['test']['class_counts'])+1)),dtype=int)
    n=len(cm); norm=cm/cm.sum(1,keepdims=True); ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(n):
        for j in range(n): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if norm[i,j]>.55 else '#172733',fontsize=9)
    ax.set(title=d,xticks=range(n),yticks=range(n),xlabel='Predicted class ID',ylabel='True class ID')
    classrows[d]=list(csv.DictReader((base/'per_class_metrics.csv').open()))
    pp=list(csv.DictReader((base/'predictions.csv').open())); y=np.array([int(p['label']) for p in pp]); prob=np.array([[float(p[f'probability_class_{k}']) for k in range(n)] for p in pp]);preds[d]=(y,prob)
fig.suptitle('Test confusion counts; colour is normalized within each true class',fontsize=12);fig.tight_layout();save(fig,'10_confusions')
fig,axs=plt.subplots(1,3,figsize=(11,3.8))
for ax,d in zip(axs,DS):
    cr=classrows[d]; vals=[float(r['recall'])*100 for r in cr];ax.bar([r['class_id'] for r in cr],vals,color=COLORS[DS.index(d)])
    ax.set(title=d,xlabel='Source class ID',ylabel='Recall (%)',ylim=(0,116))
    for i,(r,v) in enumerate(zip(cr,vals)): ax.text(i,v+2,f"{round(v*int(r['support'])/100)}/{r['support']}",ha='center',fontsize=8)
fig.tight_layout();save(fig,'11_class_recall')
for mode in ['roc','pr']:
    fig,axs=plt.subplots(1,3,figsize=(12,3.9))
    for ax,d in zip(axs,DS):
        y,p=preds[d]
        for k in range(p.shape[1]):
            binary=(y==k)
            if mode=='roc':
                xx,yy,_=roc_curve(binary,p[:,k]); label=f'Class {k} AUC {float(classrows[d][k]["auc_one_vs_rest"]):.3f}'
            else:
                yy,xx,_=precision_recall_curve(binary,p[:,k]); label=f'Class {k} AP {average_precision_score(binary,p[:,k]):.3f}'
            ax.plot(xx,yy,label=label,lw=1.5)
        if mode=='roc':ax.plot([0,1],[0,1],':',color='gray',lw=1)
        ax.set(title=d,xlabel='False positive rate' if mode=='roc' else 'Recall',ylabel='True positive rate' if mode=='roc' else 'Precision',xlim=(0,1),ylim=(0,1.04));ax.legend(fontsize=7,loc='lower right' if mode=='roc' else 'lower left')
    fig.tight_layout();save(fig,'12_roc' if mode=='roc' else '13_precision_recall')
diagram('14_audit',[[('Feature verification','SHA-256 and finite-value scans'),('Software verification','Masks, metrics, CLI and reload'),('Full baselines','30 epochs on all 3 datasets')],[('Held-out evaluation','Saved checkpoint + predictions'),('Independent summary','Metrics checked from saved CSVs'),('Report documentation','No retraining or new inference')]],'Evidence chain for the reported results')
print('Created',len(list(OUT.glob('*.png'))),'PNG figures and SVG equivalents')
