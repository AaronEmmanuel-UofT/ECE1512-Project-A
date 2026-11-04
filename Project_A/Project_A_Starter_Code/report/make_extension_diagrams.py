"""Static scientific figures for the combined report; saved artifacts only."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).parent/'word_figures';OUT.mkdir(exist_ok=True)
DS=['C16','C17','BRACS'];plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
def save(f,name):
    f.savefig(OUT/(name+'.png'),dpi=230,bbox_inches='tight',facecolor='white')
    f.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(f)
def node(ax,x,y,w,h,title,body):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#edf3f7',edgecolor='#8295a2'))
    ax.text(x+w/2,y+h*.70,title,ha='center',va='center',weight='bold',fontsize=11)
    ax.text(x+w/2,y+h*.30,body,ha='center',va='center',fontsize=9)
def arrow(ax,a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#35576c','lw':1.5})
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.13,label,fontsize=9,ha='center')
f,ax=plt.subplots(figsize=(10,7));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
node(ax,2.8,6,4.3,.75,'Same projected patch features','384 -> 256; fixed encoder features')
node(ax,.2,4.4,3,.85,'Attention pooling','z_A = sum a_i h_i')
node(ax,3.7,4.4,2.7,.85,'Suppression probe','Top 10% patch equivalents')
node(ax,7,4.4,2.7,.85,'Mean pooling','z_M = sum h_i / N')
arrow(ax,(4.9,5.9),(1.7,5.3));arrow(ax,(4.9,5.9),(5.05,5.3));arrow(ax,(4.9,5.9),(8.3,5.3))
node(ax,.2,2.9,3,.8,'Original classifier','p_A from attention vector')
node(ax,3.7,2.9,2.7,.8,'Same classifier','p_P after renormalizing')
arrow(ax,(1.7,4.3),(1.7,3.8));arrow(ax,(5.05,4.3),(5.05,3.8))
node(ax,1.8,1.4,4.8,.8,'Detached sensitivity gate','alpha = 0.5 sum |p_A - p_P|')
arrow(ax,(1.7,2.8),(3.0,2.3));arrow(ax,(5.05,2.8),(4.9,2.3))
node(ax,6.9,1.4,2.8,.8,'Mix feature vectors','(1-alpha) z_A + alpha z_M')
arrow(ax,(6.7,1.8),(6.85,1.8));arrow(ax,(8.35,4.3),(8.35,2.3))
ax.plot([.1,.03,.03,6.65,6.65],[4.8,4.8,.95,.95,1.8],color='#35576c',lw=1.5)
arrow(ax,(6.65,1.8),(6.85,1.8));ax.text(.28,1.03,'Attention feature vector z_A',fontsize=9)
node(ax,6.9,.1,2.8,.8,'Final classifier','Class logits and cross entropy')
arrow(ax,(8.3,1.3),(8.3,1.0))
ax.text(.3,.13,'The final loss trains the original learned layers through\nthe mixed feature vector. The sensitivity measurement\nis outside the gradient graph.',fontsize=9)
save(f,'fragility_architecture')
f,ax=plt.subplots(figsize=(9,3.9));x=np.arange(4);a=np.array([.7,.1,.1,.1]);alpha=.4;w=(1-alpha)*a+alpha/4
ax.bar(x-.18,a,.36,label='Attention weights',color='#156a9a');ax.bar(x+.18,w,.36,label='Effective hybrid weights',color='#bd5e25')
for i in range(4):
    ax.text(i-.18,a[i]+.02,f'{a[i]:.2f}',ha='center');ax.text(i+.18,w[i]+.02,f'{w[i]:.2f}',ha='center')
ax.set(xticks=x,xticklabels=['Patch 1','Patch 2','Patch 3','Patch 4'],ylim=(0,.9),ylabel='Pooling contribution',title='Illustrative smoothing when alpha is 0.40');ax.legend();f.tight_layout();save(f,'effective_weights')
