"""Original teaching schematics for the report, not experimental results.

Run with the project interpreter: python make_literature_figures.py
PNG files are embedded in Word; SVG files remain editable. Paper numbers
refer to literature_references.bib and the report bibliography.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/project_a_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent / 'figures' / 'literature'
OUT.mkdir(parents=True, exist_ok=True)
REVIEW_ONLY = False
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
BLUE, PURPLE, ORANGE = '#DDEAF2', '#E6E0EF', '#F7E4CD'
INK, LINE = '#20262C', '#52616B'

def canvas(height):
    f, ax = plt.subplots(figsize=(10, height))
    f.subplots_adjust(left=.015, right=.985, top=.98, bottom=.02)
    ax.set(xlim=(0, 10), ylim=(0, height)); ax.axis('off')
    return f, ax

def text(ax, x, y, label, size=12, **kw):
    ax.text(x, y, label, ha='center', va='center', fontsize=size, color=INK, **kw)

def box(ax, x, y, w, h, label, color=BLUE, dashed=False, size=12):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=LINE,
                           linewidth=1.05, linestyle='--' if dashed else '-'))
    text(ax, x+w/2, y+h/2, label, size)

def arrow(ax, a, b, dashed=False, color=LINE):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11,
                 linewidth=1.1, color=color, linestyle='--' if dashed else '-'))

def save(f, name):
    f.savefig(OUT / (name+'.png'), dpi=300)
    f.savefig(OUT / (name+'.svg'))
    plt.close(f)

def pipeline():
    f,a=canvas(3.15)
    text(a,1.0,2.91,'Whole slide',13,fontweight='bold')
    # A schematic tile grid, intentionally not a histology image.
    for row in range(4):
        for col in range(4):
            a.add_patch(Rectangle((.25+col*.36,1.1+row*.36),.33,.33,
                        facecolor=ORANGE if (row,col)==(2,1) else BLUE,
                        edgecolor='white'))
    text(a,1.0,.85,'Tissue patches',11)
    arrow(a,(1.85,1.95),(2.22,1.95))
    box(a,2.25,1.45,1.55,1.0,'Fixed\nimage encoder')
    arrow(a,(3.8,1.95),(4.18,1.95))
    box(a,4.2,1.45,1.5,1.0,'Feature bag\nN × D' if REVIEW_ONLY else 'Feature bag\nN × 384')
    arrow(a,(5.7,1.95),(6.08,1.95))
    box(a,6.1,1.45,1.7,1.,'MIL pooling\n+ classifier',PURPLE)
    arrow(a,(7.8,1.95),(8.18,1.95))
    box(a,8.2,1.45,1.55,1.,'Slide class\nprobabilities',PURPLE)
    text(a,4.96,2.84,'N varies between slides',11)
    text(a,2.98,1.05,'Already computed',10.5)
    box(a,6.1,.15,1.7,.62,'Slide label y',ORANGE,size=11.5)
    arrow(a,(7.8,.46),(8.95,.46),True)
    arrow(a,(8.95,.46),(8.95,1.4),True)
    text(a,9.05,.95,'Loss',10.5)
    text(a,2.85,.34,'No patch labels supplied to MIL training',11)
    save(f,'01_slide_pipeline')

def pooling():
    f,a=canvas(3.3)
    text(a,5,3.03,'Illustrative scalar features   h = [0, 0, 0, 1]',13,fontweight='bold')
    for x,title in [(1.6,'Mean'),(5,'Max'),(8.35,'Attention')]:
        text(a,x,2.55,title,13,fontweight='bold')
    weights=[[.25]*4,[0,0,0,1],[.1,.1,.1,.7]]
    for center,w in zip([1.6,5,8.35],weights):
        for i,v in enumerate(w):
            x=center-1.05+i*.7
            a.add_patch(Rectangle((x-.24,1.1),.48,v*1.1,facecolor=PURPLE,
                                 edgecolor=LINE,linewidth=.8))
            text(a,x,.88,str(i+1),10.5)
            text(a,x,1.25+v*1.1,f'{v:.2f}',10.5)
    for x,value in [(1.6,'z = 0.25'),(5,'z = 1.00'),(8.35,'z = 0.70')]:
        text(a,x,.46,value,13,fontweight='bold')
    text(a,5,.09,'Bars show pooling weights; horizontal labels are patch indices. Values are chosen for illustration.',10.5)
    save(f,'02_pooling_example')

def supervision():
    f,a=canvas(4.2)
    text(a,.9,3.85,'ABMIL [1]',12.5,fontweight='bold')
    text(a,.9,2.55,'CLAM [2]',12.5,fontweight='bold')
    text(a,.9,.8,'DSMIL [3]',12.5,fontweight='bold')
    box(a,1.9,3.45,1.35,.75,'Features')
    box(a,3.8,3.45,2.2,.75,'Attention pooling',PURPLE)
    box(a,7.1,3.45,2.55,.75,'Slide classifier',PURPLE)
    arrow(a,(3.25,3.82),(3.78,3.82));arrow(a,(6,3.82),(7.08,3.82))
    box(a,1.9,2.15,1.35,.75,'Features')
    box(a,3.8,2.15,2.2,.75,'Attention pooling',PURPLE)
    box(a,7.1,2.15,2.55,.75,'Slide classifier',PURPLE)
    arrow(a,(3.25,2.52),(3.78,2.52));arrow(a,(6,2.52),(7.08,2.52))
    box(a,3.8,1.2,5.85,.58,'Top / bottom attention → pseudo labels → clustering loss',ORANGE,True,10.7)
    arrow(a,(4.9,2.14),(4.9,1.8),True)
    box(a,1.9,.02,1.35,.75,'Features')
    box(a,3.8,.02,2.2,.75,'Critical instance',PURPLE)
    box(a,7.1,.02,2.55,.75,'Similarity weighted\nbag stream',PURPLE,size=11)
    arrow(a,(3.25,.39),(3.78,.39));arrow(a,(6,.39),(7.08,.39))
    text(a,6.55,.95,'Instance and bag scores are combined',10.5)
    save(f,'03_supervision_comparison')

def contextual():
    f,a=canvas(3.6)
    text(a,2.05,3.31,'ABMIL   independent scoring',13,fontweight='bold')
    text(a,7.1,3.31,'TransMIL   contextual tokens [4]',13,fontweight='bold')
    for j,y in enumerate([2.68,2.03,1.38]):
        box(a,.2,y-.23,.85,.46,f'h{j+1}',size=11)
        box(a,1.65,y-.23,1.1,.46,'Scorer',PURPLE,size=11)
        arrow(a,(1.05,y),(1.62,y))
        arrow(a,(2.75,y),(3.4,2.03))
    box(a,3.4,1.64,1.15,.78,'Softmax\n+ pooling',PURPLE,size=10.5)
    for i,x in enumerate([5.38,6.17,6.96]):
        box(a,x,2.52,.62,.48,f'h{i+1}',size=11)
    for x in [5.69,6.48,7.27]:
        arrow(a,(x,2.5),(6.48,2.1))
    box(a,5.1,1.52,2.77,.58,'Self attention layer',PURPLE,size=11)
    box(a,5.1,.7,2.77,.58,'Position module PPEG',PURPLE,size=11)
    arrow(a,(6.48,1.52),(6.48,1.3))
    box(a,8.35,.7,1.45,1.4,'Second\nself attention\n+ class token\nreadout',PURPLE,size=10.5)
    arrow(a,(7.87,.99),(8.33,.99))
    text(a,2.32,.68,'Other patches affect normalization,\nnot the initial score of a patch.',11)
    text(a,5,.19,'Token interactions are learned; a sequence grid is not proof of spatial adjacency.',10.2)
    save(f,'04_contextual_aggregation')

def pseudobags():
    f,a=canvas(3.45)
    box(a,.15,1.37,1.35,.85,'One slide\nlabel y',BLUE)
    text(a,.83,.8,'Random split',10.5)
    for y,label in [(2.64,'Pseudo bag 1'),(1.61,'Pseudo bag 2'),(.58,'Pseudo bag 3')]:
        box(a,2.05,y-.27,1.55,.6,label,BLUE,size=11)
        arrow(a,(1.5,1.8),(2.03,y))
        box(a,4.1,y-.27,1.8,.6,'Tier 1 MIL',PURPLE,size=11)
        arrow(a,(3.6,y),(4.08,y))
        text(a,4.98,y+.46,'Target y',10.5)
        arrow(a,(5.9,y),(6.45,1.61))
    box(a,6.45,1.23,1.35,.8,'Distilled\nfeatures',PURPLE,size=11)
    arrow(a,(7.8,1.63),(8.28,1.63))
    box(a,8.3,1.23,1.5,.8,'Tier 2 MIL\nslide output',PURPLE,size=11)
    text(a,9.05,2.34,'Target y',10.5)
    text(a,5,.1,'Inherited labels can be noisy: a positive slide can yield a pseudo bag without a positive patch.',11)
    save(f,'05_pseudo_bags')

def masking():
    f,a=canvas(4.15)
    text(a,5,3.94,'Training mechanisms   dashed paths are used during training',12.5,fontweight='bold')
    text(a,.85,2.77,'MHIM-MIL [6]' if REVIEW_ONLY else 'MHIM [6]',11.5 if REVIEW_ONLY else 12.5,fontweight='bold')
    box(a,1.8,3.0,1.25,.6,'Full bag',size=11)
    box(a,3.6,3.,1.75,.6,'Momentum\nteacher' if REVIEW_ONLY else 'EMA teacher',PURPLE,True,11)
    arrow(a,(3.05,3.3),(3.58,3.3),True)
    box(a,3.6,1.82,1.75,.6,'Masked bag',BLUE,True,11)
    arrow(a,(3.05,3.3),(3.6,2.15),True)
    arrow(a,(4.47,3.0),(4.47,2.44),True)
    box(a,6.05,1.82,1.7,.6,'Student MIL',PURPLE,size=11)
    arrow(a,(5.35,2.12),(6.03,2.12))
    box(a,8.32,1.82,1.5,1.78,'Class loss\n+ teacher\nconsistency',ORANGE,True,11)
    arrow(a,(7.75,2.12),(8.3,2.12),True)
    arrow(a,(5.35,3.3),(8.3,3.3),True)
    text(a,6.78,2.9,'Attention selects masks',10.5)
    text(a,.85,.92,'ACMIL [7]',12.5,fontweight='bold')
    box(a,1.8,.58,1.25,.7,'Full bag',size=11)
    box(a,3.6,.58,2.05,.7,'Multiple attention\nbranches',PURPLE,size=11)
    arrow(a,(3.05,.93),(3.58,.93))
    box(a,6.05,.58,1.7,.7,'Top K masking',PURPLE,True,10.7)
    arrow(a,(5.65,.93),(6.03,.93))
    box(a,8.32,.58,1.5,.7,'Class losses\n+ diversity',ORANGE,True,10.7)
    arrow(a,(7.75,.93),(8.3,.93),True)
    text(a,5,.13,'Inference: full-bag student (MHIM-MIL); no stochastic masking (ACMIL).' if REVIEW_ONLY else 'Inference: MHIM uses the full bag in the student; ACMIL disables stochastic masking.',11)
    save(f,'06_attention_training')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--literature-only', action='store_true', help='Use generic features and expanded method labels for the standalone review')
    args = parser.parse_args()
    REVIEW_ONLY = args.literature_only
    if REVIEW_ONLY:
        OUT = OUT.parent / 'literature_review'
        OUT.mkdir(parents=True, exist_ok=True)
    for make in [pipeline,pooling,supervision,contextual,pseudobags,masking]:
        make()
    print(f'Saved six original PNG/SVG diagrams to {OUT}')
