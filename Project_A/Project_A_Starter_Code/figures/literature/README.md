# Literature illustrations

These are original explanatory schematics, not copied paper figures,
histology images or measured model outputs. Report captions provide paper
citations. The bibliography is in `../../literature_references.bib`.

| Figure | File stem | Source and scope |
| --- | --- | --- |
| 1 | 01_slide_pipeline | MIL [1] and our implementation [9,10]; a schematic tile grid and fixed 384-dimensional feature setting |
| 2 | 02_pooling_example | Original arithmetic illustrating pooling [1]; selected weights and scalar features, not learned values |
| 3 | 03_supervision_comparison | ABMIL [1], CLAM [2], DSMIL [3]; simplified aggregation and supervision, with encoders omitted |
| 4 | 04_contextual_aggregation | ABMIL [1] and TransMIL [4]; conceptual token flow, with class-token input and internal transformer operations omitted |
| 5 | 05_pseudo_bags | DTFD-MIL [5]; two-tier flow, with the distillation strategy left abstract |
| 6 | 06_attention_training | MHIM-MIL [6] and ACMIL [7]; training mechanisms, with EMA updates, auxiliary heads and detailed loss connections omitted |

Run `make_literature_figures.py` with the project interpreter, or use the
PyCharm **05 Literature figures** configuration in the parent Project_A
project. Each run replaces six PNG and six editable SVG files in this folder.
SVG labels remain text. Word embeds the PNGs; regenerating these files does
not change the embedded pictures until they are replaced in Word.

The worked scalar example uses h = [0,0,0,1]. The mean is 0.25, the maximum
is 1.00, and weights [0.1,0.1,0.1,0.7] produce 0.70. The pseudo-bag and
class-imbalance examples in the text are also explicitly illustrative.
