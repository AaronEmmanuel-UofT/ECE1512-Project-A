# Literature sources

The expanded literature review is in ../ECE1512_Project_A_Report_Draft.docx (pages 1–8), with six original diagrams and worked examples. References are on page 15. See ../LITERATURE_REVIEW.md for coverage and the next steps, and literature_references.bib for reusable citations. The two starting readings below are retained as a guide to the baseline and CLAM. Review the cited methods before choosing the extension.

1. **Ilse, Tomczak and Welling (2018), Attention-based Deep Multiple Instance Learning. ICML, PMLR 80:2127-2136.**
   https://proceedings.mlr.press/v80/ilse18a.html
   Start here: this is the baseline paper cited by the assignment. Explain the bag/instance distinction, permutation invariance, learned attention pooling and gated attention. Map the equations to model.py.

2. **Lu et al. (2021), Data-efficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering 5:555-570.**
   https://www.nature.com/articles/s41551-020-00682-w
   Open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/
   Compare CLAM's attention-based aggregation and instance-level clustering with the simpler ABMIL baseline. Separate improvements to aggregation/training from differences in the pretrained feature encoder.

For each paper record: supervision level, bag representation, aggregator, training objective, evaluation split unit, datasets, strengths, limitations, and how it motivates or challenges your proposed extension. Use your own explanation and cite the source.
