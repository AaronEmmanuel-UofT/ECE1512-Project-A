# Literature sources

The literature review appears in Section 2 of the [Project A report](../../Project_A_Report.pdf), with explanatory diagrams, worked examples and a unified reference list. Reusable citations are provided in literature_references.bib. The two starting readings below provide additional context for ABMIL and CLAM.

1. **Ilse, Tomczak and Welling (2018), Attention-based Deep Multiple Instance Learning. ICML, PMLR 80:2127-2136.**
   https://proceedings.mlr.press/v80/ilse18a.html
   Start here: this is the baseline paper cited by the assignment. Explain the bag/instance distinction, permutation invariance, learned attention pooling and gated attention. Map the equations to model.py.

2. **Lu et al. (2021), Data-efficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering 5:555-570.**
   https://www.nature.com/articles/s41551-020-00682-w
   Open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/
   Compare CLAM's attention-based aggregation and instance-level clustering with the simpler ABMIL baseline. Separate improvements to aggregation/training from differences in the pretrained feature encoder.

For each paper record: supervision level, bag representation, aggregator, training objective, evaluation split unit, datasets, strengths, limitations, and how it motivates or challenges your proposed extension. Use your own explanation and cite the source.
