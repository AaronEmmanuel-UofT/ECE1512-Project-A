# Full comparison protocol

Recorded November 2, 2026 before the six full training runs and their held-out evaluations.

Use the October 30 fragility rule unchanged, probe fraction 0.10, seed 42, 30 epochs, batch size 1, AdamW learning rate 0.0001 and weight decay 0.00001, dropout 0, two CPU threads and no scheduler. Retain the earliest checkpoint maximizing validation macro AUROC plus macro F1. Preserve all original split IDs, configurations and feature files.

Complete fragility-guided pooling on C16, C17 and BRACS, followed by pure-mean controls on each dataset. Finish all six training runs before opening their held-out results. No parameter selection follows test evaluation. Reuse the recorded fixed-50/50 Mac runs and baseline evaluations. Report both the group C17 baseline (Windows, epoch 21) and the earlier Mac C17 baseline (epoch 19) when interpreting platform sensitivity; do not choose the better baseline after observing extension results. The group baseline remains the primary report reference. The C16 and BRACS baseline measurements agree across the retained report sources.

Export probabilities, per-class metrics, confusion matrices and gate diagnostics for the new methods. Existing fixed-mixture controls have aggregate held-out records, not saved slide probabilities; disclose this difference and do not invent per-class comparisons for those controls. Preserve the single-seed, cross-platform, C17 validation patient overlap and previous test exposure limitations.

The common-subset perturbation experiment is outside this fixed comparison. Do not claim improved robustness, causal interpretation, calibrated uncertainty or statistical superiority. Interpret negative results without changing the model.
