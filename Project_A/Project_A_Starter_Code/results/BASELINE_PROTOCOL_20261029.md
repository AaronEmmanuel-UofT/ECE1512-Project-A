# Baseline execution protocol - October 29, 2026

This records the existing training protocol being executed, not a new tuning study.

- ABMIL only; no hybrid training or adaptive gates in this baseline evaluation.
- C16: reuse the completed 30-epoch run and its original validation-selected checkpoint.
- C17 and BRACS: fresh 30-epoch runs, seed 42, supplied features, original class mappings and split logic.
- Batch size 1; AdamW, learning rate 0.0001, weight decay 0.00001; dropout 0; no learning-rate scheduler or warmup.
- CPU with two training threads per process. Runs use separate processes, each initialised with seed 42.
- Selection: maximum validation macro AUROC + macro F1, retaining the earliest tied epoch.
- Test evaluation after completion: load the selected checkpoint without retraining; verify training-source fingerprints and exact split IDs.
- Fixed argmax classification; no threshold or hyperparameter tuning from test results.
- Save per-slide probabilities, confusion matrices, per-class metrics and checkpoint fingerprints. Derive report tables from saved predictions, without repeated test inference.
- Report slide-level test accuracy (%), macro AUROC, macro F1 and cross-entropy loss; inspect training/validation curves and per-class errors.
- Disclose the single seed and 27 patients shared between C17 training and validation. No patient-level staging or clinical validation claim.

The protocol was inherited from the verified baseline implementation. It is not described as a prospectively registered experiment.
