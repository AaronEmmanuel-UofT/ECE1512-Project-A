# MacBook experiment protocol

Fixed on October 28, 2026, before new real-data model fitting or held-out predictions.

- Use the supplied split logic and metadata, verified feature revision
  `6df7ef4b42b7ecd11633722a392a1efc6d80bcc7`, and original YAML files.
- Seed 42; 30 epochs; batch size 1; AdamW; learning rate 0.0001;
  weight decay 0.00001; no scheduler or warmup; CPU with two threads.
- Preserve 2 C16, 4 C17 and 7 BRACS classes. Infer the feature dimension.
- Select the checkpoint with greatest validation macro AUROC + macro F1;
  retain the first epoch in a tie, as implemented by the supplied runner.
- Compare ABMIL to attention_mean with mean weight fixed at 0.5 on all three
  datasets. No grid search, additional seeds or architecture changes in this run.
- Reuse George's completed 30-epoch C16 baseline and selected epoch-3 checkpoint.
  Preserve its Windows environment record. Cross-platform numerical differences
  are a comparison limitation; record the new Mac environment independently.
- Complete C17/BRACS ABMIL, then C16/C17/BRACS hybrid runs. Check split identity
  against handed-off runs and between methods before held-out evaluation.
- Evaluate held-out data only after all five new validation runs have finished,
  using the selected saved checkpoints. Save separate results without modifying
  George's original summaries. Report loss, accuracy (%), macro AUROC and macro F1.
- Disclose 27 C17 patients shared across train/validation; test patients are
  separate. No patient-independent validation claim. Retain the supplied protocol.
- Debug and synthetic metrics are software evidence. C16's perfect validation
  metrics cover 27 slides and are not held-out results. Historical notebook debug
  test exposure is documented in LOCAL_VERIFICATION.md; do not imply test data
  have never previously been inspected anywhere in the project history.
- Report negative results; no improvement, novelty or end-to-end encoder training
  is assumed. Literature review, interpretation and the report require separate work.

The original handoff source files are preserved. Mac automation and evaluation
tools are in this directory. No original-slide downloads are required for these
feature-based experiments. Keep the laptop plugged in and the lid open; the
background workers inhibit idle system sleep while active.
