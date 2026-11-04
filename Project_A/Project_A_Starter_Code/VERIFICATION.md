# Verification performed in the working environment

This file records the earlier environment. See `LOCAL_VERIFICATION.md` for the
current Windows/PyCharm checks; the saved `C16_debug` run below was not produced
by the current local interpreter.

- All Python modules and completed notebook code cells passed syntax checks.
- The original student notebook, data loader, utilities and dataset configurations match the uploaded starter byte for byte.
- Synthetic smoke checks passed: normalized attention, zero padded-patch attention, padding invariance, patch-order invariance, all-invalid-mask rejection, HDF5 reading, disjoint split IDs, decreasing training loss, metric evaluation and checkpoint reload.
- The standalone runner completed two synthetic epochs, selected a validation checkpoint, evaluated its synthetic test set, and saved every expected output including a loss plot.
- The completed notebook's setup, loaders, model sanity check, training, evaluation and attention-inspection cells executed successfully against a synthetic HDF5 file. Optional WSI visualization was not executed.

Synthetic results are software checks only and must not be reported as dataset performance. Windows/PyCharm execution has not been verified because the user's desktop is not accessible to this session.

## Real CAMELYON16 debugging run

Completed two epochs on the supplied C16 HDF5 file using CPU, seed 42, and the original data/config split logic: 242 training, 27 validation, 129 held-out test slides. Features are 384-dimensional. Each epoch took about 28 seconds in this environment.

Validation-only debugging results:

| Epoch | Training loss | Validation loss | Validation accuracy (%) | Validation macro AUROC | Validation macro F1 |
|---|---|---|---|---|---|
| 1 | 0.6744 | 0.4984 | 77.78 | 0.9148 | 0.7500 |
| 2 | 0.4493 | 0.2187 | 92.59 | 1.0000 | 0.9206 |

These are two-epoch validation results on only 27 slides, not final baseline performance. The held-out test set was not evaluated. Run the full configured 30 epochs in a fresh experiment before reporting a baseline. Saved debugging outputs are in results/C16_debug. The HDF5 file is excluded from the distributable ZIP.
