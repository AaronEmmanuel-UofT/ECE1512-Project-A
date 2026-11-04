# Local Windows/PyCharm verification - October 27, 2026

Software checks passed for the baseline and hybrid method on all three dataset
paths. All real feature files passed full scans and fingerprint checks. Real
two-epoch debug runs are complete for C16, C17 and BRACS. Full
30-epoch benchmark performance remains to be established.

Project: `C:/Users/gmikh/Documents/Project A/Project_A_Starter_Code`.
Interpreter: parent `.venv/Scripts/python.exe` (Python 3.9.5).

## Initial findings

- PyCharm was using an empty virtual environment. The first smoke check failed
  at `import h5py`. Dependencies were then installed from `requirements.txt`.
- The saved `results/C16_debug` metrics came from Python 3.12.14 / Torch 2.14.1,
  not this Windows environment. They remain debugging evidence only.
- No feature HDF5 was present in the project. The 7,222,943,888-byte C16 feature
  file has now been downloaded from the course-linked Hugging Face repository.
  Its SHA-256 matches the published file fingerprint:
  `f22c22c4dc7c5314a2b4b16b8322f8c9cab8e8330d0942a5ab0cbf4f7b27fd6f`
  (repository revision `6df7ef4b42b7ecd11633722a392a1efc6d80bcc7`).
- The completed notebook evaluated held-out test slides after two debugging
  epochs. Testing is now opt-in, and its attention example uses validation data.

## Checks and improvements

- All eleven Python modules parse with Python 3.9 grammar. All 18 non-magic code
  cells in the completed notebook parse, and its model matches `model.py`.
- `dataloader.py`, `utils.py`, and the original student notebook match the
  downloaded starter ZIP byte for byte.
- `preflight.py` audits feature shape, class coverage, slide overlap, optional
  NaN/Inf scanning and C17 patient overlap. It reads data without fitting a model.
- `verify_baseline.py` exercises full synthetic runs for all three datasets and
  checks metrics against scikit-learn. Synthetic results are never benchmarks.
- New training runs record the interpreter, dependency versions, CPU thread
  count, command and source fingerprints. `--threads` allows CPU tuning without
  changing the learning protocol.
- Eight named PyCharm Run configurations are available in the parent `.run` folder,
  including all three baselines and the optional C16 extension.
- Proposal 1 is implemented in `model_hybrid.py`. Its smoke checks passed for
  zero-weight exact baseline equivalence, mean-only endpoint, padding/order
  invariance, single-patch bags, parameter counts and finite gradients.
- The refactored ABMIL matches the original assisted baseline ZIP exactly for
  logits and attention on both padded and unpadded test bags.
- The notebook's test guard was checked with a mocked evaluator: selected weights
  are always restored, and test evaluation runs only when explicitly enabled.

## Execution status

The full C16 feature scan passed: 398 slides, class counts 238/160, 384 feature
dimensions, 153-55,917 patches per slide, and no NaN/Inf values. See
`results/C16_raw_feature_audit.json`. Its file fingerprint also matches the source.

The requirements installed successfully and `pip check` found no broken
requirements. Package versions are saved in `results/windows_packages.txt`.
The baseline smoke check passed inside PyCharm with exit code 0, including
synthetic learning, metric calculation, masks and checkpoint reload. The extension
smoke check also passed using the same interpreter.

The full synthetic runner verification completed for C16, C17 and BRACS, for both
ABMIL and the hybrid method. It passed metric cross-checks against scikit-learn
for 2/4/7 classes, corrupted-value rejection, finite training, checkpoint selection,
output generation, test opt-in, identical baseline/extension split IDs and output
overwrite protection. Machine-readable evidence: `results/windows_verification.json`.
These synthetic runs are software checks only, not dataset performance.

The real C16 two-epoch local run completed and uses exactly the same split IDs as
`results/C16_debug`. Both epochs closely reproduce the earlier debugging run.

| Epoch | Training loss | Validation loss | Validation accuracy (%) | Validation AUC | Validation macro F1 |
|---|---|---|---|---|---|
| 1 | 0.674386 | 0.498421 | 77.78 | 0.914773 | 0.750000 |
| 2 | 0.449325 | 0.218735 | 92.59 | 1.000000 | 0.920588 |

These are validation-only debugging metrics on 27 validation slides, not a final
baseline benchmark. The local run did not evaluate held-out test predictions.
Outputs, checkpoint and the inspected loss plot are in `results/C16_debug_windows`.
Epochs took 174 and 153 seconds while other verification work was running.

C17 passed the full value scan and split audit: 269 train, 30 validation and 200
test slides, with all four classes present in every split and 384 feature
dimensions. There are 27 shared patients between train and validation; no
patients are shared with test. This preserves the supplied slide-level split,
and must be disclosed when discussing validation and generalization.

Its two real debug epochs completed successfully. Training loss decreased from
1.05534 to 0.96519 and validation loss from 1.03690 to 1.00120. Validation AUC
increased from 0.50831 to 0.56082. Validation accuracy stayed at 66.67% and macro
F1 at 0.20, consistent with majority-class predictions at this early stage.
These results do not establish good C17 performance or convergence. Outputs are
in `results/C17_debug_windows`; held-out test prediction was not evaluated.

BRACS also passed the full value scan and split audit: 389 train, 62 validation,
86 test slides, all seven classes in each split, and 384 feature dimensions.
All three datasets have full scan reports in `results/*_preflight_windows.json`
and verified source fingerprints in `results/*_feature_provenance.json`.

Both real BRACS debug epochs completed. Training loss decreased from 1.60187 to
1.24250 and validation loss from 1.81873 to 1.75727. Validation AUC increased from
0.68270 to 0.74369, while accuracy fell from 33.87% to 27.42% and macro F1 from
0.19178 to 0.13391. This mixed result is useful debugging evidence, not a claim
of strong performance after only two epochs. Outputs are in
`results/BRACS_debug_windows`; held-out test prediction was not evaluated.

All three real debug runs generated the expected checkpoint, configuration,
environment, split IDs, two history rows, loss plot and summary. Full 30-epoch
baseline and extension comparisons remain necessary before reporting benchmark
results. No real held-out test predictions were evaluated during this session.

## Remaining experiment requirements

Update October 28, 2026: the full C16 validation baseline completed all 30 epochs.
Validation selected epoch 3, with macro AUC 1.0, accuracy 100% and macro F1 1.0
on only 27 validation slides. Held-out test prediction remains unevaluated; these
are validation results, not test performance. The run used Run entry
`04 C16 baseline validation`. Its saved environment confirms 30 epochs, CPU,
two threads and the project interpreter. Completed outputs are in
`results/C16_baseline_windows`, including all 30 history rows and the selected
checkpoint. This run kept held-out test evaluation disabled and
does not automatically launch the other datasets or extension experiments.

Run the fixed 30-epoch baseline on C16, C17 and BRACS, using validation for model
selection and held-out testing only after the protocol is fixed. All three
feature files are downloaded and match their published fingerprints. The supplied C17 split is by held-out centers
for testing and by slides for train/validation; disclose patient overlap if found.
Then evaluate the implemented proposal 1 in `EXPERIMENT_PLAN.md` and compare it on
the same splits. Literature review, report and experimental analysis remain work
to complete. No improvement or pathology performance is claimed by software checks.
