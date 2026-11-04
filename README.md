# ECE1512 Project A

Aaron Emmanuel and George Mikhaiel

Attention-based multiple instance learning (ABMIL) baseline evaluation and
sensitivity-controlled attention–mean pooling on CAMELYON16, CAMELYON17 and
BReAst Carcinoma Subtyping (BRACS). The report reviews related work, develops
two proposals and evaluates the pooling proposal. Normalized attention entropy
regularization is the second proposal; it is not implemented.

## Report and Source

- [PDF report](Project_A_Report.pdf)
- [Editable Word report](Project_A_Report.docx)
- [Source, notebooks, results and figures](ECE1512_ProjectA_Source_and_Results.zip)

Clone this repository, or extract the complete code/results archive into an empty directory. It contains a `Project_A` folder.
The commands below run inside `Project_A/Project_A_Starter_Code`. The archive
contains selected checkpoints, full training histories and split identifiers;
large features and source whole-slide images are obtained separately. The
SHA-256 manifest verifies packaged files.

## Environment and Data

The full extension/control experiments used Python 3.12.14, PyTorch 2.8.0,
CPU execution and two threads. Primary baselines used George's recorded
Python 3.9.5 Windows environment. Per-run environment files preserve these
differences; identical seeds do not guarantee identical platform trajectories.

```sh
cd Project_A/Project_A_Starter_Code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-locked.txt
python -m ipykernel install --user --name ece1512 --display-name "ECE1512 project"
python fetch_features.py --dataset all
```

On Windows, activate with `.venv\Scripts\activate` instead. The download command
obtains about 14 GB of fixed-revision HDF5 features and verifies all three complete
SHA-256 hashes. Source URLs and sizes are in `results/feature_downloads`.
Feature revision: `6df7ef4b42b7ecd11633722a392a1efc6d80bcc7` from
[LeonYe/ECE1512-WSI-Feature](https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature/tree/6df7ef4b42b7ecd11633722a392a1efc6d80bcc7).
No encoder is trained or downloaded for these feature-based experiments.

## Review the Saved Results

```sh
python research/summarize_comparison.py
python ../macbook/show_results.py
```

These commands read saved predictions and experiment records; they do not run
models on the test set. The summarizer checks checkpoint/source hashes, exact
test slide order and metric agreement with scikit-learn. It regenerates the
comparison CSV, class/gate summaries and final scientific plots.

`ECE1512_ProjectA_Baseline.ipynb` is executed with saved outputs and has an HTML
view beside it. Its default mode reads existing checkpoints/results. Inspecting
batches requires the feature files. Optional whole-slide figure regeneration
also requires the source slides described in `wsi_visualization.py`; the saved
attention overlays and raw arrays are included. The untouched student notebook
is retained for attribution and scaffold context.

## Measured Held-Out Results

All rows use 30 training epochs and seed 42. Checkpoints maximize validation
macro AUROC plus macro F1, with the earliest tie retained. Accuracy is a
percentage; AUROC and F1 are macro averages, and CE is slide-mean cross-entropy.

| Dataset | Pooling | Epoch | Accuracy % | Macro AUROC | Macro F1 | CE |
|---|---|---:|---:|---:|---:|---:|
| C16 | ABMIL | 3 | 88.37 | 0.9224 | 0.8714 | 0.2700 |
| C16 | Fixed 50/50 | 3 | 90.70 | 0.9128 | 0.8986 | 0.2738 |
| C16 | Mean | 16 | 65.89 | 0.6107 | 0.5831 | 0.6551 |
| C16 | Fragility | 12 | 86.05 | 0.9069 | 0.8480 | 0.3799 |
| C17 | ABMIL | 21 | 78.50 | 0.8226 | 0.5520 | 0.7969 |
| C17 | Fixed 50/50 | 7 | 81.00 | 0.8640 | 0.4922 | 0.5054 |
| C17 | Mean | 2 | 60.50 | 0.7045 | 0.1885 | 0.9928 |
| C17 | Fragility | 19 | 78.50 | 0.8428 | 0.5218 | 0.6753 |
| BRACS | ABMIL | 3 | 46.51 | 0.7855 | 0.3716 | 1.4110 |
| BRACS | Fixed 50/50 | 12 | 44.19 | 0.7860 | 0.3671 | 1.4786 |
| BRACS | Mean | 20 | 36.05 | 0.6529 | 0.2648 | 1.8416 |
| BRACS | Fragility | 19 | 33.72 | 0.7051 | 0.2316 | 2.1113 |

The primary baseline table uses `results/{dataset}_baseline_windows`. An earlier
Mac C17 baseline is also retained: epoch 19, accuracy 73.00%, macro AUROC 0.8325,
macro F1 0.5426. It is a separate environment reference, not a replacement for
the group baseline. Fixed 50/50 controls are the original completed Mac runs.
Their aggregate held-out records are in `../macbook/held_out`; individual fixed-
mixture probabilities were not retained. Baseline, adaptive and mean predictions
and class metrics are included. C17 training and validation share 27 patients;
test patients are separate. Results are slide-level, single-seed and do not
establish clinical validity, statistical superiority or causal explanations.

## Fresh Experiments

The full queue trains all six new configurations before any test evaluation:

```sh
python research/complete_comparison.py --output-root results/reproduced_comparison
python research/summarize_comparison.py --output-root results/reproduced_comparison
```

That queue reuses the saved ABMIL and fixed-mixture references. For an individual
fresh adaptive experiment, use a new output path:

```sh
python research/train_extension.py --dataset C16 --method fragility --epochs 30 --threads 2 --output results/C16_fragility_reproduced
python research/evaluate_extension.py --run results/C16_fragility_reproduced --output results/C16_fragility_reproduced/heldout_test --threads 2
```

Replace C16 with C17 or BRACS for the other datasets. Use `--method mean` for
the pure-mean control. The fixed-mixture and original baseline commands are:

```sh
python train.py --dataset C16 --method attention_mean --mean-weight 0.5 --epochs 30 --threads 2 --output results/C16_fixed_reproduced
python train.py --dataset C16 --method abmil --epochs 30 --threads 2 --output results/C16_abmil_reproduced
python evaluate_run.py --run results/C16_fixed_reproduced --output results/C16_fixed_reproduced/heldout_test --threads 2
python evaluate_run.py --run results/C16_abmil_reproduced --output results/C16_abmil_reproduced/heldout_test --threads 2
```

Repeat with C17 and BRACS for the complete baseline/control protocol. Train.py
defaults to validation-only training. Evaluate only after fixing the experiment
protocol; saved held-out results should ordinarily be reused. Runners reject
existing experiment outputs, and evaluators reject repeated test evaluation.
The queue rejects incomplete existing runs rather than silently restarting them.
Use CPU-only execution for comparison with these records.

## Software Checks and Figures

```sh
python smoke_test.py
python smoke_test_extension.py
python -m unittest research.test_fragility -v
python report/make_figures.py
python report/make_extension_diagrams.py
python make_literature_figures.py
```

The seven extension tests cover the gate formula, parameter equality, ties,
padding and permutation invariance, numerical extremes and synthetic training
through all three dataset formats. Synthetic checks are not pathology results.
`research/summarize_comparison.py` generates the final comparison plots without
inference. Report images are preserved in `figures/report` with their captions.
The illustration sources and assumptions are distinguished from measured data
in the report. No common-subset robustness benchmark is claimed.

## Attribution

The model and training pipeline extend the supplied ECE1512 Project A scaffold.
Baseline method: Ilse, Tomczak and Welling, *Attention-based Deep Multiple
Instance Learning*, ICML 2018, https://proceedings.mlr.press/v80/ilse18a.html.
The report cites the datasets and all additional research sources. AI assistance
supported software checks, analysis and report editing; Aaron Emmanuel and
George Mikhaiel are responsible for the submitted work. No license is asserted
over course materials or datasets beyond their original terms.
