# AERIS: screening future extreme intermittency in high-renewable power systems

Code accompanying the article *Future Extreme Intermittency Screening and Decision
Support for High-Penetration Renewable Power Systems* (Discover Applied Sciences,
under review).

AERIS scores each 15-minute interval by the risk that the next interval is an
extreme renewable-intermittency event. It combines a residual dilated temporal
convolution encoder, attention pooling with a context gate, a variational latent
state and an auxiliary regression head on the continuous event score. A kernel
density estimate on the latent states adds a confidence value to every alert.
The model is evaluated on one year (2025) of Belgian transmission-system data
against tabular, sequence, anomaly-detection and persistence baselines.

## Contents

```
scripts/revision/     the complete experiment pipeline (rev00 ... rev32c)
  rev_common.py       data pipeline, event label, chronological split, score store
  rev_models.py       AERIS, its ablation variants and the sequence baselines
  rev_eval.py         metrics, moving-block bootstrap, alert budgets, costs
outputs/revision/
  tables/             every result table (CSV / JSON) reported in the article
  figures/            the article figures
  scores/             validation and test scores of every model, the deployed
                      AERIS checkpoint (AERIS_deployed.pt) and its latent states
dataset/              place the raw Elia files here (see dataset/README.md)
```

The shipped `outputs/revision/` lets you recompute every metric, significance test
and figure without retraining anything.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
```

The code was run with Python 3.13 on CPU; a GPU is used automatically if present.

## Data

Download ODS031, ODS032 and ODS134 from the Elia Open Data Platform as described
in `dataset/README.md`, then build the quarter-hour feature table (35,040 rows,
146 columns):

```bash
python -m scripts.revision.rev00_build_feature_table
```

Run all commands from the repository root.

## Reproducing the results

Evaluation from the shipped scores, without retraining (`rev07` also needs the
feature table built above; the other three need only the shipped results):

```bash
python -m scripts.revision.rev07_evaluation
python -m scripts.revision.rev11_extra_figures
python -m scripts.revision.rev12_make_tables
python -m scripts.revision.rev14_report
```

Full pipeline, from the raw data to every table and figure (about a day on eight
CPU threads; the random searches `rev09`, `rev09b` and `rev10*` and the horizon
study `rev21`/`rev27` take most of it). It runs every step in dependency order
with the options used for the article and stops at the first failure:

```bash
bash scripts/revision/run_pipeline.sh            # or: ... run_pipeline.sh <first-step>
```

Each script writes its results to `outputs/revision/` and overwrites the shipped
files. Independent steps can be run side by side; `REV_TORCH_THREADS=<n>` then caps
the threads PyTorch uses in each process.

| Step | Scripts | Purpose |
| --- | --- | --- |
| Data and label | `rev00_build_feature_table`, `rev01_label_validation`, `rev01b_score_structure`, `rev08_feature_dictionary` | feature table from the raw files, label prevalence and external validation, structure of the composite score, numerical check of every derived feature |
| Tabular baselines | `rev02_baselines`, `rev09_baseline_search`, `rev09b_regression_baselines` | default and tuned linear, tree, boosting (HistGBM, XGBoost, LightGBM; classification and regression formulations), anomaly-detection and persistence baselines |
| AERIS selection | `rev10_aeris_search`, `rev10b_finalise_aeris`, `rev10c_multitask_search`, `rev17_feature_selection`, `rev16_ensembles`, `rev22_dualbranch_search`, `rev23_final_selection`, `rev28_deploy_best` | validation-only configuration search, input selection, seed ensembling, dual-branch variant, greedy architecture search, deployment of the validation winner |
| Sequence models and ablation | `rev03_sequence_models`, `rev04_ablation` | LSTM, GRU, Transformer and TCN under a common recipe; component ablation |
| Robustness | `rev05_robustness`, `rev06_kde_confidence`, `rev06b_kde_confidence`, `rev31_block_sensitivity` | event quantiles, information-availability audit, rolling origin, perturbations, density confidence layer, bootstrap block length |
| Horizons | `rev21_horizon_retune`, `rev27_horizon_significance`, `rev30_horizon_vs_persistence`, `rev33_horizon_lightgbm` | per-horizon re-selection from 15 to 120 minutes and paired tests against the tuned regressors and the persistence rule |
| Evaluation | `rev26_calibrate`, `rev07_evaluation`, `rev32_audit_tuned_regressors`, `rev32b_quantiles_tuned_regressor`, `rev32c_audit_tuned_classifier`, `rev11_extra_figures`, `rev34_schematic_figures`, `rev12_make_tables`, `rev14_report` | calibration, metrics with bootstrap intervals, paired tests, alert budgets, costs, audits against the tuned boosting models, result and schematic figures, tables |

The docstring at the top of each script describes what it does and what it writes;
`run_pipeline.sh` lists the options used for the article.

## Note on the photovoltaic series

ODS032 lists Belgium together with its regions and provinces for every
quarter-hour. `rev00_build_feature_table` uses the national record only; summing
all records, as an earlier version of this pipeline did, counts photovoltaic
generation about three times.

## Design

* One data pipeline: `rev_common.build_matrices` builds the windows, labels and
  chronological 70/10/20 split, with switches for every audit (label
  normalisation, threshold source, excluded inputs, publication lag, event
  quantile, horizon).
* One training recipe: `rev_models.ScreeningNet` implements AERIS, its ablation
  variants and the sequence baselines behind switches, so they share optimiser,
  schedule, loss bookkeeping, epoch budget and checkpoint rule.
* One score store: every model writes its scores through `rev_common.save_scores`,
  so all metrics and tests are computed from identical predictions.
* All model selection uses the validation partition only; the test partition is
  touched once, for the reported numbers.

## Citation

If you use this code, please cite the article (reference to be added on
publication).

## License

MIT, see `LICENSE`. The Elia data are not part of this repository and remain
subject to Elia's terms of use.
