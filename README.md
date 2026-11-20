# BForTProb · Anonymous review code

This archive contains the benchmark implementation, evaluation notebooks, and supplementary analyses. It includes saved score tables for the main daily and weekly comparisons and a standalone scoring example with its two input files.

## Quick check

From the extracted archive directory, run:

```sh
python -m pip install -r examples/qss/requirements.txt
python examples/qss/reproduce_qss.py
```

The expected daily AUD/USD Moirai QSS at the 0.10 quantile is **−0.00684772** across 495 forecast origins. The example evaluates saved predictions and requires no model checkpoint.

## Code map

| Task | Entry point |
| --- | --- |
| Build chronological windows and train/test splits | `forecasting/data_split.ipynb` |
| Fit ARIMA, Gaussian GARCH, LightGBM, and LSTM | Corresponding model directories under `forecasting/` |
| Run the zero-shot models | Prediction scripts under `forecasting/` |
| Score RMSE, MAE, persistence-normalized MAE, nine-quantile CRPS approximation, and lower-tail QSS | `forecasting/evaluation/metrics.ipynb` |
| Compare average ranks and run Friedman–Nemenyi analyses | `forecasting/evaluation/statistical_significance.ipynb` |
| Fit path-simulated Student-t GARCH | `notebooks/student_t_garch.ipynb` |
| Compare zero-return and historical empirical references | `notebooks/zero_return.ipynb` and `notebooks/empirical_9_quantile_baseline.ipynb` |
| Evaluate quantile reliability, coverage, width, and interval scores | `notebooks/calibration_diagnostics.ipynb` and `notebooks/MSIS.ipynb` |
| Estimate paired effects, time-block uncertainty, and rank sensitivity | `notebooks/QSS_effect_size_summary.ipynb`, `notebooks/calendar_bootstrap.ipynb`, `notebooks/daily_synchronised_calendar_bootstrap.py`, and `notebooks/ranking_robustness.ipynb` |
| Check scaling, sample counts, and mean versus median forecasts | `notebooks/full_normalization_extended.ipynb`, `notebooks/convergence_extended.ipynb`, and `notebooks/TSFM_mean_vs_median.ipynb` |
| Run Timestral experiments | `forecasting/` and `notebooks/timestral_full_evaluation.ipynb` |

Filenames that contain `with_timestral` refer to expanded comparisons.

## Reproduce the rank comparisons from saved scores

Install the analysis dependencies with `python -m pip install -r requirements-analysis.txt`, launch JupyterLab from the extracted archive root, and open `forecasting/evaluation/statistical_significance.ipynb`.

1. Run the path setup cell and the **Overall CRPS and MASE** and **Overall RMSE and QSS** sections. Both daily and weekly inputs are bundled under `BForTFin_results/standardised/output_<frequency>/`.
2. For the horizon-specific comparisons, run the testing cells that read the bundled horizon CSV tables. The earlier cell that rebuilds CRPS and MASE horizon tables expects per-model forecast outputs from a full run.
3. For weekly results, set `BFORTPROB_FREQUENCY=weekly` before starting JupyterLab. The default is daily. The path setup cell uses this setting for the main pipeline. Supplementary notebooks with explicit daily and weekly sections retain those settings.

The original labels in saved artifacts use MASE in places for the persistence-normalized MAE measure. Use the metric definitions in the manuscript and rebuttal when interpreting these files.

## Rerun forecasting and metrics

1. Stage the chronological log-return series in `data/chronological_log_returns/output_<frequency>/<asset-class>/`. The splitter expects one numeric target column in each input CSV, ordered from oldest to newest. Run `forecasting/data_split.ipynb` for each frequency. It writes the complete supervised windows to `BForTFin_log_return/output_<frequency>/` and the train/test files to `BForTFin_log_return/arima_split_datasets/output_<frequency>/`.
2. Run the fitted-model notebooks and the zero-shot entry points for each asset class. Prediction scripts expose flags for the input CSV, output path, and model configuration. Put outputs in the frequency, model, and asset-class directories named in the evaluation notebooks. The submitted Chronos checkpoint is `amazon/chronos-t5-small`, selected explicitly with `--model-id`. For Moirai-MoE, select `Salesforce/moirai-moe-1.0-R-base` and the corresponding model family explicitly. The Lag-Llama runner accepts a checkpoint path via `--checkpoint`.
3. Run the relevant metric cells against the matching held-out observations and forecasts, then run the rank and supplementary analyses. Some supplementary notebooks expect generated forecasts, dated source data, or intermediate CSVs. Their input paths are declared near the start of each analysis. 

The analysis dependencies cover the CPU evaluation notebooks. The forecasting backends have separate environments. The scripts reference Chronos Forecasting, Uni2TS, Lag-Llama and GluonTS, the TimesFM source, Transformers for Sundial and Timestral, and Toto 2.0. The corresponding model folders preserve available dependency files. If using the bundled TimesFM implementation, put `model_experiments_git/timesfm/src` on `PYTHONPATH`. Match local model dependencies and accelerator support to the checkpoint implementation being run.

