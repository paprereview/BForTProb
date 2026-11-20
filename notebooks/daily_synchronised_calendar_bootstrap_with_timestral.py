"""Daily synchronised calendar-grid moving-block bootstrap (q0.1 loss).

Secondary sensitivity analysis restricted to dates shared by all 72 daily series.
Per-origin differences are empirical q0.1 reference pinball loss minus model loss,
averaged across 20 forecast horizons. All series receive identical resampled dates.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r".")
BASE_DIR = PROJECT_ROOT / "BForTFin_results" / "standardised" / "output_daily"
DATA_DIR = PROJECT_ROOT / "BForTFin_log_return" / "arima_split_datasets" / "output_daily"
DATE_CSV = PROJECT_ROOT / "BForTFin_log_return" / "evaluation_date_provenance" / "daily_forecast_origin_dates.csv"
OUT_DIR = BASE_DIR / "synchronised_calendar_bootstrap"
LOOKBACK, HORIZON, TAU = 250, 20, 0.1
N_BOOT, BLOCK_LENGTHS, SEED = 2000, [10, 20, 40], 42

ASSET_FOLDERS = {"G7": "FX", "EM": "EM FX", "MSCI": "MSCI", "Rates": "Rates",
                 "REIT": "REIT", "Equities": "Equities", "ETF": "ETF", "Crypto": "Crypto"}
CLASS_TO_FOLDER = {v: k for k, v in ASSET_FOLDERS.items()}
MODEL_FOLDERS = {
    "GARCH(1,1)": BASE_DIR / "garch_standardized_train_test",
    "Student-t GARCH": BASE_DIR / "garch_t_1_1_standardized_train_test",
    "LightGBM": BASE_DIR / "lightgbm_standardized_train_test",
    "Sundial": BASE_DIR / "sundial_standardized_train_test",
    "Lag-Llama": BASE_DIR / "lag-llama_standardized_train_test",
    "Chronos-T5": BASE_DIR / "chronos_standardized_train_test",
    "Toto": BASE_DIR / "toto_standardized_train_test",
    "TimesFM": BASE_DIR / "timesfm_standardized_train_test",
    "Moirai": BASE_DIR / "moirai_standardized_train_test",
    "Moirai-MoE": BASE_DIR / "moirai_moe_standardized_train_test",
    "Timestral": BASE_DIR / "timestral_standardized_train_test",
}



# Quantile-file suffix used by each model.
MODEL_SUFFIXES = {
    "GARCH(1,1)": "_garch_1_1",
    "Student-t GARCH": "_garch-t-1-1-simulated",
    "LightGBM": "_lightgbm",
    "Sundial": "_sundial",
    "Lag-Llama": "_lag-llama",
    "Chronos-T5": "_chronos-t5-small",
    "Toto": "_toto",
    "TimesFM": "_timesfm-2.5-200m",
    "Timestral": "_timestral-209m_standardized_zero-shot",
    "Moirai": "_moirai-2.0-r-small",
    "Moirai-MoE": "_moirai-moe-1.0-r-base",
    "ARIMA": "_arima",
    "LSTM": "_lstm",
}


def pinball(y, q):
    error = y - q
    return np.maximum(TAU * error, (TAU - 1.0) * error)


def clean_garch_name(name):
    return (name.lower().replace(" historical data", "")
            .replace("_no_preprocess_lookback-250_horizon-20_slide-1", "")
            .replace(" ", "_").replace(".", ""))


def one_match(paths, description):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {description}; found {len(paths)}: {paths[:5]}")
    return paths[0]


def get_test_path(folder, dataset):
    test_dir = DATA_DIR / folder
    exact = test_dir / f"test_{dataset}_no_preprocess_lookback-250_horizon-20_slide-1.csv"
    if exact.exists():
        return exact
    return one_match(test_dir.glob(f"test_{dataset}*.csv"), f"test CSV for {folder}/{dataset}")


def get_quantile_path(model, folder, dataset):
    root = BASE_DIR / MODEL_FOLDERS[model] / folder
    if model == "GARCH(1,1)":
        clean = clean_garch_name(dataset)
        exact = root / f"{folder.lower()}_{clean}_garch_1_1_train_test_quantiles.npz"
        if exact.exists():
            return exact
        return one_match(root.glob(f"*{clean}*garch_1_1_train_test*quantiles.npz"),
                         f"GARCH quantiles for {folder}/{dataset}")
    suffix = MODEL_SUFFIXES[model]
    return one_match(root.glob(f"test_{dataset}_no_preprocess_lookback-250_horizon-20_slide-1{suffix}_quantiles.npz"),
                     f"{model} quantiles for {folder}/{dataset}")


def load_losses(dataset, asset_class):
    folder = CLASS_TO_FOLDER[asset_class]
    test_path = get_test_path(folder, dataset)
    test = pd.read_csv(test_path)
    x = test[[f"X_{j}" for j in range(LOOKBACK)]].to_numpy(dtype=float)
    y = test[[f"Y_{j}" for j in range(HORIZON)]].to_numpy(dtype=float)
    q_ref = np.quantile(x, TAU, axis=1)[:, None]
    ref_loss = pinball(y, q_ref)
    losses = {}
    for model in MODEL_FOLDERS:
        path = get_quantile_path(model, folder, dataset)
        with np.load(path) as f:
            quantiles = np.asarray(f["quantiles"], dtype=float)
            levels = np.asarray(f["quantile_levels"], dtype=float)
        q10_matches = np.flatnonzero(np.isclose(levels, TAU, atol=1e-8, rtol=0))
        if len(q10_matches) != 1:
            raise ValueError(f"{model}/{dataset}: expected exactly one 0.1 quantile, got {levels}")
        if quantiles.ndim != 3 or quantiles.shape[:2] != y.shape or quantiles.shape[2] != len(levels):
            raise ValueError(f"{model}/{dataset}: Q shape {quantiles.shape}, Y {y.shape}, levels {len(levels)}")
        q10 = quantiles[:, :, q10_matches[0]]
        diff = (ref_loss - pinball(y, q10)).mean(axis=1)
        if not np.isfinite(diff).all():
            raise ValueError(f"{model}/{dataset}: nonfinite losses")
        losses[model] = diff
    return len(test), losses


def shared_block_indices(n_dates, block_len, n_boot, rng):
    if block_len > n_dates:
        raise ValueError(f"Block {block_len} exceeds {n_dates} common dates")
    n_blocks = int(np.ceil(n_dates / block_len))
    starts = rng.integers(0, n_dates - block_len + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    return (starts[:, :, None] + offsets).reshape(n_boot, -1)[:, :n_dates]


def main():
    dates = pd.read_csv(DATE_CSV, parse_dates=["forecast_origin_date"])
    required = {"asset_class", "dataset", "origin_idx", "forecast_origin_date"}
    if not required.issubset(dates.columns):
        raise ValueError(f"Missing origin mapping columns: {required - set(dates.columns)}")
    if dates.duplicated(["dataset", "forecast_origin_date"]).any():
        raise ValueError("Duplicate dates for a dataset")
    if dates.duplicated(["dataset", "origin_idx"]).any():
        raise ValueError("Duplicate origin indices for a dataset")
    meta = dates[["dataset", "asset_class"]].drop_duplicates().sort_values("dataset")
    if meta.dataset.duplicated().any():
        raise ValueError("Ambiguous dataset names across asset classes; use compound keys")
    if len(meta) != 72 or set(meta.asset_class) != set(CLASS_TO_FOLDER):
        raise ValueError(f"Expected 72 daily series and 8 classes; got {len(meta)} and {set(meta.asset_class)}")
    counts = dates.groupby("forecast_origin_date")["dataset"].nunique()
    common = pd.DatetimeIndex(counts[counts == len(meta)].index).sort_values()
    if len(common) < max(BLOCK_LENGTHS):
        raise ValueError(f"Only {len(common)} common origin dates; adjust block sensitivity")
    if len(common) < 60:
        raise ValueError("Common-date sample unexpectedly short; inspect mapping")
    print(f"Common dates: {len(common)} from {common[0].date()} to {common[-1].date()}")
    # Each row is one series, each column one common forecast-origin date.
    # All models share the exact same dates and forecast targets.
    model_names = list(MODEL_FOLDERS)
    cube = np.empty((len(model_names), len(meta), len(common)), dtype=float)
    series_records = []
    for j, row in enumerate(meta.itertuples(index=False)):
        dataset, asset_class = row.dataset, row.asset_class
        per_series = dates[dates.dataset == dataset].sort_values("origin_idx")
        n_origins, loss_by_model = load_losses(dataset, asset_class)
        if len(per_series) != n_origins or not np.array_equal(per_series.origin_idx.to_numpy(), np.arange(n_origins)):
            raise ValueError(f"Origin ordering/count mismatch for {dataset}: map {len(per_series)}, test {n_origins}")
        by_date = per_series.set_index("forecast_origin_date")
        loc = by_date.loc[common, "origin_idx"].to_numpy(dtype=int)
        for k, model in enumerate(model_names):
            cube[k, j] = loss_by_model[model][loc]
        series_records.append({"asset_class": asset_class, "dataset": dataset, "n_full_origins": n_origins,
                               "n_common_dates": len(common)})
        print(f"Loaded {j+1:02d}/{len(meta)}: {dataset}")
    classes = meta.asset_class.to_numpy()
    series_w = np.ones(len(meta), dtype=float) / len(meta)
    class_w = np.array([1.0 / (len(set(classes)) * np.sum(classes == cl)) for cl in classes])
    assert np.isclose(class_w.sum(), 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series_records).to_csv(OUT_DIR / "daily_common_sample_inventory_with_timestral.csv", index=False)
    pd.DataFrame({"forecast_origin_date": common}).to_csv(OUT_DIR / "daily_common_origin_dates_with_timestral.csv", index=False)
    summary = []
    for block in BLOCK_LENGTHS:
        rng = np.random.default_rng(SEED + block)
        draw_idx = shared_block_indices(len(common), block, N_BOOT, rng)
        # cube shape: model x dataset x date. Shared draw_idx for *every* model and dataset.
        for label, weights in [("equal_series", series_w), ("equal_asset_class", class_w)]:
            # At each shared calendar date, aggregate paired loss differences across series.
            aggregate_by_date = np.einsum("msd,s->md", cube, weights)
            point = aggregate_by_date.mean(axis=1)
            boot = aggregate_by_date[:, draw_idx].mean(axis=2)  # model x bootstrap
            lower, upper = np.quantile(boot, [0.025, 0.975], axis=1)
            for k, model in enumerate(model_names):
                summary.append({"model": model, "weighting": label, "block_length": block,
                                "n_boot": N_BOOT, "n_datasets": len(meta), "n_common_dates": len(common),
                                "first_common_date": common[0].date(), "last_common_date": common[-1].date(),
                                "mean_loss_difference": point[k], "ci_lower_95": lower[k], "ci_upper_95": upper[k],
                                "significantly_positive": bool(lower[k] > 0),
                                "significantly_negative": bool(upper[k] < 0)})
    res = pd.DataFrame(summary)
    res.to_csv(OUT_DIR / "daily_synchronised_q10_bootstrap_all_blocks_with_timestral.csv", index=False)
    primary = res[res.block_length == 20].sort_values(["weighting", "mean_loss_difference"], ascending=[True, False])
    primary.to_csv(OUT_DIR / "daily_synchronised_q10_bootstrap_primary_with_timestral.csv", index=False)
    print("\nPRIMARY RESULTS (BLOCK 20)")
    print(primary[["model", "weighting", "mean_loss_difference", "ci_lower_95", "ci_upper_95"]].to_string(index=False))
    print(f"\nSaved results in {OUT_DIR}")
    print("INTERPRETATION: This is a restricted common-date sensitivity analysis, not the full-sample estimate.")
    print("Blocks comprise consecutive positions on the shared observed-date grid, not calendar-day intervals.")


if __name__ == "__main__":
    main()
