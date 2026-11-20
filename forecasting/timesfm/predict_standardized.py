import argparse
import math
import random
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

LOOKBACK = 250
PREDICTION_LENGTH = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_TIMESFM_VERSION = "2.5"
DEFAULT_PREDICTION_MODE = "direct"
DEFAULT_ROLLING_FORECAST_HORIZON = 1
MODEL_IDS = {
    "2.5": "google/timesfm-2.5-200m-pytorch",
    "2.0": "google/timesfm-2.0-500m-pytorch",
}
OUTPUT_SUFFIXES = {
    "2.5": "_timesfm-2.5-200m_standardized_zero-shot",
    "2.0": "_timesfm-2.0-500m_standardized_zero-shot",
}


def require_runtime_dependencies(timesfm_version: str) -> None:
    required_modules = ["torch", "pandas"]
    if timesfm_version == "2.5":
        required_modules.append("timesfm")
    elif timesfm_version == "2.0":
        required_modules.append("transformers")

    missing = []
    for module_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "Missing TimesFM runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in timesfm/.venv, e.g. "
            + "`uv pip install -e '.[torch]' pandas`."
        )


def resolve_model_id(timesfm_version: str, model_id: Optional[str]) -> str:
    if model_id is not None:
        return model_id
    return MODEL_IDS[timesfm_version]


def get_output_suffix(timesfm_version: str) -> str:
    return OUTPUT_SUFFIXES[timesfm_version]


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def round_up_to_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# changed here
def standardize_inputs(
    inputs: list[np.ndarray],
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """
    Fit a separate StandardScaler to each context window.

    returns:
        standardized_inputs: list of (lookback,) arrays
        context_means:       shape (N,)
        context_scales:      shape (N,)
    """

    standardized_inputs = []

    context_means = np.empty(
        len(inputs),
        dtype=np.float32
    )

    context_scales = np.empty(
        len(inputs),
        dtype=np.float32
    )

    for i, x in enumerate(inputs):

        x = np.asarray(
            x,
            dtype=np.float32
        )

        scaler = StandardScaler()

        x_std = scaler.fit_transform(
            x.reshape(-1, 1)
        ).ravel()

        standardized_inputs.append(
            x_std.astype(np.float32)
        )

        context_means[i] = scaler.mean_[0]
        context_scales[i] = scaler.scale_[0]

    return (
        standardized_inputs,
        context_means,
        context_scales
    )


def load_input_csv(
    input_csv: Path,
    lookback: int,
    limit: Optional[int],
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)

    x_cols = [f"X_{i}" for i in range(lookback)]
    missing_cols = [col for col in x_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing input columns: {missing_cols[:5]}")

    inputs = [
        row.astype(np.float32)
        for row in df[x_cols].to_numpy(dtype=np.float32)
    ]
    return df, inputs


def load_timesfm_2p5_model(
    model_id: str,
    lookback: int,
    horizon: int,
    batch_size: int,
    device: str,
):
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.model.device = torch.device(device)
    model.model.device_count = torch.cuda.device_count() if device == "cuda" else 1
    model.model.to(model.model.device)
    model.compile(
        timesfm.ForecastConfig(
            max_context=lookback,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=batch_size,
        )
    )
    return model


def forecast_timesfm_2p5_inputs(
    model,
    inputs: list[np.ndarray],
    horizon: int
) -> tuple[np.ndarray, np.ndarray]:

    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=inputs
    )

    point_forecast = np.asarray(
        point_forecast,
        dtype=np.float32
    )

    quantile_forecast = np.asarray(
        quantile_forecast,
        dtype=np.float32
    )

    print(
        "TimesFM point shape:",
        point_forecast.shape
    )

    print(
        "TimesFM quantile shape:",
        quantile_forecast.shape
    )

    return point_forecast, quantile_forecast


def forecast_timesfm_2p5_inputs_rolling_multi_horizon(
    model,
    inputs: list[np.ndarray],
    horizon: int,
    rolling_forecast_horizon: int,
) -> np.ndarray:
    rolling_inputs = [np.asarray(row, dtype=np.float32).copy() for row in inputs]
    predictions = np.empty((len(rolling_inputs), horizon), dtype=np.float32)

    for step in range(horizon):
        step_predictions = forecast_timesfm_2p5_inputs(
            model,
            list(rolling_inputs),
            rolling_forecast_horizon,
        )[:, 0]
        if len(step_predictions) != len(rolling_inputs):
            raise ValueError(
                f"TimesFM 2.5 rolling prediction rows {len(step_predictions)} "
                f"do not match input rows {len(rolling_inputs)}."
            )
        predictions[:, step] = step_predictions
        rolling_inputs = [
            np.concatenate([row[1:], np.asarray([prediction], dtype=np.float32)])
            for row, prediction in zip(rolling_inputs, step_predictions)
        ]
        print(f"Rolling step: {step + 1}/{horizon}", flush=True)

    return predictions


def load_timesfm_2p0_model(model_id: str, lookback: int, horizon: int, device: str):
    from transformers import TimesFmConfig, TimesFmModelForPrediction

    config = TimesFmConfig.from_pretrained(model_id)
    context_len = round_up_to_multiple(lookback, config.patch_length)
    config.context_length = context_len
    model = TimesFmModelForPrediction.from_pretrained(
        model_id,
        config=config,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.context_len = context_len
    model.horizon_len = horizon
    model.eval()
    return model


def forecast_timesfm_2p0_inputs(
    model,
    inputs: list[np.ndarray],
    horizon: int,
    batch_size: int,
    freq: int,
) -> np.ndarray:
    import torch

    forecasts = []
    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        past_values = [
            torch.tensor(ts, dtype=torch.float32).to(model.device)
            for ts in batch_inputs
        ]
        frequency_input = torch.tensor(
            [freq] * len(batch_inputs),
            dtype=torch.long,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(
                past_values=past_values,
                freq=frequency_input,
                return_dict=True,
            )
        forecasts.append(outputs.mean_predictions[:, :horizon].float().cpu().numpy())
    return np.asarray(np.concatenate(forecasts, axis=0), dtype=np.float32)


def forecast_timesfm_2p0_inputs_rolling_multi_horizon(
    model,
    inputs: list[np.ndarray],
    horizon: int,
    rolling_forecast_horizon: int,
    batch_size: int,
    freq: int,
) -> np.ndarray:
    rolling_inputs = [np.asarray(row, dtype=np.float32).copy() for row in inputs]
    predictions = np.empty((len(rolling_inputs), horizon), dtype=np.float32)

    for step in range(horizon):
        step_predictions = forecast_timesfm_2p0_inputs(
            model=model,
            inputs=list(rolling_inputs),
            horizon=rolling_forecast_horizon,
            batch_size=batch_size,
            freq=freq,
        )[:, 0]
        if len(step_predictions) != len(rolling_inputs):
            raise ValueError(
                f"TimesFM 2.0 rolling prediction rows {len(step_predictions)} "
                f"do not match input rows {len(rolling_inputs)}."
            )
        predictions[:, step] = step_predictions
        rolling_inputs = [
            np.concatenate([row[1:], np.asarray([prediction], dtype=np.float32)])
            for row, prediction in zip(rolling_inputs, step_predictions)
        ]
        print(f"Rolling step: {step + 1}/{horizon}", flush=True)

    return predictions


def build_output_path(
    input_csv: Path,
    output_dir: Path,
    output_path: Optional[Path],
    suffix: str,
) -> Path:
    if output_path is not None:
        return output_path
    return output_dir / f"{input_csv.stem}{suffix}{input_csv.suffix}"


def save_forecast_csv(
    df: pd.DataFrame,
    predictions: np.ndarray,
    input_csv: Path,
    output_dir: Path,
    output_path: Optional[Path],
    horizon: int,
    suffix: str,
) -> Path:
    y_cols = [f"Y_{i}" for i in range(horizon)]
    missing_cols = [col for col in y_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing output columns: {missing_cols[:5]}")

    expected_shape = (len(df), horizon)
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match output shape {expected_shape}"
        )

    result_df = df.copy()
    result_df.loc[:, y_cols] = predictions

    output_csv = build_output_path(input_csv, output_dir, output_path, suffix=suffix)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    return output_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--timesfm-version",
        choices=sorted(MODEL_IDS),
        default=DEFAULT_TIMESFM_VERSION,
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=PREDICTION_LENGTH)
    parser.add_argument(
        "--prediction-mode",
        choices=["direct", "rolling_multi_horizon"],
        default=DEFAULT_PREDICTION_MODE,
    )
    parser.add_argument("--rolling-forecast-horizon", type=int, default=DEFAULT_ROLLING_FORECAST_HORIZON)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--freq", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_dependencies(args.timesfm_version)
    set_seed(args.seed)
    if args.rolling_forecast_horizon <= 0:
        raise ValueError("--rolling-forecast-horizon must be positive.")
    model_id = resolve_model_id(args.timesfm_version, args.model_id)
    output_suffix = get_output_suffix(args.timesfm_version)
    device = resolve_device()

    df, inputs = load_input_csv(args.input_csv, args.lookback, args.limit)
    print(f"CSV data successfully loaded, total samples: {len(df)}")
    print(f"Using TimesFM {args.timesfm_version} model: {model_id}")
    print(
        f"Inference parameters: horizon={args.horizon}, prediction_mode={args.prediction_mode}, "
        f"rolling_forecast_horizon={args.rolling_forecast_horizon}, batch_size={args.batch_size}, "
        f"seed={args.seed}, freq={args.freq}, device={device}"
    )

    # added
    inputs_std, context_means, context_scales = (
        standardize_inputs(inputs)
    )

    # Sanity checks
    x_std_check = np.stack(
        inputs_std,
        axis=0
    )

    print(
        "Standardized context mean range:",
        x_std_check.mean(axis=1).min(),
        x_std_check.mean(axis=1).max()
    )

    print(
        "Standardized context std range:",
        x_std_check.std(axis=1).min(),
        x_std_check.std(axis=1).max()
    )

    model_horizon = max(args.horizon, args.rolling_forecast_horizon)
    if args.timesfm_version == "2.5":
        model = load_timesfm_2p5_model(
            model_id=model_id,
            lookback=args.lookback,
            horizon=model_horizon,
            batch_size=args.batch_size,
            device=device,
        )

        if args.prediction_mode == "rolling_multi_horizon":
            predictions = forecast_timesfm_2p5_inputs_rolling_multi_horizon(
                model=model,
                inputs=inputs,
                horizon=args.horizon,
                rolling_forecast_horizon=args.rolling_forecast_horizon,
            )
        else:
            predictions_std, quantiles_std = forecast_timesfm_2p5_inputs(
                model,
                inputs_std,
                args.horizon
            )

            predictions = (
                predictions_std
                * context_scales[:, None]
                + context_means[:, None]
            )

            quantiles = (
                quantiles_std
                * context_scales[:, None, None]
                + context_means[:, None, None]
            )

            if predictions.shape != (
                len(df),
                args.horizon
            ):
                raise ValueError(
                    f"Unexpected inverse-transformed "
                    f"prediction shape: {predictions.shape}"
                )

            if quantiles.shape[:2] != (
                len(df),
                args.horizon
            ):
                raise ValueError(
                    f"Unexpected inverse-transformed "
                    f"quantile shape: {quantiles.shape}"
                )

            if not np.all(np.isfinite(predictions)):
                raise ValueError(
                    "Non-finite TimesFM predictions."
                )

            if not np.all(np.isfinite(quantiles)):
                raise ValueError(
                    "Non-finite TimesFM quantiles."
                )

    elif args.timesfm_version == "2.0":
        model = load_timesfm_2p0_model(
            model_id=model_id,
            lookback=args.lookback,
            horizon=model_horizon,
            device=device,
        )
        if args.prediction_mode == "rolling_multi_horizon":
            predictions = forecast_timesfm_2p0_inputs_rolling_multi_horizon(
                model=model,
                inputs=inputs,
                horizon=args.horizon,
                rolling_forecast_horizon=args.rolling_forecast_horizon,
                batch_size=args.batch_size,
                freq=args.freq,
            )
        else:
            predictions = forecast_timesfm_2p0_inputs(
                model=model,
                inputs=inputs,
                horizon=args.horizon,
                batch_size=args.batch_size,
                freq=args.freq,
            )
    else:
        raise ValueError(f"Unsupported TimesFM version: {args.timesfm_version}")

    output_csv = save_forecast_csv(
        df=df,
        predictions=predictions,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_path=args.output_path,
        horizon=args.horizon,
        suffix=output_suffix,
    )
    print(f"TimesFM zero-shot prediction results have been saved to: {output_csv}")

    quantile_path = (
            args.output_dir /
            f"{args.input_csv.stem}"
            f"{output_suffix}_quantiles.npz"
        )
    
    np.savez_compressed(
        quantile_path,
        quantiles=quantiles,
        context_means=context_means,
        context_scales=context_scales,
    )

    print(
        "TimesFM native quantile forecasts saved to:",
        quantile_path
    )


if __name__ == "__main__":
    main()
