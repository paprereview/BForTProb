import argparse
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_SUFFIX = "_toto_standardized_zero-shot"

LOOKBACK = 250
PREDICTION_LENGTH = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_ID = "Datadog/Toto-2.0-313m"
DEFAULT_DECODE_BLOCK_SIZE = 768
DEFAULT_PATCH_SIZE = 32
MEDIAN_QUANTILE_INDEX = 4
DEFAULT_PREDICTION_MODE = "direct"
DEFAULT_ROLLING_FORECAST_HORIZON = 1
DEFAULT_SEGMENT_ROLLING_STEPS = 5
DEFAULT_ENSEMBLE_DIRECT_WEIGHT = 0.5
DEFAULT_ROLLING_COMBINE = "median"
DEFAULT_ROLLING_QUANTILE_INDICES = "2,4,6"


def require_runtime_dependencies() -> None:
    missing = []
    for module_name in ("torch", "toto2", "pandas"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "Missing Toto runtime dependencies: "
            + ", ".join(missing)
            + ". Toto 2.0 requires Python 3.12+. Create toto/.venv with "
            + "`uv venv --python 3.12 toto/.venv`, then install torch, pandas, and toto2."
        )


def get_default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return get_default_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is not available.")
    if (
        device == "mps"
        and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    ):
        raise ValueError("Requested --device mps, but MPS is not available.")
    return device


def resolve_torch_dtype(dtype: str, device: str):
    import torch

    if dtype == "auto":
        dtype = "float32"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype], dtype


def load_input_csv(
    input_csv: Path,
    lookback: int,
    limit: Optional[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)

    x_cols = [f"X_{i}" for i in range(lookback)]
    missing_cols = [col for col in x_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing input columns: {missing_cols[:5]}")

    contexts = df[x_cols].to_numpy(dtype=np.float32)
    return df, contexts

def standardize_contexts(
    contexts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a separate StandardScaler to each context window.

    Parameters
    ----------
    contexts:
        shape (N, lookback)

    Returns
    -------
    contexts_std:
        shape (N, lookback)

    context_means:
        shape (N,)

    context_scales:
        shape (N,)
    """

    contexts = np.asarray(
        contexts,
        dtype=np.float32
    )

    contexts_std = np.empty_like(
        contexts,
        dtype=np.float32
    )

    context_means = np.empty(
        len(contexts),
        dtype=np.float32
    )

    context_scales = np.empty(
        len(contexts),
        dtype=np.float32
    )

    for i, x in enumerate(contexts):

        scaler = StandardScaler()

        x_std = scaler.fit_transform(
            x.reshape(-1, 1)
        ).ravel()

        contexts_std[i] = x_std
        context_means[i] = scaler.mean_[0]
        context_scales[i] = scaler.scale_[0]

    return (
        contexts_std,
        context_means,
        context_scales
    )

def load_toto_model(model_id: str, device: str, dtype: str):
    from toto2 import Toto2Model

    torch_dtype, dtype_name = resolve_torch_dtype(dtype, device)
    print(f"Toto dtype: {dtype_name}")
    try:
        model = Toto2Model.from_pretrained(model_id, local_files_only=True)
        print("Using Hugging Face cache to load Toto model")
    except (OSError, TypeError):
        print("Local cache not found, downloading Toto model from Hugging Face")
        model = Toto2Model.from_pretrained(model_id)
    return model.to(device=device, dtype=torch_dtype)


def make_toto_batch(data, device: str, dtype, patch_size: int = DEFAULT_PATCH_SIZE):
    import torch

    if torch.is_tensor(data):
        target = data.to(device=device, dtype=dtype)
    else:
        target = torch.tensor(data, dtype=dtype, device=device)
    if target.ndim == 2:
        target = target.unsqueeze(1)
    if target.ndim != 3:
        raise ValueError(f"Toto target must have shape (batch, n_variates, time), got {tuple(target.shape)}")

    target_mask = torch.ones_like(target, dtype=torch.bool)
    padding_length = (patch_size - (target.shape[-1] % patch_size)) % patch_size
    if padding_length:
        pad_values = target[..., :1].repeat_interleave(padding_length, dim=-1)
        target = torch.cat([pad_values, target], dim=-1)
        pad_mask = torch.zeros(
            *target_mask.shape[:-1],
            padding_length,
            dtype=torch.bool,
            device=target_mask.device,
        )
        target_mask = torch.cat([pad_mask, target_mask], dim=-1)

    series_ids = torch.zeros(target.shape[0], target.shape[1], dtype=torch.long, device=target.device)
    has_missing_values = not bool(target_mask.all().item())
    return (
        {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        },
        has_missing_values,
    )


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Quantile indices must be non-negative.")
    return values


def parse_float_list(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated float list.") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Weights must be non-negative.")
    if sum(values) == 0:
        raise argparse.ArgumentTypeError("At least one weight must be greater than zero.")
    return values


def reduce_quantile_forecast(
    quantiles,
    horizon: int,
    quantile_index: int = MEDIAN_QUANTILE_INDEX,
) -> np.ndarray:
    values = quantiles[quantile_index, :, 0, :horizon]
    return values.detach().float().cpu().numpy().astype(np.float32)


def forecast_contexts(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
    decode_block_size: int,
    patch_size: int,
    device: str,
    quantile_index: int = MEDIAN_QUANTILE_INDEX,
) -> np.ndarray:
    import torch

    model.eval()
    dtype = next(model.parameters()).dtype
    predictions = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            toto_batch, has_missing_values = make_toto_batch(
                batch,
                device=device,
                dtype=dtype,
                patch_size=patch_size,
            )
            quantiles = model.forecast(
                toto_batch,
                horizon=horizon,
                decode_block_size=decode_block_size,
                has_missing_values=has_missing_values,
            )
            quantile_values = (
                quantiles[..., :horizon]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            print(
                "Raw Toto quantile shape:",
                quantile_values.shape
            )

            predictions.append(quantile_values)
            print(f"Completed prediction: {min(start + batch_size, len(contexts))}/{len(contexts)}", flush=True)

    all_quantiles = np.concatenate(
        predictions,
        axis=1
    ).astype(np.float32)

    print(
        "Final Toto quantile shape:",
        all_quantiles.shape
    )

    return all_quantiles


def forecast_contexts_rolling(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
    decode_block_size: int,
    patch_size: int,
    device: str,
    rolling_forecast_horizon: int = DEFAULT_ROLLING_FORECAST_HORIZON,
    quantile_index: int = MEDIAN_QUANTILE_INDEX,
) -> np.ndarray:
    rolling_contexts = contexts.copy().astype(np.float32)
    predictions = np.empty((len(contexts), horizon), dtype=np.float32)
    step_horizon = max(1, rolling_forecast_horizon)

    for step in range(horizon):
        step_predictions = forecast_contexts(
            model=model,
            contexts=rolling_contexts,
            horizon=step_horizon,
            batch_size=batch_size,
            decode_block_size=decode_block_size,
            patch_size=patch_size,
            device=device,
            quantile_index=quantile_index,
        )[:, 0]
        predictions[:, step] = step_predictions
        rolling_contexts = np.concatenate(
            [rolling_contexts[:, 1:], step_predictions[:, None]],
            axis=1,
        ).astype(np.float32)
        print(f"Completed rolling step: {step + 1}/{horizon}", flush=True)

    return predictions


def combine_forecast_paths(
    paths: np.ndarray,
    combine: str,
    quantile_indices: list[int],
    weights: Optional[list[float]],
) -> np.ndarray:
    if combine == "median":
        return np.median(paths, axis=0).astype(np.float32)
    if combine == "mean":
        return np.mean(paths, axis=0).astype(np.float32)

    if weights is None:
        weights_array = np.array(
            [1.0 / (abs(index - MEDIAN_QUANTILE_INDEX) + 1.0) for index in quantile_indices],
            dtype=np.float32,
        )
    else:
        if len(weights) != len(quantile_indices):
            raise ValueError(
                "--rolling-quantile-weights must have the same length as "
                "--rolling-quantile-indices."
            )
        weights_array = np.array(weights, dtype=np.float32)

    weights_array = weights_array / weights_array.sum()
    return np.average(paths, axis=0, weights=weights_array).astype(np.float32)


def forecast_contexts_rolling_quantile(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
    decode_block_size: int,
    patch_size: int,
    device: str,
    rolling_forecast_horizon: int,
    quantile_indices: list[int],
    combine: str,
    weights: Optional[list[float]],
) -> np.ndarray:
    paths = []
    for quantile_index in quantile_indices:
        print(f"Starting rolling quantile path: index={quantile_index}", flush=True)
        paths.append(
            forecast_contexts_rolling(
                model=model,
                contexts=contexts,
                horizon=horizon,
                batch_size=batch_size,
                decode_block_size=decode_block_size,
                patch_size=patch_size,
                device=device,
                rolling_forecast_horizon=rolling_forecast_horizon,
                quantile_index=quantile_index,
            )
        )

    return combine_forecast_paths(
        paths=np.stack(paths, axis=0),
        combine=combine,
        quantile_indices=quantile_indices,
        weights=weights,
    )


def forecast_contexts_ensemble(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
    decode_block_size: int,
    patch_size: int,
    device: str,
    direct_weight: float,
) -> np.ndarray:
    direct_predictions = forecast_contexts(
        model=model,
        contexts=contexts,
        horizon=horizon,
        batch_size=batch_size,
        decode_block_size=decode_block_size,
        patch_size=patch_size,
        device=device,
    )
    rolling_predictions = forecast_contexts_rolling(
        model=model,
        contexts=contexts,
        horizon=horizon,
        batch_size=batch_size,
        decode_block_size=decode_block_size,
        patch_size=patch_size,
        device=device,
    )
    return (
        direct_weight * direct_predictions
        + (1.0 - direct_weight) * rolling_predictions
    ).astype(np.float32)


def forecast_contexts_segmented(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
    decode_block_size: int,
    patch_size: int,
    device: str,
    segment_rolling_steps: int,
) -> np.ndarray:
    direct_predictions = forecast_contexts(
        model=model,
        contexts=contexts,
        horizon=horizon,
        batch_size=batch_size,
        decode_block_size=decode_block_size,
        patch_size=patch_size,
        device=device,
    )
    rolling_steps = min(max(0, segment_rolling_steps), horizon)
    if rolling_steps == 0:
        return direct_predictions

    rolling_predictions = forecast_contexts_rolling(
        model=model,
        contexts=contexts,
        horizon=rolling_steps,
        batch_size=batch_size,
        decode_block_size=decode_block_size,
        patch_size=patch_size,
        device=device,
    )
    predictions = direct_predictions.copy()
    predictions[:, :rolling_steps] = rolling_predictions
    return predictions.astype(np.float32)


def build_output_path(
    input_csv: Path,
    output_dir: Path,
    output_path: Optional[Path],
    suffix: str = OUTPUT_SUFFIX,
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
    suffix: str = OUTPUT_SUFFIX,
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
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=PREDICTION_LENGTH)
    parser.add_argument(
        "--prediction-mode",
        choices=[
            "direct",
            "rolling",
            "ensemble",
            "rolling_multi_horizon",
            "segmented",
            "rolling_quantile",
        ],
        default=DEFAULT_PREDICTION_MODE,
    )
    parser.add_argument("--rolling-forecast-horizon", type=int, default=DEFAULT_ROLLING_FORECAST_HORIZON)
    parser.add_argument("--segment-rolling-steps", type=int, default=DEFAULT_SEGMENT_ROLLING_STEPS)
    parser.add_argument("--ensemble-direct-weight", type=float, default=DEFAULT_ENSEMBLE_DIRECT_WEIGHT)
    parser.add_argument(
        "--rolling-combine",
        choices=["median", "mean", "weighted"],
        default=DEFAULT_ROLLING_COMBINE,
    )
    parser.add_argument(
        "--rolling-quantile-indices",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_ROLLING_QUANTILE_INDICES),
    )
    parser.add_argument("--rolling-quantile-weights", type=parse_float_list, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--decode-block-size", type=int, default=DEFAULT_DECODE_BLOCK_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_dependencies()
    device = resolve_device(args.device)
    if args.rolling_forecast_horizon <= 0:
        raise ValueError("--rolling-forecast-horizon must be positive.")
    if not 0.0 <= args.ensemble_direct_weight <= 1.0:
        raise ValueError("--ensemble-direct-weight must be between 0 and 1.")

    df, contexts = load_input_csv(args.input_csv, args.lookback, args.limit)
    (
        contexts_std,
        context_means,
        context_scales
    ) = standardize_contexts(contexts)

    print(
        "Standardized context mean range:",
        contexts_std.mean(axis=1).min(),
        contexts_std.mean(axis=1).max()
    )

    print(
        "Standardized context std range:",
        contexts_std.std(axis=1).min(),
        contexts_std.std(axis=1).max()
    )
    print(f"CSV data successfully loaded, total samples: {len(df)}")
    print(f"Using Toto model: {args.model_id}")
    print(
        "Inference parameters:",
        f"horizon={args.horizon},",
        f"prediction_mode={args.prediction_mode},",
        f"rolling_forecast_horizon={args.rolling_forecast_horizon},",
        f"ensemble_direct_weight={args.ensemble_direct_weight},",
        f"segment_rolling_steps={args.segment_rolling_steps},",
        f"rolling_combine={args.rolling_combine},",
        f"rolling_quantile_indices={args.rolling_quantile_indices},",
        f"batch_size={args.batch_size},",
        f"decode_block_size={args.decode_block_size},",
        f"patch_size={args.patch_size}",
    )
    print(f"Using device: {device}")

    model = load_toto_model(args.model_id, device, args.dtype)

    quantiles_std = forecast_contexts(
        model=model,
        contexts=contexts_std,
        horizon=args.horizon,
        batch_size=args.batch_size,
        decode_block_size=args.decode_block_size,
        patch_size=args.patch_size,
        device=device,
    )

    quantiles = (
        quantiles_std
        * context_scales[None, :, None, None]
        + context_means[None, :, None, None]
    )

    print("Native Toto quantiles:", quantiles.shape)

    median_predictions = quantiles[
        MEDIAN_QUANTILE_INDEX,
        :,
        0,
        :
    ]

    print(
        "Median prediction shape:",
        median_predictions.shape
    )

    print(
        "Standardized Toto quantile range:",
        np.min(quantiles_std),
        np.max(quantiles_std)
    )

    print(
        "Inverse-transformed Toto quantile range:",
        np.min(quantiles),
        np.max(quantiles)
    )

    print(
        "Final Toto quantile shape:",
        quantiles.shape
    )

    if not np.all(np.isfinite(quantiles)):
        raise ValueError(
            "Non-finite inverse-transformed Toto quantiles."
        )

    if args.prediction_mode == "rolling":
        predictions = forecast_contexts_rolling(
            model=model,
            contexts=contexts,
            horizon=args.horizon,
            batch_size=args.batch_size,
            decode_block_size=args.decode_block_size,
            patch_size=args.patch_size,
            device=device,
        )
    elif args.prediction_mode == "ensemble":
        predictions = forecast_contexts_ensemble(
            model=model,
            contexts=contexts,
            horizon=args.horizon,
            batch_size=args.batch_size,
            decode_block_size=args.decode_block_size,
            patch_size=args.patch_size,
            device=device,
            direct_weight=args.ensemble_direct_weight,
        )
    elif args.prediction_mode == "rolling_multi_horizon":
        predictions = forecast_contexts_rolling(
            model=model,
            contexts=contexts,
            horizon=args.horizon,
            batch_size=args.batch_size,
            decode_block_size=args.decode_block_size,
            patch_size=args.patch_size,
            device=device,
            rolling_forecast_horizon=args.rolling_forecast_horizon,
        )
    elif args.prediction_mode == "segmented":
        predictions = forecast_contexts_segmented(
            model=model,
            contexts=contexts,
            horizon=args.horizon,
            batch_size=args.batch_size,
            decode_block_size=args.decode_block_size,
            patch_size=args.patch_size,
            device=device,
            segment_rolling_steps=args.segment_rolling_steps,
        )
    elif args.prediction_mode == "rolling_quantile":
        predictions = forecast_contexts_rolling_quantile(
            model=model,
            contexts=contexts,
            horizon=args.horizon,
            batch_size=args.batch_size,
            decode_block_size=args.decode_block_size,
            patch_size=args.patch_size,
            device=device,
            rolling_forecast_horizon=args.rolling_forecast_horizon,
            quantile_indices=args.rolling_quantile_indices,
            combine=args.rolling_combine,
            weights=args.rolling_quantile_weights,
        )
    else:
        predictions = median_predictions

    output_csv = save_forecast_csv(
        df=df,
        predictions=median_predictions,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_path=args.output_path,
        horizon=args.horizon,
    )
    quantiles_path = (
        args.output_dir /
        f"{args.input_csv.stem}"
        f"{OUTPUT_SUFFIX}_quantiles.npz"
    )

    np.savez_compressed(
        quantiles_path,
        quantiles=quantiles,
        context_means=context_means,
        context_scales=context_scales
    )

    print(
        "Toto native quantiles saved to:",
        quantiles_path
    )
    print(f"Toto zero-shot prediction results have been saved to: {output_csv}")


if __name__ == "__main__":
    main()
