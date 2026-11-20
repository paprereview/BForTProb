import argparse
import gc
import time
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from common import (
    DEFAULT_MODEL_ID,
    FREQ,
    LOOKBACK,
    OUTPUT_DIR,
    PREDICTION_LENGTH,
    read_supervised_csv,
    save_forecast_csv,
)


OUTPUT_SUFFIX = "_chronos_zero-shot"
DEFAULT_BATCH_SIZE = 128
DEFAULT_DEVICE = "auto"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
MEDIAN_QUANTILE = "0.5"
MODEL_ID_ALIASES = {
    "chronos-2": "autogluon/chronos-2",
    "chronos-2-small": "autogluon/chronos-2-small",
    "chronos-t5-tiny": "amazon/chronos-t5-tiny",
}
CHRONOS2_FAMILY = "chronos2"
T5_FAMILY = "t5"

def standardize_contexts(
    contexts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a separate StandardScaler to each context window.

    contexts:
        shape (N, lookback)

    returns:
        contexts_std:   shape (N, lookback)
        context_means:  shape (N,)
        context_scales: shape (N,)
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

def require_runtime_dependencies() -> None:
    missing = []
    for module_name in ("torch", "chronos", "pandas"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "Missing Chronos runtime dependencies: "
            + ", ".join(missing)
            + ". Install `chronos-forecasting`, `torch`, `pandas`, and `numpy`."
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


def resolve_model_id(model_id: str) -> str:
    return MODEL_ID_ALIASES.get(model_id, model_id)


def infer_model_family(model_id: str) -> str:
    model_name = model_id.split("/")[-1]
    if "chronos-2" in model_name:
        return CHRONOS2_FAMILY
    if "chronos-t5" in model_name:
        return T5_FAMILY
    raise ValueError(
        f"Cannot infer Chronos model family from model id: {model_id}. "
        "Supported aliases are: " + ", ".join(sorted(MODEL_ID_ALIASES))
    )


def output_suffix_for_model(model_id: str) -> str:
    model_name = model_id.split("/")[-1]
    return f"_{model_name}_standardized_zero-shot"


def load_chronos2_model(model_id: str, device: str):
    from chronos import Chronos2Pipeline

    start_time = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device,
    )
    return pipeline, time.perf_counter() - start_time


def load_t5_model(model_id: str, device: str):
    import torch
    from chronos import ChronosPipeline

    start_time = time.perf_counter()
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float32,
    )
    return pipeline, time.perf_counter() - start_time


def contexts_to_chronos_df(
    contexts: np.ndarray,
    freq: str = FREQ,
) -> pd.DataFrame:
    n_rows, lookback = contexts.shape
    timestamps = pd.date_range("2000-01-01", periods=lookback, freq=freq)
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_rows, dtype=np.int64), lookback),
            "timestamp": np.tile(timestamps, n_rows),
            "target": contexts.reshape(-1).astype(np.float32),
        }
    )


def predictions_from_chronos_df(
    pred_df: pd.DataFrame,
    n_rows: int,
    horizon: int,
) -> np.ndarray:
    median_col = MEDIAN_QUANTILE
    if median_col not in pred_df.columns:
        median_col = 0.5
    if median_col not in pred_df.columns:
        raise ValueError(
            f"Chronos prediction output is missing median quantile column: {MEDIAN_QUANTILE}"
        )

    sorted_df = pred_df.sort_values(["id", "timestamp"])
    predictions = sorted_df[median_col].to_numpy(dtype=np.float32).reshape(n_rows, horizon)
    return predictions


def forecast_chronos2_contexts(
    pipeline,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    forecast_start = time.perf_counter()
    for start in range(0, len(contexts), batch_size):
        end = min(start + batch_size, len(contexts))
        context_df = contexts_to_chronos_df(contexts[start:end])
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=horizon,
            quantile_levels=QUANTILE_LEVELS,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        predictions.append(
            predictions_from_chronos_df(
                pred_df=pred_df,
                n_rows=end - start,
                horizon=horizon,
            )
        )
        print(f"forecasted rows {start}:{end}", flush=True)

    y_pred = np.vstack(predictions)
    print(f"forecast seconds: {time.perf_counter() - forecast_start:.2f}", flush=True)
    return y_pred


def forecast_t5_contexts(
    pipeline,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
    num_samples: int,
) -> np.ndarray:
    import torch

    all_samples = []
    forecast_start = time.perf_counter()

    for start in range(0, len(contexts), batch_size):

        end = min(
            start + batch_size,
            len(contexts)
        )

        context = torch.tensor(
            contexts[start:end],
            dtype=torch.float32
        )

        forecast = pipeline.predict(
            context,
            prediction_length=horizon,
            num_samples=num_samples
        )

        samples = (
            forecast
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        print(
            "Chronos-T5 raw sample shape:",
            samples.shape
        )

        all_samples.append(samples)

        print(
            f"forecasted rows {start}:{end}",
            flush=True
        )

    samples = np.concatenate(
        all_samples,
        axis=0
    )

    print(
        "Final Chronos-T5 samples shape:",
        samples.shape
    )

    print(
        f"forecast seconds: "
        f"{time.perf_counter() - forecast_start:.2f}",
        flush=True
    )

    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=PREDICTION_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=DEFAULT_DEVICE,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100
    )
    return parser.parse_args()


def run_zero_shot(
    input_csv: Path,
    model_id: str = DEFAULT_MODEL_ID,
    lookback: int = LOOKBACK,
    horizon: int = PREDICTION_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_samples: int = 100,
    limit: Optional[int] = None,
    device: str = DEFAULT_DEVICE,
    output_dir: Path = OUTPUT_DIR,
    output_path: Optional[Path] = None,
) -> Path:

    df, contexts, _ = read_supervised_csv(
        input_csv,
        lookback=lookback,
        horizon=horizon,
        require_y=False,
        limit=limit,
    )

    contexts_std, context_means, context_scales = (
	standardize_contexts(contexts)
    )

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

    print(f"CSV loaded: {input_csv} ({len(df)} rows)")

    resolved_model_id = resolve_model_id(model_id)
    model_family = infer_model_family(resolved_model_id)

    print(f"Chronos zero-shot model: {resolved_model_id}")
    print(f"Chronos model family: {model_family}")

    resolved_device = resolve_device(device)
    print(f"Chronos device: {resolved_device}")

    if model_family == CHRONOS2_FAMILY:

        raise ValueError(
            "Chronos-2 does not provide native forecast samples. "
            "Use a Chronos-T5 model."
        )

    elif model_family == T5_FAMILY:

        pipeline, load_seconds = load_t5_model(
            resolved_model_id,
            resolved_device
        )

        print(
            f"model load seconds: {load_seconds:.2f}",
            flush=True
        )

        samples_std = forecast_t5_contexts(
            pipeline=pipeline,
            contexts=contexts_std,
            horizon=horizon,
            batch_size=batch_size,
            num_samples=num_samples,
        )
	
        samples = (
            samples_std
            * context_scales[:, None, None]
            + context_means[:, None, None]
        )
	
        print(
            "Standardized sample range:",
            np.nanmin(samples_std),
            np.nanmax(samples_std)
        )

        print(
            "Inverse-transformed sample range:",
            np.nanmin(samples),
            np.nanmax(samples)
        )
	
    else:
        raise ValueError(
            f"Unsupported Chronos model family: {model_family}"
        )

    # ---------------------------------
    # Everything below stays INSIDE
    # run_zero_shot()
    # ---------------------------------

    y_cols = [f"Y_{i}" for i in range(horizon)]

    expected_sample_shape = (
        len(df),
        num_samples,
        horizon
    )

    if samples.shape != expected_sample_shape:
        raise ValueError(
            f"Unexpected Chronos-T5 sample shape: "
            f"{samples.shape}, expected "
            f"{expected_sample_shape}"
        )

    median_predictions = np.median(
        samples,
        axis=1
    )

    result_df = df.copy()
    result_df.loc[:, y_cols] = median_predictions

    suffix = output_suffix_for_model(
        resolved_model_id
    )

    output_csv = (
        output_dir /
        f"{input_csv.stem}{suffix}.csv"
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        output_csv,
        index=False
    )

    samples_path = (
        output_dir /
        f"{input_csv.stem}{suffix}_samples.npz"
    )

    np.savez_compressed(
	samples_path,
	samples=samples,
	context_means=context_means,
	context_scales=context_scales
    )

    print(
        "Chronos-T5 point predictions saved to:",
        output_csv
    )

    print(
        "Chronos-T5 native samples saved to:",
        samples_path
    )

    del pipeline
    gc.collect()

    return output_csv

def main() -> None:
    args = parse_args()
    require_runtime_dependencies()
    output_csv = run_zero_shot(
        input_csv=args.input_csv,
        model_id=args.model_id,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        output_dir=args.output_dir,
        output_path=args.output_path,
        num_samples=args.num_samples
    )
    print(f"Chronos zero-shot predictions saved to: {output_csv}")


if __name__ == "__main__":
    main()
