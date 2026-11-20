import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Union
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
CKPT_PATH = MODEL_DIR / "lag-llama.ckpt"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_SUFFIX = "_lag-llama_standardized_zero-shot"

LOOKBACK = 250
DEFAULT_CONTEXT_LENGTH = 250
PREDICTION_LENGTH = 20
DATA_FREQ = "1D"
NUM_SAMPLES = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_PROGRESS_EVERY = 64

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated.*",
    category=UserWarning,
)

LAG_LLAMA_SOURCE = MODEL_DIR.parents[1] / "model_experiments_git" / "lag-llama"
if str(LAG_LLAMA_SOURCE) not in sys.path:
    sys.path.insert(0, str(LAG_LLAMA_SOURCE))

import gluonts
import gluonts.torch
import gluonts.torch.modules

# Some Lag-Llama checkpoints reference older GluonTS module paths/classes.
import gluonts.torch.distributions as g_dist
import gluonts.torch.util as g_util

sys.modules["gluonts.torch.distributions"] = g_dist
sys.modules["gluonts.torch.modules.loss"] = g_util

if not hasattr(g_util, "DistributionLoss"):

    class DistributionLoss:
        def __init__(self, *args, **kwargs):
            pass

    g_util.DistributionLoss = DistributionLoss

if not hasattr(g_util, "NegativeLogLikelihood"):

    class NegativeLogLikelihood:
        def __init__(self, *args, **kwargs):
            pass

    g_util.NegativeLogLikelihood = NegativeLogLikelihood

from lag_llama.gluon.estimator import LagLlamaEstimator


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def csv_to_lag_llama_dataset(
    csv_path: Path,
    lookback: int = LOOKBACK,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    freq: str = DATA_FREQ,
    limit: Optional[int] = None,
) -> tuple[
    ListDataset,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)

    if context_length <= 0:
        raise ValueError(
            "--context-length must be positive."
        )

    if context_length > lookback:
        raise ValueError(
            "--context-length cannot be larger than --lookback."
        )

    x_cols = [
        f"X_{i}"
        for i in range(lookback)
    ]

    missing_cols = [
        col
        for col in x_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"CSV missing input columns: "
            f"{missing_cols[:5]}"
        )

    # Use the latest context_length values.
    # With context_length=250, this is X_0...X_249.
    context_cols = x_cols[-context_length:]

    x_data = df[
        context_cols
    ].to_numpy(dtype=np.float32)

    # ---------------------------------------------
    # Per-window StandardScaler
    # Fit ONLY on each observed context.
    # ---------------------------------------------

    x_std = np.empty_like(
        x_data,
        dtype=np.float32
    )

    context_means = np.empty(
        len(x_data),
        dtype=np.float32
    )

    context_scales = np.empty(
        len(x_data),
        dtype=np.float32
    )

    for i, x in enumerate(x_data):

        scaler = StandardScaler()

        standardized = scaler.fit_transform(
            x.reshape(-1, 1)
        ).ravel()

        x_std[i] = standardized
        context_means[i] = scaler.mean_[0]
        context_scales[i] = scaler.scale_[0]

    # ---------------------------------------------
    # Sanity checks
    # ---------------------------------------------

    if not np.all(np.isfinite(x_std)):
        raise ValueError(
            "Non-finite standardized contexts."
        )

    print(
        "Standardized context mean range:",
        x_std.mean(axis=1).min(),
        x_std.mean(axis=1).max()
    )

    print(
        "Standardized context std range:",
        x_std.std(axis=1).min(),
        x_std.std(axis=1).max()
    )

    start_time = pd.Period(
        "2026-01-01",
        freq=freq
    )

    data = [
        {
            FieldName.START: start_time,
            FieldName.TARGET: row,
            FieldName.ITEM_ID: f"series_{idx}",
        }
        for idx, row in enumerate(x_std)
    ]

    return (
        ListDataset(data, freq=freq),
        df,
        context_means,
        context_scales,
    )


def get_lag_llama_predictions(
    dataset,
    ckpt_path: Path = CKPT_PATH,
    adapter_dir: Optional[Path] = None,
    prediction_length: int = PREDICTION_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    device: Union[torch.device, str] = torch.device("cpu"),
    use_rope_scaling: bool = True,
    num_samples: int = NUM_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    total_series: Optional[int] = None,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    estimator_args = checkpoint["hyper_parameters"]["model_kwargs"]
    trained_context_length = estimator_args["context_length"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / trained_context_length),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=str(ckpt_path),
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
        batch_size=batch_size,
        num_parallel_samples=num_samples,
        device=torch.device(device),
    )

    lightning_module = estimator.create_lightning_module()
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "PEFT is required to load --adapter-dir. Install it in the "
                "lag-llama environment, e.g. `uv add peft safetensors`."
            ) from exc

        lightning_module.model = PeftModel.from_pretrained(
            lightning_module.model,
            str(adapter_dir),
            is_trainable=False,
        )

    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples,
    )

    forecasts = []
    for idx, forecast in enumerate(forecast_it, start=1):
        forecasts.append(forecast)
        if progress_every and (
            idx == 1 or idx % progress_every == 0 or idx == total_series
        ):
            total_msg = f"/{total_series}" if total_series else ""
            print(f"Completed prediction: {idx}{total_msg}", flush=True)

    return forecasts, list(ts_it)


def build_output_path(
    input_path: Path,
    output_path: Optional[Path] = None,
    output_dir: Path = OUTPUT_DIR,
    output_suffix: str = OUTPUT_SUFFIX,
) -> Path:
    if output_path is not None:
        return output_path
    return output_dir / f"{input_path.stem}{output_suffix}{input_path.suffix}"


def save_predictions_to_csv(
    df: pd.DataFrame,
    forecasts,
    context_means: np.ndarray,
    context_scales: np.ndarray,
    input_path: Path,
    output_path: Optional[Path] = None,
    output_dir: Path = OUTPUT_DIR,
    output_suffix: str = OUTPUT_SUFFIX,
    prediction_length: int = PREDICTION_LENGTH,
) -> Path:
    y_cols = [f"Y_{i}" for i in range(prediction_length)]
    missing_cols = [col for col in y_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing output columns: {missing_cols[:5]}")

    samples_std = np.stack(
        [
            forecast.samples
            for forecast in forecasts
        ]
    ).astype(np.float32)

    print(
        "Standardized Lag-Llama samples shape:",
        samples_std.shape
    )

    # Expected:
    # (N, num_samples, horizon)

    if samples_std.ndim != 3:
        raise ValueError(
            f"Unexpected Lag-Llama sample shape: "
            f"{samples_std.shape}"
        )

    if samples_std.shape[0] != len(context_means):
        raise ValueError(
            "Number of forecast windows does not "
            "match number of scalers."
        )

    # ---------------------------------------------
    # Back to original log-return units
    # ---------------------------------------------

    samples = (
        samples_std
        * context_scales[:, None, None]
        + context_means[:, None, None]
    ).astype(np.float32)

    print(
        "Inverse-transformed Lag-Llama "
        "samples shape:",
        samples.shape
    )

    print(
        "Standardized sample range:",
        np.min(samples_std),
        np.max(samples_std)
    )

    print(
        "Inverse-transformed sample range:",
        np.min(samples),
        np.max(samples)
    )

    if not np.all(np.isfinite(samples)):
        raise ValueError(
            "Non-finite inverse-transformed "
            "Lag-Llama samples."
        )

    predictions = np.median(
        samples,
        axis=1
    ).astype(np.float32)

    result_df = df.copy()
    result_df.loc[:, y_cols] = predictions

    output_csv_path = build_output_path(
        input_path,
        output_path,
        output_dir,
        output_suffix,
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)

    samples_path = output_dir / (
        f"{input_path.stem}"
        f"{output_suffix}_samples.npz"
    )

    np.savez_compressed(
        samples_path,
        samples=samples,
        context_means=context_means,
        context_scales=context_scales
    )

    print(
        f"Native Lag-Llama forecast samples saved to: "
        f"{samples_path}"
    )

    return output_csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--freq",
        type=str,
        default=DATA_FREQ,
        help="Time-series frequency, e.g. 1D for daily or W for weekly."
    )
    parser.add_argument("--checkpoint", type=Path, default=CKPT_PATH)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=(
            "Number of latest X columns passed to Lag-Llama. "
            "--lookback still controls how many X columns are expected in the CSV."
        ),
    )
    parser.add_argument("--horizon", type=int, default=PREDICTION_LENGTH)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    (
        predict_ds,
        df,
        context_means,
        context_scales,
    ) = csv_to_lag_llama_dataset(
        args.input_csv,
        lookback=args.lookback,
        context_length=args.context_length,
        freq=args.freq,
        limit=args.limit,
    )
    print(f"CSV data successfully loaded, total samples: {len(df)}")
    print(f"Using model: {args.checkpoint}")
    device = get_default_device()
    print(f"Using device: {device}")
    print(
        "Inference parameters: "
        f"lookback={args.lookback}, "
        f"context_length={args.context_length}, "
        f"horizon={args.horizon}, "
        f"num_samples={args.num_samples}, "
        f"batch_size={args.batch_size}"
    )

    print("\n--- Starting Lag-Llama zero-shot inference ---")
    start_perf = time.time()
    forecasts, _ = get_lag_llama_predictions(
        dataset=predict_ds,
        ckpt_path=args.checkpoint,
        adapter_dir=args.adapter_dir,
        prediction_length=args.horizon,
        context_length=args.context_length,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        total_series=len(df),
        progress_every=args.progress_every,
    )

    print("Forecast object type:", type(forecasts[0]))
    print("Forecast samples shape:", forecasts[0].samples.shape)

    print(f"\nZero-shot inference completed, total time: {time.time() - start_perf:.2f} seconds")
    print(f"Successfully obtained prediction objects: {len(forecasts)}")

    output_csv_path = save_predictions_to_csv(
        df=df,
        forecasts=forecasts,
        context_means=context_means,
        context_scales=context_scales,
        input_path=args.input_csv,
        output_path=args.output_path,
        output_dir=args.output_dir,
        prediction_length=args.horizon,
    )
    print(f"Zero-shot prediction results have been saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
