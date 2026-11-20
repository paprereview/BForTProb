import argparse
import gc
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import torch

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_SUFFIX = "_moirai-2.0-r-small_standardized_zero-shot"

LOOKBACK = 250
PREDICTION_LENGTH = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_ID = "Salesforce/moirai-2.0-R-small"


def require_runtime_dependencies() -> None:
    missing = []
    for module_name in ("torch", "uni2ts", "numpy", "pandas"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "Missing Moirai runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in moirai/.venv, e.g. "
            + "`uv pip install --python moirai/.venv/bin/python uni2ts pandas`."
        )


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: str) -> str:
    if device == "auto":
        return get_default_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is not available.")
    if device == "mps":
        raise ValueError(
            "Moirai 2.0 does not currently support MPS: its prediction path uses "
            "a PyTorch operation that MPS has not implemented. Use --device cpu."
        )
    return device


def load_input_csv(
    input_csv: Path,
    lookback: int,
    limit: Optional[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)

    x_cols = [f"X_{index}" for index in range(lookback)]
    missing_cols = [column for column in x_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing input columns: {missing_cols[:5]}")

    contexts = df[x_cols].to_numpy(dtype=np.float32)
    if np.isinf(contexts).any():
        raise ValueError(f"{input_csv.name} contains infinite input values.")
    if np.all(np.isnan(contexts), axis=1).any():
        raise ValueError(f"{input_csv.name} contains a context row with no observed values.")
    return df, contexts

def standardize_contexts(
    contexts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a separate StandardScaler to every context window.

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
        context_scales,
    )

def load_moirai_model(
    model_id: str,
    horizon: int,
    lookback: int,
    device: str,
):
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(model_id),
        prediction_length=horizon,
        context_length=lookback,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    return model.to(device).eval()


def median_quantile_index(model) -> int:
    levels = np.asarray(model.module.quantile_levels, dtype=np.float32)
    matches = np.flatnonzero(np.isclose(levels, 0.5))
    if len(matches) != 1:
        raise ValueError(f"Moirai quantile levels do not contain a unique median: {levels}")
    return int(matches[0])


def forecast_contexts(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:

    quantile_levels = np.asarray(
        model.module.quantile_levels,
        dtype=np.float32
    )

    all_quantiles = []

    for start in range(
        0,
        len(contexts),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(contexts)
        )

        raw = np.asarray(
            model.predict(
                list(contexts[start:end])
            ),
            dtype=np.float32
        )

        print(
            "Raw Moirai quantile shape:",
            raw.shape
        )

        # Expected:
        # (batch, Q, horizon)
        if (
            raw.ndim == 3
            and raw.shape == (
                end - start,
                len(quantile_levels),
                horizon
            )
        ):
            batch_quantiles = raw

        # Alternative:
        # (batch, Q, horizon, 1)
        elif (
            raw.ndim == 4
            and raw.shape == (
                end - start,
                len(quantile_levels),
                horizon,
                1
            )
        ):
            batch_quantiles = raw[..., 0]

        else:
            raise ValueError(
                "Unexpected Moirai prediction shape: "
                f"{raw.shape}"
            )

        all_quantiles.append(
            batch_quantiles
        )

        print(
            f"Completed prediction: "
            f"{end}/{len(contexts)}",
            flush=True
        )

    quantiles = np.concatenate(
        all_quantiles,
        axis=0
    ).astype(np.float32)

    print(
        "Final Moirai quantile shape:",
        quantiles.shape
    )

    print(
        "Moirai quantile levels:",
        quantile_levels
    )

    return (
        quantiles,
        quantile_levels
    )


def build_output_path(input_csv: Path, output_dir: Path, output_path: Optional[Path]) -> Path:
    if output_path is not None:
        return output_path
    return output_dir / f"{input_csv.stem}{OUTPUT_SUFFIX}{input_csv.suffix}"


def save_forecast_csv(
    df: pd.DataFrame,
    predictions: np.ndarray,
    input_csv: Path,
    output_dir: Path,
    output_path: Optional[Path],
    horizon: int,
) -> Path:
    y_cols = [f"Y_{index}" for index in range(horizon)]
    missing_cols = [column for column in y_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing output columns: {missing_cols[:5]}")
    if predictions.shape != (len(df), horizon):
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match {(len(df), horizon)}"
        )

    result_df = df.copy()
    result_df.loc[:, y_cols] = predictions
    output_csv = build_output_path(input_csv, output_dir, output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    return output_csv


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
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_dependencies()
    device = resolve_device(args.device)
    df, contexts = load_input_csv(args.input_csv, args.lookback, args.limit)
    (
        contexts_std,
        context_means,
        context_scales,
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
    print(f"Using Moirai model: {args.model_id}")
    print(
        f"Inference parameters: horizon={args.horizon}, batch_size={args.batch_size}, device={device}"
    )

    model = load_moirai_model(
        model_id=args.model_id,
        horizon=args.horizon,
        lookback=args.lookback,
        device=device,
    )
    (
        quantiles_std,
        quantile_levels
    ) = forecast_contexts(
        model=model,
        contexts=contexts_std,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )

    # inverse transform quantiles
    quantiles = (
        quantiles_std
        * context_scales[:, None, None]
        + context_means[:, None, None]
    ).astype(np.float32)

    print(
        "Standardized Moirai quantile range:",
        np.min(quantiles_std),
        np.max(quantiles_std)
    )

    print(
        "Inverse-transformed Moirai quantile range:",
        np.min(quantiles),
        np.max(quantiles)
    )

    print(
        "Final quantile shape:",
        quantiles.shape
    )

    if not np.all(np.isfinite(quantiles)):
        raise ValueError(
            "Non-finite inverse-transformed "
            "Moirai quantiles."
        )

    median_matches = np.flatnonzero(
        np.isclose(
            quantile_levels,
            0.5
        )
    )

    if len(median_matches) != 1:
        raise ValueError(
            "Moirai does not contain a unique "
            "0.5 quantile."
        )

    median_index = int(
        median_matches[0]
    )

    predictions = quantiles[
        :,
        median_index,
        :
    ]

    output_csv = save_forecast_csv(
        df=df,
        predictions=predictions,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_path=args.output_path,
        horizon=args.horizon,
    )
    del model
    gc.collect()
    print(f"Moirai zero-shot prediction results have been saved to: {output_csv}")

    quantiles_path = (
        args.output_dir /
        f"{args.input_csv.stem}"
        f"{OUTPUT_SUFFIX}_quantiles.npz"
    )

    quantiles_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        quantiles_path,
        quantiles=quantiles,
        quantile_levels=quantile_levels,
        context_means=context_means,
        context_scales=context_scales,
    )

    print(
        "Moirai native quantiles saved to:",
        quantiles_path
    )

if __name__ == "__main__":
    main()
