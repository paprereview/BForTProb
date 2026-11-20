import argparse
import gc
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOOKBACK = 250
PREDICTION_LENGTH = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_FAMILY = "moirai2"
DEFAULT_MOE_PATCH_SIZE = 16
DEFAULT_MOE_NUM_SAMPLES = 100
MOIRAI2_FAMILY = "moirai2"
MOIRAI_MOE_FAMILY = "moirai-moe"
DEFAULT_MODEL_IDS = {
    MOIRAI2_FAMILY: "Salesforce/moirai-2.0-R-small",
    MOIRAI_MOE_FAMILY: "Salesforce/moirai-moe-1.0-R-base",
}



def require_runtime_dependencies(model_family: str) -> None:
    missing = []
    for module_name in ("torch", "uni2ts", "numpy", "pandas"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if model_family == MOIRAI_MOE_FAMILY:
        try:
            __import__("gluonts")
        except ImportError:
            missing.append("gluonts")
    if missing:
        raise ImportError(
            "Missing Moirai runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in moirai/.venv, e.g. "
            + "`uv add torch uni2ts numpy pandas`."
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

def load_moirai_model(
    model_id: str,
    model_family: str,
    horizon: int,
    lookback: int,
    device: str,
    moe_patch_size: int,
    moe_num_samples: int,
):
    if model_family == MOIRAI2_FAMILY:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(model_id),
            prediction_length=horizon,
            context_length=lookback,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        return model.to(device).eval()

    if model_family == MOIRAI_MOE_FAMILY:
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

        return MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(model_id),
            prediction_length=horizon,
            context_length=lookback,
            patch_size=moe_patch_size,
            num_samples=moe_num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).eval()

    raise ValueError(f"Unsupported Moirai model family: {model_family}")


def infer_model_family(model_id: str) -> str:
    normalized_id = model_id.lower()
    if "moirai-moe" in normalized_id:
        return MOIRAI_MOE_FAMILY
    if "moirai-2" in normalized_id:
        return MOIRAI2_FAMILY
    raise ValueError(
        f"Cannot infer Moirai model family from model id: {model_id}. "
        "Pass --model-family moirai2 or --model-family moirai-moe."
    )


def resolve_model_selection(
    model_id: Optional[str],
    requested_family: str,
) -> tuple[str, str]:
    if model_id is None:
        if requested_family == "auto":
            requested_family = DEFAULT_MODEL_FAMILY
        return DEFAULT_MODEL_IDS[requested_family], requested_family

    inferred_family = infer_model_family(model_id)
    if requested_family not in ("auto", inferred_family):
        raise ValueError(
            f"Model id '{model_id}' belongs to the '{inferred_family}' family, "
            f"but --model-family is '{requested_family}'."
        )
    return model_id, inferred_family


def output_suffix_for_model(
    model_id: str
) -> str:

    model_name = (
        model_id
        .rsplit("/", maxsplit=1)[-1]
        .lower()
    )

    return (
        f"_{model_name}_"
        f"standardized_zero-shot"
    )


def median_quantile_index(model) -> int:
    levels = np.asarray(model.module.quantile_levels, dtype=np.float32)
    matches = np.flatnonzero(np.isclose(levels, 0.5))
    if len(matches) != 1:
        raise ValueError(f"Moirai quantile levels do not contain a unique median: {levels}")
    return int(matches[0])


def forecast_moirai2_contexts(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
) -> np.ndarray:
    median_index = median_quantile_index(model)
    predictions = []
    for start in range(0, len(contexts), batch_size):
        end = min(start + batch_size, len(contexts))
        # Moirai's native predict method accepts a list of univariate arrays and
        # performs the model's standard missing-value imputation internally.
        quantiles = np.asarray(model.predict(list(contexts[start:end])), dtype=np.float32)
        if (
            quantiles.ndim == 3
            and quantiles.shape == (end - start, len(model.module.quantile_levels), horizon)
        ):
            batch_predictions = quantiles[:, median_index, :]
        elif (
            quantiles.ndim == 4
            and quantiles.shape
            == (end - start, len(model.module.quantile_levels), horizon, 1)
        ):
            batch_predictions = quantiles[:, median_index, :, 0]
        else:
            raise ValueError(
                "Unexpected Moirai prediction shape: "
                f"{quantiles.shape}; expected (batch, quantiles, {horizon}) "
                f"or (batch, quantiles, {horizon}, 1)."
            )
        predictions.append(batch_predictions)
        print(f"Completed prediction: {end}/{len(contexts)}", flush=True)

    return np.concatenate(predictions, axis=0).astype(np.float32)


def forecast_moirai_moe_contexts(
    model,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
    device: str,
    freq: str = "D",
) -> np.ndarray:

    from gluonts.dataset.common import ListDataset

    dataset = ListDataset(
        [
            {
                "start": pd.Period(
                    "2000-01-01",
                    freq=freq
                ),
                "target": context,
            }
            for context in contexts
        ],
        freq=freq,
    )

    predictor = model.create_predictor(
        batch_size=batch_size,
        device=device
    )

    all_samples = []

    for index, forecast in enumerate(
        predictor.predict(dataset),
        start=1
    ):

        samples = np.asarray(
            forecast.samples,
            dtype=np.float32
        )

        # Sometimes target_dim=1 gives
        # (num_samples, horizon, 1)
        if (
            samples.ndim == 3
            and samples.shape[-1] == 1
        ):
            samples = samples[..., 0]

        if (
            samples.ndim != 2
            or samples.shape[1] != horizon
        ):
            raise ValueError(
                "Unexpected Moirai-MoE sample "
                f"forecast shape: {samples.shape}; "
                f"expected (num_samples, {horizon})."
            )

        all_samples.append(samples)

        if (
            index % batch_size == 0
            or index == len(contexts)
        ):
            print(
                f"Completed prediction: "
                f"{index}/{len(contexts)}",
                flush=True
            )

    samples = np.stack(
        all_samples,
        axis=0
    ).astype(np.float32)

    print(
        "Final standardized Moirai-MoE "
        "sample shape:",
        samples.shape
    )

    # Expected:
    # (N, num_samples, horizon)

    return samples


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
    output_csv = build_output_path(input_csv, output_dir, output_path, suffix)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    return output_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model override. The default is selected by --model-family.",
    )
    parser.add_argument(
        "--model-family",
        choices=["auto", MOIRAI2_FAMILY, MOIRAI_MOE_FAMILY],
        default=DEFAULT_MODEL_FAMILY,
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=PREDICTION_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--freq",
        type=str,
        default="D",
        help="Time-series frequency, e.g. D for daily or W for weekly."
    )
    parser.add_argument("--moe-patch-size", type=int, default=DEFAULT_MOE_PATCH_SIZE)
    parser.add_argument("--moe-num-samples", type=int, default=DEFAULT_MOE_NUM_SAMPLES)
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
    model_id, model_family = resolve_model_selection(args.model_id, args.model_family)
    require_runtime_dependencies(model_family)
    device = resolve_device(args.device)
    df, contexts = load_input_csv(
        args.input_csv,
        args.lookback,
        args.limit
    )

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
    print(f"Using Moirai model: {model_id}")
    print(f"Moirai model family: {model_family}")
    print(
        f"Inference parameters: horizon={args.horizon}, batch_size={args.batch_size}, device={device}"
    )

    model = load_moirai_model(
        model_id=model_id,
        model_family=model_family,
        horizon=args.horizon,
        lookback=args.lookback,
        device=device,
        moe_patch_size=args.moe_patch_size,
        moe_num_samples=args.moe_num_samples,
    )
    if model_family == MOIRAI2_FAMILY:
        predictions = forecast_moirai2_contexts(
            model=model,
            contexts=contexts,
            horizon=args.horizon,
            batch_size=args.batch_size,
        )

    else:

        # --------------------------------------------
        # FORECAST IN STANDARDIZED SPACE
        # --------------------------------------------

        samples_std = forecast_moirai_moe_contexts(
            model=model,
            contexts=contexts_std,
            horizon=args.horizon,
            batch_size=args.batch_size,
            device=device,
            freq=args.freq,
        )

        # --------------------------------------------
        # INVERSE TRANSFORM SAMPLES
        # --------------------------------------------

        samples = (
            samples_std
            * context_scales[:, None, None]
            + context_means[:, None, None]
        ).astype(np.float32)

        print(
            "Standardized Moirai-MoE sample range:",
            np.min(samples_std),
            np.max(samples_std)
        )

        print(
            "Inverse-transformed Moirai-MoE sample range:",
            np.min(samples),
            np.max(samples)
        )

        if not np.all(np.isfinite(samples)):
            raise ValueError(
                "Non-finite inverse-transformed "
                "Moirai-MoE samples."
            )

        # --------------------------------------------
        # POINT FORECAST FOR CSV
        # --------------------------------------------

        predictions = np.median(
            samples,
            axis=1
        ).astype(np.float32)

    output_csv = save_forecast_csv(
        df=df,
        predictions=predictions,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_path=args.output_path,
        horizon=args.horizon,
        suffix=output_suffix_for_model(model_id),
    )

    if model_family == MOIRAI_MOE_FAMILY:

        samples_path = (
            args.output_dir
            / (
                f"{args.input_csv.stem}"
                f"{output_suffix_for_model(model_id)}"
                f"_samples.npz"
            )
        )

        samples_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez_compressed(
            samples_path,
            samples=samples,
            context_means=context_means,
            context_scales=context_scales
        )

        print(
            "Moirai-MoE native samples saved to:",
            samples_path
        )

    del model
    gc.collect()
    print(f"Moirai zero-shot prediction results have been saved to: {output_csv}")


if __name__ == "__main__":
    main()
