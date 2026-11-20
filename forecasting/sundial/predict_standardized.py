import argparse
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_SUFFIX = "_sundial_standardized_zero-shot"

LOOKBACK = 250
PREDICTION_LENGTH = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_ID = "thuml/sundial-base-128m"
DEFAULT_NUM_SAMPLES = 100


def require_runtime_dependencies() -> None:
    missing = []
    for module_name in ("torch", "transformers", "pandas"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise ImportError(
            "Missing Sundial runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in sundial/.venv, e.g. "
            + "`uv pip install torch transformers pandas accelerate`."
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
        dtype = "float16" if device == "cuda" else "float32"
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
    Fit a separate StandardScaler to every 250-point context window.

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

def load_sundial_model(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype, dtype_name = resolve_torch_dtype(dtype, device)
    print(f"Sundial dtype: {dtype_name}")
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            **model_kwargs,
        )
        print("Using Hugging Face cache to load Sundial model")
    except OSError:
        print("Local cache not found, starting download from Hugging Face")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

    return model.to(device)


def extract_generated_samples(
    generated,
    batch_size: int,
    horizon: int,
    num_samples: int,
) -> np.ndarray:

    values = generated.detach().float().cpu().numpy()

    expected_shape = (batch_size, num_samples, horizon)

    if values.shape == expected_shape:
        return values.astype(np.float32)

    # In case Sundial returns extra time steps
    if (
        values.ndim == 3
        and values.shape[0] == batch_size
        and values.shape[1] == num_samples
        and values.shape[2] >= horizon
    ):
        return values[:, :, -horizon:].astype(np.float32)

    raise ValueError(
        f"Unexpected Sundial output shape: {values.shape}. "
        f"Expected approximately {expected_shape}."
    )


def forecast_contexts(
    model,
    contexts: np.ndarray,
    horizon: int,
    num_samples: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            seqs = torch.tensor(batch, dtype=torch.float32, device=device)
            generated = model.generate(
                seqs,
                max_new_tokens=horizon,
                num_samples=num_samples,
            )

            print("Raw generated shape:", generated.shape)

            predictions.append(
                extract_generated_samples(
                    generated,
                    batch_size=len(batch),
                    horizon=horizon,
                    num_samples=num_samples,
                )
            )
            print(f"已完成预测: {min(start + batch_size, len(contexts))}/{len(contexts)}", flush=True)

    return np.concatenate(predictions, axis=0).astype(np.float32)


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
    samples: np.ndarray,
    context_means: np.ndarray,
    context_scales: np.ndarray,
    input_csv: Path,
    output_dir: Path,
    output_path: Optional[Path],
    horizon: int,
    suffix: str = OUTPUT_SUFFIX,
) -> tuple[Path, Path]:

    y_cols = [f"Y_{i}" for i in range(horizon)]

    missing_cols = [col for col in y_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"CSV missing output columns: {missing_cols[:5]}"
        )

    expected_rows = len(df)

    if (
        samples.ndim != 3
        or samples.shape[0] != expected_rows
        or samples.shape[2] != horizon
    ):
        raise ValueError(
            f"Unexpected sample shape {samples.shape}. "
            f"Expected (rows, samples, {horizon})."
        )

    # ---------------------------------
    # 1. Median point-forecast CSV
    # ---------------------------------

    median_predictions = np.median(samples, axis=1)

    result_df = df.copy()
    result_df.loc[:, y_cols] = median_predictions

    output_csv = build_output_path(
        input_csv,
        output_dir,
        output_path,
        suffix=suffix
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        output_csv,
        index=False
    )

    # ---------------------------------
    # 2. Full probabilistic samples
    # ---------------------------------

    samples_path = output_dir / (
        f"{input_csv.stem}"
        f"{suffix}_samples.npz"
    )

    np.savez_compressed(
        samples_path,
        samples=samples,
        context_means=context_means,
        context_scales=context_scales
    )

    return output_csv, samples_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=PREDICTION_LENGTH)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
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

    df, contexts = load_input_csv(
        args.input_csv,
        args.lookback,
        args.limit
    )

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
    print(f"Using Sundial model: {args.model_id}")
    print(
        "Inference parameters:",
        f"horizon={args.horizon},",
        f"num_samples={args.num_samples},",
        f"batch_size={args.batch_size}",
    )
    print(f"Using device: {device}")

    model = load_sundial_model(args.model_id, device, args.dtype)
    samples_std = forecast_contexts(
        model=model,
        contexts=contexts_std,
        horizon=args.horizon,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
    )

    samples = (
        samples_std
        * context_scales[:, None, None]
        + context_means[:, None, None]
    ).astype(np.float32)

    print(
        "Standardized Sundial sample shape:",
        samples_std.shape
    )

    print(
        "Inverse-transformed Sundial sample shape:",
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
            "Non-finite inverse-transformed Sundial samples."
        )

    output_csv, samples_path = save_forecast_csv(
        df=df,
        samples=samples,
        context_means=context_means,
        context_scales=context_scales,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_path=args.output_path,
        horizon=args.horizon,
    )
    print(
    f"Sundial zero-shot point predictions saved to: "
    f"{output_csv}"
    )

    print(
        f"Sundial probabilistic samples saved to: "
        f"{samples_path}"
    )


if __name__ == "__main__":
    main()
