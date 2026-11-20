"""Zero-shot Timestral forecasting without external context standardization.

Inputs and outputs stay in the supplied CSV units (expected: log returns).
Model-internal preprocessing is unchanged.

Dependencies (Python >=3.10):
    pip install git+https://github.com/mistralai/timestral.git pandas

See https://github.com/mistralai/timestral for the official model interface.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 250
HORIZON = 20
DEFAULT_MODEL_ID = "mistralai/Timestral-209M"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    p.add_argument("--output-path", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.lookback < 1 or args.horizon < 1 or args.batch_size < 1:
        raise ValueError("lookback, horizon, and batch-size must all be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    import torch
    from timestral import TimestralForecaster

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    df = pd.read_csv(args.input_csv)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    if df.empty:
        raise ValueError("No input rows")
    x_cols = [f"X_{i}" for i in range(args.lookback)]
    y_cols = [f"Y_{i}" for i in range(args.horizon)]
    missing_x = [c for c in x_cols if c not in df.columns]
    missing_y = [c for c in y_cols if c not in df.columns]
    if missing_x or missing_y:
        raise ValueError(f"Missing columns: X={missing_x[:5]}, Y={missing_y[:5]}")
    contexts = df[x_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(contexts).all():
        raise ValueError("This script expects finite context values")

    print(f"Loading {args.model_id} on {device}", flush=True)
    forecaster = TimestralForecaster.from_pretrained(args.model_id, device=device)
    pieces = []
    quantile_levels = None
    for start in range(0, len(df), args.batch_size):
        end = min(start + args.batch_size, len(df))
        # Official API accepts a (batch, time) array for univariate forecasts.
        result = forecaster.forecast(contexts[start:end], prediction_length=args.horizon)
        values = result.quantiles
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float32)
        # Official output: (batch, variates=1, horizon, quantiles=21)
        expected_prefix = (end - start, 1, args.horizon)
        if values.ndim != 4 or tuple(values.shape[:3]) != expected_prefix:
            raise ValueError(f"Unexpected Timestral quantile shape {values.shape}; expected {expected_prefix + ('Q',)}")
        levels = np.asarray(result.quantile_levels, dtype=np.float32)
        if values.shape[-1] != len(levels):
            raise ValueError("Quantile level count differs from forecast output")
        if quantile_levels is None:
            quantile_levels = levels
        elif not np.array_equal(quantile_levels, levels):
            raise ValueError("Quantile levels changed between batches")
        pieces.append(values[:, 0, :, :].transpose(0, 2, 1))  # (batch, Q, horizon)
        print(f"Forecasted {end}/{len(df)} windows", flush=True)

    # Evaluate on the same nine deciles used by the other FYP models.
    target_levels = np.arange(1, 10, dtype=np.float32) / 10.0
    indices = []
    for target in target_levels:
        matches = np.flatnonzero(np.isclose(quantile_levels, target, atol=1e-6))
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate Timestral quantile {target}: {quantile_levels}")
        indices.append(int(matches[0]))

    quantiles = np.concatenate(pieces, axis=0)[:, indices, :]
    quantile_levels = target_levels
    if not np.isfinite(quantiles).all():
        raise ValueError("Non-finite quantiles")
    median_idx = np.flatnonzero(np.isclose(quantile_levels, 0.5))
    if len(median_idx) != 1:
        raise ValueError(f"Expected exactly one 0.5 quantile, got {quantile_levels}")
    predictions = quantiles[:, int(median_idx[0]), :]
    if predictions.shape != (len(df), args.horizon):
        raise ValueError("Unexpected point-prediction shape")

    checkpoint_name = args.model_id.rstrip("/").split("/")[-1].lower()
    suffix = f"_{checkpoint_name}_unstandardized_zero-shot"
    output_csv = args.output_path or (args.output_dir / f"{args.input_csv.stem}{suffix}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df = df.copy()
    result_df.loc[:, y_cols] = predictions
    result_df.to_csv(output_csv, index=False)
    quantiles_path = output_csv.with_name(f"{output_csv.stem}_quantiles.npz")
    np.savez_compressed(
        quantiles_path,
        quantiles=quantiles,
        quantile_levels=quantile_levels,
        context_means=np.zeros(len(contexts), dtype=np.float32),
        context_scales=np.ones(len(contexts), dtype=np.float32),
        external_normalization=np.asarray("none"),
    )
    print(f"CSV saved: {output_csv}")
    print(f"Nine evaluation quantiles saved: {quantiles_path}; shape={quantiles.shape} (N,Q,H)")
    print(f"Quantile levels: {quantile_levels.tolist()}")


if __name__ == "__main__":
    main()
