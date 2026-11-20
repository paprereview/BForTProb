"""Reproduce lower-tail QSS from held-out observations and saved quantiles."""

import argparse
import csv
from pathlib import Path

import numpy as np


LOOKBACK = 250
HORIZON = 20
TAU = 0.10


def load_test_windows(path):
    columns = [f"X_{i}" for i in range(LOOKBACK)]
    columns += [f"Y_{i}" for i in range(HORIZON)]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = [name for name in columns if name not in header]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=[header.index(name) for name in columns],
        ndmin=2,
        encoding="utf-8-sig",
    )
    if not values.size or not np.isfinite(values).all():
        raise ValueError("The test CSV must contain finite values and at least one origin.")
    return values[:, :LOOKBACK], values[:, LOOKBACK:]


def load_lower_quantile(path, target_shape):
    with np.load(path, allow_pickle=False) as forecast:
        levels = np.asarray(forecast["quantile_levels"], dtype=float)
        quantiles = np.asarray(forecast["quantiles"], dtype=float)
    if levels.ndim != 1 or quantiles.shape != (*target_shape, levels.size):
        raise ValueError(
            f"Expected forecast shape {(*target_shape, levels.size)}, got {quantiles.shape}."
        )
    matches = np.flatnonzero(np.isclose(levels, TAU))
    if matches.size != 1:
        raise ValueError("Expected exactly one 0.10 forecast quantile.")
    lower = quantiles[:, :, matches[0]]
    if not np.isfinite(lower).all():
        raise ValueError("The 0.10 forecasts must be finite.")
    return lower


def pinball_loss(target, forecast):
    error = target - forecast
    return np.maximum(TAU * error, (TAU - 1.0) * error)


def calculate_qss(context, target, model_q10):
    empirical_q10 = np.quantile(context, TAU, axis=1)[:, None]
    model_loss = pinball_loss(target, model_q10)
    reference_loss = pinball_loss(target, empirical_q10)
    model_mean = model_loss.mean()
    reference_mean = reference_loss.mean()
    reference_by_lead = reference_loss.mean(axis=0)
    if np.any(reference_by_lead <= 0):
        raise ValueError("Reference pinball loss must be positive at each lead.")
    return (
        model_mean,
        reference_mean,
        1.0 - model_mean / reference_mean,
        1.0 - model_loss.mean(axis=0) / reference_by_lead,
    )


def main():
    data = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=data / "test.csv")
    parser.add_argument("--forecasts", type=Path, default=data / "forecasts.npz")
    args = parser.parse_args()
    try:
        context, target = load_test_windows(args.test)
        lower = load_lower_quantile(args.forecasts, target.shape)
        model_loss, reference_loss, qss, qss_by_lead = calculate_qss(
            context, target, lower
        )
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Forecast origins:         {target.shape[0]}")
    print(f"Forecast leads:           {target.shape[1]}")
    print(f"Model pinball loss:       {model_loss:.10f}")
    print(f"Reference pinball loss:   {reference_loss:.10f}")
    print(f"QSS (tau = {TAU:.2f}):         {qss:.8f}")
    print("Lead,QSS")
    for lead, value in enumerate(qss_by_lead, start=1):
        print(f"{lead},{value:.8f}")


if __name__ == "__main__":
    main()
