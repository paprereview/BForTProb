from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

LOOKBACK = 250
PREDICTION_LENGTH = 20
FREQ = "D"

DEFAULT_MODEL_ID = "chronos-2-small"
DEFAULT_MODEL_NAME = "chronos"


def x_columns(lookback: int) -> list[str]:
    return [f"X_{i}" for i in range(lookback)]


def y_columns(horizon: int) -> list[str]:
    return [f"Y_{i}" for i in range(horizon)]


def output_columns(lookback: int, horizon: int) -> list[str]:
    return x_columns(lookback) + y_columns(horizon)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df


def check_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing[:10]}")


def read_supervised_csv(
    csv_path: Path,
    lookback: int,
    horizon: int,
    require_y: bool,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    df = clean_columns(pd.read_csv(csv_path))
    if limit is not None:
        df = df.head(limit)

    x_cols = x_columns(lookback)
    y_cols = y_columns(horizon)
    check_columns(df, x_cols, csv_path.name)
    if require_y:
        check_columns(df, y_cols, csv_path.name)

    x = df[x_cols].to_numpy(dtype=np.float32)
    y = None
    values_to_check = x

    if require_y:
        y = df[y_cols].to_numpy(dtype=np.float32)
        values_to_check = np.concatenate([x, y], axis=1)

    if np.isnan(values_to_check).any() or np.isinf(values_to_check).any():
        raise ValueError(f"{csv_path.name} contains NaN or Inf.")

    return df, x, y


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
    lookback: int,
    horizon: int,
    suffix: str,
) -> Path:
    expected_shape = (len(df), horizon)
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match {expected_shape}"
        )

    x_cols = x_columns(lookback)
    y_cols = y_columns(horizon)
    check_columns(df, x_cols, input_csv.name)
    check_columns(df, y_cols, input_csv.name)

    result_df = df.copy()
    for h, col in enumerate(y_cols):
        result_df[col] = predictions[:, h]

    output_csv = build_output_path(input_csv, output_dir, output_path, suffix)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    return output_csv
