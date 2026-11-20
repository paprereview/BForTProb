#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
set -uo pipefail

DATA_ROOT="$PROJECT_DIR/BForTFin_log_return/output_weekly"
RESULTS_ROOT="$PROJECT_DIR/BForTFin_results/standardised/output_weekly/timestral_standardized"
PYTHON="${PYTHON:-python}"
PREDICT_SCRIPT="$SCRIPT_DIR/predict_standardized.py"
MODEL_ID="mistralai/Timestral-209M"
BATCH_SIZE=16

if ! command -v "$PYTHON" >/dev/null 2>&1 || [ ! -f "$PREDICT_SCRIPT" ]; then
  echo "ERROR: missing $PYTHON or $PREDICT_SCRIPT" >&2
  exit 1
fi
if [ ! -d "$DATA_ROOT" ] || ! find "$DATA_ROOT" -type f -name '*.csv' -print -quit | grep -q .; then
  echo "ERROR: no input CSVs under $DATA_ROOT" >&2
  exit 1
fi
mkdir -p "$RESULTS_ROOT"

for asset_dir in "$DATA_ROOT"/*; do
  [ -d "$asset_dir" ] || continue
  folder_name=$(basename "$asset_dir")
  output_dir="$RESULTS_ROOT/$folder_name"
  mkdir -p "$output_dir"
  for file in "$asset_dir"/*.csv; do
    [ -f "$file" ] || continue
    stem=$(basename "$file" .csv)
    lower_name=$(printf '%s' "$stem" | tr '[:upper:]' '[:lower:]')
    if [[ "${folder_name,,}" == *crypto* && "$lower_name" == *tether* ]]; then
      echo "SKIPPING Tether: $file"
      continue
    fi
    if [[ "${folder_name,,}" == *crypto* && "$lower_name" == *solana* ]]; then
      echo "SKIPPING Solana: $file"
      continue
    fi
    suffix="_timestral-209m_standardized_zero-shot"
    quantiles_path="$output_dir/${stem}${suffix}_quantiles.npz"
    csv_path="$output_dir/${stem}${suffix}.csv"
    if [ -s "$quantiles_path" ] && [ -s "$csv_path" ]; then
      echo "ALREADY DONE: $file"
      continue
    fi
    echo "RUNNING: $file @ $(date)"
    if "$PYTHON" "$PREDICT_SCRIPT" \
      --input-csv "$file" \
      --output-dir "$output_dir" \
      --model-id "$MODEL_ID" \
      --lookback 250 --horizon 20 \
      --batch-size "$BATCH_SIZE" --device cuda --seed 42; then
      echo "SUCCESS: $file"
    else
      echo "FAILED: $file" >&2
    fi
  done
done
echo "WEEKLY TIMESTRAL DATASETS COMPLETE @ $(date)"
