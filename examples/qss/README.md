# BForTProb · Reproduce one lower-tail QSS result

This example reproduces the evaluation of saved Moirai 2.0 forecasts for the daily AUD/USD series. It uses 495 held-out rolling forecast origins, 250 input returns per origin, and 20 forecast leads. The bundled quantiles were generated with the benchmark's standardized inference protocol and are in the same return units as the test observations.

## Run

From the extracted folder, run the following with Python 3 and NumPy.

```sh
python -m pip install -r requirements.txt
python reproduce_qss.py
```

The script prints the aggregate model and reference pinball losses, overall QSS, and QSS at each of the 20 leads. Expected aggregate values are shown below.

| Quantity | Value |
| --- | ---: |
| Forecast origins | 495 |
| Model pinball loss at the 0.10 quantile | 0.0010727369 |
| Reference pinball loss at the 0.10 quantile | 0.0010654411 |
| Overall QSS at the 0.10 quantile | −0.00684772 |

At each origin, the reference forecast is the empirical 10th percentile of the 250 preceding returns, reused across the 20 forecast leads. The script takes the 0.10 quantile from the saved model forecasts and evaluates both forecasts on the same held-out observations. Overall QSS is one minus the ratio of the two aggregate mean pinball losses. The example's negative QSS corresponds to about 0.68% higher pinball loss than the rolling empirical reference.

## Inputs

- `data/test.csv` contains the held-out windows with columns `X_0` through `X_249` and targets `Y_0` through `Y_19` in forecast-origin order.
- `data/forecasts.npz` contains `quantile_levels` and a `quantiles` array of shape `(495, 20, 9)`, ordered by forecast origin, lead, and quantile level.

The package reproduces the scoring stage for this model and series from saved predictions. It requires no model checkpoint, GPU, or network connection once NumPy is installed.
