# Workforce Demand Forecasting under a Structural Regime Change

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S_pla9J-3DcWeggbtJj_plwRjQejM6nj?usp=sharing)

## Overview

This project develops an end-to-end time-series forecasting workflow for daily workforce demand.

The goal is to forecast the number of employees required each day while evaluating how different forecasting approaches behave when the underlying demand regime changes.

The project was completed as part of the **SDAIA Academy — Time Series Forecasting AI Systems** training programme.

**Trainee:** Norah Almadhi
**Cohort:** [20/9/2026 – 22/9/2026]

---

## Dataset

The project uses:

`data/workforce_demand.csv`

from the course repository:

https://github.com/MohammadYusif/time-series-forecasting-ai-systems

The dataset contains one daily time series:

- **Date range:** 2024-01-01 to 2025-12-31
- **Observations:** 731 daily records
- **Target:** `required_headcount`
- **Frequency:** Daily
- **Primary seasonality:** Weekly
- **Documented structural break:** 2025-04-01

The dataset is synthetic and was generated for educational use by the course. It should not be interpreted as data from a real employer or contact centre.

### Why this dataset?

I selected the workforce-demand series because its structural break creates a useful forecasting problem beyond ordinary train/test evaluation.

A model can perform well after the new regime has been established while still failing badly at the point where the shift first occurs.

This makes the dataset particularly suitable for demonstrating why walk-forward backtesting and fold-level diagnostics matter.

---

## Project Objectives

The notebook implements all seven capstone requirements:

1. **Time-series diagnostics**
   - STL decomposition
   - ACF and PACF
   - Augmented Dickey-Fuller stationarity test
   - first-order differencing

2. **Classical forecasting**
   - Holt-Winters exponential smoothing
   - weekly seasonality
   - seasonal-naive baseline
   - Ljung-Box residual diagnostic

3. **Machine-learning forecasting**
   - LightGBM
   - lag features
   - rolling statistics
   - calendar features
   - leakage-safe recursive multi-step forecasting

4. **Walk-forward validation**
   - 40 expanding-window folds
   - 10-day forecast horizon
   - structural-break fold analysis

5. **Forecast evaluation**
   - MAE
   - RMSE
   - WAPE
   - MASE

6. **Probabilistic forecasting**
   - LightGBM quantile regression
   - 80% prediction interval
   - empirical coverage
   - interval width
   - pinball loss

7. **Model comparison and recommendation**
   - predictive accuracy
   - history requirements
   - interpretability
   - interval support
   - computational complexity
   - structural-break behavior

---

## Forecasting Approaches

### Holt-Winters Exponential Smoothing

The classical model uses:

- additive trend
- additive weekly seasonality
- seasonal period of 7 days

The model provides a compact and interpretable representation of the recurring workforce pattern.

### LightGBM

The machine-learning model uses temporal features including:

- lag 1
- lag 7
- lag 14
- lag 28
- rolling means
- rolling standard deviation
- day of week
- month
- annual sine/cosine calendar encoding

Multi-step forecasts are generated recursively.

Future actual observations are never used to construct forecast-time lag or rolling features.

---

## Leakage Prevention

Avoiding temporal leakage is a central part of the project.

The following safeguards are applied:

- data is never randomly shuffled;
- train/test splits preserve chronological order;
- rolling statistics use `shift(1)` before rolling;
- future test targets are never used as lag features;
- LightGBM forecasts are generated recursively;
- each backtest model is fitted only on the training data available for its own fold;
- assertions verify that training windows do not overlap their test windows.

The documented structural-break date is used only for retrospective analysis and is not supplied to the forecasting models as a predictor.

---

## Walk-Forward Backtesting

The main validation design uses:

- **40 expanding-window folds**
- **10-day forecast horizon**
- **minimum training history of 320 observations**

An expanding window was selected because it allows each model to accumulate all historical observations while making the effect of the April 2025 regime change visible.

The wide set of folds ensures that the evaluation begins before the break rather than evaluating only the stable post-break period.

---

## Key Results

Across the 40 walk-forward folds, the approximate mean results were:

| Model | MAE | RMSE | WAPE | MASE |
|---|---:|---:|---:|---:|
| Holt-Winters | 5.05 | 6.22 | 6.39% | 1.16 |
| Recursive LightGBM | 5.33 | 6.72 | 6.68% | 1.22 |
| Seasonal Naive | 5.68 | 7.37 | 7.13% | 1.30 |

The most important finding is visible around the structural break.

The fold crossing **2025-04-01** produces WAPE of approximately:

- **17.4%** for Holt-Winters
- **18.9%** for LightGBM
- **17.1%** for seasonal naive

All model families therefore experience a major error increase when the new regime first appears.

Performance recovers once post-break observations begin entering the training windows.

This demonstrates why a single recent holdout can hide important failure modes.

---

## Probabilistic Forecasting

Quantile LightGBM models were trained for the:

- 10th percentile
- 50th percentile
- 90th percentile

The 10th and 90th percentiles form a nominal **80% prediction interval**.

On the final 60-day holdout:

- **Empirical coverage:** approximately 81.7%
- **Mean interval width:** approximately 17.6 headcount units

Coverage and interval width are reported together because coverage alone does not indicate whether an interval is usefully sharp.

---

## Model Recommendation

For this individual workforce series, the recommended primary point-forecast model is **Holt-Winters exponential smoothing**.

The recommendation is based on more than forecast accuracy.

### History

The dataset contains enough daily history for both classical and machine-learning models, but only one series is being forecast. Therefore, LightGBM does not gain the global-model advantage it would have across many related series.

### Interpretability

Holt-Winters expresses the series using understandable level, trend, and weekly seasonal components, making the model easier to communicate to workforce-planning stakeholders.

### Compute

Holt-Winters is computationally lightweight and requires much less feature-engineering and recursive-inference infrastructure.

### Prediction intervals

Quantile LightGBM provides a useful uncertainty-aware extension and produced an 80% interval with empirical coverage close to the nominal target.

### Structural change

Neither model family anticipated the unexpected April 2025 regime shift.

The primary operational control should therefore include frequent refitting, walk-forward monitoring, and investigation of sudden increases in forecast error.

---

## Limitations

- The dataset is synthetic and should not be interpreted as real company data.
- No external explanatory variables are included.
- A future regime change would not ordinarily be known in advance.
- Holt-Winters residuals retain some autocorrelation according to the Ljung-Box diagnostic.
- Prediction-interval calibration may deteriorate during a new structural break.
- Conclusions apply to this series and validation design rather than establishing one universally superior forecasting family.

---

## Repository Structure

```text
.
├── README.md
├── workforce_demand_forecasting_capstone.ipynb
```

The course dataset and utility modules are downloaded automatically when the notebook is opened in a fresh Colab runtime.

---

## How to Run

### Google Colab

Use the **Open in Colab** badge at the top of this README.

Then select:

`Runtime → Run all`

The notebook installs its required open-source libraries and retrieves the dataset and shared course utility modules automatically.

No API key or credential is required.

### Local execution

Clone this project repository and open the notebook in a Jupyter-compatible environment.

Internet access is required on the first run if the course dataset and utilities are not already present locally.

---

## Reproducibility

The project fixes the random seed:

```python
RNG_SEED = 20260912
```

The notebook is designed to execute from top to bottom without manual data manipulation.

Forecast metrics use the course-provided `common/metrics.py`, and the walk-forward split logic uses `common/backtest.py`.

---

## Training Programme

This project was completed under:

**SDAIA Academy — Time Series Forecasting AI Systems**

SDAIA Academy GitHub:

https://github.com/SDAIAAcademy

Course repository:

https://github.com/MohammadYusif/time-series-forecasting-ai-systems

---

## Author

**Norah Almadhi**


