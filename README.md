# Workforce Demand Forecasting under a Structural Regime Change

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S_pla9J-3DcWeggbtJj_plwRjQejM6nj?usp=sharing)

## Overview

This project develops an end-to-end time-series forecasting workflow for daily workforce demand.

The objective is to forecast required workforce headcount while evaluating how classical, machine-learning, and probabilistic forecasting approaches behave under strong weekly seasonality and an abrupt structural regime change.

The project was completed as part of the **SDAIA Academy — Time Series Forecasting AI Systems** training programme.

**Trainee:** Norah Almadhi  
**Cohort:** 20/9/2026 – 22/9/2026

---

## Dataset

The project uses:

`data/workforce_demand.csv`

from the course repository:

https://github.com/MohammadYusif/time-series-forecasting-ai-systems

The dataset contains one synthetic daily workforce-demand series:

- **Date range:** 2024-01-01 to 2025-12-31
- **Observations:** 731 daily records
- **Target:** `required_headcount`
- **Frequency:** Daily
- **Dominant seasonality:** Weekly
- **Documented structural break:** 2025-04-01

The dataset was generated specifically for the course and does not represent a real employer or contact centre.

### Why this dataset?

I selected the workforce-demand dataset because its structural break creates a forecasting problem where validation design matters as much as model choice.

A model may perform well once the post-break regime is established while still failing substantially when the shift first occurs.

This makes the series particularly useful for demonstrating:

- time-series diagnostics,
- classical and machine-learning forecasting,
- leakage-safe feature engineering,
- expanding and rolling walk-forward validation,
- probabilistic forecasting,
- and model-family selection under regime change.

---

## Project Scope

The notebook implements the full capstone workflow.

### 1. Time-Series Structure and Diagnostics

The diagnostic analysis includes:

- STL decomposition,
- ACF and PACF,
- Augmented Dickey-Fuller testing,
- regular differencing,
- seasonal differencing,
- and analysis of additive versus multiplicative seasonal behavior.

Key findings include:

- Raw-series ADF p-value: approximately **0.879**
- Pre-break ADF p-value: approximately **0.870**
- First-differenced series: stationary at the 5% level
- Seasonal-differenced series: stationary at the 5% level
- ACF at lag 7: approximately **0.953**
- ACF at lag 14: approximately **0.937**

The weekend-to-weekday mean ratio remains nearly constant across regimes:

- Pre-break: approximately **0.452**
- Post-break: approximately **0.450**

while the absolute weekday-weekend gap increases from approximately **39** to **56** employees.

This supports the use of **multiplicative weekly seasonality**.

---

## Classical Forecasting

Two classical approaches are evaluated:

- Exponential Smoothing
- SARIMAX

### Exponential Smoothing

Several trend and seasonal specifications are compared using AIC and BIC.

The AIC-selected specification is:

- **Trend:** None
- **Seasonality:** Multiplicative
- **Seasonal period:** 7 days

Its AIC is approximately:

**2171.91**

compared with approximately:

**2269.03**

for the original additive-trend/additive-seasonality benchmark.

The selected model's Ljung-Box p-values at lags 7, 14, and 21 are approximately:

- 0.252
- 0.427
- 0.111

so the residual-autocorrelation null is not rejected at those tested lags.

The final 60-day holdout is intentionally not used as the sole model-selection criterion. On that holdout:

- Additive Holt-Winters WAPE: approximately **6.03%**
- AIC-selected multiplicative ES WAPE: approximately **7.36%**

This disagreement motivates the later repeated walk-forward evaluation.

### SARIMAX

Multiple SARIMAX orders are evaluated using AIC and BIC.

The AIC-selected model is:

`SARIMAX(2,1,2)(0,1,1,7)`

Its final-holdout WAPE is approximately:

**8.84%**

Its Ljung-Box residual checks also show no significant autocorrelation at the tested weekly lags.

---

## Machine-Learning Forecasting

A recursive LightGBM model is trained using leakage-safe temporal features.

### Lag features

- lag 1
- lag 7
- lag 14
- lag 28

### Rolling features

- 7-day rolling mean
- 7-day rolling standard deviation
- 28-day rolling mean

### Calendar features

- day of week
- month
- annual sine encoding
- annual cosine encoding

All rolling features use shifted target values so that the current target cannot enter its own predictors.

Multi-step forecasting is performed recursively. Future holdout targets are never used to build later lag or rolling features.

### LightGBM Holdout Performance

On the final 60-day holdout:

- **MAE:** approximately 5.52
- **RMSE:** approximately 6.59
- **WAPE:** approximately 6.43%
- **MASE:** approximately 1.15

### Gain-Based Feature Importance

The most influential features are:

- **Day of week:** approximately 61.8% of total gain
- **28-day rolling mean:** approximately 24.0%
- **Lag 7:** approximately 6.0%

This is consistent with the strong weekly pattern found during the diagnostic analysis.

---

## Walk-Forward Backtesting

The project evaluates both:

- **Expanding-window validation**
- **Rolling-window validation**

The main configuration uses:

- **40 folds**
- **10-day forecast horizon**
- **320-day rolling training window**
- **7-day seasonal-naive baseline**

Both window designs evaluate exactly the same test periods.

### Expanding-Window Results

| Model | MAE | RMSE | WAPE | MASE |
|---|---:|---:|---:|---:|
| AIC-selected Exponential Smoothing | **4.47** | **5.67** | **5.64%** | **1.03** |
| Recursive LightGBM | 5.33 | 6.72 | 6.68% | 1.22 |
| Seasonal Naive | 5.68 | 7.37 | 7.13% | 1.30 |

### Rolling-Window Results

| Model | MAE | RMSE | WAPE | MASE |
|---|---:|---:|---:|---:|
| AIC-selected Exponential Smoothing | **4.53** | **5.73** | **5.72%** | **0.95** |
| Recursive LightGBM | 5.46 | 6.82 | 6.86% | 1.14 |
| Seasonal Naive | 5.68 | 7.37 | 7.13% | 1.19 |

The expanding window performs slightly better overall, so removing older observations does not improve average forecasting performance for this dataset.

---

## Structural-Break Analysis

The fold crossing the documented structural break on **2025-04-01** produces a large temporary increase in forecast error.

Expanding-window WAPE on the break fold:

- AIC-selected Exponential Smoothing: **18.08%**
- Recursive LightGBM: **18.92%**
- Seasonal Naive: **17.11%**

For the selected exponential-smoothing model, WAPE then falls to approximately:

- **6.10%** on the following fold
- **6.40%** on the next fold

as post-break observations enter the training history.

The result demonstrates that none of the tested models can anticipate an abrupt regime change before observing evidence of the new regime.

---

## Evaluation Metrics

The project reports complementary point-forecast metrics:

- **MAE**
- **RMSE**
- **WAPE**
- **MASE**

MAE and RMSE retain the original headcount scale.

WAPE provides an aggregate percentage interpretation.

MASE scales forecast error against an in-sample weekly seasonal-naive benchmark using a seasonal period of 7.

Metrics are interpreted together rather than using any single value as the model-selection criterion.

---

## Probabilistic Forecasting

Four uncertainty-aware approaches are evaluated on the same final 60-day holdout:

1. Quantile LightGBM
2. Prophet native intervals
3. sktime ThetaForecaster intervals
4. Split-conformal intervals around the selected exponential-smoothing model

All approaches are evaluated using both:

- empirical coverage,
- mean interval width.

Quantile LightGBM is additionally evaluated with pinball loss.

### 80% Prediction-Interval Results

| Method | Holdout WAPE | Empirical Coverage | Mean Interval Width |
|---|---:|---:|---:|
| Quantile LightGBM | 6.45% | **81.7%** | 17.57 |
| sktime ThetaForecaster | 6.61% | **80.0%** | 17.71 |
| Selected ES + Conformal | 7.36% | 60.0% | **13.54** |
| Prophet | 8.24% | 63.3% | 16.65 |

The Quantile LightGBM interval covers 49 of 60 holdout observations.

Its observed 81.7% coverage has a 95% Wilson interval of approximately:

**70.1% to 89.4%**

so the result should not be interpreted as proof of perfect calibration.

The conformal interval is calibrated using only historical walk-forward forecast errors that occur before the final holdout. The final holdout itself is not used to determine the conformal interval width.

---

## Model-Family Comparison

The final decision considers more than forecast accuracy.

### History Length

The dataset contains 731 daily observations, providing enough history for both classical and tree-based forecasting.

Because only one series is forecast, LightGBM does not gain the cross-series pooling advantage that it can offer in large multi-series forecasting problems.

### Interpretability

The selected exponential-smoothing model has a directly interpretable level and multiplicative weekly seasonal component.

SARIMAX also provides explicit statistical structure and residual diagnostics.

Prophet provides interpretable trend and seasonal decomposition.

LightGBM is less directly interpretable, although gain-based feature importance identifies the dominant predictors.

### Prediction Intervals

Prophet and sktime provide native interval interfaces.

LightGBM requires separate quantile models or an additional interval procedure.

The project also demonstrates a split-conformal wrapper around the selected exponential-smoothing point model.

### Compute and Operational Complexity

The selected exponential-smoothing model is inexpensive to fit and does not require a feature-engineering or recursive-inference pipeline.

LightGBM requires additional infrastructure for:

- lag creation,
- rolling features,
- calendar features,
- recursive inference,
- leakage control,
- and multiple quantile models.

---

## Deployment Recommendation

For this single workforce-demand series, the recommended primary point-forecast model is the:

**AIC-selected exponential-smoothing model with multiplicative weekly seasonality**

The recommendation is based on the full validation evidence rather than on one holdout.

Although LightGBM performs better on the final 60-day holdout, the repeated backtests favor exponential smoothing.

The selected model achieves:

- expanding-window WAPE of approximately **5.64%**
- rolling-window WAPE of approximately **5.72%**

compared with:

- LightGBM expanding WAPE of approximately **6.68%**
- LightGBM rolling WAPE of approximately **6.86%**

The selected model also offers greater interpretability and lower implementation complexity for a single workforce series.

The structural-break analysis shows that no tested model is inherently robust to an unexpected regime shift. A production deployment should therefore include:

- frequent model refitting,
- ongoing forecast-error monitoring,
- repeated time-based validation,
- and investigation of sudden error increases.

For uncertainty-aware forecasting, sktime and Quantile LightGBM provide useful benchmarks, while the conformal and Prophet results demonstrate that nominal interval coverage does not guarantee calibrated holdout performance.

---

## Limitations

- The dataset is synthetic and does not represent a real employer.
- No external explanatory variables are available.
- The date of a future regime change would not ordinarily be known in advance.
- The initial additive exponential-smoothing model retained significant residual autocorrelation.
- The selected multiplicative model passes the tested Ljung-Box checks, but this does not prove that every possible temporal dependency has been removed.
- Prediction-interval coverage is evaluated on only 60 temporally related holdout observations.
- SARIMAX, Prophet, and sktime were not all included in the full 40-fold model-family backtest, so their reported results should not be interpreted as repeated-validation rankings.
- Interval calibration could change substantially after a future structural break.

---

## Leakage Prevention

Temporal leakage is explicitly controlled throughout the project.

The notebook:

- never randomly shuffles the time series,
- preserves chronological train/test order,
- shifts the target before calculating rolling features,
- never uses future holdout targets as lag features,
- produces recursive LightGBM forecasts,
- refits models separately inside each backtest fold,
- and verifies that each training period ends before its test window begins.

For rolling-window LightGBM evaluation, calendar features are rebuilt using the actual first date of each rolling training window.

---

## Reproducibility

The notebook:

- downloads the exact course dataset and shared utilities when required,
- fixes the random seed using `RNG_SEED = 20260912`,
- uses the course `common/backtest.py` implementation,
- uses the course `common/metrics.py` implementation,
- preserves chronological ordering,
- and is designed to run sequentially in a fresh Google Colab runtime.

No API key or credential is required.

---

## Repository Structure

```text
workforce-demand-time-series-forecasting/
├── README.md
├── Workforce_Demand_Forecasting.ipynb
```

The dataset and course utility modules are retrieved automatically by the notebook.

---

## How to Run

### Google Colab

Click the **Open in Colab** badge at the top of this README.

Then select:

`Runtime → Run all`

The notebook installs the required open-source packages and downloads the required course files automatically.

### Local Execution

Clone the repository and open:

`workforce_demand_forecasting_capstone.ipynb`

in a Jupyter-compatible environment.

Internet access is required on the first run if the course data and utilities have not already been downloaded.

---

## Training Programme

This project was completed under:

**SDAIA Academy — Time Series Forecasting AI Systems**

**Cohort:** 20/9/2026 – 22/9/2026

SDAIA Academy GitHub:

https://github.com/SDAIAAcademy

Course repository:

https://github.com/MohammadYusif/time-series-forecasting-ai-systems

---

## Author

**Norah Almadhi**
