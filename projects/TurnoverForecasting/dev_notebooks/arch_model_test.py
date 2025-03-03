# %%
import pandas as pd
import os 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import itertools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.filters.hp_filter import hpfilter
from numpy.polynomial.polynomial import Polynomial
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

req_naics = 336111
df = pd.read_csv(f"../data/processed_data_{req_naics}.csv")

print(df.head())
# Convert 'Year' column to datetime if it's not already
df['year'] = pd.to_datetime(df['year'], format='%Y')

# Select the column to decompose (e.g., 'VSHIP' for turnover forecasting)
series = df['vship']  # Replace with the relevant column


plot_acf(series, lags=20)  # See if there’s a peak around lag=12
plt.show()

freq_spectrum = np.fft.fft(series.dropna())
plt.plot(np.abs(freq_spectrum))
plt.title("Frequency Spectrum")
plt.show()


# Apply Hodrick-Prescott Filter to extract trend
cycle, trend = hpfilter(series, lamb=1600)  # lambda=1600 is common for annual data

# Plot trend vs original series
plt.figure(figsize=(10, 5))
plt.plot(series, label="Original Series", color="blue", alpha=0.6)
plt.plot(trend, label="Extracted Trend (HP Filter)", color="red", linewidth=2)
plt.legend()
plt.title("Trend Extraction Using HP Filter")
plt.show()

# Create time index
X = np.arange(len(series))  # Convert years to numerical values
y = series.values

# Fit a 2nd-degree polynomial trend model
p = Polynomial.fit(X, y, deg=5)

# Plot trend
plt.figure(figsize=(10, 5))
plt.plot(series, label="Original Series", color="blue", alpha=0.6)
plt.plot(series.index, p(X), label="Quadratic Trend", color="red", linewidth=2)
plt.legend()
plt.title("Polynomial Trend Fitting")
plt.show()

series_diff = series.diff().dropna()

# Recheck ACF after differencing
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series_diff, lags=20)
plt.show()

from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(series_diff)
print(f"ADF Statistic: {adf_result[0]}")
print(f"P-value: {adf_result[1]}")

if adf_result[1] < 0.05:
    print("Series is stationary after differencing.")
else:
    print("Series is still non-stationary, further differencing")

# Apply second-order differencing
series_diff2 = series.diff().diff().dropna()

# Recheck ACF after second differencing
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series_diff2, lags=20)
plt.show()

# Perform ADF test again
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(series_diff2)
print(f"ADF Statistic: {adf_result[0]}")
print(f"P-value: {adf_result[1]}")

if adf_result[1] < 0.05:
    print("Series is now stationary after second differencing.")
else:
    print("Series is still non-stationary, further transformations may be needed.")


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ACF (for q selection)
sm.graphics.tsa.plot_acf(series.diff().diff().dropna(), lags=20, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")

# PACF (for p selection)
sm.graphics.tsa.plot_pacf(series.diff().diff().dropna(), lags=20, ax=axes[1])
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.show()

# Assuming 'series' is your Pandas Series with a DateTime index
train_size = int(len(series) * 0.8)  # 80% train, 20% holdout
train, holdout = series.iloc[:train_size], series.iloc[train_size:]

# Plot train/holdout split
plt.figure(figsize=(10, 4))
plt.plot(train, label="Training Data")
plt.plot(holdout, label="Holdout Data", color="red")
plt.title("Train-Holdout Split")
plt.legend()
plt.show()


tscv = TimeSeriesSplit(n_splits=5)  # 5 folds

p, d, q = 1, 2, 2  # Set based on ACF/PACF analysis
garch_p, garch_q = 1, 1  # GARCH hyperparameters

arima_errors = []
garch_volatility_forecasts = []

for train_idx, test_idx in tscv.split(train):
    train_fold, test_fold = train.iloc[train_idx], train.iloc[test_idx]

    # Train ARIMA model on expanding training set
    arima_model = ARIMA(train_fold, order=(p, d, q))
    arima_fit = arima_model.fit()

    # Forecast ARIMA on test fold
    arima_forecast = arima_fit.forecast(steps=len(test_fold))

    # Calculate MAE for ARIMA prediction
    error = mean_absolute_error(test_fold, arima_forecast)
    arima_errors.append(error)

    # Extract ARIMA residuals
    residuals = arima_fit.resid

    # Fit GARCH model on residuals
    garch_model = arch_model(residuals, vol='Garch', p=garch_p, q=garch_q)
    garch_fit = garch_model.fit(disp="off")

    # Forecast volatility for the test fold
    garch_forecast = garch_fit.forecast(start=len(residuals), horizon=len(test_fold))
    volatility_forecast = np.sqrt(garch_forecast.variance.iloc[-1])  # Extract volatility

    garch_volatility_forecasts.append(volatility_forecast)

# Print cross-validation errors
print(f"Average ARIMA MAE Across Folds: {np.mean(arima_errors)}")