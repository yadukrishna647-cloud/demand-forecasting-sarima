# Demand Forecasting with ARIMA / SARIMA
# Dataset: Kaggle "Store Item Demand Forecasting Challenge" -> data/train.csv
# Columns: date, store, item, sales
# In VS Code, each "# %%" line is a cell: click "Run Cell" above it (needs the Jupyter extension).

# %% 1. Imports and settings
import os
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)

STORE, ITEM = 1, 1      # the single series we forecast
TEST_DAYS = 90          # hold-out period
SEASON = 7              # weekly seasonality in daily data

# %% 2. Load and clean
df = pd.read_csv("data/train.csv", parse_dates=["date"])
print(df.head())
print(df.shape)

series = (
    df[(df["store"] == STORE) & (df["item"] == ITEM)]
    .set_index("date")["sales"]
    .sort_index()
    .asfreq("D")            # force a daily frequency (inserts missing dates as NaN)
)
print("Missing values:", series.isna().sum())
series = series.interpolate()   # fill any gaps
print(series.describe())

plt.figure(figsize=(12, 4))
series.plot(title=f"Daily sales - store {STORE}, item {ITEM}")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("images/01_raw_series.png", dpi=150)
plt.show()

# %% 3. Explore: trend and seasonality
decomp = seasonal_decompose(series, model="additive", period=SEASON)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig("images/02_decomposition.png", dpi=150)
plt.show()

# Average sales by weekday and by month
print(series.groupby(series.index.dayofweek).mean())
print(series.groupby(series.index.month).mean())

# %% 4. Stationarity (ADF test)
def adf_report(x, name):
    stat, pvalue, *_ = adfuller(x.dropna())
    verdict = "stationary" if pvalue < 0.05 else "NOT stationary"
    print(f"{name:<28} ADF = {stat:8.3f}   p = {pvalue:.4f}   -> {verdict}")

adf_report(series, "original")
adf_report(series.diff(), "1st difference")
adf_report(series.diff().diff(SEASON), "1st diff + seasonal diff")

# %% 5. ACF / PACF (helps choose p and q)
stationary = series.diff().diff(SEASON).dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(stationary, lags=40, ax=axes[0])
plot_pacf(stationary, lags=40, ax=axes[1], method="ywm")
plt.tight_layout()
plt.savefig("images/03_acf_pacf.png", dpi=150)
plt.show()
# Reading tips:
#  - ACF cuts off after lag q            -> MA(q)
#  - PACF cuts off after lag p           -> AR(p)
#  - spikes at lags 7, 14, 21...         -> seasonal terms (P, Q)

# %% 6. Train / test split (time order, never shuffle)
train = series.iloc[:-TEST_DAYS]
test = series.iloc[-TEST_DAYS:]
print("Train:", train.index.min().date(), "to", train.index.max().date(), len(train))
print("Test: ", test.index.min().date(), "to", test.index.max().date(), len(test))

# %% 7. Baselines
def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))

def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))

# Naive: repeat the last observed value
naive_fc = np.repeat(train.iloc[-1], len(test))

# Seasonal naive: repeat the last 7 days
last_week = train.iloc[-SEASON:].values
seasonal_naive_fc = np.tile(last_week, int(np.ceil(len(test) / SEASON)))[: len(test)]

results = {
    "Naive": (mae(test, naive_fc), rmse(test, naive_fc)),
    "Seasonal Naive": (mae(test, seasonal_naive_fc), rmse(test, seasonal_naive_fc)),
}
for k, v in results.items():
    print(f"{k:<16} MAE = {v[0]:.2f}  RMSE = {v[1]:.2f}")

# %% 8. ARIMA (no seasonality) as an intermediate model
arima_model = ARIMA(train, order=(2, 1, 2)).fit()
print(arima_model.summary())

arima_fc = np.asarray(arima_model.forecast(steps=len(test)))
results["ARIMA(2,1,2)"] = (mae(test, arima_fc), rmse(test, arima_fc))
print("ARIMA  MAE = %.2f  RMSE = %.2f" % results["ARIMA(2,1,2)"])

# %% 9. SARIMA: small grid search using AIC (takes a few minutes)
p_range = q_range = [0, 1, 2]
P_range = Q_range = [0, 1]
d, D = 1, 1

grid_rows = []
for p, q, P, Q in itertools.product(p_range, q_range, P_range, Q_range):
    try:
        m = SARIMAX(
            train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, SEASON),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        grid_rows.append({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "AIC": m.aic})
    except Exception as e:
        print("Failed:", (p, d, q), (P, D, Q), e)

grid = pd.DataFrame(grid_rows).sort_values("AIC").reset_index(drop=True)
grid.to_csv("results/sarima_grid_search.csv", index=False)
print(grid.head(10))

best = grid.iloc[0]
best_order = (int(best.p), int(best.d), int(best.q))
best_seasonal = (int(best.P), int(best.D), int(best.Q), SEASON)
print("Best order:", best_order, "seasonal:", best_seasonal)

# %% 10. Fit the best SARIMA and forecast the test period
sarima_model = SARIMAX(
    train,
    order=best_order,
    seasonal_order=best_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

sarima_fc = np.asarray(sarima_model.get_forecast(steps=len(test)).predicted_mean)
label = f"SARIMA{best_order}x{best_seasonal}"
results[label] = (mae(test, sarima_fc), rmse(test, sarima_fc))
print(label, "MAE = %.2f  RMSE = %.2f" % results[label])

# %% 11. Residual diagnostics (should look like white noise)
resid = sarima_model.resid.iloc[SEASON + 2:]   # skip the first values, distorted by differencing

fig = sarima_model.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.savefig("images/04_residual_diagnostics.png", dpi=150)
plt.show()

lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
print(lb)
print("p-values above 0.05 mean no significant leftover autocorrelation (good).")

# %% 12. Compare all models
table = pd.DataFrame(results, index=["MAE", "RMSE"]).T.round(2)
base_mae = table.loc["Seasonal Naive", "MAE"]
table["MAE improvement vs Seasonal Naive (%)"] = ((base_mae - table["MAE"]) / base_mae * 100).round(1)
table = table.sort_values("MAE")
table.to_csv("results/model_comparison.csv")
print(table)

plt.figure(figsize=(12, 5))
plt.plot(train.iloc[-120:], label="Train (last 120 days)", color="gray")
plt.plot(test, label="Actual", color="black")
plt.plot(test.index, seasonal_naive_fc, label="Seasonal Naive", linestyle=":")
plt.plot(test.index, arima_fc, label="ARIMA(2,1,2)", linestyle="--")
plt.plot(test.index, sarima_fc, label=label, linewidth=2)
plt.title("Actual vs forecast on the 90-day test set")
plt.legend()
plt.tight_layout()
plt.savefig("images/05_forecast_comparison.png", dpi=150)
plt.show()

# %% 13. Final model: refit on ALL data and forecast the next 30 days
final_model = SARIMAX(
    series,
    order=best_order,
    seasonal_order=best_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

future = final_model.get_forecast(steps=30)
future_mean = future.predicted_mean
future_ci = future.conf_int()

plt.figure(figsize=(12, 5))
plt.plot(series.iloc[-120:], label="History", color="gray")
plt.plot(future_mean, label="30-day forecast", color="tab:blue")
plt.fill_between(
    future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
    color="tab:blue", alpha=0.2, label="95% interval",
)
plt.title("Next 30 days forecast")
plt.legend()
plt.tight_layout()
plt.savefig("images/06_future_forecast.png", dpi=150)
plt.show()

future_df = pd.DataFrame(
    {"forecast": future_mean, "lower_95": future_ci.iloc[:, 0], "upper_95": future_ci.iloc[:, 1]}
).round(1)
future_df.to_csv("results/future_30_day_forecast.csv")
print(future_df.head())